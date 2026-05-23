from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect

import numpy as np

from src.config import ExperimentConfig
from src.data import (
    apply_label_mapping,
    build_label_mapping,
    ensure_smoke_data,
    load_text_classification_csv,
    save_label_mapping,
    tokenize_frame,
)
from src.evaluate import evaluate_model_on_eval_sets
from src.metrics import classification_metrics
from src.methods.adapter import apply_adapter
from src.methods.full_ft import apply_full_finetuning
from src.methods.lora import apply_lora
from src.model_factory import load_b0_model_and_tokenizer, load_base_model_and_tokenizer
from src.seed import detect_device, set_seed
from src.utils import count_parameters, read_json, timer, write_json


def _training_arguments(cfg: ExperimentConfig, output_dir: Path):
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise ImportError("Training requires the 'transformers' package.") from exc

    training = cfg.training
    device = detect_device()
    use_fp16 = bool(training.get("fp16", False)) and device == "cuda"
    use_bf16 = bool(training.get("bf16", False)) and device == "cuda"
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training.get("epochs", 3)),
        "per_device_train_batch_size": int(training.get("batch_size", 16)),
        "per_device_eval_batch_size": int(training.get("eval_batch_size", 32)),
        "learning_rate": float(training.get("learning_rate", 2e-5)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.06)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "fp16": use_fp16,
        "bf16": use_bf16,
        "save_strategy": training.get("save_strategy", "epoch"),
        "load_best_model_at_end": bool(training.get("load_best_model_at_end", True)),
        "metric_for_best_model": training.get("metric_for_best_model", "f1_macro"),
        "greater_is_better": bool(training.get("greater_is_better", True)),
        "logging_steps": int(training.get("logging_steps", 50)),
        "report_to": [],
        "save_total_limit": 2,
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    strategy_value = training.get("evaluation_strategy", training.get("eval_strategy", "epoch"))
    if "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = strategy_value
    else:
        kwargs["eval_strategy"] = strategy_value
    return TrainingArguments(**kwargs)


def _compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return classification_metrics(labels.tolist(), predictions.tolist())


def train_sequence_classifier(
    cfg: ExperimentConfig,
    run_id: str,
    model: Any,
    tokenizer: Any,
    train_frame,
    valid_frame,
    label_mapping: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    try:
        from transformers import EarlyStoppingCallback, Trainer
    except ImportError as exc:
        raise ImportError("Training requires the 'transformers' package.") from exc

    label2id = {label: int(index) for label, index in label_mapping["label2id"].items()}
    train_data = tokenize_frame(apply_label_mapping(train_frame, cfg.label_column, label2id), tokenizer, cfg.text_column, cfg.max_length)
    valid_data = tokenize_frame(apply_label_mapping(valid_frame, cfg.label_column, label2id), tokenizer, cfg.text_column, cfg.max_length)
    callbacks = []
    if cfg.training.get("early_stopping_patience") is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(cfg.training["early_stopping_patience"])))

    trainer_kwargs = {
        "model": model,
        "args": _training_arguments(cfg, output_dir),
        "train_dataset": train_data,
        "eval_dataset": valid_data,
        "compute_metrics": _compute_metrics,
        "callbacks": callbacks,
        **_trainer_processing_kwargs(Trainer, tokenizer),
    }
    trainer = Trainer(**trainer_kwargs)
    with timer() as timing:
        train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "run_id": run_id,
        "training_time_seconds": timing["elapsed"],
        **count_parameters(model),
        **train_result.metrics,
    }
    return metrics


def _trainer_processing_kwargs(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    signature = inspect.signature(trainer_cls.__init__).parameters
    if "tokenizer" in signature:
        return {"tokenizer": tokenizer}
    if "processing_class" in signature:
        return {"processing_class": tokenizer}
    return {}


def _load_baseline_frames(cfg: ExperimentConfig):
    baseline = cfg.data["baseline"]
    return (
        load_text_classification_csv(cfg, baseline["train"]),
        load_text_classification_csv(cfg, baseline["valid"]),
        load_text_classification_csv(cfg, baseline["test"]),
    )


def _apply_method(cfg: ExperimentConfig, model: Any, run: dict[str, Any]) -> Any:
    method = run["method"]
    method_cfg = cfg.methods.get(method, {})
    if method == "full_ft":
        return apply_full_finetuning(model)
    if method == "lora":
        lora_cfg = cfg.methods.get("lora", {})
        return apply_lora(
            model,
            rank=int(run["lora_rank"]),
            lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
            include_attention_modules=bool(lora_cfg.get("include_attention_modules", True)),
            include_mlp_modules=bool(lora_cfg.get("include_mlp_modules", False)),
            train_classifier_head=bool(lora_cfg.get("train_classifier_head", True)),
        )
    if method == "adapter":
        return apply_adapter(
            model,
            bottleneck_size=int(run.get("adapter_bottleneck_size", method_cfg.get("bottleneck_size", 64))),
            nonlinearity=method_cfg.get("nonlinearity", "gelu"),
            train_classifier_head=bool(method_cfg.get("train_classifier_head", True)),
        )
    raise ValueError(f"Unsupported method: {method}")


def run_baseline_training(cfg: ExperimentConfig) -> list[dict[str, Any]]:
    set_seed(int(cfg.project.get("seed", 42)))
    ensure_smoke_data(cfg)
    train_frame, valid_frame, _test_frame = _load_baseline_frames(cfg)
    label_mapping = build_label_mapping(train_frame, cfg.label_column)
    save_label_mapping(cfg, label_mapping)

    model, tokenizer = load_base_model_and_tokenizer(cfg, label_mapping)
    model = apply_full_finetuning(model)
    output_dir = cfg.resolve_path(cfg.baseline.get("save_dir", "outputs/checkpoints/B0"))
    train_metrics = train_sequence_classifier(cfg, "B0", model, tokenizer, train_frame, valid_frame, label_mapping, output_dir)
    eval_metrics = evaluate_model_on_eval_sets(cfg, "B0", model, tokenizer, label_mapping)
    for row in eval_metrics:
        row.update({key: value for key, value in train_metrics.items() if key not in row})
    write_json(cfg.output_dir / "metrics" / "baseline_metrics.json", eval_metrics)
    return eval_metrics


def run_treatment_matrix(cfg: ExperimentConfig, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    set_seed(int(cfg.project.get("seed", 42)))
    ensure_smoke_data(cfg)
    label_mapping = read_json(cfg.output_dir / "label_mapping.json")
    all_metrics: list[dict[str, Any]] = []
    for run in runs:
        print(f"Starting treatment run {run['run_id']}")
        model, tokenizer = load_b0_model_and_tokenizer(cfg, label_mapping)
        model = _apply_method(cfg, model, run)
        paths = cfg.data["domains"][run["domain"]]
        train_frame = load_text_classification_csv(cfg, paths["train"])
        valid_frame = load_text_classification_csv(cfg, paths["valid"])
        output_dir = cfg.output_dir / "checkpoints" / run["run_id"]
        train_metrics = train_sequence_classifier(cfg, run["run_id"], model, tokenizer, train_frame, valid_frame, label_mapping, output_dir)
        eval_metrics = evaluate_model_on_eval_sets(cfg, run["run_id"], model, tokenizer, label_mapping)
        for row in eval_metrics:
            row.update(run)
            row.update({key: value for key, value in train_metrics.items() if key not in row})
        all_metrics.extend(eval_metrics)
    write_json(cfg.output_dir / "metrics" / "treatment_metrics.json", all_metrics)
    return all_metrics
