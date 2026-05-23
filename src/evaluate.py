from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.data import apply_label_mapping, load_text_classification_csv, tokenize_frame
from src.metrics import bias_summary, classification_metrics, robustness_summary
from src.utils import timer


def eval_set_paths(cfg: ExperimentConfig) -> dict[str, str]:
    domains = cfg.data.get("domains", {})
    return {
        "baseline_test": cfg.data["baseline"]["test"],
        "twitter_test": domains["twitter"]["test"],
        "general_test": domains["general"]["test"],
        "bias_eval": cfg.data.get("bias_eval"),
        "robustness_eval": cfg.data.get("robustness_eval"),
    }


def evaluate_predictions_frame(
    frame: pd.DataFrame,
    label_column: str,
    prediction_column: str,
    demographic_group_column: str | None = None,
    perturbation_type_column: str | None = None,
) -> dict[str, Any]:
    metrics = classification_metrics(frame[label_column].tolist(), frame[prediction_column].tolist())
    if demographic_group_column and demographic_group_column in frame.columns:
        metrics.update(bias_summary(frame, label_column, prediction_column, demographic_group_column))
    if perturbation_type_column and perturbation_type_column in frame.columns:
        metrics.update(robustness_summary(frame, label_column, prediction_column, perturbation_type_column))
    return metrics


def evaluate_model_on_eval_sets(
    cfg: ExperimentConfig,
    run_id: str,
    model: Any,
    tokenizer: Any,
    label_mapping: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        from transformers import Trainer
    except ImportError as exc:
        raise ImportError("Model evaluation requires the 'transformers' package.") from exc

    label2id = {label: int(index) for label, index in label_mapping["label2id"].items()}
    id2label = {int(index): label for index, label in label_mapping["id2label"].items()}
    rows: list[dict[str, Any]] = []
    trainer = Trainer(model=model, tokenizer=tokenizer)

    for eval_name, path in eval_set_paths(cfg).items():
        if not path:
            continue
        frame = load_text_classification_csv(cfg, path)
        frame = apply_label_mapping(frame, cfg.label_column, label2id)
        dataset = tokenize_frame(frame, tokenizer, cfg.text_column, cfg.max_length)
        with timer() as timing:
            prediction_output = trainer.predict(dataset)
        logits = prediction_output.predictions[0] if isinstance(prediction_output.predictions, tuple) else prediction_output.predictions
        predicted_ids = np.argmax(logits, axis=-1)
        prediction_frame = frame.copy()
        prediction_frame["prediction"] = [id2label[int(index)] for index in predicted_ids]
        metrics = evaluate_predictions_frame(
            prediction_frame,
            cfg.label_column,
            "prediction",
            cfg.data.get("demographic_group_column"),
            cfg.data.get("perturbation_type_column"),
        )
        metrics.update(
            {
                "run_id": run_id,
                "eval_set": eval_name,
                "num_examples": len(frame),
                "inference_time_per_1000_examples": (timing["elapsed"] / max(len(frame), 1)) * 1000,
            }
        )
        rows.append(metrics)
    return rows
