from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import ExperimentConfig
from src.seed import detect_device


def load_tokenizer(model_name_or_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Tokenizer loading requires the 'transformers' package.") from exc

    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, normalization=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)


def load_sequence_classifier(
    model_name_or_path: str | Path,
    num_labels: int,
    label_mapping: dict[str, Any],
):
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError("Model loading requires the 'transformers' package.") from exc

    id2label = {int(index): label for index, label in label_mapping["id2label"].items()}
    label2id = {label: int(index) for label, index in label_mapping["label2id"].items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_name_or_path),
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    device = detect_device()
    if device != "cpu":
        model.to(device)
    return model


def load_base_model_and_tokenizer(cfg: ExperimentConfig, label_mapping: dict[str, Any]):
    tokenizer = load_tokenizer(cfg.model["base_model_name"])
    model = load_sequence_classifier(
        cfg.model["base_model_name"],
        len(label_mapping["label2id"]),
        label_mapping,
    )
    return model, tokenizer


def load_b0_model_and_tokenizer(cfg: ExperimentConfig, label_mapping: dict[str, Any]):
    checkpoint = cfg.resolve_path(cfg.baseline.get("save_dir", "outputs/checkpoints/B0"))
    tokenizer = load_tokenizer(str(checkpoint) if checkpoint.exists() else cfg.model["base_model_name"])
    model = load_sequence_classifier(checkpoint, len(label_mapping["label2id"]), label_mapping)
    return model, tokenizer
