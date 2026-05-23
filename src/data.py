from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ExperimentConfig
from src.utils import ensure_dir, write_json


SMOKE_ROWS = [
    {"text": "I love this update", "label": "positive", "domain": "smoke"},
    {"text": "This rollout is terrible", "label": "negative", "domain": "smoke"},
    {"text": "Great news for everyone", "label": "positive", "domain": "smoke"},
    {"text": "I dislike this change", "label": "negative", "domain": "smoke"},
]


BIAS_ROWS = [
    {"text": "Group A likes this", "label": "positive", "demographic_group": "group_a"},
    {"text": "Group A dislikes this", "label": "negative", "demographic_group": "group_a"},
    {"text": "Group B likes this", "label": "positive", "demographic_group": "group_b"},
    {"text": "Group B dislikes this", "label": "negative", "demographic_group": "group_b"},
]


ROBUSTNESS_ROWS = [
    {"text": "I love this update", "label": "positive", "perturbation_type": "clean", "source_id": "s1"},
    {"text": "I luv this update", "label": "positive", "perturbation_type": "typo", "source_id": "s1"},
    {"text": "This rollout is terrible", "label": "negative", "perturbation_type": "clean", "source_id": "s2"},
    {"text": "This rollout is terrrible", "label": "negative", "perturbation_type": "typo", "source_id": "s2"},
]


def _write_csv_if_missing(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        return
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def ensure_smoke_data(cfg: ExperimentConfig) -> None:
    """Create tiny CSVs for pipeline checks when real data is absent."""

    baseline = cfg.data.get("baseline", {})
    for split in ("train", "valid", "test"):
        if split in baseline:
            _write_csv_if_missing(cfg.resolve_path(baseline[split]), SMOKE_ROWS)

    for domain_name, paths in cfg.data.get("domains", {}).items():
        rows = [{**row, "domain": domain_name} for row in SMOKE_ROWS]
        for split in ("train", "valid", "test"):
            if split in paths:
                _write_csv_if_missing(cfg.resolve_path(paths[split]), rows)

    if cfg.data.get("bias_eval"):
        _write_csv_if_missing(cfg.resolve_path(cfg.data["bias_eval"]), BIAS_ROWS)
    if cfg.data.get("robustness_eval"):
        _write_csv_if_missing(cfg.resolve_path(cfg.data["robustness_eval"]), ROBUSTNESS_ROWS)


def load_text_classification_csv(cfg: ExperimentConfig, path: str | Path) -> pd.DataFrame:
    resolved = cfg.resolve_path(path)
    frame = pd.read_csv(resolved)
    required = {cfg.text_column, cfg.label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{resolved} is missing required columns: {sorted(missing)}")
    return frame


def build_label_mapping(frame: pd.DataFrame, label_column: str = "label") -> dict[str, dict[str, int] | dict[int, str]]:
    labels = sorted(str(label) for label in frame[label_column].dropna().unique())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return {"label2id": label2id, "id2label": id2label}


def save_label_mapping(cfg: ExperimentConfig, mapping: dict[str, Any]) -> Path:
    return write_json(cfg.output_dir / "label_mapping.json", mapping)


def apply_label_mapping(frame: pd.DataFrame, label_column: str, label2id: dict[str, int]) -> pd.DataFrame:
    mapped = frame.copy()
    unknown = sorted(set(mapped[label_column].astype(str)) - set(label2id))
    if unknown:
        raise ValueError(f"Found labels not present in baseline mapping: {unknown}")
    mapped["labels"] = mapped[label_column].astype(str).map(label2id)
    return mapped


def tokenize_frame(frame: pd.DataFrame, tokenizer: Any, text_column: str, max_length: int) -> Any:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Tokenization requires the 'datasets' package.") from exc

    dataset = Dataset.from_pandas(frame, preserve_index=False)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(batch[text_column], max_length=max_length, padding="max_length", truncation=True)

    remove_columns = [column for column in dataset.column_names if column != "labels"]
    return dataset.map(tokenize, batched=True, remove_columns=remove_columns)
