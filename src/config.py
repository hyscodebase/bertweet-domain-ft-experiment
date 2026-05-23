from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Thin wrapper around experiment YAML with path resolution helpers."""

    raw: dict[str, Any]
    config_path: Path
    root_dir: Path

    @property
    def project(self) -> dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def baseline(self) -> dict[str, Any]:
        return self.raw.get("baseline", {})

    @property
    def training(self) -> dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def methods(self) -> dict[str, Any]:
        return self.raw.get("methods", {})

    @property
    def text_column(self) -> str:
        return str(self.data.get("text_column", "text"))

    @property
    def label_column(self) -> str:
        return str(self.data.get("label_column", "label"))

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.project.get("output_dir", "outputs"))

    @property
    def max_length(self) -> int:
        return int(self.model.get("max_length", 128))

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.root_dir / candidate


def _infer_root_dir(config_path: Path) -> Path:
    if config_path.parent.name == "configs":
        return config_path.parent.parent
    return config_path.parent


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return ExperimentConfig(raw=raw, config_path=config_path, root_dir=_infer_root_dir(config_path))
