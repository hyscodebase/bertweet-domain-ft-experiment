from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, data: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    pd.DataFrame(rows).to_csv(target, index=False)
    return target


def count_parameters(model: Any) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    ratio = trainable / total if total else 0.0
    return {"total_params": total, "trainable_params": trainable, "trainable_param_ratio": ratio}


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    state: dict[str, float] = {"start": time.perf_counter(), "elapsed": 0.0}
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - state["start"]
