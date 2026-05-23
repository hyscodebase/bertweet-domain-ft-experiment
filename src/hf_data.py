from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from src.config import ExperimentConfig, load_config
from src.utils import ensure_dir


DATASET_SERVER = "https://datasets-server.huggingface.co"


def normalize_hf_rows(api_rows: list[dict[str, Any]], source: dict[str, Any]) -> pd.DataFrame:
    text_column = source.get("text_column", "text")
    label_column = source.get("label_column", "label")
    label_map = {str(key): value for key, value in source.get("label_map", {}).items()}
    drop_labels = {str(label) for label in source.get("drop_labels", [])}
    domain = source.get("domain")
    records: list[dict[str, Any]] = []

    for item in api_rows:
        row = item.get("row", item)
        raw_label = str(row[label_column])
        if raw_label in drop_labels:
            continue
        if label_map and raw_label not in label_map:
            continue
        records.append(
            {
                "text": str(row[text_column]).strip(),
                "label": label_map.get(raw_label, raw_label),
                "domain": domain or source.get("name", "hf"),
            }
        )
    return pd.DataFrame(records, columns=["text", "label", "domain"])


def fetch_rows(
    dataset: str,
    config: str | None,
    split: str,
    offset: int,
    length: int,
    token: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_offset = offset
    headers = {"Authorization": f"Bearer {token}"} if token else None
    while len(rows) < length:
        page_length = min(100, length - len(rows))
        params = {"dataset": dataset, "split": split, "offset": page_offset, "length": page_length}
        if config:
            params["config"] = config
        url = f"{DATASET_SERVER}/rows?{urlencode(params)}"
        response = _get_with_retries(url, headers=headers)
        response.raise_for_status()
        payload = response.json()
        page_rows = payload.get("rows", [])
        if not page_rows:
            break
        rows.extend(page_rows)
        page_offset += len(page_rows)
        if len(page_rows) < page_length:
            break
    return rows


def _get_with_retries(url: str, headers: dict[str, str] | None = None, max_attempts: int = 6) -> requests.Response:
    for attempt in range(max_attempts):
        response = requests.get(url, headers=headers, timeout=60)
        if response.status_code != 429:
            return response
        retry_after = response.headers.get("Retry-After")
        wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 30)
        time.sleep(wait_seconds)
    return response


def split_limit_across_offsets(limit: int, offsets: list[int]) -> list[tuple[int, int]]:
    if not offsets:
        return []
    base = limit // len(offsets)
    remainder = limit % len(offsets)
    return [(offset, base + (1 if index < remainder else 0)) for index, offset in enumerate(offsets)]


def fetch_normalized_split(source: dict[str, Any], split_spec: dict[str, Any], default_limit: int) -> pd.DataFrame:
    requested = int(split_spec.get("limit", default_limit))
    rows: list[dict[str, Any]] = []
    offsets = [int(offset) for offset in split_spec.get("offsets", [split_spec.get("offset", 0)])]

    for offset, offset_limit in split_limit_across_offsets(requested, offsets):
        rows.extend(_fetch_normalized_from_offset(source, split_spec, offset, offset_limit))
    return pd.DataFrame(rows[:requested], columns=["text", "label", "domain"])


def _fetch_normalized_from_offset(
    source: dict[str, Any],
    split_spec: dict[str, Any],
    offset: int,
    requested: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_offset = offset
    token = source.get("token")

    while len(rows) < requested:
        remaining = requested - len(rows)
        fetch_length = min(100, remaining * 2 if source.get("drop_labels") else remaining)
        page = fetch_rows(
            dataset=source["dataset"],
            config=source.get("config"),
            split=split_spec["split"],
            offset=page_offset,
            length=fetch_length,
            token=token,
        )
        if not page:
            break
        normalized = normalize_hf_rows(page, source)
        rows.extend(normalized.to_dict("records"))
        page_offset += len(page)
    return rows[:requested]


def _target_paths(cfg: ExperimentConfig) -> dict[str, dict[str, Path]]:
    data = cfg.data
    return {
        "baseline": {split: cfg.resolve_path(path) for split, path in data["baseline"].items()},
        "twitter": {split: cfg.resolve_path(path) for split, path in data["domains"]["twitter"].items()},
        "general": {split: cfg.resolve_path(path) for split, path in data["domains"]["general"].items()},
    }


def import_hf_data(cfg: ExperimentConfig, overwrite: bool = False) -> dict[str, Path]:
    hf_data = cfg.raw.get("hf_data", {})
    if not hf_data.get("enabled", True):
        raise ValueError("hf_data.enabled is false in the config.")

    limits = {
        "train": int(hf_data.get("max_train_examples", 512)),
        "valid": int(hf_data.get("max_valid_examples", 128)),
        "test": int(hf_data.get("max_test_examples", 128)),
    }
    target_paths = _target_paths(cfg)
    written: dict[str, Path] = {}

    for name, source in hf_data.get("sources", {}).items():
        source = {**source, "name": name, "domain": source.get("domain", name)}
        if name not in target_paths:
            continue
        for target_split, split_spec in source.get("splits", {}).items():
            target_path = target_paths[name][target_split]
            if target_path.exists() and not overwrite:
                written[f"{name}.{target_split}"] = target_path
                continue
            frame = fetch_normalized_split(source, split_spec, limits[target_split])
            if frame.empty:
                raise ValueError(f"No rows imported for {name}.{target_split}")
            ensure_dir(target_path.parent)
            frame.to_csv(target_path, index=False)
            written[f"{name}.{target_split}"] = target_path

    _write_derived_evals(cfg, overwrite=overwrite)
    return written


def _write_derived_evals(cfg: ExperimentConfig, overwrite: bool) -> None:
    baseline_test = cfg.resolve_path(cfg.data["baseline"]["test"])
    if not baseline_test.exists():
        return
    source = pd.read_csv(baseline_test)

    bias_path = cfg.resolve_path(cfg.data["bias_eval"])
    if overwrite or not bias_path.exists():
        bias = source.copy()
        bias["demographic_group"] = ["proxy_group_a" if index % 2 == 0 else "proxy_group_b" for index in range(len(bias))]
        ensure_dir(bias_path.parent)
        bias[["text", "label", "demographic_group"]].to_csv(bias_path, index=False)

    robustness_path = cfg.resolve_path(cfg.data["robustness_eval"])
    if overwrite or not robustness_path.exists():
        clean = source.copy()
        clean["perturbation_type"] = "clean"
        clean["source_id"] = [f"hf_{index}" for index in range(len(clean))]
        perturbed = clean.copy()
        perturbed["text"] = perturbed["text"].map(_simple_typo)
        perturbed["perturbation_type"] = "typo"
        robustness = pd.concat([clean, perturbed], ignore_index=True)
        ensure_dir(robustness_path.parent)
        robustness[["text", "label", "perturbation_type", "source_id"]].to_csv(robustness_path, index=False)


def _simple_typo(text: str) -> str:
    text = str(text)
    return text.replace("the", "teh", 1) if "the" in text.lower() else f"{text} ..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Import configured Hugging Face datasets into project CSV files.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = import_hf_data(load_config(args.config), overwrite=args.overwrite)
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
