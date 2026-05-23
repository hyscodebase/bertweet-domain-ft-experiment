from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ExperimentConfig, load_config
from src.metrics import composite_ranking, compute_delta_rows, domain_gap_summary
from src.utils import read_json, write_rows_csv


def _load_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("rows", "metrics"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def _efficiency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = row.get("run_id")
        if run_id and run_id not in seen:
            seen[run_id] = {
                "run_id": run_id,
                "method": row.get("method"),
                "domain": row.get("domain"),
                "trainable_params": row.get("trainable_params"),
                "total_params": row.get("total_params"),
                "trainable_param_ratio": row.get("trainable_param_ratio"),
                "training_time_seconds": row.get("training_time_seconds"),
                "inference_time_per_1000_examples": row.get("inference_time_per_1000_examples"),
            }
    return list(seen.values())


def _domain_gap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row.get("eval_set") in {"twitter_test", "general_test"}:
            grouped[row["run_id"]][row["eval_set"]] = row.get("f1_macro")
    output = []
    for run_id, scores in grouped.items():
        if "twitter_test" in scores and "general_test" in scores:
            output.append(domain_gap_summary(run_id, scores["twitter_test"], scores["general_test"]))
    return output


def _summary_for_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("run_id")].append(row)
    summary = []
    for run_id, group_rows in grouped.items():
        if not run_id:
            continue
        f1_values = [row.get("f1_macro") for row in group_rows if row.get("f1_macro") is not None]
        bias_gaps = [row.get("max_group_f1_gap") for row in group_rows if row.get("max_group_f1_gap") is not None]
        robustness_drops = [row.get("robustness_drop") for row in group_rows if row.get("robustness_drop") is not None]
        first = group_rows[0]
        summary.append(
            {
                "run_id": run_id,
                "domain": first.get("domain"),
                "method": first.get("method"),
                "f1_macro": sum(f1_values) / len(f1_values) if f1_values else None,
                "trainable_param_ratio": first.get("trainable_param_ratio"),
                "training_time_seconds": first.get("training_time_seconds"),
                "max_group_f1_gap": max(bias_gaps) if bias_gaps else None,
                "robustness_drop": max(robustness_drops) if robustness_drops else None,
            }
        )
    return summary


def compare_metrics(cfg: ExperimentConfig) -> dict[str, Path]:
    baseline_rows = _load_metric_rows(cfg.output_dir / "metrics" / "baseline_metrics.json")
    treatment_rows = _load_metric_rows(cfg.output_dir / "metrics" / "treatment_metrics.json")
    tables_dir = cfg.output_dir / "tables"
    outputs = {
        "baseline_metrics": write_rows_csv(tables_dir / "baseline_metrics.csv", baseline_rows),
        "treatment_metrics": write_rows_csv(tables_dir / "treatment_metrics.csv", treatment_rows),
        "delta_vs_baseline": write_rows_csv(
            tables_dir / "delta_vs_baseline.csv",
            compute_delta_rows(baseline_rows, treatment_rows),
        ),
        "efficiency": write_rows_csv(tables_dir / "efficiency.csv", _efficiency_rows(baseline_rows + treatment_rows)),
        "domain_gap_summary": write_rows_csv(
            tables_dir / "domain_gap_summary.csv",
            _domain_gap_rows(baseline_rows + treatment_rows),
        ),
    }
    bias_rows = [row for row in baseline_rows + treatment_rows if row.get("max_group_f1_gap") is not None]
    robustness_rows = [row for row in baseline_rows + treatment_rows if row.get("robustness_drop") is not None]
    outputs["bias_summary"] = write_rows_csv(tables_dir / "bias_summary.csv", bias_rows)
    outputs["robustness_summary"] = write_rows_csv(tables_dir / "robustness_summary.csv", robustness_rows)
    ranking_rows = composite_ranking(_summary_for_ranking(treatment_rows), cfg.raw.get("ranking", {}).get("weights", {}))
    outputs["final_ranking"] = write_rows_csv(tables_dir / "final_ranking.csv", ranking_rows)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare treatment metrics against B0.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    outputs = compare_metrics(load_config(args.config))
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
