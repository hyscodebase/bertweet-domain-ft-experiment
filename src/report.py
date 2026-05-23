from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from src.config import ExperimentConfig, load_config
from src.utils import ensure_dir


def _table_preview(path: Path, max_rows: int = 12) -> str:
    if not path.exists():
        return "_Not generated yet._"
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return "_No rows._"
    if frame.empty:
        return "_No rows._"
    preview = frame.head(max_rows).fillna("")
    columns = list(preview.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _index, row in preview.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_report(cfg: ExperimentConfig) -> Path:
    tables_dir = cfg.output_dir / "tables"
    report_path = cfg.output_dir / "report.md"
    ensure_dir(report_path.parent)
    sections = [
        "# BERTweet Domain Fine-Tuning Baseline Delta Report",
        "",
        "## Research Question",
        "",
        "Starting from the same BERTweet-based baseline model B0, how do domain-specific fine-tuning methods change performance, efficiency, bias, and robustness across Twitter and general-domain evaluation sets?",
        "",
        "## Interpretation Rule",
        "",
        "Every treatment row should be read relative to B0 on the same evaluation set. Positive `delta_f1_macro` means the treatment improved over B0. Positive `delta_bias_gap` or `delta_robustness_drop` means the treatment worsened that risk metric.",
        "",
        "## Baseline Metrics",
        "",
        _table_preview(tables_dir / "baseline_metrics.csv"),
        "",
        "## Treatment Metrics",
        "",
        _table_preview(tables_dir / "treatment_metrics.csv"),
        "",
        "## Delta Vs Baseline",
        "",
        _table_preview(tables_dir / "delta_vs_baseline.csv"),
        "",
        "## Final Ranking",
        "",
        _table_preview(tables_dir / "final_ranking.csv"),
        "",
        "## Caution",
        "",
        "Synthetic smoke-test data only verifies the pipeline. Research claims require real, sufficiently sized, well-labeled domain, bias, and robustness evaluation datasets.",
        "",
    ]
    report_path.write_text("\n".join(sections), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown experiment report.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    path = generate_report(load_config(args.config))
    print(f"Report written to {path}")


if __name__ == "__main__":
    main()
