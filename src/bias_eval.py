from __future__ import annotations

import argparse

import pandas as pd

from src.config import load_config
from src.metrics import bias_summary
from src.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bias metrics from a predictions CSV.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--predictions", required=True, help="CSV containing label and prediction columns.")
    parser.add_argument("--prediction-column", default="prediction")
    args = parser.parse_args()
    cfg = load_config(args.config)
    frame = pd.read_csv(args.predictions)
    metrics = bias_summary(
        frame,
        cfg.label_column,
        args.prediction_column,
        cfg.data.get("demographic_group_column", "demographic_group"),
    )
    output_path = cfg.output_dir / "metrics" / "bias_eval_metrics.json"
    write_json(output_path, metrics)
    print(f"Bias metrics written to {output_path}")


if __name__ == "__main__":
    main()
