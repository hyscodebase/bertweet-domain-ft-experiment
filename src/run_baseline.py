from __future__ import annotations

import argparse

from src.config import load_config
from src.train import run_baseline_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and evaluate the shared B0 baseline model.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics = run_baseline_training(cfg)
    print(f"Saved B0 metrics for {len(metrics)} evaluation rows.")


if __name__ == "__main__":
    main()
