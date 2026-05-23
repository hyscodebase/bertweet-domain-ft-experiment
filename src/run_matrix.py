from __future__ import annotations

import argparse
from typing import Any

from src.config import ExperimentConfig, load_config
from src.utils import write_json


def _raw_config(config: ExperimentConfig | dict[str, Any]) -> dict[str, Any]:
    return config.raw if isinstance(config, ExperimentConfig) else config


def build_experiment_matrix(config: ExperimentConfig | dict[str, Any]) -> list[dict[str, Any]]:
    raw = _raw_config(config)
    matrix = raw.get("experiment_matrix", {})
    methods = raw.get("methods", {})
    domains = matrix.get("domains", [])
    method_names = matrix.get("methods", [])
    runs: list[dict[str, Any]] = []

    for domain in domains:
        for method in method_names:
            if method == "full_ft":
                runs.append({"domain": domain, "method": "full_ft", "run_id": f"{domain}_full_ft"})
            elif method == "lora":
                for rank in methods.get("lora", {}).get("ranks", []):
                    runs.append(
                        {
                            "domain": domain,
                            "method": "lora",
                            "lora_rank": int(rank),
                            "run_id": f"{domain}_lora_r{int(rank)}",
                        }
                    )
            elif method == "adapter":
                bottleneck = int(methods.get("adapter", {}).get("bottleneck_size", 64))
                runs.append(
                    {
                        "domain": domain,
                        "method": "adapter",
                        "adapter_bottleneck_size": bottleneck,
                        "run_id": f"{domain}_adapter_b{bottleneck}",
                    }
                )
            else:
                raise ValueError(f"Unsupported fine-tuning method: {method}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the B0-starting treatment experiment matrix.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only write the expanded matrix.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runs = build_experiment_matrix(cfg)
    output_path = cfg.output_dir / "metrics" / "experiment_matrix.json"
    write_json(output_path, runs)
    print(f"Saved experiment matrix with {len(runs)} runs to {output_path}")

    if not args.dry_run:
        from src.train import run_treatment_matrix

        run_treatment_matrix(cfg, runs)


if __name__ == "__main__":
    main()
