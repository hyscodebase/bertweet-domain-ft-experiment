#!/usr/bin/env bash
set -euo pipefail

python -m src.run_baseline --config configs/experiment.yaml
python -m src.run_matrix --config configs/experiment.yaml
python -m src.compare --config configs/experiment.yaml
python -m src.report --config configs/experiment.yaml
