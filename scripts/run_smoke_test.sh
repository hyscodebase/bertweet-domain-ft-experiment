#!/usr/bin/env bash
set -euo pipefail

python3 -m compileall src
python3 -m pytest -q
python3 -m src.run_matrix --config configs/experiment.yaml --dry-run
