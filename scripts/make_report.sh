#!/usr/bin/env bash
set -euo pipefail

python -m src.compare --config configs/experiment.yaml
python -m src.report --config configs/experiment.yaml
