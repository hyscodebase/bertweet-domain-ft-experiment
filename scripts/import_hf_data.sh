#!/usr/bin/env bash
set -euo pipefail

python -m src.hf_data --config configs/experiment.yaml --overwrite
