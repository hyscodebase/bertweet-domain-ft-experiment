# BERTweet Domain Fine-Tuning Baseline Delta Experiment

This repository implements a B0-centered BERTweet fine-tuning study.

The goal is not to ask whether full fine-tuning, LoRA, or adapters are best in isolation. The goal is to create one shared baseline model, B0, then measure how each domain-specific treatment changes performance, efficiency, bias, and robustness relative to that same B0.

## Experimental Flow

```text
vinai/bertweet-base
  -> baseline dataset + common recipe
  -> B0 baseline checkpoint
  -> twitter/general additional fine-tuning from B0
  -> evaluate every model on the same eval sets
  -> report raw metrics and B0-relative deltas
```

## Matrix

| Run | Domain | Method | Detail |
| --- | --- | --- | --- |
| B0 | baseline | full_ft | common baseline recipe |
| T1 | twitter | full_ft | all parameters trainable |
| T2 | twitter | lora | rank 8 |
| T3 | twitter | lora | rank 16 |
| T4 | twitter | lora | rank 32 |
| T5 | twitter | adapter | bottleneck 64 |
| G1 | general | full_ft | all parameters trainable |
| G2 | general | lora | rank 8 |
| G3 | general | lora | rank 16 |
| G4 | general | lora | rank 32 |
| G5 | general | adapter | bottleneck 64 |

## Data Schema

All data files are CSV files. Sequence classification files require:

- `text`
- `label`

Optional metadata columns:

- `domain`
- `demographic_group`
- `perturbation_type`
- `source_id`

The label mapping is built from `data/baseline/train.csv` and saved to `outputs/label_mapping.json`. Treatment datasets should use labels compatible with that baseline mapping.

If expected CSV files are missing, the code can create tiny synthetic smoke-test files. These are only for pipeline checks and must not be used for research conclusions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

`vinai/bertweet-base` requires internet access or a local Hugging Face cache.

## Google Colab GPU

Use [notebooks/bertweet_domain_ft_colab_gpu.ipynb](notebooks/bertweet_domain_ft_colab_gpu.ipynb) in Google Colab.

1. Open the notebook in Colab.
2. Select a GPU runtime in Colab.
3. Push this repo to GitHub and set `REPO_URL` in the notebook, or upload/clone the repo to `/content/bertweet-domain-ft-experiment`.
4. Run the notebook cells in order.

The Colab notebook uses `configs/experiment_colab.yaml`, installs `requirements.txt`, imports the Hugging Face datasets, trains B0 on GPU, writes `outputs_colab/tables/baseline_metrics.csv`, and generates `outputs_colab/report.md`.

## Commands

Import default Hugging Face datasets into the CSV layout:

```bash
python -m src.hf_data --config configs/experiment.yaml --overwrite
```

Smoke checks:

```bash
bash scripts/run_smoke_test.sh
```

Create B0:

```bash
python -m src.run_baseline --config configs/experiment.yaml
```

Run the treatment matrix:

```bash
python -m src.run_matrix --config configs/experiment.yaml
```

Compare against B0:

```bash
python -m src.compare --config configs/experiment.yaml
```

Generate report:

```bash
python -m src.report --config configs/experiment.yaml
```

## Outputs

- `outputs/checkpoints/B0`
- `outputs/metrics/baseline_metrics.json`
- `outputs/metrics/treatment_metrics.json`
- `outputs/tables/baseline_metrics.csv`
- `outputs/tables/treatment_metrics.csv`
- `outputs/tables/delta_vs_baseline.csv`
- `outputs/tables/efficiency.csv`
- `outputs/tables/bias_summary.csv`
- `outputs/tables/domain_gap_summary.csv`
- `outputs/tables/robustness_summary.csv`
- `outputs/tables/final_ranking.csv`
- `outputs/report.md`

## How To Read Deltas

Every treatment is compared against B0 on the same evaluation set.

- `delta_f1_macro > 0`: treatment improved over B0.
- `delta_accuracy > 0`: treatment improved accuracy over B0.
- `delta_bias_gap > 0`: demographic gap worsened relative to B0.
- `delta_domain_gap > 0`: domain imbalance worsened relative to B0.
- `delta_robustness_drop > 0`: robustness worsened relative to B0.
- `f1_retention_vs_baseline`: treatment F1 divided by B0 F1 on the same eval set.

The final ranking combines F1, efficiency, fairness, and robustness with configurable weights. It is intentionally not an F1-only ranking.

## Research Warning

Results depend on dataset size, label quality, demographic annotation quality, and domain distribution. Synthetic data is only a wiring check.
