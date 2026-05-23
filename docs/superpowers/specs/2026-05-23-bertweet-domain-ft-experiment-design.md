# BERTweet Domain Fine-Tuning Experiment Design

## Purpose

This project builds a reproducible experiment repository for studying domain-specific additional fine-tuning from a shared BERTweet baseline. The experiment is not a direct contest between full fine-tuning, LoRA, and adapters. Instead, it first creates a shared baseline model, B0, then measures how each domain-specific treatment changes performance, efficiency, bias, and robustness relative to that same B0.

The central research question is:

> Starting from the same BERTweet-based baseline model B0, how do domain-specific fine-tuning methods change performance, efficiency, bias, and robustness across Twitter and general-domain evaluation sets?

## Project Root

The repository lives at:

`/Users/hong-yuseog/Desktop/AIX/bertweet-domain-ft-experiment`

This keeps the experiment isolated from other AIX work while leaving room for future papers, slides, and related projects under `/Users/hong-yuseog/Desktop/AIX`.

## Baseline-Centered Flow

1. Load the shared pretrained checkpoint `vinai/bertweet-base`.
2. Add a sequence classification head.
3. Fine-tune on the baseline dataset using the common baseline recipe.
4. Save the resulting checkpoint as `outputs/checkpoints/B0`.
5. Evaluate B0 on all evaluation sets.
6. Start every treatment run from `outputs/checkpoints/B0`.
7. Fine-tune treatments by domain and method.
8. Evaluate every treatment on the same evaluation sets as B0.
9. Report raw metrics and B0-relative deltas.

Treatment domains:

- `twitter`
- `general`

Treatment methods:

- `full_ft`
- `lora` with ranks `8`, `16`, and `32`
- `adapter` with configurable bottleneck size, default `64`

## Repository Structure

```text
bertweet-domain-ft-experiment/
  README.md
  requirements.txt
  configs/
    experiment.yaml
  data/
    README.md
    baseline/
      train.csv
      valid.csv
      test.csv
    twitter/
      train.csv
      valid.csv
      test.csv
    general/
      train.csv
      valid.csv
      test.csv
    bias/
      bias_eval.csv
    robustness/
      robustness_eval.csv
  docs/
    superpowers/
      specs/
      plans/
  src/
    __init__.py
    config.py
    seed.py
    data.py
    preprocessing.py
    model_factory.py
    methods/
      __init__.py
      full_ft.py
      lora.py
      adapter.py
    train.py
    evaluate.py
    bias_eval.py
    robustness_eval.py
    metrics.py
    run_baseline.py
    run_matrix.py
    compare.py
    report.py
    utils.py
  scripts/
    run_smoke_test.sh
    run_baseline.sh
    run_all_experiments.sh
    make_report.sh
  outputs/
    checkpoints/
    metrics/
    tables/
    figures/
    report.md
  tests/
    test_data_loading.py
    test_metrics.py
    test_experiment_matrix.py
```

## Data Design

All data files are CSV files. Sequence classification datasets require:

- `text`
- `label`

Optional metadata columns:

- `domain`
- `demographic_group`
- `perturbation_type`
- `source_id`

The code must not assume fixed label names. It builds `label2id` and `id2label` from the baseline training data and saves the mapping to `outputs/label_mapping.json`. Domain-specific treatment data must use labels compatible with that baseline mapping.

If real data files are missing, the repository provides tiny synthetic smoke-test CSVs. These are explicitly marked as smoke-test data only and are not suitable for research conclusions.

## Configuration

`configs/experiment.yaml` is the single control surface for:

- project output directory and seed
- base model id
- data paths and column names
- training hyperparameters
- LoRA ranks and LoRA hyperparameters
- adapter bottleneck size
- experiment matrix
- evaluation set list
- ranking weights

The default task type is `sequence_classification`. The code should keep task boundaries clear enough that token classification can be added later without rewriting the full repository.

## Training Architecture

The implementation uses PyTorch, Hugging Face Transformers, Datasets, scikit-learn, PEFT, and PyYAML. Hugging Face `Trainer` is preferred for the first version.

Core modules:

- `config.py`: load and validate YAML config into a structured object.
- `seed.py`: seed Python, NumPy, PyTorch, and CUDA when available.
- `data.py`: load CSV files, create smoke data if needed, build labels, and tokenize datasets.
- `model_factory.py`: create sequence classification models from base model or B0 checkpoint.
- `methods/full_ft.py`: mark all parameters trainable.
- `methods/lora.py`: apply PEFT LoRA, auto-detect RoBERTa-style target modules, and keep the classification head trainable.
- `methods/adapter.py`: use an adapter-compatible implementation when available, otherwise inject a simple bottleneck adapter into transformer layer outputs.
- `train.py`: shared Trainer orchestration and run metric persistence.
- `evaluate.py`: evaluate one model on all configured eval sets.
- `run_baseline.py`: create B0 and save B0 metrics.
- `run_matrix.py`: run all treatment combinations from B0.
- `compare.py`: compute B0-relative deltas and summary tables.
- `report.py`: generate the final Markdown report.

## Fine-Tuning Methods

### Full Fine-Tuning

All model parameters are trainable. This is the high-cost and high-flexibility condition.

### LoRA

LoRA starts from B0, freezes base model parameters, and trains LoRA parameters plus the classification head. It uses PEFT `LoraConfig` and `get_peft_model`. Rank values come from config. The code prints the exact target module names receiving LoRA.

Default target module strategy:

- include attention projections for RoBERTa/BERTweet-style architectures
- include MLP or dense modules only when enabled in config

### Adapter

Adapter starts from B0, freezes base model parameters, and trains adapter parameters plus the classification head. If an external adapter library is missing or incompatible, the fallback adapter is:

- down projection: `hidden_size -> bottleneck_size`
- activation: GELU
- up projection: `bottleneck_size -> hidden_size`
- residual: `hidden_states + adapter(hidden_states)`

The fallback adapter is inserted after transformer layer outputs where feasible. The code prints adapter parameter counts.

## Evaluation

B0 and every treatment model are evaluated on:

- `baseline_test`
- `twitter_test`
- `general_test`
- `bias_eval`
- `robustness_eval`

Classification metrics:

- accuracy
- precision_macro
- recall_macro
- f1_macro
- f1_weighted
- per_class_f1

Efficiency metrics:

- trainable_params
- total_params
- trainable_param_ratio
- training_time_seconds
- inference_time_per_1000_examples
- peak_gpu_memory_mb when CUDA is available

Convergence metrics:

- best_eval_f1_macro
- best_epoch
- global_step_at_best
- convergence_step_to_95_percent_best_score when available from training history

## Bias, Domain, And Robustness Analysis

Bias evaluation uses `demographic_group` when present and computes:

- group_f1_macro
- group_accuracy
- max_group_f1_gap
- max_group_accuracy_gap
- worst_group
- best_group
- binary false positive and false negative gaps when applicable

Domain bias compares each model's Twitter and general test performance:

- `domain_gap_f1 = f1_macro_twitter_test - f1_macro_general_test`
- `absolute_domain_gap_f1 = abs(domain_gap_f1)`

Robustness evaluation uses `perturbation_type` when present and computes:

- f1_macro by perturbation type
- `robustness_drop = clean_eval_f1_macro - perturbed_eval_f1_macro`
- pairwise original/perturbed comparison when `source_id` exists

Every treatment's bias, domain gap, and robustness metrics are compared against B0.

## Output Artifacts

The repository generates:

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

## Delta Rules

The comparison layer must always compute deltas against B0 on the same evaluation set:

- `delta_f1_macro = model_f1_macro - B0_f1_macro_on_same_eval_set`
- `delta_accuracy = model_accuracy - B0_accuracy_on_same_eval_set`
- `delta_bias_gap = model_bias_gap - B0_bias_gap`
- `delta_domain_gap = model_domain_gap - B0_domain_gap`
- `delta_robustness_drop = model_robustness_drop - B0_robustness_drop`
- `f1_retention_vs_baseline = model_f1_macro / B0_f1_macro_on_same_eval_set`

The code should handle zero or missing baseline denominators without crashing and mark retention as null when it cannot be computed.

## Final Ranking

The final ranking is not based only on F1. It uses configurable weights:

```text
composite_score =
  0.40 * normalized_f1_macro
  + 0.20 * normalized_efficiency
  + 0.20 * normalized_robustness
  + 0.20 * normalized_fairness
```

Efficiency rewards lower trainable parameter ratio and lower training time. Fairness rewards lower demographic bias gaps. Robustness rewards lower robustness drop.

## Testing Strategy

The first tests cover stable, offline logic:

- data loading creates and reads smoke-test CSVs
- label mapping is built from training data and applied consistently
- classification, bias, domain, and robustness metrics are computed correctly
- experiment matrix expands to the expected ten treatment runs
- delta tables compare treatment metrics against B0 on matching evaluation sets

Model download and full training are not required for unit tests. The smoke test command verifies repository wiring on tiny synthetic data, but it may still require model access unless a local tiny model is explicitly configured later.

## Risks And Constraints

- `vinai/bertweet-base` requires internet access or a local Hugging Face cache.
- Adapter libraries can be version-sensitive, so the fallback bottleneck adapter is part of the design.
- Synthetic smoke data proves the pipeline shape only. It must not be used for research claims.
- Real conclusions depend on dataset size, label quality, demographic annotation quality, and domain distribution.
- MPS support can differ from CUDA support, so peak GPU memory is guaranteed only for CUDA.

## Acceptance Criteria

The first complete version is accepted when:

1. The documented repository structure exists.
2. `configs/experiment.yaml` includes the full B0-centered matrix.
3. Missing data paths can be filled with clearly marked synthetic smoke data.
4. `python -m src.run_baseline --config configs/experiment.yaml` is implemented.
5. `python -m src.run_matrix --config configs/experiment.yaml` is implemented.
6. `python -m src.compare --config configs/experiment.yaml` is implemented.
7. `python -m src.report --config configs/experiment.yaml` is implemented.
8. Unit tests pass for data loading, metrics, and matrix generation.
9. A smoke test script exists.
10. README explains the B0-relative interpretation clearly.
