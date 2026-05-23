# BERTweet Domain Fine-Tuning Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible BERTweet fine-tuning experiment repository where every domain-specific treatment starts from B0 and is compared against B0 with performance, efficiency, bias, domain, and robustness deltas.

**Architecture:** The repository is a config-driven Python package. Offline unit tests cover data loading, metric computation, experiment matrix expansion, delta computation, and report generation, while training modules use Hugging Face Trainer and can run when model dependencies and data are available.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, Datasets, PEFT, scikit-learn, pandas, PyYAML, pytest.

---

## File Structure

- Create `README.md`: user-facing purpose, setup, commands, and interpretation guide.
- Create `requirements.txt`: runtime and test dependencies.
- Create `configs/experiment.yaml`: default B0-centered experiment config.
- Create `data/README.md`: CSV schemas and smoke-test warning.
- Create `src/config.py`: YAML loading and path helpers.
- Create `src/seed.py`: reproducibility utilities.
- Create `src/data.py`: CSV loading, smoke data creation, label mapping, and optional tokenization helpers.
- Create `src/metrics.py`: classification, bias, domain, robustness, efficiency, normalization, and delta metrics.
- Create `src/model_factory.py`: model and tokenizer construction.
- Create `src/methods/full_ft.py`: full fine-tuning parameter policy.
- Create `src/methods/lora.py`: PEFT LoRA setup and module detection.
- Create `src/methods/adapter.py`: fallback bottleneck adapter setup.
- Create `src/train.py`: shared Trainer orchestration.
- Create `src/evaluate.py`: model evaluation helpers.
- Create `src/bias_eval.py`: CLI wrapper for bias summaries.
- Create `src/robustness_eval.py`: CLI wrapper for robustness summaries.
- Create `src/run_baseline.py`: B0 training/evaluation CLI.
- Create `src/run_matrix.py`: treatment matrix CLI.
- Create `src/compare.py`: summary table and delta CLI.
- Create `src/report.py`: Markdown report CLI.
- Create `src/utils.py`: filesystem, JSON, CSV, timer, and parameter helpers.
- Create shell scripts in `scripts/`: smoke test, baseline, matrix, report.
- Create tests in `tests/`: data loading, metrics/deltas, matrix generation.

## Task 1: Data Loading And Matrix Tests

**Files:**
- Create: `tests/test_data_loading.py`
- Create: `tests/test_experiment_matrix.py`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/data.py`
- Create: `src/run_matrix.py`
- Create: `src/utils.py`
- Create: `configs/experiment.yaml`

- [ ] **Step 1: Write failing tests for smoke data and label mapping**

```python
from pathlib import Path

from src.config import load_config
from src.data import ensure_smoke_data, load_text_classification_csv, build_label_mapping


def test_smoke_data_is_created_when_missing(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
project:
  name: test
  output_dir: outputs
  seed: 7
model:
  base_model_name: hf-internal-testing/tiny-random-roberta
  task_type: sequence_classification
  num_labels: null
  max_length: 32
data:
  text_column: text
  label_column: label
  demographic_group_column: demographic_group
  perturbation_type_column: perturbation_type
  source_id_column: source_id
  baseline:
    train: data/baseline/train.csv
    valid: data/baseline/valid.csv
    test: data/baseline/test.csv
  domains:
    twitter:
      train: data/twitter/train.csv
      valid: data/twitter/valid.csv
      test: data/twitter/test.csv
    general:
      train: data/general/train.csv
      valid: data/general/valid.csv
      test: data/general/test.csv
  bias_eval: data/bias/bias_eval.csv
  robustness_eval: data/robustness/robustness_eval.csv
baseline:
  run_id: B0
  method: full_ft
  save_dir: outputs/checkpoints/B0
training:
  epochs: 1
  batch_size: 2
  eval_batch_size: 2
  learning_rate: 0.00002
  weight_decay: 0.01
  warmup_ratio: 0.0
  gradient_accumulation_steps: 1
  fp16: false
  bf16: false
methods:
  lora:
    ranks: [8, 16, 32]
  adapter:
    bottleneck_size: 64
experiment_matrix:
  domains: [twitter, general]
  methods: [full_ft, lora, adapter]
ranking:
  weights:
    f1_macro: 0.4
    efficiency: 0.2
    robustness: 0.2
    fairness: 0.2
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    ensure_smoke_data(cfg)
    train = load_text_classification_csv(cfg, cfg.data["baseline"]["train"])
    mapping = build_label_mapping(train, cfg.label_column)
    assert set(train.columns) >= {"text", "label"}
    assert mapping["label2id"] == {"negative": 0, "positive": 1}
```

- [ ] **Step 2: Write failing tests for the experiment matrix**

```python
from src.run_matrix import build_experiment_matrix


def test_experiment_matrix_has_expected_treatments():
    config = {
        "experiment_matrix": {
            "domains": ["twitter", "general"],
            "methods": ["full_ft", "lora", "adapter"],
        },
        "methods": {"lora": {"ranks": [8, 16, 32]}, "adapter": {"bottleneck_size": 64}},
    }
    runs = build_experiment_matrix(config)
    assert len(runs) == 10
    assert {"domain": "twitter", "method": "full_ft", "run_id": "twitter_full_ft"} in runs
    assert {"domain": "general", "method": "lora", "lora_rank": 32, "run_id": "general_lora_r32"} in runs
    assert {"domain": "twitter", "method": "adapter", "adapter_bottleneck_size": 64, "run_id": "twitter_adapter_b64"} in runs
```

- [ ] **Step 3: Verify tests fail**

Run: `pytest tests/test_data_loading.py tests/test_experiment_matrix.py -q`

Expected: FAIL because the modules do not exist yet.

- [ ] **Step 4: Implement config, data, and matrix helpers**

Implement the files listed above with enough behavior to satisfy the tests and support later tasks.

- [ ] **Step 5: Verify tests pass**

Run: `pytest tests/test_data_loading.py tests/test_experiment_matrix.py -q`

Expected: PASS.

## Task 2: Metrics, Bias, Robustness, Ranking, And Delta Tests

**Files:**
- Create: `tests/test_metrics.py`
- Create: `src/metrics.py`
- Modify: `src/utils.py`

- [ ] **Step 1: Write failing metric tests**

```python
import pandas as pd

from src.metrics import (
    classification_metrics,
    bias_summary,
    domain_gap_summary,
    robustness_summary,
    compute_delta_rows,
    composite_ranking,
)


def test_classification_metrics_include_macro_scores():
    result = classification_metrics(["pos", "neg", "pos"], ["pos", "neg", "neg"])
    assert result["accuracy"] == 2 / 3
    assert "f1_macro" in result
    assert "per_class_f1" in result


def test_bias_domain_robustness_and_delta_metrics():
    df = pd.DataFrame(
        {
            "label": ["pos", "neg", "pos", "neg"],
            "prediction": ["pos", "neg", "neg", "neg"],
            "demographic_group": ["a", "a", "b", "b"],
            "perturbation_type": ["clean", "typo", "clean", "typo"],
        }
    )
    bias = bias_summary(df, "label", "prediction", "demographic_group")
    assert bias["max_group_f1_gap"] >= 0
    robustness = robustness_summary(df, "label", "prediction", "perturbation_type")
    assert "robustness_drop" in robustness
    domain = domain_gap_summary("run", 0.8, 0.6)
    assert domain["domain_gap_f1"] == 0.2

    baseline = [{"run_id": "B0", "eval_set": "twitter_test", "f1_macro": 0.5, "accuracy": 0.6}]
    treatment = [{"run_id": "twitter_lora_r8", "eval_set": "twitter_test", "f1_macro": 0.7, "accuracy": 0.8}]
    deltas = compute_delta_rows(baseline, treatment)
    assert deltas[0]["delta_f1_macro"] == 0.2
    assert deltas[0]["f1_retention_vs_baseline"] == 1.4


def test_composite_ranking_rewards_balanced_models():
    rows = [
        {"run_id": "a", "f1_macro": 0.9, "trainable_param_ratio": 1.0, "training_time_seconds": 10, "max_group_f1_gap": 0.4, "robustness_drop": 0.3},
        {"run_id": "b", "f1_macro": 0.85, "trainable_param_ratio": 0.1, "training_time_seconds": 2, "max_group_f1_gap": 0.1, "robustness_drop": 0.1},
    ]
    ranked = composite_ranking(rows, {"f1_macro": 0.4, "efficiency": 0.2, "robustness": 0.2, "fairness": 0.2})
    assert ranked[0]["composite_score"] >= ranked[1]["composite_score"]
```

- [ ] **Step 2: Verify tests fail**

Run: `pytest tests/test_metrics.py -q`

Expected: FAIL because `src.metrics` does not exist yet.

- [ ] **Step 3: Implement metric functions**

Implement classification, bias, domain, robustness, delta, and composite ranking logic. Use scikit-learn for classification metrics and pandas-friendly plain dictionaries for outputs.

- [ ] **Step 4: Verify tests pass**

Run: `pytest tests/test_metrics.py -q`

Expected: PASS.

## Task 3: Training, Method, Compare, And Report Scaffolding

**Files:**
- Create: `src/model_factory.py`
- Create: `src/methods/__init__.py`
- Create: `src/methods/full_ft.py`
- Create: `src/methods/lora.py`
- Create: `src/methods/adapter.py`
- Create: `src/train.py`
- Create: `src/evaluate.py`
- Create: `src/run_baseline.py`
- Create: `src/compare.py`
- Create: `src/report.py`
- Create: `src/bias_eval.py`
- Create: `src/robustness_eval.py`

- [ ] **Step 1: Write CLI smoke tests if needed**

If the first two test files are passing, add small tests only for pure functions in `compare.py` or `report.py`. Avoid model download in unit tests.

- [ ] **Step 2: Implement method scaffolding**

Implement trainable parameter policies and optional LoRA/adapter setup without forcing those dependencies to import during pure unit tests.

- [ ] **Step 3: Implement CLIs**

Each CLI accepts `--config configs/experiment.yaml`. Training CLIs run real model code when dependencies and model access are available. Compare and report CLIs work from JSON/CSV metrics already present in `outputs`.

- [ ] **Step 4: Verify imports**

Run: `python -m compileall src`

Expected: all source files compile.

## Task 4: Documentation, Scripts, And Final Verification

**Files:**
- Create: `README.md`
- Create: `requirements.txt`
- Create: `data/README.md`
- Create: `scripts/run_smoke_test.sh`
- Create: `scripts/run_baseline.sh`
- Create: `scripts/run_all_experiments.sh`
- Create: `scripts/make_report.sh`
- Create output directories with `.gitkeep` files where needed.

- [ ] **Step 1: Document setup and commands**

README must explain B0, treatment matrix, deltas, data schema, smoke test, baseline run, matrix run, compare, report, and interpretation warnings.

- [ ] **Step 2: Add scripts**

Scripts call the documented commands and set `set -euo pipefail`.

- [ ] **Step 3: Run verification**

Run:

```bash
pytest -q
python -m compileall src
```

Expected: all tests pass and all Python files compile.

- [ ] **Step 4: Commit implementation**

Run:

```bash
git add .
git commit -m "feat: scaffold bertweet baseline delta experiment"
```
