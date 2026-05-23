from src.config import load_config
from src.data import build_label_mapping, ensure_smoke_data, load_text_classification_csv


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
