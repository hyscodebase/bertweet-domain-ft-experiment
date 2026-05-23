import pandas as pd

from src.metrics import (
    bias_summary,
    classification_metrics,
    composite_ranking,
    compute_delta_rows,
    domain_gap_summary,
    robustness_summary,
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
        {
            "run_id": "a",
            "f1_macro": 0.9,
            "trainable_param_ratio": 1.0,
            "training_time_seconds": 10,
            "max_group_f1_gap": 0.4,
            "robustness_drop": 0.3,
        },
        {
            "run_id": "b",
            "f1_macro": 0.85,
            "trainable_param_ratio": 0.1,
            "training_time_seconds": 2,
            "max_group_f1_gap": 0.1,
            "robustness_drop": 0.1,
        },
    ]
    ranked = composite_ranking(
        rows,
        {"f1_macro": 0.4, "efficiency": 0.2, "robustness": 0.2, "fairness": 0.2},
    )
    assert ranked[0]["composite_score"] >= ranked[1]["composite_score"]
