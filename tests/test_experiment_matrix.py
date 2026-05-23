from src.run_matrix import build_experiment_matrix


def test_experiment_matrix_has_expected_treatments():
    config = {
        "experiment_matrix": {
            "domains": ["twitter", "general"],
            "methods": ["full_ft", "lora", "adapter"],
        },
        "methods": {
            "lora": {"ranks": [8, 16, 32]},
            "adapter": {"bottleneck_size": 64},
        },
    }
    runs = build_experiment_matrix(config)
    assert len(runs) == 10
    assert {"domain": "twitter", "method": "full_ft", "run_id": "twitter_full_ft"} in runs
    assert {
        "domain": "general",
        "method": "lora",
        "lora_rank": 32,
        "run_id": "general_lora_r32",
    } in runs
    assert {
        "domain": "twitter",
        "method": "adapter",
        "adapter_bottleneck_size": 64,
        "run_id": "twitter_adapter_b64",
    } in runs
