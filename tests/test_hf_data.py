from src.hf_data import normalize_hf_rows, split_limit_across_offsets


def test_normalize_hf_rows_maps_labels_and_drops_filtered_labels():
    api_rows = [
        {"row": {"text": "bad tweet", "label": 0}},
        {"row": {"text": "neutral tweet", "label": 1}},
        {"row": {"text": "good tweet", "label": 2}},
    ]
    source = {
        "text_column": "text",
        "label_column": "label",
        "label_map": {"0": "negative", "2": "positive"},
        "drop_labels": [1],
        "domain": "twitter",
    }
    frame = normalize_hf_rows(api_rows, source)
    assert frame.to_dict("records") == [
        {"text": "bad tweet", "label": "negative", "domain": "twitter"},
        {"text": "good tweet", "label": "positive", "domain": "twitter"},
    ]


def test_split_limit_across_offsets_balances_requested_rows():
    assert split_limit_across_offsets(10, [0, 100]) == [(0, 5), (100, 5)]
    assert split_limit_across_offsets(11, [0, 100]) == [(0, 6), (100, 5)]
