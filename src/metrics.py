from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _round(value: Any, digits: int = 10) -> Any:
    if value is None:
        return None
    return round(float(value), digits)


def classification_metrics(y_true: list[Any], y_pred: list[Any]) -> dict[str, Any]:
    labels = sorted({str(value) for value in y_true} | {str(value) for value in y_pred})
    true = [str(value) for value in y_true]
    pred = [str(value) for value in y_pred]
    precision, recall, f1, _support = precision_recall_fscore_support(
        true,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    _precision_weighted, _recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision_macro": _round(precision_macro),
        "recall_macro": _round(recall_macro),
        "f1_macro": _round(f1_macro),
        "f1_weighted": _round(f1_weighted),
        "per_class_f1": {label: _round(score) for label, score in zip(labels, f1)},
        "per_class_precision": {label: _round(score) for label, score in zip(labels, precision)},
        "per_class_recall": {label: _round(score) for label, score in zip(labels, recall)},
    }


def _binary_rates(frame: pd.DataFrame, label_column: str, prediction_column: str) -> dict[str, float]:
    labels = sorted({str(value) for value in frame[label_column]} | {str(value) for value in frame[prediction_column]})
    if len(labels) != 2:
        return {}
    negative, positive = labels
    true = frame[label_column].astype(str)
    pred = frame[prediction_column].astype(str)
    false_positive = ((true == negative) & (pred == positive)).sum()
    true_negative = ((true == negative) & (pred == negative)).sum()
    false_negative = ((true == positive) & (pred == negative)).sum()
    true_positive = ((true == positive) & (pred == positive)).sum()
    fpr_denominator = false_positive + true_negative
    fnr_denominator = false_negative + true_positive
    return {
        "false_positive_rate": _round(false_positive / fpr_denominator) if fpr_denominator else 0.0,
        "false_negative_rate": _round(false_negative / fnr_denominator) if fnr_denominator else 0.0,
    }


def bias_summary(
    frame: pd.DataFrame,
    label_column: str,
    prediction_column: str,
    group_column: str,
) -> dict[str, Any]:
    if group_column not in frame.columns:
        return {"group_f1_macro": {}, "group_accuracy": {}, "max_group_f1_gap": None, "max_group_accuracy_gap": None}

    group_f1: dict[str, float] = {}
    group_accuracy: dict[str, float] = {}
    fpr_by_group: dict[str, float] = {}
    fnr_by_group: dict[str, float] = {}

    for group, group_frame in frame.groupby(group_column):
        group_key = str(group)
        metrics = classification_metrics(group_frame[label_column].tolist(), group_frame[prediction_column].tolist())
        group_f1[group_key] = metrics["f1_macro"]
        group_accuracy[group_key] = metrics["accuracy"]
        rates = _binary_rates(group_frame, label_column, prediction_column)
        if rates:
            fpr_by_group[group_key] = rates["false_positive_rate"]
            fnr_by_group[group_key] = rates["false_negative_rate"]

    max_group_f1_gap = _gap(group_f1.values())
    max_group_accuracy_gap = _gap(group_accuracy.values())
    worst_group = min(group_f1, key=group_f1.get) if group_f1 else None
    best_group = max(group_f1, key=group_f1.get) if group_f1 else None

    return {
        "group_f1_macro": group_f1,
        "group_accuracy": group_accuracy,
        "max_group_f1_gap": max_group_f1_gap,
        "max_group_accuracy_gap": max_group_accuracy_gap,
        "worst_group": worst_group,
        "best_group": best_group,
        "false_positive_rate_by_group": fpr_by_group,
        "false_negative_rate_by_group": fnr_by_group,
        "fpr_gap": _gap(fpr_by_group.values()) if fpr_by_group else None,
        "fnr_gap": _gap(fnr_by_group.values()) if fnr_by_group else None,
    }


def _gap(values: Any) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return _round(max(numeric) - min(numeric))


def domain_gap_summary(run_id: str, twitter_f1_macro: float, general_f1_macro: float) -> dict[str, Any]:
    gap = _round(twitter_f1_macro - general_f1_macro)
    return {
        "run_id": run_id,
        "twitter_test_f1": _round(twitter_f1_macro),
        "general_test_f1": _round(general_f1_macro),
        "domain_gap_f1": gap,
        "absolute_domain_gap_f1": _round(abs(gap)),
    }


def robustness_summary(
    frame: pd.DataFrame,
    label_column: str,
    prediction_column: str,
    perturbation_column: str,
) -> dict[str, Any]:
    if perturbation_column not in frame.columns:
        return {"f1_macro_by_perturbation_type": {}, "robustness_drop": None}

    by_type: dict[str, float] = {}
    for perturbation, group_frame in frame.groupby(perturbation_column):
        metrics = classification_metrics(group_frame[label_column].tolist(), group_frame[prediction_column].tolist())
        by_type[str(perturbation)] = metrics["f1_macro"]

    clean_f1 = by_type.get("clean")
    perturbed = frame[frame[perturbation_column].astype(str) != "clean"]
    if clean_f1 is None or perturbed.empty:
        drop = None
        perturbed_f1 = None
    else:
        perturbed_f1 = classification_metrics(perturbed[label_column].tolist(), perturbed[prediction_column].tolist())[
            "f1_macro"
        ]
        drop = _round(clean_f1 - perturbed_f1)

    return {
        "f1_macro_by_perturbation_type": by_type,
        "clean_eval_f1_macro": clean_f1,
        "perturbed_eval_f1_macro": perturbed_f1,
        "robustness_drop": drop,
    }


def compute_delta_rows(
    baseline_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_eval = {row.get("eval_set"): row for row in baseline_rows}
    deltas: list[dict[str, Any]] = []
    for row in treatment_rows:
        eval_set = row.get("eval_set")
        baseline = baseline_by_eval.get(eval_set)
        if not baseline:
            continue
        merged = dict(row)
        merged["baseline_run_id"] = baseline.get("run_id", "B0")
        merged["baseline_f1_macro"] = baseline.get("f1_macro")
        merged["baseline_accuracy"] = baseline.get("accuracy")
        merged["delta_f1_macro"] = _safe_delta(row.get("f1_macro"), baseline.get("f1_macro"))
        merged["delta_accuracy"] = _safe_delta(row.get("accuracy"), baseline.get("accuracy"))
        merged["delta_bias_gap"] = _safe_delta(row.get("max_group_f1_gap"), baseline.get("max_group_f1_gap"))
        merged["delta_domain_gap"] = _safe_delta(row.get("absolute_domain_gap_f1"), baseline.get("absolute_domain_gap_f1"))
        merged["delta_robustness_drop"] = _safe_delta(row.get("robustness_drop"), baseline.get("robustness_drop"))
        denominator = baseline.get("f1_macro")
        merged["f1_retention_vs_baseline"] = (
            _round(float(row["f1_macro"]) / float(denominator))
            if denominator not in (None, 0) and row.get("f1_macro") is not None
            else None
        )
        deltas.append(merged)
    return deltas


def _safe_delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return _round(float(value) - float(baseline))


def composite_ranking(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    if not rows:
        return []
    f1_scores = _normalize([row.get("f1_macro") for row in rows], higher_is_better=True)
    param_scores = _normalize([row.get("trainable_param_ratio") for row in rows], higher_is_better=False)
    time_scores = _normalize([row.get("training_time_seconds") for row in rows], higher_is_better=False)
    fairness_scores = _normalize([row.get("max_group_f1_gap") for row in rows], higher_is_better=False)
    robustness_scores = _normalize([row.get("robustness_drop") for row in rows], higher_is_better=False)

    ranked: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        efficiency = (param_scores[index] + time_scores[index]) / 2
        score = (
            weights.get("f1_macro", 0.4) * f1_scores[index]
            + weights.get("efficiency", 0.2) * efficiency
            + weights.get("robustness", 0.2) * robustness_scores[index]
            + weights.get("fairness", 0.2) * fairness_scores[index]
        )
        enriched = dict(row)
        enriched["normalized_f1_macro"] = _round(f1_scores[index])
        enriched["normalized_efficiency"] = _round(efficiency)
        enriched["normalized_robustness"] = _round(robustness_scores[index])
        enriched["normalized_fairness"] = _round(fairness_scores[index])
        enriched["composite_score"] = _round(score)
        ranked.append(enriched)
    return sorted(ranked, key=lambda item: item["composite_score"], reverse=True)


def _normalize(values: list[Any], higher_is_better: bool) -> list[float]:
    numeric = [None if value is None else float(value) for value in values]
    present = [value for value in numeric if value is not None]
    if not present:
        return [0.0 for _ in values]
    minimum = min(present)
    maximum = max(present)
    if maximum == minimum:
        return [1.0 if value is not None else 0.0 for value in numeric]
    scores: list[float] = []
    for value in numeric:
        if value is None:
            scores.append(0.0)
            continue
        scaled = (value - minimum) / (maximum - minimum)
        scores.append(scaled if higher_is_better else 1.0 - scaled)
    return scores
