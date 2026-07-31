from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _unavailable(reason: str) -> dict[str, Any]:
    return {"value": None, "available": False, "reason": reason}


def _value(value: float | int) -> dict[str, Any]:
    return {"value": value, "available": True, "reason": None}


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "completed"]
    scores = np.asarray([float(row["raw_image_score"]) for row in successful])
    labels = [row["label"] for row in successful]
    label_counts = Counter(labels)
    result: dict[str, Any] = {
        "sample_count": len(rows),
        "successful_count": len(successful),
        "error_count": len(rows) - len(successful),
        "quality_failure_count": sum(bool(row.get("quality_flags")) for row in rows),
        "label_counts": dict(label_counts),
    }
    if scores.size:
        result["raw_score"] = {
            "min": float(scores.min()),
            "median": float(np.median(scores)),
            "p90": float(np.quantile(scores, 0.90)),
            "p95": float(np.quantile(scores, 0.95)),
            "p99": float(np.quantile(scores, 0.99)),
            "max": float(scores.max()),
        }
        for key in ("preprocessing_ms", "inference_ms", "total_ms"):
            values = np.asarray([float(row[key]) for row in successful])
            result[key] = {
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
    else:
        result["raw_score"] = None

    predictions = [row.get("prediction") for row in successful]
    calibrated = successful and all(value in ("normal", "fault") for value in predictions)
    if calibrated:
        normal_rows = [row for row in successful if row["label"] == "normal"]
        if normal_rows:
            false_positives = sum(row["prediction"] == "fault" for row in normal_rows)
            result["false_positives"] = _value(false_positives)
            result["false_positive_rate"] = _value(false_positives / len(normal_rows))
        else:
            reason = "Requires at least one reviewed normal sample"
            result["false_positives"] = _unavailable(reason)
            result["false_positive_rate"] = _unavailable(reason)
    else:
        reason = "Model has no declared image-score threshold"
        result["false_positives"] = _unavailable(reason)
        result["false_positive_rate"] = _unavailable(reason)

    has_both_classes = set(labels) >= {"normal", "fault"}
    defect_reason = "Requires both reviewed normal and fault samples"
    for metric in ("auroc", "f1", "precision", "recall"):
        result[metric] = _unavailable(defect_reason)
    if has_both_classes and scores.size:
        normal = scores[np.asarray(labels) == "normal"]
        fault = scores[np.asarray(labels) == "fault"]
        wins = sum(float(fault_score > normal_score) + 0.5 * float(fault_score == normal_score)
                   for fault_score in fault for normal_score in normal)
        result["auroc"] = _value(wins / (len(fault) * len(normal)))
        if calibrated:
            y_true = np.asarray([label == "fault" for label in labels])
            y_pred = np.asarray([value == "fault" for value in predictions])
            tp = int(np.sum(y_true & y_pred)); fp = int(np.sum(~y_true & y_pred))
            fn = int(np.sum(y_true & ~y_pred))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            result["precision"] = _value(precision)
            result["recall"] = _value(recall)
            result["f1"] = _value(f1)
    return result
