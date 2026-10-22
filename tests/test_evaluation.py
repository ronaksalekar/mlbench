"""
Unit tests for Evaluator.
Verifies mathematical correctness of classification metrics,
confusion matrix unpacking, and ROC-AUC computation.
"""

import pytest
import numpy as np
from mlbench.evaluation import Evaluator


def test_metric_mathematical_definitions():
    # Known confusion matrix setup:
    # y_true: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] (4 positive, 6 negative)
    # y_pred: [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    # TP = 3, FN = 1
    # FP = 1, TN = 5
    # Total = 10
    # Accuracy = (3 + 5) / 10 = 0.8
    # Precision = 3 / (3 + 1) = 0.75
    # Recall = 3 / (3 + 1) = 0.75
    # F1 = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
    # Specificity = 5 / (5 + 1) = 5/6 = 0.8333
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4],
                       [0.4, 0.6], [0.8, 0.2], [0.9, 0.1], [0.7, 0.3], [0.85, 0.15], [0.95, 0.05]])

    metrics = Evaluator.calculate_metrics(y_true, y_pred, y_prob=y_prob, pos_label=1)

    assert metrics["accuracy"] == pytest.approx(0.8, abs=1e-3)
    assert metrics["precision"] == pytest.approx(0.75, abs=1e-3)
    assert metrics["recall"] == pytest.approx(0.75, abs=1e-3)
    assert metrics["f1_score"] == pytest.approx(0.75, abs=1e-3)
    assert metrics["specificity"] == pytest.approx(5/6, abs=1e-3)

    counts = metrics["binary_counts"]
    assert counts["tp"] == 3
    assert counts["tn"] == 5
    assert counts["fp"] == 1
    assert counts["fn"] == 1

    assert metrics["roc_auc"] is not None
    assert metrics["roc_auc"] > 0.5


def test_zero_division_safety():
    # Model predicts all 0s -> TP = 0, FP = 0 -> Precision denominator is 0
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    metrics = Evaluator.calculate_metrics(y_true, y_pred, y_prob=None, pos_label=1)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["accuracy"] == 0.5


def test_multiclass_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])

    metrics = Evaluator.calculate_metrics(y_true, y_pred, y_prob=None)
    assert metrics["is_binary"] is False
    assert metrics["average_method"] == "weighted"
    assert 0.0 <= metrics["accuracy"] <= 1.0
