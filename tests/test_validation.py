"""
Unit tests for DatasetValidator.
"""

import pytest
import pandas as pd
import numpy as np
from mlbench.validation import DatasetValidator


def test_validate_target_valid_binary():
    df = pd.DataFrame({
        "feat": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    res = DatasetValidator.validate_target_column(df, "target")
    assert res["is_valid"] is True
    assert res["is_binary"] is True
    assert res["num_classes"] == 2


def test_validate_target_continuous_rejected():
    df = pd.DataFrame({
        "feat": range(30),
        "target": [x * 1.345 for x in range(30)]
    })
    with pytest.raises(ValueError, match="continuous numerical variable"):
        DatasetValidator.validate_target_column(df, "target")


def test_validate_target_single_class_rejected():
    df = pd.DataFrame({
        "feat": range(15),
        "target": [1] * 15
    })
    with pytest.raises(ValueError, match="requires at least 2 distinct classes"):
        DatasetValidator.validate_target_column(df, "target")


def test_validate_features_prevents_leakage():
    df = pd.DataFrame({
        "id": range(15),
        "feat1": range(15),
        "target": [0, 1] * 7 + [0]
    })
    # Target column must be excluded even if passed in features list
    res = DatasetValidator.validate_features(
        df,
        target_col="target",
        feature_cols=["id", "feat1", "target"],
        dropped_cols=["id"]
    )
    assert "target" not in res["valid_features"]
    assert "id" not in res["valid_features"]
    assert res["valid_features"] == ["feat1"]


def test_validate_cv_folds():
    class_counts = {"class_0": 10, "class_1": 2}
    is_valid, msg = DatasetValidator.validate_cv_folds(class_counts, n_splits=5)
    assert is_valid is False
    assert "cannot exceed" in msg

    is_valid_ok, _ = DatasetValidator.validate_cv_folds({"class_0": 20, "class_1": 15}, n_splits=5)
    assert is_valid_ok is True
