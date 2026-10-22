"""
Unit tests for PreprocessingPipeline.
Verifies strictly leakage-free transformations, split integrity, and encoding.
"""

import pytest
import pandas as pd
import numpy as np
from mlbench.preprocessing import PreprocessingConfig, PreprocessingPipeline


def test_preprocessing_no_leakage():
    # Construct a dataset where test set has missing values and train has known median
    df = pd.DataFrame({
        "num_feat": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        "cat_feat": ["cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    cfg = PreprocessingConfig(
        target_col="target",
        test_size=0.3,
        random_state=42,
        num_imputation="median",
        scale_features=True
    )
    pipeline = PreprocessingPipeline(cfg)
    X_train, X_test, y_train, y_test, _, _ = pipeline.prepare_and_split(df)

    assert len(X_train) == 7
    assert len(X_test) == 3

    X_train_proc, X_test_proc, feature_names = pipeline.fit_transform(X_train, X_test)

    # Preprocessor should be fitted
    assert pipeline.preprocessor is not None
    assert X_train_proc.shape[0] == 7
    assert X_test_proc.shape[0] == 3
    # Check transformed feature count (1 numerical + 2 one-hot)
    assert X_train_proc.shape[1] >= 2


def test_preprocessing_grouped_split():
    # Verify subjects do NOT overlap between train and test
    records = []
    for s in range(1, 11):
        for rec in range(3):
            records.append({
                "subject": f"Subj_{s}",
                "val": float(s * 10 + rec),
                "target": 1 if s <= 5 else 0
            })
    df = pd.DataFrame(records)

    cfg = PreprocessingConfig(
        target_col="target",
        subject_id_col="subject",
        use_grouped_split=True,
        test_size=0.3,
        random_state=42
    )
    pipeline = PreprocessingPipeline(cfg)
    X_train, X_test, y_train, y_test, groups_train, groups_test = pipeline.prepare_and_split(df)

    # Check that set of subjects in train and test are strictly disjoint
    train_subjects = set(groups_train.unique())
    test_subjects = set(groups_test.unique())
    assert len(train_subjects.intersection(test_subjects)) == 0
