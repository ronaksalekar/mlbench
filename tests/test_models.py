"""
Unit tests for ModelFactory and classifier implementations.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from mlbench.models import ModelFactory


@pytest.fixture
def synthetic_data():
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.mark.parametrize("model_name", [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine (SVM)",
    "XGBoost"
])
def test_all_models_train_and_predict(synthetic_data, model_name):
    X, y = synthetic_data
    model = ModelFactory.create_model(model_name=model_name, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})


def test_feature_importance_extraction(synthetic_data):
    X, y = synthetic_data
    feat_names = [f"f_{i}" for i in range(X.shape[1])]

    rf = ModelFactory.create_model("Random Forest", random_state=42)
    rf.fit(X, y)
    importances = ModelFactory.extract_feature_importance(rf, feat_names)

    assert importances is not None
    assert len(importances) == len(feat_names)
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-2)
