"""
Unit tests for ReportGenerator and export serialization.
"""

import pytest
import json
from mlbench.reporting import ReportGenerator


@pytest.fixture
def dummy_experiment_results():
    return {
        "Random Forest": {
            "hyperparameters": {"n_estimators": 50},
            "training_time_sec": 0.123,
            "cv_mean": 0.88,
            "cv_std": 0.02,
            "metrics": {
                "accuracy": 0.90,
                "precision": 0.89,
                "recall": 0.91,
                "f1_score": 0.90,
                "specificity": 0.89,
                "roc_auc": 0.95,
                "confusion_matrix": [[9, 1], [1, 9]]
            },
            "feature_importance": {"feat_A": 0.6, "feat_B": 0.4}
        },
        "Logistic Regression": {
            "hyperparameters": {"C": 1.0},
            "training_time_sec": 0.045,
            "cv_mean": 0.82,
            "cv_std": 0.04,
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.84,
                "recall": 0.86,
                "f1_score": 0.85,
                "specificity": 0.84,
                "roc_auc": 0.90,
                "confusion_matrix": [[8, 2], [1, 9]]
            },
            "feature_importance": {"feat_A": 0.5, "feat_B": 0.5}
        }
    }


def test_leaderboard_generation(dummy_experiment_results):
    df = ReportGenerator.generate_leaderboard_df(dummy_experiment_results, ranking_metric="F1-Score")
    assert len(df) == 2
    assert "Rank" in df.columns
    # Random Forest has F1 0.90 vs 0.85, so it must be rank 1
    assert df.iloc[0]["Model"] == "Random Forest"
    assert df.iloc[0]["Rank"] == 1


def test_json_summary(dummy_experiment_results):
    exp_data = {
        "dataset_info": {"dataset_name": "TestDS", "rows": 100},
        "target_col": "status",
        "ranking_metric": "F1-Score",
        "results": dummy_experiment_results
    }
    json_str = ReportGenerator.generate_json_summary(exp_data)
    parsed = json.loads(json_str)
    assert parsed["platform"] == "MLBench - Comparative ML Model Evaluation Platform"
    assert len(parsed["models_summary"]) == 2


def test_html_report(dummy_experiment_results):
    exp_data = {
        "dataset_info": {"dataset_name": "Parkinsons", "rows": 120, "features_count": 22},
        "target_col": "status",
        "ranking_metric": "F1-Score",
        "results": dummy_experiment_results
    }
    html = ReportGenerator.generate_html_report(exp_data)
    assert "<!DOCTYPE html>" in html
    assert "Random Forest" in html
    assert "Parkinsons" in html
    assert "Educational & Research Disclaimer" in html
