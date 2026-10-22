"""
Model factory and wrappers for MLBench classification algorithms.
Implements Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Check XGBoost availability gracefully
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    XGBOOST_ERROR = None
except ImportError as e:
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(e)


class ModelFactory:
    """Instantiates configured scikit-learn and XGBoost classifiers."""

    AVAILABLE_MODELS = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine (SVM)",
        "XGBoost"
    ]

    @classmethod
    def is_xgboost_available(cls) -> bool:
        return XGBOOST_AVAILABLE

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Dict[str, Any]]:
        """Default hyperparameters for each algorithm."""
        return {
            "Logistic Regression": {
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42
            },
            "Decision Tree": {
                "max_depth": 6,
                "min_samples_leaf": 2,
                "criterion": "gini",
                "random_state": 42
            },
            "Random Forest": {
                "n_estimators": 100,
                "max_depth": 8,
                "min_samples_leaf": 2,
                "random_state": 42
            },
            "Support Vector Machine (SVM)": {
                "C": 1.0,
                "kernel": "rbf",
                "probability": True,
                "random_state": 42
            },
            "XGBoost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 4,
                "subsample": 0.8,
                "random_state": 42
            }
        }

    @classmethod
    def create_model(
        cls,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ) -> Any:
        """
        Factory method to instantiate a classifier with given hyperparameters.
        """
        defaults = cls.get_default_hyperparameters().get(model_name, {})
        cfg = {**defaults, **(params or {})}
        cfg["random_state"] = random_state

        if model_name == "Logistic Regression":
            return LogisticRegression(
                C=float(cfg.get("C", 1.0)),
                max_iter=int(cfg.get("max_iter", 1000)),
                solver=str(cfg.get("solver", "lbfgs")),
                random_state=random_state
            )

        elif model_name == "Decision Tree":
            max_depth = cfg.get("max_depth", 6)
            return DecisionTreeClassifier(
                max_depth=int(max_depth) if max_depth is not None else None,
                min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
                criterion=str(cfg.get("criterion", "gini")),
                random_state=random_state
            )

        elif model_name == "Random Forest":
            max_depth = cfg.get("max_depth", 8)
            return RandomForestClassifier(
                n_estimators=int(cfg.get("n_estimators", 100)),
                max_depth=int(max_depth) if max_depth is not None else None,
                min_samples_leaf=int(cfg.get("min_samples_leaf", 2)),
                random_state=random_state,
                n_jobs=-1
            )

        elif model_name == "Support Vector Machine (SVM)":
            return SVC(
                C=float(cfg.get("C", 1.0)),
                kernel=str(cfg.get("kernel", "rbf")),
                probability=bool(cfg.get("probability", True)),
                random_state=random_state
            )

        elif model_name == "XGBoost":
            if not XGBOOST_AVAILABLE:
                raise ImportError(
                    f"XGBoost is not installed or available: {XGBOOST_ERROR}. "
                    "Install xgboost via `pip install xgboost` or uncheck XGBoost from model selection."
                )
            max_depth = cfg.get("max_depth", 4)
            return XGBClassifier(
                n_estimators=int(cfg.get("n_estimators", 100)),
                learning_rate=float(cfg.get("learning_rate", 0.1)),
                max_depth=int(max_depth) if max_depth is not None else 4,
                subsample=float(cfg.get("subsample", 0.8)),
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1
            )

        else:
            raise ValueError(f"Unknown model name '{model_name}'. Supported models: {cls.AVAILABLE_MODELS}")

    @staticmethod
    def extract_feature_importance(
        model: Any,
        feature_names: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Extract feature importances or linear coefficients if available.
        Returns dict {feature_name: importance_value} sorted descending.
        """
        # Tree-based models (Random Forest, Decision Tree, XGBoost)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            if len(importances) == len(feature_names):
                fi = {name: float(imp) for name, imp in zip(feature_names, importances)}
                return dict(sorted(fi.items(), key=lambda item: item[1], reverse=True))

        # Linear models (Logistic Regression, Linear SVM)
        if hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef_mag = np.mean(np.abs(coef), axis=0)
            else:
                coef_mag = np.abs(coef)
            if len(coef_mag) == len(feature_names):
                fi = {name: float(val) for name, val in zip(feature_names, coef_mag)}
                return dict(sorted(fi.items(), key=lambda item: item[1], reverse=True))

        return None
