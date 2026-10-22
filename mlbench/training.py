"""
Training pipeline and experiment execution runner for MLBench.
Coordinates model training, holdout evaluation, stratified/grouped cross-validation,
and execution timing.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.inspection import permutation_importance

from mlbench.models import ModelFactory
from mlbench.evaluation import Evaluator


class TrainingRunner:
    """Orchestrates model fitting, cross-validation, and metrics collection."""

    @staticmethod
    def run_experiment(
        selected_models: List[str],
        hyperparams: Dict[str, Dict[str, Any]],
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        class_mapping: Dict[int, str],
        enable_cv: bool = True,
        cv_folds: int = 5,
        cv_metric: str = "f1",
        groups_train: Optional[np.ndarray] = None,
        random_state: int = 42,
        progress_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute training and evaluation across all selected models.
        Returns a dictionary containing completed model results, leaderboard records,
        and any failed model error logs.
        """
        results = {}
        errors = {}
        total_models = len(selected_models)

        # Prepare CV scoring metric string
        # scikit-learn scoring names: 'accuracy', 'f1_weighted' or 'f1', 'precision_weighted', 'recall_weighted', 'roc_auc'
        is_binary = (len(np.unique(y_train)) == 2)
        if cv_metric == "f1":
            scoring_name = "f1" if is_binary else "f1_weighted"
        elif cv_metric == "precision":
            scoring_name = "precision" if is_binary else "precision_weighted"
        elif cv_metric == "recall":
            scoring_name = "recall" if is_binary else "recall_weighted"
        elif cv_metric == "roc_auc":
            scoring_name = "roc_auc" if is_binary else "roc_auc_ovr_weighted"
        else:
            scoring_name = "accuracy"

        # Setup cross-validation splitter
        cv_splitter = None
        if enable_cv:
            if groups_train is not None and len(groups_train) == len(y_train):
                # Check that number of groups >= cv_folds
                unique_groups = len(np.unique(groups_train))
                if unique_groups >= cv_folds:
                    cv_splitter = GroupKFold(n_splits=cv_folds)
                else:
                    cv_splitter = StratifiedKFold(n_splits=min(cv_folds, 5), shuffle=True, random_state=random_state)
            else:
                cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        for idx, model_name in enumerate(selected_models):
            if progress_callback:
                progress_callback(idx / total_models, f"Training {model_name}...")

            model_params = hyperparams.get(model_name, {})

            try:
                # 1. Instantiate model
                model = ModelFactory.create_model(
                    model_name=model_name,
                    params=model_params,
                    random_state=random_state
                )

                # 2. Cross-validation on training data ONLY
                cv_mean, cv_std = None, None
                cv_scores = []
                if enable_cv and cv_splitter is not None:
                    try:
                        if isinstance(cv_splitter, GroupKFold) and groups_train is not None:
                            scores = cross_val_score(
                                model, X_train, y_train,
                                groups=groups_train,
                                cv=cv_splitter,
                                scoring=scoring_name,
                                n_jobs=-1
                            )
                        else:
                            scores = cross_val_score(
                                model, X_train, y_train,
                                cv=cv_splitter,
                                scoring=scoring_name,
                                n_jobs=-1
                            )
                        cv_scores = scores.tolist()
                        cv_mean = float(np.mean(scores))
                        cv_std = float(np.std(scores))
                    except Exception as cv_e:
                        # Fallback if scoring metric or fold failed
                        cv_mean, cv_std = None, None
                        cv_scores = []

                # 3. Fit model on full training set and measure time
                start_time = time.perf_counter()
                model.fit(X_train, y_train)
                train_time = time.perf_counter() - start_time

                # 4. Predict on holdout test set
                y_pred = model.predict(X_test)

                # 5. Predict probabilities if available
                y_prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_prob = model.predict_proba(X_test)
                    except Exception:
                        y_prob = None
                elif hasattr(model, "decision_function"):
                    try:
                        df_scores = model.decision_function(X_test)
                        if df_scores.ndim == 1:
                            # Convert 1D decision scores to pseudo-probabilities via sigmoid
                            probs = 1 / (1 + np.exp(-df_scores))
                            y_prob = np.column_stack([1 - probs, probs])
                    except Exception:
                        y_prob = None

                # 6. Evaluate metrics
                metrics = Evaluator.calculate_metrics(
                    y_true=y_test,
                    y_pred=y_pred,
                    y_prob=y_prob,
                    classes=[class_mapping[i] for i in sorted(class_mapping.keys())],
                    pos_label=1
                )

                # 7. Extract feature importance
                feat_importance = ModelFactory.extract_feature_importance(model, feature_names)

                # Store model result
                results[model_name] = {
                    "model_name": model_name,
                    "model_obj": model,
                    "hyperparameters": model_params,
                    "training_time_sec": round(train_time, 4),
                    "metrics": metrics,
                    "cv_mean": round(cv_mean, 4) if cv_mean is not None else None,
                    "cv_std": round(cv_std, 4) if cv_std is not None else None,
                    "cv_scores": cv_scores,
                    "cv_metric_name": cv_metric,
                    "feature_importance": feat_importance,
                    "y_pred": y_pred.tolist(),
                    "y_prob": y_prob.tolist() if y_prob is not None else None,
                }

            except Exception as e:
                errors[model_name] = str(e)

        if progress_callback:
            progress_callback(1.0, "Experiment completed!")

        return {
            "results": results,
            "errors": errors,
            "total_selected": total_models,
            "successful_count": len(results),
            "failed_count": len(errors),
            "class_mapping": class_mapping
        }

    @staticmethod
    def compute_permutation_importance(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 5,
        random_state: int = 42
    ) -> Dict[str, float]:
        """Model-agnostic permutation importance on holdout test set."""
        try:
            r = permutation_importance(
                model, X_test, y_test,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=-1
            )
            pi_dict = {
                name: float(mean_imp)
                for name, mean_imp in zip(feature_names, r.importances_mean)
            }
            return dict(sorted(pi_dict.items(), key=lambda item: item[1], reverse=True))
        except Exception:
            return {}
