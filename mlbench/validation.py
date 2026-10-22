"""
Validation and data-integrity verification utilities for MLBench.
Ensures valid classification tasks, guards against target leakage, and validates splits.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class DatasetValidator:
    """Validates datasets, target specifications, and feature candidate subsets."""

    MIN_TOTAL_SAMPLES = 10
    MIN_CLASS_SAMPLES = 2
    MAX_CLASSES_FOR_CLASSIFICATION = 20

    @classmethod
    def validate_target_column(
        cls,
        df: pd.DataFrame,
        target_col: str
    ) -> Dict[str, Any]:
        """
        Validates target column for supervised classification.
        Raises ValueError with clear educational error message if invalid.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' does not exist in dataset.")

        target_series = df[target_col].dropna()

        if len(target_series) == 0:
            raise ValueError(f"Target column '{target_col}' contains only missing (NaN) values.")

        if len(target_series) < cls.MIN_TOTAL_SAMPLES:
            raise ValueError(
                f"Target column contains only {len(target_series)} non-null samples. "
                f"A minimum of {cls.MIN_TOTAL_SAMPLES} samples is required for evaluation."
            )

        unique_vals = target_series.unique()
        n_unique = len(unique_vals)

        # Check for continuous / regression target
        if pd.api.types.is_float_dtype(target_series):
            # Check if floats are actually integers (e.g. 0.0, 1.0)
            is_effectively_int = np.all(np.equal(np.mod(target_series, 1), 0))
            if not is_effectively_int and n_unique > cls.MAX_CLASSES_FOR_CLASSIFICATION:
                raise ValueError(
                    f"Selected column '{target_col}' appears to be a continuous numerical variable "
                    f"with {n_unique} unique floating-point values. MLBench currently evaluates classification models. "
                    "Please select a discrete classification target."
                )

        if n_unique < 2:
            raise ValueError(
                f"Selected target column '{target_col}' has only {n_unique} unique class ({unique_vals[0]}). "
                "Supervised classification requires at least 2 distinct classes."
            )

        if n_unique > cls.MAX_CLASSES_FOR_CLASSIFICATION:
            raise ValueError(
                f"Target column '{target_col}' has {n_unique} distinct classes, which exceeds the supported limit "
                f"({cls.MAX_CLASSES_FOR_CLASSIFICATION}) for tabular benchmark classifiers. Please choose a classification target."
            )

        # Value counts check for minimum per-class samples
        counts = target_series.value_counts()
        min_count = counts.min()
        rare_classes = counts[counts < cls.MIN_CLASS_SAMPLES].index.tolist()

        if min_count < cls.MIN_CLASS_SAMPLES:
            raise ValueError(
                f"Class(es) {rare_classes} have fewer than {cls.MIN_CLASS_SAMPLES} samples. "
                "Cannot perform stratified evaluation or train classifiers reliably."
            )

        is_binary = (n_unique == 2)
        task_type = "Binary Classification" if is_binary else "Multiclass Classification"

        # Check class balance
        majority_ratio = counts.max() / len(target_series)
        is_imbalanced = majority_ratio >= 0.75

        warnings = []
        if is_imbalanced:
            warnings.append(
                f"Dataset shows class imbalance: majority class accounts for {round(majority_ratio*100, 1)}% "
                "of observations. In imbalanced datasets, Accuracy can be misleading; focus on Precision, Recall, F1, and Specificity."
            )

        return {
            "is_valid": True,
            "task_type": task_type,
            "is_binary": is_binary,
            "num_classes": n_unique,
            "classes": [str(c) for c in counts.index],
            "class_counts": counts.to_dict(),
            "min_class_samples": int(min_count),
            "is_imbalanced": is_imbalanced,
            "warnings": warnings,
        }

    @classmethod
    def validate_features(
        cls,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        dropped_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validates feature columns ensuring target is excluded, no empty feature set,
        and flags potential leakage from identifiers or direct target proxies.
        """
        dropped_cols = dropped_cols or []

        # Target MUST NOT be in features
        safe_features = [c for c in feature_cols if c != target_col and c not in dropped_cols]

        if not safe_features:
            raise ValueError(
                "No predictive features remain after excluding the target and dropped columns. "
                "Please select at least one input feature."
            )

        warnings = []
        potential_leakage = []

        # Detect identifiers or high cardinality string columns
        for col in safe_features:
            s = df[col]
            n_unique = s.nunique(dropna=True)
            col_l = str(col).lower()

            # Check if column is constant
            if n_unique <= 1:
                warnings.append(f"Feature '{col}' has only {n_unique} unique value (constant). It will provide no predictive value.")

            # Check for perfect unique identifiers
            if n_unique == len(df) and len(df) > 20 and not pd.api.types.is_numeric_dtype(s):
                potential_leakage.append(col)
                warnings.append(
                    f"Feature '{col}' has 100% unique values and appears to be an identifier. "
                    "Including it may cause overfitting and data leakage. Consider dropping it."
                )
            elif any(id_word in col_l for id_word in ["patient_id", "recording_id", "subject_id", "patientid"]):
                warnings.append(
                    f"Feature '{col}' looks like a subject/patient identifier. "
                    "Consider excluding it from features and using it for subject-level grouped validation instead."
                )

        return {
            "valid_features": safe_features,
            "num_features": len(safe_features),
            "warnings": warnings,
            "potential_leakage": potential_leakage
        }

    @classmethod
    def validate_cv_folds(
        cls,
        class_counts: Dict[Any, int],
        n_splits: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates that each class has at least n_splits samples for StratifiedKFold.
        """
        min_class_count = min(class_counts.values()) if class_counts else 0
        if n_splits > min_class_count:
            return (
                False,
                f"Requested {n_splits}-fold cross validation, but the smallest class has only "
                f"{min_class_count} sample(s). Number of folds cannot exceed the minimum class count ({min_class_count})."
            )
        return True, None
