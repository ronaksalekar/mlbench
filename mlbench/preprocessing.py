"""
Data Preprocessing and Train/Test Pipeline for MLBench.
Implements strict leakage-free transformers, imputation, scaling, encoding,
and stratified or grouped dataset partitioning.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class PreprocessingConfig:
    """Configuration container for dataset preprocessing."""
    def __init__(
        self,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        num_imputation: str = "median",  # 'median', 'mean'
        cat_imputation: str = "most_frequent",
        scale_features: bool = True,
        encode_categorical: bool = True,
        dropped_cols: Optional[List[str]] = None,
        subject_id_col: Optional[str] = None,
        use_grouped_split: bool = False,
        drop_missing_row_threshold: float = 0.8,
        drop_missing_col_threshold: float = 0.6,
    ):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.num_imputation = num_imputation
        self.cat_imputation = cat_imputation
        self.scale_features = scale_features
        self.encode_categorical = encode_categorical
        self.dropped_cols = dropped_cols or []
        self.subject_id_col = subject_id_col
        self.use_grouped_split = use_grouped_split
        self.drop_missing_row_threshold = drop_missing_row_threshold
        self.drop_missing_col_threshold = drop_missing_col_threshold


class PreprocessingPipeline:
    """
    Leakage-free preprocessing pipeline.
    Splits raw data FIRST, then fits imputers, encoders, and scalers strictly on training data.
    """

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_mapping: Dict[int, str] = {}
        self.feature_names: List[str] = []
        self.transformed_feature_names: List[str] = []
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.warnings: List[str] = []

    def prepare_and_split(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series], Optional[pd.Series]]:
        """
        Step 1: Clean raw dataset, separate features and target, and split into train & test sets.
        Returns:
            X_train_raw, X_test_raw, y_train_raw, y_test_raw, groups_train, groups_test
        """
        self.warnings = []
        clean_df = df.copy()

        # Check target existence
        if self.config.target_col not in clean_df.columns:
            raise ValueError(f"Target column '{self.config.target_col}' not found in dataframe.")

        # Drop rows where target is NaN
        initial_count = len(clean_df)
        clean_df = clean_df.dropna(subset=[self.config.target_col])
        dropped_target_nans = initial_count - len(clean_df)
        if dropped_target_nans > 0:
            self.warnings.append(f"Dropped {dropped_target_nans} rows with missing target values.")

        # Drop columns with excessive missing values
        col_thresh = self.config.drop_missing_col_threshold
        excessive_missing_cols = [
            c for c in clean_df.columns
            if c != self.config.target_col and (clean_df[c].isna().mean() > col_thresh)
        ]
        if excessive_missing_cols:
            clean_df = clean_df.drop(columns=excessive_missing_cols)
            self.warnings.append(
                f"Dropped columns with >{int(col_thresh*100)}% missing values: {excessive_missing_cols}"
            )

        # Drop rows with excessive missing features
        row_missing_frac = clean_df.drop(columns=[self.config.target_col]).isna().mean(axis=1)
        row_thresh = self.config.drop_missing_row_threshold
        bad_rows = row_missing_frac > row_thresh
        if bad_rows.sum() > 0:
            clean_df = clean_df[~bad_rows].reset_index(drop=True)
            self.warnings.append(
                f"Dropped {bad_rows.sum()} rows containing >{int(row_thresh*100)}% missing feature values."
            )

        # Separate target
        y_raw = clean_df[self.config.target_col]

        # Extract subject groups if enabled
        groups_series = None
        if self.config.subject_id_col and self.config.subject_id_col in clean_df.columns:
            groups_series = clean_df[self.config.subject_id_col].astype(str)

        # Determine feature columns
        cols_to_exclude = set([self.config.target_col] + self.config.dropped_cols)
        if self.config.subject_id_col:
            cols_to_exclude.add(self.config.subject_id_col)

        feature_cols = [c for c in clean_df.columns if c not in cols_to_exclude]

        if not feature_cols:
            raise ValueError("No feature columns available after dropping target and specified columns.")

        X_raw = clean_df[feature_cols]
        self.feature_names = feature_cols

        # Identify numerical and categorical columns
        self.num_cols = list(X_raw.select_dtypes(include=[np.number]).columns)
        self.cat_cols = list(X_raw.select_dtypes(exclude=[np.number]).columns)

        # Encode target with LabelEncoder
        self.label_encoder = LabelEncoder()
        y_encoded = pd.Series(self.label_encoder.fit_transform(y_raw), index=y_raw.index)
        self.class_mapping = {int(i): str(cls_name) for i, cls_name in enumerate(self.label_encoder.classes_)}

        # Split data
        groups_train, groups_test = None, None

        if self.config.use_grouped_split and groups_series is not None:
            # Grouped split: prevents subjects from leaking across train and test sets
            n_groups = groups_series.nunique()
            if n_groups < 2:
                raise ValueError("Grouped split requires at least 2 distinct subjects/groups.")

            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            train_idx, test_idx = next(gss.split(X_raw, y_encoded, groups=groups_series))
            X_train = X_raw.iloc[train_idx].copy()
            X_test = X_raw.iloc[test_idx].copy()
            y_train = y_encoded.iloc[train_idx].copy()
            y_test = y_encoded.iloc[test_idx].copy()
            groups_train = groups_series.iloc[train_idx].copy()
            groups_test = groups_series.iloc[test_idx].copy()
            self.warnings.append(
                f"Subject-level Grouped Split applied: {groups_train.nunique()} subjects in training set, "
                f"{groups_test.nunique()} subjects in holdout test set (no subject overlap)."
            )
        else:
            # Stratified split for classification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_raw,
                    y_encoded,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y_encoded
                )
            except ValueError:
                # If a class has too few instances to stratify, fall back to random split with clear warning
                self.warnings.append(
                    "Stratified split was not possible due to extremely small class count; fell back to random split."
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X_raw,
                    y_encoded,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state
                )

        return X_train, X_test, y_train, y_test, groups_train, groups_test

    def build_transformer(self) -> ColumnTransformer:
        """
        Build scikit-learn ColumnTransformer configured with imputation,
        optional standard scaling, and one-hot encoding.
        """
        transformers = []

        if self.num_cols:
            num_steps = [
                ("imputer", SimpleImputer(strategy=self.config.num_imputation))
            ]
            if self.config.scale_features:
                num_steps.append(("scaler", StandardScaler()))
            transformers.append(("num", Pipeline(num_steps), self.num_cols))

        if self.cat_cols and self.config.encode_categorical:
            cat_steps = [
                ("imputer", SimpleImputer(strategy=self.config.cat_imputation)),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
            transformers.append(("cat", Pipeline(cat_steps), self.cat_cols))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return self.preprocessor

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fit preprocessor ONLY on X_train, then transform both X_train and X_test.
        Returns:
            X_train_proc, X_test_proc, feature_names
        """
        if self.preprocessor is None:
            self.build_transformer()

        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # Extract output feature names
        out_features = []
        for name, transformer, cols in self.preprocessor.transformers_:
            if name == "num":
                out_features.extend(cols)
            elif name == "cat":
                try:
                    onehot_step = transformer.named_steps["onehot"]
                    cat_names = onehot_step.get_feature_names_out(cols)
                    out_features.extend(list(cat_names))
                except Exception:
                    out_features.extend([f"{c}_cat" for c in cols])

        self.transformed_feature_names = out_features if out_features else [f"feature_{i}" for i in range(X_train_proc.shape[1])]
        return X_train_proc, X_test_proc, self.transformed_feature_names
