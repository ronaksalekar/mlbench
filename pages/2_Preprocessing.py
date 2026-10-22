"""
MLBench — Page 2: Data Preprocessing
Data imputation, encoding, scaling, leakage prevention, and train/test partition controls.
"""

import streamlit as st
import pandas as pd
import numpy as np
from mlbench.state import init_session_state, get_css
from mlbench.validation import DatasetValidator
from mlbench.preprocessing import PreprocessingConfig, PreprocessingPipeline
from mlbench.data_loader import DataLoader

st.set_page_config(page_title="Preprocessing — MLBench", page_icon="⚙️", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)
init_session_state()

st.title("⚙️ Data Preprocessing & Leakage Protection")
st.caption("Configure strict train/test isolation, imputation, scaling, categorical encoding, and subject grouping.")

st.markdown("---")

if st.session_state.df_raw is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset in 'Dataset Explorer' first.")
    st.stop()

df = st.session_state.df_raw
target_col = st.session_state.target_col

if not target_col or target_col not in df.columns:
    st.error("Please select a valid target column in 'Dataset Explorer' first.")
    st.stop()

# Validate target column using DatasetValidator
try:
    val_res = DatasetValidator.validate_target_column(df, target_col)
    st.success(f"Target Confirmed: **{target_col}** ({val_res['task_type']}, {val_res['num_classes']} classes)")
    if val_res["warnings"]:
        for w in val_res["warnings"]:
            st.warning(w)
except ValueError as e:
    st.error(f"Target Validation Error: {e}")
    st.stop()

st.markdown("### 1. Feature Selection & Identifier Exclusion")
all_cols = list(df.columns)
potential_features = [c for c in all_cols if c != target_col]

# Detect potential identifier columns (e.g. name, id, patient)
detected_id_cols = [
    c for c in potential_features
    if any(k in c.lower() for k in ["id", "name", "patient", "subject", "sample"]) or df[c].nunique() == len(df)
]

col_f1, col_f2 = st.columns(2)

with col_f1:
    dropped_cols = st.multiselect(
        "Columns to Exclude (Identifiers, Leaky Features, Keys):",
        options=potential_features,
        default=[c for c in detected_id_cols if c in potential_features],
        help="Identifiers or subject keys should NOT be fed as input features to classifiers."
    )
    st.session_state.dropped_cols = dropped_cols

with col_f2:
    st.markdown("#### Subject Grouping (Intra-Subject Leakage Defense)")
    inferred_subject_col = None
    for c in all_cols:
        if c.lower() in ["name", "subject", "patient_id", "subject_id", "patient"]:
            inferred_subject_col = c
            break

    use_grouped = st.checkbox(
        "Enable Subject-Level Grouped Split",
        value=st.session_state.use_grouped_split or (inferred_subject_col is not None),
        help="Essential for speech or biomedical datasets with multiple recordings per subject. "
             "Ensures all recordings from any individual remain in either train OR test set, never both."
    )
    st.session_state.use_grouped_split = use_grouped

    subject_col = None
    if use_grouped:
        subject_opts = [c for c in all_cols if c != target_col]
        default_s_idx = subject_opts.index(inferred_subject_col) if inferred_subject_col in subject_opts else 0
        subject_col = st.selectbox(
            "Select Subject / Patient Identifier Column:",
            options=subject_opts,
            index=default_s_idx,
            help="Observations sharing the same value in this column will be grouped together."
        )
        st.session_state.subject_id_col = subject_col
        st.info(f"Grouped Split will enforce subject independence using '{subject_col}'.")
    else:
        st.session_state.subject_id_col = None

# Feature validation check
try:
    feat_val = DatasetValidator.validate_features(
        df,
        target_col=target_col,
        feature_cols=potential_features,
        dropped_cols=dropped_cols
    )
    st.session_state.feature_cols = feat_val["valid_features"]
    if feat_val["warnings"]:
        with st.expander("Feature Warnings", expanded=False):
            for fw in feat_val["warnings"]:
                st.warning(fw)
except ValueError as fe:
    st.error(f"Feature Validation Error: {fe}")
    st.stop()

st.markdown("### 2. Preprocessing & Partitioning Configuration")

p_col1, p_col2, p_col3 = st.columns(3)

with p_col1:
    num_impute = st.selectbox(
        "Numerical Imputation Strategy:",
        options=["median", "mean"],
        index=0,
        help="Strategy to replace missing numerical values. Median is robust to outliers."
    )
    cat_impute = st.selectbox(
        "Categorical Imputation Strategy:",
        options=["most_frequent"],
        index=0,
        help="Replaces missing categorical entries with the modal value."
    )

with p_col2:
    scale_features = st.checkbox(
        "Apply Standard Scaling (Z-score)",
        value=True,
        help="Crucial for distance-based and gradient algorithms like Logistic Regression and SVM."
    )
    onehot_encode = st.checkbox(
        "One-Hot Encode Categorical Features",
        value=True,
        help="Transforms non-numerical categories into binary indicator columns."
    )

with p_col3:
    test_size = st.slider(
        "Holdout Test Set Size (%):",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data strictly reserved for final unbiased model evaluation."
    ) / 100.0

    random_state = st.number_input(
        "Random Seed (Reproducibility):",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
        help="Fixed seed guarantees reproducible train/test splits and consistent benchmarking."
    )

st.markdown("---")

# Execution Button
if st.button("🚀 Execute Preprocessing & Create Shared Holdout Split", type="primary", use_container_width=True):
    with st.spinner("Processing dataset and creating strict train/test split..."):
        try:
            cfg = PreprocessingConfig(
                target_col=target_col,
                test_size=test_size,
                random_state=random_state,
                num_imputation=num_impute,
                cat_imputation=cat_impute,
                scale_features=scale_features,
                encode_categorical=onehot_encode,
                dropped_cols=dropped_cols,
                subject_id_col=st.session_state.subject_id_col,
                use_grouped_split=st.session_state.use_grouped_split
            )

            pipeline = PreprocessingPipeline(cfg)
            X_train_raw, X_test_raw, y_train, y_test, g_train, g_test = pipeline.prepare_and_split(df)
            X_train_proc, X_test_proc, feat_names = pipeline.fit_transform(X_train_raw, X_test_raw)

            # Store in session state
            st.session_state.preprocessing_pipeline = pipeline
            st.session_state.X_train_proc = X_train_proc
            st.session_state.X_test_proc = X_test_proc
            st.session_state.y_train_raw = y_train.values
            st.session_state.y_test_raw = y_test.values
            st.session_state.groups_train = g_train.values if g_train is not None else None
            st.session_state.groups_test = g_test.values if g_test is not None else None
            st.session_state.transformed_feature_names = feat_names
            st.session_state.class_mapping = pipeline.class_mapping
            st.session_state.preprocessing_done = True
            st.session_state.experiment_results = None  # Reset prior experiment for fresh run

            st.success("✅ Preprocessing and split successfully completed without data leakage!")

        except Exception as e:
            st.error(f"Preprocessing Failed: {str(e)}")

# Display split statistics if preprocessing completed
if st.session_state.preprocessing_done:
    st.markdown("### 📊 Partitioning Summary")
    s1, s2, s3, s4 = st.columns(4)

    s1.metric("Training Samples", len(st.session_state.y_train_raw))
    s2.metric("Holdout Test Samples", len(st.session_state.y_test_raw))
    s3.metric("Transformed Features", len(st.session_state.transformed_feature_names))
    s4.metric("Distinct Classes", len(st.session_state.class_mapping))

    if st.session_state.use_grouped_split and st.session_state.groups_train is not None:
        n_tr_subj = len(np.unique(st.session_state.groups_train))
        n_te_subj = len(np.unique(st.session_state.groups_test))
        st.info(f"Grouped Split Verified: {n_tr_subj} subjects in Training, {n_te_subj} subjects in Holdout Test (Zero Subject Overlap).")

    st.markdown("👉 **Next Step:** Proceed to **Model Training** in the sidebar to configure and benchmark classification algorithms.")
