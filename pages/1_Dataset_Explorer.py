"""
MLBench — Page 1: Dataset Explorer
Dataset ingestion, exploratory data analysis, quality auditing, and target selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from mlbench.state import init_session_state, get_css
from mlbench.data_loader import DataLoader
from mlbench.visualization import Visualizer

st.set_page_config(page_title="Dataset Explorer — MLBench", page_icon="🔍", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)
init_session_state()

st.title("🔍 Dataset Explorer & Quality Audit")
st.caption("Upload tabular CSV datasets, inspect feature types, audit missing values, and select prediction targets.")

st.markdown("---")

# Section 1: Ingestion
col_up1, col_up2 = st.columns([2, 1])

with col_up1:
    uploaded_file = st.file_uploader(
        "Upload Tabular Dataset (CSV)",
        type=["csv"],
        help="Upload a comma-separated values file containing labelled classification data."
    )

with col_up2:
    st.markdown("#### Handling Duplicates")
    remove_dups = st.checkbox(
        "Remove duplicate rows",
        value=False,
        help="Warning: In some biomedical or repeated-measurement datasets, duplicate feature observations may be valid."
    )
    if remove_dups:
        st.warning("Duplicate rows will be dropped upon loading. Verify that this aligns with your domain.")

# Process File Upload
if uploaded_file is not None:
    try:
        df, meta = DataLoader.load_csv(uploaded_file, drop_duplicates=remove_dups)
        st.session_state.dataset_name = uploaded_file.name
        st.session_state.df_raw = df
        st.session_state.df_clean = df.copy()
        st.session_state.dataset_metadata = meta
        st.session_state.total_datasets_loaded += 1
        st.session_state.preprocessing_done = False
        st.session_state.experiment_results = None
        st.success(f"Successfully loaded '{uploaded_file.name}' with {len(df):,} rows and {len(df.columns)} columns!")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# Check if dataset is available
if st.session_state.df_raw is None:
    st.info("👋 No dataset currently loaded. Please upload a CSV above or load the Parkinson's sample dataset from the Home page.")
    st.stop()

df = st.session_state.df_raw

# Section 2: Summary Metrics
st.markdown("### 📊 Dataset Overview")
m1, m2, m3, m4 = st.columns(4)

total_missing = int(df.isna().sum().sum())
total_cells = df.shape[0] * df.shape[1]
missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0

m1.metric("Total Rows", f"{len(df):,}")
m2.metric("Total Columns", len(df.columns))
m3.metric("Duplicate Rows", f"{st.session_state.dataset_metadata.get('duplicate_count', 0):,}")
m4.metric("Missing Cells", f"{total_missing:,} ({missing_pct}%)")

# Section 3: Data Preview
with st.expander("👁️ View Dataset Preview", expanded=True):
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=50, value=10)
    st.dataframe(df.head(preview_rows), use_container_width=True)

# Section 4: Column Health & Types
st.markdown("### 📋 Column Inventory & Role Inference")
col_summary_df = DataLoader.get_column_summary(df)
st.dataframe(col_summary_df, use_container_width=True, hide_index=True)

# Quality Warnings
warnings = []
for _, row in col_summary_df.iterrows():
    if row["Missing (%)"] > 40.0:
        warnings.append(f"⚠️ Column **{row['Column Name']}** has a high missing rate ({row['Missing (%)']}%).")
    if row["Unique Values"] <= 1:
        warnings.append(f"⚠️ Column **{row['Column Name']}** is constant (only {row['Unique Values']} unique value).")

if warnings:
    with st.expander("⚠️ Data Quality Alerts", expanded=True):
        for w in warnings:
            st.warning(w)

# Section 5: Target Column Selection
st.markdown("### 🎯 Target Variable Selection")
candidate_targets = DataLoader.detect_candidate_targets(df)
default_idx = 0
if st.session_state.target_col in candidate_targets:
    default_idx = candidate_targets.index(st.session_state.target_col)

selected_target = st.selectbox(
    "Select the Classification Target Column:",
    options=candidate_targets,
    index=default_idx,
    help="Select the column containing the class labels to predict. Ranked candidates are listed first."
)

st.session_state.target_col = selected_target

# Target Distribution Analysis
target_series = df[selected_target].dropna()
dist_df = DataLoader.get_class_distribution(target_series)

t_col1, t_col2 = st.columns([1, 1.5])

with t_col1:
    st.markdown("#### Class Distribution Table")
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # Class balance check
    majority_pct = dist_df["Percentage (%)"].max() if not dist_df.empty else 0
    if majority_pct >= 75.0:
        st.warning(
            f"Class Imbalance Detected: The majority class represents {majority_pct}% of the observations. "
            "Accuracy alone will be misleading; evaluate with Precision, Recall, and F1-Score."
        )

with t_col2:
    st.plotly_chart(Visualizer.plot_target_distribution(dist_df), use_container_width=True)

# Section 6: Feature Correlation Analysis
st.markdown("### 📈 Numerical Feature Correlation")
with st.expander("View Correlation Heatmap", expanded=False):
    corr_df = DataLoader.get_numerical_correlation(df)
    if not corr_df.empty:
        st.plotly_chart(Visualizer.plot_correlation_heatmap(corr_df), use_container_width=True)
    else:
        st.info("No numerical features found for correlation analysis.")
