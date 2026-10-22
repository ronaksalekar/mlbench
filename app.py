"""
MLBench — Comparative Machine Learning Model Evaluation Platform
Home Dashboard and Session Overview.
"""

import streamlit as st
import os
import pandas as pd
from mlbench.state import init_session_state, get_css
from mlbench.data_loader import DataLoader

# Page configuration
st.set_page_config(
    page_title="MLBench — Model Evaluation Platform",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styling and initialize state
st.markdown(get_css(), unsafe_allow_html=True)
init_session_state()

# Header Section
st.title("⚖️ MLBench")
st.subheader("Comparative Machine Learning Model Evaluation Platform")
st.caption("Final-Year Academic Mini-Project in Machine Learning • Department of Computer Engineering")

st.markdown("---")

# Session Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Datasets Loaded",
        value=st.session_state.total_datasets_loaded,
        help="Number of datasets uploaded or loaded in the current session."
    )

with col2:
    st.metric(
        label="Models Trained",
        value=st.session_state.total_models_trained,
        help="Total number of classifier instances trained and evaluated."
    )

with col3:
    exp_status = "Ready / No Runs"
    if st.session_state.experiment_results is not None:
        exp_status = "Completed"
    st.metric(
        label="Latest Experiment Status",
        value=exp_status,
        help="Status of the most recent evaluation experiment run."
    )

with col4:
    best_model_label = "None (No experiments run)"
    if st.session_state.experiment_results is not None:
        res = st.session_state.experiment_results.get("results", {})
        metric_key = st.session_state.ranking_metric.lower().replace("-", "_")
        best_name = "None"
        best_val = -1.0
        for m_name, m_data in res.items():
            val = m_data.get("metrics", {}).get(metric_key, 0.0)
            if val is not None and val > best_val:
                best_val = val
                best_name = m_name
        if best_name != "None":
            best_model_label = f"{best_name} ({best_val:.4f})"

    st.metric(
        label=f"Top Model ({st.session_state.ranking_metric})",
        value=best_model_label,
        help="Champion model from the latest completed evaluation run."
    )

st.markdown("---")

# Quick Actions and Sample Loader
st.markdown("### 🚀 Quick Start & Laboratory Workflow")

act_col1, act_col2 = st.columns([1.5, 1])

with act_col1:
    st.markdown("""
    **MLBench** is an interactive benchmarking platform designed to systematically evaluate and compare supervised classification algorithms on tabular data.
    
    #### Experimental Methodology:
    1. **Data Ingestion & Quality Audit:** Upload CSV datasets, analyze missingness, inspect target distributions, and flag duplicate records.
    2. **Leakage-Free Preprocessing:** Fit imputation, scaling, and one-hot encoding exclusively on training partitions to prevent optimistic bias.
    3. **Grouped vs Stratified Splits:** Support subject-level grouping (e.g. for speech and biomedical recordings) to prevent intra-patient correlation leakage.
    4. **Rigorous Benchmarking:** Train Logistic Regression, Decision Trees, Random Forests, Support Vector Machines, and XGBoost on identical train/test splits.
    5. **Multi-Metric Evaluation:** Compute Accuracy, Precision, Recall, F1-Score, Specificity, and ROC-AUC directly from real holdout test predictions.
    6. **Academic Export:** Download reproducible CSV leaderboards, JSON run metadata, and comprehensive HTML evaluation reports.
    """)

with act_col2:
    st.markdown("#### 🧪 Academic Demonstration")
    st.info(
        "For Machine Learning laboratory demonstrations, you can instantly load the included "
        "**UCI Parkinson's Disease Voice Dataset** or upload your own CSV file in the Explorer."
    )

    sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "parkinsons_sample.csv")
    if os.path.exists(sample_path):
        if st.button("📁 Load Parkinson's Dataset Example", type="primary", use_container_width=True):
            try:
                df, meta = DataLoader.load_csv(sample_path)
                st.session_state.dataset_name = "UCI Parkinson's Voice Sample (120 records)"
                st.session_state.df_raw = df
                st.session_state.df_clean = df.copy()
                st.session_state.dataset_metadata = meta
                st.session_state.target_col = "status"
                st.session_state.subject_id_col = "name"
                st.session_state.dropped_cols = ["name"]
                st.session_state.use_grouped_split = True
                st.session_state.total_datasets_loaded += 1
                st.session_state.preprocessing_done = False
                st.session_state.experiment_results = None
                st.success("Loaded Parkinson's dataset successfully! Proceed to 'Dataset Explorer' or 'Data Preprocessing'.")
            except Exception as e:
                st.error(f"Failed to load sample dataset: {e}")

    if st.session_state.df_raw is not None:
        st.success(f"Active Dataset: **{st.session_state.dataset_name}** ({len(st.session_state.df_raw)} rows)")

st.markdown("---")

# Navigation Guide
st.markdown("### 🧭 Platform Navigation")
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.markdown("#### 1. Dataset Explorer")
    st.caption("Inspect columns, missingness, class distributions, and correlations.")

with nav2:
    st.markdown("#### 2. Preprocessing")
    st.caption("Configure imputation, scaling, identifier exclusion, and grouped splits.")

with nav3:
    st.markdown("#### 3. Model Training")
    st.caption("Select classifiers, tune hyperparameters, and run K-fold cross-validation.")

with nav4:
    st.markdown("#### 4. Comparison & Reports")
    st.caption("Analyze leaderboards, confusion matrices, ROC curves, and export reports.")
