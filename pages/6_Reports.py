"""
MLBench — Page 6: Experiment History & Reports
Downloadable evaluation reports (CSV, JSON, HTML) and syllabus mapping.
"""

import streamlit as st
import pandas as pd
import json
from mlbench.state import init_session_state, get_css
from mlbench.reporting import ReportGenerator

st.set_page_config(page_title="Reports & History — MLBench", page_icon="📑", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)
init_session_state()

st.title("📑 Experiment Reports & Session History")
st.caption("Export reproducible benchmark leaderboards, JSON run summaries, and full academic HTML reports.")

st.markdown("---")

store = st.session_state.experiment_store
all_exps = store.list_experiments()

# Section 1: Session Experiment Registry
st.markdown("### 🗃️ Session Experiment Runs")
if not all_exps:
    st.info("No completed experiment runs logged in this session yet. Complete an evaluation in 'Model Training' to record a run.")
else:
    hist_rows = []
    for exp in all_exps:
        hist_rows.append({
            "Experiment ID": exp["id"],
            "Timestamp": exp["timestamp"],
            "Dataset": exp["dataset_name"],
            "Target": exp["target_col"],
            "Models Trained": exp["num_models"],
            "Ranking Metric": exp["ranking_metric"],
            "Top Model": exp["best_model"],
            "Top Score": exp["best_score"]
        })
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# Section 2: Export Options for Latest Experiment
st.markdown("### 📥 Download Benchmark Reports")

if st.session_state.experiment_results is None:
    st.warning("⚠️ No active experiment results to download. Train models first.")
    st.stop()

results = st.session_state.experiment_results.get("results", {})
ranking_metric = st.session_state.ranking_metric

exp_data = {
    "dataset_info": {
        "dataset_name": st.session_state.dataset_name or "Custom CSV",
        "rows": len(st.session_state.df_raw) if st.session_state.df_raw is not None else 0,
        "features_count": len(st.session_state.transformed_feature_names)
    },
    "target_col": st.session_state.target_col,
    "class_distribution": st.session_state.dataset_metadata.get("class_counts", {}),
    "preprocessing_config": {
        "test_size": 0.2,
        "scale_features": True,
        "use_grouped_split": st.session_state.use_grouped_split
    },
    "ranking_metric": ranking_metric,
    "results": results
}

# 1. Leaderboard CSV
lb_df = ReportGenerator.generate_leaderboard_df(results, ranking_metric)
csv_bytes = lb_df.to_csv(index=False).encode("utf-8")

# 2. JSON Summary
json_str = ReportGenerator.generate_json_summary(exp_data)

# 3. Comprehensive Academic HTML Report
html_report_str = ReportGenerator.generate_html_report(exp_data)

d_col1, d_col2, d_col3 = st.columns(3)

with d_col1:
    st.download_button(
        label="📄 Download Leaderboard (CSV)",
        data=csv_bytes,
        file_name=f"mlbench_leaderboard_{ranking_metric.lower()}.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.caption("Tabular model leaderboard with accuracy, precision, recall, specificity, F1, and CV scores.")

with d_col2:
    st.download_button(
        label="📦 Download Run Metadata (JSON)",
        data=json_str,
        file_name="mlbench_experiment_metadata.json",
        mime="application/json",
        use_container_width=True
    )
    st.caption("Machine-readable JSON specification of hyperparameters, metrics, and data shapes.")

with d_col3:
    st.download_button(
        label="📑 Download Full Report (HTML)",
        data=html_report_str,
        file_name="mlbench_evaluation_report.html",
        mime="text/html",
        type="primary",
        use_container_width=True
    )
    st.caption("Self-contained, printable academic evaluation report with styled tables and charts.")

st.markdown("---")

# Section 3: In-App HTML Report Preview
with st.expander("👁️ Preview Generated Academic HTML Report", expanded=False):
    st.components.v1.html(html_report_str, height=700, scrolling=True)

st.markdown("---")

# Section 4: Syllabus Mapping Table
st.markdown("### 🎓 Academic Syllabus & Curriculum Alignment")
st.markdown("""
The following table documents the direct mapping between MLBench's implementation modules and the Computer Engineering Machine Learning syllabus topics:
""")

syllabus_mapping = [
    {
        "Syllabus Topic": "Classification & Regression Concepts",
        "MLBench Module": "`mlbench/validation.py`, `mlbench/data_loader.py`",
        "Implementation Detail": "Differentiates discrete classification targets from continuous regression variables; verifies class sample thresholds."
    },
    {
        "Syllabus Topic": "Decision Tree Classification",
        "MLBench Module": "`mlbench/models.py`",
        "Implementation Detail": "Implements `DecisionTreeClassifier` with configurable depth, leaf criteria, and feature importance extraction."
    },
    {
        "Syllabus Topic": "Ensemble Learning & Random Forest",
        "MLBench Module": "`mlbench/models.py`, `mlbench/training.py`",
        "Implementation Detail": "Implements `RandomForestClassifier` bagging ensemble; extracts Gini impurity feature importances."
    },
    {
        "Syllabus Topic": "Gradient Boosting (XGBoost)",
        "MLBench Module": "`mlbench/models.py`",
        "Implementation Detail": "Implements gradient-boosted trees via `XGBClassifier` with learning rate, max depth, and regularization controls."
    },
    {
        "Syllabus Topic": "Support Vector Machines (SVM)",
        "MLBench Module": "`mlbench/models.py`",
        "Implementation Detail": "Implements `SVC` with Linear and RBF kernels, standard scaling, and calibrated probability estimates for ROC."
    },
    {
        "Syllabus Topic": "K-Fold Cross-Validation",
        "MLBench Module": "`mlbench/training.py`",
        "Implementation Detail": "Implements Stratified K-Fold and GroupKFold cross-validation exclusively on training splits to prevent data leakage."
    },
    {
        "Syllabus Topic": "Confusion Matrix & Diagnostic Metrics",
        "MLBench Module": "`mlbench/evaluation.py`",
        "Implementation Detail": "Calculates exact True Positives, True Negatives, False Positives, False Negatives, Sensitivity, and Specificity."
    },
    {
        "Syllabus Topic": "Precision, Recall & F-Measure",
        "MLBench Module": "`mlbench/evaluation.py`",
        "Implementation Detail": "Implements mathematical definitions for Precision (PPV), Recall (Sensitivity), and harmonic F1-score with zero-division handling."
    },
    {
        "Syllabus Topic": "ROC Curve & AUC",
        "MLBench Module": "`mlbench/evaluation.py`, `mlbench/visualization.py`",
        "Implementation Detail": "Computes True Positive Rate vs False Positive Rate curves and Area Under Curve (AUC) across classification thresholds."
    },
    {
        "Syllabus Topic": "Dimensionality Reduction (PCA)",
        "MLBench Module": "`mlbench/visualization.py`",
        "Implementation Detail": "Implements 2D Principal Component Analysis to visualize cluster separability and explained variance ratios."
    }
]

st.dataframe(pd.DataFrame(syllabus_mapping), use_container_width=True, hide_index=True)
