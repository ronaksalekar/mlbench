"""
MLBench — Page 4: Model Comparison Dashboard
Interactive leaderboard, multi-metric benchmarking charts, and head-to-head model comparison.
"""

import streamlit as st
import pandas as pd
from mlbench.state import init_session_state, get_css
from mlbench.reporting import ReportGenerator
from mlbench.visualization import Visualizer

st.set_page_config(page_title="Model Comparison — MLBench", page_icon="🏆", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)
init_session_state()

st.title("🏆 Comparative Model Evaluation Dashboard")
st.caption("Benchmark trained classification algorithms across holdout test accuracy, precision, recall, specificity, and ROC-AUC.")

st.markdown("---")

if st.session_state.experiment_results is None:
    st.info("ℹ️ No model evaluation results found. Please configure and run 'Model Training' first.")
    st.stop()

results = st.session_state.experiment_results.get("results", {})
if not results:
    st.error("No models were successfully trained in the last experiment.")
    st.stop()

# Ranking metric selector
m_opts = ["F1-Score", "Accuracy", "Precision", "Recall", "Specificity", "ROC-AUC"]
cur_metric = st.session_state.ranking_metric if st.session_state.ranking_metric in m_opts else "F1-Score"

sort_metric = st.selectbox(
    "Sort Leaderboard By Metric:",
    options=m_opts,
    index=m_opts.index(cur_metric),
    help="Select the key metric by which the model leaderboard is ranked."
)

st.session_state.ranking_metric = sort_metric

# Generate leaderboard DataFrame
leaderboard_df = ReportGenerator.generate_leaderboard_df(results, ranking_metric=sort_metric)

best_model = leaderboard_df.iloc[0]["Model"]
best_score = leaderboard_df.iloc[0][sort_metric]

# Banner for Best Model
st.markdown(
    f"""
    <div style="background-color: #ECFDF5; border: 1px solid #10B981; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;">
        <h3 style="color: #065F46; margin: 0 0 0.5rem 0;">🏆 Champion Model: {best_model}</h3>
        <p style="color: #047857; margin: 0; font-size: 1rem;">
            Achieved the highest <strong>{sort_metric}</strong> score of <strong>{best_score:.4f}</strong> on the shared holdout test set.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Leaderboard Table
st.markdown("### 📊 Official Model Leaderboard")
st.dataframe(
    leaderboard_df.style.highlight_max(subset=[m for m in m_opts if m in leaderboard_df.columns], color="#D1FAE5"),
    use_container_width=True,
    hide_index=True
)

st.caption("✨ Shaded in light green: highest score per column across all trained algorithms.")

st.markdown("---")

# Visualizations
st.markdown("### 📈 Visual Benchmarks")

v_tab1, v_tab2 = st.tabs(["Leaderboard Ranking Chart", "Multi-Metric Comparative Chart"])

with v_tab1:
    st.plotly_chart(Visualizer.plot_leaderboard_bar(leaderboard_df, metric_col=sort_metric), use_container_width=True)

with v_tab2:
    st.plotly_chart(Visualizer.plot_multi_metric_comparison(leaderboard_df), use_container_width=True)

st.markdown("---")

# Head-to-Head Side-by-Side Comparison
st.markdown("### ⚔️ Head-to-Head Model Comparison")
comp_col1, comp_col2 = st.columns(2)

all_model_names = list(results.keys())
if len(all_model_names) >= 2:
    with comp_col1:
        m1 = st.selectbox("Select Model 1:", options=all_model_names, index=0)
    with comp_col2:
        m2 = st.selectbox("Select Model 2:", options=all_model_names, index=1 if len(all_model_names) > 1 else 0)

    if m1 and m2:
        m1_metrics = results[m1]["metrics"]
        m2_metrics = results[m2]["metrics"]

        comp_data = []
        for met in ["accuracy", "precision", "recall", "f1_score", "specificity", "roc_auc"]:
            v1 = m1_metrics.get(met, 0.0)
            v2 = m2_metrics.get(met, 0.0)
            v1_str = f"{v1:.4f}" if v1 is not None else "N/A"
            v2_str = f"{v2:.4f}" if v2 is not None else "N/A"
            diff = (v1 - v2) if (v1 is not None and v2 is not None) else 0.0
            comp_data.append({
                "Metric": met.replace("_", " ").title(),
                f"{m1}": v1_str,
                f"{m2}": v2_str,
                "Delta (M1 - M2)": f"{diff:+.4f}" if v1 is not None and v2 is not None else "N/A"
            })

        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
else:
    st.info("Train at least 2 models in 'Model Training' to view head-to-head comparisons.")

st.markdown("---")

# Educational Syllabus Tooltips
with st.expander("📚 Academic Metric Reference & Syllabus Guide", expanded=False):
    st.markdown("""
    - **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$. Proportion of correct predictions overall. Can be deceptively high in imbalanced classes.
    - **Precision (PPV):** $TP / (TP + FP)$. Measures how many predicted positives were actually positive (minimizes false alarms).
    - **Recall (Sensitivity / TPR):** $TP / (TP + FN)$. Measures how many actual positives were correctly caught (crucial for medical diagnosis).
    - **Specificity (TNR):** $TN / (TN + FP)$. Measures how many actual negatives were correctly identified as negative.
    - **F1-Score:** $2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$. Harmonic mean balance between Precision and Recall.
    - **ROC-AUC:** Area Under the Receiver Operating Characteristic curve. Measures discrimination capability across all classification thresholds.
    """)
