"""
MLBench — Page 5: Model Analysis
In-depth Confusion Matrix, ROC curves, Feature Importance, and PCA Projections.
"""

import streamlit as st
import numpy as np
import pandas as pd
from mlbench.state import init_session_state, get_css
from mlbench.visualization import Visualizer
from mlbench.training import TrainingRunner

st.set_page_config(page_title="Model Analysis — MLBench", page_icon="🔬", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)
init_session_state()

st.title("🔬 Deep Diagnostic & Feature Analysis")
st.caption("Inspect per-model confusion matrices, ROC curves, feature importances, and PCA projections.")

st.markdown("---")

if st.session_state.experiment_results is None:
    st.info("ℹ️ No trained models found. Please train models in 'Model Training' first.")
    st.stop()

results = st.session_state.experiment_results.get("results", {})
if not results:
    st.error("No model results available.")
    st.stop()

model_names = list(results.keys())
selected_model = st.selectbox(
    "Select Model for Detailed Diagnostic Analysis:",
    options=model_names,
    index=0
)

res = results[selected_model]
metrics = res.get("metrics", {})
class_mapping = st.session_state.class_mapping
class_names = [class_mapping[k] for k in sorted(class_mapping.keys())]

# Metric Banner for Selected Model
st.markdown(f"### 🎯 Performance Breakdown: **{selected_model}**")
mb1, mb2, mb3, mb4, mb5 = st.columns(5)
mb1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
mb2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
mb3.metric("Recall (Sensitivity)", f"{metrics.get('recall', 0):.4f}")
mb4.metric("Specificity", f"{metrics.get('specificity', 0):.4f}")
mb5.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")

st.markdown("---")

# Section 1: Confusion Matrix & ROC Curve
st.markdown("### 1. Confusion Matrix & ROC Curve")
cm_col, roc_col = st.columns([1, 1.2])

with cm_col:
    cm = metrics.get("confusion_matrix", [])
    if cm:
        st.plotly_chart(
            Visualizer.plot_confusion_matrix(cm, class_names=class_names, model_name=selected_model),
            use_container_width=True
        )

        # Show binary breakdown counts if binary
        if metrics.get("is_binary") and metrics.get("binary_counts"):
            b_cnt = metrics["binary_counts"]
            st.markdown(
                f"""
                <div style="font-size: 0.9rem; line-height: 1.6; background: #F8FAFC; padding: 12px; border-radius: 6px; border: 1px solid #E2E8F0;">
                    • <b>True Positives (TP):</b> {b_cnt['tp']} (Correctly detected positive cases)<br>
                    • <b>True Negatives (TN):</b> {b_cnt['tn']} (Correctly identified negative cases)<br>
                    • <b>False Positives (FP):</b> {b_cnt['fp']} (Type I Error / False alarms)<br>
                    • <b>False Negatives (FN):</b> {b_cnt['fn']} (Type II Error / Missed detections)
                </div>
                """,
                unsafe_allow_html=True
            )

with roc_col:
    roc_data = metrics.get("roc_data")
    if roc_data:
        roc_fig = Visualizer.plot_roc_curve(roc_data, model_name=selected_model)
        if roc_fig:
            st.plotly_chart(roc_fig, use_container_width=True)
    else:
        st.warning(
            f"ROC Curve unavailable for {selected_model}: "
            f"{metrics.get('roc_error_reason', 'Model does not support probability predictions.')}"
        )

st.markdown("---")

# Section 2: Feature Importance Analysis
st.markdown("### 2. Feature Importance & Attribution")
fi_col1, fi_col2 = st.columns([1.5, 1])

with fi_col1:
    fi_dict = res.get("feature_importance")
    perm_toggle = st.checkbox("Compute Model-Agnostic Permutation Importance on Test Set", value=False)

    if perm_toggle:
        with st.spinner("Computing permutation importance across 5 repeats..."):
            model_obj = res.get("model_obj")
            if model_obj and st.session_state.X_test_proc is not None:
                fi_dict = TrainingRunner.compute_permutation_importance(
                    model=model_obj,
                    X_test=st.session_state.X_test_proc,
                    y_test=st.session_state.y_test_raw,
                    feature_names=st.session_state.transformed_feature_names
                )
                st.caption("Showing permutation feature importance calculated by shuffling test set features.")

    if fi_dict:
        st.plotly_chart(
            Visualizer.plot_feature_importance(
                fi_dict,
                title=f"Top Predictive Features — {selected_model}",
                top_n=15
            ),
            use_container_width=True
        )
    else:
        st.info("Direct feature importance weights are not available for this model type. Enable 'Permutation Importance' above.")

with fi_col2:
    st.markdown("#### Top 10 Attributes Table")
    if fi_dict:
        fi_df = pd.DataFrame(
            [{"Feature": k, "Importance": round(v, 4)} for k, v in list(fi_dict.items())[:10]]
        )
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

    st.info(
        "⚠️ **Academic Note:** Feature importance indicates statistical association with the prediction outcome, "
        "not medical or mechanistic causality."
    )

st.markdown("---")

# Section 3: PCA 2D Dimensionality Reduction
st.markdown("### 3. Principal Component Analysis (PCA) Projection")
pca_col1, pca_col2 = st.columns([1.5, 1])

with pca_col1:
    if st.session_state.X_test_proc is not None and st.session_state.y_test_raw is not None:
        pca_fig = Visualizer.plot_pca_projection(
            X=st.session_state.X_test_proc,
            y=st.session_state.y_test_raw,
            class_mapping=st.session_state.class_mapping
        )
        st.plotly_chart(pca_fig, use_container_width=True)

with pca_col2:
    st.markdown("#### Dimensionality Reduction Context")
    st.markdown("""
    - **PCA** projects high-dimensional feature spaces onto orthogonal axes of maximum variance.
    - Inspect whether classes form separable clusters or show significant overlap in the reduced 2D plane.
    - **Note:** While PCA aids 2D visualization, linear projections do not capture all nonlinear boundaries that tree ensembles or kernel SVMs can learn.
    """)
