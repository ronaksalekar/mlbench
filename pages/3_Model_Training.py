"""
MLBench — Page 3: Model Training
Train Logistic Regression, Decision Trees, Random Forests, SVM, and XGBoost with Cross-Validation.
"""

import streamlit as st
import numpy as np
import time
from mlbench.state import init_session_state, get_css
from mlbench.models import ModelFactory
from mlbench.training import TrainingRunner
from mlbench.validation import DatasetValidator

st.set_page_config(page_title="Model Training — MLBench", page_icon="🏋️", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)
init_session_state()

st.title("🏋️ Model Training & Cross-Validation")
st.caption("Configure hyperparameters, execute stratified/grouped K-fold cross-validation, and evaluate on shared holdout data.")

st.markdown("---")

if not st.session_state.preprocessing_done or st.session_state.X_train_proc is None:
    st.warning("⚠️ Preprocessing has not been executed yet. Please complete 'Data Preprocessing' first.")
    st.stop()

X_train = st.session_state.X_train_proc
X_test = st.session_state.X_test_proc
y_train = st.session_state.y_train_raw
y_test = st.session_state.y_test_raw
feat_names = st.session_state.transformed_feature_names
class_mapping = st.session_state.class_mapping
groups_tr = st.session_state.groups_train

# Section 1: Model Selection
st.markdown("### 1. Select Classifiers for Evaluation")

available_models = ModelFactory.AVAILABLE_MODELS.copy()
xgb_avail = ModelFactory.is_xgboost_available()

default_selection = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine (SVM)"
]
if xgb_avail:
    default_selection.append("XGBoost")

selected_models = st.multiselect(
    "Choose Algorithms to Train & Benchmark:",
    options=available_models,
    default=default_selection,
    help="Select which classification algorithms to train on the shared training partition."
)

if not xgb_avail:
    st.info("ℹ️ Note: XGBoost is not installed in the current environment. Other models will run normally.")

if not selected_models:
    st.error("Please select at least one algorithm to train.")
    st.stop()

# Section 2: Hyperparameter Tuning Expanders
st.markdown("### 2. Algorithm Hyperparameters")
hyperparams = {}

for m_name in selected_models:
    with st.expander(f"⚙️ Hyperparameters: {m_name}", expanded=False):
        if m_name == "Logistic Regression":
            col_lr1, col_lr2, col_lr3 = st.columns(3)
            c_val = col_lr1.number_input("Inverse Regularization Strength (C):", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
            max_iter = col_lr2.number_input("Maximum Iterations:", min_value=100, max_value=5000, value=1000, step=100)
            solver = col_lr3.selectbox("Optimization Solver:", options=["lbfgs", "liblinear", "saga"], index=0)
            hyperparams[m_name] = {"C": c_val, "max_iter": max_iter, "solver": solver}

        elif m_name == "Decision Tree":
            col_dt1, col_dt2, col_dt3 = st.columns(3)
            max_d = col_dt1.slider("Max Tree Depth (None = Unlimited):", min_value=1, max_value=20, value=6)
            min_leaf = col_dt2.number_input("Min Samples per Leaf:", min_value=1, max_value=20, value=2, step=1)
            crit = col_dt3.selectbox("Split Criterion:", options=["gini", "entropy", "log_loss"], index=0)
            hyperparams[m_name] = {"max_depth": max_d, "min_samples_leaf": min_leaf, "criterion": crit}

        elif m_name == "Random Forest":
            col_rf1, col_rf2, col_rf3 = st.columns(3)
            n_est = col_rf1.slider("Number of Trees (n_estimators):", min_value=10, max_value=500, value=100, step=10)
            max_d_rf = col_rf2.slider("Max Depth:", min_value=1, max_value=25, value=8)
            min_leaf_rf = col_rf3.number_input("Min Samples per Leaf:", min_value=1, max_value=20, value=2, step=1, key="rf_leaf")
            hyperparams[m_name] = {"n_estimators": n_est, "max_depth": max_d_rf, "min_samples_leaf": min_leaf_rf}

        elif m_name == "Support Vector Machine (SVM)":
            col_svm1, col_svm2 = st.columns(2)
            c_svm = col_svm1.number_input("Regularization Parameter (C):", min_value=0.01, max_value=100.0, value=1.0, step=0.1, key="svm_c")
            kernel = col_svm2.selectbox("Kernel Type:", options=["rbf", "linear", "poly", "sigmoid"], index=0)
            st.caption("ℹ️ Probability estimation enabled for ROC-AUC computation (incurs small calibration computational cost).")
            hyperparams[m_name] = {"C": c_svm, "kernel": kernel, "probability": True}

        elif m_name == "XGBoost":
            col_xgb1, col_xgb2, col_xgb3 = st.columns(3)
            n_est_xgb = col_xgb1.slider("Boosting Rounds (n_estimators):", min_value=20, max_value=300, value=100, step=10)
            lr = col_xgb2.number_input("Learning Rate (eta):", min_value=0.01, max_value=1.0, value=0.1, step=0.02)
            max_d_xgb = col_xgb3.slider("Max Depth:", min_value=1, max_value=15, value=4, key="xgb_depth")
            hyperparams[m_name] = {"n_estimators": n_est_xgb, "learning_rate": lr, "max_depth": max_d_xgb, "subsample": 0.8}

# Section 3: Cross-Validation & Metric Selection
st.markdown("### 3. Evaluation & Validation Strategy")
cv_col1, cv_col2, cv_col3 = st.columns(3)

with cv_col1:
    enable_cv = st.checkbox(
        "Enable K-Fold Cross-Validation on Train Split",
        value=True,
        help="Estimates generalization variance across training folds before evaluating the untouched holdout test set."
    )

with cv_col2:
    cv_folds = st.slider("Cross-Validation Folds (K):", min_value=2, max_value=10, value=5, disabled=not enable_cv)
    if enable_cv:
        # Check fold count feasibility
        unique_classes, counts = np.unique(y_train, return_counts=True)
        min_class_cnt = int(np.min(counts))
        if cv_folds > min_class_cnt:
            st.warning(
                f"Requested K={cv_folds}, but the smallest class in training data has only {min_class_cnt} observations. "
                f"K cannot exceed {min_class_cnt} for stratified evaluation."
            )

with cv_col3:
    ranking_metric = st.selectbox(
        "Primary Leaderboard Metric:",
        options=["F1-Score", "Accuracy", "Precision", "Recall", "Specificity", "ROC-AUC"],
        index=0,
        help="Metric used to rank models in the comparison leaderboard."
    )
    st.session_state.ranking_metric = ranking_metric

st.markdown("---")

# Section 4: Training Execution
if st.button("🚀 Start Training All Selected Models", type="primary", use_container_width=True):
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(pct, msg):
        progress_bar.progress(pct)
        status_text.text(msg)

    with st.spinner("Training models and calculating metrics..."):
        cv_metric_name = ranking_metric.lower().replace("-", "_")
        exp_output = TrainingRunner.run_experiment(
            selected_models=selected_models,
            hyperparams=hyperparams,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feat_names,
            class_mapping=class_mapping,
            enable_cv=enable_cv,
            cv_folds=cv_folds,
            cv_metric=cv_metric_name,
            groups_train=groups_tr,
            random_state=42,
            progress_callback=update_progress
        )

        st.session_state.experiment_results = exp_output
        st.session_state.total_models_trained += exp_output["successful_count"]

        # Save to experiment store
        exp_id = st.session_state.experiment_store.add_experiment(
            name=f"Run_{st.session_state.dataset_name}_{ranking_metric}",
            dataset_name=st.session_state.dataset_name or "Unknown Dataset",
            target_col=st.session_state.target_col or "target",
            dataset_info={
                "dataset_name": st.session_state.dataset_name,
                "rows": len(df),
                "features_count": len(feat_names)
            },
            preprocessing_config={
                "test_size": 0.2,
                "scale_features": True,
                "use_grouped_split": st.session_state.use_grouped_split
            },
            results=exp_output["results"],
            ranking_metric=ranking_metric,
            class_distribution=st.session_state.dataset_metadata.get("class_counts", {}),
            feature_names=feat_names
        )
        st.session_state.last_experiment_id = exp_id

        progress_bar.progress(1.0)
        status_text.text("Training run complete!")
        st.success(f"🎉 Successfully trained and evaluated {exp_output['successful_count']} model(s) on the holdout test set!")

        if exp_output["errors"]:
            st.error("Some models encountered errors during training:")
            for m_err_name, err_msg in exp_output["errors"].items():
                st.warning(f"• **{m_err_name}**: {err_msg}")

if st.session_state.experiment_results is not None:
    res = st.session_state.experiment_results["results"]
    st.markdown("### 📋 Quick Run Summary")
    card_cols = st.columns(len(res))
    for c_idx, (m_name, m_data) in enumerate(res.items()):
        with card_cols[c_idx % len(card_cols)]:
            m_metrics = m_data.get("metrics", {})
            st.metric(
                label=m_name,
                value=f"{m_metrics.get(st.session_state.ranking_metric.lower().replace('-', '_'), 0.0):.4f}",
                delta=f"{m_data.get('training_time_sec', 0):.3f}s train time"
            )

    st.markdown("👉 **Next Step:** Open **Model Comparison** in the sidebar to view the interactive leaderboard, charts, and metrics.")
