"""
Streamlit session state management helper for MLBench.
Ensures uniform state keys and persistent experiment registry across pages.
"""

import streamlit as st
import pandas as pd
from typing import Optional
from mlbench.experiment_store import ExperimentStore


def init_session_state():
    """Initializes standard MLBench session keys if not already present."""
    defaults = {
        "dataset_name": None,
        "df_raw": None,
        "df_clean": None,
        "dataset_metadata": {},
        "target_col": None,
        "feature_cols": [],
        "dropped_cols": [],
        "subject_id_col": None,
        "use_grouped_split": False,
        "X_train_raw": None,
        "X_test_raw": None,
        "y_train_raw": None,
        "y_test_raw": None,
        "groups_train": None,
        "groups_test": None,
        "X_train_proc": None,
        "X_test_proc": None,
        "transformed_feature_names": [],
        "class_mapping": {},
        "preprocessing_pipeline": None,
        "preprocessing_done": False,
        "experiment_results": None,
        "ranking_metric": "F1-Score",
        "experiment_store": ExperimentStore(),
        "total_datasets_loaded": 0,
        "total_models_trained": 0,
        "last_experiment_id": None
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_css() -> str:
    """Returns custom CSS for an academic, modern look."""
    return """
    <style>
        /* Main container styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #1E40AF !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            color: #475569 !important;
            font-weight: 500 !important;
        }

        /* Headers */
        h1, h2, h3 {
            color: #0F172A;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-weight: 700;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        /* Information cards */
        .mlbench-card {
            background-color: #ffffff;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
        }
        .mlbench-pill {
            background-color: #EFF6FF;
            color: #1D4ED8;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
    </style>
    """
