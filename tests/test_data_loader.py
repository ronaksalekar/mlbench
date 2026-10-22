"""
Unit tests for DataLoader.
"""

import pytest
import pandas as pd
import io
from mlbench.data_loader import DataLoader


def test_load_csv_from_string():
    csv_data = "colA,colB,target\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n"
    df, meta = DataLoader.load_csv(io.StringIO(csv_data))
    assert len(df) == 3
    assert meta["column_count"] == 3
    assert "target" in meta["columns"]
    assert meta["duplicate_count"] == 0


def test_load_csv_duplicates():
    csv_data = "colA,target\n1,0\n1,0\n2,1\n"
    # Without drop
    df, meta = DataLoader.load_csv(io.StringIO(csv_data), drop_duplicates=False)
    assert len(df) == 3
    assert meta["duplicate_count"] == 1

    # With drop
    df_dedup, meta_dedup = DataLoader.load_csv(io.StringIO(csv_data), drop_duplicates=True)
    assert len(df_dedup) == 2
    assert meta_dedup["dropped_duplicates"] is True


def test_load_csv_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        DataLoader.load_csv(io.StringIO(""))


def test_column_summary():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "feature": [10.5, None, 12.3],
        "category": ["A", "B", "A"],
        "target": [0, 1, 0]
    })
    summary = DataLoader.get_column_summary(df)
    assert len(summary) == 4
    feature_row = summary[summary["Column Name"] == "feature"].iloc[0]
    assert feature_row["Missing Values"] == 1
    assert feature_row["Missing (%)"] == 33.33


def test_detect_candidate_targets():
    df = pd.DataFrame({
        "patient_id": [101, 102, 103, 104],
        "feature1": [1.1, 2.2, 3.3, 4.4],
        "status": [0, 1, 0, 1]
    })
    candidates = DataLoader.detect_candidate_targets(df)
    assert len(candidates) > 0
    assert candidates[0] == "status"


def test_extract_subject_ids_uci_pattern():
    df = pd.DataFrame({
        "name": ["phon_R01_S01_1", "phon_R01_S01_2", "phon_R01_S02_1"],
        "val": [1, 2, 3]
    })
    subjects = DataLoader.extract_subject_ids(df, "name")
    assert subjects is not None
    assert list(subjects) == ["phon_R01_S01", "phon_R01_S01", "phon_R01_S02"]
