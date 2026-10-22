"""
Data loading and exploratory analysis utilities for MLBench.
Handles CSV ingestion, data type inference, quality checks, and metadata extraction.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import io
import re

MAX_FILE_SIZE_MB = 50
MAX_ROW_COUNT = 100_000


class DataLoader:
    """Utility class for loading and inspecting tabular datasets."""

    @staticmethod
    def load_csv(
        file_or_path: Any,
        drop_duplicates: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load CSV from a file path, BytesIO, or uploaded file buffer.
        
        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        try:
            if isinstance(file_or_path, (str, io.BytesIO, io.StringIO)) or hasattr(file_or_path, 'read'):
                # Read CSV with pandas
                df = pd.read_csv(file_or_path)
            else:
                raise ValueError("Unsupported file format or source provided.")

            if df.empty:
                raise ValueError("Uploaded CSV file is completely empty.")

            if len(df) > MAX_ROW_COUNT:
                raise ValueError(
                    f"Dataset contains {len(df):,} rows, exceeding the educational platform limit of {MAX_ROW_COUNT:,}."
                )

            initial_rows = len(df)
            duplicate_count = int(df.duplicated().sum())

            if drop_duplicates and duplicate_count > 0:
                df = df.drop_duplicates().reset_index(drop=True)

            metadata = {
                "initial_row_count": initial_rows,
                "current_row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "duplicate_count": duplicate_count,
                "dropped_duplicates": drop_duplicates and duplicate_count > 0,
            }

            return df, metadata

        except pd.errors.EmptyDataError:
            raise ValueError("Uploaded CSV file is completely empty and contains no data or headers.")
        except pd.errors.ParserError as pe:
            raise ValueError(f"CSV Parsing Error: {str(pe)}")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Failed to load dataset: {str(e)}")

    @staticmethod
    def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a comprehensive summary table for each column.
        Includes dtype, missing count, missing percentage, unique count, and sample values.
        """
        summary_rows = []
        n_rows = len(df)

        for col in df.columns:
            s = df[col]
            missing_cnt = int(s.isna().sum())
            missing_pct = round((missing_cnt / n_rows * 100) if n_rows > 0 else 0, 2)
            n_unique = int(s.nunique(dropna=True))
            dtype_str = str(s.dtype)

            # Heuristic detection of inferred column role
            col_lower = str(col).lower()
            is_constant = (n_unique <= 1)
            is_id = False

            if n_unique == n_rows and n_rows > 1:
                is_id = True
            elif any(k in col_lower for k in ["id", "name", "subject", "patient", "sample", "recording"]):
                is_id = True

            inferred_type = "Numerical" if pd.api.types.is_numeric_dtype(s) else "Categorical"
            if is_id:
                inferred_role = "Identifier / Key"
            elif is_constant:
                inferred_role = "Constant Column"
            elif col_lower in ["status", "target", "label", "class", "diagnosis", "outcome", "y"]:
                inferred_role = "Potential Target"
            else:
                inferred_role = f"Feature ({inferred_type})"

            sample_vals = ", ".join([str(v) for v in s.dropna().unique()[:3]])

            summary_rows.append({
                "Column Name": col,
                "Data Type": dtype_str,
                "Inferred Role": inferred_role,
                "Missing Values": missing_cnt,
                "Missing (%)": missing_pct,
                "Unique Values": n_unique,
                "Sample Values": sample_vals
            })

        return pd.DataFrame(summary_rows)

    @staticmethod
    def detect_candidate_targets(df: pd.DataFrame) -> List[str]:
        """
        Rank and list likely candidate target columns.
        Prioritizes columns named target, status, label, class, outcome, diagnosis,
        as well as low-cardinality discrete columns.
        """
        candidates = []
        scores = {}

        common_target_names = {"status", "target", "label", "class", "outcome", "diagnosis", "y", "result"}

        for col in df.columns:
            col_l = str(col).lower()
            n_unique = df[col].nunique(dropna=True)
            score = 0

            # Direct name match
            if col_l in common_target_names:
                score += 50
            elif any(name in col_l for name in common_target_names):
                score += 25

            # Binary or small multiclass target
            if n_unique == 2:
                score += 30
            elif 3 <= n_unique <= 10:
                score += 15

            # Penalize unique id columns
            if n_unique == len(df) or "id" in col_l or "name" in col_l:
                score -= 40

            # Penalize columns with high missing rates (>50%)
            if df[col].isna().mean() > 0.5:
                score -= 30

            scores[col] = score

        # Sort descending by score
        sorted_cols = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        # Return all columns with positive scores or at least all columns ordered
        return sorted_cols

    @staticmethod
    def extract_subject_ids(df: pd.DataFrame, id_col: Optional[str] = None) -> Optional[pd.Series]:
        """
        Extract or infer subject/patient group IDs for grouped cross-validation.
        For Parkinson's UCI dataset (e.g. phon_R01_S01_1), the subject identifier
        is the prefix before the last underscore.
        """
        if id_col and id_col in df.columns:
            col_vals = df[id_col].astype(str)
            # Check for UCI Parkinson's pattern: 'phon_R01_S01_1' -> 'phon_R01_S01'
            has_uci_pattern = col_vals.str.contains(r'_S\d+_\d+', regex=True).all()
            if has_uci_pattern:
                subjects = col_vals.apply(lambda x: re.sub(r'_\d+$', '', x))
                return subjects
            return col_vals

        # Auto-detect subject/patient columns
        for col in df.columns:
            col_l = col.lower()
            if any(term in col_l for term in ["patient", "subject", "person"]):
                return df[col].astype(str)
            if col_l in ["name", "id"]:
                sample = df[col].astype(str)
                if sample.str.contains(r'_S\d+_\d+', regex=True).any():
                    subjects = sample.apply(lambda x: re.sub(r'_\d+$', '', x))
                    return subjects

        return None

    @staticmethod
    def get_class_distribution(series: pd.Series) -> pd.DataFrame:
        """Calculate class distribution with counts, proportions, and percentages."""
        clean_s = series.dropna()
        counts = clean_s.value_counts().sort_index()
        proportions = clean_s.value_counts(normalize=True).sort_index() * 100

        dist_df = pd.DataFrame({
            "Class Label": [str(c) for c in counts.index],
            "Count": counts.values,
            "Percentage (%)": [round(p, 2) for p in proportions.values]
        })
        return dist_df

    @staticmethod
    def get_numerical_correlation(df: pd.DataFrame, max_cols: int = 30) -> pd.DataFrame:
        """Compute correlation matrix for numerical features."""
        num_df = df.select_dtypes(include=[np.number])
        if num_df.empty:
            return pd.DataFrame()
        # Limit columns if too wide for readable heatmap
        if num_df.shape[1] > max_cols:
            num_df = num_df.iloc[:, :max_cols]
        return num_df.corr().round(3)
