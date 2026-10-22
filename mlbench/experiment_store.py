"""
Experiment Store for managing, persisting, and comparing historical evaluation runs in MLBench.
"""

from typing import Dict, Any, List, Optional
import datetime
import uuid


class ExperimentStore:
    """Manages an in-memory session registry and optional serialization of experiments."""

    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def add_experiment(
        self,
        name: str,
        dataset_name: str,
        target_col: str,
        dataset_info: Dict[str, Any],
        preprocessing_config: Dict[str, Any],
        results: Dict[str, Any],
        ranking_metric: str,
        class_distribution: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None
    ) -> str:
        """Add a completed experiment run to the store."""
        exp_id = str(uuid.uuid4())[:8]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine best model
        best_model = "None"
        best_score = -1.0
        for m_name, m_res in results.items():
            metrics = m_res.get("metrics", {})
            m_score = metrics.get(ranking_metric.lower().replace("-", "_"), 0.0)
            if m_score is None:
                m_score = metrics.get("accuracy", 0.0)
            if m_score > best_score:
                best_score = m_score
                best_model = m_name

        self.experiments[exp_id] = {
            "id": exp_id,
            "name": name or f"Experiment_{exp_id}",
            "timestamp": timestamp,
            "dataset_name": dataset_name,
            "target_col": target_col,
            "dataset_info": dataset_info,
            "preprocessing_config": preprocessing_config,
            "ranking_metric": ranking_metric,
            "results": results,
            "best_model": best_model,
            "best_score": round(best_score, 4),
            "num_models": len(results),
            "class_distribution": class_distribution or {},
            "feature_names": feature_names or []
        }
        return exp_id

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        return self.experiments.get(exp_id)

    def list_experiments(self) -> List[Dict[str, Any]]:
        return list(self.experiments.values())

    def clear(self):
        self.experiments.clear()
