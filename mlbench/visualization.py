"""
Plotly visualization builders for MLBench.
Provides interactive, aesthetic charts for model comparison, confusion matrices,
ROC curves, feature importances, PCA projection, and data distributions.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA


PRIMARY_BLUE = "#1E40AF"
ACCENT_CYAN = "#0EA5E9"
SUCCESS_GREEN = "#10B981"
DANGER_RED = "#EF4444"
NEUTRAL_DARK = "#1E293B"
NEUTRAL_LIGHT = "#F8FAFC"
BORDER_GRAY = "#E2E8F0"

PLOT_LAYOUT_DEFAULTS = dict(
    font=dict(family="Inter, Roboto, sans-serif", size=12, color="#1E293B"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=40, t=50, b=40),
)


class Visualizer:
    """Creates polished Plotly figures for MLBench."""

    @staticmethod
    def plot_leaderboard_bar(
        leaderboard_df: pd.DataFrame,
        metric_col: str = "F1-Score"
    ) -> go.Figure:
        """Single-metric comparative bar chart with the top model highlighted."""
        df_sorted = leaderboard_df.sort_values(by=metric_col, ascending=True).copy()
        max_val = df_sorted[metric_col].max()

        colors = [
            SUCCESS_GREEN if val == max_val else PRIMARY_BLUE
            for val in df_sorted[metric_col]
        ]

        fig = go.Figure(
            go.Bar(
                x=df_sorted[metric_col],
                y=df_sorted["Model"],
                orientation="h",
                marker=dict(color=colors, line=dict(color="#1E3A8A", width=1)),
                text=[f"{v:.4f}" for v in df_sorted[metric_col]],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>" + f"{metric_col}: " + "%{x:.4f}<extra></extra>"
            )
        )

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            title=f"Model Comparison Ranked by {metric_col} (Best Highlighted)",
            xaxis_title=metric_col,
            yaxis_title="",
            height=350 + len(df_sorted) * 35,
            xaxis=dict(range=[0, min(1.05, max(1.0, max_val * 1.1)), ], gridcolor=BORDER_GRAY),
            yaxis=dict(gridcolor="rgba(0,0,0,0)")
        )
        return fig

    @staticmethod
    def plot_multi_metric_comparison(
        leaderboard_df: pd.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> go.Figure:
        """Grouped bar chart comparing multiple classification metrics across models."""
        if metrics is None:
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]

        valid_metrics = [m for m in metrics if m in leaderboard_df.columns]

        fig = go.Figure()
        palette = ["#1E40AF", "#0EA5E9", "#10B981", "#8B5CF6", "#F59E0B", "#EC4899"]

        for idx, metric in enumerate(valid_metrics):
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=leaderboard_df["Model"],
                    y=leaderboard_df[metric],
                    marker_color=palette[idx % len(palette)],
                    hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.4f}}<extra></extra>"
                )
            )

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            title="Comprehensive Multi-Metric Model Benchmark",
            barmode="group",
            xaxis_title="Classification Algorithm",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.05], gridcolor=BORDER_GRAY),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450
        )
        return fig

    @staticmethod
    def plot_confusion_matrix(
        cm: List[List[int]],
        class_names: List[str],
        model_name: str = ""
    ) -> go.Figure:
        """Heatmap of confusion matrix with counts and percentages."""
        cm_array = np.array(cm)
        total = np.sum(cm_array)

        # Prepare cell annotations
        annotations = []
        is_binary = (cm_array.shape == (2, 2))
        labels_matrix = [["TN", "FP"], ["FN", "TP"]] if is_binary else None

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                cnt = cm_array[i, j]
                pct = (cnt / total * 100) if total > 0 else 0
                label_txt = f"{cnt}<br>({pct:.1f}%)"
                if is_binary and labels_matrix:
                    label_txt = f"<b>{labels_matrix[i][j]}</b><br>{cnt}<br>({pct:.1f}%)"

                annotations.append(
                    dict(
                        x=class_names[j],
                        y=class_names[i],
                        text=label_txt,
                        showarrow=False,
                        font=dict(color="white" if cnt > cm_array.max() * 0.5 else "#1E293B", size=13)
                    )
                )

        fig = go.Figure(
            data=go.Heatmap(
                z=cm_array,
                x=class_names,
                y=class_names,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Count"),
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            )
        )

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            title=f"Confusion Matrix — {model_name}" if model_name else "Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            annotations=annotations,
            height=420,
            width=480,
            yaxis=dict(autorange="reversed")
        )
        return fig

    @staticmethod
    def plot_roc_curve(
        roc_data: Optional[Dict[str, Any]],
        model_name: str = ""
    ) -> Optional[go.Figure]:
        """Interactive ROC curve with reference chance diagonal."""
        if not roc_data:
            return None

        fig = go.Figure()

        # Diagonal 50% baseline
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#94A3B8", dash="dash", width=2),
                name="Random Chance (AUC = 0.50)"
            )
        )

        if roc_data.get("type") == "binary":
            fpr = roc_data.get("fpr", [])
            tpr = roc_data.get("tpr", [])
            auc = roc_data.get("auc", 0.0)

            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    line=dict(color=PRIMARY_BLUE, width=3),
                    name=f"{model_name} (AUC = {auc:.4f})",
                    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
                )
            )

        elif roc_data.get("type") == "multiclass_ovr":
            class_curves = roc_data.get("class_curves", {})
            auc = roc_data.get("auc", 0.0)

            colors = px.colors.qualitative.Plotly
            for idx, (cls_name, curve) in enumerate(class_curves.items()):
                fig.add_trace(
                    go.Scatter(
                        x=curve.get("fpr", []),
                        y=curve.get("tpr", []),
                        mode="lines",
                        line=dict(color=colors[idx % len(colors)], width=2),
                        name=f"Class {cls_name} (OvR)"
                    )
                )

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            title=f"Receiver Operating Characteristic (ROC) — {model_name}",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall / Sensitivity)",
            xaxis=dict(range=[-0.02, 1.02], gridcolor=BORDER_GRAY),
            yaxis=dict(range=[-0.02, 1.05], gridcolor=BORDER_GRAY),
            legend=dict(x=0.55, y=0.1, bgcolor="rgba(255,255,255,0.8)"),
            height=440
        )
        return fig

    @staticmethod
    def plot_feature_importance(
        fi_dict: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 15
    ) -> go.Figure:
        """Horizontal bar chart for top N important features."""
        items = list(fi_dict.items())[:top_n]
        items.reverse()  # Largest on top

        names = [k for k, _ in items]
        values = [v for _, v in items]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                marker_color="#3B82F6",
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
            )
        )

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            title=title,
            xaxis_title="Relative Importance Score",
            yaxis_title="",
            height=max(350, len(items) * 25),
            xaxis=dict(gridcolor=BORDER_GRAY)
        )
        return fig

    @staticmethod
    def plot_pca_projection(
        X: np.ndarray,
        y: np.ndarray,
        class_mapping: Dict[int, str]
    ) -> go.Figure:
        """2D PCA projection scatter plot colored by target class."""
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        var_ratio = pca.explained_variance_ratio_ * 100

        labels = [class_mapping.get(int(val), str(val)) for val in y]

        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Class": labels
        })

        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Class",
            color_discrete_sequence=[PRIMARY_BLUE, "#EF4444", "#10B981", "#F59E0B"],
            title=f"2D PCA Projection (Explained Variance: PC1={var_ratio[0]:.1f}%, PC2={var_ratio[1]:.1f}%)"
        )

        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            xaxis_title=f"Principal Component 1 ({var_ratio[0]:.1f}%)",
            yaxis_title=f"Principal Component 2 ({var_ratio[1]:.1f}%)",
            height=450
        )
        return fig

    @staticmethod
    def plot_target_distribution(dist_df: pd.DataFrame) -> go.Figure:
        """Target class distribution bar chart."""
        fig = go.Figure(
            go.Bar(
                x=dist_df["Class Label"],
                y=dist_df["Count"],
                marker_color=PRIMARY_BLUE,
                text=[f"{cnt} ({pct}%)" for cnt, pct in zip(dist_df["Count"], dist_df["Percentage (%)"])],
                textposition="auto"
            )
        )
        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            title="Target Class Distribution",
            xaxis_title="Class Label",
            yaxis_title="Observation Count",
            height=320,
            yaxis=dict(gridcolor=BORDER_GRAY)
        )
        return fig

    @staticmethod
    def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
        """Correlation heatmap for numerical features."""
        fig = go.Figure(
            go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Pearson r")
            )
        )
        fig.update_layout(
            **PLOT_LAYOUT_DEFAULTS,
            title="Numerical Feature Correlation Heatmap",
            height=550,
            yaxis=dict(autorange="reversed")
        )
        return fig
