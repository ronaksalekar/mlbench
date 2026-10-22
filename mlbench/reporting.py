"""
Reporting and export utilities for MLBench.
Generates CSV leaderboards, JSON experiment summaries, and comprehensive HTML academic reports.
"""

from typing import Dict, Any, List, Optional
import json
import pandas as pd
from datetime import datetime


class ReportGenerator:
    """Generates downloadable summaries and comprehensive HTML evaluation reports."""

    @staticmethod
    def generate_leaderboard_df(
        results: Dict[str, Any],
        ranking_metric: str = "F1-Score"
    ) -> pd.DataFrame:
        """
        Construct a clean, ranked pandas DataFrame from experiment results.
        """
        rows = []
        metric_key_map = {
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall",
            "F1-Score": "f1_score",
            "Specificity": "specificity",
            "ROC-AUC": "roc_auc"
        }

        for model_name, res in results.items():
            metrics = res.get("metrics", {})
            cv_mean = res.get("cv_mean")
            cv_std = res.get("cv_std")
            cv_str = f"{cv_mean:.4f} +/- {cv_std:.4f}" if cv_mean is not None and cv_std is not None else "N/A"

            row = {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", 0.0),
                "Precision": metrics.get("precision", 0.0),
                "Recall": metrics.get("recall", 0.0),
                "F1-Score": metrics.get("f1_score", 0.0),
                "Specificity": metrics.get("specificity", 0.0),
                "ROC-AUC": metrics.get("roc_auc") if metrics.get("roc_auc") is not None else 0.0,
                "CV Score (Mean ± Std)": cv_str,
                "Training Time (s)": res.get("training_time_sec", 0.0)
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Sort by selected ranking metric if present
        if ranking_metric in df.columns:
            df = df.sort_values(by=ranking_metric, ascending=False).reset_index(drop=True)
            df.insert(0, "Rank", range(1, len(df) + 1))
        return df

    @staticmethod
    def generate_json_summary(
        experiment_data: Dict[str, Any]
    ) -> str:
        """Serializes experiment configuration and results to JSON (excluding raw objects)."""
        export_dict = {
            "platform": "MLBench - Comparative ML Model Evaluation Platform",
            "generated_at": datetime.now().isoformat(),
            "dataset_info": experiment_data.get("dataset_info", {}),
            "target_col": experiment_data.get("target_col"),
            "class_distribution": experiment_data.get("class_distribution", {}),
            "preprocessing": experiment_data.get("preprocessing_config", {}),
            "ranking_metric": experiment_data.get("ranking_metric"),
            "models_summary": []
        }

        results = experiment_data.get("results", {})
        for model_name, res in results.items():
            export_dict["models_summary"].append({
                "model_name": model_name,
                "hyperparameters": res.get("hyperparameters", {}),
                "training_time_sec": res.get("training_time_sec"),
                "cv_mean": res.get("cv_mean"),
                "cv_std": res.get("cv_std"),
                "metrics": res.get("metrics", {}),
                "feature_importance_top5": dict(list((res.get("feature_importance") or {}).items())[:5])
            })

        return json.dumps(export_dict, indent=2)

    @staticmethod
    def generate_html_report(
        experiment_data: Dict[str, Any]
    ) -> str:
        """
        Creates a self-contained, publication-ready academic HTML evaluation report.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dataset_info = experiment_data.get("dataset_info", {})
        results = experiment_data.get("results", {})
        ranking_metric = experiment_data.get("ranking_metric", "F1-Score")
        lb_df = ReportGenerator.generate_leaderboard_df(results, ranking_metric)

        best_model_name = lb_df.iloc[0]["Model"] if not lb_df.empty else "N/A"
        best_score = lb_df.iloc[0].get(ranking_metric, 0.0) if not lb_df.empty else 0.0

        # Build table rows for leaderboard
        lb_rows_html = ""
        for _, row in lb_df.iterrows():
            is_best = (row["Model"] == best_model_name)
            highlight_cls = "class='best-row'" if is_best else ""
            roc_val = f"{row['ROC-AUC']:.4f}" if row['ROC-AUC'] > 0 else "N/A"
            lb_rows_html += f"""
            <tr {highlight_cls}>
                <td><strong>#{row.get('Rank', '-')}</strong></td>
                <td><strong>{row['Model']}</strong> {'⭐ <em>(Best)</em>' if is_best else ''}</td>
                <td>{row['Accuracy']:.4f}</td>
                <td>{row['Precision']:.4f}</td>
                <td>{row['Recall']:.4f}</td>
                <td>{row['F1-Score']:.4f}</td>
                <td>{row['Specificity']:.4f}</td>
                <td>{roc_val}</td>
                <td>{row['CV Score (Mean ± Std)']}</td>
                <td>{row['Training Time (s)']:.3f}s</td>
            </tr>
            """

        # Model details cards
        model_cards_html = ""
        for model_name, res in results.items():
            metrics = res.get("metrics", {})
            cm = metrics.get("confusion_matrix", [])
            cm_html = "<table class='mini-cm'><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>"
            if len(cm) == 2:
                cm_html += f"<tr><th>Actual 0</th><td>{cm[0][0]} (TN)</td><td>{cm[0][1]} (FP)</td></tr>"
                cm_html += f"<tr><th>Actual 1</th><td>{cm[1][0]} (FN)</td><td>{cm[1][1]} (TP)</td></tr>"
            else:
                for idx, row in enumerate(cm):
                    cm_html += f"<tr><th>Actual {idx}</th>" + "".join([f"<td>{v}</td>" for v in row]) + "</tr>"
            cm_html += "</table>"

            fi_items = list((res.get("feature_importance") or {}).items())[:5]
            fi_html = "<ul>" + "".join([f"<li><code>{k}</code>: {v:.4f}</li>" for k, v in fi_items]) + "</ul>" if fi_items else "<em>Not available for this model type.</em>"

            model_cards_html += f"""
            <div class="model-card">
                <h3>{model_name}</h3>
                <div class="model-grid">
                    <div>
                        <p><strong>Training Time:</strong> {res.get('training_time_sec', 0):.4f} seconds</p>
                        <p><strong>Holdout Accuracy:</strong> {metrics.get('accuracy', 0):.4f}</p>
                        <p><strong>Holdout F1-Score:</strong> {metrics.get('f1_score', 0):.4f}</p>
                        <p><strong>Holdout Specificity:</strong> {metrics.get('specificity', 0):.4f}</p>
                        <p><strong>ROC-AUC:</strong> {metrics.get('roc_auc', 'N/A')}</p>
                        <p><strong>Cross-Validation Score:</strong> {res.get('cv_mean', 'N/A')} ± {res.get('cv_std', 'N/A')}</p>
                    </div>
                    <div>
                        <h4>Confusion Matrix</h4>
                        {cm_html}
                    </div>
                    <div>
                        <h4>Top Predictive Features</h4>
                        {fi_html}
                    </div>
                </div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLBench Experiment Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #1e293b;
            background-color: #f8fafc;
            margin: 0;
            padding: 40px 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            padding: 40px;
        }}
        .header {{
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #1e40af;
            font-size: 28px;
        }}
        .meta-pill {{
            display: inline-block;
            background: #eff6ff;
            color: #1e40af;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-right: 10px;
        }}
        .best-banner {{
            background: #ecfdf5;
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 25px 0;
            color: #065f46;
        }}
        table.benchmark {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }}
        table.benchmark th, table.benchmark td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        table.benchmark th {{
            background: #f1f5f9;
            color: #334155;
            font-weight: 600;
        }}
        tr.best-row {{
            background-color: #f0fdf4;
            font-weight: bold;
        }}
        .model-card {{
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            background: #ffffff;
        }}
        .model-card h3 {{
            margin-top: 0;
            color: #1e3a8a;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        .model-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }}
        table.mini-cm {{
            border-collapse: collapse;
            font-size: 13px;
        }}
        table.mini-cm th, table.mini-cm td {{
            border: 1px solid #cbd5e1;
            padding: 6px 10px;
            text-align: center;
        }}
        table.mini-cm th {{
            background: #f8fafc;
        }}
        .callout-warning {{
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 14px 18px;
            border-radius: 4px;
            margin: 25px 0;
            color: #92400e;
        }}
        .footer {{
            margin-top: 40px;
            border-top: 1px solid #e2e8f0;
            padding-top: 20px;
            font-size: 12px;
            color: #64748b;
            text-align: center;
        }}
        @media print {{
            body {{ background: #fff; padding: 0; }}
            .container {{ box-shadow: none; border: none; padding: 10px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MLBench — Comparative Machine Learning Model Evaluation Report</h1>
            <p>Comprehensive academic benchmark report generated on <strong>{now_str}</strong></p>
            <span class="meta-pill">Dataset: {dataset_info.get('dataset_name', 'Uploaded Dataset')}</span>
            <span class="meta-pill">Rows: {dataset_info.get('rows', 'N/A')}</span>
            <span class="meta-pill">Features: {dataset_info.get('features_count', 'N/A')}</span>
            <span class="meta-pill">Target: {experiment_data.get('target_col', 'N/A')}</span>
            <span class="meta-pill">Ranking Metric: {ranking_metric}</span>
        </div>

        <div class="best-banner">
            <h3 style="margin: 0 0 5px 0;">🏆 Top Performing Algorithm: {best_model_name}</h3>
            <p style="margin: 0;">Achieved highest {ranking_metric} of <strong>{best_score:.4f}</strong> on the shared holdout test set.</p>
        </div>

        <h2>1. Executive Model Leaderboard</h2>
        <p>All models evaluated on identical holdout test split with standardized preprocessing pipeline fitted strictly on training observations.</p>
        <table class="benchmark">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model Name</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Specificity</th>
                    <th>ROC-AUC</th>
                    <th>CV Score (Mean ± Std)</th>
                    <th>Training Time</th>
                </tr>
            </thead>
            <tbody>
                {lb_rows_html}
            </tbody>
        </table>

        <h2>2. Detailed Model Evaluation & Confusion Matrices</h2>
        {model_cards_html}

        <h2>3. Experimental Methodology & Rigor</h2>
        <ul>
            <li><strong>Leakage Prevention:</strong> Preprocessing pipelines (imputers, one-hot encoders, standard scalers) were fitted exclusively on training partitions (<code>X_train</code>) and applied without refitting to test partitions.</li>
            <li><strong>Cross-Validation:</strong> K-Fold Stratified Cross-Validation was performed exclusively on the training partition to estimate out-of-fold generalization stability before final holdout test evaluation.</li>
            <li><strong>Metric Definitions:</strong> Evaluated using exact definitions of Accuracy, Precision (PPV), Recall (Sensitivity), Specificity (True Negative Rate), and F1-score (Harmonic Mean).</li>
        </ul>

        <div class="callout-warning">
            <strong>Educational & Research Disclaimer:</strong><br>
            This software is designed as an educational and academic benchmarking platform. While it includes support for medical and speech classification datasets (such as the UCI Parkinson's voice dataset), it is NOT a certified medical diagnostic device. For biomedical speech datasets containing multiple recordings per subject, grouped cross-validation is strongly recommended to prevent optimistic estimates arising from intra-subject voice correlation.
        </div>

        <div class="footer">
            <p>MLBench Academic Mini-Project Platform • Generated with scikit-learn, XGBoost, and Streamlit</p>
        </div>
    </div>
</body>
</html>
        """
        return html_content
