"""
Evaluation metrics and statistical computations for MLBench.
Implements exact mathematical definitions for Accuracy, Precision, Recall,
Specificity, F1-Score, ROC-AUC, and Confusion Matrix metrics.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    classification_report
)


class Evaluator:
    """Evaluates classification models using consistent holdout test metrics."""

    @staticmethod
    def calculate_metrics(
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_prob: Optional[np.ndarray] = None,
        classes: Optional[List[str]] = None,
        pos_label: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate full suite of classification evaluation metrics.
        Handles both binary and multiclass tasks.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)
        is_binary = (n_classes == 2)

        # 1. Accuracy: (TP + TN) / Total
        acc = float(accuracy_score(y_true, y_pred))

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # 3. Precision, Recall, F1, Specificity
        tp, tn, fp, fn = 0, 0, 0, 0
        specificity = 0.0

        if is_binary:
            # Binary classification metrics using pos_label
            # Find confusion matrix entries
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                # Specificity = TN / (TN + FP)
                denom_spec = tn + fp
                specificity = float(tn / denom_spec) if denom_spec > 0 else 0.0
            else:
                specificity = 0.0

            prec = float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            rec = float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
            avg_method = "binary"

        else:
            # Multiclass metrics: use weighted and macro
            prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            avg_method = "weighted"

            # Multiclass specificity (macro average of per-class specificities)
            per_class_spec = []
            for i in range(len(cm)):
                # Class i is positive, all others negative
                c_tp = cm[i, i]
                c_fn = np.sum(cm[i, :]) - c_tp
                c_fp = np.sum(cm[:, i]) - c_tp
                c_tn = np.sum(cm) - (c_tp + c_fn + c_fp)
                denom = c_tn + c_fp
                per_class_spec.append(c_tn / denom if denom > 0 else 0.0)
            specificity = float(np.mean(per_class_spec))

        # 4. ROC-AUC calculation
        roc_auc = None
        roc_data = None
        roc_error_reason = None

        if y_prob is not None:
            try:
                if is_binary:
                    # Probability for positive class
                    if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                        prob_pos = y_prob[:, 1]
                    else:
                        prob_pos = y_prob.ravel()

                    roc_auc = float(roc_auc_score(y_true, prob_pos))
                    fpr, tpr, thresholds = roc_curve(y_true, prob_pos, pos_label=pos_label)
                    roc_data = {
                        "type": "binary",
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist(),
                        "auc": roc_auc
                    }
                else:
                    # Multiclass ROC-AUC (One-vs-Rest)
                    if y_prob.ndim == 2 and y_prob.shape[1] == n_classes:
                        roc_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
                        # Generate OvR curve for each class
                        class_curves = {}
                        for idx, cls_val in enumerate(unique_classes):
                            bin_y = (y_true == cls_val).astype(int)
                            if len(np.unique(bin_y)) > 1:
                                c_fpr, c_tpr, _ = roc_curve(bin_y, y_prob[:, idx])
                                class_curves[str(cls_val)] = {
                                    "fpr": c_fpr.tolist(),
                                    "tpr": c_tpr.tolist()
                                }
                        roc_data = {
                            "type": "multiclass_ovr",
                            "class_curves": class_curves,
                            "auc": roc_auc
                        }
                    else:
                        roc_error_reason = "Probability matrix dimension does not match number of classes."
            except Exception as e:
                roc_auc = None
                roc_error_reason = str(e)
        else:
            roc_error_reason = "Model does not produce probability estimates."

        # Detailed classification report dictionary
        report_dict = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "roc_error_reason": roc_error_reason,
            "roc_data": roc_data,
            "confusion_matrix": cm.tolist(),
            "binary_counts": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn)
            } if is_binary else None,
            "is_binary": is_binary,
            "average_method": avg_method,
            "classification_report": report_dict
        }
