"""
Comprehensive evaluation metrics for academic performance prediction.
Calculates ROC-AUC, PR-AUC, F1-scores, Recall, and Confusion Matrices.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Calculate complete classification metrics suite."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "model": model_name,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_high_performing": float(f1_score(y_true, y_pred, pos_label=1)),
        "recall_high_performing": float(recall_score(y_true, y_pred, pos_label=1)),
        "precision_high_performing": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": specificity,
        "f1_low_performing": float(f1_score(y_true, y_pred, pos_label=0)),
        "recall_low_performing": float(recall_score(y_true, y_pred, pos_label=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    logger.info(
        f"[{model_name}] ROC-AUC: {metrics['roc_auc']:.4f} | "
        f"PR-AUC: {metrics['pr_auc']:.4f} | "
        f"F1 (High): {metrics['f1_high_performing']:.4f} | "
        f"Recall (High): {metrics['recall_high_performing']:.4f} | "
        f"Specificity: {metrics['specificity']:.4f} | "
        f"Balanced Acc: {metrics['balanced_accuracy']:.4f}"
    )

    return metrics


def format_metrics_table(metrics_list: List[Dict[str, Any]], output_path: str = "outputs/tables/benchmark_results.csv") -> pd.DataFrame:
    """Format multiple model metrics into comparison table and persist to CSV."""
    df = pd.DataFrame(metrics_list)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved benchmark metrics to {path}")
    return df
