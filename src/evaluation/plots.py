"""
Publication-ready plotting routines for ROC curves, PR curves, Confusion Matrices,
and Real vs TabRep synthetic distributions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Style settings
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11


def plot_roc_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str = "outputs/figures/roc_curves.png",
) -> Path:
    """
    Plot ROC Curves for multiple models.
    models_dict: {model_name: (y_true, y_prob)}
    """
    plt.figure(figsize=(8, 6), dpi=300)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (name, (y_true, y_prob)) in enumerate(models_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    logger.info(f"Saved ROC Curve figure to: {out_file}")
    return out_file


def plot_precision_recall_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str = "outputs/figures/precision_recall_curves.png",
) -> Path:
    """Plot Precision-Recall curves with baseline prevalence."""
    plt.figure(figsize=(8, 6), dpi=300)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    baseline = 0.331  # Argentina PISA class 1 prevalence
    for idx, (name, (y_true, y_prob)) in enumerate(models_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        color = colors[idx % len(colors)]
        plt.plot(recall, precision, color=color, lw=2.2, label=f"{name} (AP = {ap:.3f})")

    plt.axhline(y=baseline, color="gray", linestyle="--", lw=1.5, label=f"No Skill Baseline ({baseline*100:.1f}%)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall (Sensitivity for High Performing)", fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    plt.title("Precision-Recall (PR) Curves for Imbalanced Classification", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    logger.info(f"Saved PR Curve figure to: {out_file}")
    return out_file


def plot_confusion_matrices(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str = "outputs/figures/confusion_matrices.png",
) -> Path:
    """Plot side-by-side normalized confusion matrices."""
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), dpi=300, squeeze=False)

    for idx, (name, (y_true, y_pred)) in enumerate(models_dict.items()):
        ax = axes[0, idx]
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Low (0)", "High (1)"],
            yticklabels=["Low (0)", "High (1)"],
        )
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("Predicted Class")
        if idx == 0:
            ax.set_ylabel("True Class")

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    logger.info(f"Saved Confusion Matrix figure to: {out_file}")
    return out_file


def plot_feature_importances(
    importances: pd.Series,
    top_n: int = 20,
    output_path: str = "outputs/figures/feature_importance.png",
) -> Path:
    """Plot top N feature importances."""
    plt.figure(figsize=(10, 7), dpi=300)
    top_feats = importances.head(top_n).sort_values(ascending=True)
    
    colors = sns.color_palette("viridis", len(top_feats))
    top_feats.plot(kind="barh", color=colors)
    plt.title(f"Top {top_n} Predictive Features for Academic Performance (Argentina PISA 2022)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Importance Score", fontweight="bold")
    plt.ylabel("Feature", fontweight="bold")
    plt.tight_layout()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    logger.info(f"Saved Feature Importance figure to: {out_file}")
    return out_file
