"""
Model Explainability module using SHAP (SHapley Additive exPlanations).
Provides publication-ready TreeExplainer routines:
- Beeswarm summary plots (showing magnitude and directionality of impact)
- Global feature importance (mean absolute SHAP values)
- Contrastive student profile waterfall plots (low risk vs high risk)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Patch SHAP TreeExplainer for modern XGBoost 3.x base_score format '[5E-1]'
try:
    import shap.explainers._tree as _tree_mod
    _orig_decode_ubjson = _tree_mod.decode_ubjson_buffer

    def _patched_decode_ubjson_buffer(fd):
        jmodel = _orig_decode_ubjson(fd)
        try:
            if "learner" in jmodel and "learner_model_param" in jmodel["learner"]:
                lmp = jmodel["learner"]["learner_model_param"]
                if "base_score" in lmp and isinstance(lmp["base_score"], str):
                    lmp["base_score"] = lmp["base_score"].strip("[]")
        except Exception:
            pass
        return jmodel

    _tree_mod.decode_ubjson_buffer = _patched_decode_ubjson_buffer
except Exception as e:
    logger.debug(f"SHAP TreeExplainer patch notice: {e}")


def compute_shap_explanations(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[List[str]] = None,
) -> Tuple[shap.TreeExplainer, shap.Explanation]:
    """
    Computes SHAP explanations using TreeExplainer for tree-based models (XGBoost, LightGBM, RF).
    
    Args:
        model: Trained tree-based model (e.g. XGBClassifier).
        X: Evaluation feature matrix (pd.DataFrame or np.ndarray).
        feature_names: Optional list of column names if X is np.ndarray.
        
    Returns:
        Tuple of (explainer, shap_values Explanation object).
    """
    if isinstance(X, np.ndarray):
        if feature_names is not None:
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    else:
        X_df = X.copy()
        
    logger.info(f"Initializing SHAP TreeExplainer on {X_df.shape[0]} samples with {X_df.shape[1]} features...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)

    # Extract Class 1 explanation for binary classifiers
    if len(shap_values.shape) == 3 and shap_values.shape[-1] == 2:
        shap_values = shap_values[:, :, 1]

    logger.info(f"SHAP explanations computed successfully (shape: {shap_values.shape}).")
    return explainer, shap_values


def plot_shap_summary(
    shap_values: shap.Explanation,
    max_display: int = 15,
    output_path: str = "outputs/figures/shap_summary_beeswarm.png",
    title: str = "SHAP Beeswarm Plot - Academic Performance Drivers",
) -> Path:
    """
    Plots a SHAP beeswarm plot displaying both feature importance and directional effect.
    High feature values are colored red; low values are colored blue.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8), dpi=300)
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    logger.info(f"Saved SHAP beeswarm plot to {out_file}")
    return out_file


def plot_shap_global_importance(
    shap_values: shap.Explanation,
    max_display: int = 15,
    output_path: str = "outputs/figures/shap_importance_bar.png",
    title: str = "Global Feature Importance (Mean |SHAP Value|)",
) -> Path:
    """
    Plots a standard global feature importance bar chart based on mean absolute SHAP values.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(9, 7), dpi=300)
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    logger.info(f"Saved SHAP importance bar plot to {out_file}")
    return out_file


def plot_shap_waterfall_cases(
    shap_values: shap.Explanation,
    y_true: Union[pd.Series, np.ndarray],
    y_pred_prob: Union[pd.Series, np.ndarray],
    output_dir: str = "outputs/figures",
    max_display: int = 12,
) -> Dict[str, Path]:
    """
    Generates waterfall explanation plots for 4 contrastive student archetypes:
    - 2 High Performers (Class 1, score >= 420.07, high predicted probability)
    - 2 Low Performers (Class 0, score < 420.07, low predicted probability / at academic risk)
    
    Args:
        shap_values: Computed SHAP Explanation object.
        y_true: Ground truth binary labels (0 = Low Performer, 1 = High Performer).
        y_pred_prob: Predicted probabilities for Class 1 (High Performance).
        output_dir: Directory to save figure images.
        max_display: Maximum features to display per waterfall plot.
        
    Returns:
        Dict mapping profile name to saved figure Path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_pred_prob)
    
    # Select representative high and low performing students
    class_1_idxs = np.where(y_true_arr == 1)[0]
    sorted_class_1 = class_1_idxs[np.argsort(-y_prob_arr[class_1_idxs])]
    high_perf_idxs = sorted_class_1[:2] if len(sorted_class_1) >= 2 else class_1_idxs[:2]

    class_0_idxs = np.where(y_true_arr == 0)[0]
    sorted_class_0 = class_0_idxs[np.argsort(y_prob_arr[class_0_idxs])]
    low_perf_idxs = sorted_class_0[:2] if len(sorted_class_0) >= 2 else class_0_idxs[:2]
    
    cases = [
        ("high_performer_case_1", high_perf_idxs[0], "Student Profile: High Academic Performer #1 (Proficiency >= Level 2)"),
        ("high_performer_case_2", high_perf_idxs[1], "Student Profile: High Academic Performer #2 (Proficiency >= Level 2)"),
        ("low_performer_case_1", low_perf_idxs[0], "Student Profile: Low Academic Performer #1 (Academic Risk < Level 2)"),
        ("low_performer_case_2", low_perf_idxs[1], "Student Profile: Low Academic Performer #2 (Academic Risk < Level 2)"),
    ]
    
    saved_paths = {}
    for case_id, sample_idx, case_title in cases:
        out_file = out_dir / f"shap_waterfall_{case_id}.png"
        prob_val = y_prob_arr[sample_idx]
        actual_val = y_true_arr[sample_idx]
        
        plt.figure(figsize=(9, 6), dpi=300)
        shap.plots.waterfall(shap_values[sample_idx], max_display=max_display, show=False)
        plt.title(
            f"{case_title}\n[Actual Class: {actual_val} | P(High Performance) = {prob_val:.3f}]",
            fontsize=11,
            fontweight="bold",
            pad=14,
        )
        plt.tight_layout()
        plt.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close()
        
        saved_paths[case_id] = out_file
        logger.info(f"Saved {case_id} waterfall plot to {out_file}")
        
    return saved_paths
