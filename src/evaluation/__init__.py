from .metrics import evaluate_predictions, format_metrics_table
from .plots import plot_roc_curves, plot_precision_recall_curves, plot_confusion_matrices, plot_feature_importances
from .statistical_tests import compare_models_cv
from .explainability import (
    compute_shap_explanations,
    plot_shap_summary,
    plot_shap_global_importance,
    plot_shap_waterfall_cases,
)

__all__ = [
    "evaluate_predictions",
    "format_metrics_table",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_confusion_matrices",
    "plot_feature_importances",
    "compare_models_cv",
    "compute_shap_explanations",
    "plot_shap_summary",
    "plot_shap_global_importance",
    "plot_shap_waterfall_cases",
]

