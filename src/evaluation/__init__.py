from .metrics import evaluate_predictions, format_metrics_table
from .plots import (
    MODEL_CURVE_COLORS,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_feature_importances,
)
from .statistical_tests import (
    nadeau_bengio_ttest,
    compute_paired_cv_inference_table,
    compute_holdout_statistical_tests,
    balance_fold_data,
)
from .tuning import (
    SplitArtifact,
    NestedFoldCacheManager,
    NestedModelTunerSelector,
    resolve_metric_name,
)
from .explainability import (
    compute_shap_explanations,
    plot_shap_summary,
    plot_shap_global_importance,
    plot_shap_waterfall_cases,
)

__all__ = [
    "evaluate_predictions",
    "format_metrics_table",
    "MODEL_CURVE_COLORS",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_confusion_matrices",
    "plot_feature_importances",
    "nadeau_bengio_ttest",
    "compute_paired_cv_inference_table",
    "compute_holdout_statistical_tests",
    "balance_fold_data",
    "SplitArtifact",
    "NestedFoldCacheManager",
    "NestedModelTunerSelector",
    "resolve_metric_name",
    "compute_shap_explanations",
    "plot_shap_summary",
    "plot_shap_global_importance",
    "plot_shap_waterfall_cases",
]
