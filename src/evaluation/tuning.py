"""
Leakage-free Nested Cross-Validation Hyperparameter Optimization and Model Selection.

Methodological Guarantees:
1. Strict Test-Set Isolation:
   All feature selection, imputation, TabRep generative balancing, target encoding,
   hyperparameter optimization, and champion model selection operate strictly inside
   `train.parquet`. The held-out `test.parquet` cohort is never accessed during tuning
   or model selection.

2. Intra-Split Feature Selection & Real-Cohort Target Encoding:
   Every outer and inner training split independently fits:
     FeatureSelector -> NativeImputer -> DatasetBalancer (TabRep-Flow) -> FeatureEncoderScaler(real).
   Fitting `FeatureEncoderScaler` strictly on REAL split training rows prevents synthetic
   `y=1` prior shift (`0.331 -> 0.500`) and rare-category synthetic label leakage.

3. Nested Cross-Validation Separation of Tuning and Model Selection:
   - Inner split(s) inside `D_outer_tr_k` optimize hyperparameters `theta_hat_k` via Optuna TPE
     (with baseline configuration enqueued as Trial 0).
   - Outer split `k` refits `M(theta_hat_k)` on `D_outer_tr_k` and evaluates ONCE on untouched
     `D_outer_val_k`.
   - Model/regime ranking (`auto` or `manual` mode) and Nadeau-Bengio variance-corrected
     paired t-tests rely exclusively on these unbiased outer-fold validation scores.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.evaluation.statistical_tests import balance_fold_data, compute_paired_cv_inference_table
from src.features.cleaner import categorize_pisa_features
from src.features.preprocessor import FeatureEncoderScaler, NativeImputer
from src.features.selector import FeatureSelector
from src.models.classifiers.baseline_comparisons import DEFAULT_XGB_PARAMS, build_xgboost
from src.models.classifiers.random_forest import DEFAULT_RF_PARAMS, build_random_forest
from src.models.classifiers.tabpfn_classifier import (
    build_tabpfn_classifier,
    is_tabpfn_available,
    predict_tabpfn_raw_logits,
    tabpfn_logits_to_probs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METRIC_ALIASES: Dict[str, str] = {
    "f1": "f1_high_performing",
    "f1_pos": "f1_high_performing",
    "f1_class1": "f1_high_performing",
    "f1_high_performing": "f1_high_performing",
    "recall": "recall_high_performing",
    "recall_pos": "recall_high_performing",
    "recall_class1": "recall_high_performing",
    "recall_high_performing": "recall_high_performing",
    "precision": "precision_high_performing",
    "precision_high_performing": "precision_high_performing",
    "balanced_accuracy": "balanced_accuracy",
    "bal_acc": "balanced_accuracy",
    "pr_auc": "pr_auc",
    "average_precision": "pr_auc",
    "roc_auc": "roc_auc",
    "auc": "roc_auc",
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "specificity": "specificity",
}

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "tabpfn": "TabPFNv3",
}

REGIME_DISPLAY_NAMES: Dict[str, str] = {
    "unbalanced": "Unbalanced",
    "balanced": "TabRep Balanced",
}

STAT_METRIC_LABELS: Dict[str, str] = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "f1_high_performing": "F1 (High Performing)",
    "recall_high_performing": "Recall (High Performing)",
    "balanced_accuracy": "Balanced Accuracy",
}


def resolve_metric_name(metric: str) -> str:
    """Normalize CLI or config metric names into canonical metric keys."""
    key = str(metric).strip().lower()
    if key not in METRIC_ALIASES:
        valid = sorted(set(METRIC_ALIASES.values()))
        raise ValueError(f"Unsupported selection metric '{metric}'. Supported metrics: {valid}")
    return METRIC_ALIASES[key]


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute full thresholded and threshold-free validation metrics without logging noise."""
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    pr = np.asarray(y_prob, dtype=np.float64)

    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "roc_auc": float(roc_auc_score(yt, pr)),
        "pr_auc": float(average_precision_score(yt, pr)),
        "accuracy": float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "f1_high_performing": float(f1_score(yt, yp, pos_label=1, zero_division=0)),
        "recall_high_performing": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "precision_high_performing": float(precision_score(yt, yp, pos_label=1, zero_division=0)),
        "specificity": spec,
        "brier_score": float(brier_score_loss(yt, pr)),
    }


@dataclass
class SplitArtifact:
    """Encapsulates leakage-free preprocessed training (unbalanced & balanced) and validation matrices."""

    split_id: str
    X_tr_unbal: pd.DataFrame
    y_tr_unbal: pd.Series
    X_tr_bal: Optional[pd.DataFrame]
    y_tr_bal: Optional[pd.Series]
    X_val: pd.DataFrame
    y_val: pd.Series
    selected_features: List[str]
    n_nominal: int
    n_numerical: int


class NestedFoldCacheManager:
    """
    Constructs and optionally caches leakage-free outer and inner CV split artifacts.
    Guarantees that FeatureSelector, NativeImputer, TabRep-Flow balancing, and real-cohort
    FeatureEncoderScaler are fitted strictly on each split's own training rows.
    """

    def __init__(
        self,
        target_k: int = 100,
        max_missing_ratio: float = 0.50,
        variance_threshold: float = 0.005,
        selection_method: str = "hybrid_importance",
        apply_feature_selection: Optional[bool] = None,
        categorical_strategy: str = "target_encoder",
        scaler_type: str = "robust",
        target_encoder_smoothing: float = 10.0,
        balancer_strategy: str = "tabrep",
        diffusion_config: Optional[Dict[str, Any]] = None,
        metadata_explorer: Optional[Any] = None,
        categorization: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        force: bool = False,
    ):
        self.target_k = target_k
        self.max_missing_ratio = max_missing_ratio
        self.variance_threshold = variance_threshold
        self.selection_method = selection_method
        self.apply_feature_selection = apply_feature_selection
        self.categorical_strategy = categorical_strategy
        self.scaler_type = scaler_type
        self.target_encoder_smoothing = target_encoder_smoothing
        self.balancer_strategy = balancer_strategy
        self.diffusion_config = diffusion_config or {}
        self.metadata_explorer = metadata_explorer
        self.categorization = categorization
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.force = force
        self._memory_cache: Dict[str, SplitArtifact] = {}

    def _fingerprint(self, X: pd.DataFrame, y: pd.Series, split_id: str, need_balanced: bool) -> str:
        payload = {
            "split_id": split_id,
            "shape": list(X.shape),
            "cols_head": list(X.columns[:15]),
            "cols_tail": list(X.columns[-15:]),
            "y_sum": int(y.sum()),
            "target_k": self.target_k,
            "max_missing_ratio": self.max_missing_ratio,
            "variance_threshold": self.variance_threshold,
            "selection_method": self.selection_method,
            "apply_feature_selection": self.apply_feature_selection,
            "categorical_strategy": self.categorical_strategy,
            "scaler_type": self.scaler_type,
            "target_encoder_smoothing": self.target_encoder_smoothing,
            "balancer_strategy": self.balancer_strategy,
            "need_balanced": need_balanced,
            "diffusion_config": self.diffusion_config,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        return f"{split_id}_{digest}"

    def get_or_build_split(
        self,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        split_id: str,
        split_seed: int,
        need_balanced: bool = True,
    ) -> SplitArtifact:
        cache_key = self._fingerprint(X_full, y_full, split_id=split_id, need_balanced=need_balanced)
        if not self.force and cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        cache_file = (self.cache_dir / f"{cache_key}.pkl") if self.cache_dir is not None else None
        if cache_file is not None and cache_file.exists() and not self.force:
            with open(cache_file, "rb") as f:
                artifact: SplitArtifact = pickle.load(f)
            self._memory_cache[cache_key] = artifact
            logger.info(f"Loaded cached split artifact: {cache_file.name}")
            return artifact

        X_tr = X_full.iloc[train_idx].copy()
        y_tr = y_full.iloc[train_idx].copy()
        X_val = X_full.iloc[val_idx].copy()
        y_val = y_full.iloc[val_idx].copy()

        run_selector = (
            self.apply_feature_selection
            if self.apply_feature_selection is not None
            else (X_tr.shape[1] > self.target_k)
        )

        has_meta = (
            self.metadata_explorer is not None
            and getattr(self.metadata_explorer, "dictionary", None)
            and len(self.metadata_explorer.dictionary) > 0
        )

        split_cat = self.categorization
        if run_selector:
            sel = FeatureSelector(
                target_k=min(self.target_k, X_tr.shape[1]),
                max_missing_ratio=self.max_missing_ratio,
                variance_threshold=self.variance_threshold,
                method=self.selection_method,
                random_state=split_seed,
            )
            sel.fit(X_tr, y_tr, metadata_explorer=self.metadata_explorer if has_meta else None)
            X_tr = sel.transform(X_tr)
            X_val = sel.transform(X_val)
            if has_meta:
                split_cat = categorize_pisa_features(X_tr, self.metadata_explorer)
            else:
                split_cat = None
        elif split_cat is None and has_meta:
            try:
                split_cat = categorize_pisa_features(X_tr, self.metadata_explorer)
            except ValueError:
                split_cat = None

        imp = NativeImputer(categorization=split_cat)
        X_tr_imp = imp.fit_transform(X_tr)
        X_val_imp = imp.transform(X_val)

        if need_balanced:
            X_tr_bal_imp, y_tr_bal = balance_fold_data(
                X_tr_imp,
                y_tr,
                strategy=self.balancer_strategy,
                random_state=split_seed,
                categorization=split_cat,
                diffusion_config=self.diffusion_config,
            )
        else:
            X_tr_bal_imp, y_tr_bal = None, None

        # Fit FeatureEncoderScaler strictly on REAL split training rows (X_tr_imp, y_tr)
        enc = FeatureEncoderScaler(
            categorical_strategy=self.categorical_strategy,
            scaler_type=self.scaler_type,
            target_encoder_smoothing=self.target_encoder_smoothing,
            categorization=split_cat,
        )
        X_tr_unbal_enc = enc.fit_transform(X_tr_imp, y_tr)
        X_tr_bal_enc = enc.transform(X_tr_bal_imp) if X_tr_bal_imp is not None else None
        X_val_enc = enc.transform(X_val_imp)

        artifact = SplitArtifact(
            split_id=split_id,
            X_tr_unbal=X_tr_unbal_enc,
            y_tr_unbal=y_tr.reset_index(drop=True),
            X_tr_bal=X_tr_bal_enc,
            y_tr_bal=y_tr_bal.reset_index(drop=True) if y_tr_bal is not None else None,
            X_val=X_val_enc,
            y_val=y_val.reset_index(drop=True),
            selected_features=list(X_tr_imp.columns),
            n_nominal=len(enc.nominal_cols_),
            n_numerical=len(enc.numerical_cols_),
        )

        self._memory_cache[cache_key] = artifact
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(artifact, f)
            logger.info(f"Cached split artifact to: {cache_file.name}")

        return artifact


class NestedModelTunerSelector:
    """
    Supports both Manual (`mode='manual'`) and Automatic (`mode='auto'`) model selection
    with Optuna TPE / Random Search / Fixed Baseline evaluation inside a strict Nested CV protocol.
    """

    def __init__(
        self,
        mode: str = "auto",
        selected_models: Optional[List[str]] = None,
        regime: str = "both",
        primary_metric: str = "f1_high_performing",
        tie_breakers: Optional[List[str]] = None,
        strategy: str = "tpe",
        outer_cv_folds: int = 5,
        inner_cv_folds: int = 1,
        inner_val_ratio: float = 0.20,
        n_trials_by_model: Optional[Dict[str, int]] = None,
        n_trials_override: Optional[int] = None,
        evaluate_tabpfn_cv: bool = False,
        classifier_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        tabpfn_tuning_grid: Optional[Dict[str, List[Any]]] = None,
        random_state: int = 42,
    ):
        if mode not in ("auto", "manual"):
            raise ValueError(f"Selection mode must be 'auto' or 'manual', got '{mode}'.")
        if regime not in ("both", "unbalanced", "balanced"):
            raise ValueError(f"Regime must be 'both', 'unbalanced', or 'balanced', got '{regime}'.")
        if strategy not in ("tpe", "random", "none"):
            raise ValueError(f"Tuning strategy must be 'tpe', 'random', or 'none', got '{strategy}'.")

        self.mode = mode
        self.regime = regime
        self.primary_metric = resolve_metric_name(primary_metric)
        self.tie_breakers = [
            resolve_metric_name(m)
            for m in (tie_breakers or ["balanced_accuracy", "pr_auc", "roc_auc"])
            if resolve_metric_name(m) != self.primary_metric
        ]
        self.strategy = strategy
        self.outer_cv_folds = int(outer_cv_folds)
        self.inner_cv_folds = int(inner_cv_folds)
        self.inner_val_ratio = float(inner_val_ratio)
        self.n_trials_by_model = dict(n_trials_by_model or {"random_forest": 20, "xgboost": 25, "tabpfn": 20})
        self.n_trials_override = n_trials_override
        self.evaluate_tabpfn_cv = bool(evaluate_tabpfn_cv)
        self.classifier_configs = classifier_configs or {}
        self.tabpfn_tuning_grid = tabpfn_tuning_grid or {
            "n_estimators": [1, 2, 4],
            "softmax_temperature": [0.75, 0.85, 0.90, 1.00, 1.10],
            "average_before_softmax": [False, True],
        }
        self.random_state = int(random_state)
        self.selected_models = self._resolve_candidate_models(selected_models)

        # Caching raw TabPFN logits per (split_id, regime, max_n_estimators) to avoid repeated GPU passes
        self._tabpfn_logits_cache: Dict[Tuple[str, str, int], Tuple[Any, np.ndarray]] = {}

    def _resolve_candidate_models(self, requested: Optional[List[str]]) -> List[str]:
        valid = {"random_forest", "xgboost", "tabpfn"}
        if self.mode == "manual":
            if not requested:
                raise ValueError("Manual selection mode requires at least one explicit --model candidate.")
            models = [m.strip().lower() for m in requested]
            unknown = [m for m in models if m not in valid]
            if unknown:
                raise ValueError(f"Unsupported model(s): {unknown}. Valid choices: {sorted(valid)}")
            if "tabpfn" in models and not is_tabpfn_available(require_token=True):
                logger.warning(
                    "TabPFNv3 was requested in manual mode, but tabpfn or TABPFN_TOKEN/HF_TOKEN is unavailable."
                )
                models = [m for m in models if m != "tabpfn"]
            return models

        # Automatic mode
        if requested:
            models = [m.strip().lower() for m in requested]
        else:
            models = ["random_forest", "xgboost"]
            if self.evaluate_tabpfn_cv:
                models.append("tabpfn")

        resolved = []
        for m in models:
            if m not in valid:
                raise ValueError(f"Unsupported model '{m}'. Valid choices: {sorted(valid)}")
            if m == "tabpfn" and not is_tabpfn_available(require_token=True):
                logger.warning(
                    "TabPFNv3 skipped in auto CV selection: package or TABPFN_TOKEN/HF_TOKEN unavailable. "
                    "Continuing cleanly with Random Forest and XGBoost."
                )
                continue
            resolved.append(m)
        return resolved

    def get_regimes(self) -> List[str]:
        if self.regime == "both":
            return ["unbalanced", "balanced"]
        return [self.regime]

    def _get_baseline_params(self, model_name: str) -> Dict[str, Any]:
        cfg = dict(self.classifier_configs.get(model_name, {}))
        if model_name == "random_forest":
            base = {
                "n_estimators": int(cfg.get("n_estimators", DEFAULT_RF_PARAMS["n_estimators"])),
                "max_depth": cfg.get("max_depth", DEFAULT_RF_PARAMS["max_depth"]),
                "min_samples_split": int(cfg.get("min_samples_split", DEFAULT_RF_PARAMS["min_samples_split"])),
                "min_samples_leaf": int(cfg.get("min_samples_leaf", DEFAULT_RF_PARAMS["min_samples_leaf"])),
                "max_features": cfg.get("max_features", DEFAULT_RF_PARAMS["max_features"]),
                "max_samples": float(cfg.get("max_samples", 1.0)) if cfg.get("max_samples") is not None else 1.0,
                "criterion": cfg.get("criterion", "gini"),
            }
            return base
        if model_name == "xgboost":
            return {
                "n_estimators": int(cfg.get("n_estimators", DEFAULT_XGB_PARAMS["n_estimators"])),
                "max_depth": int(cfg.get("max_depth", DEFAULT_XGB_PARAMS["max_depth"])),
                "learning_rate": float(cfg.get("learning_rate", DEFAULT_XGB_PARAMS["learning_rate"])),
                "min_child_weight": int(cfg.get("min_child_weight", 1)),
                "subsample": float(cfg.get("subsample", DEFAULT_XGB_PARAMS["subsample"])),
                "colsample_bytree": float(cfg.get("colsample_bytree", DEFAULT_XGB_PARAMS["colsample_bytree"])),
                "gamma": float(cfg.get("gamma", 0.0)),
                "reg_alpha": float(cfg.get("reg_alpha", 1e-4)),
                "reg_lambda": float(cfg.get("reg_lambda", 1.0)),
            }
        if model_name == "tabpfn":
            return {
                "n_estimators": int(cfg.get("n_estimators", 4)),
                "softmax_temperature": float(cfg.get("softmax_temperature", 0.9)),
                "average_before_softmax": bool(cfg.get("average_before_softmax", False)),
                "balance_probabilities": False,
            }
        raise ValueError(f"Unknown model_name: {model_name}")

    def _suggest_params(self, trial: Any, model_name: str) -> Dict[str, Any]:
        if model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=50),
                "max_depth": trial.suggest_categorical("max_depth", [8, 12, 16, 18, 20, 24, None]),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 16),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.25]),
                "max_samples": trial.suggest_float("max_samples", 0.70, 1.00),
                "criterion": trial.suggest_categorical("criterion", ["gini", "log_loss"]),
            }
        if model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "subsample": trial.suggest_float("subsample", 0.60, 1.00),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.50, 1.00),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            }
        raise ValueError(f"Optuna continuous/categorical sampler not used for model '{model_name}'.")

    @staticmethod
    def _extract_regime_data(split: SplitArtifact, regime: str) -> Tuple[pd.DataFrame, pd.Series]:
        if regime == "unbalanced":
            return split.X_tr_unbal, split.y_tr_unbal
        if regime == "balanced":
            if split.X_tr_bal is None or split.y_tr_bal is None:
                raise ValueError(f"Split '{split.split_id}' was built without balanced training data.")
            return split.X_tr_bal, split.y_tr_bal
        raise ValueError(f"Unknown regime '{regime}'.")

    def evaluate_params_on_split(
        self,
        model_name: str,
        regime: str,
        params: Dict[str, Any],
        split: SplitArtifact,
        seed: int,
    ) -> Dict[str, float]:
        """Fit a candidate configuration on a single split's training set and score on its validation set."""
        X_tr, y_tr = self._extract_regime_data(split, regime)
        X_val, y_val = split.X_val, split.y_val

        if model_name == "random_forest":
            run_params = dict(self.classifier_configs.get("random_forest", {}))
            run_params.update(params)
            run_params["random_state"] = seed
            clf = build_random_forest(**run_params)
            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            return compute_binary_metrics(y_val.values, y_pred, y_prob)

        if model_name == "xgboost":
            run_params = dict(self.classifier_configs.get("xgboost", {}))
            run_params.update(params)
            run_params["random_state"] = seed
            clf = build_xgboost(**run_params)
            clf.fit(X_tr, y_tr)
            y_prob = clf.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            return compute_binary_metrics(y_val.values, y_pred, y_prob)

        if model_name == "tabpfn":
            base_cfg = dict(self.classifier_configs.get("tabpfn", {}))
            grid_ests = [int(k) for k in self.tabpfn_tuning_grid.get("n_estimators", [1, 2, 4])]
            max_est = max([int(params.get("n_estimators", 4)), int(base_cfg.get("n_estimators", 4))] + grid_ests)
            cache_key = (split.split_id, regime, max_est)

            if cache_key not in self._tabpfn_logits_cache:
                clf_Init = dict(base_cfg)
                clf_Init["n_estimators"] = max_est
                clf_Init["random_state"] = seed
                clf = build_tabpfn_classifier(**clf_Init)
                clf.fit(X_tr.values.astype(np.float32), y_tr.values.astype(int))
                raw_logits = predict_tabpfn_raw_logits(clf, X_val.values.astype(np.float32))
                self._tabpfn_logits_cache[cache_key] = (clf, raw_logits)
            else:
                clf, raw_logits = self._tabpfn_logits_cache[cache_key]

            y_prob = tabpfn_logits_to_probs(
                clf,
                raw_logits,
                n_estimators=int(params.get("n_estimators", 4)),
                softmax_temperature=float(params.get("softmax_temperature", 0.9)),
                average_before_softmax=bool(params.get("average_before_softmax", False)),
                balance_probabilities=bool(params.get("balance_probabilities", False)),
            )
            y_pred = (y_prob >= 0.5).astype(int)
            return compute_binary_metrics(y_val.values, y_pred, y_prob)

        raise ValueError(f"Unsupported model_name: {model_name}")

    def _optimize_on_inner_splits(
        self,
        model_name: str,
        regime: str,
        inner_splits: List[SplitArtifact],
        study_seed: int,
    ) -> Tuple[Dict[str, Any], float, int]:
        """Tune hyperparameters strictly on inner split(s), returning (best_params, best_inner_score, n_trials_run)."""
        baseline_params = self._get_baseline_params(model_name)

        n_trials = (
            int(self.n_trials_override)
            if self.n_trials_override is not None
            else int(self.n_trials_by_model.get(model_name, 20))
        )

        if self.strategy == "none" or n_trials <= 0 or len(inner_splits) == 0:
            if not inner_splits:
                return baseline_params, float("nan"), 0
            scores = [
                self.evaluate_params_on_split(model_name, regime, baseline_params, sp, study_seed + idx)[
                    self.primary_metric
                ]
                for idx, sp in enumerate(inner_splits)
            ]
            return baseline_params, float(np.mean(scores)), 1

        if model_name == "tabpfn":
            # Exact grid evaluation over single-pass raw logits
            max_cfg_est = int(self.classifier_configs.get("tabpfn", {}).get("n_estimators", 4))
            est_grid = [int(k) for k in self.tabpfn_tuning_grid.get("n_estimators", [1, 2, 4]) if int(k) <= max_cfg_est]
            if not est_grid:
                est_grid = [max_cfg_est]
            temp_grid = [float(t) for t in self.tabpfn_tuning_grid.get("softmax_temperature", [0.75, 0.85, 0.90, 1.0, 1.1])]
            avg_grid = [bool(a) for a in self.tabpfn_tuning_grid.get("average_before_softmax", [False, True])]

            candidates: List[Dict[str, Any]] = [baseline_params]
            for n_est in est_grid:
                for temp in temp_grid:
                    for avg_b in avg_grid:
                        cand = {
                            "n_estimators": n_est,
                            "softmax_temperature": temp,
                            "average_before_softmax": avg_b,
                            "balance_probabilities": False,
                        }
                        if cand not in candidates:
                            candidates.append(cand)

            if self.n_trials_override is not None and self.n_trials_override < len(candidates):
                candidates = candidates[: max(1, int(self.n_trials_override))]

            best_cand = baseline_params
            best_score = -np.inf
            for cand in candidates:
                cand_scores = [
                    self.evaluate_params_on_split(model_name, regime, cand, sp, study_seed + idx)[self.primary_metric]
                    for idx, sp in enumerate(inner_splits)
                ]
                mean_score = float(np.mean(cand_scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_cand = cand
            return best_cand, best_score, len(candidates)

        import warnings
        import optuna
        from optuna.exceptions import ExperimentalWarning

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if self.strategy == "tpe":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ExperimentalWarning)
                sampler = optuna.samplers.TPESampler(seed=study_seed, multivariate=True)
        else:
            sampler = optuna.samplers.RandomSampler(seed=study_seed)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.enqueue_trial(baseline_params)

        def objective(trial: optuna.Trial) -> float:
            cand = self._suggest_params(trial, model_name)
            fold_vals = [
                self.evaluate_params_on_split(model_name, regime, cand, sp, study_seed + idx)[self.primary_metric]
                for idx, sp in enumerate(inner_splits)
            ]
            return float(np.mean(fold_vals))

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        return best_params, float(study.best_value), len(study.trials)

    def _select_canonical_deploy_params(
        self,
        model_name: str,
        regime: str,
        candidate_param_sets: List[Dict[str, Any]],
        outer_splits: List[SplitArtifact],
    ) -> Tuple[Dict[str, Any], float]:
        """
        Select canonical deployment hyperparameters for the final 100%-train refit by scoring
        the unique inner-fold winners + baseline across all cached outer splits (strictly inside train.parquet).
        Does NOT alter the already-recorded unbiased Nested CV outer validation scores.
        """
        baseline = self._get_baseline_params(model_name)
        if self.strategy == "none" or not candidate_param_sets:
            return baseline, float("nan")

        unique_candidates: List[Dict[str, Any]] = []
        for cand in [baseline] + candidate_param_sets:
            if cand not in unique_candidates:
                unique_candidates.append(cand)

        if len(unique_candidates) == 1:
            return unique_candidates[0], float("nan")

        best_params = unique_candidates[0]
        best_key: Tuple[float, ...] = (-np.inf,)
        for cand in unique_candidates:
            per_fold = [
                self.evaluate_params_on_split(model_name, regime, cand, sp, self.random_state + idx + 1)
                for idx, sp in enumerate(outer_splits)
            ]
            ranking_key = tuple(
                float(np.mean([m[k] for m in per_fold]))
                for k in ([self.primary_metric] + self.tie_breakers)
            )
            if ranking_key > best_key:
                best_key = ranking_key
                best_params = cand

        return best_params, float(best_key[0])

    def run_nested_selection(
        self,
        X_clean: pd.DataFrame,
        y: pd.Series,
        cache_manager: NestedFoldCacheManager,
        output_tables_dir: Optional[str] = "outputs/tables",
    ) -> Dict[str, Any]:
        """
        Execute full Nested CV tuning and unbiased outer-fold model/regime selection on `train.parquet`.
        """
        if not self.selected_models:
            raise RuntimeError("No available candidate models to evaluate.")

        regimes = self.get_regimes()
        need_balanced = "balanced" in regimes
        do_inner_tuning = self.strategy != "none" and (
            self.n_trials_override is None or int(self.n_trials_override) > 0
        )

        outer_skf = StratifiedKFold(
            n_splits=self.outer_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        outer_splits: List[SplitArtifact] = []
        inner_splits_by_outer: List[List[SplitArtifact]] = []

        logger.info(
            f"Building/loading leakage-free Nested CV splits (outer_folds={self.outer_cv_folds}, "
            f"inner_folds={self.inner_cv_folds if do_inner_tuning else 0}, need_balanced={need_balanced})..."
        )

        for outer_idx, (o_tr_idx, o_val_idx) in enumerate(outer_skf.split(X_clean, y), start=1):
            outer_seed = self.random_state + outer_idx
            outer_art = cache_manager.get_or_build_split(
                X_full=X_clean,
                y_full=y,
                train_idx=o_tr_idx,
                val_idx=o_val_idx,
                split_id=f"outer_{outer_idx}",
                split_seed=outer_seed,
                need_balanced=need_balanced,
            )
            outer_splits.append(outer_art)

            fold_inner_list: List[SplitArtifact] = []
            if do_inner_tuning:
                X_outer_tr = X_clean.iloc[o_tr_idx].reset_index(drop=True)
                y_outer_tr = y.iloc[o_tr_idx].reset_index(drop=True)
                inner_base_seed = self.random_state + 100 * outer_idx

                if self.inner_cv_folds <= 1:
                    sss = StratifiedShuffleSplit(
                        n_splits=1,
                        test_size=self.inner_val_ratio,
                        random_state=inner_base_seed,
                    )
                    inner_iter = list(sss.split(X_outer_tr, y_outer_tr))
                else:
                    iskf = StratifiedKFold(
                        n_splits=self.inner_cv_folds,
                        shuffle=True,
                        random_state=inner_base_seed,
                    )
                    inner_iter = list(iskf.split(X_outer_tr, y_outer_tr))

                for inner_idx, (i_tr_rel, i_val_rel) in enumerate(inner_iter, start=1):
                    i_tr_abs = o_tr_idx[i_tr_rel]
                    i_val_abs = o_tr_idx[i_val_rel]
                    inner_art = cache_manager.get_or_build_split(
                        X_full=X_clean,
                        y_full=y,
                        train_idx=i_tr_abs,
                        val_idx=i_val_abs,
                        split_id=f"outer_{outer_idx}_inner_{inner_idx}",
                        split_seed=inner_base_seed + inner_idx,
                        need_balanced=need_balanced,
                    )
                    fold_inner_list.append(inner_art)

            inner_splits_by_outer.append(fold_inner_list)

        tuning_rows: List[Dict[str, Any]] = []
        selection_rows: List[Dict[str, Any]] = []
        best_hyperparams_payload: Dict[str, Any] = {
            "protocol": {
                "mode": self.mode,
                "regime": self.regime,
                "primary_metric": self.primary_metric,
                "tie_breakers": self.tie_breakers,
                "strategy": self.strategy,
                "outer_cv_folds": self.outer_cv_folds,
                "inner_cv_folds": self.inner_cv_folds if do_inner_tuning else 0,
                "inner_val_ratio": self.inner_val_ratio,
                "random_state": self.random_state,
            },
            "candidates": {},
        }

        # Store outer fold metric vectors per (model_name, regime) for Nadeau-Bengio inference
        outer_metric_vectors: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
        canonical_params_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for model_name in self.selected_models:
            best_hyperparams_payload["candidates"][model_name] = {}
            for reg_name in regimes:
                cand_label = f"{MODEL_DISPLAY_NAMES[model_name]} ({REGIME_DISPLAY_NAMES[reg_name]})"
                logger.info(
                    f"=== Nested CV [{self.mode.upper()} | strategy={self.strategy}]: {cand_label} ==="
                )

                fold_best_params: List[Dict[str, Any]] = []
                outer_fold_metrics: List[Dict[str, float]] = []

                for k_idx, outer_sp in enumerate(outer_splits):
                    fold_num = k_idx + 1
                    study_seed = self.random_state + 1000 * fold_num + (17 if reg_name == "balanced" else 0)
                    best_k, best_inner_val, n_trials_run = self._optimize_on_inner_splits(
                        model_name=model_name,
                        regime=reg_name,
                        inner_splits=inner_splits_by_outer[k_idx],
                        study_seed=study_seed,
                    )
                    fold_best_params.append(best_k)

                    # Refit M(theta_hat_k) on full D_outer_tr_k and score ONCE on untouched D_outer_val_k
                    outer_m = self.evaluate_params_on_split(
                        model_name=model_name,
                        regime=reg_name,
                        params=best_k,
                        split=outer_sp,
                        seed=self.random_state + fold_num,
                    )
                    outer_fold_metrics.append(outer_m)

                    tuning_rows.append({
                        "model_key": model_name,
                        "model": MODEL_DISPLAY_NAMES[model_name],
                        "regime": reg_name,
                        "candidate": cand_label,
                        "outer_fold": fold_num,
                        "strategy": self.strategy,
                        "n_trials": n_trials_run,
                        "primary_metric": self.primary_metric,
                        "inner_best_score": best_inner_val,
                        "outer_val_primary": outer_m[self.primary_metric],
                        "outer_val_f1_high": outer_m["f1_high_performing"],
                        "outer_val_recall_high": outer_m["recall_high_performing"],
                        "outer_val_balanced_accuracy": outer_m["balanced_accuracy"],
                        "outer_val_pr_auc": outer_m["pr_auc"],
                        "outer_val_roc_auc": outer_m["roc_auc"],
                        "best_params_json": json.dumps(best_k, sort_keys=True),
                    })

                canonical_params, consensus_score = self._select_canonical_deploy_params(
                    model_name=model_name,
                    regime=reg_name,
                    candidate_param_sets=fold_best_params,
                    outer_splits=outer_splits,
                )
                canonical_params_map[(model_name, reg_name)] = canonical_params

                metric_keys = list(outer_fold_metrics[0].keys())
                vec_dict = {mk: [fm[mk] for fm in outer_fold_metrics] for mk in metric_keys}
                outer_metric_vectors[(model_name, reg_name)] = vec_dict

                ddof = 1 if len(outer_fold_metrics) > 1 else 0
                summary_entry: Dict[str, Any] = {
                    "model_key": model_name,
                    "model": MODEL_DISPLAY_NAMES[model_name],
                    "regime": reg_name,
                    "candidate": cand_label,
                    "selection_mode": self.mode,
                    "tuning_strategy": self.strategy,
                    "primary_metric": self.primary_metric,
                    "outer_folds": len(outer_fold_metrics),
                }
                for mk in metric_keys:
                    vals = np.asarray(vec_dict[mk], dtype=np.float64)
                    summary_entry[f"mean_{mk}"] = float(np.mean(vals))
                    summary_entry[f"std_{mk}"] = float(np.std(vals, ddof=ddof))
                summary_entry["canonical_deploy_params_json"] = json.dumps(canonical_params, sort_keys=True)
                selection_rows.append(summary_entry)

                best_hyperparams_payload["candidates"][model_name][reg_name] = {
                    "canonical_deploy_params": canonical_params,
                    "consensus_cv_primary_score": consensus_score,
                    "nested_outer_mean_primary": summary_entry[f"mean_{self.primary_metric}"],
                    "nested_outer_std_primary": summary_entry[f"std_{self.primary_metric}"],
                    "per_outer_fold_inner_best_params": fold_best_params,
                }

                logger.info(
                    f"[{cand_label}] Unbiased Nested CV {self.primary_metric}: "
                    f"{summary_entry[f'mean_{self.primary_metric}']:.4f} +/- "
                    f"{summary_entry[f'std_{self.primary_metric}']:.4f} | "
                    f"Deploy params: {canonical_params}"
                )

        # Rank all candidates strictly by outer-fold Nested CV primary_metric + tie_breakers
        sort_cols = [f"mean_{m}" for m in ([self.primary_metric] + self.tie_breakers)]
        selection_df = pd.DataFrame(selection_rows).sort_values(by=sort_cols, ascending=False).reset_index(drop=True)
        selection_df["rank"] = np.arange(1, len(selection_df) + 1)
        selection_df["is_selected_champion"] = selection_df["rank"] == 1

        champion_row = selection_df.iloc[0].to_dict()
        best_hyperparams_payload["champion"] = {
            "model_key": champion_row["model_key"],
            "model": champion_row["model"],
            "regime": champion_row["regime"],
            "candidate": champion_row["candidate"],
            "primary_metric": self.primary_metric,
            "nested_outer_mean_primary": champion_row[f"mean_{self.primary_metric}"],
            "nested_outer_std_primary": champion_row[f"std_{self.primary_metric}"],
            "canonical_deploy_params": canonical_params_map[(champion_row["model_key"], champion_row["regime"])],
        }

        # Paired Nadeau-Bengio CV inference whenever both regimes were evaluated
        stat_frames: List[pd.DataFrame] = []
        if "unbalanced" in regimes and "balanced" in regimes and len(outer_splits) >= 2:
            n_tr_fold = len(outer_splits[-1].y_tr_unbal)
            n_val_fold = len(outer_splits[-1].y_val)
            for model_name in self.selected_models:
                u_vec = outer_metric_vectors[(model_name, "unbalanced")]
                b_vec = outer_metric_vectors[(model_name, "balanced")]
                u_named = {label: u_vec[key] for key, label in STAT_METRIC_LABELS.items() if key in u_vec}
                b_named = {label: b_vec[key] for key, label in STAT_METRIC_LABELS.items() if key in b_vec}
                df_m = compute_paired_cv_inference_table(
                    model_display_name=MODEL_DISPLAY_NAMES[model_name],
                    unbal_scores=u_named,
                    bal_scores=b_named,
                    n_train=n_tr_fold,
                    n_val=n_val_fold,
                    output_path=None,
                )
                stat_frames.append(df_m)

        stat_df = pd.concat(stat_frames, ignore_index=True) if stat_frames else pd.DataFrame()
        tuning_df = pd.DataFrame(tuning_rows)

        if output_tables_dir is not None:
            out_dir = Path(output_tables_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            tuning_df.to_csv(out_dir / "hyperparameter_tuning_summary.csv", index=False)
            selection_df.to_csv(out_dir / "cv_model_selection_results.csv", index=False)
            with open(out_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
                json.dump(best_hyperparams_payload, f, indent=2, default=str)
            if not stat_df.empty:
                stat_df.to_csv(out_dir / "statistical_significance.csv", index=False)
            logger.info(
                f"Persisted Nested CV artifacts to {out_dir} | Champion: {champion_row['candidate']} "
                f"({self.primary_metric}={champion_row[f'mean_{self.primary_metric}']:.4f})"
            )

        return {
            "champion": best_hyperparams_payload["champion"],
            "canonical_params_map": canonical_params_map,
            "selection_df": selection_df,
            "tuning_df": tuning_df,
            "statistical_df": stat_df,
            "best_hyperparameters": best_hyperparams_payload,
        }

