"""
Master pipeline orchestrator for the Academic Prediction Diffusion Project.
Supports execution of individual steps or complete end-to-end runs, with both
automatic (`--mode auto`) and manual (`--mode manual --model ...`) model selection
under a leakage-free Nested Cross-Validation protocol.
All steps are idempotent unless `force=True` (or explicit CLI selection overrides) are passed.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.data.converter import convert_sas_to_parquet
from src.data.downloader import ensure_raw_dataset
from src.data.loader import prepare_train_test_data
from src.evaluation.explainability import (
    compute_shap_explanations,
    plot_shap_global_importance,
    plot_shap_summary,
    plot_shap_waterfall_cases,
)
from src.evaluation.metrics import evaluate_predictions, format_metrics_table
from src.evaluation.plots import (
    plot_confusion_matrices,
    plot_feature_importances,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.evaluation.statistical_tests import compute_holdout_statistical_tests
from src.evaluation.tuning import (
    MODEL_DISPLAY_NAMES,
    REGIME_DISPLAY_NAMES,
    NestedFoldCacheManager,
    NestedModelTunerSelector,
)
from src.features.cleaner import categorize_pisa_features, clean_pisa_survey_codes
from src.features.preprocessor import FeatureEncoderScaler, NativeImputer
from src.features.selector import FeatureSelector
from src.models.classifiers.baseline_comparisons import train_xgboost
from src.models.classifiers.random_forest import train_random_forest
from src.models.classifiers.tabpfn_classifier import is_tabpfn_available, train_tabpfn
from src.models.diffusion.balancer import DatasetBalancer
from src.models.diffusion.tabrep_adapter import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MasterPipeline:
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        features_path: str = "configs/features.yaml",
        models_path: str = "configs/models.yaml",
    ):
        self.config = self._load_yaml(config_path)
        self.feat_cfg = self._load_yaml(features_path)
        self.model_cfg = self._load_yaml(models_path)

        from src.data.metadata_explorer import PISAMetadataExplorer

        cfg_d = self.config["data"]
        sas_data_path = str(Path(cfg_d["raw_dir"]) / cfg_d["sas_filename"])
        sas_format_path = str(Path(cfg_d["raw_dir"]) / cfg_d.get("sas_format_filename", "CY08MSP_STU_QQQ.FORMAT.SAS"))
        cache_path = cfg_d.get("metadata_dict_path", "data/interim/pisa_metadata_dictionary.json")
        self.explorer = PISAMetadataExplorer(
            sas_data_path=sas_data_path,
            sas_format_path=sas_format_path,
            cache_path=cache_path,
        )

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run_download(self) -> Path:
        """Step 1: Ensure raw dataset and format definitions are present locally."""
        cfg = self.config["data"]
        return ensure_raw_dataset(
            raw_dir=cfg["raw_dir"],
            sas_filename=cfg["sas_filename"],
            sas_format_filename=cfg.get("sas_format_filename", "CY08MSP_STU_QQQ.FORMAT.SAS"),
            zip_filename=cfg["zip_filename"],
            download_url=cfg["download_url"],
        )

    def run_convert(self, force: bool = False) -> Path:
        """Step 2: Convert SAS7BDAT to compressed Parquet."""
        cfg = self.config["data"]
        sas_path = Path(cfg["raw_dir"]) / cfg["sas_filename"]
        return convert_sas_to_parquet(
            sas_path=str(sas_path),
            parquet_path=cfg["interim_parquet"],
            force=force,
        )

    def run_metadata_dictionary(self, force: bool = False) -> Dict[str, Any]:
        """Step 2b: Build and cache authoritative OECD metadata dictionary."""
        dictionary = self.explorer.build_dictionary(force_reload=force)
        logger.info(f"Metadata dictionary verified: {len(dictionary)} variables in {self.explorer.cache_path}")
        return dictionary

    def run_prepare(self, force: bool = False):
        """Step 3: Filter Argentina, define binary target, stratified split."""
        cfg_d = self.config["data"]
        cfg_s = self.config["scope"]
        cfg_sp = self.config["split"]
        return prepare_train_test_data(
            interim_parquet=cfg_d["interim_parquet"],
            train_out=cfg_d["train_parquet"],
            test_out=cfg_d["test_parquet"],
            country_code=cfg_s["country_code"],
            target_col=cfg_s["target_column"],
            threshold=cfg_s["threshold"],
            test_size=cfg_sp["test_size"],
            random_state=cfg_sp["random_state"],
            stratify=bool(cfg_sp.get("stratify", True)),
            force=force,
            metadata_explorer=self.explorer,
        )

    def run_feature_selection(self, force: bool = False):
        """
        Step 4: Fit top-k feature selection and native imputation on the full training split
        for final holdout refit and TabRep synthesis (`train_selected.parquet` / `test_selected.parquet`).
        Note: Cross-validation inside `run_benchmark` re-runs `FeatureSelector` independently
        within each CV fold on `train.parquet` to prevent fold-level feature-selection leakage.
        """
        processed_dir = Path(self.config["data"]["processed_dir"])
        train_sel_path = processed_dir / "train_selected.parquet"
        test_sel_path = processed_dir / "test_selected.parquet"
        scores_path = Path(self.config["outputs"]["tables_dir"]) / "feature_selection_scores.csv"

        if not force and train_sel_path.exists() and test_sel_path.exists():
            logger.info(f"Cached selected datasets found in {processed_dir}. Skipping (use --force to recompute).")
            X_train_proc = pd.read_parquet(train_sel_path)
            X_test_proc = pd.read_parquet(test_sel_path)
            feat_cols = [c for c in X_train_proc.columns if c != "target"]
            self.cat_registry_ = categorize_pisa_features(X_train_proc[feat_cols], self.explorer)
            return X_train_proc, X_test_proc, feat_cols

        train_df = pd.read_parquet(self.config["data"]["train_parquet"])
        test_df = pd.read_parquet(self.config["data"]["test_parquet"])

        target_col = "target"
        X_train_raw = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test_raw = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        sel_cfg = self.feat_cfg["selection"]
        prep_cfg = self.feat_cfg["preprocessing"]

        if prep_cfg.get("clean_survey_codes", True):
            logger.info("Applying metadata-driven PISA survey code cleaning before feature selection...")
            X_train_raw = clean_pisa_survey_codes(X_train_raw, self.explorer)
            X_test_raw = clean_pisa_survey_codes(X_test_raw, self.explorer)

        selector = FeatureSelector(
            target_k=sel_cfg["target_k"],
            max_missing_ratio=sel_cfg["max_missing_ratio"],
            variance_threshold=sel_cfg["variance_threshold"],
            method=sel_cfg.get("method", "hybrid_importance"),
            random_state=self.config["execution"]["seed"],
        )
        selector.fit(X_train_raw, y_train, metadata_explorer=self.explorer)
        selector.save_scores(str(scores_path))

        X_train_sel = selector.transform(X_train_raw)
        X_test_sel = selector.transform(X_test_raw)

        cat_registry = categorize_pisa_features(X_train_sel, self.explorer)
        self.cat_registry_ = cat_registry

        imputer = NativeImputer(
            categorization=cat_registry,
            numerical_strategy=prep_cfg.get("numerical_imputer", "median"),
            nominal_missing_token=prep_cfg.get("nominal_missing_token", "MISSING"),
        )
        X_train_proc = imputer.fit_transform(X_train_sel)
        X_test_proc = imputer.transform(X_test_sel)

        processed_dir.mkdir(parents=True, exist_ok=True)
        X_train_proc = X_train_proc.copy()
        X_train_proc["target"] = y_train.values
        X_test_proc = X_test_proc.copy()
        X_test_proc["target"] = y_test.values

        X_train_proc.to_parquet(train_sel_path, index=False)
        X_test_proc.to_parquet(test_sel_path, index=False)
        logger.info(f"Saved native processed train/test datasets to {processed_dir}")

        return X_train_proc, X_test_proc, selector.selected_features_

    def run_diffusion_balancing(self, force: bool = False):
        """Step 5: Train Official TabRep-Flow model on minority class to balance training data 1:1."""
        processed_dir = Path(self.config["data"]["processed_dir"])
        train_path = processed_dir / "train_selected.parquet"
        if not train_path.exists():
            self.run_feature_selection(force=force)

        train_df = pd.read_parquet(train_path)
        diff_cfg = self.model_cfg["diffusion"]
        net_cfg = diff_cfg.get("network", {})
        train_cfg = diff_cfg.get("training", {})

        tables_dir = Path(self.config["outputs"]["tables_dir"])
        figures_dir = Path(self.config["outputs"]["figures_dir"])
        use_gpu = bool(self.config["execution"].get("use_gpu", True))
        resolved_device = "auto" if use_gpu else "cpu"

        balancer = DatasetBalancer(
            strategy="tabrep",
            epochs=train_cfg.get("epochs", 600),
            batch_size=train_cfg.get("batch_size", 512),
            lr=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 5e-4),
            timesteps=train_cfg.get("timesteps", 50),
            hidden_dims=net_cfg.get("hidden_dims", [1024, 2048, 2048, 1024]),
            dim_t=net_cfg.get("dim_t", 1024),
            dropout=net_cfg.get("dropout", 0.0),
            activation=net_cfg.get("activation", "relu"),
            architecture=net_cfg.get("architecture", "tabrep_mlp"),
            snap_discrete_levels=diff_cfg.get("snap_discrete_levels", True),
            lr_scheduler_factor=train_cfg.get("lr_scheduler_factor", 0.9),
            lr_scheduler_patience=train_cfg.get("lr_scheduler_patience", 30),
            solver=train_cfg.get("solver", "heun"),
            device=resolved_device,
            random_state=self.config["execution"]["seed"],
        )

        reg = getattr(self, "cat_registry_", None)
        if reg is None:
            feature_cols = [c for c in train_df.columns if c != "target"]
            reg = categorize_pisa_features(train_df[feature_cols], self.explorer)
            self.cat_registry_ = reg

        balanced_df = balancer.balance_training_data(
            train_df=train_df,
            target_col="target",
            output_path=self.config["data"]["train_balanced_parquet"],
            force=force,
            categorization=reg,
            save_diagnostics=True,
            diagnostics_table_path=str(tables_dir / "tabrep_distribution_metrics.csv"),
            diagnostics_fig_path=str(figures_dir / "tabrep_distribution_audit.png"),
        )
        return balanced_df

    def run_benchmark(
        self,
        force: bool = False,
        mode: Optional[str] = None,
        selected_models: Optional[List[str]] = None,
        regime: Optional[str] = None,
        primary_metric: Optional[str] = None,
        n_trials_override: Optional[int] = None,
        no_tune: bool = False,
    ) -> pd.DataFrame:
        """
        Step 6: Execute leakage-free Nested CV tuning & model selection on `train.parquet`,
        refit models with CV-selected hyperparameters on 100% training data (using real-cohort
        target encoding), evaluate ONCE on vaulted `test.parquet`, and generate statistical
        inference tables + SHAP interpretability figures.
        """
        table_dir = Path(self.config["outputs"]["tables_dir"])
        fig_dir = Path(self.config["outputs"]["figures_dir"])
        bench_csv = table_dir / "benchmark_results.csv"
        cv_sel_csv = table_dir / "cv_model_selection_results.csv"

        has_cli_override = any([
            mode is not None,
            selected_models is not None,
            regime is not None,
            primary_metric is not None,
            n_trials_override is not None,
            no_tune,
        ])
        if not force and not has_cli_override and bench_csv.exists() and cv_sel_csv.exists():
            logger.info(f"Cached benchmark tables found in {table_dir}. Skipping (use --force to recompute).")
            return pd.read_csv(bench_csv)

        sel_cfg = self.model_cfg.get("selection", {})
        tune_cfg = self.model_cfg.get("tuning", {})
        val_cfg = self.model_cfg.get("validation", {})
        feat_sel_cfg = self.feat_cfg["selection"]
        enc_cfg = self.feat_cfg["encoding"]
        prep_cfg = self.feat_cfg["preprocessing"]

        resolved_mode = mode or sel_cfg.get("mode", "auto")
        resolved_regime = regime or sel_cfg.get("regime", "both")
        resolved_metric = primary_metric or sel_cfg.get("primary_metric", "f1_high_performing")
        resolved_strategy = "none" if no_tune else tune_cfg.get("strategy", "tpe")
        seed = int(self.config["execution"]["seed"])
        use_gpu = bool(self.config["execution"].get("use_gpu", True))

        classifier_cfgs = {k: dict(v) for k, v in self.model_cfg["classifiers"].items()}
        if not use_gpu and "tabpfn" in classifier_cfgs:
            classifier_cfgs["tabpfn"]["device"] = "cpu"

        # ------------------------------------------------------------------
        # Stage 6A: Leakage-Free Nested CV on full `train.parquet` ONLY
        # ------------------------------------------------------------------
        train_raw_df = pd.read_parquet(self.config["data"]["train_parquet"])
        y_train_full = train_raw_df["target"].astype(int)
        X_train_full = train_raw_df.drop(columns=["target"])
        if prep_cfg.get("clean_survey_codes", True):
            X_train_full = clean_pisa_survey_codes(X_train_full, self.explorer)

        cache_manager = NestedFoldCacheManager(
            target_k=feat_sel_cfg["target_k"],
            max_missing_ratio=feat_sel_cfg["max_missing_ratio"],
            variance_threshold=feat_sel_cfg["variance_threshold"],
            selection_method=feat_sel_cfg.get("method", "hybrid_importance"),
            apply_feature_selection=True,
            categorical_strategy=enc_cfg["categorical_strategy"],
            scaler_type=prep_cfg["scaler"],
            target_encoder_smoothing=enc_cfg["target_encoder_smoothing"],
            numerical_imputer=prep_cfg.get("numerical_imputer", "median"),
            nominal_missing_token=prep_cfg.get("nominal_missing_token", "MISSING"),
            balancer_strategy="tabrep",
            diffusion_config=self.model_cfg["diffusion"],
            metadata_explorer=self.explorer,
            cache_dir=self.config["data"].get("cv_cache_dir", "data/processed/cv_cache"),
            force=force,
        )

        tuner = NestedModelTunerSelector(
            mode=resolved_mode,
            selected_models=selected_models,
            regime=resolved_regime,
            primary_metric=resolved_metric,
            tie_breakers=sel_cfg.get("tie_breakers", ["balanced_accuracy", "pr_auc", "roc_auc"]),
            strategy=resolved_strategy,
            outer_cv_folds=tune_cfg.get("outer_cv_folds", val_cfg.get("cv_folds", 5)),
            inner_cv_folds=tune_cfg.get("inner_cv_folds", 1),
            inner_val_ratio=tune_cfg.get("inner_val_ratio", 0.20),
            n_trials_by_model=tune_cfg.get("n_trials", {"random_forest": 20, "xgboost": 25, "tabpfn": 30}),
            n_trials_override=n_trials_override,
            evaluate_tabpfn_cv=val_cfg.get("evaluate_tabpfn_cv", False),
            classifier_configs=classifier_cfgs,
            tabpfn_tuning_grid=tune_cfg.get("tabpfn_grid"),
            random_state=seed,
        )

        if not tuner.selected_models:
            logger.warning(
                "No runnable models remained after checking package/token availability. "
                "Verify TABPFN_TOKEN / HF_TOKEN in .env or choose --model random_forest / xgboost."
            )
            return pd.DataFrame()

        nested_out = tuner.run_nested_selection(
            X_clean=X_train_full,
            y=y_train_full,
            cache_manager=cache_manager,
            output_tables_dir=str(table_dir),
        )
        canonical_params_map = nested_out["canonical_params_map"]
        champion_info = nested_out["champion"]
        selection_df = nested_out["selection_df"]

        # ------------------------------------------------------------------
        # Stage 6B: Final 100%-Train Refit & Single Vaulted Test Evaluation
        # ------------------------------------------------------------------
        processed_dir = Path(self.config["data"]["processed_dir"])
        train_sel_path = processed_dir / "train_selected.parquet"
        test_sel_path = processed_dir / "test_selected.parquet"
        balanced_path = Path(self.config["data"]["train_balanced_parquet"])

        if not train_sel_path.exists() or not test_sel_path.exists():
            self.run_feature_selection(force=force)

        train_df = pd.read_parquet(train_sel_path)
        test_df = pd.read_parquet(test_sel_path)

        active_regimes = tuner.get_regimes()
        if "balanced" in active_regimes:
            if not balanced_path.exists():
                self.run_diffusion_balancing(force=force)
            balanced_df = pd.read_parquet(balanced_path)
        else:
            balanced_df = None

        target_col = "target"
        X_train_unbal = train_df.drop(columns=[target_col])
        y_train_unbal = train_df[target_col].astype(int)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].astype(int)

        if balanced_df is not None:
            drop_cols = [c for c in [target_col, "is_synthetic"] if c in balanced_df.columns]
            X_train_bal = balanced_df.drop(columns=drop_cols)
            y_train_bal = balanced_df[target_col].astype(int)
            feature_order = [c for c in X_train_unbal.columns if c in X_train_bal.columns]
            X_train_bal = X_train_bal[feature_order]
        else:
            X_train_bal, y_train_bal = None, None
            feature_order = list(X_train_unbal.columns)

        X_train_unbal = X_train_unbal[feature_order]
        X_test = X_test[feature_order]

        cat_reg = getattr(self, "cat_registry_", None)
        if cat_reg is None:
            cat_reg = categorize_pisa_features(X_train_unbal, self.explorer)
            self.cat_registry_ = cat_reg

        # Fit FeatureEncoderScaler on observed training rows (X_train_unbal, y_train_unbal)
        encoder_real = FeatureEncoderScaler(
            categorical_strategy=enc_cfg["categorical_strategy"],
            scaler_type=prep_cfg["scaler"],
            target_encoder_smoothing=enc_cfg["target_encoder_smoothing"],
            categorization=cat_reg,
        )
        X_train_unbal_enc = encoder_real.fit_transform(X_train_unbal, y_train_unbal)
        X_train_bal_enc = encoder_real.transform(X_train_bal) if X_train_bal is not None else None
        X_test_enc = encoder_real.transform(X_test)

        set_seed(seed)

        # Determine holdout model list: in auto mode, also benchmark zero-shot TabPFNv3 on holdout if authenticated
        holdout_models = list(tuner.selected_models)
        if resolved_mode == "auto" and "tabpfn" not in holdout_models:
            if is_tabpfn_available(require_token=True):
                holdout_models.append("tabpfn")
            else:
                logger.warning(
                    "TabPFNv3 holdout evaluation skipped (TABPFN_TOKEN / HF_TOKEN not configured). "
                    "Executing benchmark cleanly with Random Forest and XGBoost."
                )

        results: List[Dict[str, Any]] = []
        models_roc: Dict[str, Any] = {}
        models_pr: Dict[str, Any] = {}
        models_cm: Dict[str, Any] = {}
        trained_objects: Dict[tuple, Any] = {}
        preds_store: Dict[tuple, tuple] = {}
        rf_importances: Optional[pd.Series] = None

        for m_key in holdout_models:
            base_cfg = dict(classifier_cfgs.get(m_key, {}))
            for reg_key in active_regimes:
                X_tr_fit = X_train_unbal_enc if reg_key == "unbalanced" else X_train_bal_enc
                y_tr_fit = y_train_unbal if reg_key == "unbalanced" else y_train_bal
                tuned_params = canonical_params_map.get((m_key, reg_key), {})
                fit_params = dict(base_cfg)
                fit_params.update(tuned_params)
                fit_params["random_state"] = seed

                disp_name = f"{MODEL_DISPLAY_NAMES[m_key]} ({REGIME_DISPLAY_NAMES[reg_key]})"
                logger.info(f"--- Holdout Final Refit & Vaulted Test Eval: {disp_name} ---")

                if m_key == "random_forest":
                    clf_obj, y_pred, y_prob, imp = train_random_forest(
                        X_train=X_tr_fit,
                        y_train=y_tr_fit,
                        X_test=X_test_enc,
                        **fit_params,
                    )
                    if rf_importances is None:
                        rf_importances = imp
                elif m_key == "xgboost":
                    clf_obj, y_pred, y_prob = train_xgboost(
                        X_train=X_tr_fit,
                        y_train=y_tr_fit,
                        X_test=X_test_enc,
                        **fit_params,
                    )
                elif m_key == "tabpfn":
                    clf_obj, y_pred, y_prob = train_tabpfn(
                        X_train=X_tr_fit,
                        y_train=y_tr_fit,
                        X_test=X_test_enc,
                        **fit_params,
                    )
                else:
                    continue

                if clf_obj is None or y_pred is None or y_prob is None:
                    continue

                is_champ = (
                    champion_info["model_key"] == m_key and champion_info["regime"] == reg_key
                )
                m_row = evaluate_predictions(y_test.values, y_pred, y_prob, model_name=disp_name)
                m_row["selection_mode"] = resolved_mode
                m_row["tuning_source"] = "nested_cv" if (m_key, reg_key) in canonical_params_map else "config_default"
                m_row["is_cv_selected_champion"] = bool(is_champ)
                results.append(m_row)

                models_roc[disp_name] = (y_test.values, y_prob)
                models_pr[disp_name] = (y_test.values, y_prob)
                models_cm[disp_name] = (y_test.values, y_pred)
                trained_objects[(m_key, reg_key)] = clf_obj
                preds_store[(m_key, reg_key)] = (y_pred, y_prob)

        res_df = format_metrics_table(results, output_path=str(bench_csv))

        # ------------------------------------------------------------------
        # Stage 6C: Independent Vaulted Test-Set Statistical Inference
        # ------------------------------------------------------------------
        if "unbalanced" in active_regimes and "balanced" in active_regimes:
            holdout_stat_frames: List[pd.DataFrame] = []
            for m_key in holdout_models:
                if (m_key, "unbalanced") in preds_store and (m_key, "balanced") in preds_store:
                    pred_u, prob_u = preds_store[(m_key, "unbalanced")]
                    pred_b, prob_b = preds_store[(m_key, "balanced")]
                    df_ht = compute_holdout_statistical_tests(
                        y_true=y_test.values,
                        probs_unbal=prob_u,
                        preds_unbal=pred_u,
                        probs_bal=prob_b,
                        preds_bal=pred_b,
                        model_name=MODEL_DISPLAY_NAMES[m_key],
                        n_bootstrap=2000,
                        random_state=seed,
                        output_path=None,
                    )
                    holdout_stat_frames.append(df_ht)
            if holdout_stat_frames:
                holdout_stat_df = pd.concat(holdout_stat_frames, ignore_index=True)
                holdout_stat_path = table_dir / "holdout_statistical_tests.csv"
                holdout_stat_df.to_csv(holdout_stat_path, index=False)
                logger.info(f"Saved held-out Exact McNemar & Bootstrap CI table to: {holdout_stat_path}")

        # ------------------------------------------------------------------
        # Stage 6D: Figures & Champion Tree SHAP Explainability
        # ------------------------------------------------------------------
        fig_dir.mkdir(parents=True, exist_ok=True)
        if models_roc:
            plot_roc_curves(models_roc, str(fig_dir / "roc_curves.png"))
            plot_precision_recall_curves(models_pr, str(fig_dir / "precision_recall_curves.png"))
            plot_confusion_matrices(models_cm, str(fig_dir / "confusion_matrices.png"))
        if rf_importances is not None:
            plot_feature_importances(rf_importances, top_n=20, output_path=str(fig_dir / "feature_importance.png"))

        logger.info("--- Stage 6E: SHAP Explainability on CV-Selected Tree Model ---")
        try:
            shap_target_key = None
            for _, r in selection_df.iterrows():
                cand_tuple = (r["model_key"], r["regime"])
                if r["model_key"] in ("xgboost", "random_forest") and cand_tuple in trained_objects:
                    shap_target_key = cand_tuple
                    break

            if shap_target_key is not None:
                shap_model = trained_objects[shap_target_key]
                _, shap_prob = preds_store[shap_target_key]
                explainer, shap_values = compute_shap_explanations(shap_model, X_test_enc)
                plot_shap_summary(shap_values, max_display=15, output_path=str(fig_dir / "shap_summary_beeswarm.png"))
                plot_shap_global_importance(
                    shap_values, max_display=15, output_path=str(fig_dir / "shap_importance_bar.png")
                )
                plot_shap_waterfall_cases(
                    shap_values, y_test.values, shap_prob, output_dir=str(fig_dir), max_display=12
                )
                logger.info(
                    f"Saved complete SHAP figures for {MODEL_DISPLAY_NAMES[shap_target_key[0]]} "
                    f"({REGIME_DISPLAY_NAMES[shap_target_key[1]]})."
                )
            else:
                logger.info("Skipping TreeExplainer SHAP plots (no tree-based model active in current selection).")
        except Exception as e:
            logger.warning(f"SHAP explanation generation notice: {e}")

        logger.info(
            f"Benchmark complete. Nested CV Champion: {champion_info['candidate']} "
            f"({champion_info['primary_metric']}={champion_info['nested_outer_mean_primary']:.4f})."
        )
        return res_df
