"""
Master pipeline orchestrator for the Academic Prediction Diffusion Project.
Supports execution of individual steps or complete end-to-end runs.
All steps are fully idempotent.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.data.downloader import ensure_raw_dataset
from src.data.converter import convert_sas_to_parquet
from src.data.loader import prepare_train_test_data
from src.features.cleaner import clean_pisa_survey_codes, categorize_pisa_features
from src.features.selector import FeatureSelector
from src.features.preprocessor import TabularPreprocessor
from src.models.diffusion.balancer import DatasetBalancer
from src.models.classifiers.random_forest import train_random_forest
from src.models.classifiers.tabpfn_classifier import train_tabpfn
from src.models.classifiers.baseline_comparisons import train_xgboost
from src.evaluation.metrics import evaluate_predictions, format_metrics_table
from src.evaluation.plots import (
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrices,
    plot_feature_importances,
)
from src.evaluation.statistical_tests import compare_models_cv
from src.evaluation.explainability import (
    compute_shap_explanations,
    plot_shap_summary,
    plot_shap_global_importance,
    plot_shap_waterfall_cases,
)

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
        with open(path, "r") as f:
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
            force=force,
            metadata_explorer=self.explorer,
        )

    def run_feature_selection(self, force: bool = False):
        """Step 4: Select top k features (k=100 benchmark) and preprocess."""
        train_df = pd.read_parquet(self.config["data"]["train_parquet"])
        test_df = pd.read_parquet(self.config["data"]["test_parquet"])

        target_col = "target"
        X_train_raw = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test_raw = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        sel_cfg = self.feat_cfg["selection"]
        enc_cfg = self.feat_cfg["encoding"]
        prep_cfg = self.feat_cfg["preprocessing"]

        # Clean format-declared survey codes
        if prep_cfg.get("clean_survey_codes", True):
            logger.info("Applying metadata-driven PISA survey code cleaning before feature selection...")
            X_train_raw = clean_pisa_survey_codes(X_train_raw, self.explorer)
            X_test_raw = clean_pisa_survey_codes(X_test_raw, self.explorer)

        # Feature selection
        selector = FeatureSelector(
            target_k=sel_cfg["target_k"],
            max_missing_ratio=sel_cfg["max_missing_ratio"],
            variance_threshold=sel_cfg["variance_threshold"],
            random_state=self.config["execution"]["seed"],
        )
        selector.fit(X_train_raw, y_train, metadata_explorer=self.explorer)
        selector.save_scores()

        X_train_sel = selector.transform(X_train_raw)
        X_test_sel = selector.transform(X_test_raw)

        # Categorize selected features
        cat_registry = categorize_pisa_features(X_train_sel, self.explorer)
        self.cat_registry_ = cat_registry

        # Preprocessing and encoding
        preprocessor = TabularPreprocessor(
            categorical_strategy=enc_cfg["categorical_strategy"],
            categorization=cat_registry,
            scaler_type=prep_cfg["scaler"],
            imputation_strategy=prep_cfg.get("numerical_imputer", "median"),
            target_encoder_smoothing=enc_cfg["target_encoder_smoothing"],
        )
        X_train_proc = preprocessor.fit_transform(X_train_sel, y_train)
        X_test_proc = preprocessor.transform(X_test_sel)

        processed_dir = Path(self.config["data"]["processed_dir"])
        X_train_proc["target"] = y_train.values
        X_test_proc["target"] = y_test.values

        X_train_proc.to_parquet(processed_dir / "train_selected.parquet", index=False)
        X_test_proc.to_parquet(processed_dir / "test_selected.parquet", index=False)
        logger.info(f"Saved processed train/test datasets to {processed_dir}")

        return X_train_proc, X_test_proc, selector.selected_features_

    def run_diffusion_balancing(self, force: bool = False):
        """Step 5: Train TabRep diffusion model on minority class to balance training data."""
        processed_dir = Path(self.config["data"]["processed_dir"])
        train_path = processed_dir / "train_selected.parquet"
        if not train_path.exists():
            self.run_feature_selection(force=force)

        train_df = pd.read_parquet(train_path)
        diff_cfg = self.model_cfg["diffusion"]

        net_cfg = diff_cfg.get("network", {})
        train_cfg = diff_cfg.get("training", {})

        balancer = DatasetBalancer(
            strategy="tabrep",
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 1e-5),
            timesteps=train_cfg["timesteps"],
            hidden_dims=net_cfg.get("hidden_dims", [256, 512, 256]),
            dropout=net_cfg.get("dropout", 0.10),
            solver=train_cfg.get("solver", "heun"),
            random_state=self.config["execution"]["seed"],
        )

        reg = getattr(self, "cat_registry_", None)
        if reg is None:
            from src.features.cleaner import categorize_pisa_features
            feature_cols = [c for c in train_df.columns if c != "target"]
            reg = categorize_pisa_features(train_df[feature_cols], self.explorer)
            self.cat_registry_ = reg

        balanced_df = balancer.balance_training_data(
            train_df=train_df,
            target_col="target",
            output_path=self.config["data"]["train_balanced_parquet"],
            force=force,
            categorization=reg,
        )
        return balanced_df

    def run_benchmark(self, force: bool = False):
        """Step 6: Train Random Forest, TabPFNv3, and compare on test data."""
        processed_dir = Path(self.config["data"]["processed_dir"])
        train_path = processed_dir / "train_selected.parquet"
        test_path = processed_dir / "test_selected.parquet"
        balanced_path = Path(self.config["data"]["train_balanced_parquet"])

        if not train_path.exists() or not test_path.exists():
            self.run_feature_selection(force=force)

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)

        if not balanced_path.exists():
            self.run_diffusion_balancing(force=force)
        balanced_df = pd.read_parquet(balanced_path)

        # Prepare X, y
        target_col = "target"
        drop_cols = [target_col, "is_synthetic"] if "is_synthetic" in balanced_df.columns else [target_col]

        X_train_unbal = train_df.drop(columns=[target_col])
        y_train_unbal = train_df[target_col]

        X_train_bal = balanced_df.drop(columns=drop_cols)
        y_train_bal = balanced_df[target_col]

        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        feature_order = [c for c in X_train_unbal.columns if c in X_train_bal.columns]
        X_train_unbal = X_train_unbal[feature_order]
        X_train_bal = X_train_bal[feature_order]
        X_test = X_test[feature_order]

        from src.models.diffusion.tabrep_adapter import set_seed
        set_seed(42)

        results = []
        models_roc = {}
        models_pr = {}
        models_cm = {}

        rf_cfg = self.model_cfg["classifiers"]["random_forest"]

        logger.info("--- Benchmark 1: Random Forest (Unbalanced Baseline) ---")
        rf_unbal, y_pred_rf1, y_prob_rf1, importances = train_random_forest(
            X_train=X_train_unbal,
            y_train=y_train_unbal,
            X_test=X_test,
            **rf_cfg,
        )
        m1 = evaluate_predictions(y_test, y_pred_rf1, y_prob_rf1, model_name="Random Forest (Unbalanced)")
        results.append(m1)
        models_roc["RF (Unbalanced)"] = (y_test.values, y_prob_rf1)
        models_pr["RF (Unbalanced)"] = (y_test.values, y_prob_rf1)
        models_cm["RF (Unbalanced)"] = (y_test.values, y_pred_rf1)

        logger.info("--- Benchmark 2: Random Forest (TabRep Diffusion Balanced) ---")
        rf_bal, y_pred_rf2, y_prob_rf2, _ = train_random_forest(
            X_train=X_train_bal,
            y_train=y_train_bal,
            X_test=X_test,
            **rf_cfg,
        )
        m2 = evaluate_predictions(y_test, y_pred_rf2, y_prob_rf2, model_name="Random Forest (TabRep Balanced)")
        results.append(m2)
        models_roc["RF (TabRep Balanced)"] = (y_test.values, y_prob_rf2)
        models_pr["RF (TabRep Balanced)"] = (y_test.values, y_prob_rf2)
        models_cm["RF (TabRep Balanced)"] = (y_test.values, y_pred_rf2)

        logger.info("--- Benchmark 3: XGBoost (Unbalanced Baseline) ---")
        xgb_cfg = self.model_cfg["classifiers"]["xgboost"]
        xgb_unbal, y_pred_xgb1, y_prob_xgb1 = train_xgboost(
            X_train=X_train_unbal,
            y_train=y_train_unbal,
            X_test=X_test,
            **xgb_cfg,
        )
        if y_pred_xgb1 is not None:
            m3 = evaluate_predictions(y_test, y_pred_xgb1, y_prob_xgb1, model_name="XGBoost (Unbalanced)")
            results.append(m3)
            models_roc["XGBoost (Unbalanced)"] = (y_test.values, y_prob_xgb1)
            models_pr["XGBoost (Unbalanced)"] = (y_test.values, y_prob_xgb1)
            models_cm["XGBoost (Unbalanced)"] = (y_test.values, y_pred_xgb1)

        logger.info("--- Benchmark 4: XGBoost (TabRep Diffusion Balanced) ---")
        xgb_bal, y_pred_xgb2, y_prob_xgb2 = train_xgboost(
            X_train=X_train_bal,
            y_train=y_train_bal,
            X_test=X_test,
            **xgb_cfg,
        )
        if y_pred_xgb2 is not None:
            m4 = evaluate_predictions(y_test, y_pred_xgb2, y_prob_xgb2, model_name="XGBoost (TabRep Balanced)")
            results.append(m4)
            models_roc["XGBoost (TabRep Balanced)"] = (y_test.values, y_prob_xgb2)
            models_pr["XGBoost (TabRep Balanced)"] = (y_test.values, y_prob_xgb2)
            models_cm["XGBoost (TabRep Balanced)"] = (y_test.values, y_pred_xgb2)

        logger.info("--- Benchmark 5: TabPFNv3 Tabular Foundation Model ---")
        tabpfn_model, y_pred_pfn, y_prob_pfn = train_tabpfn(
            X_train=X_train_unbal,
            y_train=y_train_unbal,
            X_test=X_test,
            device=self.model_cfg["classifiers"]["tabpfn"].get("device", "auto"),
            n_estimators=self.model_cfg["classifiers"]["tabpfn"].get("n_estimators", 4),
            random_state=self.model_cfg["classifiers"]["tabpfn"].get("random_state", 42),
        )
        if y_pred_pfn is not None:
            m5 = evaluate_predictions(y_test, y_pred_pfn, y_prob_pfn, model_name="TabPFNv3 (Unbalanced)")
            results.append(m5)
            models_roc["TabPFNv3"] = (y_test.values, y_prob_pfn)
            models_pr["TabPFNv3"] = (y_test.values, y_prob_pfn)
            models_cm["TabPFNv3"] = (y_test.values, y_pred_pfn)

        res_df = format_metrics_table(results)

        fig_dir = Path(self.config["outputs"]["figures_dir"])
        plot_roc_curves(models_roc, str(fig_dir / "roc_curves.png"))
        plot_precision_recall_curves(models_pr, str(fig_dir / "precision_recall_curves.png"))
        plot_confusion_matrices(models_cm, str(fig_dir / "confusion_matrices.png"))
        plot_feature_importances(importances, top_n=20, output_path=str(fig_dir / "feature_importance.png"))

        # Explainability Analysis with SHAP
        logger.info("--- Step 6b: SHAP Explainability & Student Profiles ---")
        try:
            best_model = xgb_bal if xgb_bal is not None else rf_bal
            best_prob = y_prob_xgb2 if xgb_bal is not None else y_prob_rf2
            explainer, shap_values = compute_shap_explanations(best_model, X_test)
            plot_shap_summary(shap_values, max_display=15, output_path=str(fig_dir / "shap_summary_beeswarm.png"))
            plot_shap_global_importance(shap_values, max_display=15, output_path=str(fig_dir / "shap_importance_bar.png"))
            plot_shap_waterfall_cases(shap_values, y_test.values, best_prob, output_dir=str(fig_dir), max_display=12)
            logger.info("Saved complete SHAP figures (beeswarm, importance bar, student waterfalls).")
        except Exception as e:
            logger.warning(f"SHAP explanation generation notice: {e}")

        # Statistical CV comparison (Random Forest, XGBoost, and optional TabPFN)
        logger.info("--- Step 7: Multi-Model Statistical Cross-Validation ---")
        val_cfg = self.model_cfg.get("validation", {})
        diff_cfg = self.model_cfg.get("diffusion", {}).get("training", {})
        cat_reg = getattr(self, "cat_registry_", None)

        df_rf_stat = compare_models_cv(
            X_train_unbal,
            y_train_unbal,
            n_splits=val_cfg.get("cv_folds", 5),
            random_state=val_cfg.get("random_state", 42),
            model_type="random_forest",
            balancer_strategy="tabrep",
            categorization=cat_reg,
            diffusion_config=diff_cfg,
            output_path=None,
        )
        df_xgb_stat = compare_models_cv(
            X_train_unbal,
            y_train_unbal,
            n_splits=val_cfg.get("cv_folds", 5),
            random_state=val_cfg.get("random_state", 42),
            model_type="xgboost",
            balancer_strategy="tabrep",
            categorization=cat_reg,
            diffusion_config=diff_cfg,
            output_path=None,
        )
        stat_frames = [df_rf_stat, df_xgb_stat]

        if val_cfg.get("evaluate_tabpfn_cv", False):
            logger.info("Running TabPFNv3 cross-validation...")
            df_tabpfn_stat = compare_models_cv(
                X_train_unbal,
                y_train_unbal,
                n_splits=val_cfg.get("cv_folds", 5),
                random_state=val_cfg.get("random_state", 42),
                model_type="tabpfn",
                balancer_strategy="tabrep",
                categorization=cat_reg,
                diffusion_config=diff_cfg,
                output_path=None,
            )
            stat_frames.append(df_tabpfn_stat)

        combined_stat = pd.concat(stat_frames, ignore_index=True)
        stat_out = Path("outputs/tables/statistical_significance.csv")
        stat_out.parent.mkdir(parents=True, exist_ok=True)
        combined_stat.to_csv(stat_out, index=False)
        logger.info(f"Saved complete multi-model paired statistical tests to: {stat_out}")

        logger.info("Pipeline execution successfully completed.")
        return res_df
