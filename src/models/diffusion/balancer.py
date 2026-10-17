"""
Dataset balancing orchestrator using TabRep diffusion flow generation.
Includes benchmark oversamplers (RandomOverSampler) and automated statistical quality verification.
"""

import logging
from pathlib import Path
from typing import Optional, Any
import pandas as pd
import numpy as np

from .tabrep_adapter import TabRepDiffusionModel
from .diagnostics import compute_distribution_metrics, plot_distribution_audit, assert_synthetic_validity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DatasetBalancer:
    def __init__(
        self,
        strategy: str = "tabrep",
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        timesteps: int = 50,
        hidden_dims: Optional[list] = None,
        dropout: float = 0.1,
        solver: str = "heun",
        random_state: int = 42,
    ):
        self.strategy = strategy.lower()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.timesteps = timesteps
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 512, 256]
        self.dropout = dropout
        self.solver = solver
        self.random_state = random_state
        self.diffusion_model_: Optional[TabRepDiffusionModel] = None

    def balance_training_data(
        self,
        train_df: pd.DataFrame,
        target_col: str = "target",
        output_path: Optional[str] = "data/processed/train_balanced_tabrep.parquet",
        force: bool = False,
        cat_cols: Optional[list] = None,
        num_cols: Optional[list] = None,
        discrete_cols: Optional[list] = None,
        categorization: Optional[Any] = None,
        save_diagnostics: bool = True,
    ) -> pd.DataFrame:
        """
        Balances the training dataset to a 1:1 ratio.
        Generates synthetic minority instances (Class 1) using TabRep Flow generation.
        """
        if output_path is not None:
            out_file = Path(output_path)
            if out_file.exists() and not force:
                cached_df = pd.read_parquet(out_file)
                expected_cols = set(train_df.columns)
                cached_cols = set(c for c in cached_df.columns if c != "is_synthetic")
                if expected_cols == cached_cols:
                    logger.info(f"Loading cached balanced dataset from: {out_file} (matches all {len(expected_cols)} features).")
                    return cached_df
                logger.info("Cached balanced dataset columns do not match current training features. Re-generating with TabRep...")

        counts = train_df[target_col].value_counts().to_dict()
        n_0 = counts.get(0, 0)
        n_1 = counts.get(1, 0)
        deficit = n_0 - n_1

        logger.info(f"Class distribution before balancing - Class 0: {n_0}, Class 1: {n_1} (Deficit: {deficit})")

        if deficit <= 0:
            logger.info("Dataset is already balanced or minority is class 0. Returning original data.")
            return train_df.copy()

        if self.strategy == "tabrep":
            logger.info(f"Training TabRep Flow model on minority class (Class 1, N={n_1}) to synthesize {deficit} samples...")
            minority_features = train_df[train_df[target_col] == 1].drop(columns=[target_col]).copy()

            self.diffusion_model_ = TabRepDiffusionModel(
                num_timesteps=self.timesteps,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                random_state=self.random_state,
            )
            self.diffusion_model_.fit(
                minority_features,
                cat_cols=cat_cols,
                num_cols=num_cols,
                discrete_cols=discrete_cols,
                categorization=categorization,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

            synthetic_samples = self.diffusion_model_.sample(
                num_samples=deficit,
                batch_size=256,
                solver=self.solver,
            )

            # Automated distribution quality audit
            metrics_df, summary = compute_distribution_metrics(minority_features, synthetic_samples)
            if save_diagnostics:
                table_out = Path("outputs/tables/tabrep_distribution_metrics.csv")
                fig_out = Path("outputs/figures/tabrep_distribution_audit.png")
                table_out.parent.mkdir(parents=True, exist_ok=True)
                fig_out.parent.mkdir(parents=True, exist_ok=True)

                metrics_df.to_csv(table_out, index=False)
                plot_distribution_audit(minority_features, synthetic_samples, metrics_df, output_path=str(fig_out))
                logger.info(f"Saved distribution audit table to {table_out} and figure to {fig_out}")
            logger.info(
                f"Synthetic Audit Summary: Mean KS = {summary['mean_ks_stat']:.4f}, "
                f"Mean Wasserstein = {summary['mean_wasserstein']:.4f}, "
                f"Mean OOB = {summary['mean_oob_pct']:.2f}%, "
                f"Corr MAE = {summary['corr_matrix_mae']:.4f}, "
                f"Exact Memorization = {summary['exact_copy_rate_pct']:.1f}%"
            )

            # Enforce statistical quality gate
            assert_synthetic_validity(minority_features, synthetic_samples)

            synthetic_samples[target_col] = 1
            synthetic_samples["is_synthetic"] = 1

            train_df_real = train_df.copy()
            train_df_real["is_synthetic"] = 0

            balanced_df = pd.concat([train_df_real, synthetic_samples], axis=0, ignore_index=True)
            balanced_df = balanced_df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        elif self.strategy == "random_oversample":
            logger.info(f"Applying Random Oversampling to Class 1 (+{deficit} samples)...")
            minority_df = train_df[train_df[target_col] == 1].sample(n=deficit, replace=True, random_state=self.random_state)
            minority_df["is_synthetic"] = 1
            train_df_real = train_df.copy()
            train_df_real["is_synthetic"] = 0
            balanced_df = pd.concat([train_df_real, minority_df], axis=0, ignore_index=True)
            balanced_df = balanced_df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        else:
            raise ValueError(f"Unsupported balancing strategy: {self.strategy}")

        new_counts = balanced_df[target_col].value_counts().to_dict()
        logger.info(f"Balanced dataset ready: Class 0: {new_counts.get(0, 0)}, Class 1: {new_counts.get(1, 0)}")

        if output_path is not None:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            balanced_df.to_parquet(out_file, index=False)
            logger.info(f"Saved balanced dataset to: {out_file}")

        return balanced_df
