"""
Leakage-free tabular preprocessing pipeline.
Fits imputers, scalers, and encoders strictly on the training set.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from .encoders import build_categorical_encoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_strategy: str = "target_encoder",
        scaler_type: str = "robust",
        target_encoder_smoothing: float = 10.0,
    ):
        self.categorical_strategy = categorical_strategy
        self.scaler_type = scaler_type
        self.target_encoder_smoothing = target_encoder_smoothing

        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.num_imputer_ = SimpleImputer(strategy="median")
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
        self.scaler_ = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.encoder_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TabularPreprocessor":
        """Fit all transformers on training data only."""
        X_df = X.copy()
        self.num_cols_ = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

        logger.info(f"Preprocessor fit: {len(self.num_cols_)} numerical features, {len(self.cat_cols_)} categorical features.")

        # Fit numerical imputer
        if self.num_cols_:
            self.num_imputer_.fit(X_df[self.num_cols_])
            imputed_num = self.num_imputer_.transform(X_df[self.num_cols_])
            self.scaler_.fit(imputed_num)

        # Fit categorical imputer and encoder
        if self.cat_cols_:
            self.cat_imputer_.fit(X_df[self.cat_cols_])
            imputed_cat = pd.DataFrame(
                self.cat_imputer_.transform(X_df[self.cat_cols_]),
                columns=self.cat_cols_,
                index=X_df.index,
            )
            self.encoder_ = build_categorical_encoder(
                strategy=self.categorical_strategy,
                cols=self.cat_cols_,
                smoothing=self.target_encoder_smoothing,
            )
            if y is not None:
                self.encoder_.fit(imputed_cat, y)
            else:
                self.encoder_.fit(imputed_cat)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform incoming data using strictly training-fitted parameters."""
        X_df = X.copy()
        dfs = []

        # Process numerical features
        if self.num_cols_:
            missing_num = [c for c in self.num_cols_ if c not in X_df.columns]
            for c in missing_num:
                X_df[c] = np.nan
            imputed_num = self.num_imputer_.transform(X_df[self.num_cols_])
            scaled_num = self.scaler_.transform(imputed_num)
            num_df = pd.DataFrame(scaled_num, columns=self.num_cols_, index=X_df.index)
            dfs.append(num_df)

        # Process categorical features
        if self.cat_cols_:
            missing_cat = [c for c in self.cat_cols_ if c not in X_df.columns]
            for c in missing_cat:
                X_df[c] = "MISSING"
            imputed_cat = pd.DataFrame(
                self.cat_imputer_.transform(X_df[self.cat_cols_]),
                columns=self.cat_cols_,
                index=X_df.index,
            )
            encoded_cat = self.encoder_.transform(imputed_cat)
            if isinstance(encoded_cat, np.ndarray):
                encoded_cols = [f"cat_{i}" for i in range(encoded_cat.shape[1])]
                encoded_cat = pd.DataFrame(encoded_cat, columns=encoded_cols, index=X_df.index)
            dfs.append(encoded_cat)

        if not dfs:
            raise ValueError("No features available to transform.")

        transformed_df = pd.concat(dfs, axis=1)
        return transformed_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
