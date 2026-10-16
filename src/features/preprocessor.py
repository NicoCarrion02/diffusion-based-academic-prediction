"""
Leakage-free tabular preprocessing pipeline.
Fits imputers, scalers, and encoders strictly on the training set.
Supports median imputation with missingness indicators to capture survey non-response patterns.
"""

import logging
from typing import List, Optional
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
        imputation_strategy: str = "median",
        target_encoder_smoothing: float = 10.0,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
    ):
        self.categorical_strategy = categorical_strategy
        self.scaler_type = scaler_type
        self.imputation_strategy = imputation_strategy
        self.target_encoder_smoothing = target_encoder_smoothing
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols

        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.indicator_cols_: List[str] = []

        add_indicator = (imputation_strategy == "median_indicator")
        self.num_imputer_ = SimpleImputer(strategy="median", add_indicator=add_indicator)
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
        self.scaler_ = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.encoder_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TabularPreprocessor":
        """Fit all transformers on training data only."""
        X_df = X.copy()
        
        self.feature_names_in_ = X_df.columns.tolist()
        
        if self.categorical_cols is not None:
            cat_list = [self.categorical_cols] if isinstance(self.categorical_cols, str) else list(self.categorical_cols)
            self.cat_cols_ = [c for c in cat_list if c in X_df.columns]
        else:
            self.cat_cols_ = []

        if self.numerical_cols is not None:
            num_list = [self.numerical_cols] if isinstance(self.numerical_cols, str) else list(self.numerical_cols)
            self.num_cols_ = [c for c in num_list if c in X_df.columns and c not in self.cat_cols_]
        else:
            self.num_cols_ = [c for c in X_df.columns if c not in self.cat_cols_]

        logger.info(
            f"Preprocessor fit: {len(self.num_cols_)} numerical/ordinal features, "
            f"{len(self.cat_cols_)} categorical features (imputer: {self.imputation_strategy})."
        )

        if self.num_cols_:
            imputed = self.num_imputer_.fit_transform(X_df[self.num_cols_])
            n_num = len(self.num_cols_)
            imputed_num = imputed[:, :n_num]
            self.scaler_.fit(imputed_num)

            if self.num_imputer_.add_indicator and self.num_imputer_.indicator_ is not None:
                self.indicator_cols_ = [
                    f"{self.num_cols_[idx]}_is_missing"
                    for idx in self.num_imputer_.indicator_.features_
                ]
                logger.info(f"Generated {len(self.indicator_cols_)} binary missingness indicators.")
            else:
                self.indicator_cols_ = []

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
        """Transform incoming data using fitted parameters."""
        X_df = X.copy()
        dfs = []

        if self.num_cols_:
            missing_num = [c for c in self.num_cols_ if c not in X_df.columns]
            for c in missing_num:
                X_df[c] = np.nan
            
            imputed = self.num_imputer_.transform(X_df[self.num_cols_])
            n_num = len(self.num_cols_)
            imputed_num = imputed[:, :n_num]
            scaled_num = self.scaler_.transform(imputed_num)

            num_df = pd.DataFrame(scaled_num, columns=self.num_cols_, index=X_df.index)

            if self.indicator_cols_ and imputed.shape[1] > n_num:
                ind_vals = imputed[:, n_num:]
                ind_df = pd.DataFrame(ind_vals, columns=self.indicator_cols_, index=X_df.index)
                num_df = pd.concat([num_df, ind_df], axis=1)

            dfs.append(num_df)

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
        base_cols = getattr(self, "feature_names_in_", X.columns.tolist())
        expected_order = [c for c in base_cols if c in transformed_df.columns] + \
                         [c for c in transformed_df.columns if c not in base_cols]
        return transformed_df[expected_order]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
