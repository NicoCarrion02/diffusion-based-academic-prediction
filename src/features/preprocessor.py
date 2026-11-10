"""
Leakage-free tabular preprocessing pipeline.
Fits imputers, scalers, and encoders strictly on the training partition:
- NativeImputer: Preserves survey-native types (strings for nominals, floats/ints for numericals/ordinals)
  for continuous-manifold generative modeling (TabRep-Flow).
- FeatureEncoderScaler: Empirical-Bayes TargetEncoder and Robust/StandardScaler for downstream classifiers.
"""

import logging
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from .encoders import build_categorical_encoder
from .cleaner import FeatureCategorization

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NativeImputer(BaseEstimator, TransformerMixin):
    """
    Survey-native imputer estimated strictly on training data.
    Imputes nominal missing values with a dedicated categorical token (default 'MISSING')
    and numerical/ordinal/binary missing values with the training-split median (or mean).
    """
    def __init__(
        self,
        nominal_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        categorization: Optional[FeatureCategorization] = None,
        numerical_strategy: str = "median",
        nominal_missing_token: str = "MISSING",
    ):
        self.nominal_cols = nominal_cols
        self.numerical_cols = numerical_cols
        self.categorization = categorization
        self.numerical_strategy = numerical_strategy
        self.nominal_missing_token = nominal_missing_token
        self.nominal_cols_: List[str] = []
        self.numerical_cols_: List[str] = []
        self.medians_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "NativeImputer":
        X_df = X.copy()
        self.feature_names_in_ = X_df.columns.tolist()

        if self.categorization is not None:
            self.nominal_cols_ = [c for c in self.categorization.nominal_cols if c in X_df.columns]
            self.numerical_cols_ = [c for c in self.categorization.numerical_cols if c in X_df.columns]
        else:
            if self.nominal_cols is None and self.numerical_cols is None:
                self.nominal_cols_ = [c for c in X_df.columns if not np.issubdtype(X_df[c].dtype, np.number)]
                self.numerical_cols_ = [c for c in X_df.columns if np.issubdtype(X_df[c].dtype, np.number)]
            else:
                self.nominal_cols_ = list(self.nominal_cols) if self.nominal_cols is not None else []
                self.numerical_cols_ = list(self.numerical_cols) if self.numerical_cols is not None else []

        overlap = set(self.nominal_cols_).intersection(set(self.numerical_cols_))
        if overlap:
            raise ValueError(f"Columns assigned to both nominal and numerical types: {list(overlap)}")

        assigned = set(self.nominal_cols_ + self.numerical_cols_)
        unassigned = [c for c in X_df.columns if c not in assigned]
        if unassigned:
            raise ValueError(
                f"Unassigned columns found in input data: {unassigned}. "
                f"All features must be categorized as nominal or numerical."
            )

        if self.numerical_strategy not in ("median", "mean"):
            raise ValueError(f"Unsupported numerical_strategy '{self.numerical_strategy}'. Use 'median' or 'mean'.")

        self.medians_ = {}
        for c in self.numerical_cols_:
            if c in X_df.columns:
                val = X_df[c].median() if self.numerical_strategy == "median" else X_df[c].mean()
                self.medians_[c] = float(val) if not pd.isna(val) else 0.0

        logger.info(
            f"NativeImputer fit: {len(self.numerical_cols_)} numerical/discrete features "
            f"({self.numerical_strategy} imputed), {len(self.nominal_cols_)} nominal features "
            f"('{self.nominal_missing_token}' category imputed)."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        for c in self.nominal_cols_:
            if c in X_df.columns:
                X_df[c] = X_df[c].fillna(self.nominal_missing_token).astype(str)
            else:
                X_df[c] = self.nominal_missing_token

        for c in self.numerical_cols_:
            if c in X_df.columns:
                med = self.medians_.get(c, 0.0)
                X_df[c] = X_df[c].fillna(med)
            else:
                X_df[c] = self.medians_.get(c, 0.0)

        base_cols = getattr(self, "feature_names_in_", X.columns.tolist())
        expected_order = [c for c in base_cols if c in X_df.columns] + \
                         [c for c in X_df.columns if c not in base_cols]
        return X_df[expected_order]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class FeatureEncoderScaler(BaseEstimator, TransformerMixin):
    """
    Applies empirical-Bayes TargetEncoder (on nominal categories) and RobustScaler/StandardScaler
    (on numerical, ordinal, and binary features).
    """
    def __init__(
        self,
        categorical_strategy: str = "target_encoder",
        scaler_type: str = "robust",
        target_encoder_smoothing: float = 10.0,
        nominal_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        categorization: Optional[FeatureCategorization] = None,
    ):
        self.categorical_strategy = categorical_strategy
        self.scaler_type = scaler_type
        self.target_encoder_smoothing = target_encoder_smoothing
        self.nominal_cols = nominal_cols
        self.numerical_cols = numerical_cols
        self.categorization = categorization
        self.nominal_cols_: List[str] = []
        self.numerical_cols_: List[str] = []
        self.scaler_ = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.encoder_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEncoderScaler":
        X_df = X.copy()
        self.feature_names_in_ = X_df.columns.tolist()

        if self.categorization is not None:
            self.nominal_cols_ = [c for c in self.categorization.nominal_cols if c in X_df.columns]
            self.numerical_cols_ = [c for c in self.categorization.numerical_cols if c in X_df.columns]
        else:
            if self.nominal_cols is None and self.numerical_cols is None:
                self.nominal_cols_ = [c for c in X_df.columns if not np.issubdtype(X_df[c].dtype, np.number)]
                self.numerical_cols_ = [c for c in X_df.columns if np.issubdtype(X_df[c].dtype, np.number)]
            else:
                self.nominal_cols_ = list(self.nominal_cols) if self.nominal_cols is not None else []
                self.numerical_cols_ = list(self.numerical_cols) if self.numerical_cols is not None else []

        overlap = set(self.nominal_cols_).intersection(set(self.numerical_cols_))
        if overlap:
            raise ValueError(f"Columns assigned to both nominal and numerical types: {list(overlap)}")

        assigned = set(self.nominal_cols_ + self.numerical_cols_)
        unassigned = [c for c in X_df.columns if c not in assigned]
        if unassigned:
            raise ValueError(
                f"Unassigned columns found in input data: {unassigned}. "
                f"All features must be categorized as nominal or numerical."
            )

        if self.numerical_cols_:
            self.scaler_.fit(X_df[self.numerical_cols_].values)

        if self.nominal_cols_:
            self.encoder_ = build_categorical_encoder(
                strategy=self.categorical_strategy,
                cols=self.nominal_cols_,
                smoothing=self.target_encoder_smoothing,
            )
            nom_df = X_df[self.nominal_cols_].astype(str)
            if y is not None:
                self.encoder_.fit(nom_df, y)
            else:
                self.encoder_.fit(nom_df)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        dfs = []

        if self.numerical_cols_:
            missing_num = [c for c in self.numerical_cols_ if c not in X_df.columns]
            for c in missing_num:
                X_df[c] = 0.0
            scaled_num = self.scaler_.transform(X_df[self.numerical_cols_].values)
            num_df = pd.DataFrame(scaled_num, columns=self.numerical_cols_, index=X_df.index)
            dfs.append(num_df)

        if self.nominal_cols_:
            missing_nom = [c for c in self.nominal_cols_ if c not in X_df.columns]
            for c in missing_nom:
                X_df[c] = "MISSING"
            nom_df = X_df[self.nominal_cols_].astype(str)
            encoded_cat = self.encoder_.transform(nom_df)
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
