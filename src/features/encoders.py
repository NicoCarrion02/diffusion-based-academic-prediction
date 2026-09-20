"""
Modular categorical encoders for tabular learning and diffusion models.
Supports TargetEncoder, OneHot, Ordinal, Frequency, and TabRep's CatConverter.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical categories by their relative frequencies."""
    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols
        self.maps_: Dict[str, Dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols is not None else X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.cols_ = cols
        for col in cols:
            freq = X[col].value_counts(normalize=True).to_dict()
            self.maps_[col] = freq
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cols_:
            if col in X_out.columns:
                m = self.maps_[col]
                X_out[col] = X_out[col].map(m).fillna(0.0).astype(float)
        return X_out


class CatConverterEncoder(BaseEstimator, TransformerMixin):
    """
    TabRep's CatConverter: Maps each of K discrete categories to a 2D point 
    on the unit circle via sine and cosine functions.
    cos(2*pi*k / K), sin(2*pi*k / K).
    Provides dense, continuous, geometry-aware embeddings for diffusion models.
    """
    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols
        self.categories_: Dict[str, List] = {}
        self.cat_to_idx_: Dict[str, Dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.cols if self.cols is not None else X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.cols_ = cols
        for col in cols:
            cats = sorted(X[col].dropna().unique().tolist())
            self.categories_[col] = cats
            self.cat_to_idx_[col] = {cat: idx for idx, cat in enumerate(cats)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cols_:
            if col not in X_out.columns:
                continue
            mapping = self.cat_to_idx_[col]
            K = max(len(mapping), 1)
            indices = X_out[col].map(mapping).fillna(0).values.astype(float)
            angles = 2.0 * np.pi * indices / K
            cos_vals = np.cos(angles)
            sin_vals = np.sin(angles)
            X_out[f"{col}_cos"] = cos_vals
            X_out[f"{col}_sin"] = sin_vals
            X_out.drop(columns=[col], inplace=True)
        return X_out

    def inverse_transform_coordinates(self, col: str, cos_vals: np.ndarray, sin_vals: np.ndarray) -> np.ndarray:
        """Inverse transforms (cos, sin) coordinates back to original discrete category."""
        if col not in self.categories_:
            raise KeyError(f"Column {col} not fitted in CatConverterEncoder")
        cats = self.categories_[col]
        K = len(cats)
        if K == 0:
            return np.array([])
        angles = np.arctan2(sin_vals, cos_vals) % (2.0 * np.pi)
        pred_idx = np.round((angles * K) / (2.0 * np.pi)).astype(int) % K
        cat_arr = np.array(cats)
        return cat_arr[pred_idx]


def build_categorical_encoder(strategy: str = "target_encoder", cols: Optional[List[str]] = None, **kwargs):
    """
    Factory function for modular categorical encoders.
    Permits flexible experimentation: 'target_encoder', 'one_hot', 'ordinal', 'frequency', 'catconverter'.
    """
    strategy = strategy.lower()
    if strategy == "target_encoder":
        try:
            import category_encoders as ce
            smoothing = kwargs.get("smoothing", 10.0)
            return ce.TargetEncoder(cols=cols, smoothing=smoothing)
        except ImportError:
            try:
                from sklearn.preprocessing import TargetEncoder as SklearnTargetEncoder
                return SklearnTargetEncoder(smooth=kwargs.get("smoothing", "auto"))
            except ImportError:
                logger.warning("TargetEncoder not available. Falling back to OrdinalEncoder.")
                return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    elif strategy == "catboost":
        try:
            import category_encoders as ce
            return ce.CatBoostEncoder(cols=cols)
        except ImportError:
            logger.warning("CatBoostEncoder not available. Falling back to OrdinalEncoder.")
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    elif strategy == "one_hot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    elif strategy == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    elif strategy == "frequency":
        return FrequencyEncoder(cols=cols)

    elif strategy == "catconverter":
        return CatConverterEncoder(cols=cols)

    else:
        raise ValueError(f"Unknown categorical encoding strategy: '{strategy}'")
