"""
Modular categorical encoders for downstream tabular classifiers.
Supports TargetEncoder, CatBoostEncoder, OneHotEncoder, OrdinalEncoder, and FrequencyEncoder.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes nominal categories by their empirical training frequencies."""
    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols
        self.maps_: Dict[str, Dict] = {}

    def fit(self, X: pd.DataFrame, y=None):
        if not self.cols:
            raise ValueError("FrequencyEncoder requires an explicit list of columns in 'cols'.")
        self.cols_ = list(self.cols)
        for col in self.cols_:
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


def build_categorical_encoder(strategy: str = "target_encoder", cols: Optional[List[str]] = None, **kwargs):
    """
    Factory function for downstream categorical encoders.
    Supported strategies: 'target_encoder', 'catboost', 'one_hot', 'ordinal', 'frequency'.
    """
    strategy = strategy.lower()
    if strategy in ("target_encoder", "target"):
        try:
            import category_encoders as ce
            smoothing = kwargs.get("smoothing", 10.0)
            return ce.TargetEncoder(
                cols=cols,
                smoothing=smoothing,
                handle_unknown="value",
                handle_missing="value",
            )
        except ImportError:
            try:
                from sklearn.preprocessing import TargetEncoder as SklearnTargetEncoder
                return SklearnTargetEncoder(smooth=kwargs.get("smoothing", "auto"), cv=5)
            except ImportError:
                raise ImportError(
                    "TargetEncoder is not available. Please install 'category_encoders' "
                    "or ensure scikit-learn >= 1.3 is installed."
                )

    elif strategy == "catboost":
        try:
            import category_encoders as ce
            return ce.CatBoostEncoder(cols=cols)
        except ImportError:
            raise ImportError("CatBoostEncoder requires 'category_encoders'. Run: pip install category_encoders")

    elif strategy == "one_hot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    elif strategy == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    elif strategy == "frequency":
        return FrequencyEncoder(cols=cols)

    else:
        raise ValueError(f"Unknown categorical encoding strategy: '{strategy}'")
