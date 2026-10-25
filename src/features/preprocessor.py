"""
Leakage-free tabular preprocessing pipeline.
Fits imputers, scalers, and encoders strictly on the training set.
Provides decoupled:
- NativeImputer: Retains survey-native types (strings for nominals, floats/ints for numericals/ordinals)
  for generative models such as TabRep.
- FeatureEncoderScaler: TargetEncoder + Robust/StandardScaler for downstream ML classifiers.
- TabularPreprocessor: Composite pipeline for backwards compatibility.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from .encoders import build_categorical_encoder
from .cleaner import FeatureCategorization

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NativeImputer(BaseEstimator, TransformerMixin):
    """
    Leakage-free native imputer.
    Fills nominal missing values with "MISSING" string (handled natively as a category token in TabRep's PSK).
    Fills ordinal/binary/continuous numerical missing values with training-set median.
    Retains native dtypes (string for nominals, float/int for numericals) so TabRep can perform CatConverter
    and discrete integer snapping.
    """
    def __init__(
        self,
        nominal_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        categorization: Optional[FeatureCategorization] = None,
    ):
        self.nominal_cols = nominal_cols
        self.numerical_cols = numerical_cols
        self.categorization = categorization
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

        self.medians_ = {}
        for c in self.numerical_cols_:
            if c in X_df.columns:
                med = X_df[c].median()
                self.medians_[c] = float(med) if not pd.isna(med) else 0.0

        logger.info(
            f"NativeImputer fit: {len(self.numerical_cols_)} numerical/discrete features (median imputed), "
            f"{len(self.nominal_cols_)} nominal features ('MISSING' category imputed)."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = X.copy()
        for c in self.nominal_cols_:
            if c in X_df.columns:
                X_df[c] = X_df[c].fillna("MISSING").astype(str)
            else:
                X_df[c] = "MISSING"

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
    Applies TargetEncoder (on nominal categories) and RobustScaler/StandardScaler (on numerical/ordinal/binary features).
    Takes imputed native DataFrames (real or synthetic) and outputs float representations ready for ML classifiers.
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


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Composite tabular preprocessing pipeline for end-to-end transformation.
    Preserves backwards compatibility with existing pipeline calls.
    """
    def __init__(
        self,
        categorical_strategy: str = "target_encoder",
        scaler_type: str = "robust",
        imputation_strategy: str = "median",
        target_encoder_smoothing: float = 10.0,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        categorization: Optional[FeatureCategorization] = None,
    ):
        self.categorical_strategy = categorical_strategy
        self.scaler_type = scaler_type
        self.imputation_strategy = imputation_strategy
        self.target_encoder_smoothing = target_encoder_smoothing
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.categorization = categorization

        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.indicator_cols_: List[str] = []

        add_indicator = (imputation_strategy == "median_indicator")
        self.num_imputer_ = SimpleImputer(strategy="median", add_indicator=add_indicator)
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
        self.scaler_ = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.encoder_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TabularPreprocessor":
        """Fit imputers, scalers, and categorical encoders on training data."""
        X_df = X.copy()
        self.feature_names_in_ = X_df.columns.tolist()

        if self.categorization is not None:
            cat_cols = [c for c in self.categorization.nominal_cols if c in X_df.columns]
            num_cols = [c for c in self.categorization.numerical_cols if c in X_df.columns]
        else:
            if self.categorical_cols is None or self.numerical_cols is None:
                raise ValueError(
                    "TabularPreprocessor requires explicit column lists (categorical_cols and numerical_cols) "
                    "or a FeatureCategorization object. Automatic type inference is disabled."
                )
            cat_cols = list(self.categorical_cols)
            num_cols = list(self.numerical_cols)

        overlap = set(cat_cols).intersection(set(num_cols))
        if overlap:
            raise ValueError(f"Columns assigned to both categorical and numerical types: {list(overlap)}")

        assigned = set(cat_cols + num_cols)
        unassigned = [c for c in X_df.columns if c not in assigned]
        if unassigned:
            raise ValueError(
                f"Unassigned columns found in input data: {unassigned}. "
                f"All features must be explicitly categorized as numerical or categorical."
            )

        self.cat_cols_ = [c for c in cat_cols if c in X_df.columns]
        self.num_cols_ = [c for c in num_cols if c in X_df.columns]

        logger.info(
            f"Preprocessor fit: {len(self.num_cols_)} numerical/discrete features, "
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
