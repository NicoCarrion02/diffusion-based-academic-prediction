"""
Gradient Boosting classifier baseline (XGBoost) for benchmark comparison.
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 250,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "tree_method": "hist",
    "random_state": 42,
    "eval_metric": "logloss",
    "n_jobs": -1,
}


def build_xgboost(**kwargs) -> Optional[object]:
    """Instantiate an XGBClassifier without touching evaluation/test data."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed. Skipping XGBoost.")
        return None

    params = dict(DEFAULT_XGB_PARAMS)
    params.update(kwargs)
    return XGBClassifier(**params)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    **kwargs,
) -> Tuple[Optional[object], Optional[np.ndarray], Optional[np.ndarray]]:
    """Train XGBoost Classifier with complete kwargs customization and predict on X_test."""
    xgb = build_xgboost(**kwargs)
    if xgb is None:
        return None, None, None

    logger.info(f"Training XGBoost Classifier on {len(X_train)} samples with {X_train.shape[1]} features...")
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    return xgb, y_pred, y_prob
