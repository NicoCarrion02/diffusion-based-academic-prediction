"""
Gradient Boosting baselines (XGBoost, LightGBM) for benchmark comparison.
"""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    **kwargs,
) -> Tuple[Optional[object], Optional[np.ndarray], Optional[np.ndarray]]:
    """Train XGBoost Classifier with complete kwargs customization."""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("xgboost not installed. Skipping XGBoost.")
        return None, None, None

    params = {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }
    params.update(kwargs)
    logger.info(f"Training XGBoost Classifier with params: {params}...")
    xgb = XGBClassifier(**params)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    return xgb, y_pred, y_prob
