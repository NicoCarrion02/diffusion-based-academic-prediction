"""
Random Forest classifier baseline for student performance prediction.
"""

import logging
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    **kwargs,
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray, pd.Series]:
    """
    Train Random Forest classifier, evaluate on test, and return predictions and probabilities.
    Supports complete hyperparameter customization via kwargs.
    """
    params = {
        "n_estimators": 300,
        "max_depth": 18,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(kwargs)
    logger.info(f"Training Random Forest on {len(X_train)} samples with {X_train.shape[1]} features (params: {params})...")
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    logger.info(f"Random Forest fit complete. Top 3 features: {importances.index[:3].tolist()}")

    return rf, y_pred, y_prob, importances
