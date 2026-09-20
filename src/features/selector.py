"""
Feature selection module for high-dimensional PISA tabular data.
Filters missingness, low-variance, ranks by Mutual Information and Tree Importance,
and preserves core socio-economic and psychological PISA domain indices.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Key PISA domain variables widely established in educational research
DOMAIN_ANCHORS = [
    "ESCS",       # Index of economic, social and cultural status
    "HOMEPOS",    # Home possessions
    "ICTRES",     # ICT resources at home
    "ST004D01T",  # Student gender
    "ANXMAT",     # Mathematics anxiety
    "BELONG",     # Sense of belonging to school
    "DISCLIM",   # Disciplinary climate in mathematics
    "TEACHSUP",   # Teacher support in mathematics
    "MATHMOT",    # Mathematics motivation
    "MATHEFF",    # Mathematics self-efficacy
    "FAMSUP",     # Family support
    "REPEAT",     # Grade repetition
    "IMMIG",      # Immigration status
    "LANGN",      # Language spoken at home
]


class FeatureSelector:
    def __init__(
        self,
        target_k: int = 100,
        max_missing_ratio: float = 0.50,
        variance_threshold: float = 0.005,
        random_state: int = 42,
    ):
        self.target_k = target_k
        self.max_missing_ratio = max_missing_ratio
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.selected_features_: List[str] = []
        self.feature_scores_: pd.DataFrame = pd.DataFrame()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """
        Fit feature selection on training data only to prevent leakage.
        """
        logger.info(f"Initial candidate features: {X.shape[1]}")

        # Step 1: Missingness filter
        missing_ratios = X.isnull().mean()
        valid_missing = missing_ratios[missing_ratios <= self.max_missing_ratio].index.tolist()
        logger.info(f"Features after <= {self.max_missing_ratio*100}% missingness filter: {len(valid_missing)}")
        X_curr = X[valid_missing].copy()

        # Step 2: Separate types for initial numerical imputation
        num_cols = X_curr.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_curr.select_dtypes(exclude=[np.number]).columns.tolist()

        # Simple imputation for scoring purposes
        X_num_imp = X_curr[num_cols].fillna(X_curr[num_cols].median())
        X_cat_encoded = pd.DataFrame(index=X_curr.index)
        for col in cat_cols:
            # Categorical factorize
            X_cat_encoded[col] = pd.factorize(X_curr[col].fillna("MISSING"))[0]

        X_numeric_all = pd.concat([X_num_imp, X_cat_encoded], axis=1)

        # Step 3: Low variance filter
        var_selector = VarianceThreshold(threshold=self.variance_threshold)
        try:
            var_selector.fit(X_numeric_all)
            kept_var_cols = X_numeric_all.columns[var_selector.get_support()].tolist()
            X_numeric_all = X_numeric_all[kept_var_cols]
            logger.info(f"Features after variance threshold ({self.variance_threshold}): {len(kept_var_cols)}")
        except Exception as e:
            logger.warning(f"Variance threshold skipped due to: {e}")

        # Step 4: Mutual Information Ranking
        logger.info("Computing Mutual Information scores...")
        mi_scores = mutual_info_classif(
            X_numeric_all,
            y,
            discrete_features="auto",
            random_state=self.random_state,
            n_neighbors=5,
        )
        mi_series = pd.Series(mi_scores, index=X_numeric_all.columns)
        mi_norm = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-9)

        # Step 5: Tree-based feature importance
        logger.info("Computing Tree-based importance scores...")
        tree_model = ExtraTreesClassifier(n_estimators=100, max_depth=12, random_state=self.random_state, n_jobs=-1)
        tree_model.fit(X_numeric_all, y)
        tree_series = pd.Series(tree_model.feature_importances_, index=X_numeric_all.columns)
        tree_norm = (tree_series - tree_series.min()) / (tree_series.max() - tree_series.min() + 1e-9)

        # Step 6: Hybrid Ranking
        hybrid_score = 0.5 * mi_norm + 0.5 * tree_norm

        scores_df = pd.DataFrame({
            "feature": X_numeric_all.columns,
            "mutual_info": mi_series.values,
            "tree_importance": tree_series.values,
            "hybrid_score": hybrid_score.values,
        }).sort_values(by="hybrid_score", ascending=False).reset_index(drop=True)

        self.feature_scores_ = scores_df

        # Step 7: Select top K, ensuring domain anchors are preserved if present in candidates
        top_candidates = scores_df["feature"].tolist()
        selected = []

        # Add domain anchors first if available
        for anchor in DOMAIN_ANCHORS:
            if anchor in X_numeric_all.columns and anchor not in selected:
                selected.append(anchor)

        # Fill remaining slots up to target_k
        for feat in top_candidates:
            if feat not in selected:
                selected.append(feat)
            if len(selected) >= self.target_k:
                break

        self.selected_features_ = selected[: self.target_k]
        logger.info(f"Final selected features count: {len(self.selected_features_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select fitted feature columns from dataframe."""
        if not self.selected_features_:
            raise ValueError("FeatureSelector is not fitted yet.")
        missing_in_X = [col for col in self.selected_features_ if col not in X.columns]
        if missing_in_X:
            logger.warning(f"{len(missing_in_X)} selected features not found in X, filling with NaN: {missing_in_X[:5]}")
        return X.reindex(columns=self.selected_features_)

    def save_scores(self, output_path: str = "outputs/tables/feature_selection_summary.csv") -> Path:
        """Save feature importance scores to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_scores_.to_csv(path, index=False)
        logger.info(f"Saved feature scores summary to {path}")
        return path
