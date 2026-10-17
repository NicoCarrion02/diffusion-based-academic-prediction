"""
Statistical significance testing comparing classifiers across cross-validation folds.
Performs rigorous paired t-tests and Wilcoxon signed-rank tests between baseline
(unbalanced) and balanced models to demonstrate statistical validity (p < 0.05).
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, recall_score, balanced_accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def balance_fold_data(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    strategy: str = "tabrep",
    random_state: int = 42,
    categorization: Optional[Any] = None,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    discrete_cols: Optional[List[str]] = None,
    diffusion_config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balances a training fold to a 1:1 ratio.
    Supports 'tabrep' (Continuous Flow Matching) and 'random_oversample'.
    """
    n_0 = (y_tr == 0).sum()
    n_1 = (y_tr == 1).sum()
    deficit = n_0 - n_1
    if deficit <= 0:
        return X_tr.copy(), y_tr.copy()

    strat = strategy.lower()
    if strat == "tabrep":
        from src.models.diffusion.balancer import DatasetBalancer

        cfg = diffusion_config or {}
        balancer = DatasetBalancer(
            strategy="tabrep",
            epochs=cfg.get("epochs", 40),
            batch_size=cfg.get("batch_size", 128),
            lr=cfg.get("learning_rate", 0.001),
            timesteps=cfg.get("timesteps", 50),
            solver=cfg.get("solver", "heun"),
            random_state=random_state,
        )
        fold_train = X_tr.copy()
        fold_train["target"] = y_tr.values

        if categorization is None and cat_cols is None and num_cols is None:
            num_cols = list(X_tr.columns)
            cat_cols = []

        balanced_fold = balancer.balance_training_data(
            train_df=fold_train,
            target_col="target",
            output_path=None,
            cat_cols=cat_cols,
            num_cols=num_cols,
            discrete_cols=discrete_cols,
            categorization=categorization,
            save_diagnostics=False,
        )
        drop_cols = [c for c in ["target", "is_synthetic"] if c in balanced_fold.columns]
        X_bal = balanced_fold.drop(columns=drop_cols)
        y_bal = balanced_fold["target"]
        return X_bal, y_bal

    elif strat == "random_oversample":
        minority_idx = y_tr[y_tr == 1].index
        sampled_idx = np.random.RandomState(random_state).choice(minority_idx, size=deficit, replace=True)
        X_syn = X_tr.loc[sampled_idx]
        y_syn = y_tr.loc[sampled_idx]
        X_bal = pd.concat([X_tr, X_syn], axis=0).reset_index(drop=True)
        y_bal = pd.concat([y_tr, y_syn], axis=0).reset_index(drop=True)
        return X_bal, y_bal

    else:
        raise ValueError(f"Unsupported fold balancing strategy: {strategy}")


def compare_models_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    model_type: str = "random_forest",
    clf_params: Optional[Dict[str, Any]] = None,
    balancer_strategy: str = "tabrep",
    categorization: Optional[Any] = None,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    discrete_cols: Optional[List[str]] = None,
    diffusion_config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = "outputs/tables/statistical_significance.csv",
) -> pd.DataFrame:
    """
    Evaluates baseline vs balanced modeling across K stratified folds and computes
    paired t-tests and Wilcoxon signed-rank tests for rigorous thesis reporting.
    Supports 'random_forest', 'xgboost', and 'tabpfn'.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics_names = ["ROC-AUC", "F1 (High Performing)", "Recall (High Performing)", "Balanced Accuracy"]
    unbal_scores = {m: [] for m in metrics_names}
    bal_scores = {m: [] for m in metrics_names}

    model_key = model_type.lower()
    is_tabpfn = "tabpfn" in model_key
    is_xgb = "xgb" in model_key or "boost" in model_key

    if is_tabpfn:
        model_display_name = "TabPFNv3"
    elif is_xgb:
        model_display_name = "XGBoost"
    else:
        model_display_name = "Random Forest"

    logger.info(
        f"Running {n_splits}-fold paired cross-validation ({model_display_name}: "
        f"Unbalanced vs. Balanced [{balancer_strategy}])..."
    )

    if is_tabpfn:
        try:
            from tabpfn import TabPFNClassifier
        except ImportError:
            raise ImportError(
                "tabpfn package is not installed. To run TabPFNv3 in cross-validation, "
                "install tabpfn: pip install tabpfn"
            )
        import torch
        tabpfn_device = (
            clf_params.get("device")
            if clf_params and "device" in clf_params
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        tabpfn_estimators = clf_params.get("n_estimators", 4) if clf_params else 4
        logger.warning(
            "TabPFNv3 evaluation in multi-fold cross-validation involves in-context attention "
            "over large tabular contexts and requires substantial GPU compute time."
        )

    elif is_xgb:
        from xgboost import XGBClassifier
        default_params = {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        if clf_params:
            default_params.update(clf_params)
    else:
        default_params = {
            "n_estimators": 100,
            "max_depth": 12,
            "n_jobs": -1,
        }
        if clf_params:
            default_params.update(clf_params)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        fold_seed = random_state + fold

        # 1. Baseline Unbalanced Model
        if is_tabpfn:
            import random
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fold_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            clf_unbal = TabPFNClassifier(
                device=tabpfn_device,
                n_estimators=tabpfn_estimators,
                random_state=fold_seed,
            )
            clf_unbal.fit(X_tr.values.astype(np.float32), y_tr.values.astype(int))
            probs_unbal = clf_unbal.predict_proba(X_val.values.astype(np.float32))[:, 1]
            preds_unbal = clf_unbal.predict(X_val.values.astype(np.float32))
        else:
            fold_params = dict(default_params)
            fold_params["random_state"] = fold_seed
            if is_xgb:
                clf_unbal = XGBClassifier(**fold_params)
            else:
                clf_unbal = RandomForestClassifier(**fold_params)
            clf_unbal.fit(X_tr, y_tr)
            probs_unbal = clf_unbal.predict_proba(X_val)[:, 1]
            preds_unbal = clf_unbal.predict(X_val)

        # 2. Balanced Model
        X_tr_bal, y_tr_bal = balance_fold_data(
            X_tr,
            y_tr,
            strategy=balancer_strategy,
            random_state=fold_seed,
            categorization=categorization,
            cat_cols=cat_cols,
            num_cols=num_cols,
            discrete_cols=discrete_cols,
            diffusion_config=diffusion_config,
        )

        if is_tabpfn:
            import random
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fold_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            clf_bal = TabPFNClassifier(
                device=tabpfn_device,
                n_estimators=tabpfn_estimators,
                random_state=fold_seed,
            )
            clf_bal.fit(X_tr_bal.values.astype(np.float32), y_tr_bal.values.astype(int))
            probs_bal = clf_bal.predict_proba(X_val.values.astype(np.float32))[:, 1]
            preds_bal = clf_bal.predict(X_val.values.astype(np.float32))
        else:
            fold_params = dict(default_params)
            fold_params["random_state"] = random_state + fold
            if is_xgb:
                clf_bal = XGBClassifier(**fold_params)
            else:
                clf_bal = RandomForestClassifier(**fold_params)
            clf_bal.fit(X_tr_bal, y_tr_bal)
            probs_bal = clf_bal.predict_proba(X_val)[:, 1]
            preds_bal = clf_bal.predict(X_val)

        unbal_scores["ROC-AUC"].append(float(roc_auc_score(y_val, probs_unbal)))
        unbal_scores["F1 (High Performing)"].append(float(f1_score(y_val, preds_unbal, pos_label=1)))
        unbal_scores["Recall (High Performing)"].append(float(recall_score(y_val, preds_unbal, pos_label=1)))
        unbal_scores["Balanced Accuracy"].append(float(balanced_accuracy_score(y_val, preds_unbal)))

        bal_scores["ROC-AUC"].append(float(roc_auc_score(y_val, probs_bal)))
        bal_scores["F1 (High Performing)"].append(float(f1_score(y_val, preds_bal, pos_label=1)))
        bal_scores["Recall (High Performing)"].append(float(recall_score(y_val, preds_bal, pos_label=1)))
        bal_scores["Balanced Accuracy"].append(float(balanced_accuracy_score(y_val, preds_bal)))

    summary_rows = []
    for m in metrics_names:
        u_vals = np.array(unbal_scores[m])
        b_vals = np.array(bal_scores[m])
        diffs = b_vals - u_vals

        mean_u, std_u = float(np.mean(u_vals)), float(np.std(u_vals))
        mean_b, std_b = float(np.mean(b_vals)), float(np.std(b_vals))
        mean_diff = float(np.mean(diffs))

        try:
            t_res = stats.ttest_rel(b_vals, u_vals)
            t_stat = float(t_res.statistic)
            p_val_t = float(t_res.pvalue)
        except Exception:
            t_stat, p_val_t = np.nan, np.nan

        try:
            w_res = stats.wilcoxon(b_vals, u_vals)
            w_stat = float(w_res.statistic)
            p_val_w = float(w_res.pvalue)
        except Exception:
            w_stat, p_val_w = np.nan, np.nan

        summary_rows.append({
            "model": model_display_name,
            "metric": m,
            "mean_unbalanced": mean_u,
            "std_unbalanced": std_u,
            "mean_balanced": mean_b,
            "std_balanced": std_b,
            "mean_difference": mean_diff,
            "t_statistic": t_stat,
            "p_value_ttest": p_val_t,
            "wilcoxon_statistic": w_stat,
            "p_value_wilcoxon": p_val_w,
            "significant_ttest_05": bool(p_val_t < 0.05) if not np.isnan(p_val_t) else False,
            "significant_wilcoxon_05": bool(p_val_w < 0.05) if not np.isnan(p_val_w) else False,
        })

        logger.info(
            f"[{model_display_name} - {m}] Unbalanced: {mean_u:.4f} vs Balanced: {mean_b:.4f} "
            f"(diff: {mean_diff:+.4f}) | t-test p={p_val_t:.4f}, Wilcoxon p={p_val_w:.4f}"
        )

    res_df = pd.DataFrame(summary_rows)
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_file, index=False)
        logger.info(f"Saved complete paired statistical tests to: {out_file}")
    return res_df
