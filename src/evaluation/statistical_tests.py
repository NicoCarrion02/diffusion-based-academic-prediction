"""
Statistical inference for classifier and generative-balancing comparisons.

Methodological Design:
1. Cross-Validation Inference (Nadeau & Bengio, 2003; Bengio & Grandvalet, 2004):
   K-fold cross-validation scores on a single dataset are correlated because training folds
   share (K - 2) / (K - 1) of student records (75% overlap at K=5).
   Naive paired Student's t-tests underestimate the standard error of the mean difference
   and inflate Type I error rates, while Wilcoxon signed-rank tests assume independent
   paired units and have a discrete two-sided floor p_min = 2 / 2^5 = 0.0625 > 0.05 at K=5.
   Accordingly, CV inference uses the Nadeau-Bengio variance-corrected resampled t-test:
       SE_NB = sqrt((1 / J + n_val / n_train) * s_d^2),   df = J - 1.

2. Independent Held-Out Test Inference (Dietterich, 1998):
   On the vaulted test cohort (N_test = 2,423 non-overlapping students), student-level
   predictions are conditionally independent given the fitted models.
   We report Exact McNemar's binomial test on discordant binary correctness vectors
   (evaluating Overall Accuracy on the full test set and Minority Recall on the Class 1 subset)
   alongside paired stratified bootstrap 95% confidence intervals (B = 2,000 resamples).
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    balanced_accuracy_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def nadeau_bengio_ttest(
    scores_bal: np.ndarray,
    scores_unbal: np.ndarray,
    n_train: int,
    n_val: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Compute the Nadeau & Bengio (2003) variance-corrected resampled t-test for K-fold CV.
    Accounts for positive inter-fold correlation induced by overlapping training splits.
    """
    b_vals = np.asarray(scores_bal, dtype=np.float64)
    u_vals = np.asarray(scores_unbal, dtype=np.float64)
    if len(b_vals) != len(u_vals) or len(b_vals) < 2:
        raise ValueError("nadeau_bengio_ttest requires equal-length score arrays with J >= 2 splits.")

    J = len(b_vals)
    diffs = b_vals - u_vals
    mean_diff = float(np.mean(diffs))
    s2_d = float(np.var(diffs, ddof=1))
    overlap_factor = (1.0 / J) + (float(n_val) / max(float(n_train), 1.0))
    se_nb = float(np.sqrt(overlap_factor * s2_d))
    df = J - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))

    if se_nb <= 1e-15:
        if abs(mean_diff) <= 1e-15:
            t_nb, p_nb = 0.0, 1.0
        else:
            t_nb = float(np.sign(mean_diff) * np.inf)
            p_nb = 0.0
        ci_low, ci_high = mean_diff, mean_diff
    else:
        t_nb = float(mean_diff / se_nb)
        p_nb = float(2.0 * stats.t.sf(abs(t_nb), df=df))
        ci_low = float(mean_diff - t_crit * se_nb)
        ci_high = float(mean_diff + t_crit * se_nb)

    # Descriptive uncorrected reference statistics across overlapping folds
    try:
        if s2_d <= 1e-15:
            t_naive = 0.0 if abs(mean_diff) <= 1e-15 else float(np.sign(mean_diff) * np.inf)
            p_naive = 1.0 if abs(mean_diff) <= 1e-15 else 0.0
        else:
            t_naive_res = stats.ttest_rel(b_vals, u_vals)
            t_naive = float(t_naive_res.statistic)
            p_naive = float(t_naive_res.pvalue)
    except Exception:
        t_naive, p_naive = np.nan, np.nan

    try:
        if np.allclose(diffs, 0.0):
            w_stat, p_wilcoxon = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(b_vals, u_vals)
            w_stat = float(w_res.statistic)
            p_wilcoxon = float(w_res.pvalue)
    except Exception:
        w_stat, p_wilcoxon = np.nan, np.nan

    return {
        "n_splits": int(J),
        "n_train_per_fold": int(n_train),
        "n_val_per_fold": int(n_val),
        "mean_difference": mean_diff,
        "std_difference": float(np.sqrt(s2_d)),
        "se_nadeau_bengio": se_nb,
        "ci95_low_nb": ci_low,
        "ci95_high_nb": ci_high,
        "t_statistic_nadeau_bengio": t_nb,
        "p_value_nadeau_bengio": p_nb,
        "significant_nadeau_bengio_05": bool(p_nb < alpha) if not np.isnan(p_nb) else False,
        "folds_improved": int(np.sum(diffs > 0)),
        # Uncorrected descriptive reference statistics
        "t_statistic": t_naive,
        "p_value_ttest": p_naive,
        "wilcoxon_statistic": w_stat,
        "p_value_wilcoxon": p_wilcoxon,
    }


def compute_paired_cv_inference_table(
    model_display_name: str,
    unbal_scores: Dict[str, List[float]],
    bal_scores: Dict[str, List[float]],
    n_train: int,
    n_val: int,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a paired CV statistical table from fold-level metric dictionaries
    using the Nadeau-Bengio variance-corrected resampled t-test.
    """
    summary_rows = []
    for metric_name, u_list in unbal_scores.items():
        if metric_name not in bal_scores:
            continue
        u_vals = np.asarray(u_list, dtype=np.float64)
        b_vals = np.asarray(bal_scores[metric_name], dtype=np.float64)
        ddof = 1 if len(u_vals) > 1 else 0

        nb = nadeau_bengio_ttest(b_vals, u_vals, n_train=n_train, n_val=n_val)
        J = nb["n_splits"]
        note = (
            f"Primary inference: Nadeau-Bengio corrected t-test (n_val/n_tr={n_val}/{n_train}). "
            f"Naive t-test/Wilcoxon ignore {int(round(100*(J-2)/max(J-1,1)))}% fold overlap"
            + ("; two-sided Wilcoxon min p=0.0625 at K=5." if J == 5 else ".")
        )

        row = {
            "model": model_display_name,
            "metric": metric_name,
            "n_splits": J,
            "mean_unbalanced": float(np.mean(u_vals)),
            "std_unbalanced": float(np.std(u_vals, ddof=ddof)),
            "mean_balanced": float(np.mean(b_vals)),
            "std_balanced": float(np.std(b_vals, ddof=ddof)),
            "mean_difference": nb["mean_difference"],
            "ci95_low_nb": nb["ci95_low_nb"],
            "ci95_high_nb": nb["ci95_high_nb"],
            "folds_improved": nb["folds_improved"],
            "t_statistic_nadeau_bengio": nb["t_statistic_nadeau_bengio"],
            "p_value_nadeau_bengio": nb["p_value_nadeau_bengio"],
            "significant_nadeau_bengio_05": nb["significant_nadeau_bengio_05"],
            "t_statistic": nb["t_statistic"],
            "p_value_ttest": nb["p_value_ttest"],
            "wilcoxon_statistic": nb["wilcoxon_statistic"],
            "p_value_wilcoxon": nb["p_value_wilcoxon"],
            "inference_note": note,
        }
        summary_rows.append(row)

        logger.info(
            f"[{model_display_name} - {metric_name}] Unbal: {row['mean_unbalanced']:.4f} vs "
            f"Bal: {row['mean_balanced']:.4f} (diff: {row['mean_difference']:+.4f}, "
            f"95% NB CI [{row['ci95_low_nb']:+.4f}, {row['ci95_high_nb']:+.4f}]) | "
            f"Nadeau-Bengio p={row['p_value_nadeau_bengio']:.4f} "
            f"(uncorrected naive p={row['p_value_ttest']:.4f})"
        )

    res_df = pd.DataFrame(summary_rows)
    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(out_file, index=False)
        logger.info(f"Saved Nadeau-Bengio paired CV inference table to: {out_file}")
    return res_df


def _exact_mcnemar_pvalue(correct_bal: np.ndarray, correct_unbal: np.ndarray) -> Tuple[int, int, float]:
    """Exact two-sided binomial McNemar test on discordant binary correctness vectors."""
    b = int(np.sum(correct_bal & (~correct_unbal)))
    c = int(np.sum((~correct_bal) & correct_unbal))
    n_disc = b + c
    if n_disc == 0:
        return 0, 0, 1.0
    p_val = float(stats.binomtest(b, n=n_disc, p=0.5, alternative="two-sided").pvalue)
    return b, c, p_val


def compute_holdout_statistical_tests(
    y_true: np.ndarray,
    probs_unbal: np.ndarray,
    preds_unbal: np.ndarray,
    probs_bal: np.ndarray,
    preds_bal: np.ndarray,
    model_name: str = "Model",
    n_bootstrap: int = 2000,
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Independent-sample statistical inference on the held-out test set:
    1. Paired Stratified Bootstrap 95% CIs (preserving exact Class 0 / Class 1 sample sizes)
       across F1 (Class 1), Recall (Class 1), Balanced Accuracy, Overall Accuracy, PR-AUC, and ROC-AUC.
    2. Exact McNemar binomial test paired strictly with the binary correctness metric it evaluates:
       - Class 1 subset (N_1 = 802) on 'Recall (High Performing)' (minority sensitivity).
       - Full test set (N = 2,423) on 'Overall Accuracy' (overall classification correctness).
    """
    y_arr = np.asarray(y_true, dtype=int)
    pu_arr = np.asarray(probs_unbal, dtype=np.float64)
    hu_arr = np.asarray(preds_unbal, dtype=int)
    pb_arr = np.asarray(probs_bal, dtype=np.float64)
    hb_arr = np.asarray(preds_bal, dtype=int)

    idx_0 = np.where(y_arr == 0)[0]
    idx_1 = np.where(y_arr == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    rng = np.random.RandomState(random_state)

    metric_fns = {
        "F1 (High Performing)": lambda y, pr, pd_: f1_score(y, pd_, pos_label=1, zero_division=0),
        "Recall (High Performing)": lambda y, pr, pd_: recall_score(y, pd_, pos_label=1, zero_division=0),
        "Balanced Accuracy": lambda y, pr, pd_: balanced_accuracy_score(y, pd_),
        "Overall Accuracy": lambda y, pr, pd_: accuracy_score(y, pd_),
        "PR-AUC": lambda y, pr, pd_: average_precision_score(y, pr),
        "ROC-AUC": lambda y, pr, pd_: roc_auc_score(y, pr),
    }

    point_u = {m: float(fn(y_arr, pu_arr, hu_arr)) for m, fn in metric_fns.items()}
    point_b = {m: float(fn(y_arr, pb_arr, hb_arr)) for m, fn in metric_fns.items()}
    boot_diffs = {m: np.empty(n_bootstrap, dtype=np.float64) for m in metric_fns}

    for b_idx in range(n_bootstrap):
        s0 = rng.choice(idx_0, size=n0, replace=True)
        s1 = rng.choice(idx_1, size=n1, replace=True)
        boot_idx = np.concatenate([s0, s1])
        yb = y_arr[boot_idx]
        pu_b, hu_b = pu_arr[boot_idx], hu_arr[boot_idx]
        pb_b, hb_b = pb_arr[boot_idx], hb_arr[boot_idx]
        for m, fn in metric_fns.items():
            boot_diffs[m][b_idx] = fn(yb, pb_b, hb_b) - fn(yb, pu_b, hu_b)

    # Exact McNemar overall (tests Overall Accuracy) and on minority Class 1 (tests Recall)
    corr_u_all = (hu_arr == y_arr)
    corr_b_all = (hb_arr == y_arr)
    mc_b_all, mc_c_all, mc_p_all = _exact_mcnemar_pvalue(corr_b_all, corr_u_all)

    corr_u_c1 = (hu_arr[idx_1] == 1)
    corr_b_c1 = (hb_arr[idx_1] == 1)
    mc_b_c1, mc_c_c1, mc_p_c1 = _exact_mcnemar_pvalue(corr_b_c1, corr_u_c1)

    rows = []
    for m in metric_fns:
        d_arr = boot_diffs[m]
        ci_low, ci_high = np.percentile(d_arr, [2.5, 97.5])
        p_pos = np.mean(d_arr <= 0.0)
        p_neg = np.mean(d_arr >= 0.0)
        p_boot = float(min(1.0, 2.0 * min(p_pos, p_neg)))

        if m == "Recall (High Performing)":
            mc_b, mc_c, mc_p = mc_b_c1, mc_c_c1, mc_p_c1
            mc_scope = f"Class 1 subset (N={n1}) — Minority Recall"
        elif m == "Overall Accuracy":
            mc_b, mc_c, mc_p = mc_b_all, mc_c_all, mc_p_all
            mc_scope = f"Full test set (N={n0 + n1}) — Overall Accuracy"
        elif m in ("F1 (High Performing)", "Balanced Accuracy"):
            mc_b, mc_c, mc_p = np.nan, np.nan, np.nan
            mc_scope = "Composite metric (Bootstrap CI)"
        else:
            mc_b, mc_c, mc_p = np.nan, np.nan, np.nan
            mc_scope = "Threshold-free curve metric (Bootstrap CI)"

        rows.append({
            "model": model_name,
            "metric": m,
            "n_test": int(n0 + n1),
            "unbalanced": point_u[m],
            "balanced": point_b[m],
            "difference": point_b[m] - point_u[m],
            "bootstrap_ci95_low": float(ci_low),
            "bootstrap_ci95_high": float(ci_high),
            "bootstrap_p_two_sided": p_boot,
            "mcnemar_bal_only_correct": mc_b,
            "mcnemar_unbal_only_correct": mc_c,
            "mcnemar_exact_pvalue": mc_p,
            "mcnemar_scope": mc_scope,
        })

    out_df = pd.DataFrame(rows)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(p, index=False)
        logger.info(f"Saved held-out statistical test results to: {p}")
    return out_df


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
    device: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balances a single training fold to a 1:1 ratio strictly using X_tr[y_tr == 1].
    Supports 'tabrep' (Official TabRep-Flow) and 'random_oversample'.
    """
    n_0 = int((y_tr == 0).sum())
    n_1 = int((y_tr == 1).sum())
    deficit = n_0 - n_1
    if deficit <= 0:
        return X_tr.copy(), y_tr.copy()

    strat = strategy.lower()
    if strat == "tabrep":
        from src.models.diffusion.balancer import DatasetBalancer

        cfg = diffusion_config or {}
        net_cfg = cfg.get("network", {}) if "network" in cfg else cfg
        tr_cfg = cfg.get("training", {}) if "training" in cfg else cfg

        balancer = DatasetBalancer(
            strategy="tabrep",
            epochs=tr_cfg.get("cv_epochs", tr_cfg.get("epochs", 500)),
            batch_size=tr_cfg.get("batch_size", 512),
            lr=tr_cfg.get("learning_rate", 0.001),
            weight_decay=tr_cfg.get("weight_decay", 5e-4),
            timesteps=tr_cfg.get("timesteps", 50),
            hidden_dims=net_cfg.get("hidden_dims", [1024, 2048, 2048, 1024]),
            dim_t=net_cfg.get("dim_t", 1024),
            dropout=net_cfg.get("dropout", 0.0),
            activation=net_cfg.get("activation", "relu"),
            architecture=net_cfg.get("architecture", "tabrep_mlp"),
            snap_discrete_levels=cfg.get("snap_discrete_levels", True),
            lr_scheduler_factor=tr_cfg.get("lr_scheduler_factor", 0.9),
            lr_scheduler_patience=tr_cfg.get("lr_scheduler_patience", 30),
            solver=tr_cfg.get("solver", "heun"),
            device=device,
            random_state=random_state,
        )
        fold_train = X_tr.copy()
        fold_train["target"] = y_tr.values

        if categorization is None and cat_cols is None and num_cols is None:
            num_cols = [c for c in X_tr.columns if np.issubdtype(X_tr[c].dtype, np.number)]
            cat_cols = [c for c in X_tr.columns if c not in num_cols]

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
        y_bal = balanced_fold["target"].astype(int)
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
