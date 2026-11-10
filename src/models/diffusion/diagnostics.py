"""
Automated Tabular Distribution Audit and Diagnostics Module.
Computes statistical fidelity metrics:
- Marginal statistics (Mean, Std, Min, Max, Quantiles) for numerical features
- Kolmogorov-Smirnov (KS) statistic & 1-Wasserstein distance for numerical features
- Total Variation Distance (TVD) for nominal categorical features (kept strictly separate from KS)
- Empirical support boundary check (noting that 0.00% OOB on continuous features is structurally
  enforced by QuantileTransformer.inverse_transform() quantile clipping)
- Pairwise Pearson correlation structure preservation strictly on numerical features
- Standardized Euclidean nearest-neighbor distance for memorization auditing
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_distribution_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute per-feature and multivariate distribution diagnostics, keeping continuous/discrete
    numerical metrics (KS, 1-Wasserstein, Quantile RMSE) strictly separated from nominal
    categorical Total Variation Distance (TVD).
    """
    if feature_names is None:
        feature_names = [c for c in real_df.columns if c in synth_df.columns and c not in ["target", "is_synthetic"]]

    per_feature_records = []

    for col in feature_names:
        r_vals = real_df[col].dropna().values
        s_vals = synth_df[col].dropna().values

        if np.issubdtype(r_vals.dtype, np.number) and np.issubdtype(s_vals.dtype, np.number):
            r_min, r_max = float(np.min(r_vals)), float(np.max(r_vals))
            s_min, s_max = float(np.min(s_vals)), float(np.max(s_vals))
            r_mean, s_mean = float(np.mean(r_vals)), float(np.mean(s_vals))
            r_std, s_std = float(np.std(r_vals)), float(np.std(s_vals))

            tol = 1e-5
            oob = float(np.mean((s_vals < r_min - tol) | (s_vals > r_max + tol)) * 100.0)

            ks_res = ks_2samp(r_vals, s_vals)
            ks_stat = float(ks_res.statistic)
            ks_pval = float(ks_res.pvalue)

            wd = float(wasserstein_distance(r_vals, s_vals))

            q_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
            r_quantiles = np.quantile(r_vals, q_levels)
            s_quantiles = np.quantile(s_vals, q_levels)
            q_rmse = float(np.sqrt(np.mean((r_quantiles - s_quantiles) ** 2)))

            per_feature_records.append({
                "feature": col,
                "feature_type": "numerical",
                "real_mean": r_mean,
                "synth_mean": s_mean,
                "mean_diff": abs(r_mean - s_mean),
                "real_std": r_std,
                "synth_std": s_std,
                "std_diff": abs(r_std - s_std),
                "real_min": r_min,
                "synth_min": s_min,
                "real_max": r_max,
                "synth_max": s_max,
                "oob_pct": oob,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pval,
                "wasserstein": wd,
                "quantile_rmse": q_rmse,
                "tv_dist": np.nan,
            })
        else:
            r_counts = pd.Series(r_vals).value_counts(normalize=True)
            s_counts = pd.Series(s_vals).value_counts(normalize=True)
            all_cats = list(set(r_counts.index).union(set(s_counts.index)))
            tv_dist = 0.5 * sum(abs(r_counts.get(c, 0.0) - s_counts.get(c, 0.0)) for c in all_cats)
            per_feature_records.append({
                "feature": col,
                "feature_type": "nominal",
                "real_mean": np.nan,
                "synth_mean": np.nan,
                "mean_diff": np.nan,
                "real_std": np.nan,
                "synth_std": np.nan,
                "std_diff": np.nan,
                "real_min": np.nan,
                "synth_min": np.nan,
                "real_max": np.nan,
                "synth_max": np.nan,
                "oob_pct": np.nan,
                "ks_stat": np.nan,
                "ks_pvalue": np.nan,
                "wasserstein": np.nan,
                "quantile_rmse": np.nan,
                "tv_dist": float(tv_dist),
            })

    feature_metrics_df = pd.DataFrame(per_feature_records)

    # Pairwise Pearson correlation difference computed strictly on numerical features
    num_cols = [
        c for c in feature_names
        if np.issubdtype(real_df[c].dtype, np.number) and np.issubdtype(synth_df[c].dtype, np.number)
    ]
    if len(num_cols) > 1:
        r_corr = real_df[num_cols].corr().fillna(0.0).values
        s_corr = synth_df[num_cols].corr().fillna(0.0).values
        corr_frobenius = float(np.linalg.norm(r_corr - s_corr, "fro") / len(num_cols))
        corr_mae = float(np.mean(np.abs(r_corr - s_corr)))
    else:
        corr_frobenius = 0.0
        corr_mae = 0.0

    # Standardized nearest-neighbor Euclidean distance for memorization audit
    if len(num_cols) > 0:
        sub_size = min(len(real_df), len(synth_df), 500)
        scaler = StandardScaler()
        scaler.fit(real_df[num_cols].fillna(0.0).values)
        sub_real = scaler.transform(real_df[num_cols].fillna(0.0).sample(n=sub_size, random_state=42).values)
        sub_synth = scaler.transform(synth_df[num_cols].fillna(0.0).sample(n=sub_size, random_state=42).values)
        dists = cdist(sub_synth, sub_real, metric="euclidean")
        min_dists = np.min(dists, axis=1)
        exact_copy_rate = float(np.mean(min_dists < 1e-5) * 100.0)
        avg_min_distance = float(np.mean(min_dists))
    else:
        exact_copy_rate = 0.0
        avg_min_distance = 0.0

    ks_series = feature_metrics_df["ks_stat"].dropna()
    tvd_series = feature_metrics_df["tv_dist"].dropna()
    wd_series = feature_metrics_df["wasserstein"].dropna()
    oob_series = feature_metrics_df["oob_pct"].dropna()

    mean_ks_num = float(ks_series.mean()) if len(ks_series) > 0 else 0.0
    median_ks_num = float(ks_series.median()) if len(ks_series) > 0 else 0.0
    mean_tvd_nom = float(tvd_series.mean()) if len(tvd_series) > 0 else 0.0
    median_tvd_nom = float(tvd_series.median()) if len(tvd_series) > 0 else 0.0

    summary_dict = {
        "mean_ks_numerical": mean_ks_num,
        "median_ks_numerical": median_ks_num,
        "mean_tvd_nominal": mean_tvd_nom,
        "median_tvd_nominal": median_tvd_nom,
        "mean_ks_stat": mean_ks_num,
        "median_ks_stat": median_ks_num,
        "mean_wasserstein": float(wd_series.mean()) if len(wd_series) > 0 else 0.0,
        "mean_oob_pct": float(oob_series.mean()) if len(oob_series) > 0 else 0.0,
        "max_oob_pct": float(oob_series.max()) if len(oob_series) > 0 else 0.0,
        "mean_mean_diff": float(feature_metrics_df["mean_diff"].dropna().mean()) if len(ks_series) > 0 else 0.0,
        "mean_std_diff": float(feature_metrics_df["std_diff"].dropna().mean()) if len(ks_series) > 0 else 0.0,
        "corr_matrix_mae": corr_mae,
        "corr_matrix_frobenius": corr_frobenius,
        "exact_copy_rate_pct": exact_copy_rate,
        "avg_min_distance_to_real": avg_min_distance,
    }

    return feature_metrics_df, summary_dict


def plot_distribution_audit(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    feature_metrics_df: pd.DataFrame,
    output_path: str = "outputs/figures/tabrep_distribution_audit.png",
    key_features: Optional[List[str]] = None,
):
    """
    Generate and save a publication-quality distribution audit figure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    num_metrics_df = feature_metrics_df[feature_metrics_df["ks_stat"].notna()].copy()

    if key_features is None:
        default_candidates = ["ESCS", "HOMEPOS", "HISEI", "BMMJ1"]
        key_features = [f for f in default_candidates if f in real_df.columns and f in synth_df.columns]
        if len(key_features) < 4:
            available_feats = num_metrics_df.sort_values(by="ks_stat")["feature"].tolist()
            for feat in available_feats:
                if feat not in key_features and feat in real_df.columns and feat in synth_df.columns:
                    key_features.append(feat)
                if len(key_features) >= 4:
                    break
    key_features = [f for f in key_features if f in real_df.columns and f in synth_df.columns]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32)

    palette = {"Real Class 1": "#2b5c8f", "Synthetic TabRep": "#d95f02"}
    for idx, feat in enumerate(key_features[:4]):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        sns.kdeplot(real_df[feat], label="Real Class 1", ax=ax, color=palette["Real Class 1"], lw=2.2, fill=True, alpha=0.15)
        sns.kdeplot(synth_df[feat], label="Synthetic TabRep", ax=ax, color=palette["Synthetic TabRep"], lw=2.2, fill=True, alpha=0.15)
        ax.set_title(f"Marginal Density: {feat}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.5)

    ax_ks = fig.add_subplot(gs[0:2, 2])
    ks_vals = num_metrics_df["ks_stat"].dropna()
    sns.histplot(ks_vals, kde=True, ax=ax_ks, color="#7570b3", bins=18)
    mean_ks = float(ks_vals.mean()) if len(ks_vals) > 0 else 0.0
    ax_ks.axvline(mean_ks, color="red", linestyle="--", lw=2, label=f"Mean Numerical KS = {mean_ks:.3f}")
    if "tv_dist" in feature_metrics_df.columns and feature_metrics_df["tv_dist"].notna().any():
        mean_tvd = float(feature_metrics_df["tv_dist"].dropna().mean())
        ax_ks.axvline(mean_tvd, color="#d95f02", linestyle=":", lw=2, label=f"Mean Nominal TVD = {mean_tvd:.3f} (n={int(feature_metrics_df['tv_dist'].notna().sum())})")
    ax_ks.set_title(f"Numerical KS Distribution (n = {len(ks_vals)} Features)", fontsize=12, fontweight="bold")
    ax_ks.set_xlabel("Kolmogorov-Smirnov Statistic", fontsize=10)
    ax_ks.set_ylabel("Feature Count", fontsize=10)
    ax_ks.legend(loc="upper right", frameon=True)
    ax_ks.grid(True, linestyle="--", alpha=0.5)

    ax_qq = fig.add_subplot(gs[2, 0])
    q_vals = np.linspace(0.01, 0.99, 100)
    r_q = np.quantile(real_df["ESCS"].dropna(), q_vals)
    s_q = np.quantile(synth_df["ESCS"].dropna(), q_vals)
    ax_qq.scatter(r_q, s_q, color="#1b9e77", alpha=0.7, edgecolors="none", s=30)
    min_val = min(r_q.min(), s_q.min())
    max_val = max(r_q.max(), s_q.max())
    ax_qq.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Identity (x = y)")
    ax_qq.set_title("Q-Q Plot: ESCS", fontsize=12, fontweight="bold")
    ax_qq.set_xlabel("Real Quantiles", fontsize=10)
    ax_qq.set_ylabel("Synthetic Quantiles", fontsize=10)
    ax_qq.legend(loc="upper left", frameon=True)
    ax_qq.grid(True, linestyle="--", alpha=0.5)

    ax_oob = fig.add_subplot(gs[2, 1])
    oob_vals = num_metrics_df["oob_pct"].dropna()
    sns.histplot(oob_vals, ax=ax_oob, color="#e7298a", bins=20)
    mean_oob = float(oob_vals.mean()) if len(oob_vals) > 0 else 0.0
    ax_oob.axvline(mean_oob, color="black", linestyle="--", lw=1.8, label=f"Mean OOB = {mean_oob:.1f}% (Quantile Clipped)")
    ax_oob.set_title("Numerical Support Bounds (Quantile-Clipped)", fontsize=12, fontweight="bold")
    ax_oob.set_xlabel("Percentage of Samples Outside [Min, Max]", fontsize=10)
    ax_oob.set_ylabel("Feature Count", fontsize=10)
    ax_oob.legend(loc="upper right", frameon=True)
    ax_oob.grid(True, linestyle="--", alpha=0.5)

    ax_corr = fig.add_subplot(gs[2, 2])
    cand_feats = num_metrics_df.sort_values(by="ks_stat")["feature"].tolist()
    sub_feats = [f for f in cand_feats if np.issubdtype(real_df[f].dtype, np.number) and np.issubdtype(synth_df[f].dtype, np.number)][:10]
    if len(sub_feats) >= 2:
        r_c = real_df[sub_feats].corr().fillna(0.0).values
        s_c = synth_df[sub_feats].corr().fillna(0.0).values
        diff_df = pd.DataFrame(np.abs(r_c - s_c), index=sub_feats, columns=sub_feats)
    else:
        diff_df = pd.DataFrame(np.zeros((1, 1)))
    sns.heatmap(
        diff_df,
        ax=ax_corr,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.25,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "|Real - Synth| Corr"},
    )
    ax_corr.set_title("Correlation Absolute Error (Top 10 Numerical)", fontsize=12, fontweight="bold")
    ax_corr.tick_params(axis="x", rotation=45, labelsize=8)
    ax_corr.tick_params(axis="y", rotation=0, labelsize=8)

    plt.suptitle("TabRep Generative Pipeline Quality Audit", fontsize=16, fontweight="bold", y=0.99)
    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.07, right=0.97, hspace=0.38, wspace=0.30)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved distribution audit figure to: {output_path}")


def assert_synthetic_validity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    max_mean_oob: float = 12.0,
    max_mean_ks: float = 0.35,
):
    """
    Quality gate: Validates that synthetic data adheres to statistical bounds.
    Raises ValueError if generation failed or produced pathological values.
    """
    # 1. Null and Infinite checks
    if synth_df.isna().sum().sum() > 0:
        raise ValueError("Synthetic generation produced NaN values!")
    if np.isinf(synth_df.select_dtypes(include=[np.number]).values).sum() > 0:
        raise ValueError("Synthetic generation produced Infinite values!")

    feature_metrics_df, summary = compute_distribution_metrics(real_df, synth_df)

    mean_oob = summary["mean_oob_pct"]
    mean_ks = summary["mean_ks_stat"]

    logger.info(
        f"Synthetic Quality Gate Check -> Mean OOB: {mean_oob:.2f}% (Threshold: {max_mean_oob}%), "
        f"Mean KS: {mean_ks:.4f} (Threshold: {max_mean_ks})"
    )

    if mean_oob > max_mean_oob:
        raise ValueError(
            f"Quality Gate Failed: Mean Out-of-Bounds rate ({mean_oob:.2f}%) exceeds tolerance ({max_mean_oob}%). "
            f"Possible distribution collapse or decoder failure."
        )

    if mean_ks > max_mean_ks:
        raise ValueError(
            f"Quality Gate Failed: Mean Kolmogorov-Smirnov statistic ({mean_ks:.4f}) exceeds tolerance ({max_mean_ks}). "
            f"Generated data does not adequately represent training distribution."
        )

    logger.info("Synthetic data PASSED all statistical quality gates.")
