"""
TabPFN v3 / v3.5 classifier wrapper for tabular academic prediction.
Utilizes PriorLabs' prior-fitted in-context learning foundation model for tabular data,
including single-pass raw-logit extraction for computationally tractable inference calibration.
"""

import inspect
import logging
import os
import random
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PLACEHOLDER_TOKENS = {
    "",
    "your_tabpfn_api_key_here",
    "your_hf_token_here",
    "your_api_key_here",
}


def has_tabpfn_auth_token() -> bool:
    """Check whether TABPFN_TOKEN or HF_TOKEN is configured in the environment or .env."""
    for env_key in ("TABPFN_TOKEN", "HF_TOKEN"):
        val = os.environ.get(env_key, "").strip()
        if val and val not in PLACEHOLDER_TOKENS:
            return True
    return False


def is_tabpfn_available(require_token: bool = True) -> bool:
    """Verify whether the tabpfn package is installed and (optionally) authenticated."""
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        return False
    return has_tabpfn_auth_token() if require_token else True


def resolve_tabpfn_device(device: Optional[str] = None) -> str:
    """Resolve 'auto' / None device safely across GPU and CPU workstations."""
    if device is None or str(device).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        logger.warning("TabPFN configured with CUDA, but CUDA is unavailable. Falling back to CPU.")
        return "cpu"
    return str(device)


def build_tabpfn_classifier(
    device: Optional[str] = "auto",
    n_estimators: int = 4,
    softmax_temperature: Any = 0.9,
    average_before_softmax: bool = False,
    balance_probabilities: bool = False,
    ignore_pretraining_limits: bool = True,
    memory_saving_mode: Any = "auto",
    random_state: int = 42,
    **kwargs,
):
    """
    Safely construct a TabPFNClassifier validated against the installed package signature.
    Unsupported legacy config keys (e.g., 'model_version', 'thinking_mode') are ignored cleanly.
    """
    from tabpfn import TabPFNClassifier

    resolved_device = resolve_tabpfn_device(device)
    sig = inspect.signature(TabPFNClassifier.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}

    candidate_kwargs: Dict[str, Any] = {
        "device": resolved_device,
        "n_estimators": n_estimators,
        "softmax_temperature": softmax_temperature,
        "average_before_softmax": average_before_softmax,
        "balance_probabilities": balance_probabilities,
        "ignore_pretraining_limits": ignore_pretraining_limits,
        "memory_saving_mode": memory_saving_mode,
        "random_state": random_state,
    }
    for k, v in kwargs.items():
        if k in accepted:
            candidate_kwargs[k] = v
        else:
            logger.debug(f"Ignoring non-TabPFNClassifier parameter '{k}={v}'.")

    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    return TabPFNClassifier(**filtered_kwargs)


def predict_tabpfn_raw_logits(clf: Any, X: np.ndarray) -> np.ndarray:
    """
    Compute per-estimator raw logits once (shape: [n_estimators, n_samples, n_classes])
    so post-hoc inference parameters (n_estimators subsetting, softmax_temperature,
    average_before_softmax) can be tuned in milliseconds without repeating Transformer passes.
    """
    X_arr = np.asarray(X, dtype=np.float32)
    if hasattr(clf, "predict_raw_logits"):
        raw = clf.predict_raw_logits(X_arr)
        return np.asarray(raw, dtype=np.float32)

    # Fallback if an older tabpfn version without predict_raw_logits is installed
    probs = clf.predict_proba(X_arr)
    eps = 1e-7
    clipped = np.clip(probs, eps, 1.0 - eps)
    log_p = np.log(clipped)[None, :, :]
    return log_p.astype(np.float32)


def tabpfn_logits_to_probs(
    clf: Any,
    raw_logits: np.ndarray,
    n_estimators: Optional[int] = None,
    softmax_temperature: float = 0.9,
    average_before_softmax: bool = False,
    balance_probabilities: bool = False,
) -> np.ndarray:
    """
    Convert cached per-estimator raw logits into Class 1 probabilities under specified
    ensemble size, temperature scaling, and softmax aggregation order.
    """
    logits_sub = raw_logits[:n_estimators] if (n_estimators is not None and raw_logits.ndim == 3) else raw_logits

    if hasattr(clf, "logits_to_probabilities"):
        probs_t = clf.logits_to_probabilities(
            logits_sub,
            softmax_temperature=float(softmax_temperature),
            average_before_softmax=bool(average_before_softmax),
            balance_probabilities=bool(balance_probabilities),
        )
        probs_np = probs_t.detach().cpu().numpy() if isinstance(probs_t, torch.Tensor) else np.asarray(probs_t)
        return probs_np[:, 1].astype(np.float64)

    # Pure NumPy fallback matching TabPFN post-processing semantics
    scaled = logits_sub / float(softmax_temperature)
    if scaled.ndim == 3:
        if average_before_softmax:
            avg_logits = np.mean(scaled, axis=0)
            exp_l = np.exp(avg_logits - np.max(avg_logits, axis=-1, keepdims=True))
            probs_np = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
        else:
            exp_l = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
            est_probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            probs_np = np.mean(est_probs, axis=0)
    else:
        exp_l = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs_np = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    return probs_np[:, 1].astype(np.float64)


def train_tabpfn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    device: Optional[str] = "auto",
    n_estimators: int = 4,
    softmax_temperature: Any = 0.9,
    average_before_softmax: bool = False,
    balance_probabilities: bool = False,
    ignore_pretraining_limits: bool = True,
    memory_saving_mode: Any = "auto",
    random_state: int = 42,
    **kwargs,
) -> Tuple[Optional[object], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit TabPFN in-context classifier on X_train and predict on X_test.
    Checks TABPFN_TOKEN and HF_TOKEN in .env / environment; skips cleanly with a warning if absent.
    """
    try:
        from tabpfn import TabPFNClassifier  # noqa: F401
    except ImportError:
        logger.warning(
            "tabpfn package is not installed. TabPFNv3 stage will be skipped. "
            "To install, run: pip install tabpfn"
        )
        return None, None, None

    if not has_tabpfn_auth_token():
        logger.warning(
            "TabPFNv3 skipped: neither TABPFN_TOKEN nor HF_TOKEN is configured. "
            "Continuing with Random Forest and XGBoost. To activate TabPFN (free for academic research), "
            "set TABPFN_TOKEN or HF_TOKEN in .env (see .env.example)."
        )
        return None, None, None

    resolved_device = resolve_tabpfn_device(device)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(
        f"Initializing TabPFNClassifier (n_estimators={n_estimators}, temp={softmax_temperature}, "
        f"avg_before_softmax={average_before_softmax}) on device: {resolved_device}..."
    )
    try:
        tabpfn = build_tabpfn_classifier(
            device=resolved_device,
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            average_before_softmax=average_before_softmax,
            balance_probabilities=balance_probabilities,
            ignore_pretraining_limits=ignore_pretraining_limits,
            memory_saving_mode=memory_saving_mode,
            random_state=random_state,
            **kwargs,
        )
        X_tr = X_train.values.astype(np.float32) if hasattr(X_train, "values") else np.asarray(X_train, dtype=np.float32)
        y_tr = y_train.values.astype(int) if hasattr(y_train, "values") else np.asarray(y_train, dtype=int)
        X_te = X_test.values.astype(np.float32) if hasattr(X_test, "values") else np.asarray(X_test, dtype=np.float32)

        logger.info(f"Fitting TabPFN on {len(X_tr)} samples with {X_tr.shape[1]} features...")
        tabpfn.fit(X_tr, y_tr)

        y_prob = tabpfn.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        logger.info("TabPFN inference complete.")
        return tabpfn, y_pred, y_prob

    except torch.cuda.OutOfMemoryError:
        logger.error(
            "TabPFN encountered CUDA Out-of-Memory. Reduce n_estimators or run on a server GPU with >= 16 GB VRAM."
        )
        torch.cuda.empty_cache()
        return None, None, None
    except Exception as e:
        logger.error(f"TabPFN execution error: {e}. Verify TABPFN_TOKEN / HF_TOKEN and cached weights.")
        return None, None, None
