"""
TabPFNv3 classifier wrapper for tabular academic prediction.
Utilizes in-context learning foundation model for tabular data.
"""

import os
import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch

# Load .env variables if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_tabpfn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    device: Optional[str] = None,
    n_estimators: int = 4,
    random_state: int = 42,
) -> Tuple[Optional[object], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Train TabPFNv3 classifier and predict on test data.
    Requires free academic registration token (TABPFN_TOKEN) in .env file or environment.
    """
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        logger.warning(
            "tabpfn package is not installed. TabPFNv3 stage will be skipped. "
            "To install, run: pip install tabpfn"
        )
        return None, None, None

    token = os.environ.get("TABPFN_TOKEN", "").strip()
    if not token or token == "your_tabpfn_api_key_here":
        logger.info(
            "TabPFNv3 skipped: TABPFN_TOKEN not configured. "
            "To activate TabPFN (100% free for research), create an account at https://ux.priorlabs.ai, "
            "copy your API key to a .env file (see .env.example) and re-run."
        )
        return None, None, None

    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    import random
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Initializing TabPFNClassifier (n_estimators={n_estimators}) on device: {device}...")
    try:
        tabpfn = TabPFNClassifier(device=device, n_estimators=n_estimators, random_state=random_state)
        X_tr = X_train.values.astype(np.float32)
        y_tr = y_train.values.astype(int)
        X_te = X_test.values.astype(np.float32)

        logger.info(f"Fitting TabPFN on {len(X_tr)} samples with {X_tr.shape[1]} features...")
        tabpfn.fit(X_tr, y_tr)

        y_pred = tabpfn.predict(X_te)
        y_prob = tabpfn.predict_proba(X_te)[:, 1]
        logger.info("TabPFN inference complete.")
        return tabpfn, y_pred, y_prob

    except torch.cuda.OutOfMemoryError:
        logger.error(
            "TabPFN encountered CUDA Out-of-Memory. In-context transformer attention over 12,000 samples "
            "exceeded local GPU VRAM. Consider running TabPFN on a server GPU with >= 16 GB VRAM."
        )
        torch.cuda.empty_cache()
        return None, None, None
    except Exception as e:
        logger.error(f"TabPFN execution error: {e}. If model weights fail to download, verify TABPFN_TOKEN.")
        return None, None, None
