"""
PISA data loader, Argentina filtering, target binarization, and stratified split.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import pyarrow.dataset as ds
from sklearn.model_selection import train_test_split

from src.data.metadata_explorer import PISAMetadataExplorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_train_test_data(
    interim_parquet: str = "data/interim/pisa2022.parquet",
    train_out: str = "data/processed/train.parquet",
    test_out: str = "data/processed/test.parquet",
    country_code: str = "ARG",
    target_col: str = "PV1MATH",
    threshold: float = 420.07,
    test_size: float = 0.20,
    random_state: int = 42,
    force: bool = False,
    metadata_explorer: Optional[PISAMetadataExplorer] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters PISA dataset for specified country, constructs binary target,
    removes leakage columns, and performs stratified train/test split.
    Idempotent: returns existing train/test if already processed and not force.
    """
    train_path = Path(train_out)
    test_path = Path(test_out)

    if train_path.exists() and test_path.exists() and not force:
        logger.info(f"Loading cached train/test splits from {train_path.parent}...")
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
        return train_df, test_df

    parquet_file = Path(interim_parquet)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Interim parquet file not found at: {parquet_file}")

    logger.info(f"Reading {country_code} subset from {parquet_file}...")
    dataset = ds.dataset(str(parquet_file), format="parquet")
    country_filter = ds.field("CNT") == country_code
    table = dataset.to_table(filter=country_filter)
    df = table.to_pandas()
    logger.info(f"Filtered {country_code} data: {df.shape[0]} rows, {df.shape[1]} columns.")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns.")

    valid_mask = df[target_col].notnull()
    if not valid_mask.all():
        dropped = (~valid_mask).sum()
        logger.warning(f"Dropping {dropped} rows with missing {target_col}.")
        df = df[valid_mask].copy()

    df["target"] = (df[target_col] >= threshold).astype(int)

    counts = df["target"].value_counts().to_dict()
    proportions = df["target"].value_counts(normalize=True).to_dict()
    logger.info(
        f"Target distribution (Threshold = {threshold}): "
        f"Class 0 (Low): {counts.get(0, 0)} ({proportions.get(0, 0.0)*100:.2f}%), "
        f"Class 1 (High): {counts.get(1, 0)} ({proportions.get(1, 0.0)*100:.2f}%)"
    )

    # Drop non-analytical columns according to metadata (PVs, survey weights, administrative IDs)
    if metadata_explorer is None:
        metadata_explorer = PISAMetadataExplorer()

    candidate_cols = [c for c in df.columns if c != "target"]
    schema = metadata_explorer.get_schema_categorization(candidate_cols)
    cols_to_drop = [c for c in schema["other_cols"] if c in df.columns]

    logger.info(f"Removing {len(cols_to_drop)} non-analytical and leakage columns according to metadata...")
    df.drop(columns=cols_to_drop, inplace=True)

    logger.info(f"Performing stratified split ({int((1-test_size)*100)}% train / {int(test_size*100)}% test, seed={random_state})...")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["target"],
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    logger.info(f"Saved train set ({train_df.shape}) to {train_path}")
    logger.info(f"Saved test set ({test_df.shape}) to {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    prepare_train_test_data()
