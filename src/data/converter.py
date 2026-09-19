"""
SAS7BDAT to Parquet converter with encoding normalization and idempotent caching.
"""

import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_sas_to_parquet(
    sas_path: str = "data/raw/CY08MSP_STU_QQQ.SAS7BDAT",
    parquet_path: str = "data/interim/pisa2022.parquet",
    force: bool = False,
) -> Path:
    """
    Convert a SAS7BDAT file to compressed Parquet.
    Idempotent: if parquet already exists and force is False, returns existing path.
    """
    out_file = Path(parquet_path)
    in_file = Path(sas_path)

    if out_file.exists() and out_file.stat().st_size > 10 * 1024 * 1024 and not force:
        logger.info(f"Parquet file already exists: {out_file} ({out_file.stat().st_size / (1024*1024):.1f} MB). Skipping conversion.")
        return out_file

    if not in_file.exists():
        raise FileNotFoundError(f"SAS file not found at: {in_file}")

    logger.info(f"Reading SAS file: {in_file} (this may take a few minutes)...")
    df = pd.read_sas(str(in_file), format="sas7bdat", encoding="latin1")
    logger.info(f"Loaded DataFrame with shape: {df.shape}")

    # Ensure byte strings are properly decoded to python str
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(lambda x: x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing compressed Parquet to: {out_file}...")
    df.to_parquet(out_file, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"Conversion complete: {out_file} ({out_file.stat().st_size / (1024*1024):.1f} MB)")

    return out_file


if __name__ == "__main__":
    convert_sas_to_parquet()
