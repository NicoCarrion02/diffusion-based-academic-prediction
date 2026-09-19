"""
Idempotent PISA 2022 dataset downloader and extractor.
"""

import os
import zipfile
import logging
import requests
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with progress tracking and safe atomic renaming."""
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Connecting to OECD data source: {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    logger.info(f"Downloading {dest_path.name} ({total_size / (1024 * 1024):.1f} MB)...")

    with open(temp_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=dest_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    if temp_path.exists():
        temp_path.replace(dest_path)
    logger.info(f"Successfully downloaded to {dest_path}")


def extract_zip(zip_path: Path, extract_to: Path, target_filename: str) -> Path:
    """Extract target file from zip archive if not already extracted."""
    logger.info(f"Extracting {target_filename} from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        namelist = zip_ref.namelist()
        matched = [name for name in namelist if name.upper().endswith(".SAS7BDAT")]
        if not matched:
            raise FileNotFoundError(f"No .SAS7BDAT file found in archive {zip_path}")
        
        extracted_file = matched[0]
        zip_ref.extract(extracted_file, path=extract_to)
        extracted_path = extract_to / extracted_file
        
        target_path = extract_to / target_filename
        if extracted_path != target_path and extracted_path.exists():
            extracted_path.replace(target_path)
            
    logger.info(f"Extraction complete: {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
    return target_path


def ensure_raw_dataset(
    raw_dir: str = "data/raw",
    sas_filename: str = "CY08MSP_STU_QQQ.SAS7BDAT",
    zip_filename: str = "STU_QQQ_SAS.zip",
    download_url: str = "https://webfs.oecd.org/pisa2022/STU_QQQ_SAS.zip",
    fallback_url: str = "https://webfs.oecd.org/pisa2022/SCH_QQQ_SAS.zip",
) -> Path:
    """
    Ensure the SAS dataset is present locally.
    Idempotent: skips download/extraction if the SAS file or valid zip is already present.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    sas_path = raw_path / sas_filename
    zip_path = raw_path / zip_filename

    # Case 1: SAS file already exists
    if sas_path.exists() and sas_path.stat().st_size > 100 * 1024 * 1024:
        logger.info(f"SAS dataset already present: {sas_path} ({sas_path.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
        return sas_path

    # Case 2: Zip archive already exists, extract it
    if zip_path.exists() and zip_path.stat().st_size > 50 * 1024 * 1024:
        logger.info(f"Archive found at {zip_path}. Extracting...")
        return extract_zip(zip_path, raw_path, sas_filename)

    # Case 3: Download zip and extract
    try:
        download_file(download_url, zip_path)
    except Exception as e:
        logger.warning(f"Primary download URL failed: {e}. Trying fallback URL: {fallback_url}")
        download_file(fallback_url, zip_path)

    return extract_zip(zip_path, raw_path, sas_filename)


if __name__ == "__main__":
    ensure_raw_dataset()
