"""
Idempotent PISA 2022 dataset downloader and extractor.
"""

import os
import zipfile
import logging
import requests
from pathlib import Path
from typing import Tuple, Optional
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


def extract_zip(
    zip_path: Path,
    extract_to: Path,
    target_data_filename: str = "CY08MSP_STU_QQQ.SAS7BDAT",
    target_format_filename: str = "CY08MSP_STU_QQQ.FORMAT.SAS",
) -> Tuple[Path, Path]:
    """
    Extracts both SAS7BDAT dataset and SAS format definition files from archive.
    Ensures complete OECD PISA data and metadata availability.
    """
    logger.info(f"Inspecting archive: {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        namelist = zip_ref.namelist()
        
        # 1. Extract SAS7BDAT dataset
        data_matches = [name for name in namelist if name.upper().endswith(".SAS7BDAT")]
        if not data_matches:
            raise FileNotFoundError(f"No .SAS7BDAT file found in archive {zip_path}")
        
        extracted_data_file = data_matches[0]
        zip_ref.extract(extracted_data_file, path=extract_to)
        extracted_data_path = extract_to / extracted_data_file
        target_data_path = extract_to / target_data_filename
        if extracted_data_path != target_data_path and extracted_data_path.exists():
            extracted_data_path.replace(target_data_path)

        # 2. Extract FORMAT.SAS file
        format_matches = [name for name in namelist if name.upper().endswith(".FORMAT.SAS") or name.upper().endswith(".SAS")]
        if format_matches:
            # Prefer explicit .FORMAT.SAS if available
            exact_fmt = [m for m in format_matches if m.upper().endswith(".FORMAT.SAS")]
            chosen_fmt = exact_fmt[0] if exact_fmt else format_matches[0]
            zip_ref.extract(chosen_fmt, path=extract_to)
            extracted_fmt_path = extract_to / chosen_fmt
            target_fmt_path = extract_to / target_format_filename
            if extracted_fmt_path != target_fmt_path and extracted_fmt_path.exists():
                extracted_fmt_path.replace(target_fmt_path)
            logger.info(f"Extracted format file: {target_fmt_path} ({target_fmt_path.stat().st_size / 1024:.1f} KB)")
        else:
            logger.warning(f"No .FORMAT.SAS file found in archive {zip_path}")
            target_fmt_path = extract_to / target_format_filename

    logger.info(f"Extraction complete: {target_data_path} ({target_data_path.stat().st_size / (1024*1024):.1f} MB)")
    return target_data_path, target_fmt_path


def ensure_raw_dataset(
    raw_dir: str = "data/raw",
    sas_filename: str = "CY08MSP_STU_QQQ.SAS7BDAT",
    sas_format_filename: str = "CY08MSP_STU_QQQ.FORMAT.SAS",
    zip_filename: str = "STU_QQQ_SAS.zip",
    download_url: str = "https://webfs.oecd.org/pisa2022/STU_QQQ_SAS.zip",
) -> Path:
    """
    Ensure both the SAS dataset and SAS format definition are present locally.
    Idempotent: skips download/extraction if the SAS data and format files are already present.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    sas_path = raw_path / sas_filename
    format_path = raw_path / sas_format_filename
    zip_path = raw_path / zip_filename

    # Case 1: Both SAS data and format files already exist and have valid sizes
    data_valid = sas_path.exists() and sas_path.stat().st_size > 100 * 1024 * 1024
    format_valid = format_path.exists() and format_path.stat().st_size > 10 * 1024

    if data_valid and format_valid:
        logger.info(
            f"OECD PISA raw files verified: data={sas_path.name} ({sas_path.stat().st_size / (1024*1024):.1f} MB), "
            f"format={format_path.name} ({format_path.stat().st_size / 1024:.1f} KB). Skipping download."
        )
        return sas_path

    # Case 2: Zip archive already exists, extract both files
    if zip_path.exists() and zip_path.stat().st_size > 50 * 1024 * 1024:
        logger.info(f"Archive found at {zip_path}. Extracting data and format files...")
        extract_zip(zip_path, raw_path, sas_filename, sas_format_filename)
        return sas_path

    # Case 3: Download student zip and extract
    logger.info(f"Downloading PISA 2022 Student Questionnaire from official OECD source: {download_url}")
    download_file(download_url, zip_path)
    extract_zip(zip_path, raw_path, sas_filename, sas_format_filename)

    return sas_path


if __name__ == "__main__":
    ensure_raw_dataset()
