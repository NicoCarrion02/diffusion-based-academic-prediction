"""
Metadata-driven PISA survey code cleaner and feature categorizer.
Cleans declared OECD non-response / missing codes per variable using the official SAS format dictionary
and categorizes features strictly according to the metadata schema.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Union, Optional
import numpy as np
import pandas as pd

from src.data.metadata_explorer import PISAMetadataExplorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class FeatureCategorization:
    nominal_cols: List[str]
    ordinal_discrete_cols: List[str]
    continuous_cols: List[str]
    binary_cols: List[str] = field(default_factory=list)

    def __iter__(self):
        return iter((self.nominal_cols, self.ordinal_discrete_cols, self.continuous_cols))

    def __getitem__(self, key: Union[int, str]):
        if isinstance(key, str):
            return getattr(self, key)
        return (self.nominal_cols, self.ordinal_discrete_cols, self.continuous_cols)[key]

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    @property
    def numerical_cols(self) -> List[str]:
        return self.continuous_cols + self.ordinal_discrete_cols

    @property
    def all_features(self) -> List[str]:
        return self.nominal_cols + self.numerical_cols

    def to_dict(self):
        return {
            "nominal_cols": self.nominal_cols,
            "ordinal_discrete_cols": self.ordinal_discrete_cols,
            "binary_cols": self.binary_cols,
            "continuous_cols": self.continuous_cols,
        }


def clean_pisa_survey_codes(
    df: pd.DataFrame,
    metadata_explorer: Optional[PISAMetadataExplorer] = None,
) -> pd.DataFrame:
    """
    Cleans survey data using the official OECD metadata schema:
    1. Maps format-declared missing/non-response codes to NaN.
    2. Formats nominal features uniformly as categorical tokens.
    """
    if metadata_explorer is None:
        metadata_explorer = PISAMetadataExplorer()

    df_clean = df.copy()
    cleaned_count = 0

    schema = metadata_explorer.get_schema_categorization(df_clean.columns.tolist())
    invalid_cols = schema.get("other_cols", [])
    if invalid_cols:
        raise ValueError(
            f"Dataset contains unmapped or non-analytical columns according to metadata: {invalid_cols}"
        )
    nominal_set = set(schema["nominal_cols"])

    for col in df_clean.columns:
        if col in ("target", "is_synthetic"):
            continue

        missing_codes = metadata_explorer.get_missing_codes(col)
        if missing_codes:
            mask = df_clean[col].isin(missing_codes)
            n_mod = mask.sum()
            if n_mod > 0:
                df_clean.loc[mask, col] = np.nan
                cleaned_count += n_mod

        if col in nominal_set:
            df_clean[col] = df_clean[col].apply(
                lambda x: f"CAT_{int(x)}" if pd.notna(x) and isinstance(x, (int, float, np.number)) and not np.isnan(x)
                else (str(x).strip() if pd.notna(x) and not (isinstance(x, float) and np.isnan(x)) else "MISSING")
            )

    logger.info(f"Metadata-driven cleaning complete. Sanitized {cleaned_count} format-declared missing codes to NaN.")
    return df_clean


def categorize_pisa_features(
    df: pd.DataFrame,
    metadata_explorer: Optional[PISAMetadataExplorer] = None,
) -> FeatureCategorization:
    """
    Categorizes features according to the PISA SAS metadata schema:
    - 'nominal_cols': Unordered categorical variables.
    - 'ordinal_discrete_cols': Ordered survey questions / Likert items / discrete quantities.
    - 'binary_cols': 2-level indicator questions.
    - 'continuous_cols': Continuous IRT latent scores and indices.
    """
    if metadata_explorer is None:
        metadata_explorer = PISAMetadataExplorer()

    cols = [c for c in df.columns if c not in ("target", "is_synthetic")]
    schema_dict = metadata_explorer.get_schema_categorization(cols)

    invalid_cols = schema_dict.get("other_cols", [])
    if invalid_cols:
        raise ValueError(
            f"Features contain unmapped or non-analytical columns according to metadata: {invalid_cols}"
        )

    combined_discrete = list(schema_dict["ordinal_discrete_cols"])
    for b in schema_dict["binary_cols"]:
        if b not in combined_discrete:
            combined_discrete.append(b)

    logger.info(
        f"Feature categorization complete: {len(schema_dict['nominal_cols'])} nominal, "
        f"{len(combined_discrete)} discrete ({len(schema_dict['binary_cols'])} binary), "
        f"{len(schema_dict['continuous_cols'])} continuous."
    )

    return FeatureCategorization(
        nominal_cols=schema_dict["nominal_cols"],
        ordinal_discrete_cols=combined_discrete,
        continuous_cols=schema_dict["continuous_cols"],
        binary_cols=schema_dict["binary_cols"],
    )
