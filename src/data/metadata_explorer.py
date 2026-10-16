"""
PISA 2022 Metadata Explorer & Dictionary Parser.
Integrates CY08MSP_STU_QQQ.SAS7BDAT column descriptors with CY08MSP_STU_QQQ.FORMAT.SAS
value labels and formatting rules to extract formal variable types and codebooks.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import pyreadstat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ORDINAL_KEYWORDS = [
    'agree', 'disagree', 'never', 'sometimes', 'often', 'always',
    'rarely', 'almost', 'less than', 'more than', 'at least',
    'none', 'low', 'medium', 'high', 'poor', 'fair', 'good', 'excellent',
    'level', 'stage', 'frequency', 'times a week', 'times a month',
    'strongly', 'not at all', 'to some extent', 'to a great extent',
    'completely', 'very', 'extremely', 'satisfied', 'dissatisfied',
    'day', 'days', 'year', 'years', 'grade', 'age', 'month', 'months',
    'hour', 'hours', 'week', 'weeks', 'time', 'times', 'lesson', 'lessons',
    'isced', 'younger', 'older', 'ladder', 'struggle', 'effort', 'easily',
    'once', 'twice'
]

ADMIN_VARS = {
    'CNTSCHID', 'CNTSTUID', 'CYC', 'STRATUM', 'SUBNATIO', 'NatCen', 'VER_DAT',
    'CNT', 'CNTRYID', 'BOOKID', 'REGION', 'LANGTEST_QQQ', 'LANGTEST_COG', 'LANGTEST_PAQ',
    'WVARSTRR', 'UNIT', 'ADMINMODE'
}

KNOWN_NOMINALS = {
    'LANGN', 'PROGN', 'OCOD1', 'OCOD2', 'OCOD3', 'COBN_S', 'COBN_M', 'COBN_F',
    'ST250D06JA', 'ST250D07JA', 'ST251D08JA', 'ST251D09JA', 'ST330D10WA'
}

MISSING_TERMS = ['missing', 'valid skip', 'not applicable', 'invalid', 'no response', 'not reached']


class PISAMetadataExplorer:
    """
    Parses and explores official OECD PISA metadata directly from SAS files.
    """

    def __init__(
        self,
        sas_data_path: str = "data/raw/CY08MSP_STU_QQQ.SAS7BDAT",
        sas_format_path: str = "data/raw/CY08MSP_STU_QQQ.FORMAT.SAS",
        cache_path: str = "data/interim/pisa_metadata_dictionary.json"
    ):
        self.sas_data_path = sas_data_path
        self.sas_format_path = sas_format_path
        self.cache_path = cache_path
        self.dictionary: Dict[str, Dict[str, Any]] = {}

    def parse_formats(self) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        """Parses PROC FORMAT value definitions and FORMAT variable mappings."""
        if not os.path.exists(self.sas_format_path):
            raise FileNotFoundError(f"Format file not found at: {self.sas_format_path}")

        logger.info(f"Parsing SAS format definitions from {self.sas_format_path}...")
        with open(self.sas_format_path, "r", encoding="latin1") as f:
            text = f.read()

        # 1. Parse value definitions: value <format_name> ... ; (support "value $ FORMAT" syntax)
        val_pattern = re.compile(r"value\s+(\$?\s*\w+)(.*?);", re.DOTALL | re.IGNORECASE)
        format_defs: Dict[str, Dict[str, Any]] = {}

        for m in val_pattern.finditer(text):
            fmt_name = m.group(1).upper().replace(" ", "")
            body = m.group(2).strip()

            categories: Dict[str, str] = {}
            missing_codes: Dict[str, str] = {}

            for line in body.split("\n"):
                if "=" in line:
                    parts = line.split("=", 1)
                    raw_k = parts[0].strip().strip("'").strip('"')
                    raw_v = parts[1].strip().strip('"').strip("'")

                    if any(term in raw_v.lower() for term in MISSING_TERMS):
                        missing_codes[raw_k] = raw_v
                    else:
                        categories[raw_k] = raw_v

            format_defs[fmt_name] = {
                "categories": categories,
                "missing_codes": missing_codes,
                "is_missing_only": len(categories) == 0
            }

        # 2. Parse variable-to-format assignments: format var1 fmt1. var2 fmt2. ... ;
        var_to_format: Dict[str, str] = {}
        fmt_assign_match = re.search(r"\bformat\s*\n(.*?);", text, re.DOTALL | re.IGNORECASE)
        if fmt_assign_match:
            for line in fmt_assign_match.group(1).strip().split("\n"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    var_name = parts[0].upper()
                    fmt_ref = parts[1].rstrip(".").upper().replace(" ", "")
                    var_to_format[var_name] = fmt_ref

        logger.info(f"Parsed {len(format_defs)} format definitions and {len(var_to_format)} variable mappings.")
        return format_defs, var_to_format

    def build_dictionary(self, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Builds the unified dictionary combining SAS7BDAT metadata and SAS format definitions.
        Uses cached JSON if available to avoid re-reading 4GB SAS header.
        """
        if not force_reload and os.path.exists(self.cache_path):
            logger.info(f"Loading metadata dictionary from cache: {self.cache_path}")
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.dictionary = json.load(f)
            return self.dictionary

        format_defs, var_to_format = self.parse_formats()

        logger.info(f"Reading SAS7BDAT column metadata from {self.sas_data_path}...")
        _, meta = pyreadstat.read_sas7bdat(self.sas_data_path, metadataonly=True)

        dictionary = {}

        for col in meta.column_names:
            col_upper = col.upper()
            label = meta.column_names_to_labels.get(col, "")
            raw_type = meta.readstat_variable_types.get(col, "unknown")
            storage_width = meta.variable_storage_width.get(col, 0)

            fmt_name = var_to_format.get(col_upper)
            clean_fmt = fmt_name.lstrip("$") if fmt_name else None

            fdef = None
            if fmt_name:
                fdef = format_defs.get(fmt_name) or format_defs.get(clean_fmt) or format_defs.get("$" + clean_fmt)

            # Determine official variable type
            if col_upper.startswith("PV"):
                var_type = "plausible_value"
            elif col_upper.startswith("W_") or "SENWT" in col_upper:
                var_type = "survey_weight"
            elif col_upper in ADMIN_VARS:
                var_type = "identifier_admin"
            elif col_upper in KNOWN_NOMINALS:
                var_type = "nominal"
            elif col_upper == "GRADE":
                var_type = "ordinal"
            elif fdef is not None and fdef["is_missing_only"]:
                # Formats like TEACHSUPF, STUBMIF, WB156Q01HAF that only mark missing codes for continuous scales
                var_type = "continuous_scale"
            elif fdef is not None and len(fdef["categories"]) > 0:
                # Categorical with defined substantive categories
                cats = fdef["categories"]
                n_cats = len(cats)
                if n_cats == 2:
                    var_type = "binary"
                else:
                    cats_lower = " ".join(cats.values()).lower()
                    if any(kw in cats_lower for kw in ORDINAL_KEYWORDS):
                        var_type = "ordinal"
                    elif all(c.strip().isdigit() for c in cats.values()):
                        var_type = "ordinal"
                    else:
                        var_type = "nominal"
            else:
                logger.warning(
                    f"Variable '{col}' (format={fmt_name}, storage={raw_type}) cannot be deterministically mapped. "
                    f"Classifying as 'unknown'."
                )
                var_type = "unknown"

            dictionary[col] = {
                "name": col,
                "label": label,
                "variable_type": var_type,
                "storage_type": raw_type,
                "storage_width": storage_width,
                "format_name": fmt_name,
                "categories": fdef["categories"] if fdef else {},
                "missing_codes": fdef["missing_codes"] if fdef else {},
            }

        self.dictionary = dictionary

        # Save to cache
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(dictionary, f, indent=2, ensure_ascii=False)
        logger.info(f"Cached metadata dictionary to {self.cache_path} ({len(dictionary)} variables).")

        return self.dictionary

    def get_variable(self, var_name: str) -> Optional[Dict[str, Any]]:
        """Returns metadata info for a specific variable."""
        if not self.dictionary:
            self.build_dictionary()
        return self.dictionary.get(var_name) or self.dictionary.get(var_name.upper())

    def search(self, query: str) -> pd.DataFrame:
        """Search variables by keyword across variable name and question label."""
        if not self.dictionary:
            self.build_dictionary()

        q = query.lower()
        matches = []
        for v in self.dictionary.values():
            if q in v["name"].lower() or q in v["label"].lower():
                matches.append({
                    "name": v["name"],
                    "variable_type": v["variable_type"],
                    "label": v["label"],
                    "format": v["format_name"],
                    "num_categories": len(v["categories"]),
                })
        return pd.DataFrame(matches)

    def get_columns_by_type(self, var_type: str) -> List[str]:
        """Returns all column names of a specific type (e.g. 'ordinal', 'continuous_scale', etc.)."""
        if not self.dictionary:
            self.build_dictionary()
        return [name for name, v in self.dictionary.items() if v["variable_type"] == var_type]

    def to_dataframe(self) -> pd.DataFrame:
        """Exports the entire metadata dictionary as a Pandas DataFrame."""
        if not self.dictionary:
            self.build_dictionary()
        rows = []
        for v in self.dictionary.values():
            rows.append({
                "name": v["name"],
                "variable_type": v["variable_type"],
                "label": v["label"],
                "storage_type": v["storage_type"],
                "format_name": v["format_name"],
                "num_categories": len(v["categories"]),
                "categories_preview": str(list(v["categories"].items())[:3]) if v["categories"] else "",
                "missing_codes": str(list(v["missing_codes"].items())) if v["missing_codes"] else "",
            })
        return pd.DataFrame(rows)

    def get_missing_codes(self, var_name: str) -> set:
        """Returns set of numeric missing codes declared in the SAS format for this variable."""
        var_info = self.get_variable(var_name)
        if not var_info or not var_info.get("missing_codes"):
            return set()
        codes = set()
        for k in var_info["missing_codes"].keys():
            clean_k = str(k).strip("'\"")
            codes.add(clean_k)
            try:
                f_val = float(clean_k)
                codes.add(f_val)
                if f_val.is_integer():
                    codes.add(int(f_val))
            except ValueError:
                pass
        return codes

    def get_schema_categorization(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Categorizes a list of feature column names into:
        - nominal_cols
        - ordinal_discrete_cols
        - binary_cols
        - continuous_cols
        """
        if not self.dictionary:
            self.build_dictionary()

        nominal_cols = []
        ordinal_discrete_cols = []
        binary_cols = []
        continuous_cols = []

        for col in columns:
            if col in ("target", "is_synthetic"):
                continue
            v = self.dictionary.get(col) or self.dictionary.get(col.upper())
            if v is None:
                continuous_cols.append(col)
                continue
            v_type = v.get("variable_type", "continuous_scale")
            if v_type == "nominal":
                nominal_cols.append(col)
            elif v_type == "ordinal":
                ordinal_discrete_cols.append(col)
            elif v_type == "binary":
                binary_cols.append(col)
            else:
                continuous_cols.append(col)

        return {
            "nominal_cols": nominal_cols,
            "ordinal_discrete_cols": ordinal_discrete_cols,
            "binary_cols": binary_cols,
            "continuous_cols": continuous_cols,
        }


if __name__ == "__main__":
    explorer = PISAMetadataExplorer()
    df_meta = explorer.build_dictionary()
    print("\nSummary of Variable Types:")
    df_table = explorer.to_dataframe()
    print(df_table["variable_type"].value_counts())
