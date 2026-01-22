"""Schema helpers for conversion weight tables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


WEIGHT_COLUMNS = [
    "period",
    "from_vintage_year",
    "to_vintage_year",
    "from_code",
    "to_code",
    "group_id",
    "weight",
]

WEIGHT_DTYPES: Dict[str, str] = {
    "period": "string",
    "from_vintage_year": "string",
    "to_vintage_year": "string",
    "from_code": "string",
    "to_code": "string",
    "group_id": "string",
    "weight": "float64",
}

DEFAULT_WEIGHTS_DIR = Path("outputs/estimate/weights")


def empty_weight_table() -> pd.DataFrame:
    """Return an empty weight table with the canonical schema."""
    return pd.DataFrame({col: pd.Series(dtype=WEIGHT_DTYPES[col]) for col in WEIGHT_COLUMNS})


def validate_weight_table(
    df: pd.DataFrame,
    *,
    schema: str = "full",
    check_bounds: bool = True,
    check_row_sums: bool = False,
    row_sum_tol: float = 1e-6,
) -> None:
    """Validate the weight table schema and optional constraints."""
    if schema not in {"full", "minimal"}:
        raise ValueError("schema must be 'full' or 'minimal'")

    required_cols = WEIGHT_COLUMNS if schema == "full" else ["from_code", "to_code", "weight"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    required = df[required_cols]
    if required.isna().any().any():
        raise ValueError("Weight table contains nulls in required columns")

    if check_bounds:
        if (required["weight"] < 0).any() or (required["weight"] > 1).any():
            raise ValueError("Weights must be within [0, 1]")

    if check_row_sums:
        if schema == "full":
            key_cols = ["period", "from_vintage_year", "to_vintage_year", "from_code"]
        else:
            key_cols = ["from_code"]
        row_sums = required.groupby(key_cols, dropna=False)["weight"].sum()
        if ((row_sums - 1.0).abs() > row_sum_tol).any():
            raise ValueError("Weight rows must sum to 1 within tolerance")
