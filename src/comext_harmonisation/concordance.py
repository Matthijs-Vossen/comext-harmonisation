"""Parse and normalize Eurostat CN concordance tables."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


_PERIOD_LEN = 8
_CODE_LEN = 8


@dataclass(frozen=True)
class ConcordancePeriod:
    period: str
    origin_year: str
    dest_year: str


def _normalize_period(value: object) -> ConcordancePeriod:
    if pd.isna(value):
        raise ValueError("Period is missing")
    period = str(value).strip()
    if period.endswith(".0"):
        period = period[:-2]
    if len(period) != _PERIOD_LEN or not period.isdigit():
        raise ValueError(f"Invalid period '{value}'; expected 8-digit YYYYYYYY")
    origin_year = period[:4]
    dest_year = period[4:]
    return ConcordancePeriod(period=period, origin_year=origin_year, dest_year=dest_year)


def _normalize_code(value: object) -> str:
    if pd.isna(value):
        raise ValueError("Code is missing")
    if isinstance(value, (int,)):
        code = str(value)
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Non-integer code '{value}'")
        code = str(int(value))
    else:
        code = str(value).strip()
        if code.endswith(".0") and re.fullmatch(r"\d+\.0", code):
            code = code[:-2]
    if not code.isdigit():
        raise ValueError(f"Invalid code '{value}'; expected digits only")
    if len(code) > _CODE_LEN:
        raise ValueError(f"Invalid code '{value}'; expected <= 8 digits")
    return code.zfill(_CODE_LEN)


def parse_concordance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a concordance dataframe to canonical columns.

    Expected input columns: 'Period', 'Origin code', 'Destination code'.
    Returns a dataframe with: period, origin_year, dest_year, origin_code, dest_code.
    """
    required = {"Period", "Origin code", "Destination code"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    normalized_rows = []
    for row in df[["Period", "Origin code", "Destination code"]].itertuples(
        index=False, name=None
    ):
        period_raw, origin_raw, dest_raw = row
        period = _normalize_period(period_raw)
        origin_code = _normalize_code(origin_raw)
        dest_code = _normalize_code(dest_raw)
        normalized_rows.append(
            {
                "period": period.period,
                "origin_year": period.origin_year,
                "dest_year": period.dest_year,
                "origin_code": origin_code,
                "dest_code": dest_code,
            }
        )

    normalized = pd.DataFrame(normalized_rows)
    if normalized.empty:
        return normalized

    normalized = normalized.drop_duplicates(
        subset=["period", "origin_code", "dest_code"], keep="first"
    ).reset_index(drop=True)
    return normalized


def read_concordance_xls(path: str, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Read and normalize the official CN concordance XLS file."""
    if sheet_name is None:
        # Use first sheet to avoid a hard dependency on a specific name.
        sheet_name = 0
    raw = pd.read_excel(path, sheet_name=sheet_name, engine="xlrd")
    return parse_concordance_df(raw)
