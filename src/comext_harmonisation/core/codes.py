"""Shared code/year normalization helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def normalize_codes(series: pd.Series) -> pd.Series:
    """Normalize commodity codes to trimmed strings, left-padding numeric codes to CN8 width."""
    codes = series.astype(str).str.strip().str.replace(" ", "", regex=False)
    mask = codes.str.isdigit()
    return codes.where(~mask, codes.str.zfill(8))


def normalize_code_set(codes: Iterable[str]) -> set[str]:
    values = list(codes)
    if not values:
        return set()
    normalized = normalize_codes(pd.Series(values))
    return set(normalized.tolist())


def normalize_year(year: str | int) -> str:
    year_str = str(year)
    if len(year_str) != 4 or not year_str.isdigit():
        raise ValueError(f"Invalid year '{year}'; expected 4-digit year")
    return year_str


def chain_periods(origin_year: str | int, target_year: str | int) -> tuple[list[str], str]:
    origin = int(origin_year)
    target = int(target_year)
    if origin == target:
        return [], "identity"
    if origin < target:
        return [f"{year}{year + 1}" for year in range(origin, target)], "a_to_b"
    return [f"{year}{year + 1}" for year in range(origin - 1, target - 1, -1)], "b_to_a"

