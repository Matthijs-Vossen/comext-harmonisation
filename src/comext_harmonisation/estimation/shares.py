"""Prepare annual trade shares for LT weight estimation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple

import pandas as pd

from ..groups import ConcordanceGroups
from ..mappings import get_ambiguous_group_summary


ANNUAL_DATA_DIR = Path("data/extracted_annual_no_confidential/products_like")

EXCLUDE_CODES_DEFAULT: Set[str] = {
    "QP",
    "QQ",
    "QR",
    "QS",
    "QU",
    "QV",
    "QW",
    "QX",
    "QY",
    "QZ",
}

AGGREGATE_CODES: Set[str] = {"XA", "XO", "XR", "XZ"}


@dataclass(frozen=True)
class EstimationShares:
    period: str
    direction: str
    measure: str
    vintage_a_year: str
    vintage_b_year: str
    shares_a: pd.DataFrame
    shares_b: pd.DataFrame
    group_totals: pd.DataFrame
    skipped_groups: pd.DataFrame


def _split_period(period: str) -> Tuple[str, str]:
    period = str(period)
    if len(period) != 8 or not period.isdigit():
        raise ValueError(f"Invalid period '{period}'; expected 8-digit YYYYYYYY")
    return period[:4], period[4:]


def _normalize_measure(measure: str) -> str:
    measure = str(measure).strip().upper()
    if measure not in {"VALUE_EUR", "QUANTITY_KG"}:
        raise ValueError("measure must be 'VALUE_EUR' or 'QUANTITY_KG'")
    return measure


def _measure_column(measure: str) -> str:
    return measure.lower()


def _load_annual_year(base_dir: Path, year: str, measure: str) -> pd.DataFrame:
    path = base_dir / f"comext_{year}.parquet"
    cols = ["REPORTER", "PARTNER", "TRADE_TYPE", "PRODUCT_NC", "FLOW", measure]
    return pd.read_parquet(path, columns=cols)


def _apply_exclusions(
    df: pd.DataFrame,
    *,
    exclude_codes: Set[str],
) -> pd.DataFrame:
    if not exclude_codes:
        return df
    mask = ~df["REPORTER"].isin(exclude_codes) & ~df["PARTNER"].isin(exclude_codes)
    return df.loc[mask]


def _prepare_side_shares(
    df: pd.DataFrame,
    *,
    period: str,
    vintage_year: str,
    group_map: pd.DataFrame,
    group_ids: Set[str],
    code_col_name: str,
    flow: str,
    exclude_codes: Set[str],
    measure: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"REPORTER", "PARTNER", "TRADE_TYPE", "PRODUCT_NC", "FLOW", measure}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    value_col = _measure_column(measure)
    df = df.copy()
    df["FLOW"] = df["FLOW"].astype(str)
    df = df[df["FLOW"] == str(flow)]
    df = _apply_exclusions(df, exclude_codes=exclude_codes)

    df = df.rename(columns={"PRODUCT_NC": code_col_name, measure: value_col})

    df = df.merge(group_map, on=code_col_name, how="inner")
    df = df[df["group_id"].isin(group_ids)]

    df = (
        df.groupby(["REPORTER", "PARTNER", "group_id", code_col_name], as_index=False)
        .agg(**{value_col: (value_col, "sum")})
    )

    if df.empty:
        shares = df.copy()
        shares["share"] = pd.Series(dtype="float64")
        totals = pd.DataFrame(
            columns=["period", "group_id", f"total_{value_col}", "n_rows", "n_pairs"]
        )
        return shares, totals

    totals = (
        df.groupby("group_id", as_index=False)
        .agg(**{f"total_{value_col}": (value_col, "sum")})
    )
    pairs = (
        df.drop_duplicates(["group_id", "REPORTER", "PARTNER"])
        .groupby("group_id", as_index=False)
        .size()
        .rename(columns={"size": "n_pairs"})
    )
    n_rows = (
        df.groupby("group_id", as_index=False)
        .size()
        .rename(columns={"size": "n_rows"})
    )

    totals = totals.merge(pairs, on="group_id", how="left").merge(n_rows, on="group_id", how="left")
    totals.insert(0, "period", period)

    df = df.merge(totals[["group_id", f"total_{value_col}"]], on="group_id", how="left")
    df["share"] = df[value_col] / df[f"total_{value_col}"]
    df.insert(0, "period", period)
    df.insert(1, "vintage_year", vintage_year)

    return df, totals


def prepare_estimation_shares_from_frames(
    *,
    period: str,
    groups: ConcordanceGroups,
    direction: str = "a_to_b",
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    measure: str = "VALUE_EUR",
    flow: str = "1",
    exclude_codes: Optional[Iterable[str]] = None,
    exclude_aggregate_codes: bool = False,
) -> EstimationShares:
    """Prepare within-group trade shares for a concordance period using dataframes."""
    vintage_a_year, vintage_b_year = _split_period(period)
    measure = _normalize_measure(measure)
    value_col = _measure_column(measure)

    summary = get_ambiguous_group_summary(groups, direction)
    period_summary = summary[summary["period"] == period]
    group_ids = set(period_summary["group_id"].tolist())

    exclude_set = set(exclude_codes) if exclude_codes else set(EXCLUDE_CODES_DEFAULT)
    if exclude_aggregate_codes:
        exclude_set |= AGGREGATE_CODES

    edges_period = groups.edges[groups.edges["period"] == period]
    a_map = edges_period[["vintage_a_code", "group_id"]].drop_duplicates()
    b_map = edges_period[["vintage_b_code", "group_id"]].drop_duplicates()

    shares_a, totals_a = _prepare_side_shares(
        data_a,
        period=period,
        vintage_year=vintage_a_year,
        group_map=a_map,
        group_ids=group_ids,
        code_col_name="vintage_a_code",
        flow=flow,
        exclude_codes=exclude_set,
        measure=measure,
    )
    shares_b, totals_b = _prepare_side_shares(
        data_b,
        period=period,
        vintage_year=vintage_b_year,
        group_map=b_map,
        group_ids=group_ids,
        code_col_name="vintage_b_code",
        flow=flow,
        exclude_codes=exclude_set,
        measure=measure,
    )

    total_a_col = f"total_{value_col}_a"
    total_b_col = f"total_{value_col}_b"

    totals = totals_a.rename(
        columns={
            f"total_{value_col}": total_a_col,
            "n_rows": "n_rows_a",
            "n_pairs": "n_pairs_a",
        }
    ).merge(
        totals_b.rename(
            columns={
                f"total_{value_col}": total_b_col,
                "n_rows": "n_rows_b",
                "n_pairs": "n_pairs_b",
            }
        ),
        on=["period", "group_id"],
        how="outer",
    )
    totals[total_a_col] = pd.to_numeric(totals[total_a_col], errors="coerce").fillna(0.0)
    totals[total_b_col] = pd.to_numeric(totals[total_b_col], errors="coerce").fillna(0.0)
    for col in ["n_rows_a", "n_rows_b", "n_pairs_a", "n_pairs_b"]:
        if col not in totals.columns:
            totals[col] = 0
        totals[col] = pd.to_numeric(totals[col], errors="coerce").fillna(0).astype(int)

    totals["skip_reason"] = ""
    totals.loc[totals[total_a_col] == 0, "skip_reason"] = "zero_total_a"
    totals.loc[totals[total_b_col] == 0, "skip_reason"] = totals["skip_reason"].replace(
        {"": "zero_total_b"}
    )
    totals.loc[
        (totals[total_a_col] == 0) & (totals[total_b_col] == 0), "skip_reason"
    ] = "zero_total_a_b"

    skipped = totals[totals["skip_reason"] != ""].copy().reset_index(drop=True)
    valid_groups = set(totals[totals["skip_reason"] == ""]["group_id"])

    shares_a = shares_a[shares_a["group_id"].isin(valid_groups)].reset_index(drop=True)
    shares_b = shares_b[shares_b["group_id"].isin(valid_groups)].reset_index(drop=True)

    return EstimationShares(
        period=period,
        direction=direction,
        measure=measure,
        vintage_a_year=vintage_a_year,
        vintage_b_year=vintage_b_year,
        shares_a=shares_a,
        shares_b=shares_b,
        group_totals=totals.reset_index(drop=True),
        skipped_groups=skipped,
    )


def prepare_estimation_shares_for_period(
    *,
    period: str,
    groups: ConcordanceGroups,
    direction: str = "a_to_b",
    base_dir: Path = ANNUAL_DATA_DIR,
    measure: str = "VALUE_EUR",
    flow: str = "1",
    exclude_codes: Optional[Iterable[str]] = None,
    exclude_aggregate_codes: bool = False,
) -> EstimationShares:
    """Prepare within-group trade shares for a concordance period from parquet files."""
    vintage_a_year, vintage_b_year = _split_period(period)
    measure = _normalize_measure(measure)
    data_a = _load_annual_year(base_dir, vintage_a_year, measure)
    data_b = _load_annual_year(base_dir, vintage_b_year, measure)
    return prepare_estimation_shares_from_frames(
        period=period,
        groups=groups,
        direction=direction,
        data_a=data_a,
        data_b=data_b,
        measure=measure,
        flow=flow,
        exclude_codes=exclude_codes,
        exclude_aggregate_codes=exclude_aggregate_codes,
    )
