"""Shared utilities for within-group share stability analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ...core.codes import normalize_codes as _normalize_codes_core


@dataclass(frozen=True)
class PanelPair:
    x_year: int
    y_year: int
    data: pd.DataFrame


def normalize_codes(series: pd.Series) -> pd.Series:
    return _normalize_codes_core(series)


def _normalize_years(years: Iterable[int]) -> list[int]:
    return [int(year) for year in years]


def _weights_for_year(
    *,
    year: int,
    target_year: int,
    weights_by_year: dict[str, pd.DataFrame],
) -> pd.DataFrame | None:
    return weights_by_year.get(str(year)) if year != target_year else None


def _validate_annual_files_exist(*, years: Sequence[int], annual_base_dir: Path) -> None:
    missing = [
        annual_base_dir / f"comext_{year}.parquet"
        for year in years
        if not (annual_base_dir / f"comext_{year}.parquet").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing annual data file: {missing[0]}")


def filter_partners(
    df: pd.DataFrame, *, exclude_reporters: Sequence[str], exclude_partners: Sequence[str]
) -> pd.DataFrame:
    if not exclude_reporters and not exclude_partners:
        return df
    mask = pd.Series(True, index=df.index)
    if exclude_reporters:
        mask &= ~df["REPORTER"].isin(exclude_reporters)
    if exclude_partners:
        mask &= ~df["PARTNER"].isin(exclude_partners)
    return df.loc[mask]


def convert_totals_to_target(
    *,
    totals: pd.DataFrame,
    weights: pd.DataFrame | None,
    assume_identity_for_missing: bool = True,
) -> pd.DataFrame:
    totals = totals.copy()
    totals["PRODUCT_NC"] = normalize_codes(totals["PRODUCT_NC"])
    if weights is None:
        return totals.rename(columns={"PRODUCT_NC": "target_code"})

    weights = weights[["from_code", "to_code", "weight"]].copy()
    weights["from_code"] = normalize_codes(weights["from_code"])
    weights["to_code"] = normalize_codes(weights["to_code"])

    missing = set(totals["PRODUCT_NC"]) - set(weights["from_code"])
    if missing:
        if not assume_identity_for_missing:
            sample = sorted(list(missing))[:10]
            raise ValueError(f"Missing weights for {len(missing)} codes; sample: {sample}")
        identity = pd.DataFrame(
            {"from_code": list(missing), "to_code": list(missing), "weight": 1.0}
        )
        weights = pd.concat([weights, identity], ignore_index=True)

    merged = totals.merge(weights, left_on="PRODUCT_NC", right_on="from_code", how="inner")
    merged["value"] = merged["value"] * merged["weight"]
    converted = (
        merged.groupby("to_code", as_index=False, sort=False)["value"].sum().rename(
            columns={"to_code": "target_code"}
        )
    )
    return converted


def compute_group_shares(
    *,
    totals: pd.DataFrame,
    group_map: pd.DataFrame,
    group_ids: set[str],
) -> pd.DataFrame:
    df = totals.merge(group_map, left_on="target_code", right_on="target_code", how="inner")
    df = df[df["group_id"].isin(group_ids)]
    group_totals = df.groupby("group_id", as_index=False, sort=False)["value"].sum()
    group_totals = group_totals.rename(columns={"value": "group_total"})
    df = df.merge(group_totals, on="group_id", how="left")
    df["share"] = df["value"] / df["group_total"]
    return df[["group_id", "target_code", "share"]]


def _load_year_totals(
    *,
    year: int,
    annual_base_dir: Path,
    measure: str,
    exclude_reporters: Sequence[str],
    exclude_partners: Sequence[str],
) -> pd.DataFrame:
    data_path = annual_base_dir / f"comext_{year}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing annual data file: {data_path}")
    data = pd.read_parquet(data_path, columns=["REPORTER", "PARTNER", "PRODUCT_NC", measure])
    data = filter_partners(
        data,
        exclude_reporters=exclude_reporters,
        exclude_partners=exclude_partners,
    )
    return (
        data.groupby("PRODUCT_NC", as_index=False, sort=False)[measure]
        .sum()
        .rename(columns={measure: "value"})
    )


def _build_values_for_groups_from_totals_impl(
    *,
    totals: pd.DataFrame,
    year: int,
    target_year: int,
    weights_by_year: dict[str, pd.DataFrame],
    group_map: pd.DataFrame,
    group_ids: set[str],
) -> pd.DataFrame:
    converted = convert_totals_to_target(
        totals=totals,
        weights=_weights_for_year(
            year=year,
            target_year=target_year,
            weights_by_year=weights_by_year,
        ),
        assume_identity_for_missing=True,
    )
    df = converted.merge(group_map, on="target_code", how="inner")
    df = df[df["group_id"].isin(group_ids)]
    df = df.rename(columns={"target_code": "product_code"})
    return df[["group_id", "product_code", "value"]]


def _build_year_share_frame_from_totals(
    *,
    totals: pd.DataFrame,
    year: int,
    target_year: int,
    weights_by_year: dict[str, pd.DataFrame],
    group_map: pd.DataFrame,
    group_ids: set[str],
) -> pd.DataFrame:
    converted = convert_totals_to_target(
        totals=totals,
        weights=_weights_for_year(
            year=year,
            target_year=target_year,
            weights_by_year=weights_by_year,
        ),
        assume_identity_for_missing=True,
    )
    shares = compute_group_shares(
        totals=converted,
        group_map=group_map,
        group_ids=group_ids,
    )
    shares = shares.rename(columns={"target_code": "product_code"})
    shares["year"] = year
    return shares


def build_year_shares(
    *,
    years: Iterable[int],
    target_year: int,
    annual_base_dir: Path,
    weights_by_year: dict[str, pd.DataFrame],
    measure: str,
    group_map: pd.DataFrame,
    group_ids: set[str],
    exclude_reporters: Sequence[str],
    exclude_partners: Sequence[str],
) -> dict[int, pd.DataFrame]:
    year_list = _normalize_years(years)
    _validate_annual_files_exist(years=year_list, annual_base_dir=annual_base_dir)
    shares_by_year: dict[int, pd.DataFrame] = {}
    for year in year_list:
        totals = _load_year_totals(
            year=year,
            annual_base_dir=annual_base_dir,
            measure=measure,
            exclude_reporters=exclude_reporters,
            exclude_partners=exclude_partners,
        )
        shares_by_year[year] = _build_year_share_frame_from_totals(
            totals=totals,
            year=year,
            target_year=target_year,
            weights_by_year=weights_by_year,
            group_map=group_map,
            group_ids=group_ids,
        )
    return shares_by_year


def build_year_shares_from_totals(
    *,
    years: Iterable[int],
    target_year: int,
    totals_by_year: dict[int, pd.DataFrame],
    weights_by_year: dict[str, pd.DataFrame],
    group_map: pd.DataFrame,
    group_ids: set[str],
) -> dict[int, pd.DataFrame]:
    year_list = _normalize_years(years)
    shares_by_year: dict[int, pd.DataFrame] = {}
    for year in year_list:
        if year not in totals_by_year:
            raise KeyError(f"Missing totals for year {year}")
        shares_by_year[year] = _build_year_share_frame_from_totals(
            totals=totals_by_year[year],
            year=year,
            target_year=target_year,
            weights_by_year=weights_by_year,
            group_map=group_map,
            group_ids=group_ids,
        )
    return shares_by_year


def build_target_values_for_groups(
    *,
    target_year: int,
    annual_base_dir: Path,
    measure: str,
    group_map: pd.DataFrame,
    group_ids: set[str],
    exclude_reporters: Sequence[str],
    exclude_partners: Sequence[str],
) -> pd.DataFrame:
    totals = _load_year_totals(
        year=target_year,
        annual_base_dir=annual_base_dir,
        measure=measure,
        exclude_reporters=exclude_reporters,
        exclude_partners=exclude_partners,
    )
    return _build_values_for_groups_from_totals_impl(
        totals=totals,
        year=target_year,
        target_year=target_year,
        weights_by_year={},
        group_map=group_map,
        group_ids=group_ids,
    )


def build_values_for_groups(
    *,
    year: int,
    target_year: int,
    annual_base_dir: Path,
    weights_by_year: dict[str, pd.DataFrame],
    measure: str,
    group_map: pd.DataFrame,
    group_ids: set[str],
    exclude_reporters: Sequence[str],
    exclude_partners: Sequence[str],
) -> pd.DataFrame:
    totals = _load_year_totals(
        year=year,
        annual_base_dir=annual_base_dir,
        measure=measure,
        exclude_reporters=exclude_reporters,
        exclude_partners=exclude_partners,
    )
    return _build_values_for_groups_from_totals_impl(
        totals=totals,
        year=year,
        target_year=target_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    )


def build_values_for_groups_from_totals(
    *,
    year: int,
    target_year: int,
    totals_by_year: dict[int, pd.DataFrame],
    weights_by_year: dict[str, pd.DataFrame],
    group_map: pd.DataFrame,
    group_ids: set[str],
) -> pd.DataFrame:
    if year not in totals_by_year:
        raise KeyError(f"Missing totals for year {year}")
    return _build_values_for_groups_from_totals_impl(
        totals=totals_by_year[year],
        year=year,
        target_year=target_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    )


def build_panel_pairs(
    *,
    start_year: int,
    end_year: int,
    year_shares: dict[int, pd.DataFrame],
    group_ids_filtered: set[str] | None = None,
) -> list[PanelPair]:
    pairs: list[PanelPair] = []
    for year in range(start_year, end_year):
        left = year_shares[year].rename(columns={"share": "share_t"})
        right = year_shares[year + 1].rename(columns={"share": "share_t1"})
        merged = left.merge(
            right,
            on=["group_id", "product_code"],
            how="inner",
        )
        if group_ids_filtered is not None:
            merged = merged[merged["group_id"].isin(group_ids_filtered)]
        pairs.append(PanelPair(x_year=year, y_year=year + 1, data=merged))
    return pairs
