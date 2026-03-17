"""Raw-data bilateral persistence analysis in the spirit of LT Table 3."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from ...concordance.mappings import get_ambiguous_group_summary
from ...estimation.runner import load_concordance_groups
from ..config import BilateralPersistenceConfig
from ..common.steps import chain_steps
from ..common.shares import normalize_codes


ROW_DETERMINISTIC_ALL = "deterministic_all_codes"
ROW_ALL = "all_codes"
ROW_ADJUSTED = "adjusted_codes"

ROW_LABELS = {
    ROW_DETERMINISTIC_ALL: "All deterministically break-comparable CN codes",
    ROW_ALL: "All break-group CN codes",
    ROW_ADJUSTED: "Adjusted CN codes",
}

ROW_ORDER = [ROW_DETERMINISTIC_ALL, ROW_ALL, ROW_ADJUSTED]

EMPTY_POSITIVE_COLUMNS = [
    "REPORTER",
    "PARTNER",
    "group_id",
    "concept_code",
    "value",
    "group_total",
    "scaled_flow",
]

EMPTY_PANEL_COLUMNS = [
    "REPORTER",
    "PARTNER",
    "group_id",
    "concept_code",
    "scaled_flow_lag",
    "scaled_flow_cur",
]


def _split_period(period: str) -> tuple[int, int]:
    text = str(period).strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"Invalid period '{period}'")
    return int(text[:4]), int(text[4:])


def _load_native_bilateral_year(
    *,
    annual_base_dir,
    year: int,
    measure: str,
    flow_code: str,
    exclude_reporters,
    exclude_partners,
) -> pd.DataFrame:
    data_path = annual_base_dir / f"comext_{year}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing annual data file: {data_path}")
    cols = ["FLOW", "REPORTER", "PARTNER", "PRODUCT_NC", measure]
    data = pd.read_parquet(data_path, columns=cols)
    data["FLOW"] = data["FLOW"].astype(str)
    data = data.loc[data["FLOW"] == str(flow_code)]
    if exclude_reporters:
        data = data.loc[~data["REPORTER"].isin(exclude_reporters)]
    if exclude_partners:
        data = data.loc[~data["PARTNER"].isin(exclude_partners)]
    data = data.rename(columns={measure: "value"})
    data["PRODUCT_NC"] = normalize_codes(data["PRODUCT_NC"])
    grouped = (
        data.groupby(["REPORTER", "PARTNER", "PRODUCT_NC"], as_index=False, sort=False)["value"]
        .sum()
    )
    grouped["value"] = grouped["value"].astype(float)
    return grouped


def _strict_one_to_one_maps(groups) -> dict[tuple[str, str], pd.DataFrame]:
    maps: dict[tuple[str, str], pd.DataFrame] = {}
    edges = groups.edges[
        ["period", "vintage_a_code", "vintage_b_code"]
    ].drop_duplicates().copy()
    edges["vintage_a_code"] = normalize_codes(edges["vintage_a_code"])
    edges["vintage_b_code"] = normalize_codes(edges["vintage_b_code"])

    for period, period_edges in edges.groupby("period", sort=False):
        a_deg = period_edges.groupby("vintage_a_code", sort=False)["vintage_b_code"].nunique()
        b_deg = period_edges.groupby("vintage_b_code", sort=False)["vintage_a_code"].nunique()
        mask = (
            period_edges["vintage_a_code"].map(a_deg).eq(1)
            & period_edges["vintage_b_code"].map(b_deg).eq(1)
        )
        matched = period_edges.loc[mask].copy()
        maps[(str(period), "a_to_b")] = matched.rename(
            columns={"vintage_a_code": "from_code", "vintage_b_code": "to_code"}
        )[["from_code", "to_code"]]
        maps[(str(period), "b_to_a")] = matched.rename(
            columns={"vintage_b_code": "from_code", "vintage_a_code": "to_code"}
        )[["from_code", "to_code"]]
    return maps


def _period_revised_code_sets(groups) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    revised_a: dict[str, set[str]] = {}
    revised_b: dict[str, set[str]] = {}
    edges = groups.edges[
        ["period", "vintage_a_code", "vintage_b_code"]
    ].drop_duplicates().copy()
    edges["vintage_a_code"] = normalize_codes(edges["vintage_a_code"])
    edges["vintage_b_code"] = normalize_codes(edges["vintage_b_code"])
    for period, period_edges in edges.groupby("period", sort=False):
        revised_a[str(period)] = set(period_edges["vintage_a_code"].astype(str).tolist())
        revised_b[str(period)] = set(period_edges["vintage_b_code"].astype(str).tolist())
    return revised_a, revised_b


def _carry_codes_to_target(
    *,
    codes: pd.Series,
    source_year: int,
    target_year: int,
    one_to_one_maps: dict[tuple[str, str], pd.DataFrame],
    revised_a_by_period: dict[str, set[str]],
    revised_b_by_period: dict[str, set[str]],
) -> pd.Series:
    current = pd.DataFrame({"concept_code": normalize_codes(codes)})
    if source_year == target_year:
        return current["concept_code"]

    for step in chain_steps(source_year, target_year):
        period = str(step["period"])
        direction = str(step["direction"])
        step_map = one_to_one_maps.get((period, direction))
        step_map = step_map if step_map is not None else pd.DataFrame(columns=["from_code", "to_code"])
        matched = current.merge(
            step_map,
            left_on="concept_code",
            right_on="from_code",
            how="inner",
        )
        if direction == "a_to_b":
            revised_codes = revised_a_by_period.get(period, set())
        else:
            revised_codes = revised_b_by_period.get(period, set())
        unchanged = current.loc[~current["concept_code"].isin(revised_codes), ["concept_code"]].copy()
        matched = matched[["to_code"]].rename(columns={"to_code": "concept_code"})
        current = pd.concat([matched, unchanged], ignore_index=True).drop_duplicates()
        if current.empty:
            return pd.Series(dtype="object")

    return current["concept_code"]


def _carry_frame_to_target(
    *,
    frame: pd.DataFrame,
    source_year: int,
    target_year: int,
    one_to_one_maps: dict[tuple[str, str], pd.DataFrame],
    revised_a_by_period: dict[str, set[str]],
    revised_b_by_period: dict[str, set[str]],
) -> pd.DataFrame:
    current = frame.rename(columns={"PRODUCT_NC": "concept_code"}).copy()
    current["concept_code"] = normalize_codes(current["concept_code"])

    if source_year == target_year:
        return current

    for step in chain_steps(source_year, target_year):
        period = str(step["period"])
        direction = str(step["direction"])
        step_map = one_to_one_maps.get((period, direction))
        step_map = step_map if step_map is not None else pd.DataFrame(columns=["from_code", "to_code"])
        matched = current.merge(
            step_map,
            left_on="concept_code",
            right_on="from_code",
            how="inner",
        )
        matched = matched.drop(columns=["concept_code", "from_code"]).rename(
            columns={"to_code": "concept_code"}
        )
        if direction == "a_to_b":
            revised_codes = revised_a_by_period.get(period, set())
        else:
            revised_codes = revised_b_by_period.get(period, set())
        unchanged = current.loc[~current["concept_code"].isin(revised_codes)].copy()
        current = pd.concat([matched, unchanged], ignore_index=True)
        if current.empty:
            return current

    return (
        current.groupby(["REPORTER", "PARTNER", "concept_code"], as_index=False, sort=False)["value"]
        .sum()
    )


def _positive_scaled_flows(
    *,
    frame: pd.DataFrame,
    group_map: pd.DataFrame,
) -> pd.DataFrame:
    df = frame.merge(group_map, on="concept_code", how="inner")
    if df.empty:
        return pd.DataFrame(columns=EMPTY_POSITIVE_COLUMNS)
    df = (
        df.groupby(["REPORTER", "PARTNER", "group_id", "concept_code"], as_index=False, sort=False)["value"]
        .sum()
    )
    totals = (
        df.groupby("group_id", as_index=False, sort=False)["value"]
        .sum()
        .rename(columns={"value": "group_total"})
    )
    df = df.merge(totals, on="group_id", how="left")
    df = df.loc[df["group_total"] > 0].copy()
    df["scaled_flow"] = df["value"] / df["group_total"]
    return df


def _panel_from_positive_flows(
    *,
    lag_positive: pd.DataFrame,
    year_positive: pd.DataFrame,
    code_universe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if code_universe.empty:
        empty = pd.DataFrame(columns=EMPTY_PANEL_COLUMNS)
        return empty, {
            "n_groups": 0,
            "n_concepts": 0,
            "n_pairs": 0,
            "n_cells": 0,
            "n_positive_lag": 0,
            "n_positive_current": 0,
            "n_both_positive": 0,
        }

    pair_universe = pd.concat(
        [
            lag_positive[["REPORTER", "PARTNER", "group_id"]],
            year_positive[["REPORTER", "PARTNER", "group_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    if pair_universe.empty:
        empty = pd.DataFrame(columns=EMPTY_PANEL_COLUMNS)
        return empty, {
            "n_groups": int(code_universe["group_id"].nunique()),
            "n_concepts": int(code_universe["concept_code"].nunique()),
            "n_pairs": 0,
            "n_cells": 0,
            "n_positive_lag": int(len(lag_positive)),
            "n_positive_current": int(len(year_positive)),
            "n_both_positive": 0,
        }

    panel = pair_universe.merge(code_universe, on="group_id", how="inner")
    lag_scaled = lag_positive[
        ["REPORTER", "PARTNER", "group_id", "concept_code", "scaled_flow"]
    ].rename(columns={"scaled_flow": "scaled_flow_lag"})
    year_scaled = year_positive[
        ["REPORTER", "PARTNER", "group_id", "concept_code", "scaled_flow"]
    ].rename(columns={"scaled_flow": "scaled_flow_cur"})
    panel = panel.merge(
        lag_scaled,
        on=["REPORTER", "PARTNER", "group_id", "concept_code"],
        how="left",
    )
    panel = panel.merge(
        year_scaled,
        on=["REPORTER", "PARTNER", "group_id", "concept_code"],
        how="left",
    )
    panel["scaled_flow_lag"] = pd.to_numeric(panel["scaled_flow_lag"], errors="coerce").fillna(0.0)
    panel["scaled_flow_cur"] = pd.to_numeric(panel["scaled_flow_cur"], errors="coerce").fillna(0.0)

    diagnostics = {
        "n_groups": int(code_universe["group_id"].nunique()),
        "n_concepts": int(code_universe["concept_code"].nunique()),
        "n_pairs": int(panel[["REPORTER", "PARTNER"]].drop_duplicates().shape[0]),
        "n_cells": int(len(panel)),
        "n_positive_lag": int(len(lag_positive)),
        "n_positive_current": int(len(year_positive)),
        "n_both_positive": int(
            ((panel["scaled_flow_lag"] > 0.0) & (panel["scaled_flow_cur"] > 0.0)).sum()
        ),
    }
    return panel, diagnostics


def _lt_no_constant_hc1(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    xx = float(np.dot(x, x))
    if xx <= 0:
        return float("nan"), float("nan")
    beta = float(np.dot(x, y) / xx)
    resid = y - beta * x
    if n <= 1:
        return beta, float("nan")
    meat = float(np.sum((x**2) * (resid**2)))
    var_hc1 = (n / (n - 1)) * meat / (xx**2)
    se = float(np.sqrt(var_hc1))
    return beta, se


def _format_cell(beta: float, se: float) -> str:
    if not np.isfinite(beta):
        return ""
    if not np.isfinite(se):
        return f"{beta:.3f}"
    return f"{beta:.3f} ({se:.3f})"


def _non_bijective_codes_from_edges(edges: pd.DataFrame) -> tuple[set[str], set[str]]:
    edges = edges[["vintage_a_code", "vintage_b_code"]].drop_duplicates().copy()
    edges["vintage_a_code"] = normalize_codes(edges["vintage_a_code"])
    edges["vintage_b_code"] = normalize_codes(edges["vintage_b_code"])
    if edges.empty:
        return set(), set()
    a_deg = edges.groupby("vintage_a_code", sort=False)["vintage_b_code"].nunique()
    b_deg = edges.groupby("vintage_b_code", sort=False)["vintage_a_code"].nunique()
    mask = (
        edges["vintage_a_code"].map(a_deg).gt(1)
        | edges["vintage_b_code"].map(b_deg).gt(1)
    )
    contaminated = edges.loc[mask]
    return (
        set(contaminated["vintage_a_code"].astype(str).tolist()),
        set(contaminated["vintage_b_code"].astype(str).tolist()),
    )


def _select_adjusted_group_ids(groups, *, break_period: str, direction: str) -> set[str]:
    if direction == "union":
        summary = groups.group_summary.loc[
            (groups.group_summary["period"] == break_period)
            & (
                groups.group_summary["a_to_b_ambiguous"]
                | groups.group_summary["b_to_a_ambiguous"]
            ),
            ["group_id"],
        ]
    else:
        summary = get_ambiguous_group_summary(groups, direction)
        summary = summary.loc[summary["period"] == break_period, ["group_id"]]
    return set(summary["group_id"].astype(str).tolist())


def _collect_contaminated_group_ids(
    *,
    groups,
    break_period: str,
    filter_years: list[int],
    one_to_one_maps: dict[tuple[str, str], pd.DataFrame],
    revised_a_by_period: dict[str, set[str]],
    revised_b_by_period: dict[str, set[str]],
    candidate_group_ids: set[str],
) -> set[str]:
    break_a_year, break_b_year = _split_period(break_period)
    break_edges = groups.edges.loc[groups.edges["period"] == break_period]
    a_group_map = break_edges[["vintage_a_code", "group_id"]].drop_duplicates().copy()
    a_group_map["vintage_a_code"] = normalize_codes(a_group_map["vintage_a_code"])
    b_group_map = break_edges[["vintage_b_code", "group_id"]].drop_duplicates().copy()
    b_group_map["vintage_b_code"] = normalize_codes(b_group_map["vintage_b_code"])

    contaminated: set[str] = set()
    for year in filter_years:
        period = f"{year}{year + 1}"
        period_edges = groups.edges.loc[groups.edges["period"] == period]
        if period_edges.empty:
            continue
        unstable_a, unstable_b = _non_bijective_codes_from_edges(period_edges)

        if year + 1 <= break_a_year:
            target_year = break_a_year
            group_map = a_group_map.rename(columns={"vintage_a_code": "concept_code"})
            carried_a = _carry_codes_to_target(
                codes=pd.Series(sorted(unstable_a)),
                source_year=year,
                target_year=target_year,
                one_to_one_maps=one_to_one_maps,
                revised_a_by_period=revised_a_by_period,
                revised_b_by_period=revised_b_by_period,
            )
            carried_b = _carry_codes_to_target(
                codes=pd.Series(sorted(unstable_b)),
                source_year=year + 1,
                target_year=target_year,
                one_to_one_maps=one_to_one_maps,
                revised_a_by_period=revised_a_by_period,
                revised_b_by_period=revised_b_by_period,
            )
        elif year >= break_b_year:
            target_year = break_b_year
            group_map = b_group_map.rename(columns={"vintage_b_code": "concept_code"})
            carried_a = _carry_codes_to_target(
                codes=pd.Series(sorted(unstable_a)),
                source_year=year,
                target_year=target_year,
                one_to_one_maps=one_to_one_maps,
                revised_a_by_period=revised_a_by_period,
                revised_b_by_period=revised_b_by_period,
            )
            carried_b = _carry_codes_to_target(
                codes=pd.Series(sorted(unstable_b)),
                source_year=year + 1,
                target_year=target_year,
                one_to_one_maps=one_to_one_maps,
                revised_a_by_period=revised_a_by_period,
                revised_b_by_period=revised_b_by_period,
            )
        else:
            continue

        carried = pd.concat([carried_a, carried_b], ignore_index=True).drop_duplicates()
        if carried.empty:
            continue
        group_ids = set(
            group_map.loc[group_map["concept_code"].isin(set(carried.astype(str))), "group_id"]
            .astype(str)
            .tolist()
        )
        contaminated |= (group_ids & candidate_group_ids)

    return contaminated


def _prepare_break_pair(
    *,
    year: int,
    lag_year: int,
    break_a_year: int,
    break_b_year: int,
    annual_by_year: dict[int, pd.DataFrame],
    one_to_one_maps: dict[tuple[str, str], pd.DataFrame],
    revised_a_by_period: dict[str, set[str]],
    revised_b_by_period: dict[str, set[str]],
    break_group_map_a: pd.DataFrame,
    break_group_map_b: pd.DataFrame,
    group_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if year <= break_a_year:
        target_year = break_a_year
        group_map = break_group_map_a
    else:
        target_year = break_b_year
        group_map = break_group_map_b
    group_map = group_map.loc[group_map["group_id"].isin(group_ids)].drop_duplicates().reset_index(drop=True)

    return _prepare_target_basis_pair(
        year=year,
        lag_year=lag_year,
        target_year=target_year,
        group_map=group_map,
        annual_by_year=annual_by_year,
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
    )


def _prepare_target_basis_pair(
    *,
    year: int,
    lag_year: int,
    target_year: int,
    group_map: pd.DataFrame,
    annual_by_year: dict[int, pd.DataFrame],
    one_to_one_maps: dict[tuple[str, str], pd.DataFrame],
    revised_a_by_period: dict[str, set[str]],
    revised_b_by_period: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    lag_frame = _carry_frame_to_target(
        frame=annual_by_year[lag_year],
        source_year=lag_year,
        target_year=target_year,
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
    )
    year_frame = _carry_frame_to_target(
        frame=annual_by_year[year],
        source_year=year,
        target_year=target_year,
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
    )
    lag_positive = _positive_scaled_flows(frame=lag_frame, group_map=group_map)
    year_positive = _positive_scaled_flows(frame=year_frame, group_map=group_map)
    panel, diagnostics = _panel_from_positive_flows(
        lag_positive=lag_positive,
        year_positive=year_positive,
        code_universe=group_map[["group_id", "concept_code"]].drop_duplicates(),
    )
    diagnostics["basis_year"] = target_year
    return panel, diagnostics


def _collect_contaminated_basis_codes(
    *,
    break_a_year: int,
    break_b_year: int,
    target_year: int,
    filter_years: list[int],
    one_to_one_maps: dict[tuple[str, str], pd.DataFrame],
    revised_a_by_period: dict[str, set[str]],
    revised_b_by_period: dict[str, set[str]],
    groups,
    candidate_codes: set[str],
) -> set[str]:
    contaminated: set[str] = set()
    for year in filter_years:
        period = f"{year}{year + 1}"
        period_edges = groups.edges.loc[groups.edges["period"] == period]
        if period_edges.empty:
            continue

        if target_year == break_a_year and year + 1 > break_a_year:
            continue
        if target_year == break_b_year and year < break_b_year:
            continue

        unstable_a, unstable_b = _non_bijective_codes_from_edges(period_edges)
        carried_a = _carry_codes_to_target(
            codes=pd.Series(sorted(unstable_a)),
            source_year=year,
            target_year=target_year,
            one_to_one_maps=one_to_one_maps,
            revised_a_by_period=revised_a_by_period,
            revised_b_by_period=revised_b_by_period,
        )
        carried_b = _carry_codes_to_target(
            codes=pd.Series(sorted(unstable_b)),
            source_year=year + 1,
            target_year=target_year,
            one_to_one_maps=one_to_one_maps,
            revised_a_by_period=revised_a_by_period,
            revised_b_by_period=revised_b_by_period,
        )
        carried = set(pd.concat([carried_a, carried_b], ignore_index=True).astype(str).tolist())
        contaminated |= (carried & candidate_codes)
    return contaminated


def _build_deterministic_basis_group_map(
    *,
    target_year: int,
    annual_by_year: dict[int, pd.DataFrame],
    break_group_map: pd.DataFrame,
    contaminated_codes: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    basis_codes = pd.DataFrame(
        {"concept_code": sorted(set(normalize_codes(annual_by_year[target_year]["PRODUCT_NC"])))}
    )
    linked_map = break_group_map.drop_duplicates().copy()
    linked_map["concept_code"] = normalize_codes(linked_map["concept_code"])
    linked_group_ids = set(linked_map["group_id"].astype(str))

    basis_map = basis_codes.merge(linked_map, on="concept_code", how="left")
    basis_map["group_id"] = basis_map["group_id"].fillna(
        basis_map["concept_code"].map(lambda code: f"singleton::{target_year}::{code}")
    )
    basis_map["is_linked_group"] = basis_map["group_id"].isin(linked_group_ids)

    pre_filter = {
        "n_linked_groups_pre_filter": int(
            basis_map.loc[basis_map["is_linked_group"], "group_id"].nunique()
        ),
        "n_singleton_groups_pre_filter": int((~basis_map["is_linked_group"]).sum()),
    }
    filtered = basis_map.loc[~basis_map["concept_code"].isin(contaminated_codes)].copy()
    diagnostics = {
        **pre_filter,
        "n_linked_groups": int(filtered.loc[filtered["is_linked_group"], "group_id"].nunique()),
        "n_singleton_groups": int((~filtered["is_linked_group"]).sum()),
    }
    return filtered[["concept_code", "group_id"]].reset_index(drop=True), diagnostics


def _prepare_deterministic_all_pair(
    *,
    year: int,
    lag_year: int,
    break_a_year: int,
    break_b_year: int,
    annual_by_year: dict[int, pd.DataFrame],
    one_to_one_maps: dict[tuple[str, str], pd.DataFrame],
    revised_a_by_period: dict[str, set[str]],
    revised_b_by_period: dict[str, set[str]],
    basis_group_map_a: pd.DataFrame,
    basis_group_map_b: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if year <= break_a_year:
        target_year = break_a_year
        group_map = basis_group_map_a
    else:
        target_year = break_b_year
        group_map = basis_group_map_b

    return _prepare_target_basis_pair(
        year=year,
        lag_year=lag_year,
        target_year=target_year,
        group_map=group_map,
        annual_by_year=annual_by_year,
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
    )


def run_bilateral_persistence_analysis(config: BilateralPersistenceConfig) -> dict[str, object]:
    groups = load_concordance_groups(
        concordance_path=config.paths.concordance_path,
        sheet_name=config.paths.concordance_sheet,
    )
    break_period = config.break_config.period
    break_a_year, break_b_year = _split_period(break_period)

    annual_years = sorted({year for year in config.years.columns} | {year - 1 for year in config.years.columns})
    annual_by_year: dict[int, pd.DataFrame] = {}
    for year in annual_years:
        annual_by_year[year] = _load_native_bilateral_year(
            annual_base_dir=config.paths.annual_base_dir,
            year=year,
            measure=config.measures.analysis_measure,
            flow_code=config.flow.flow_code,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
        )

    one_to_one_maps = _strict_one_to_one_maps(groups)
    revised_a_by_period, revised_b_by_period = _period_revised_code_sets(groups)

    break_edges = groups.edges.loc[groups.edges["period"] == break_period].copy()
    if break_edges.empty:
        raise ValueError(f"Missing concordance edges for break period {break_period}")
    break_edges["vintage_a_code"] = normalize_codes(break_edges["vintage_a_code"])
    break_edges["vintage_b_code"] = normalize_codes(break_edges["vintage_b_code"])
    break_group_map_a = break_edges[["vintage_a_code", "group_id"]].drop_duplicates().rename(
        columns={"vintage_a_code": "concept_code"}
    )
    break_group_map_b = break_edges[["vintage_b_code", "group_id"]].drop_duplicates().rename(
        columns={"vintage_b_code": "concept_code"}
    )

    all_break_group_ids_pre = set(break_edges["group_id"].astype(str).tolist())
    adjusted_group_ids_pre = _select_adjusted_group_ids(
        groups,
        break_period=break_period,
        direction=config.break_config.direction,
    )
    contaminated_groups = _collect_contaminated_group_ids(
        groups=groups,
        break_period=break_period,
        filter_years=list(config.adjusted_filter.years),
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
        candidate_group_ids=all_break_group_ids_pre,
    )
    all_break_group_ids = all_break_group_ids_pre - contaminated_groups
    adjusted_group_ids = adjusted_group_ids_pre & all_break_group_ids
    adjusted_contaminated_groups = adjusted_group_ids_pre - adjusted_group_ids
    det_contaminated_a = _collect_contaminated_basis_codes(
        break_a_year=break_a_year,
        break_b_year=break_b_year,
        target_year=break_a_year,
        filter_years=list(config.adjusted_filter.years),
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
        groups=groups,
        candidate_codes=set(normalize_codes(annual_by_year[break_a_year]["PRODUCT_NC"])),
    )
    det_contaminated_b = _collect_contaminated_basis_codes(
        break_a_year=break_a_year,
        break_b_year=break_b_year,
        target_year=break_b_year,
        filter_years=list(config.adjusted_filter.years),
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
        groups=groups,
        candidate_codes=set(normalize_codes(annual_by_year[break_b_year]["PRODUCT_NC"])),
    )
    deterministic_group_map_a, deterministic_diag_a = _build_deterministic_basis_group_map(
        target_year=break_a_year,
        annual_by_year=annual_by_year,
        break_group_map=break_group_map_a,
        contaminated_codes=det_contaminated_a,
    )
    deterministic_group_map_b, deterministic_diag_b = _build_deterministic_basis_group_map(
        target_year=break_b_year,
        annual_by_year=annual_by_year,
        break_group_map=break_group_map_b,
        contaminated_codes=det_contaminated_b,
    )

    regression_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for year in config.years.columns:
        lag_year = year - 1

        det_panel, det_diag = _prepare_deterministic_all_pair(
            year=year,
            lag_year=lag_year,
            break_a_year=break_a_year,
            break_b_year=break_b_year,
            annual_by_year=annual_by_year,
            one_to_one_maps=one_to_one_maps,
            revised_a_by_period=revised_a_by_period,
            revised_b_by_period=revised_b_by_period,
            basis_group_map_a=deterministic_group_map_a,
            basis_group_map_b=deterministic_group_map_b,
        )
        beta, se = _lt_no_constant_hc1(
            det_panel["scaled_flow_lag"].to_numpy(),
            det_panel["scaled_flow_cur"].to_numpy(),
        )
        det_meta = deterministic_diag_a if int(det_diag["basis_year"]) == break_a_year else deterministic_diag_b
        regression_rows.append(
            {
                "row_key": ROW_DETERMINISTIC_ALL,
                "row_label": ROW_LABELS[ROW_DETERMINISTIC_ALL],
                "year": year,
                "lag_year": lag_year,
                "basis_year": int(det_diag["basis_year"]),
                "sample_basis": "break_filtered_deterministic_all",
                "coef": beta,
                "se": se,
                "n_obs": int(len(det_panel)),
            }
        )
        sample_rows.append(
            {
                "row_key": ROW_DETERMINISTIC_ALL,
                "row_label": ROW_LABELS[ROW_DETERMINISTIC_ALL],
                "year": year,
                "lag_year": lag_year,
                "basis_year": int(det_diag["basis_year"]),
                "sample_basis": "break_filtered_deterministic_all",
                "n_groups_pre_filter": int(
                    det_meta["n_linked_groups_pre_filter"] + det_meta["n_singleton_groups_pre_filter"]
                ),
                "n_groups_contaminated": int(
                    det_meta["n_linked_groups_pre_filter"]
                    + det_meta["n_singleton_groups_pre_filter"]
                    - det_meta["n_linked_groups"]
                    - det_meta["n_singleton_groups"]
                ),
                "n_groups": det_diag["n_groups"],
                "n_linked_groups_pre_filter": int(det_meta["n_linked_groups_pre_filter"]),
                "n_linked_groups": int(det_meta["n_linked_groups"]),
                "n_singleton_groups_pre_filter": int(det_meta["n_singleton_groups_pre_filter"]),
                "n_singleton_groups": int(det_meta["n_singleton_groups"]),
                "n_concepts": det_diag["n_concepts"],
                "n_pairs": det_diag["n_pairs"],
                "n_cells": det_diag["n_cells"],
                "n_positive_lag": det_diag["n_positive_lag"],
                "n_positive_current": det_diag["n_positive_current"],
                "n_both_positive": det_diag["n_both_positive"],
                "n_obs": int(len(det_panel)),
            }
        )

        broad_panel, broad_diag = _prepare_break_pair(
            lag_year=lag_year,
            year=year,
            break_a_year=break_a_year,
            break_b_year=break_b_year,
            annual_by_year=annual_by_year,
            one_to_one_maps=one_to_one_maps,
            revised_a_by_period=revised_a_by_period,
            revised_b_by_period=revised_b_by_period,
            break_group_map_a=break_group_map_a,
            break_group_map_b=break_group_map_b,
            group_ids=all_break_group_ids,
        )
        beta, se = _lt_no_constant_hc1(
            broad_panel["scaled_flow_lag"].to_numpy(),
            broad_panel["scaled_flow_cur"].to_numpy(),
        )
        regression_rows.append(
            {
                "row_key": ROW_ALL,
                "row_label": ROW_LABELS[ROW_ALL],
                "year": year,
                "lag_year": lag_year,
                "basis_year": int(broad_diag["basis_year"]),
                "sample_basis": "break_filtered_break_groups",
                "coef": beta,
                "se": se,
                "n_obs": int(len(broad_panel)),
            }
        )
        sample_rows.append(
            {
                "row_key": ROW_ALL,
                "row_label": ROW_LABELS[ROW_ALL],
                "year": year,
                "lag_year": lag_year,
                "basis_year": int(broad_diag["basis_year"]),
                "sample_basis": "break_filtered_break_groups",
                "n_groups_pre_filter": int(len(all_break_group_ids_pre)),
                "n_groups_contaminated": int(len(contaminated_groups)),
                "n_groups": broad_diag["n_groups"],
                "n_linked_groups_pre_filter": int(len(all_break_group_ids_pre)),
                "n_linked_groups": broad_diag["n_groups"],
                "n_singleton_groups_pre_filter": 0,
                "n_singleton_groups": 0,
                "n_concepts": broad_diag["n_concepts"],
                "n_pairs": broad_diag["n_pairs"],
                "n_cells": broad_diag["n_cells"],
                "n_positive_lag": broad_diag["n_positive_lag"],
                "n_positive_current": broad_diag["n_positive_current"],
                "n_both_positive": broad_diag["n_both_positive"],
                "n_obs": int(len(broad_panel)),
            }
        )

        adjusted_panel, adj_diag = _prepare_break_pair(
            year=year,
            lag_year=lag_year,
            break_a_year=break_a_year,
            break_b_year=break_b_year,
            annual_by_year=annual_by_year,
            one_to_one_maps=one_to_one_maps,
            revised_a_by_period=revised_a_by_period,
            revised_b_by_period=revised_b_by_period,
            break_group_map_a=break_group_map_a,
            break_group_map_b=break_group_map_b,
            group_ids=adjusted_group_ids,
        )
        beta, se = _lt_no_constant_hc1(
            adjusted_panel["scaled_flow_lag"].to_numpy(),
            adjusted_panel["scaled_flow_cur"].to_numpy(),
        )
        regression_rows.append(
            {
                "row_key": ROW_ADJUSTED,
                "row_label": ROW_LABELS[ROW_ADJUSTED],
                "year": year,
                "lag_year": lag_year,
                "basis_year": int(adj_diag["basis_year"]),
                "sample_basis": "break_filtered_adjusted",
                "coef": beta,
                "se": se,
                "n_obs": int(len(adjusted_panel)),
            }
        )
        sample_rows.append(
            {
                "row_key": ROW_ADJUSTED,
                "row_label": ROW_LABELS[ROW_ADJUSTED],
                "year": year,
                "lag_year": lag_year,
                "basis_year": int(adj_diag["basis_year"]),
                "sample_basis": "break_filtered_adjusted",
                "n_groups_pre_filter": int(len(adjusted_group_ids_pre)),
                "n_groups_contaminated": int(len(adjusted_contaminated_groups)),
                "n_groups": adj_diag["n_groups"],
                "n_linked_groups_pre_filter": int(len(adjusted_group_ids_pre)),
                "n_linked_groups": adj_diag["n_groups"],
                "n_singleton_groups_pre_filter": 0,
                "n_singleton_groups": 0,
                "n_concepts": adj_diag["n_concepts"],
                "n_pairs": adj_diag["n_pairs"],
                "n_cells": adj_diag["n_cells"],
                "n_positive_lag": adj_diag["n_positive_lag"],
                "n_positive_current": adj_diag["n_positive_current"],
                "n_both_positive": adj_diag["n_both_positive"],
                "n_obs": int(len(adjusted_panel)),
            }
        )

    row_order = {row_key: idx for idx, row_key in enumerate(ROW_ORDER)}
    details = pd.DataFrame(regression_rows)
    details["_row_order"] = details["row_key"].map(row_order)
    details = details.sort_values(["_row_order", "year"]).drop(columns="_row_order").reset_index(drop=True)
    diagnostics = pd.DataFrame(sample_rows)
    diagnostics["_row_order"] = diagnostics["row_key"].map(row_order)
    diagnostics = (
        diagnostics.sort_values(["_row_order", "year"]).drop(columns="_row_order").reset_index(drop=True)
    )

    table_rows = []
    for row_key in ROW_ORDER:
        subset = details.loc[details["row_key"] == row_key].copy()
        row = {"row_label": ROW_LABELS[row_key]}
        for year in config.years.columns:
            match = subset.loc[subset["year"] == year]
            if match.empty:
                row[str(year)] = ""
            else:
                row[str(year)] = _format_cell(
                    float(match.iloc[0]["coef"]),
                    float(match.iloc[0]["se"]),
                )
        table_rows.append(row)
    table = pd.DataFrame(table_rows)

    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(config.output.details_csv, index=False)
    diagnostics.to_csv(config.output.sample_diagnostics_csv, index=False)
    table.to_csv(config.output.table_csv, index=False)
    tex = table.to_latex(index=False, escape=False)
    config.output.table_tex.write_text(tex)

    return {
        "table_csv": str(config.output.table_csv),
        "details_csv": str(config.output.details_csv),
        "sample_diagnostics_csv": str(config.output.sample_diagnostics_csv),
        "config": asdict(config),
    }
