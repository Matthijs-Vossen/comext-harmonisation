"""Share-stability analysis for within-group trade shares."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import pandas as pd
import numpy as np

from ..config import ShareStabilityConfig
from ..common.metrics import (
    mae_weighted,
    r2_45,
    r2_45_weighted,
    r2_45_weighted_symmetric,
    weighted_mean,
)
from ..common.plotting import plot_share_panels
from ..common.progress import progress
from ..common.chain_sampling import map_codes_to_target as _map_codes_to_target_common
from ..common.shares import (
    PanelPair,
    build_panel_pairs,
    build_values_for_groups_from_totals,
    build_year_shares_from_totals,
    normalize_codes,
)
from ..common.steps import compute_step_metrics
from ..common.steps import chain_steps
from ..common.steps import load_annual_totals
from ...chaining.engine import build_chained_weights_for_range, build_code_universe_from_annual
from ...estimation.runner import load_concordance_groups
from ...concordance.mappings import get_ambiguous_group_summary


def _unstable_codes_from_edges(edges: pd.DataFrame) -> tuple[set[str], set[str]]:
    edges = edges[["vintage_a_code", "vintage_b_code"]].drop_duplicates().copy()
    edges["vintage_a_code"] = normalize_codes(edges["vintage_a_code"])
    edges["vintage_b_code"] = normalize_codes(edges["vintage_b_code"])

    unstable_a: set[str] = set()
    unstable_b: set[str] = set()

    a_groups = edges.groupby("vintage_a_code", sort=False)["vintage_b_code"].apply(list)
    for code, targets in a_groups.items():
        unique_targets = set(targets)
        if len(unique_targets) != 1 or next(iter(unique_targets)) != code:
            unstable_a.add(code)

    b_groups = edges.groupby("vintage_b_code", sort=False)["vintage_a_code"].apply(list)
    for code, sources in b_groups.items():
        unique_sources = set(sources)
        if len(unique_sources) != 1 or next(iter(unique_sources)) != code:
            unstable_b.add(code)

    return unstable_a, unstable_b


def _map_codes_to_target(codes: set[str], weights: pd.DataFrame | None) -> set[str]:
    return _map_codes_to_target_common(
        codes,
        weights,
        preserve_unmapped=False,
    )


def _collect_unstable_target_codes(
    *,
    groups,
    years: Sequence[int],
    weights_by_year: dict[str, pd.DataFrame],
    target_year: int,
) -> set[str]:
    unstable_target_codes: set[str] = set()
    edges = groups.edges
    for year in years:
        period = f"{year}{year + 1}"
        period_edges = edges[edges["period"] == period]
        if period_edges.empty:
            continue
        unstable_a, unstable_b = _unstable_codes_from_edges(period_edges)

        weights_a = weights_by_year.get(str(year)) if year != target_year else None
        if year != target_year and weights_a is None:
            raise ValueError(f"Missing chained weights for {year} -> target")
        weights_b = weights_by_year.get(str(year + 1)) if (year + 1) != target_year else None
        if (year + 1) != target_year and weights_b is None:
            raise ValueError(f"Missing chained weights for {year + 1} -> target")

        unstable_target_codes |= _map_codes_to_target(unstable_a, weights_a)
        unstable_target_codes |= _map_codes_to_target(unstable_b, weights_b)

    return unstable_target_codes


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


def _lineage_to_year(
    *,
    side_map: pd.DataFrame,
    source_year: int,
    target_year: int,
    one_to_one_maps: dict[tuple[str, str], pd.DataFrame],
    revised_a_by_period: dict[str, set[str]],
    revised_b_by_period: dict[str, set[str]],
) -> pd.DataFrame:
    current = side_map[["group_id", "lineage_code", "native_code"]].copy()
    current["lineage_code"] = normalize_codes(current["lineage_code"])
    current["native_code"] = normalize_codes(current["native_code"])
    if source_year == target_year:
        current["complete"] = True
        return current

    for step in chain_steps(source_year, target_year):
        period = str(step["period"])
        direction = str(step["direction"])
        step_map = one_to_one_maps.get((period, direction))
        if step_map is None:
            step_map = pd.DataFrame(columns=["from_code", "to_code"])

        if direction == "a_to_b":
            revised_codes = revised_a_by_period.get(period, set())
        else:
            revised_codes = revised_b_by_period.get(period, set())

        matched = current.merge(
            step_map,
            left_on="native_code",
            right_on="from_code",
            how="left",
        )
        is_revised = matched["native_code"].isin(revised_codes)
        matched["complete"] = (~is_revised) | matched["to_code"].notna()
        matched["native_code"] = np.where(
            is_revised,
            matched["to_code"],
            matched["native_code"],
        )
        current = matched[["group_id", "lineage_code", "native_code", "complete"]].copy()
        current = current.loc[current["complete"]].drop_duplicates()
        if current.empty:
            return current

    current["native_code"] = normalize_codes(current["native_code"])
    return current


def _complete_lineage_maps(
    *,
    base_map: pd.DataFrame,
    base_year: int,
    target_years: Sequence[int],
    one_to_one_maps: dict[tuple[str, str], pd.DataFrame],
    revised_a_by_period: dict[str, set[str]],
    revised_b_by_period: dict[str, set[str]],
) -> tuple[dict[int, pd.DataFrame], set[str], dict[str, int]]:
    side_map = base_map.rename(columns={"target_code": "lineage_code"}).copy()
    side_map["native_code"] = side_map["lineage_code"]
    side_map["lineage_code"] = normalize_codes(side_map["lineage_code"])
    side_map["native_code"] = normalize_codes(side_map["native_code"])
    group_sizes = side_map.groupby("group_id", sort=False)["lineage_code"].nunique()

    maps_by_year: dict[int, pd.DataFrame] = {}
    retained_group_ids = set(group_sizes.index.astype(str).tolist())
    for year in target_years:
        lineage = _lineage_to_year(
            side_map=side_map,
            source_year=base_year,
            target_year=int(year),
            one_to_one_maps=one_to_one_maps,
            revised_a_by_period=revised_a_by_period,
            revised_b_by_period=revised_b_by_period,
        )
        counts = lineage.groupby("group_id", sort=False)["lineage_code"].nunique()
        complete_groups = set(counts.loc[counts.eq(group_sizes)].index.astype(str).tolist())
        retained_group_ids &= complete_groups
        maps_by_year[int(year)] = lineage

    maps_by_year = {
        year: lineage.loc[lineage["group_id"].isin(retained_group_ids)].copy()
        for year, lineage in maps_by_year.items()
    }
    diagnostics = {
        "n_groups_pre_lineage": int(len(group_sizes)),
        "n_groups_retained": int(len(retained_group_ids)),
        "n_groups_dropped_lineage": int(len(group_sizes) - len(retained_group_ids)),
    }
    return maps_by_year, retained_group_ids, diagnostics


def _shares_from_lineage_totals(
    *,
    totals: pd.DataFrame,
    lineage_map: pd.DataFrame,
) -> pd.DataFrame:
    values = totals[["PRODUCT_NC", "value"]].copy()
    values["PRODUCT_NC"] = normalize_codes(values["PRODUCT_NC"])
    values = values.rename(columns={"PRODUCT_NC": "native_code", "value": "native_value"})
    df = lineage_map.merge(values, on="native_code", how="left")
    df["value"] = pd.to_numeric(df["native_value"], errors="coerce").fillna(0.0)
    df = (
        df.groupby(["group_id", "lineage_code"], as_index=False, sort=False)["value"]
        .sum()
        .rename(columns={"lineage_code": "product_code"})
    )
    group_totals = (
        df.groupby("group_id", as_index=False, sort=False)["value"]
        .sum()
        .rename(columns={"value": "group_total"})
    )
    df = df.merge(group_totals, on="group_id", how="left")
    df = df.loc[df["group_total"] > 0].copy()
    df["share"] = df["value"] / df["group_total"]
    return df[["group_id", "product_code", "share", "value"]]


def _build_deterministic_lineage_placebo_panels(
    *,
    years,
    groups,
    break_period: str,
    group_ids: set[str],
    edges_period: pd.DataFrame,
    totals_by_year: dict[int, pd.DataFrame],
    focal_pair: PanelPair,
) -> tuple[list[PanelPair], pd.DataFrame]:
    break_a_year = int(break_period[:4])
    break_b_year = int(break_period[4:])
    if years.target != break_a_year:
        raise ValueError("deterministic_lineage_placebo currently requires target year to be break vintage A")

    one_to_one_maps = _strict_one_to_one_maps(groups)
    revised_a_by_period, revised_b_by_period = _period_revised_code_sets(groups)
    edges = edges_period.copy()
    edges["vintage_a_code"] = normalize_codes(edges["vintage_a_code"])
    edges["vintage_b_code"] = normalize_codes(edges["vintage_b_code"])
    focal_a_map = (
        edges[["vintage_a_code", "group_id"]]
        .drop_duplicates()
        .rename(columns={"vintage_a_code": "target_code"})
    )
    focal_b_map = (
        edges[["vintage_b_code", "group_id"]]
        .drop_duplicates()
        .rename(columns={"vintage_b_code": "target_code"})
    )
    focal_a_map = focal_a_map.loc[focal_a_map["group_id"].isin(group_ids)].copy()
    focal_b_map = focal_b_map.loc[focal_b_map["group_id"].isin(group_ids)].copy()

    pre_years = [year for year in range(years.start, break_a_year + 1)]
    post_years = [year for year in range(break_b_year, years.end + 1)]
    pre_maps, pre_retained, pre_diag = _complete_lineage_maps(
        base_map=focal_a_map,
        base_year=break_a_year,
        target_years=pre_years,
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
    )
    post_maps, post_retained, post_diag = _complete_lineage_maps(
        base_map=focal_b_map,
        base_year=break_b_year,
        target_years=post_years,
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
    )
    retained_group_ids = pre_retained & post_retained
    pre_maps = {
        year: lineage.loc[lineage["group_id"].isin(retained_group_ids)].copy()
        for year, lineage in pre_maps.items()
    }
    post_maps = {
        year: lineage.loc[lineage["group_id"].isin(retained_group_ids)].copy()
        for year, lineage in post_maps.items()
    }

    pre_shares = {
        year: _shares_from_lineage_totals(
            totals=totals_by_year[year],
            lineage_map=lineage,
        )
        for year, lineage in pre_maps.items()
    }
    post_shares = {
        year: _shares_from_lineage_totals(
            totals=totals_by_year[year],
            lineage_map=lineage,
        )
        for year, lineage in post_maps.items()
    }

    panel_pairs: list[PanelPair] = []
    diag_rows: list[dict[str, object]] = []
    for year in range(years.start, years.end):
        if year < break_a_year:
            left = pre_shares[year].rename(columns={"share": "share_t"})
            right = pre_shares[year + 1].rename(columns={"share": "share_t1"})
            panel = left.merge(
                right,
                on=["group_id", "product_code"],
                how="inner",
                suffixes=("_t", "_t1"),
            )
            mode = "deterministic_lineage"
            basis = f"CN{break_a_year}-side lineage"
            retained = retained_group_ids
            dropped = len(group_ids) - len(retained_group_ids)
        elif year == break_a_year:
            panel = focal_pair.data.loc[focal_pair.data["group_id"].isin(retained_group_ids)].copy()
            mode = "lt_converted_focal"
            basis = f"CN{break_a_year}"
            retained = retained_group_ids
            dropped = len(group_ids) - len(retained_group_ids)
        else:
            left = post_shares[year].rename(columns={"share": "share_t"})
            right = post_shares[year + 1].rename(columns={"share": "share_t1"})
            panel = left.merge(
                right,
                on=["group_id", "product_code"],
                how="inner",
                suffixes=("_t", "_t1"),
            )
            mode = "deterministic_lineage"
            basis = f"CN{break_b_year}-side lineage"
            retained = retained_group_ids
            dropped = len(group_ids) - len(retained_group_ids)
        panel_pairs.append(PanelPair(x_year=year, y_year=year + 1, data=panel))
        diag_rows.append(
            {
                "year_t": year,
                "year_t1": year + 1,
                "panel_mode": mode,
                "lineage_basis": basis,
                "n_groups_pre_lineage": int(len(group_ids)),
                "n_groups_retained": int(len(retained)),
                "n_groups_dropped_lineage": int(dropped),
                "n_points": int(len(panel)),
                "n_codes": int(panel["product_code"].nunique()) if not panel.empty else 0,
                "n_groups_in_panel": int(panel["group_id"].nunique()) if not panel.empty else 0,
                "pre_side_groups_retained": int(pre_diag["n_groups_retained"]),
                "post_side_groups_retained": int(post_diag["n_groups_retained"]),
            }
        )

    return panel_pairs, pd.DataFrame(diag_rows)


def run_share_stability_analysis(config: ShareStabilityConfig) -> dict[str, object]:
    """Run share-stability analysis and return metadata."""
    years = config.years
    measures = config.measures
    metrics_set = {name.lower() for name in config.metrics}

    groups = load_concordance_groups(
        concordance_path=config.paths.concordance_path,
        sheet_name=config.paths.concordance_sheet,
    )

    summary = get_ambiguous_group_summary(groups, config.break_config.direction)
    period_summary = summary[summary["period"] == config.break_config.period]
    group_ids = set(period_summary["group_id"].tolist())

    edges_period = groups.edges[groups.edges["period"] == config.break_config.period]
    vintage_a_year = config.break_config.period[:4]
    vintage_b_year = config.break_config.period[4:]
    if str(years.target) == vintage_a_year:
        group_map = edges_period[["vintage_a_code", "group_id"]].drop_duplicates()
        group_map = group_map.rename(columns={"vintage_a_code": "target_code"})
    elif str(years.target) == vintage_b_year:
        group_map = edges_period[["vintage_b_code", "group_id"]].drop_duplicates()
        group_map = group_map.rename(columns={"vintage_b_code": "target_code"})
    else:
        raise ValueError("target year must match the break period vintage")

    code_universe = build_code_universe_from_annual(
        annual_base_dir=config.paths.annual_base_dir,
        years=range(years.start, years.end + 1),
    )

    chained = build_chained_weights_for_range(
        start_year=years.start,
        end_year=years.end,
        target_year=years.target,
        measures=[measures.weights_source],
        code_universe=code_universe,
        weights_dir=config.paths.weights_dir,
        output_weights_dir=config.paths.output_dir / "chain" / f"CN{years.target}",
        output_diagnostics_dir=config.paths.output_dir / "chain" / f"CN{years.target}",
        finalize_weights=config.chaining.finalize_weights,
        neg_tol=config.chaining.neg_tol,
        pos_tol=config.chaining.pos_tol,
        row_sum_tol=config.chaining.row_sum_tol,
        fail_on_missing=True,
    )

    weights_by_year: dict[str, pd.DataFrame] = {}
    for output in chained:
        weights_by_year[output.origin_year] = output.weights

    group_ids_filtered = set(group_ids)
    unstable_target_codes: set[str] = set()
    if config.stability_filter.enabled and config.stability_filter.years:
        unstable_target_codes = _collect_unstable_target_codes(
            groups=groups,
            years=config.stability_filter.years,
            weights_by_year=weights_by_year,
            target_year=years.target,
        )
        if unstable_target_codes:
            unstable_groups = set(
                group_map[group_map["target_code"].isin(unstable_target_codes)]["group_id"].unique()
            )
            group_ids_filtered = group_ids_filtered - unstable_groups

    totals_by_year: dict[int, pd.DataFrame] = {}
    for year in range(years.start, years.end + 1):
        totals_by_year[year] = load_annual_totals(
            annual_base_dir=config.paths.annual_base_dir,
            year=year,
            measure=measures.analysis_measure,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
        )
    if years.target not in totals_by_year:
        totals_by_year[years.target] = load_annual_totals(
            annual_base_dir=config.paths.annual_base_dir,
            year=years.target,
            measure=measures.analysis_measure,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
        )
    year_shares = build_year_shares_from_totals(
        years=range(years.start, years.end + 1),
        target_year=years.target,
        totals_by_year=totals_by_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    )

    pairs: list[tuple[int, pd.DataFrame]] = []
    panel_stats: dict[int, dict[str, int]] = {}
    annotation_by_year: dict[int, str] = {}

    sample_group_ids = group_ids_filtered if group_ids_filtered != group_ids else group_ids
    sample_target_codes = set(
        group_map.loc[group_map["group_id"].isin(sample_group_ids), "target_code"]
        .astype(str)
        .tolist()
    )
    values_by_year: dict[int, pd.DataFrame] = {}

    def _values_for_year(year: int) -> pd.DataFrame:
        if year not in values_by_year:
            values_by_year[year] = build_values_for_groups_from_totals(
                year=year,
                target_year=years.target,
                totals_by_year=totals_by_year,
                weights_by_year=weights_by_year,
                group_map=group_map,
                group_ids=sample_group_ids,
            )
        return values_by_year[year]
    panel_pairs = build_panel_pairs(
        start_year=years.start,
        end_year=years.end,
        year_shares=year_shares,
        group_ids_filtered=group_ids_filtered if group_ids_filtered != group_ids else None,
    )
    for pair in progress(panel_pairs, desc="share_stability panels", total=len(panel_pairs)):
        merged = pair.data
        if group_ids_filtered != group_ids:
            left = year_shares[pair.x_year].rename(columns={"share": "share_t"})
            right = year_shares[pair.y_year].rename(columns={"share": "share_t1"})
            merged_all = left.merge(
                right,
                on=["group_id", "product_code"],
                how="inner",
            )
            n_codes_before = merged_all["product_code"].nunique()
            n_codes_after = merged["product_code"].nunique()
            panel_stats[pair.x_year] = {
                "n_codes_before": int(n_codes_before),
                "n_codes_after": int(n_codes_after),
                "n_codes_filtered": int(n_codes_before - n_codes_after),
            }
        else:
            panel_stats[pair.x_year] = {
                "n_codes_before": int(merged["product_code"].nunique()),
                "n_codes_after": int(merged["product_code"].nunique()),
                "n_codes_filtered": 0,
            }
        pairs.append((pair.x_year, merged))

    panel_diagnostics = pd.DataFrame(
        [
            {
                "year_t": year,
                "year_t1": year + 1,
                "panel_mode": "common_target",
                "lineage_basis": f"CN{years.target}",
                "n_groups_pre_lineage": int(len(sample_group_ids)),
                "n_groups_retained": int(len(sample_group_ids)),
                "n_groups_dropped_lineage": 0,
                "n_points": int(len(df)),
                "n_codes": int(df["product_code"].nunique()) if not df.empty else 0,
                "n_groups_in_panel": int(df["group_id"].nunique()) if not df.empty else 0,
            }
            for year, df in pairs
        ]
    )

    if config.comparison.mode == "deterministic_lineage_placebo":
        focal_year = int(vintage_a_year)
        focal_matches = [pair for pair in panel_pairs if pair.x_year == focal_year]
        if not focal_matches:
            raise ValueError("deterministic_lineage_placebo requires a focal break panel")
        panel_pairs, panel_diagnostics = _build_deterministic_lineage_placebo_panels(
            years=years,
            groups=groups,
            break_period=config.break_config.period,
            group_ids=sample_group_ids,
            edges_period=edges_period,
            totals_by_year=totals_by_year,
            focal_pair=focal_matches[0],
        )
        pairs = [(pair.x_year, pair.data) for pair in panel_pairs]
        panel_stats = {
            int(row.year_t): {
                "n_codes_before": int(row.n_codes),
                "n_codes_after": int(row.n_codes),
                "n_codes_filtered": 0,
            }
            for row in panel_diagnostics.itertuples(index=False)
        }

    summary_path = config.paths.output_dir / "summary.csv"
    panel_diagnostics_path = config.paths.output_dir / "panel_diagnostics.csv"
    comparison_path = config.paths.output_dir / "comparison_to_common_target.csv"
    summary_rows = []
    want_exposure = "exposure_weighted" in metrics_set
    want_diffuseness = "diffuseness_weighted" in metrics_set
    metrics_cache: dict[int, tuple[float, float]] = {}
    step_weights_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    feasible_map_cache: dict[tuple[str, str], dict[str, list[str]]] = {}

    def _metrics_for_year(year: int) -> tuple[float, float]:
        if year in metrics_cache:
            return metrics_cache[year]
        if year == years.target:
            metrics_cache[year] = (float("nan"), float("nan"))
            return metrics_cache[year]
        step_rows_chain = compute_step_metrics(
            base_year=year,
            target_year=years.target,
            sample_target_codes=sample_target_codes,
            weights_by_year=weights_by_year,
            groups=groups,
            annual_base_dir=config.paths.annual_base_dir,
            measure=measures.analysis_measure,
            weights_dir=config.paths.weights_dir,
            weights_source=measures.weights_source,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
            compute_exposure=want_exposure,
            compute_diffuseness=want_diffuseness,
            totals_by_year=totals_by_year,
            step_weights_cache=step_weights_cache,
            feasible_map_cache=feasible_map_cache,
        )
        steps_df = pd.DataFrame(step_rows_chain)
        exposure_weighted = float("nan")
        diffuseness_weighted = float("nan")
        if not steps_df.empty:
            if want_exposure:
                exp_vals = steps_df["ambiguity_exposure"].to_numpy(dtype=float)
                exp_w = steps_df["total_trade_sample"].to_numpy(dtype=float)
                exposure_weighted = weighted_mean(exp_vals, exp_w)
            if want_diffuseness:
                diff_vals = steps_df["diffuseness"].to_numpy(dtype=float)
                diff_w = steps_df["ambiguous_trade"].to_numpy(dtype=float)
                diffuseness_weighted = weighted_mean(diff_vals, diff_w)
        metrics_cache[year] = (exposure_weighted, diffuseness_weighted)
        return metrics_cache[year]

    for year, df in pairs:
        exposure_weighted = float("nan")
        diffuseness_weighted = float("nan")
        panel_step_years = {year, year + 1}
        if (want_exposure or want_diffuseness) and (years.target in panel_step_years):
            source_year = year if (year + 1) == years.target else year + 1
            exposure_weighted, diffuseness_weighted = _metrics_for_year(source_year)

        lines = []
        if "r2_45" in metrics_set:
            r2_val = r2_45(df["share_t"].to_numpy(), df["share_t1"].to_numpy())
            lines.append(rf"$R^2$ = {r2_val:.3f}")
        else:
            r2_val = float("nan")
        r2_w = float("nan")
        r2_sym = float("nan")
        mae_w = float("nan")
        need_weighted = {
            "r2_45_weighted",
            "r2_45_weighted_symmetric",
            "mae_weighted",
        } & metrics_set
        if need_weighted:
            values_y = _values_for_year(pair.y_year).rename(columns={"value": "value_y"})
            weights_y = df.merge(
                values_y, on=["group_id", "product_code"], how="left"
            )
            weights_y["value_y"] = weights_y["value_y"].fillna(0.0)
            if "r2_45_weighted" in metrics_set:
                r2_w = r2_45_weighted(
                    weights_y["share_t"].to_numpy(),
                    weights_y["share_t1"].to_numpy(),
                    weights_y["value_y"].to_numpy(),
                )
                lines.append(rf"$R^2_w$ = {r2_w:.3f}")
            if "r2_45_weighted_symmetric" in metrics_set or "mae_weighted" in metrics_set:
                values_x = _values_for_year(pair.x_year).rename(columns={"value": "value_x"})
                weights_xy = weights_y.merge(
                    values_x, on=["group_id", "product_code"], how="left"
                )
                weights_xy["value_x"] = weights_xy["value_x"].fillna(0.0)
                if "r2_45_weighted_symmetric" in metrics_set:
                    r2_sym = r2_45_weighted_symmetric(
                        weights_xy["share_t"].to_numpy(),
                        weights_xy["share_t1"].to_numpy(),
                        weights_xy["value_x"].to_numpy(),
                        weights_xy["value_y"].to_numpy(),
                    )
                    lines.append(rf"$R^2_{{sym}}$ = {r2_sym:.3f}")
                if "mae_weighted" in metrics_set:
                    mae_w = mae_weighted(
                        weights_xy["share_t"].to_numpy(),
                        weights_xy["share_t1"].to_numpy(),
                        weights_xy["value_x"].to_numpy(),
                        weights_xy["value_y"].to_numpy(),
                    )
                    lines.append(rf"wMAE = {mae_w:.3f}")
        if want_exposure:
            lines.append(rf"$E_w$ = {exposure_weighted:.3f}")
        if want_diffuseness:
            lines.append(rf"$H_w$ = {diffuseness_weighted:.3f}")
        diag_match = panel_diagnostics.loc[panel_diagnostics["year_t"] == year]
        panel_mode = str(diag_match.iloc[0]["panel_mode"]) if not diag_match.empty else "common_target"
        if config.comparison.mode == "deterministic_lineage_placebo":
            label = "LT focal" if panel_mode == "lt_converted_focal" else "native"
            lines.append(label)
        if lines:
            annotation_by_year[year] = "\n".join(lines)

        stats = panel_stats.get(year, {})
        summary_rows.append(
            {
                "year_t": year,
                "year_t1": year + 1,
                "n_points": len(df),
                "r2_45": r2_val,
                "ambiguity_exposure_weighted": (
                    exposure_weighted if want_exposure else float("nan")
                ),
                "diffuseness_weighted": (
                    diffuseness_weighted if want_diffuseness else float("nan")
                ),
                "n_codes_filtered": stats.get("n_codes_filtered", 0),
                "r2_45_weighted": r2_w,
                "r2_45_weighted_symmetric": r2_sym,
                "mae_weighted": mae_w,
                "panel_mode": panel_mode,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    panel_diagnostics.to_csv(panel_diagnostics_path, index=False)

    comparison_csv = None
    if (
        config.comparison.mode == "deterministic_lineage_placebo"
        and config.comparison.common_target_summary_path is not None
        and config.comparison.common_target_summary_path.exists()
    ):
        common = pd.read_csv(config.comparison.common_target_summary_path)
        comparison = common[["year_t", "year_t1", "r2_45"]].rename(
            columns={"r2_45": "r2_common_target"}
        ).merge(
            summary[["year_t", "year_t1", "r2_45", "panel_mode"]].rename(
                columns={"r2_45": "r2_deterministic_placebo"}
            ),
            on=["year_t", "year_t1"],
            how="outer",
        )
        comparison["delta_r2"] = (
            comparison["r2_deterministic_placebo"] - comparison["r2_common_target"]
        )
        comparison.to_csv(comparison_path, index=False)
        comparison_csv = str(comparison_path)

    plot_share_panels(
        pairs=panel_pairs,
        output_path=config.plot.output_path,
        title=config.plot.title,
        point_alpha=config.plot.point_alpha,
        point_size=config.plot.point_size,
        axis_padding=config.plot.axis_padding,
        point_color=config.plot.point_color,
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
        annotation_by_year=annotation_by_year if annotation_by_year else None,
        annotation_pos=(
            (0.05, 0.82)
            if config.comparison.mode == "deterministic_lineage_placebo"
            else (0.05, 0.9)
        ),
    )

    return {
        "output_plot": str(config.plot.output_path),
        "summary_csv": str(summary_path),
        "panel_diagnostics_csv": str(panel_diagnostics_path),
        "comparison_csv": comparison_csv,
        "config": asdict(config),
    }
