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
    build_panel_pairs,
    build_values_for_groups_from_totals,
    build_year_shares_from_totals,
    normalize_codes,
)
from ..common.steps import compute_step_metrics
from ..common.steps import load_annual_totals
from ...estimation.chaining import build_chained_weights_for_range, build_code_universe_from_annual
from ...estimation.runner import load_concordance_groups
from ...mappings import get_ambiguous_group_summary


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

    summary_path = config.paths.output_dir / "summary.csv"
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
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

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
    )

    return {
        "output_plot": str(config.plot.output_path),
        "summary_csv": str(summary_path),
        "config": asdict(config),
    }
