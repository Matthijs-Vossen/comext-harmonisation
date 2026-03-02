"""Stress-test share stability analysis across long chains to a target vintage."""

from __future__ import annotations

from dataclasses import asdict
import pandas as pd

from ..common.metrics import (
    mae_weighted,
    r2_45,
    r2_45_weighted,
    r2_45_weighted_symmetric,
    weighted_mean,
)
from ..common.plotting import plot_share_panels
from ..common.progress import progress
from ..common.chain_sampling import (
    build_chain_group_map as _build_chain_group_map_common,
)
from ..common.shares import (
    PanelPair,
    build_values_for_groups_from_totals,
    build_year_shares_from_totals,
)
from ..common.steps import compute_step_metrics, load_annual_totals
from ..config import StressConfig
from ...estimation.chaining import build_chained_weights_for_range, build_code_universe_from_annual
from ...estimation.runner import load_concordance_groups


def _build_chain_group_map(
    *,
    groups,
    base_year: int,
    target_year: int,
    weights_by_year: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, set[str], set[str]]:
    return _build_chain_group_map_common(
        groups=groups,
        base_year=base_year,
        target_year=target_year,
        weights_by_year=weights_by_year,
        preserve_unmapped=True,
    )


def _merge_shares(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.merge(right, on=["group_id", "product_code"], how="inner")


def _build_share_pair(
    *,
    year_x: int,
    year_y: int,
    target_year: int,
    totals_by_year: dict[int, pd.DataFrame],
    weights_by_year: dict[str, pd.DataFrame],
    group_map: pd.DataFrame,
    group_ids: set[str],
) -> pd.DataFrame:
    shares_by_year = build_year_shares_from_totals(
        years=[year_x, year_y],
        target_year=target_year,
        totals_by_year=totals_by_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    )
    left = shares_by_year[year_x].rename(columns={"share": "share_t"})
    right = shares_by_year[year_y].rename(columns={"share": "share_t1"})
    return _merge_shares(left, right)


def _build_weighted_pair_frame(
    *,
    merged: pd.DataFrame,
    year_x: int,
    year_y: int,
    target_year: int,
    totals_by_year: dict[int, pd.DataFrame],
    weights_by_year: dict[str, pd.DataFrame],
    group_map: pd.DataFrame,
    group_ids: set[str],
) -> pd.DataFrame:
    values_y = build_values_for_groups_from_totals(
        year=year_y,
        target_year=target_year,
        totals_by_year=totals_by_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    ).rename(columns={"value": "value_y"})
    weighted = merged.merge(values_y, on=["group_id", "product_code"], how="left")
    weighted["value_y"] = weighted["value_y"].fillna(0.0)

    values_x = build_values_for_groups_from_totals(
        year=year_x,
        target_year=target_year,
        totals_by_year=totals_by_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    ).rename(columns={"value": "value_x"})
    weighted = weighted.merge(values_x, on=["group_id", "product_code"], how="left")
    weighted["value_x"] = weighted["value_x"].fillna(0.0)
    return weighted


def _compute_weighted_pair_metrics(
    *,
    merged: pd.DataFrame,
    base_year: int,
    compare_year: int,
    target_year: int,
    totals_by_year: dict[int, pd.DataFrame],
    weights_by_year: dict[str, pd.DataFrame],
    group_map: pd.DataFrame,
    group_ids: set[str],
    metrics_set: set[str],
) -> tuple[float, float]:
    r2_sym = float("nan")
    mae_w = float("nan")
    need_weighted = {"r2_45_weighted_symmetric", "mae_weighted"} & metrics_set
    if not need_weighted:
        return r2_sym, mae_w

    weighted = _build_weighted_pair_frame(
        merged=merged,
        year_x=base_year,
        year_y=compare_year,
        target_year=target_year,
        totals_by_year=totals_by_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    )
    if "r2_45_weighted_symmetric" in metrics_set:
        r2_sym = r2_45_weighted_symmetric(
            weighted["share_t"].to_numpy(),
            weighted["share_t1"].to_numpy(),
            weighted["value_x"].to_numpy(),
            weighted["value_y"].to_numpy(),
        )
    if "mae_weighted" in metrics_set:
        mae_w = mae_weighted(
            weighted["share_t"].to_numpy(),
            weighted["share_t1"].to_numpy(),
            weighted["value_x"].to_numpy(),
            weighted["value_y"].to_numpy(),
        )
    return r2_sym, mae_w


def _compute_step_aggregates(
    *,
    step_rows_chain: list[dict[str, object]],
    metrics_set: set[str],
) -> tuple[float, float, float]:
    exposure_weighted = float("nan")
    diffuseness_weighted = float("nan")
    diffuse_exposure_weighted = float("nan")
    steps_df = pd.DataFrame(step_rows_chain)
    if steps_df.empty:
        return exposure_weighted, diffuseness_weighted, diffuse_exposure_weighted

    if "exposure_weighted" in metrics_set:
        exp_vals = steps_df["ambiguity_exposure"].to_numpy(dtype=float)
        exp_w = steps_df["total_trade_sample"].to_numpy(dtype=float)
        exposure_weighted = weighted_mean(exp_vals, exp_w)
    if "diffuseness_weighted" in metrics_set:
        diff_vals = steps_df["diffuseness"].to_numpy(dtype=float)
        diff_w = steps_df["ambiguous_trade"].to_numpy(dtype=float)
        diffuseness_weighted = weighted_mean(diff_vals, diff_w)
    if "diffuse_exposure" in metrics_set:
        d_vals = steps_df["diffuse_exposure"].to_numpy(dtype=float)
        d_w = steps_df["total_trade_sample"].to_numpy(dtype=float)
        diffuse_exposure_weighted = weighted_mean(d_vals, d_w)
    return exposure_weighted, diffuseness_weighted, diffuse_exposure_weighted


def run_stress_test_analysis(config: StressConfig) -> dict[str, object]:
    years = config.years
    measures = config.measures
    metrics_set = {name.lower() for name in config.metrics}
    want_exposure = "exposure_weighted" in metrics_set
    want_diffuse_exposure = "diffuse_exposure" in metrics_set
    want_diffuseness = ("diffuseness_weighted" in metrics_set) or want_diffuse_exposure

    groups = load_concordance_groups(
        concordance_path=config.paths.concordance_path,
        sheet_name=config.paths.concordance_sheet,
    )

    base_years = [chain.base_year for chain in years.chains]
    compare_years = [chain.compare_year for chain in years.chains]
    min_year = min([years.target, *base_years, *compare_years])
    max_year = max([years.target, *base_years, *compare_years])

    code_universe = build_code_universe_from_annual(
        annual_base_dir=config.paths.annual_base_dir,
        years=range(min_year, max_year + 1),
    )

    chained = build_chained_weights_for_range(
        start_year=min_year,
        end_year=max_year,
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

    totals_by_year: dict[int, pd.DataFrame] = {}
    for year in range(min_year, max_year + 1):
        totals_by_year[year] = load_annual_totals(
            annual_base_dir=config.paths.annual_base_dir,
            year=year,
            measure=measures.analysis_measure,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
        )
    step_weights_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    feasible_map_cache: dict[tuple[str, str], dict[str, list[str]]] = {}

    panel_pairs: list[PanelPair] = []
    annotation_by_year: dict[int, str] = {}
    summary_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []

    for chain in progress(years.chains, desc="stress_test chains", total=len(years.chains)):
        base_year = int(chain.base_year)
        compare_year = int(chain.compare_year)
        group_map, group_ids, sample_codes = _build_chain_group_map(
            groups=groups,
            base_year=base_year,
            target_year=years.target,
            weights_by_year=weights_by_year,
        )

        if group_map.empty or not group_ids or not sample_codes:
            empty = pd.DataFrame(
                columns=["group_id", "product_code", "share_t", "share_t1"]
            )
            panel_pairs.append(PanelPair(x_year=base_year, y_year=compare_year, data=empty))
            summary_rows.append(
                {
                    "base_year": base_year,
                    "compare_year": compare_year,
                    "n_points": 0,
                    "r2_45": float("nan"),
                    "ambiguity_exposure_weighted": float("nan"),
                    "diffuseness_weighted": float("nan"),
                }
            )
            continue

        step_rows_chain = compute_step_metrics(
            base_year=base_year,
            target_year=years.target,
            sample_target_codes=sample_codes,
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

        merged = _build_share_pair(
            year_x=base_year,
            year_y=compare_year,
            target_year=years.target,
            totals_by_year=totals_by_year,
            weights_by_year=weights_by_year,
            group_map=group_map,
            group_ids=group_ids,
        )
        panel_pairs.append(PanelPair(x_year=base_year, y_year=compare_year, data=merged))
        r2_val = (
            r2_45(merged["share_t"].to_numpy(), merged["share_t1"].to_numpy())
            if "r2_45" in metrics_set
            else float("nan")
        )
        r2_sym, mae_w = _compute_weighted_pair_metrics(
            merged=merged,
            base_year=base_year,
            compare_year=compare_year,
            target_year=years.target,
            totals_by_year=totals_by_year,
            weights_by_year=weights_by_year,
            group_map=group_map,
            group_ids=group_ids,
            metrics_set=metrics_set,
        )
        exposure_weighted, diffuseness_weighted, diffuse_exposure_weighted = _compute_step_aggregates(
            step_rows_chain=step_rows_chain,
            metrics_set=metrics_set,
        )
        lines = []
        if "r2_45" in metrics_set:
            lines.append(rf"$R^2$ = {r2_val:.3f}")
        if "r2_45_weighted_symmetric" in metrics_set:
            lines.append(rf"$R^2_w$ = {r2_sym:.3f}")
        if "mae_weighted" in metrics_set:
            lines.append(rf"wMAE = {mae_w:.3f}")
        if "exposure_weighted" in metrics_set:
            lines.append(rf"$E_w$ = {exposure_weighted:.3f}")
        if "diffuseness_weighted" in metrics_set:
            lines.append(rf"$H_w$ = {diffuseness_weighted:.3f}")
        if "diffuse_exposure" in metrics_set:
            lines.append(rf"$D_w$ = {diffuse_exposure_weighted:.3f}")
        annotation_by_year[base_year] = "\n".join(lines)
        summary_rows.append(
            {
                "base_year": base_year,
                "compare_year": compare_year,
                "n_points": int(len(merged)),
                "r2_45": r2_val,
                "r2_45_weighted_symmetric": r2_sym,
                "mae_weighted": mae_w,
                "ambiguity_exposure_weighted": (
                    exposure_weighted if "exposure_weighted" in metrics_set else float("nan")
                ),
                "diffuseness_weighted": (
                    diffuseness_weighted if "diffuseness_weighted" in metrics_set else float("nan")
                ),
                "diffuse_exposure_weighted": (
                    diffuse_exposure_weighted
                    if "diffuse_exposure" in metrics_set
                    else float("nan")
                ),
            }
        )
        step_rows.extend(step_rows_chain)

    if annotation_by_year:
        line_count = max(1, len(next(iter(annotation_by_year.values())).splitlines()))
        annotation_y = 0.9 - 0.06 * (line_count - 1)
    else:
        annotation_y = 0.9

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
        annotation_by_year=annotation_by_year,
        annotation_pos=(0.05, annotation_y),
    )

    summary = pd.DataFrame(summary_rows)
    summary_path = config.paths.output_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    step_metrics = pd.DataFrame(step_rows)
    step_metrics_path = config.paths.output_dir / "step_metrics.csv"
    if not step_metrics.empty:
        step_metrics.to_csv(step_metrics_path, index=False)

    return {
        "output_plot": str(config.plot.output_path),
        "summary_csv": str(summary_path),
        "step_metrics_csv": str(step_metrics_path) if step_rows else "",
        "config": asdict(config),
    }
