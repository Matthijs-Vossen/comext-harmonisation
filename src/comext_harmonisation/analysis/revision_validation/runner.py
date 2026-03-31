"""Revision-by-revision backward validation heatmap analysis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Mapping

import pandas as pd

from ..common.metrics import mae, mae_weighted, r2_45, weighted_mean
from ..common.plotting import plot_revision_validation_heatmap
from ..common.progress import progress
from ..common.shares import (
    build_panel_pairs,
    build_values_for_groups_from_totals,
    build_year_shares_from_totals,
)
from ..common.steps import load_annual_totals
from ..config import RevisionValidationConfig
from ..sampling_robustness.runner import (
    _annotate_for_holdout,
    _build_bin_assignments,
    _build_link_summary,
    _estimate_ambiguous_weights,
    _filter_holdout_rows,
    _group_maps_for_period,
    _importance_frame,
    _load_annual_frame,
    _reported_ambiguous_weights,
)
from ..share_stability.runner import _collect_unstable_target_codes
from ...chaining.engine import build_chained_weights_for_range, build_code_universe_from_annual
from ...concordance.mappings import get_ambiguous_group_summary
from ...estimation.runner import load_concordance_groups


def _enumerate_internal_target_years(*, min_year: int, max_year: int) -> list[int]:
    return list(range(min_year + 1, max_year - 1))


def _period_for_target_year(target_year: int) -> str:
    return f"{target_year}{target_year + 1}"


def _panel_label(*, x_year: int, target_year: int) -> str:
    if x_year == target_year - 1:
        return "pre_immediate"
    if x_year == target_year:
        return "break"
    if x_year == target_year + 1:
        return "post_immediate"
    raise ValueError(f"Unexpected panel year {x_year} for target year {target_year}")


def _importance_weighted_mean_instability(link_summary: pd.DataFrame) -> float:
    return weighted_mean(
        pd.to_numeric(link_summary["max_minus_min"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(link_summary["importance_product"], errors="coerce").to_numpy(dtype=float),
    )


def _skipped_summary_row(*, period: str, target_year: int, reason: str) -> dict[str, object]:
    return {
        "period": period,
        "target_year": int(target_year),
        "status": "skipped",
        "skip_reason": reason,
        "pre_fit_immediate": float("nan"),
        "break_fit": float("nan"),
        "post_fit_immediate": float("nan"),
        "placebo_baseline_fit": float("nan"),
        "break_penalty": float("nan"),
        "pre_mae_immediate": float("nan"),
        "break_mae": float("nan"),
        "break_year_mae": float("nan"),
        "post_mae_immediate": float("nan"),
        "non_revised_mae": float("nan"),
        "excess_break_mae": float("nan"),
        "pre_wmae_immediate": float("nan"),
        "break_wmae": float("nan"),
        "post_wmae_immediate": float("nan"),
        "non_revised_wmae": float("nan"),
        "excess_break_wmae": float("nan"),
        "n_points_pre_immediate": float("nan"),
        "n_points_break": float("nan"),
        "n_points_post_immediate": float("nan"),
        "n_observations": float("nan"),
        "n_groups": float("nan"),
        "n_links": float("nan"),
        "coverage_complete_share": float("nan"),
        "max_missing_run_count": float("nan"),
        "instability_p50": float("nan"),
        "instability_p90": float("nan"),
        "instability_importance_weighted_mean": float("nan"),
    }


def _summarize_panel_metrics(panel_details: pd.DataFrame) -> dict[str, float | int]:
    panel_by_label = panel_details.set_index("panel_label")
    pre_immediate = float(panel_by_label.loc["pre_immediate", "r2_45"])
    break_fit = float(panel_by_label.loc["break", "r2_45"])
    post_immediate = float(panel_by_label.loc["post_immediate", "r2_45"])
    placebo_baseline_fit = 0.5 * (pre_immediate + post_immediate)
    pre_mae = float(panel_by_label.loc["pre_immediate", "mae"])
    break_mae = float(panel_by_label.loc["break", "mae"])
    post_mae = float(panel_by_label.loc["post_immediate", "mae"])
    non_revised_mae = 0.5 * (pre_mae + post_mae)
    pre_wmae = float(panel_by_label.loc["pre_immediate", "mae_weighted"])
    break_wmae = float(panel_by_label.loc["break", "mae_weighted"])
    post_wmae = float(panel_by_label.loc["post_immediate", "mae_weighted"])
    non_revised_wmae = 0.5 * (pre_wmae + post_wmae)
    return {
        "pre_fit_immediate": pre_immediate,
        "break_fit": break_fit,
        "post_fit_immediate": post_immediate,
        "placebo_baseline_fit": placebo_baseline_fit,
        "break_penalty": placebo_baseline_fit - break_fit,
        "pre_mae_immediate": pre_mae,
        "break_mae": break_mae,
        "break_year_mae": break_mae,
        "post_mae_immediate": post_mae,
        "non_revised_mae": non_revised_mae,
        "excess_break_mae": break_mae - non_revised_mae,
        "pre_wmae_immediate": pre_wmae,
        "break_wmae": break_wmae,
        "post_wmae_immediate": post_wmae,
        "non_revised_wmae": non_revised_wmae,
        "excess_break_wmae": break_wmae - non_revised_wmae,
        "n_points_pre_immediate": int(panel_by_label.loc["pre_immediate", "n_points"]),
        "n_points_break": int(panel_by_label.loc["break", "n_points"]),
        "n_points_post_immediate": int(panel_by_label.loc["post_immediate", "n_points"]),
    }


def _compute_panel_details_for_period(
    *,
    config: RevisionValidationConfig,
    groups,
    period: str,
    target_year: int,
    group_ids: set[str],
    code_universe: Mapping[int, set[str]],
    totals_by_year: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    if not group_ids:
        raise ValueError(f"revision_validation {period}: no ambiguous groups found")

    edges_period = groups.edges[groups.edges["period"] == period]
    group_map = (
        edges_period[["vintage_a_code", "group_id"]]
        .drop_duplicates()
        .rename(columns={"vintage_a_code": "target_code"})
    )

    chained = build_chained_weights_for_range(
        start_year=target_year - 1,
        end_year=target_year + 2,
        target_year=target_year,
        measures=[config.measures.weights_source],
        code_universe=code_universe,
        weights_dir=config.paths.weights_dir,
        output_weights_dir=config.paths.output_dir / "chain" / f"CN{target_year}",
        output_diagnostics_dir=config.paths.output_dir / "chain" / f"CN{target_year}",
        finalize_weights=config.chaining.finalize_weights,
        neg_tol=config.chaining.neg_tol,
        pos_tol=config.chaining.pos_tol,
        row_sum_tol=config.chaining.row_sum_tol,
        fail_on_missing=True,
    )
    weights_by_year = {output.origin_year: output.weights for output in chained}

    group_ids_filtered = set(group_ids)
    unstable_target_codes = _collect_unstable_target_codes(
        groups=groups,
        years=[target_year - 1, target_year + 1],
        weights_by_year=weights_by_year,
        target_year=target_year,
    )
    if unstable_target_codes:
        unstable_groups = set(
            group_map[group_map["target_code"].isin(unstable_target_codes)]["group_id"].unique()
        )
        group_ids_filtered -= unstable_groups
    if not group_ids_filtered:
        raise ValueError(
            f"revision_validation {period}: stability filter removed all focal groups"
        )

    year_shares = build_year_shares_from_totals(
        years=range(target_year - 1, target_year + 3),
        target_year=target_year,
        totals_by_year=totals_by_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    )
    panel_pairs = build_panel_pairs(
        start_year=target_year - 1,
        end_year=target_year + 2,
        year_shares=year_shares,
        group_ids_filtered=group_ids_filtered if group_ids_filtered != group_ids else None,
    )
    values_by_year: dict[int, pd.DataFrame] = {}

    def _values_for_year(year: int) -> pd.DataFrame:
        if year not in values_by_year:
            values_by_year[year] = build_values_for_groups_from_totals(
                year=year,
                target_year=target_year,
                totals_by_year=totals_by_year,
                weights_by_year=weights_by_year,
                group_map=group_map,
                group_ids=group_ids_filtered,
            )
        return values_by_year[year]

    rows: list[dict[str, object]] = []
    for pair in panel_pairs:
        values_y = _values_for_year(pair.y_year).rename(columns={"value": "value_y"})
        weights_y = pair.data.merge(values_y, on=["group_id", "product_code"], how="left")
        weights_y["value_y"] = weights_y["value_y"].fillna(0.0)
        values_x = _values_for_year(pair.x_year).rename(columns={"value": "value_x"})
        weights_xy = weights_y.merge(values_x, on=["group_id", "product_code"], how="left")
        weights_xy["value_x"] = weights_xy["value_x"].fillna(0.0)
        rows.append(
            {
                "period": period,
                "target_year": int(target_year),
                "panel_label": _panel_label(x_year=pair.x_year, target_year=target_year),
                "year_t": int(pair.x_year),
                "year_t1": int(pair.y_year),
                "r2_45": float(
                    r2_45(
                        pair.data["share_t"].to_numpy(),
                        pair.data["share_t1"].to_numpy(),
                    )
                ),
                "mae": float(
                    mae(
                        pair.data["share_t"].to_numpy(),
                        pair.data["share_t1"].to_numpy(),
                    )
                ),
                "mae_weighted": float(
                    mae_weighted(
                        weights_xy["share_t"].to_numpy(),
                        weights_xy["share_t1"].to_numpy(),
                        weights_xy["value_x"].to_numpy(),
                        weights_xy["value_y"].to_numpy(),
                    )
                ),
                "n_points": int(len(pair.data)),
                "n_groups": int(pair.data["group_id"].nunique()),
                "n_codes": int(pair.data["product_code"].nunique()),
                "n_focal_groups_before_filter": int(len(group_ids)),
                "n_focal_groups_after_filter": int(len(group_ids_filtered)),
            }
        )
    return pd.DataFrame(rows).sort_values("year_t").reset_index(drop=True)


def _compute_sampling_robustness_for_period(
    *,
    config: RevisionValidationConfig,
    groups,
    period: str,
    target_year: int,
    raw_frames_by_year: dict[int, pd.DataFrame],
) -> tuple[dict[str, float | int], pd.DataFrame]:
    measure = config.measures.weights_source
    raw_a = raw_frames_by_year[target_year]
    raw_b = raw_frames_by_year[target_year + 1]

    full_estimation, full_matrices, full_weights, _ = _estimate_ambiguous_weights(
        groups=groups,
        period=period,
        direction=config.break_config.direction,
        measure=measure,
        flow_code=config.flow.flow_code,
        data_a=raw_a,
        data_b=raw_b,
    )
    full_weights = _reported_ambiguous_weights(full_weights)
    if full_weights.empty:
        raise ValueError(f"revision_validation {period}: no ambiguous weights estimated")

    assignments = _build_bin_assignments(
        matrices=full_matrices,
        n_bins=config.run.n_bins,
        seed=config.run.seed,
    )
    a_map, b_map = _group_maps_for_period(groups=groups, period=period)
    raw_a_annotated = _annotate_for_holdout(raw_a, group_map=a_map)
    raw_b_annotated = _annotate_for_holdout(raw_b, group_map=b_map)

    run_frames: list[pd.DataFrame] = []
    for omitted_bin in range(config.run.n_bins):
        holdout_keys = set(
            assignments.loc[assignments["bin_id"] == omitted_bin, "holdout_key"]
            .astype(str)
            .tolist()
        )
        subsample_a = _filter_holdout_rows(
            raw_a_annotated,
            holdout_keys=holdout_keys,
            measure=measure,
        )
        subsample_b = _filter_holdout_rows(
            raw_b_annotated,
            holdout_keys=holdout_keys,
            measure=measure,
        )
        _, _, run_weights, _ = _estimate_ambiguous_weights(
            groups=groups,
            period=period,
            direction=config.break_config.direction,
            measure=measure,
            flow_code=config.flow.flow_code,
            data_a=subsample_a,
            data_b=subsample_b,
        )
        run_weights = _reported_ambiguous_weights(run_weights).copy()
        run_weights["run_id"] = int(omitted_bin)
        run_weights["omitted_bin"] = int(omitted_bin)
        run_frames.append(run_weights)

    run_weights = pd.concat(run_frames, ignore_index=True)
    importance = _importance_frame(
        estimation=full_estimation,
        direction=config.break_config.direction,
        full_weights=full_weights,
    )
    link_summary = _build_link_summary(
        full_weights=full_weights,
        run_weights=run_weights,
        importance=importance,
        n_runs=config.run.n_bins,
    )
    if not bool(link_summary["coverage_complete"].all()):
        raise RuntimeError(f"revision_validation {period}: incomplete link coverage across runs")

    link_summary["period"] = period
    link_summary["target_year"] = int(target_year)
    instability = pd.to_numeric(link_summary["max_minus_min"], errors="coerce")
    return (
        {
            "n_observations": int(len(assignments)),
            "n_groups": int(assignments["group_id"].nunique()),
            "n_links": int(len(link_summary)),
            "coverage_complete_share": float(link_summary["coverage_complete"].mean()),
            "max_missing_run_count": int(link_summary["missing_run_count"].max()),
            "instability_p50": float(instability.quantile(0.50)),
            "instability_p90": float(instability.quantile(0.90)),
            "instability_importance_weighted_mean": float(
                _importance_weighted_mean_instability(link_summary)
            ),
        },
        link_summary.sort_values(["group_id", "from_code", "to_code"]).reset_index(drop=True),
    )


def _compute_revision_result(
    *,
    config: RevisionValidationConfig,
    groups,
    period: str,
    target_year: int,
    group_ids: set[str],
    code_universe: Mapping[int, set[str]],
    totals_by_year: dict[int, pd.DataFrame],
    raw_frames_by_year: dict[int, pd.DataFrame],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    panel_details = _compute_panel_details_for_period(
        config=config,
        groups=groups,
        period=period,
        target_year=target_year,
        group_ids=group_ids,
        code_universe=code_universe,
        totals_by_year=totals_by_year,
    )
    sampling_summary, link_summary = _compute_sampling_robustness_for_period(
        config=config,
        groups=groups,
        period=period,
        target_year=target_year,
        raw_frames_by_year=raw_frames_by_year,
    )
    summary_row = {
        "period": period,
        "target_year": int(target_year),
        "status": "ok",
        "skip_reason": "",
        **_summarize_panel_metrics(panel_details),
        **sampling_summary,
    }
    return summary_row, panel_details, link_summary


def run_revision_validation_analysis(config: RevisionValidationConfig) -> dict[str, object]:
    groups = load_concordance_groups(
        concordance_path=config.paths.concordance_path,
        sheet_name=config.paths.concordance_sheet,
    )
    period_summary = get_ambiguous_group_summary(groups, config.break_config.direction)
    group_ids_by_period = period_summary.groupby("period", sort=False)["group_id"].apply(
        lambda s: set(s.tolist())
    ).to_dict()

    target_years = _enumerate_internal_target_years(
        min_year=config.years.min_year,
        max_year=config.years.max_year,
    )
    code_universe = build_code_universe_from_annual(
        annual_base_dir=config.paths.annual_base_dir,
        years=list(range(config.years.min_year, config.years.max_year + 1)),
    )
    totals_by_year = {
        year: load_annual_totals(
            annual_base_dir=config.paths.annual_base_dir,
            year=year,
            measure=config.measures.analysis_measure,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
        )
        for year in range(config.years.min_year, config.years.max_year + 1)
    }
    raw_frames_by_year = {
        year: _load_annual_frame(
            annual_base_dir=config.paths.annual_base_dir,
            year=year,
            measure=config.measures.weights_source,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
        )
        for year in range(config.years.min_year, config.years.max_year + 1)
    }

    revision_specs = [
        (
            target_year,
            _period_for_target_year(target_year),
            set(group_ids_by_period.get(_period_for_target_year(target_year), set())),
        )
        for target_year in target_years
    ]

    results: list[tuple[dict[str, object], pd.DataFrame, pd.DataFrame]] = []
    if config.run.max_workers == 1:
        for target_year, period, group_ids in progress(
            revision_specs,
            desc="revision_validation periods",
            total=len(revision_specs),
        ):
            if not group_ids:
                results.append(
                    (
                        _skipped_summary_row(
                            period=period,
                            target_year=target_year,
                            reason="no_ambiguous_groups",
                        ),
                        pd.DataFrame(),
                        pd.DataFrame(),
                    )
                )
                continue
            results.append(
                _compute_revision_result(
                    config=config,
                    groups=groups,
                    period=period,
                    target_year=target_year,
                    group_ids=group_ids,
                    code_universe=code_universe,
                    totals_by_year=totals_by_year,
                    raw_frames_by_year=raw_frames_by_year,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=config.run.max_workers) as executor:
            future_to_spec = {
                executor.submit(
                    _compute_revision_result,
                    config=config,
                    groups=groups,
                    period=period,
                    target_year=target_year,
                    group_ids=group_ids,
                    code_universe=code_universe,
                    totals_by_year=totals_by_year,
                    raw_frames_by_year=raw_frames_by_year,
                ): (target_year, period)
                for target_year, period, group_ids in revision_specs
                if group_ids
            }
            for target_year, period, group_ids in revision_specs:
                if not group_ids:
                    results.append(
                        (
                            _skipped_summary_row(
                                period=period,
                                target_year=target_year,
                                reason="no_ambiguous_groups",
                            ),
                            pd.DataFrame(),
                            pd.DataFrame(),
                        )
                    )
            for future in progress(
                as_completed(future_to_spec),
                desc="revision_validation periods",
                total=len(future_to_spec),
            ):
                target_year, period = future_to_spec[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # pragma: no cover - exercised via future wrapping
                    raise RuntimeError(
                        f"revision_validation {period}: revision worker failed"
                    ) from exc

    summary_rows = [summary_row for summary_row, _, _ in results]
    panel_frames = [panel_details for _, panel_details, _ in results if not panel_details.empty]
    link_frames = [link_summary for _, _, link_summary in results if not link_summary.empty]

    summary = pd.DataFrame(summary_rows).sort_values("target_year").reset_index(drop=True)
    if panel_frames:
        panel_details = (
            pd.concat(panel_frames, ignore_index=True)
            .sort_values(["target_year", "year_t"])
            .reset_index(drop=True)
        )
    else:
        panel_details = pd.DataFrame()
    if link_frames:
        link_summary = (
            pd.concat(link_frames, ignore_index=True)
            .sort_values(["target_year", "group_id", "from_code", "to_code"])
            .reset_index(drop=True)
        )
    else:
        link_summary = pd.DataFrame()

    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.output.summary_csv, index=False)
    panel_details.to_csv(config.output.panel_details_csv, index=False)
    link_summary.to_csv(config.output.link_summary_csv, index=False)

    plot_revision_validation_heatmap(
        data=summary,
        output_path=config.plot.output_path,
        title=config.plot.title,
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
        show_annotations=config.plot.show_annotations,
    )

    return {
        "output_plot": config.plot.output_path,
        "summary_csv": config.output.summary_csv,
        "panel_details_csv": config.output.panel_details_csv,
        "link_summary_csv": config.output.link_summary_csv,
    }
