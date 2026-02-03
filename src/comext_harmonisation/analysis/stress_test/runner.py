"""Stress-test share stability analysis across long chains to a target vintage."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
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
from ..common.shares import (
    PanelPair,
    build_values_for_groups_from_totals,
    build_year_shares_from_totals,
    normalize_codes,
)
from ..common.steps import compute_step_metrics, chain_steps, load_annual_totals
from ..config import StressConfig
from ...estimation.chaining import build_chained_weights_for_range, build_code_universe_from_annual
from ...estimation.runner import load_concordance_groups
from ...mappings import get_ambiguous_group_summary


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: str) -> str:
        self.add(item)
        parent = self._parent[item]
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self._size[root_left] < self._size[root_right]:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        self._size[root_left] += self._size[root_right]


def _map_codes_to_target(codes: Iterable[str], weights: pd.DataFrame | None) -> set[str]:
    codes_list = [str(code) for code in codes]
    if not codes_list:
        return set()
    codes_series = normalize_codes(pd.Series(codes_list))
    if weights is None:
        return set(codes_series.tolist())

    weights = weights[["from_code", "to_code"]].copy()
    weights["from_code"] = normalize_codes(weights["from_code"])
    weights["to_code"] = normalize_codes(weights["to_code"])

    mapped = weights[weights["from_code"].isin(codes_series)]["to_code"].unique().tolist()
    mapped_set = set(mapped)
    missing = set(codes_series.tolist()) - set(weights["from_code"].unique())
    if missing:
        mapped_set |= missing
    return mapped_set


def _ambiguous_edges_for_step(
    *,
    groups,
    period: str,
    direction: str,
) -> pd.DataFrame:
    summary = get_ambiguous_group_summary(groups, direction)
    summary_period = summary.loc[summary["period"] == period]
    if summary_period.empty:
        return groups.edges.iloc[0:0].copy()
    return groups.edges.merge(summary_period[["period", "group_id"]], on=["period", "group_id"], how="inner")


def _build_chain_group_map(
    *,
    groups,
    base_year: int,
    target_year: int,
    weights_by_year: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, set[str], set[str]]:
    uf = _UnionFind()
    sample_codes: set[str] = set()

    for step in chain_steps(base_year, target_year):
        period = str(step["period"])
        direction = str(step["direction"])
        step_edges = _ambiguous_edges_for_step(groups=groups, period=period, direction=direction)
        if step_edges.empty:
            continue

        year_a = int(step_edges["vintage_a_year"].iloc[0])
        year_b = int(step_edges["vintage_b_year"].iloc[0])
        weights_a = weights_by_year.get(str(year_a)) if year_a != target_year else None
        weights_b = weights_by_year.get(str(year_b)) if year_b != target_year else None

        for group_id, edges in step_edges.groupby("group_id", sort=False):
            codes_a = set(edges["vintage_a_code"].tolist())
            codes_b = set(edges["vintage_b_code"].tolist())
            mapped = set()
            mapped |= _map_codes_to_target(codes_a, weights_a)
            mapped |= _map_codes_to_target(codes_b, weights_b)
            mapped = set(normalize_codes(pd.Series(list(mapped))).tolist())
            if not mapped:
                continue
            sample_codes |= mapped
            mapped_list = list(mapped)
            for code in mapped_list:
                uf.add(code)
            anchor = mapped_list[0]
            for code in mapped_list[1:]:
                uf.union(anchor, code)

    if not sample_codes:
        return (
            pd.DataFrame(columns=["target_code", "group_id"]),
            set(),
            set(),
        )

    components: dict[str, list[str]] = {}
    for code in sample_codes:
        root = uf.find(code)
        components.setdefault(root, []).append(code)

    component_rows: list[tuple[str, str]] = []
    for codes in components.values():
        codes_sorted = sorted(codes)
        component_rows.append((codes_sorted[0], codes_sorted))
    component_rows.sort(key=lambda item: item[0])

    group_map_rows: list[dict[str, str]] = []
    group_ids: set[str] = set()
    for idx, (_, codes) in enumerate(component_rows, start=1):
        group_id = f"{base_year}to{target_year}_g{idx:06d}"
        group_ids.add(group_id)
        for code in codes:
            group_map_rows.append({"target_code": code, "group_id": group_id})

    return pd.DataFrame(group_map_rows), group_ids, sample_codes


def _merge_shares(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.merge(right, on=["group_id", "product_code"], how="inner")


def run_stress_test_analysis(config: StressConfig) -> dict[str, object]:
    years = config.years
    measures = config.measures
    metrics_set = {name.lower() for name in config.metrics}
    want_exposure = "exposure_weighted" in metrics_set
    want_diffuseness = "diffuseness_weighted" in metrics_set

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

        shares_by_year = build_year_shares_from_totals(
            years=[base_year, compare_year],
            target_year=years.target,
            totals_by_year=totals_by_year,
            weights_by_year=weights_by_year,
            group_map=group_map,
            group_ids=group_ids,
        )

        left = shares_by_year[base_year].rename(columns={"share": "share_t"})
        right = shares_by_year[compare_year].rename(columns={"share": "share_t1"})
        merged = _merge_shares(left, right)
        panel_pairs.append(PanelPair(x_year=base_year, y_year=compare_year, data=merged))
        r2_val = (
            r2_45(merged["share_t"].to_numpy(), merged["share_t1"].to_numpy())
            if "r2_45" in metrics_set
            else float("nan")
        )
        r2_w = float("nan")
        r2_sym = float("nan")
        mae_w = float("nan")
        need_weighted = {
            "r2_45_weighted",
            "r2_45_weighted_symmetric",
            "mae_weighted",
        } & metrics_set
        if need_weighted:
            compare_values = build_values_for_groups_from_totals(
                year=compare_year,
                target_year=years.target,
                totals_by_year=totals_by_year,
                weights_by_year=weights_by_year,
                group_map=group_map,
                group_ids=group_ids,
            ).rename(columns={"value": "value_y"})
            weights_y = merged.merge(
                compare_values, on=["group_id", "product_code"], how="left"
            )
            weights_y["value_y"] = weights_y["value_y"].fillna(0.0)
            if "r2_45_weighted" in metrics_set:
                r2_w = r2_45_weighted(
                    weights_y["share_t"].to_numpy(),
                    weights_y["share_t1"].to_numpy(),
                    weights_y["value_y"].to_numpy(),
                )
            if "r2_45_weighted_symmetric" in metrics_set or "mae_weighted" in metrics_set:
                base_values = build_values_for_groups_from_totals(
                    year=base_year,
                    target_year=years.target,
                    totals_by_year=totals_by_year,
                    weights_by_year=weights_by_year,
                    group_map=group_map,
                    group_ids=group_ids,
                ).rename(columns={"value": "value_x"})
                weights_xy = weights_y.merge(
                    base_values, on=["group_id", "product_code"], how="left"
                )
                weights_xy["value_x"] = weights_xy["value_x"].fillna(0.0)
                if "r2_45_weighted_symmetric" in metrics_set:
                    r2_sym = r2_45_weighted_symmetric(
                        weights_xy["share_t"].to_numpy(),
                        weights_xy["share_t1"].to_numpy(),
                        weights_xy["value_x"].to_numpy(),
                        weights_xy["value_y"].to_numpy(),
                    )
                if "mae_weighted" in metrics_set:
                    mae_w = mae_weighted(
                        weights_xy["share_t"].to_numpy(),
                        weights_xy["share_t1"].to_numpy(),
                        weights_xy["value_x"].to_numpy(),
                        weights_xy["value_y"].to_numpy(),
                    )
        exposure_weighted = float("nan")
        diffuseness_weighted = float("nan")
        steps_df = pd.DataFrame(step_rows_chain)
        if not steps_df.empty:
            if "exposure_weighted" in metrics_set:
                exp_vals = steps_df["ambiguity_exposure"].to_numpy(dtype=float)
                exp_w = steps_df["total_trade_sample"].to_numpy(dtype=float)
                exposure_weighted = weighted_mean(exp_vals, exp_w)
            if "diffuseness_weighted" in metrics_set:
                diff_vals = steps_df["diffuseness"].to_numpy(dtype=float)
                diff_w = steps_df["ambiguous_trade"].to_numpy(dtype=float)
                diffuseness_weighted = weighted_mean(diff_vals, diff_w)
        lines = []
        if "r2_45" in metrics_set:
            lines.append(rf"$R^2$ = {r2_val:.3f}")
        if "r2_45_weighted" in metrics_set:
            lines.append(rf"$R^2_w$ = {r2_w:.3f}")
        if "r2_45_weighted_symmetric" in metrics_set:
            lines.append(rf"$R^2_{{sym}}$ = {r2_sym:.3f}")
        if "mae_weighted" in metrics_set:
            lines.append(rf"wMAE = {mae_w:.3f}")
        if "exposure_weighted" in metrics_set:
            lines.append(rf"$E_w$ = {exposure_weighted:.3f}")
        if "diffuseness_weighted" in metrics_set:
            lines.append(rf"$H_w$ = {diffuseness_weighted:.3f}")
        annotation_by_year[base_year] = "\n".join(lines)
        summary_rows.append(
            {
                "base_year": base_year,
                "compare_year": compare_year,
                "n_points": int(len(merged)),
                "r2_45": r2_val,
                "r2_45_weighted": r2_w,
                "r2_45_weighted_symmetric": r2_sym,
                "mae_weighted": mae_w,
                "ambiguity_exposure_weighted": (
                    exposure_weighted if "exposure_weighted" in metrics_set else float("nan")
                ),
                "diffuseness_weighted": (
                    diffuseness_weighted if "diffuseness_weighted" in metrics_set else float("nan")
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
