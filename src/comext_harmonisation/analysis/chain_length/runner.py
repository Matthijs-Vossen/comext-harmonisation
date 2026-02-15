"""Chain-length sensitivity analysis (R^2_sym, E_w, H_w vs chain length)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ChainLengthConfig
from ..common.metrics import mae_weighted, r2_45_weighted_symmetric, weighted_mean
from ..common.plotting import plot_chain_length_delta_panels, plot_chain_length_panels
from ..common.progress import progress
from ..common.shares import (
    build_values_for_groups_from_totals,
    build_year_shares_from_totals,
    normalize_codes,
)
from ..common.steps import (
    compute_step_metrics,
    chain_steps,
    load_annual_totals,
    load_step_weights,
    feasible_target_map,
)
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
            parent = self.find(parent)
            self._parent[item] = parent
        return parent

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self._size[root_a] < self._size[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._size[root_a] += self._size[root_b]


def _ambiguous_edges_for_step(groups, period: str, direction: str) -> pd.DataFrame:
    summary = get_ambiguous_group_summary(groups, direction)
    summary_period = summary.loc[summary["period"] == period]
    if summary_period.empty:
        return groups.edges.iloc[0:0].copy()
    return groups.edges.merge(
        summary_period[["period", "group_id"]], on=["period", "group_id"], how="inner"
    )


def _map_codes_to_target(codes: set[str], weights: pd.DataFrame | None) -> set[str]:
    if not codes:
        return set()
    codes_series = normalize_codes(pd.Series(list(codes)))
    if weights is None:
        return set(codes_series.tolist())
    weights = weights[["from_code", "to_code"]].copy()
    weights["from_code"] = normalize_codes(weights["from_code"])
    weights["to_code"] = normalize_codes(weights["to_code"])
    mapped = weights[weights["from_code"].isin(codes_series)]["to_code"].unique().tolist()
    return set(mapped)


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

        for _, edges in step_edges.groupby("group_id", sort=False):
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

    component_rows: list[tuple[str, list[str]]] = []
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


def _compute_chain_point(
    *,
    base_year: int,
    target_year: int,
    groups,
    weights_by_year: dict[str, pd.DataFrame],
    config: ChainLengthConfig,
    totals_by_year: dict[int, pd.DataFrame],
    step_weights_cache: dict[tuple[str, str, str], pd.DataFrame],
    feasible_map_cache: dict[tuple[str, str], dict[str, list[str]]],
    fixed_group_map: pd.DataFrame | None = None,
    fixed_group_ids: set[str] | None = None,
    fixed_sample_codes: set[str] | None = None,
) -> tuple[float, float, float, float, float, float, int, list[dict[str, object]]]:
    if fixed_group_map is not None and fixed_group_ids is not None and fixed_sample_codes is not None:
        group_map = fixed_group_map
        group_ids = fixed_group_ids
        sample_codes = fixed_sample_codes
    else:
        group_map, group_ids, sample_codes = _build_chain_group_map(
            groups=groups,
            base_year=base_year,
            target_year=target_year,
            weights_by_year=weights_by_year,
        )
    if not group_ids:
        raise ValueError(f"No sample groups for chain {base_year}->{target_year}")

    shares_by_year = build_year_shares_from_totals(
        years=[base_year, target_year],
        target_year=target_year,
        totals_by_year=totals_by_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    )
    left = shares_by_year[base_year].rename(columns={"share": "share_t"})
    right = shares_by_year[target_year].rename(columns={"share": "share_t1"})
    merged = left.merge(right, on=["group_id", "product_code"], how="inner")
    if merged.empty:
        raise ValueError(f"No merged shares for chain {base_year}->{target_year}")

    values_base = build_values_for_groups_from_totals(
        year=base_year,
        target_year=target_year,
        totals_by_year=totals_by_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    ).rename(columns={"value": "value_x"})
    values_target = build_values_for_groups_from_totals(
        year=target_year,
        target_year=target_year,
        totals_by_year=totals_by_year,
        weights_by_year=weights_by_year,
        group_map=group_map,
        group_ids=group_ids,
    ).rename(columns={"value": "value_y"})

    weights_df = merged.merge(values_base, on=["group_id", "product_code"], how="left")
    weights_df = weights_df.merge(values_target, on=["group_id", "product_code"], how="left")
    weights_df["value_x"] = weights_df["value_x"].fillna(0.0)
    weights_df["value_y"] = weights_df["value_y"].fillna(0.0)

    r2_sym = r2_45_weighted_symmetric(
        weights_df["share_t"].to_numpy(),
        weights_df["share_t1"].to_numpy(),
        weights_df["value_x"].to_numpy(),
        weights_df["value_y"].to_numpy(),
    )
    mae_w = mae_weighted(
        weights_df["share_t"].to_numpy(),
        weights_df["share_t1"].to_numpy(),
        weights_df["value_x"].to_numpy(),
        weights_df["value_y"].to_numpy(),
    )

    mae_step = float("nan")
    if abs(base_year - target_year) >= 2:
        adjacent_year = base_year - 1 if base_year > target_year else base_year + 1
        step_shares = build_year_shares_from_totals(
            years=[base_year, adjacent_year],
            target_year=target_year,
            totals_by_year=totals_by_year,
            weights_by_year=weights_by_year,
            group_map=group_map,
            group_ids=group_ids,
        )
        step_left = step_shares[base_year].rename(columns={"share": "share_t"})
        step_right = step_shares[adjacent_year].rename(columns={"share": "share_t1"})
        step_merged = step_left.merge(step_right, on=["group_id", "product_code"], how="inner")
        if not step_merged.empty:
            step_values_base = build_values_for_groups_from_totals(
                year=base_year,
                target_year=target_year,
                totals_by_year=totals_by_year,
                weights_by_year=weights_by_year,
                group_map=group_map,
                group_ids=group_ids,
            ).rename(columns={"value": "value_x"})
            step_values_adj = build_values_for_groups_from_totals(
                year=adjacent_year,
                target_year=target_year,
                totals_by_year=totals_by_year,
                weights_by_year=weights_by_year,
                group_map=group_map,
                group_ids=group_ids,
            ).rename(columns={"value": "value_y"})
            step_weights_df = step_merged.merge(
                step_values_base, on=["group_id", "product_code"], how="left"
            )
            step_weights_df = step_weights_df.merge(
                step_values_adj, on=["group_id", "product_code"], how="left"
            )
            step_weights_df["value_x"] = step_weights_df["value_x"].fillna(0.0)
            step_weights_df["value_y"] = step_weights_df["value_y"].fillna(0.0)
            mae_step = mae_weighted(
                step_weights_df["share_t"].to_numpy(),
                step_weights_df["share_t1"].to_numpy(),
                step_weights_df["value_x"].to_numpy(),
                step_weights_df["value_y"].to_numpy(),
            )

    step_rows = compute_step_metrics(
        base_year=base_year,
        target_year=target_year,
        sample_target_codes=sample_codes,
        weights_by_year=weights_by_year,
        groups=groups,
        annual_base_dir=config.paths.annual_base_dir,
        measure=config.measures.analysis_measure,
        weights_dir=config.paths.weights_dir,
        weights_source=config.measures.weights_source,
        exclude_reporters=config.sample.exclude_reporters,
        exclude_partners=config.sample.exclude_partners,
        compute_exposure=True,
        compute_diffuseness=True,
        totals_by_year=totals_by_year,
        step_weights_cache=step_weights_cache,
        feasible_map_cache=feasible_map_cache,
    )
    steps_df = pd.DataFrame(step_rows)
    if steps_df.empty:
        raise ValueError(f"No step metrics for chain {base_year}->{target_year}")
    first_step = steps_df.loc[steps_df["step_index"] == 1]
    if first_step.empty:
        raise ValueError(f"No step_index=1 metrics for chain {base_year}->{target_year}")
    exposure = float(first_step.iloc[0]["ambiguity_exposure"])
    diffuseness = float(first_step.iloc[0]["diffuseness"])
    diffuse_exposure = float(first_step.iloc[0]["diffuse_exposure"])
    return r2_sym, mae_w, mae_step, exposure, diffuseness, diffuse_exposure, len(merged), step_rows


def _chain_length_points(
    *, min_year: int, max_year: int, backward_anchor: int, forward_anchor: int
) -> list[dict[str, int | str]]:
    points: list[dict[str, int | str]] = []
    for base_year in range(backward_anchor + 1, max_year + 1):
        points.append(
            {
                "direction": "backward",
                "anchor_year": backward_anchor,
                "base_year": base_year,
                "chain_length": base_year - backward_anchor,
            }
        )
    for base_year in range(min_year, forward_anchor):
        points.append(
            {
                "direction": "forward",
                "anchor_year": forward_anchor,
                "base_year": base_year,
                "chain_length": forward_anchor - base_year,
            }
        )
    return points


def run_chain_length_analysis(config: ChainLengthConfig) -> dict[str, object]:
    years = config.years

    groups = load_concordance_groups(
        concordance_path=config.paths.concordance_path,
        sheet_name=config.paths.concordance_sheet,
    )

    min_year = years.min_year
    max_year = years.max_year

    code_universe = build_code_universe_from_annual(
        annual_base_dir=config.paths.annual_base_dir,
        years=range(min_year, max_year + 1),
    )

    outputs_backward = build_chained_weights_for_range(
        start_year=min_year,
        end_year=max_year,
        target_year=years.backward_anchor,
        measures=[config.measures.weights_source],
        code_universe=code_universe,
        weights_dir=config.paths.weights_dir,
        output_weights_dir=config.paths.output_dir
        / "chain"
        / f"CN{years.backward_anchor}",
        output_diagnostics_dir=config.paths.output_dir
        / "chain"
        / f"CN{years.backward_anchor}",
        finalize_weights=config.chaining.finalize_weights,
        neg_tol=config.chaining.neg_tol,
        pos_tol=config.chaining.pos_tol,
        row_sum_tol=config.chaining.row_sum_tol,
        fail_on_missing=True,
    )
    weights_backward: dict[str, pd.DataFrame] = {
        output.origin_year: output.weights for output in outputs_backward
    }

    outputs_forward = build_chained_weights_for_range(
        start_year=min_year,
        end_year=max_year,
        target_year=years.forward_anchor,
        measures=[config.measures.weights_source],
        code_universe=code_universe,
        weights_dir=config.paths.weights_dir,
        output_weights_dir=config.paths.output_dir
        / "chain"
        / f"CN{years.forward_anchor}",
        output_diagnostics_dir=config.paths.output_dir
        / "chain"
        / f"CN{years.forward_anchor}",
        finalize_weights=config.chaining.finalize_weights,
        neg_tol=config.chaining.neg_tol,
        pos_tol=config.chaining.pos_tol,
        row_sum_tol=config.chaining.row_sum_tol,
        fail_on_missing=True,
    )
    weights_forward: dict[str, pd.DataFrame] = {
        output.origin_year: output.weights for output in outputs_forward
    }

    totals_by_year: dict[int, pd.DataFrame] = {}
    for year in range(min_year, max_year + 1):
        totals_by_year[year] = load_annual_totals(
            annual_base_dir=config.paths.annual_base_dir,
            year=year,
            measure=config.measures.analysis_measure,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
        )
    step_weights_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    feasible_map_cache: dict[tuple[str, str], dict[str, list[str]]] = {}
    required_steps: set[tuple[str, str]] = set()
    for year in range(years.backward_anchor + 1, max_year + 1):
        required_steps.add((f"{year - 1}{year}", "b_to_a"))
    for year in range(min_year, years.forward_anchor):
        required_steps.add((f"{year}{year + 1}", "a_to_b"))
    for period, direction in sorted(required_steps):
        step_weights_cache[(period, direction, config.measures.weights_source)] = load_step_weights(
            period=period,
            direction=direction,
            measure=config.measures.weights_source,
            weights_dir=config.paths.weights_dir,
        )
        period_edges = groups.edges.loc[groups.edges["period"] == period]
        feasible_map_cache[(period, direction)] = feasible_target_map(period_edges, direction)

    rows: list[dict[str, object]] = []
    step_rows_all: list[dict[str, object]] = []

    fixed_backward: tuple[pd.DataFrame, set[str], set[str]] | None = None
    fixed_forward: tuple[pd.DataFrame, set[str], set[str]] | None = None
    if config.sample.sample_mode == "max_chain":
        fixed_backward = _build_chain_group_map(
            groups=groups,
            base_year=max_year,
            target_year=years.backward_anchor,
            weights_by_year=weights_backward,
        )
        fixed_forward = _build_chain_group_map(
            groups=groups,
            base_year=min_year,
            target_year=years.forward_anchor,
            weights_by_year=weights_forward,
        )

    # backward: base_year > anchor
    for base_year in progress(
        range(years.backward_anchor + 1, max_year + 1),
        desc="chain_length backward",
        total=max(0, max_year - years.backward_anchor),
    ):
        r2_sym, mae_w, mae_step, exposure, diffuseness, diffuse_exposure, n_points, step_rows = _compute_chain_point(
            base_year=base_year,
            target_year=years.backward_anchor,
            groups=groups,
            weights_by_year=weights_backward,
            config=config,
            totals_by_year=totals_by_year,
            step_weights_cache=step_weights_cache,
            feasible_map_cache=feasible_map_cache,
            fixed_group_map=fixed_backward[0] if fixed_backward else None,
            fixed_group_ids=fixed_backward[1] if fixed_backward else None,
            fixed_sample_codes=fixed_backward[2] if fixed_backward else None,
        )
        length = base_year - years.backward_anchor
        rows.append(
            {
                "direction": "backward",
                "anchor_year": years.backward_anchor,
                "base_year": base_year,
                "chain_length": length,
                "one_minus_r2_sym": 1.0 - r2_sym,
                "r2_sym": r2_sym,
                "mae_weighted": mae_w,
                "mae_weighted_step": mae_step,
                "exposure_weighted": exposure,
                "diffuseness_weighted": diffuseness,
                "diffuse_exposure": diffuse_exposure,
                "n_points": int(n_points),
            }
        )
        for row in step_rows:
            row["direction"] = "backward"
            row["anchor_year"] = years.backward_anchor
            row["chain_length"] = length
        step_rows_all.extend(step_rows)

    # forward: base_year < anchor
    for base_year in progress(
        range(min_year, years.forward_anchor),
        desc="chain_length forward",
        total=max(0, years.forward_anchor - min_year),
    ):
        r2_sym, mae_w, mae_step, exposure, diffuseness, diffuse_exposure, n_points, step_rows = _compute_chain_point(
            base_year=base_year,
            target_year=years.forward_anchor,
            groups=groups,
            weights_by_year=weights_forward,
            config=config,
            totals_by_year=totals_by_year,
            step_weights_cache=step_weights_cache,
            feasible_map_cache=feasible_map_cache,
            fixed_group_map=fixed_forward[0] if fixed_forward else None,
            fixed_group_ids=fixed_forward[1] if fixed_forward else None,
            fixed_sample_codes=fixed_forward[2] if fixed_forward else None,
        )
        length = years.forward_anchor - base_year
        rows.append(
            {
                "direction": "forward",
                "anchor_year": years.forward_anchor,
                "base_year": base_year,
                "chain_length": length,
                "one_minus_r2_sym": 1.0 - r2_sym,
                "r2_sym": r2_sym,
                "mae_weighted": mae_w,
                "mae_weighted_step": mae_step,
                "exposure_weighted": exposure,
                "diffuseness_weighted": diffuseness,
                "diffuse_exposure": diffuse_exposure,
                "n_points": int(n_points),
            }
        )
        for row in step_rows:
            row["direction"] = "forward"
            row["anchor_year"] = years.forward_anchor
            row["chain_length"] = length
        step_rows_all.extend(step_rows)

    summary = pd.DataFrame(rows).sort_values(["direction", "chain_length"])
    summary["delta_one_minus_r2_sym"] = summary.groupby("direction")[
        "one_minus_r2_sym"
    ].diff()
    summary["delta_mae_weighted"] = summary.groupby("direction")[
        "mae_weighted"
    ].diff()
    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    step_metrics_path = output_dir / "step_metrics.csv"
    if step_rows_all:
        pd.DataFrame(step_rows_all).to_csv(step_metrics_path, index=False)

    plot_path = config.plot.output_path
    plot_chain_length_panels(
        data=summary,
        output_path=plot_path,
        title=config.plot.title,
        point_color=config.plot.point_color,
        point_size=config.plot.point_size,
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
        metrics=config.metrics,
    )
    delta_plot_path = plot_path.with_name(f"{plot_path.stem}_delta.png")
    spearman_by_direction: dict[str, float] = {}
    for direction in ["backward", "forward"]:
        subset = summary.loc[
            summary["direction"] == direction, ["mae_weighted_step", "diffuse_exposure"]
        ].dropna()
        subset = subset[subset["diffuse_exposure"] > 0]
        if len(subset) < 2:
            spearman_by_direction[direction] = float("nan")
        else:
            spearman_by_direction[direction] = float(
                subset["mae_weighted_step"].corr(subset["diffuse_exposure"], method="spearman")
            )

    plot_chain_length_delta_panels(
        data=summary,
        output_path=delta_plot_path,
        title=config.plot.title,
        point_color=config.plot.point_color,
        point_size=max(1.0, config.plot.point_size - 1.0),
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
        spearman_by_direction=spearman_by_direction,
    )

    return {
        "summary_csv": summary_path,
        "step_metrics_csv": step_metrics_path if step_rows_all else "",
        "output_plot": plot_path,
        "output_plot_delta": delta_plot_path,
        "spearman_by_direction": spearman_by_direction,
        "config": config,
    }
