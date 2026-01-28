"""Stress-test share stability analysis across long chains to a target vintage."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..common.metrics import r2_45
from ..common.plotting import plot_share_panels
from ..common.shares import PanelPair, build_year_shares, normalize_codes
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


def _chain_steps(base_year: int, target_year: int) -> list[dict[str, int | str]]:
    steps: list[dict[str, int | str]] = []
    if base_year == target_year:
        return steps
    if base_year < target_year:
        for year in range(base_year, target_year):
            steps.append(
                {
                    "period": f"{year}{year + 1}",
                    "direction": "a_to_b",
                    "source_year": year,
                    "target_year": year + 1,
                }
            )
    else:
        for year in range(base_year - 1, target_year - 1, -1):
            steps.append(
                {
                    "period": f"{year}{year + 1}",
                    "direction": "b_to_a",
                    "source_year": year + 1,
                    "target_year": year,
                }
            )
    return steps


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


def _load_annual_totals(
    *,
    annual_base_dir: Path,
    year: int,
    measure: str,
    exclude_reporters: Iterable[str],
    exclude_partners: Iterable[str],
) -> pd.DataFrame:
    data_path = annual_base_dir / f"comext_{year}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing annual data file: {data_path}")
    cols = ["REPORTER", "PARTNER", "PRODUCT_NC", measure]
    data = pd.read_parquet(data_path, columns=cols)
    if exclude_reporters:
        data = data.loc[~data["REPORTER"].isin(exclude_reporters)]
    if exclude_partners:
        data = data.loc[~data["PARTNER"].isin(exclude_partners)]
    totals = (
        data.groupby("PRODUCT_NC", as_index=False, sort=False)[measure]
        .sum()
        .rename(columns={measure: "value"})
    )
    totals["PRODUCT_NC"] = normalize_codes(totals["PRODUCT_NC"])
    return totals


def _sample_source_codes(
    *,
    sample_target_codes: set[str],
    weights_to_target: pd.DataFrame | None,
) -> set[str]:
    if not sample_target_codes:
        return set()
    if weights_to_target is None:
        return set(sample_target_codes)
    weights = weights_to_target[["from_code", "to_code"]].copy()
    weights["from_code"] = normalize_codes(weights["from_code"])
    weights["to_code"] = normalize_codes(weights["to_code"])
    matched = weights[weights["to_code"].isin(sample_target_codes)]["from_code"].unique().tolist()
    return set(matched)


def _load_step_weights(
    *,
    period: str,
    direction: str,
    measure: str,
    weights_dir: Path,
) -> pd.DataFrame:
    measure_tag = measure.lower()
    weights_path = weights_dir / period / direction / measure_tag / "weights_ambiguous.csv"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    weights = pd.read_csv(weights_path)
    if weights.empty:
        return pd.DataFrame(columns=["from_code", "to_code", "weight"])
    weights = weights[["from_code", "to_code", "weight"]].copy()
    weights["from_code"] = normalize_codes(weights["from_code"])
    weights["to_code"] = normalize_codes(weights["to_code"])
    weights["weight"] = weights["weight"].astype(float)
    return weights


def _feasible_target_map(
    period_edges: pd.DataFrame,
    direction: str,
) -> dict[str, list[str]]:
    if period_edges.empty:
        return {}
    if direction == "a_to_b":
        grouped = period_edges.groupby("vintage_a_code", sort=False)["vintage_b_code"].unique()
    else:
        grouped = period_edges.groupby("vintage_b_code", sort=False)["vintage_a_code"].unique()
    mapping: dict[str, list[str]] = {}
    for code, targets in grouped.items():
        mapping[str(code)] = normalize_codes(pd.Series(list(targets))).tolist()
    return mapping


def _build_chain_group_map(
    *,
    groups,
    base_year: int,
    target_year: int,
    weights_by_year: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, set[str], set[str]]:
    uf = _UnionFind()
    sample_codes: set[str] = set()

    for step in _chain_steps(base_year, target_year):
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
    want_exposure = bool(metrics_set & {"exposure", "exposure_weighted"})
    want_diffuseness = bool(metrics_set & {"diffuseness", "diffuseness_weighted"})

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

    panel_pairs: list[PanelPair] = []
    annotation_by_year: dict[int, str] = {}
    summary_rows: list[dict[str, object]] = []
    step_rows: list[dict[str, object]] = []

    for chain in years.chains:
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
                    "ambiguity_exposure": float("nan"),
                    "diffuseness": float("nan"),
                }
            )
            continue

        chain_steps = _chain_steps(base_year, years.target)
        exposures: list[float] = []
        diffuseness: list[float] = []
        step_rows_chain: list[dict[str, object]] = []

        if want_exposure or want_diffuseness:
            for step_idx, step in enumerate(chain_steps, start=1):
                period = str(step["period"])
                direction = str(step["direction"])
                source_year = int(step["source_year"])
                weights_to_target = weights_by_year.get(str(source_year)) if source_year != years.target else None
                sample_source_codes = _sample_source_codes(
                    sample_target_codes=sample_codes,
                    weights_to_target=weights_to_target,
                )

                totals = _load_annual_totals(
                    annual_base_dir=config.paths.annual_base_dir,
                    year=source_year,
                    measure=measures.analysis_measure,
                    exclude_reporters=config.sample.exclude_reporters,
                    exclude_partners=config.sample.exclude_partners,
                )
                totals = totals.loc[totals["PRODUCT_NC"].isin(sample_source_codes)]
                total_trade = float(totals["value"].sum())

                step_weights = _load_step_weights(
                    period=period,
                    direction=direction,
                    measure=measures.weights_source,
                    weights_dir=config.paths.weights_dir,
                )
                period_edges = groups.edges.loc[groups.edges["period"] == period]
                feasible_map = _feasible_target_map(period_edges, direction)
                ambiguous_sources = {
                    code
                    for code, targets in feasible_map.items()
                    if len(targets) > 1
                }
                estimable_sources = set(step_weights["from_code"].unique().tolist())
                ambiguous_sources = (
                    ambiguous_sources
                    & estimable_sources
                    & set(totals["PRODUCT_NC"].unique().tolist())
                )
                totals = totals.loc[totals["PRODUCT_NC"].isin(estimable_sources)]

                ambiguous_trade = float(
                    totals.loc[totals["PRODUCT_NC"].isin(ambiguous_sources), "value"].sum()
                )
                exposure = float("nan")
                if want_exposure and total_trade > 0:
                    exposure = ambiguous_trade / total_trade
                    exposures.append(exposure)

                step_entropy = float("nan")
                if want_diffuseness and ambiguous_trade > 0 and ambiguous_sources:
                    entropy_rows = []
                    for code in ambiguous_sources:
                        weights_code = step_weights.loc[step_weights["from_code"] == code, "weight"]
                        weights_code = weights_code[weights_code > 0]
                        weights_sum = float(weights_code.sum())
                        if weights_sum <= 0:
                            raise ValueError(
                                f"Entropy computation failed for {period} {direction}: "
                                f"row {code} has no positive weights after clipping."
                            )
                        probs = (weights_code / weights_sum).to_numpy()
                        positive_probs = probs[probs > 0]
                        k_est = len(feasible_map.get(code, []))
                        if k_est <= 1:
                            h_norm = 0.0
                        else:
                            h_val = float(-(positive_probs * np.log(positive_probs)).sum())
                            h_norm = h_val / float(np.log(k_est))
                        trade_val = float(totals.loc[totals["PRODUCT_NC"] == code, "value"].sum())
                        entropy_rows.append((trade_val, h_norm))
                    if entropy_rows:
                        weights_trade = np.array([row[0] for row in entropy_rows], dtype=float)
                        values = np.array([row[1] for row in entropy_rows], dtype=float)
                        denom = float(weights_trade.sum())
                        if denom > 0:
                            step_entropy = float((weights_trade * values).sum() / denom)
                            diffuseness.append(step_entropy)

                step_rows_chain.append(
                    {
                        "base_year": base_year,
                        "compare_year": compare_year,
                        "target_year": years.target,
                        "step_index": step_idx,
                        "period": period,
                        "direction": direction,
                        "source_year": source_year,
                        "total_trade_sample": total_trade,
                        "ambiguous_trade": ambiguous_trade,
                        "ambiguity_exposure": exposure,
                        "diffuseness": step_entropy,
                        "n_ambiguous_sources": int(len(ambiguous_sources)),
                    }
                )

        shares_by_year = build_year_shares(
            years=[base_year, compare_year],
            target_year=years.target,
            annual_base_dir=config.paths.annual_base_dir,
            weights_by_year=weights_by_year,
            measure=measures.analysis_measure,
            group_map=group_map,
            group_ids=group_ids,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
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
        exposure_mean = float(np.nanmean(exposures)) if exposures else float("nan")
        diffuseness_mean = float(np.nanmean(diffuseness)) if diffuseness else float("nan")
        exposure_weighted = float("nan")
        diffuseness_weighted = float("nan")
        steps_df = pd.DataFrame(step_rows_chain)
        if not steps_df.empty:
            if "exposure_weighted" in metrics_set:
                exp_vals = steps_df["ambiguity_exposure"].to_numpy(dtype=float)
                exp_w = steps_df["total_trade_sample"].to_numpy(dtype=float)
                exp_mask = np.isfinite(exp_vals) & np.isfinite(exp_w) & (exp_w > 0)
                if exp_mask.any():
                    exposure_weighted = float(
                        (exp_w[exp_mask] * exp_vals[exp_mask]).sum() / exp_w[exp_mask].sum()
                    )
            if "diffuseness_weighted" in metrics_set:
                diff_vals = steps_df["diffuseness"].to_numpy(dtype=float)
                diff_w = steps_df["ambiguous_trade"].to_numpy(dtype=float)
                diff_mask = np.isfinite(diff_vals) & np.isfinite(diff_w) & (diff_w > 0)
                if diff_mask.any():
                    diffuseness_weighted = float(
                        (diff_w[diff_mask] * diff_vals[diff_mask]).sum() / diff_w[diff_mask].sum()
                    )
        lines = []
        if "r2_45" in metrics_set:
            lines.append(rf"$R^2$ = {r2_val:.3f}")
        if "exposure" in metrics_set:
            lines.append(rf"$E$ = {exposure_mean:.3f}")
        if "exposure_weighted" in metrics_set:
            lines.append(rf"$E_w$ = {exposure_weighted:.3f}")
        if "diffuseness" in metrics_set:
            lines.append(rf"$H$ = {diffuseness_mean:.3f}")
        if "diffuseness_weighted" in metrics_set:
            lines.append(rf"$H_w$ = {diffuseness_weighted:.3f}")
        annotation_by_year[base_year] = "\n".join(lines)
        summary_rows.append(
            {
                "base_year": base_year,
                "compare_year": compare_year,
                "n_points": int(len(merged)),
                "r2_45": r2_val,
                "ambiguity_exposure": exposure_mean if "exposure" in metrics_set else float("nan"),
                "diffuseness": diffuseness_mean if "diffuseness" in metrics_set else float("nan"),
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
