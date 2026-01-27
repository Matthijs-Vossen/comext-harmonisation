"""Figure-3 style analysis for within-group share stability."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Fig3Config
from ..estimation.chaining import build_chained_weights_for_range, build_code_universe_from_annual
from ..estimation.runner import load_concordance_groups
from ..mappings import get_ambiguous_group_summary


def _normalize_codes(series: pd.Series) -> pd.Series:
    codes = series.astype(str).str.strip().str.replace(" ", "", regex=False)
    mask = codes.str.isdigit()
    codes = codes.where(~mask, codes.str.zfill(8))
    return codes


def _filter_partners(df: pd.DataFrame, *, exclude_reporters: Sequence[str], exclude_partners: Sequence[str]) -> pd.DataFrame:
    if not exclude_reporters and not exclude_partners:
        return df
    mask = pd.Series(True, index=df.index)
    if exclude_reporters:
        mask &= ~df["REPORTER"].isin(exclude_reporters)
    if exclude_partners:
        mask &= ~df["PARTNER"].isin(exclude_partners)
    return df.loc[mask]


def _convert_totals_to_target(
    *,
    totals: pd.DataFrame,
    weights: pd.DataFrame | None,
    assume_identity_for_missing: bool = True,
) -> pd.DataFrame:
    totals = totals.copy()
    totals["PRODUCT_NC"] = _normalize_codes(totals["PRODUCT_NC"])
    if weights is None:
        return totals.rename(columns={"PRODUCT_NC": "target_code"})

    weights = weights[["from_code", "to_code", "weight"]].copy()
    weights["from_code"] = _normalize_codes(weights["from_code"])
    weights["to_code"] = _normalize_codes(weights["to_code"])

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


def _compute_group_shares(
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


def _unstable_codes_from_edges(edges: pd.DataFrame) -> tuple[set[str], set[str]]:
    edges = edges[["vintage_a_code", "vintage_b_code"]].drop_duplicates().copy()
    edges["vintage_a_code"] = _normalize_codes(edges["vintage_a_code"])
    edges["vintage_b_code"] = _normalize_codes(edges["vintage_b_code"])

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
    if not codes:
        return set()
    codes_series = pd.Series(list(codes))
    codes_series = _normalize_codes(codes_series)
    if weights is None:
        return set(codes_series.tolist())

    weights = weights[["from_code", "to_code"]].copy()
    weights["from_code"] = _normalize_codes(weights["from_code"])
    weights["to_code"] = _normalize_codes(weights["to_code"])
    mapped = weights[weights["from_code"].isin(codes_series)]["to_code"].unique().tolist()
    return set(mapped)


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


def _r2_45(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    y_bar = float(np.mean(y))
    sse = float(np.sum((y - x) ** 2))
    sst = float(np.sum((y - y_bar) ** 2))
    if sst == 0:
        return float("nan")
    return 1.0 - sse / sst


def _plot_panels(
    pairs: list[tuple[int, pd.DataFrame]],
    output_path: Path,
    point_alpha: float,
    point_size: float,
    axis_padding: float,
    point_color: str,
    use_latex: bool,
    latex_preamble: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    if use_latex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["text.latex.preamble"] = latex_preamble

    n_panels = len(pairs)
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6), squeeze=False)

    for idx, (year, df) in enumerate(pairs):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        x = df["share_t"].to_numpy()
        y = df["share_t1"].to_numpy()
        label_scatter = "trade shares" if idx == 0 else None
        label_line = "identity line ($y=x$)" if idx == 0 else None
        ax.scatter(
            x,
            y,
            s=point_size,
            alpha=point_alpha,
            color=point_color,
            edgecolors="none",
            label=label_scatter,
        )
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, label=label_line)
        pad = max(axis_padding, 0.0)
        ax.set_xlim(-pad, 1 + pad)
        ax.set_ylim(-pad, 1 + pad)
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel(f"{year} Trade Shares")
        ax.set_ylabel(f"{year + 1} Trade Shares")
        r2 = _r2_45(x, y)
        ax.text(0.05, 0.9, f"$R^2$ = {r2:.3f}", transform=ax.transAxes)

    for idx in range(n_panels, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.delaxes(axes[row][col])

    if pairs:
        from matplotlib.lines import Line2D

        handles, labels = axes[0][0].get_legend_handles_labels()
        legend_handles = []
        legend_labels = []
        for handle, label in zip(handles, labels):
            if label == "identity line ($y=x$)":
                legend_handles.append(
                    Line2D(
                        [],
                        [],
                        linestyle="None",
                        marker=(2, 0, 135),
                        markersize=10,
                        markeredgewidth=1.0,
                        color="black",
                    )
                )
                legend_labels.append(label)
            else:
                legend_handles.append(handle)
                legend_labels.append(label)
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=2,
                frameon=False,
                handletextpad=0.4,
                columnspacing=1.0,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_fig3_analysis(config: Fig3Config) -> dict[str, object]:
    """Run Figure-3 style analysis and return metadata."""
    years = config.years
    measures = config.measures

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

    year_shares: dict[int, pd.DataFrame] = {}
    for year in range(years.start, years.end + 1):
        data_path = config.paths.annual_base_dir / f"comext_{year}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing annual data file: {data_path}")
        data = pd.read_parquet(data_path, columns=["REPORTER", "PARTNER", "PRODUCT_NC", measures.analysis_measure])
        data = _filter_partners(
            data,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
        )
        totals = (
            data.groupby("PRODUCT_NC", as_index=False, sort=False)[measures.analysis_measure]
            .sum()
            .rename(columns={measures.analysis_measure: "value"})
        )
        weights = weights_by_year.get(str(year)) if year != years.target else None
        converted = _convert_totals_to_target(
            totals=totals,
            weights=weights,
            assume_identity_for_missing=True,
        )
        shares = _compute_group_shares(
            totals=converted,
            group_map=group_map,
            group_ids=group_ids,
        )
        shares = shares.rename(columns={"target_code": "product_code"})
        shares["year"] = year
        year_shares[year] = shares

    pairs: list[tuple[int, pd.DataFrame]] = []
    panel_stats: dict[int, dict[str, int]] = {}
    for year in range(years.start, years.end):
        left = year_shares[year].rename(columns={"share": "share_t"})
        right = year_shares[year + 1].rename(columns={"share": "share_t1"})
        merged_all = left.merge(
            right,
            on=["group_id", "product_code"],
            how="inner",
        )
        if group_ids_filtered != group_ids:
            merged = merged_all[merged_all["group_id"].isin(group_ids_filtered)]
            n_codes_before = merged_all["product_code"].nunique()
            n_codes_after = merged["product_code"].nunique()
            panel_stats[year] = {
                "n_codes_before": int(n_codes_before),
                "n_codes_after": int(n_codes_after),
                "n_codes_filtered": int(n_codes_before - n_codes_after),
            }
        else:
            merged = merged_all
            panel_stats[year] = {
                "n_codes_before": int(merged["product_code"].nunique()),
                "n_codes_after": int(merged["product_code"].nunique()),
                "n_codes_filtered": 0,
            }
        pairs.append((year, merged))

    _plot_panels(
        pairs=pairs,
        output_path=config.plot.output_path,
        point_alpha=config.plot.point_alpha,
        point_size=config.plot.point_size,
        axis_padding=config.plot.axis_padding,
        point_color=config.plot.point_color,
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
    )

    summary_path = config.paths.output_dir / "summary.csv"
    summary_rows = []
    for year, df in pairs:
        stats = panel_stats.get(year, {})
        summary_rows.append(
            {
                "year_t": year,
                "year_t1": year + 1,
                "n_points": len(df),
                "r2_45": _r2_45(df["share_t"].to_numpy(), df["share_t1"].to_numpy()),
                "n_codes_filtered": stats.get("n_codes_filtered", 0),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    return {
        "output_plot": str(config.plot.output_path),
        "summary_csv": str(summary_path),
        "config": asdict(config),
    }
