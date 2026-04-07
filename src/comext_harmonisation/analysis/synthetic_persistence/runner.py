"""Qualitative synthetic-persistence analysis for LT-converted CN8 series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ...chaining.engine import build_chained_weights_for_range, build_code_universe_from_annual
from ...concordance.io import read_concordance_xls
from ...core.codes import normalize_codes
from ..config import SyntheticPersistenceConfig


PREHISTORY_SET = "prehistory"
AFTERLIFE_SET = "afterlife"


@dataclass(frozen=True)
class CandidateTiming:
    obs_first_year: int | None
    obs_last_year: int | None
    concordance_intro_year: int | None
    concordance_sunset_year: int | None


def _normalize_code_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    out[column] = normalize_codes(out[column])
    return out


def _load_import_totals(
    *,
    annual_base_dir: Path,
    year: int,
    measure: str,
    flow_code: str,
    exclude_reporters: Iterable[str],
    exclude_partners: Iterable[str],
) -> pd.DataFrame:
    data_path = annual_base_dir / f"comext_{year}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing annual data file: {data_path}")
    cols = ["FLOW", "REPORTER", "PARTNER", "PRODUCT_NC", measure]
    data = pd.read_parquet(data_path, columns=cols)
    data = data.loc[data["FLOW"].astype(str) == str(flow_code)]
    if exclude_reporters:
        data = data.loc[~data["REPORTER"].isin(exclude_reporters)]
    if exclude_partners:
        data = data.loc[~data["PARTNER"].isin(exclude_partners)]

    totals = (
        data.groupby("PRODUCT_NC", as_index=False, sort=False)[measure]
        .sum()
        .rename(columns={"PRODUCT_NC": "code", measure: "value"})
    )
    totals = _normalize_code_frame(totals, "code")
    totals["value"] = totals["value"].astype(float)
    return totals


def _build_totals_cache(config: SyntheticPersistenceConfig) -> tuple[
    dict[int, pd.DataFrame], dict[int, float], dict[str, list[int]]
]:
    totals_by_year: dict[int, pd.DataFrame] = {}
    total_trade_by_year: dict[int, float] = {}
    observed_years_by_code: dict[str, list[int]] = {}

    for year in range(config.years.start, config.years.end + 1):
        totals = _load_import_totals(
            annual_base_dir=config.paths.annual_base_dir,
            year=year,
            measure=config.measures.analysis_measure,
            flow_code=config.flow.flow_code,
            exclude_reporters=config.sample.exclude_reporters,
            exclude_partners=config.sample.exclude_partners,
        )
        totals_by_year[year] = totals
        total_trade_by_year[year] = float(totals["value"].sum())
        for code in totals["code"].unique().tolist():
            observed_years_by_code.setdefault(str(code), []).append(year)

    return totals_by_year, total_trade_by_year, observed_years_by_code


def _concordance_timing_by_code(
    *,
    concordance_path: Path,
    concordance_sheet: str | int | None,
) -> tuple[dict[str, int], dict[str, int]]:
    edges = read_concordance_xls(str(concordance_path), sheet_name=concordance_sheet)
    edges = _normalize_code_frame(edges, "vintage_a_code")
    edges = _normalize_code_frame(edges, "vintage_b_code")

    intro_series = (
        edges.groupby("vintage_b_code", sort=False)["vintage_b_year"].min().astype(int)
    )
    sunset_series = (
        edges.groupby("vintage_a_code", sort=False)["vintage_a_year"].max().astype(int)
    )
    intro_by_code = {str(code): int(year) for code, year in intro_series.items()}
    sunset_by_code = {str(code): int(year) for code, year in sunset_series.items()}
    return intro_by_code, sunset_by_code


def _build_chain_weights(
    *,
    config: SyntheticPersistenceConfig,
    target_year: int,
    code_universe: dict[int, set[str]],
) -> dict[int, pd.DataFrame]:
    outputs = build_chained_weights_for_range(
        start_year=config.years.start,
        end_year=config.years.end,
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
        fail_on_missing=config.chaining.fail_on_missing,
        strict_revised_link_validation=config.chaining.strict_revised_link_validation,
        write_unresolved_details=config.chaining.write_unresolved_details,
    )

    weights_by_year: dict[int, pd.DataFrame] = {}
    for output in outputs:
        year = int(output.origin_year)
        weights = output.weights[["from_code", "to_code", "weight"]].copy()
        weights = _normalize_code_frame(weights, "from_code")
        weights = _normalize_code_frame(weights, "to_code")
        weights["weight"] = weights["weight"].astype(float)
        weights_by_year[year] = weights
    return weights_by_year


def _converted_value_for_code(
    *,
    year: int,
    code: str,
    anchor_year: int,
    totals_by_year: dict[int, pd.DataFrame],
    weights_by_year: dict[int, pd.DataFrame],
) -> float:
    totals = totals_by_year[year]
    if year == anchor_year:
        return float(totals.loc[totals["code"] == code, "value"].sum())

    if year not in weights_by_year:
        raise KeyError(f"Missing chained weights for year {year} -> CN{anchor_year}")
    weights = weights_by_year[year]
    mapped = weights.loc[weights["to_code"] == code]
    if mapped.empty:
        return 0.0

    merged = mapped.merge(totals, left_on="from_code", right_on="code", how="left")
    merged["value"] = merged["value"].fillna(0.0)
    return float((merged["value"] * merged["weight"]).sum())


def _build_candidate_timing(
    *,
    all_codes: Iterable[str],
    observed_years_by_code: dict[str, list[int]],
    intro_by_code: dict[str, int],
    sunset_by_code: dict[str, int],
) -> dict[str, CandidateTiming]:
    timing: dict[str, CandidateTiming] = {}
    for code in sorted(set(str(c) for c in all_codes)):
        years = observed_years_by_code.get(code, [])
        timing[code] = CandidateTiming(
            obs_first_year=min(years) if years else None,
            obs_last_year=max(years) if years else None,
            concordance_intro_year=intro_by_code.get(code) if code in intro_by_code else None,
            concordance_sunset_year=int(sunset_by_code.get(code)) if code in sunset_by_code else None,
        )
    return timing


def _afterlife_concept_check(timing: CandidateTiming, *, start_year: int) -> bool:
    return (
        timing.concordance_sunset_year is None
        and timing.concordance_intro_year is not None
        and timing.concordance_intro_year >= start_year
    )


def _classify_candidate_status(
    *,
    dimension: str,
    timing: CandidateTiming,
    start_year: int,
    end_year: int,
) -> tuple[bool, str | None]:
    if timing.obs_first_year is None:
        if timing.concordance_intro_year is not None and timing.concordance_intro_year > end_year:
            return False, "introduced_outside_window"
        return False, "absent_in_window_data"

    if dimension == AFTERLIFE_SET and _afterlife_concept_check(timing, start_year=start_year):
        return False, "concept_not_afterlife"

    return True, None


def _candidate_rows(config: SyntheticPersistenceConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for idx, code in enumerate(config.candidates.prehistory, start=1):
        rows.append(
            {
                "dimension": PREHISTORY_SET,
                "set_name": PREHISTORY_SET,
                "code": str(code),
                "label": config.candidates.display_labels.get(str(code), str(code)),
                "display_order": idx,
            }
        )
    for idx, code in enumerate(config.candidates.afterlife, start=1):
        rows.append(
            {
                "dimension": AFTERLIFE_SET,
                "set_name": AFTERLIFE_SET,
                "code": str(code),
                "label": config.candidates.display_labels.get(str(code), str(code)),
                "display_order": idx,
            }
        )
    return rows


def _build_code_catalog(
    *,
    config: SyntheticPersistenceConfig,
    timing_by_code: dict[str, CandidateTiming],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in _candidate_rows(config):
        code = str(item["code"])
        timing = timing_by_code.get(
            code,
            CandidateTiming(
                obs_first_year=None,
                obs_last_year=None,
                concordance_intro_year=None,
                concordance_sunset_year=None,
            ),
        )
        included_in_series, exclusion_reason = _classify_candidate_status(
            dimension=str(item["dimension"]),
            timing=timing,
            start_year=config.years.start,
            end_year=config.years.end,
        )
        rows.append(
            {
                **item,
                "obs_first_year": timing.obs_first_year,
                "obs_last_year": timing.obs_last_year,
                "concordance_intro_year": timing.concordance_intro_year,
                "concordance_sunset_year": timing.concordance_sunset_year,
                "included_in_series": bool(included_in_series),
                "exclusion_reason": exclusion_reason,
            }
        )

    return pd.DataFrame(rows).sort_values(["set_name", "display_order"]).reset_index(drop=True)


def _prehistory_intro_year(timing: CandidateTiming) -> int | None:
    if timing.obs_first_year is not None:
        return timing.obs_first_year
    return timing.concordance_intro_year


def _build_candidate_series(
    *,
    code_catalog: pd.DataFrame,
    timing_by_code: dict[str, CandidateTiming],
    years: Iterable[int],
    total_trade_by_year: dict[int, float],
    totals_by_year: dict[int, pd.DataFrame],
    prehistory_anchor: int,
    afterlife_anchor: int,
    prehistory_weights: dict[int, pd.DataFrame],
    afterlife_weights: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    included = code_catalog.loc[code_catalog["included_in_series"]].copy()

    for item in included.itertuples(index=False):
        code = str(item.code)
        timing = timing_by_code[code]
        intro_year = _prehistory_intro_year(timing)
        last_year = timing.obs_last_year
        dimension = str(item.dimension)

        if dimension == PREHISTORY_SET:
            anchor = prehistory_anchor
            weights = prehistory_weights
        else:
            anchor = afterlife_anchor
            weights = afterlife_weights

        for year in years:
            value_conv = _converted_value_for_code(
                year=year,
                code=code,
                anchor_year=anchor,
                totals_by_year=totals_by_year,
                weights_by_year=weights,
            )
            total_trade = float(total_trade_by_year[year])
            share_conv = value_conv / total_trade if total_trade > 0 else np.nan

            if dimension == PREHISTORY_SET:
                is_synthetic_window = intro_year is not None and year < intro_year
                is_inlife_window = intro_year is not None and year >= intro_year
            else:
                is_synthetic_window = last_year is not None and year > last_year
                is_inlife_window = last_year is not None and year <= last_year

            rows.append(
                {
                    "dimension": dimension,
                    "set_name": str(item.set_name),
                    "code": code,
                    "label": str(item.label),
                    "display_order": int(item.display_order),
                    "year": int(year),
                    "value_conv": float(value_conv),
                    "share_conv": float(share_conv),
                    "total_trade": total_trade,
                    "is_synthetic_window": bool(is_synthetic_window),
                    "is_inlife_window": bool(is_inlife_window),
                    "intro_year": intro_year,
                    "last_observed_year": last_year,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["set_name", "display_order", "year"])
        .reset_index(drop=True)
    )


def _peak_point(frame: pd.DataFrame) -> tuple[float, int | None]:
    valid = frame.dropna(subset=["share_conv"])
    if valid.empty:
        return float("nan"), None
    idx = int(valid["share_conv"].idxmax())
    row = valid.loc[idx]
    return float(row["share_conv"]), int(row["year"])


def _compute_code_evidence(candidate_series: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (dimension, set_name, code), frame in candidate_series.groupby(
        ["dimension", "set_name", "code"], sort=False
    ):
        frame = frame.sort_values("year")
        synthetic = frame.loc[frame["is_synthetic_window"]].sort_values("year")
        inlife = frame.loc[frame["is_inlife_window"]].sort_values("year")

        peak_share, peak_year = _peak_point(frame)
        synthetic_peak_share, synthetic_peak_year = _peak_point(synthetic)
        inlife_peak_share, inlife_peak_year = _peak_point(inlife)

        cumulative_synthetic = float(synthetic["share_conv"].sum())
        cumulative_inlife = float(inlife["share_conv"].sum())
        cumulative_ratio = (
            cumulative_synthetic / cumulative_inlife if cumulative_inlife > 0 else float("nan")
        )

        synthetic_first_share = (
            float(synthetic.iloc[0]["share_conv"]) if not synthetic.empty else float("nan")
        )
        synthetic_last_share = (
            float(synthetic.iloc[-1]["share_conv"]) if not synthetic.empty else float("nan")
        )

        rows.append(
            {
                "dimension": dimension,
                "set_name": set_name,
                "code": code,
                "display_order": int(frame["display_order"].iloc[0])
                if "display_order" in frame.columns
                else 0,
                "synthetic_years": int(len(synthetic)),
                "inlife_years": int(len(inlife)),
                "peak_share": peak_share,
                "peak_year": peak_year,
                "synthetic_peak_share": synthetic_peak_share,
                "synthetic_peak_year": synthetic_peak_year,
                "synthetic_median_share": float(synthetic["share_conv"].median())
                if not synthetic.empty
                else float("nan"),
                "synthetic_cumulative_share": cumulative_synthetic,
                "inlife_peak_share": inlife_peak_share,
                "inlife_peak_year": inlife_peak_year,
                "inlife_median_share": float(inlife["share_conv"].median())
                if not inlife.empty
                else float("nan"),
                "inlife_cumulative_share": cumulative_inlife,
                "cumulative_ratio_synth_to_inlife": float(cumulative_ratio),
                "share_start_year": float(frame.iloc[0]["share_conv"]),
                "share_end_year": float(frame.iloc[-1]["share_conv"]),
                "synthetic_first_year_share": synthetic_first_share,
                "synthetic_last_year_share": synthetic_last_share,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["set_name", "display_order"])
        .reset_index(drop=True)
    )


def _plot_style(use_latex: bool, latex_preamble: str) -> None:
    from matplotlib import rcParams

    if use_latex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["text.latex.preamble"] = latex_preamble


def _format_share_axis(ax, *, y_axis_unit: str) -> None:
    y_min, y_max = ax.get_ylim()
    if y_min > 0.0:
        ax.set_ylim(bottom=-y_min, top=y_max)

    if y_axis_unit == "percent":
        from matplotlib.ticker import PercentFormatter

        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=3))


def _apply_panel_axis_override(ax, *, code: str) -> None:
    if str(code).strip() == "88062210":
        ax.set_ylim(-5e-06, 3.1e-04)
        ax.set_yticks([0.0, 1e-04, 2e-04, 3e-04])


def _split_regime_segments_for_plot(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return observed and synthetic line segments with boundary bridge points."""
    base = (
        frame[["year", "share_conv", "is_synthetic_window"]]
        .dropna(subset=["year", "share_conv"])
        .sort_values("year")
        .copy()
    )
    observed = base.loc[~base["is_synthetic_window"], ["year", "share_conv"]].copy()
    synthetic = base.loc[base["is_synthetic_window"], ["year", "share_conv"]].copy()
    if observed.empty or synthetic.empty:
        return observed, synthetic

    obs_min_year = int(observed["year"].min())
    obs_max_year = int(observed["year"].max())
    synth_min_year = int(synthetic["year"].min())
    synth_max_year = int(synthetic["year"].max())

    if synth_max_year < obs_min_year:
        # Pre-history pattern: synthetic years precede observed years.
        observed = pd.concat(
            [synthetic.tail(1), observed],
            ignore_index=True,
        )
        synthetic = pd.concat(
            [synthetic, observed.iloc[[1]]],
            ignore_index=True,
        )
    elif obs_max_year < synth_min_year:
        # Afterlife pattern: observed years precede synthetic years.
        observed = pd.concat(
            [observed, synthetic.head(1)],
            ignore_index=True,
        )
        synthetic = pd.concat(
            [observed.iloc[[-2]], synthetic],
            ignore_index=True,
        )

    return observed.sort_values("year"), synthetic.sort_values("year")


def _plot_small_multiples_section(
    *,
    candidate_series: pd.DataFrame,
    dimension_set_name: str,
    output_axes: list[object],
    line_width: float,
    font_scale: float,
    y_axis_unit: str,
) -> None:
    series = candidate_series.loc[candidate_series["set_name"] == dimension_set_name].copy()
    sort_columns = ["year"]
    if "display_order" in series.columns:
        sort_columns = ["display_order", "year"]
    series = series.sort_values(sort_columns)
    codes = list(series["code"].drop_duplicates())
    years = series["year"]
    x_min = int(years.min()) if not years.empty else None
    x_max = int(years.max()) if not years.empty else None

    for idx, ax in enumerate(output_axes):
        if idx >= len(codes):
            ax.set_axis_off()
            continue

        code = codes[idx]
        frame = series.loc[series["code"] == code].sort_values("year")
        label = str(frame["label"].iloc[0]) if not frame.empty else code
        observed, synthetic = _split_regime_segments_for_plot(frame)
        color = "black"
        if not observed.empty:
            ax.plot(
                observed["year"],
                observed["share_conv"],
                linewidth=line_width,
                color=color,
                linestyle="-",
            )
        if not synthetic.empty:
            ax.plot(
                synthetic["year"],
                synthetic["share_conv"],
                linewidth=line_width,
                color=color,
                linestyle="--",
            )

        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
        _apply_panel_axis_override(ax, code=code)
        _format_share_axis(ax, y_axis_unit=y_axis_unit)
        ax.set_title(_format_panel_title(label, code), fontsize=8.5 * font_scale, pad=4)
        ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.25)
        ax.tick_params(labelsize=7.5 * font_scale)


def _format_panel_title(label: str, code: str) -> str:
    short_label = str(label).strip()
    short_code = str(code).strip()
    if not short_label or short_label == short_code:
        return short_code
    return f"{short_label}\n({short_code})"


def _format_section_title(title: str, *, use_latex: bool) -> str:
    if use_latex:
        return rf"\textbf{{{title}}}"
    return title


def _panel_columns(code_count: int, *, max_columns: int) -> int:
    return max(1, min(int(code_count), max_columns))


def _plot_summary(
    *,
    candidate_series: pd.DataFrame,
    output_path: Path,
    use_latex: bool,
    latex_preamble: str,
    line_width: float,
    font_scale: float,
    section_title_scale: float,
    y_axis_unit: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _plot_style(use_latex=use_latex, latex_preamble=latex_preamble)

    pre_codes = candidate_series.loc[
        candidate_series["set_name"] == PREHISTORY_SET, "code"
    ].drop_duplicates()
    after_codes = candidate_series.loc[
        candidate_series["set_name"] == AFTERLIFE_SET, "code"
    ].drop_duplicates()
    ncols_pre = _panel_columns(len(pre_codes), max_columns=3)
    ncols_after = _panel_columns(len(after_codes), max_columns=4)
    nrows_pre = max(1, int(np.ceil(len(pre_codes) / ncols_pre)))
    nrows_after = max(1, int(np.ceil(len(after_codes) / ncols_after)))
    fig_height = max(6.5, (1.75 * font_scale) * (nrows_pre + nrows_after) + 1.2)

    fig = plt.figure(figsize=(10.5, fig_height), layout="constrained")
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[nrows_pre, 0.08, nrows_after, 0.035],
        hspace=0.04,
    )

    pre_fig = fig.add_subfigure(outer[0])
    pre_fig.suptitle(
        _format_section_title(
            "Forward conversion trajectories (CN2023 anchor)",
            use_latex=use_latex,
        ),
        y=1.08,
        fontsize=11 * font_scale * section_title_scale,
    )
    pre_fig.supylabel("Share of total annual trade", fontsize=9 * font_scale)
    pre_axes_grid = pre_fig.subplots(nrows_pre, ncols_pre, squeeze=False, sharex=True)
    pre_axes = [ax for row in pre_axes_grid for ax in row]

    _plot_small_multiples_section(
        candidate_series=candidate_series,
        dimension_set_name=PREHISTORY_SET,
        output_axes=pre_axes,
        line_width=line_width,
        font_scale=font_scale,
        y_axis_unit=y_axis_unit,
    )

    title_ax = fig.add_subplot(outer[1])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.5,
        _format_section_title(
            "Backward conversion trajectories (CN1988 anchor)",
            use_latex=use_latex,
        ),
        ha="center",
        va="center",
        fontsize=11 * font_scale * section_title_scale,
    )

    after_fig = fig.add_subfigure(outer[2])
    after_fig.supxlabel("Year", fontsize=9 * font_scale)
    after_fig.supylabel("Share of total annual trade", fontsize=9 * font_scale)
    after_axes_grid = after_fig.subplots(nrows_after, ncols_after, squeeze=False, sharex=True)
    after_axes = [ax for row in after_axes_grid for ax in row]

    _plot_small_multiples_section(
        candidate_series=candidate_series,
        dimension_set_name=AFTERLIFE_SET,
        output_axes=after_axes,
        line_width=line_width,
        font_scale=font_scale,
        y_axis_unit=y_axis_unit,
    )

    legend_ax = fig.add_subplot(outer[3])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color="black", linewidth=line_width, linestyle="-"),
            Line2D([0], [0], color="black", linewidth=line_width, linestyle="--"),
        ],
        labels=["Observed share", "Converted share"],
        loc="center",
        ncol=2,
        frameon=False,
        fontsize=8.5 * font_scale,
        handlelength=2.8,
        columnspacing=1.8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def run_synthetic_persistence_analysis(config: SyntheticPersistenceConfig) -> dict[str, object]:
    years = list(range(config.years.start, config.years.end + 1))

    totals_by_year, total_trade_by_year, observed_years_by_code = _build_totals_cache(config)

    all_codes = list(config.candidates.prehistory) + list(config.candidates.afterlife)
    intro_by_code, sunset_by_code = _concordance_timing_by_code(
        concordance_path=config.paths.concordance_path,
        concordance_sheet=config.paths.concordance_sheet,
    )
    timing_by_code = _build_candidate_timing(
        all_codes=all_codes,
        observed_years_by_code=observed_years_by_code,
        intro_by_code=intro_by_code,
        sunset_by_code=sunset_by_code,
    )

    code_catalog = _build_code_catalog(config=config, timing_by_code=timing_by_code)

    code_universe = build_code_universe_from_annual(
        annual_base_dir=config.paths.annual_base_dir,
        years=years,
    )
    prehistory_weights = _build_chain_weights(
        config=config,
        target_year=config.years.prehistory_anchor,
        code_universe=code_universe,
    )
    afterlife_weights = _build_chain_weights(
        config=config,
        target_year=config.years.afterlife_anchor,
        code_universe=code_universe,
    )

    candidate_series = _build_candidate_series(
        code_catalog=code_catalog,
        timing_by_code=timing_by_code,
        years=years,
        total_trade_by_year=total_trade_by_year,
        totals_by_year=totals_by_year,
        prehistory_anchor=config.years.prehistory_anchor,
        afterlife_anchor=config.years.afterlife_anchor,
        prehistory_weights=prehistory_weights,
        afterlife_weights=afterlife_weights,
    )

    code_evidence = _compute_code_evidence(candidate_series)

    output_dir = config.paths.output_dir
    code_catalog_path = _write_csv(code_catalog, output_dir / "code_catalog.csv")
    candidate_series_path = _write_csv(candidate_series, output_dir / "candidate_series.csv")
    code_evidence_path = _write_csv(code_evidence, output_dir / "code_evidence.csv")

    _plot_summary(
        candidate_series=candidate_series,
        output_path=config.plot.summary_output_path,
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
        line_width=config.plot.line_width,
        font_scale=config.plot.font_scale,
        section_title_scale=config.plot.section_title_scale,
        y_axis_unit=config.plot.y_axis_unit,
    )

    return {
        "output_plot": config.plot.summary_output_path,
        "code_catalog_csv": code_catalog_path,
        "candidate_series_csv": candidate_series_path,
        "code_evidence_csv": code_evidence_path,
    }
