"""Plotting utilities for analysis figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .shares import PanelPair
from .metrics import r2_45


def plot_share_panels(
    *,
    pairs: Sequence[PanelPair],
    output_path: Path,
    title: str | None,
    point_alpha: float,
    point_size: float,
    axis_padding: float,
    point_color: str,
    use_latex: bool,
    latex_preamble: str,
    annotation_by_year: Mapping[int, str] | None = None,
    annotation_pos: tuple[float, float] = (0.05, 0.9),
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D

    if use_latex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["text.latex.preamble"] = latex_preamble

    n_panels = len(pairs)
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6), squeeze=False)

    for idx, pair in enumerate(pairs):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        x = pair.data["share_t"].to_numpy()
        y = pair.data["share_t1"].to_numpy()
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
        ax.set_xlabel(f"{pair.x_year} Trade Shares")
        ax.set_ylabel(f"{pair.y_year} Trade Shares")
        if annotation_by_year and pair.x_year in annotation_by_year:
            text = annotation_by_year[pair.x_year]
        else:
            r2 = r2_45(x, y)
            text = rf"$R^2$ = {r2:.3f}"
        ax.text(annotation_pos[0], annotation_pos[1], text, transform=ax.transAxes)

    for idx in range(n_panels, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        fig.delaxes(axes[row][col])

    legend_y = 0.98
    top_margin = 0.94
    if title:
        fig.suptitle(title, y=0.965)
        legend_y = 0.93
        top_margin = 0.92

    if pairs:
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
                bbox_to_anchor=(0.5, legend_y),
                ncol=2,
                frameon=False,
                handletextpad=0.4,
                columnspacing=1.0,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, top_margin])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_chain_length_panels(
    *,
    data: pd.DataFrame,
    output_path: Path,
    title: str | None,
    point_color: str,
    point_size: float = 6.0,
    use_latex: bool,
    latex_preamble: str,
    metrics: Sequence[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    if use_latex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["text.latex.preamble"] = latex_preamble

    if metrics is None:
        metrics = [
            "r2_45_weighted_symmetric",
            "exposure_weighted",
            "diffuseness_weighted",
        ]
    metric_map = {
        "r2_45_weighted_symmetric": ("one_minus_r2_sym", "1 − $R^2_{sym}$"),
        "delta_r2_45_weighted_symmetric": (
            "delta_one_minus_r2_sym",
            "$\\Delta(1 - R^2_{sym})$",
        ),
        "mae_weighted": ("mae_weighted", "wMAE"),
        "delta_mae_weighted": ("delta_mae_weighted", "$\\Delta$ wMAE"),
        "exposure_weighted": ("exposure_weighted", "$E_t$"),
        "diffuseness_weighted": ("diffuseness_weighted", "$H_t$"),
        "diffuse_exposure": ("diffuse_exposure", "$D_t$"),
    }
    metrics = [name.lower() for name in metrics]
    metric_rows = [metric_map[name] for name in metrics if name in metric_map]
    if not metric_rows:
        metric_rows = [metric_map["r2_45_weighted_symmetric"]]
    directions = ["backward", "forward"]
    fig, axes = plt.subplots(
        nrows=len(metric_rows),
        ncols=len(directions),
        figsize=(8, 6),
        sharex=False,
        squeeze=False,
    )

    y_limits: dict[str, tuple[float, float]] = {}
    for metric, _ in metric_rows:
        vals = data.loc[data[metric].notna(), metric]
        if vals.empty:
            y_limits[metric] = (0.0, 1.0)
            continue
        lo = float(vals.min())
        hi = float(vals.max())
        span = hi - lo
        if span <= 0:
            pad = 0.05
        else:
            pad = min(0.02, span * 0.15)
        lower = lo - pad
        upper = hi + pad
        if "delta" not in metric:
            lower = max(0.0, lower)
        lower = max(-1.0, lower)
        upper = min(1.0, upper)
        y_limits[metric] = (lower, upper)

    for col, direction in enumerate(directions):
        df_dir = data[data["direction"] == direction].sort_values("chain_length")
        for row, (metric, label) in enumerate(metric_rows):
            ax = axes[row][col]
            marker = "o"
            if metric == "mae_weighted":
                marker = None
            ax.plot(
                df_dir["chain_length"],
                df_dir[metric],
                marker=marker,
                markersize=point_size if marker else 0.0,
                color=point_color,
                linewidth=1.0,
            )
            ax.set_ylim(*y_limits[metric])
            ax.set_xlabel("Chain length")
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"{direction.title()} chaining")

    if title:
        fig.suptitle(title, y=0.98)
        top_margin = 0.92
    else:
        top_margin = 0.95

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, top_margin])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_chain_length_delta_panels(
    *,
    data: pd.DataFrame,
    output_path: Path,
    title: str | None,
    point_color: str,
    point_size: float = 6.0,
    use_latex: bool,
    latex_preamble: str,
    spearman_by_direction: dict[str, float] | None = None,
    metrics: Sequence[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import MaxNLocator

    if use_latex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["text.latex.preamble"] = latex_preamble

    metric_map = {
        "mae_weighted": ("mae_weighted", "wMAE"),
        "mae_weighted_step": ("mae_weighted_step", "$\\mathrm{wMAE}_\\ell$"),
        "diffuse_exposure": ("diffuse_exposure", "$\\mathrm{w}D_\\ell$"),
    }
    if metrics is None:
        metrics = ["mae_weighted", "mae_weighted_step", "diffuse_exposure"]
    selected = [name.lower() for name in metrics]
    invalid = [name for name in selected if name not in metric_map]
    if invalid:
        raise ValueError(
            f"Unknown chain-length delta metric(s): {invalid}. Allowed: {sorted(metric_map)}"
        )
    metric_rows = [metric_map[name] for name in selected]
    if not metric_rows:
        raise ValueError("No metrics selected for chain-length delta plot.")
    directions = ["backward", "forward"]
    fig, axes = plt.subplots(
        nrows=len(metric_rows),
        ncols=len(directions),
        figsize=(8, 6),
        sharex=False,
        squeeze=False,
    )

    y_limits: dict[str, tuple[float, float]] = {}
    for metric, _ in metric_rows:
        vals = data.loc[data[metric].notna(), metric]
        if vals.empty:
            y_limits[metric] = (0.0, 1.0)
            continue
        lo = float(vals.min())
        hi = float(vals.max())
        span = hi - lo
        if span <= 0:
            pad = 0.05
        else:
            pad = min(0.02, span * 0.15)
        lower = lo - pad
        upper = hi + pad
        if "delta" not in metric:
            lower = max(0.0, lower)
        lower = max(-1.0, lower)
        upper = min(1.0, upper)
        y_limits[metric] = (lower, upper)

    for col, direction in enumerate(directions):
        df_dir = data[data["direction"] == direction].sort_values("chain_length")
        anchor_year = None
        if not df_dir.empty and "anchor_year" in df_dir:
            anchor_year = int(df_dir["anchor_year"].iloc[0])
        for row, (metric, label) in enumerate(metric_rows):
            ax = axes[row][col]
            x_vals = df_dir["chain_length"]
            y_vals = df_dir[metric]
            if metric == "mae_weighted":
                ax.plot(
                    x_vals,
                    y_vals,
                    marker=None,
                    markersize=0.0,
                    color=point_color,
                    linewidth=1.0,
                )
            else:
                markerline, stemlines, _ = ax.stem(
                    x_vals,
                    y_vals,
                    linefmt="-",
                    markerfmt="o",
                    basefmt=" ",
                )
                markerline.set_markersize(max(1.5, point_size - 1.5))
                markerline.set_color(point_color)
                stem_color = point_color
                if str(point_color).lower() in {"black", "#000000"}:
                    stem_color = "#666666"
                stemlines.set_linewidth(0.5)
                stemlines.set_color(stem_color)
            ax.set_ylim(*y_limits[metric])
            if metric in {
                "mae_weighted",
                "mae_weighted_step",
                "diffuse_exposure",
            }:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if row == len(metric_rows) - 1:
                ax.set_xlabel("Base year vintage (HS/CN revision)")
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("")
            if row == 0:
                header = f"{direction.title()} chaining"
                if spearman_by_direction and direction in spearman_by_direction:
                    rho_val = spearman_by_direction[direction]
                    if np.isfinite(rho_val):
                        header = (
                            f"{header}\n"
                            + r"$\rho_S(\mathrm{wMAE}_\ell, \mathrm{w}D_\ell)$"
                            + f" = {rho_val:.2f}"
                        )
                ax.set_title(header)
            if anchor_year is not None:
                hs_years = [2022, 2017, 2012, 2007, 2002, 1996, 1992, 1988]
                ticks: list[tuple[int, str]] = []
                for year in hs_years:
                    length = abs(anchor_year - year)
                    if not x_vals.empty and length >= int(x_vals.min()) and length <= int(
                        x_vals.max()
                    ):
                        ticks.append((length, f"HS {year}"))
                ticks = sorted(ticks, key=lambda item: item[0])
                if ticks:
                    ax.set_xticks([item[0] for item in ticks])
                    ax.set_xticklabels([item[1] for item in ticks], rotation=30, ha="right")
                    for x_tick, _ in ticks:
                        ax.axvline(x_tick, color="#cccccc", linewidth=0.6, alpha=0.3)
            if metric == "mae_weighted":
                ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.25)

    if title:
        fig.suptitle(title, y=0.98)
        top_margin = 0.92
    else:
        top_margin = 0.95

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, top_margin])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_revision_validation_heatmap(
    *,
    data: pd.DataFrame,
    output_path: Path,
    title: str | None,
    use_latex: bool,
    latex_preamble: str,
    show_annotations: bool = False,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec

    if use_latex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["text.latex.preamble"] = latex_preamble

    metric_blocks = [
        (
            "Local persistence",
            [
                ("non_revised_mae", "Local baseline MAE"),
                ("break_year_mae", "Break-year MAE"),
                ("excess_break_mae", "Excess break MAE"),
            ],
        ),
        (
            "Weight robustness",
            [
                ("instability_p50", "Median instability"),
                (
                    "instability_importance_weighted_mean",
                    "Importance-weighted instability",
                ),
            ],
        ),
        (
            "Break-year sample size",
            [
                ("n_points_break", "Observations"),
            ],
        ),
    ]
    periods = data["period"].astype(str).tolist()
    labels = [f"{period[:4]}-{period[6:]}" for period in periods]

    def _format_scale_value(column: str, value: float) -> str:
        if column == "n_points_break":
            return f"{int(round(value))}"
        return f"{value:.3f}"

    def _build_block_arrays(
        metric_rows: list[tuple[str, str]],
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float] | None]]:
        plot_data = np.full((len(metric_rows), len(data)), np.nan, dtype=float)
        value_data = np.full((len(metric_rows), len(data)), np.nan, dtype=float)
        row_ranges: list[tuple[float, float] | None] = []
        for row_idx, (column, _label) in enumerate(metric_rows):
            values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
            value_data[row_idx, :] = values
            finite = np.isfinite(values)
            if not finite.any():
                row_ranges.append(None)
                continue
            raw_vals = values[finite]
            if column == "n_points_break":
                row_vals = np.log1p(raw_vals)
            else:
                row_vals = raw_vals
            lo = float(np.min(row_vals))
            hi = float(np.max(row_vals))
            row_ranges.append((float(np.min(raw_vals)), float(np.max(raw_vals))))
            if hi > lo:
                if column == "n_points_break":
                    scaled = (np.log1p(values) - lo) / (hi - lo)
                    scaled = 1.0 - scaled
                else:
                    scaled = (values - lo) / (hi - lo)
            else:
                scaled = np.full_like(values, 0.5, dtype=float)
            plot_data[row_idx, finite] = scaled[finite]
        return plot_data, value_data, row_ranges

    fig_width = 9.4
    fig_height = 3.3 if title else 3.0
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    grid = GridSpec(
        nrows=3,
        ncols=2,
        figure=fig,
        width_ratios=[22, 3.8],
        height_ratios=[3, 2, 1],
    )
    axes_flat = [fig.add_subplot(grid[row_idx, 0]) for row_idx in range(3)]
    scale_axes = [fig.add_subplot(grid[row_idx, 1]) for row_idx in range(3)]
    for ax in axes_flat[1:]:
        ax.sharex(axes_flat[0])
    cmap = LinearSegmentedColormap.from_list(
        "revision_validation_gray_blue",
        ["#edf2f7", "#cbd5e1", "#94a3b8", "#475569"],
    )

    for block_idx, (block_title, metric_rows) in enumerate(metric_blocks):
        ax = axes_flat[block_idx]
        scale_ax = scale_axes[block_idx]
        plot_data, value_data, row_ranges = _build_block_arrays(metric_rows)
        ax.imshow(plot_data, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(block_title, fontsize=10, fontweight="bold", pad=8)
        ax.set_yticks(np.arange(len(metric_rows)))
        ax.set_yticklabels([label for _, label in metric_rows])
        if block_idx == len(metric_blocks) - 1:
            tick_idx = np.arange(1, len(labels), 2)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(
                [labels[idx] for idx in tick_idx],
                rotation=55,
                ha="right",
                rotation_mode="anchor",
            )
            ax.tick_params(axis="x", which="major", labelbottom=True)
        else:
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="major", labelbottom=False)
        for row_idx in range(len(metric_rows)):
            for col_idx in range(len(data)):
                value = value_data[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                if show_annotations:
                    shade = plot_data[row_idx, col_idx]
                    text_color = "white" if np.isfinite(shade) and shade >= 0.58 else "black"
                    column_name = metric_rows[row_idx][0]
                    if column_name == "n_points_break":
                        text = f"{int(round(value))}"
                    else:
                        text = f"{value:.3f}"
                    ax.text(
                        col_idx,
                        row_idx,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=text_color,
                    )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(metric_rows), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0)

        gradient = np.linspace(0.0, 1.0, 256, dtype=float).reshape(1, -1)
        scale_ax.set_xlim(0.0, 1.0)
        scale_ax.set_ylim(len(metric_rows) - 0.5, -0.5)
        scale_ax.axis("off")
        for row_idx, (column, _label) in enumerate(metric_rows):
            row_range = row_ranges[row_idx]
            if row_range is None:
                continue
            row_gradient = gradient[:, ::-1] if column == "n_points_break" else gradient
            scale_ax.imshow(
                row_gradient,
                aspect="auto",
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                extent=(0.23, 0.77, row_idx + 0.15, row_idx - 0.15),
            )
            scale_ax.text(
                0.17,
                row_idx,
                _format_scale_value(column, row_range[0]),
                ha="right",
                va="center",
                fontsize=7.5,
                color="black",
            )
            scale_ax.text(
                0.83,
                row_idx,
                _format_scale_value(column, row_range[1]),
                ha="left",
                va="center",
                fontsize=7.5,
                color="black",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if title:
        fig.suptitle(title, y=0.98)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_chained_link_distribution_panels(
    *,
    data: pd.DataFrame,
    output_path: Path,
    title: str | None,
    use_latex: bool,
    latex_preamble: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import PercentFormatter
    from matplotlib.patches import Patch

    order = ["1:1", "m:1", "1:n", "m:n"]
    panel_titles = {
        "backward": "Focal CN2024 codes",
        "forward": "Focal CN1988 codes",
    }

    def _year_ticks(years: np.ndarray) -> tuple[list[int], list[int]]:
        years_int = sorted(int(year) for year in years.tolist())
        if not years_int:
            return [], []
        start = years_int[0]
        end = years_int[-1]
        span = end - start
        if span <= 12:
            step = 1
        elif span <= 24:
            step = 2
        else:
            step = 3
        major = list(range(start, end + 1, step))
        if major[-1] != end:
            major.append(end)
        if major[0] != start:
            major.insert(0, start)
        minor = list(range(start, end + 1))
        return sorted(set(major)), minor

    sequential_gray_blue = ["#edf2f7", "#cbd5e1", "#94a3b8", "#475569"]

    def _render(*, render_with_latex: bool) -> None:
        rcParams["text.usetex"] = bool(render_with_latex)
        if render_with_latex:
            rcParams["font.family"] = "serif"
            rcParams["text.latex.preamble"] = latex_preamble

        palette_values = sequential_gray_blue[: len(order)]
        palette = {key: value for key, value in zip(order, palette_values)}

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=False, squeeze=False)
        axes_flat = axes[:, 0]

        for ax, panel_direction in zip(axes_flat, ["backward", "forward"]):
            panel = data.loc[data["panel_direction"] == panel_direction].copy()
            if panel.empty:
                continue
            pivot = (
                panel.pivot_table(
                    index="compare_year",
                    columns="relationship",
                    values="share_anchor_codes",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .reindex(columns=order, fill_value=0.0)
                .sort_index()
            )
            x = pivot.index.to_numpy(dtype=float)
            y = [pivot[col].to_numpy(dtype=float) for col in order]
            colors = [palette[col] for col in order]
            ax.stackplot(x, y, colors=colors, linewidth=0.4, edgecolor="black", zorder=1)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(float(x.min()), float(x.max()))
            ax.margins(x=0.0, y=0.0)
            ax.set_ylabel("Share of focal codes")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.set_yticks(np.linspace(0.0, 1.0, 6))
            major_ticks, minor_ticks = _year_ticks(pivot.index.to_numpy(dtype=int))
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.tick_params(axis="x", labelsize=8, length=3.5, width=0.6, color="#4d4d4d")
            ax.tick_params(axis="y", labelsize=8, length=3.5, width=0.6, color="#4d4d4d")
            ax.set_axisbelow(False)
            ax.grid(
                axis="y",
                which="major",
                color="#3a3a3a",
                linewidth=0.7,
                alpha=0.35,
                linestyle="-",
                zorder=5,
            )
            ax.grid(
                axis="x",
                which="major",
                color="#3a3a3a",
                linewidth=0.5,
                alpha=0.18,
                linestyle="-",
                zorder=5,
            )
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#666666")
            ax.set_title(panel_titles.get(panel_direction, panel_direction.title()))
            ax.set_xlabel("Comparison year")

        legend_handles = [Patch(facecolor=palette[key], edgecolor="black", label=key) for key in order]
        legend_y = 0.98
        top_margin = 0.93
        if title:
            fig.suptitle(title, y=0.98)
            legend_y = 0.94
            top_margin = 0.9
        fig.legend(
            legend_handles,
            order,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=4,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, top_margin])
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    try:
        _render(render_with_latex=use_latex)
    except RuntimeError:
        if not use_latex:
            raise
        _render(render_with_latex=False)


def plot_chained_link_distribution_bar_panels(
    *,
    data: pd.DataFrame,
    output_path: Path,
    title: str | None,
    use_latex: bool,
    latex_preamble: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import PercentFormatter
    from matplotlib.patches import Patch

    order = ["1:1", "m:1", "1:n", "m:n"]
    panel_titles = {
        "backward": "Focal CN2024 codes",
        "forward": "Focal CN1988 codes",
    }
    sequential_gray_blue = ["#edf2f7", "#cbd5e1", "#94a3b8", "#475569"]

    def _year_ticks(years: np.ndarray) -> list[int]:
        years_int = sorted(int(year) for year in years.tolist())
        if not years_int:
            return []
        start = years_int[0]
        end = years_int[-1]
        span = end - start
        if span <= 12:
            step = 1
        elif span <= 24:
            step = 2
        else:
            step = 3
        major = list(range(start, end + 1, step))
        if major[-1] != end:
            major.append(end)
        if major[0] != start:
            major.insert(0, start)
        return sorted(set(major))

    def _render(*, render_with_latex: bool) -> None:
        rcParams["text.usetex"] = bool(render_with_latex)
        if render_with_latex:
            rcParams["font.family"] = "serif"
            rcParams["text.latex.preamble"] = latex_preamble

        palette = {key: value for key, value in zip(order, sequential_gray_blue[: len(order)])}
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=False, squeeze=False)
        axes_flat = axes[:, 0]

        for ax, panel_direction in zip(axes_flat, ["backward", "forward"]):
            panel = data.loc[data["panel_direction"] == panel_direction].copy()
            if panel.empty:
                continue
            pivot = (
                panel.pivot_table(
                    index="compare_year",
                    columns="relationship",
                    values="share_anchor_codes",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                .reindex(columns=order, fill_value=0.0)
                .sort_index()
            )
            x = pivot.index.to_numpy(dtype=float)
            bottom = np.zeros(len(pivot), dtype=float)
            width = 0.9
            for relationship in order:
                values = pivot[relationship].to_numpy(dtype=float)
                ax.bar(
                    x,
                    values,
                    bottom=bottom,
                    width=width,
                    color=palette[relationship],
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=1,
                    align="center",
                )
                bottom += values

            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(float(x.min()) - 0.5, float(x.max()) + 0.5)
            ax.set_ylabel("Share of focal codes")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.set_yticks(np.linspace(0.0, 1.0, 6))
            ax.set_xticks(_year_ticks(pivot.index.to_numpy(dtype=int)))
            ax.tick_params(axis="x", labelsize=8, length=3.5, width=0.6, color="#4d4d4d")
            ax.tick_params(axis="y", labelsize=8, length=3.5, width=0.6, color="#4d4d4d")
            ax.set_axisbelow(False)
            ax.grid(axis="y", which="major", color="#3a3a3a", linewidth=0.7, alpha=0.35, zorder=5)
            ax.grid(axis="x", which="major", color="#3a3a3a", linewidth=0.5, alpha=0.15, zorder=5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#666666")
            ax.set_title(panel_titles.get(panel_direction, panel_direction.title()))
            ax.set_xlabel("Comparison year")

        legend_handles = [Patch(facecolor=palette[key], edgecolor="black", label=key) for key in order]
        legend_y = 0.98
        top_margin = 0.93
        if title:
            fig.suptitle(title, y=0.98)
            legend_y = 0.94
            top_margin = 0.9
        fig.legend(
            legend_handles,
            order,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=4,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, top_margin])
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    try:
        _render(render_with_latex=use_latex)
    except RuntimeError:
        if not use_latex:
            raise
        _render(render_with_latex=False)


def plot_crm_revision_exposure_panels(
    *,
    data: pd.DataFrame,
    output_path: Path,
    title: str | None,
    use_latex: bool,
    latex_preamble: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import PercentFormatter
    from matplotlib.lines import Line2D

    anchor_year = int(data["anchor_year"].iloc[0]) if not data.empty else 2023
    population_styles = {
        "crm_anchor_codes": {
            "label": f"CRM-related CN{anchor_year} codes",
            "color": "#334155",
            "linestyle": "-",
        },
        "all_anchor_codes": {
            "label": f"All observed CN{anchor_year} codes",
            "color": "#94a3b8",
            "linestyle": "--",
        },
    }
    metric = "ever_unknown_weight_step"

    def _year_ticks(years: np.ndarray) -> tuple[list[int], list[int]]:
        years_int = sorted(int(year) for year in years.tolist())
        if not years_int:
            return [], []
        start = years_int[0]
        end = years_int[-1]
        minor = list(range(start, end + 1))
        revision_years = [2022, 2017, 2012, 2007, 2002, 1996, 1992, 1988]
        major = [year for year in revision_years if start <= year <= end]
        if start not in major:
            major.append(start)
        return sorted(set(major)), minor

    def _render(*, render_with_latex: bool) -> None:
        rcParams["text.usetex"] = bool(render_with_latex)
        if render_with_latex:
            rcParams["font.family"] = "serif"
            rcParams["text.latex.preamble"] = latex_preamble

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8.2, 3.8), squeeze=True)
        backward = data.loc[data["panel_direction"] == "backward"].copy()
        backward_metric = backward.loc[backward["metric"] == metric].copy()
        all_years = sorted(set(backward_metric["compare_year"].astype(int).tolist()))
        if anchor_year not in all_years:
            all_years.append(anchor_year)
            all_years = sorted(all_years)

        for population, style in population_styles.items():
            back_series = (
                backward_metric.loc[backward_metric["population"] == population]
                .sort_values("compare_year")
                .copy()
            )
            if not back_series.empty:
                anchor_row = back_series.iloc[[-1]].copy()
                anchor_row["compare_year"] = anchor_year
                anchor_row["n_codes"] = 0
                anchor_row["share_codes"] = 0.0
                back_series = pd.concat([back_series, anchor_row], ignore_index=True)
                back_series = back_series.sort_values("compare_year")
                ax.plot(
                    back_series["compare_year"],
                    back_series["share_codes"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=1.9,
                    zorder=3,
                )

        ax.set_ylabel("Share of anchor codes")
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_axisbelow(False)
        ax.grid(axis="y", which="major", color="#3a3a3a", linewidth=0.7, alpha=0.30, zorder=5)
        ax.grid(axis="x", which="major", color="#3a3a3a", linewidth=0.55, alpha=0.18, zorder=5)
        ax.grid(axis="x", which="minor", color="#3a3a3a", linewidth=0.35, alpha=0.08, zorder=5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#666666")

        if all_years:
            major_ticks, minor_ticks = _year_ticks(np.array(all_years, dtype=int))
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_xlim(float(min(all_years)), float(max(all_years)))
            ax.set_xticklabels([f"HS {year}" for year in major_ticks], rotation=0, ha="center")
            for x_tick in major_ticks:
                ax.axvline(x_tick, color="#b8b8b8", linewidth=0.6, alpha=0.25, zorder=4)
        ax.tick_params(axis="x", which="major", labelsize=8, length=4.0, width=0.6, color="#4d4d4d")
        ax.tick_params(axis="x", which="minor", length=2.0, width=0.45, color="#777777")
        ax.tick_params(axis="y", labelsize=8, length=3.5, width=0.6, color="#4d4d4d")
        ax.set_xlabel("Comparison year")

        legend_handles = [
            Line2D(
                [],
                [],
                color=population_styles[key]["color"],
                linestyle=population_styles[key]["linestyle"],
                linewidth=1.8,
                label=population_styles[key]["label"],
            )
            for key in ["crm_anchor_codes", "all_anchor_codes"]
        ]

        legend_y = 0.955
        top_margin = 0.94
        if title:
            fig.suptitle(title, y=0.965)
            legend_y = 0.935
            top_margin = 0.955
        fig.legend(
            legend_handles,
            [handle.get_label() for handle in legend_handles],
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=2,
            frameon=False,
            handletextpad=0.5,
            columnspacing=1.0,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, top_margin])
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    try:
        _render(render_with_latex=use_latex)
    except RuntimeError:
        if not use_latex:
            raise
        _render(render_with_latex=False)


def plot_sampling_robustness_panels(
    *,
    data: pd.DataFrame,
    output_path: Path,
    title: str | None,
    point_alpha: float,
    point_size: float,
    point_color: str,
    histogram_bins: int,
    use_latex: bool,
    latex_preamble: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    if use_latex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["text.latex.preamble"] = latex_preamble

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.6), squeeze=False)
    ax_hist = axes[0][0]
    ax_scatter = axes[0][1]

    instability = data["max_minus_min"].astype(float).to_numpy()
    importance = data["importance_product"].astype(float).to_numpy()

    ax_hist.hist(
        instability,
        bins=max(1, histogram_bins),
        color="#5f9ed1",
        edgecolor="#3a6c92",
        linewidth=0.6,
    )
    ax_hist.set_xlabel("Difference between Maximum and Minimum Coefficient")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(0.0, 1.0)

    ax_scatter.scatter(
        instability,
        importance,
        s=point_size,
        alpha=point_alpha,
        color=point_color,
        edgecolors="none",
    )
    ax_scatter.set_xlabel("Difference between Maximum and Minimum Coefficient")
    ax_scatter.set_ylabel("Relative Importance in Group")
    ax_scatter.set_xlim(0.0, 1.0)
    ax_scatter.set_ylim(0.0, 1.0)

    if title:
        fig.suptitle(title, y=0.98)
        top_margin = 0.9
    else:
        top_margin = 0.95

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, top_margin])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
