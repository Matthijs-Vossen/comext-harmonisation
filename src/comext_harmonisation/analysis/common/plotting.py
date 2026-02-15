"""Plotting utilities for share-stability analyses."""

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
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import MaxNLocator

    if use_latex:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "serif"
        rcParams["text.latex.preamble"] = latex_preamble

    metric_rows = [
        ("mae_weighted", "wMAE"),
        ("mae_weighted_step", "$\\mathrm{wMAE}_\\ell$"),
        ("diffuse_exposure", "$D_\\ell$"),
    ]
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
                            + r"$\rho_S(\mathrm{wMAE}_\ell, D_\ell)$"
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
