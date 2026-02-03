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
        "mae_weighted": ("mae_weighted", "wMAE"),
        "exposure_weighted": ("exposure_weighted", "$E_w$"),
        "diffuseness_weighted": ("diffuseness_weighted", "$H_w$"),
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
        else:
            lo = float(vals.min())
            hi = float(vals.max())
            pad = 0.02 if hi > lo else 0.05
            y_limits[metric] = (max(0.0, lo - pad), min(1.0, hi + pad))

    for col, direction in enumerate(directions):
        df_dir = data[data["direction"] == direction].sort_values("chain_length")
        for row, (metric, label) in enumerate(metric_rows):
            ax = axes[row][col]
            ax.plot(
                df_dir["chain_length"],
                df_dir[metric],
                marker="o",
                markersize=point_size,
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
