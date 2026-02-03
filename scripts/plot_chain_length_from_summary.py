#!/usr/bin/env python3
"""Rebuild chain-length plot from an existing summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _ensure_src_on_path() -> None:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot chain-length analysis from summary CSV.")
    parser.add_argument(
        "--summary",
        default="outputs/analysis/chain_length/summary.csv",
        help="Path to chain-length summary.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/analysis/chain_length/chain_length.png",
        help="Output plot path",
    )
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument("--point-color", default="gray", help="Line/marker color")
    parser.add_argument("--point-size", type=float, default=6.0, help="Marker size")
    parser.add_argument("--use-latex", action="store_true", help="Enable LaTeX rendering")
    parser.add_argument(
        "--latex-preamble",
        default=r"\usepackage{newtxtext,newtxmath}",
        help="LaTeX preamble when --use-latex is set",
    )
    parser.add_argument(
        "--metrics",
        default="r2_45_weighted_symmetric,exposure_weighted,diffuseness_weighted",
        help="Comma-separated metrics to plot (e.g. r2_45_weighted_symmetric,mae_weighted,exposure_weighted)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure_src_on_path()
    from comext_harmonisation.analysis.common.plotting import plot_chain_length_panels
    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    data = pd.read_csv(summary_path)
    metrics = [item.strip() for item in str(args.metrics).split(",") if item.strip()]
    plot_chain_length_panels(
        data=data,
        output_path=Path(args.output),
        title=args.title,
        point_color=args.point_color,
        point_size=args.point_size,
        use_latex=args.use_latex,
        latex_preamble=args.latex_preamble,
        metrics=metrics,
    )
    print("plot:", args.output)


if __name__ == "__main__":
    main()
