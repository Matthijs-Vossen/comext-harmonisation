#!/usr/bin/env python3
"""Rebuild revision-validation heatmap from an existing summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot revision-validation analysis from summary CSV."
    )
    parser.add_argument(
        "--summary",
        default="outputs/analysis/revision_validation/summary.csv",
        help="Path to revision-validation summary.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/analysis/revision_validation/revision_validation_heatmap.png",
        help="Output plot path",
    )
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument(
        "--show-annotations",
        action="store_true",
        help="Show numeric in-cell annotations",
    )
    parser.add_argument(
        "--use-latex",
        action="store_true",
        default=True,
        help="Enable LaTeX rendering",
    )
    parser.add_argument(
        "--latex-preamble",
        default=r"\usepackage{newtxtext,newtxmath}",
        help="LaTeX preamble when --use-latex is set",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from ..analysis.common.plotting import plot_revision_validation_heatmap

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    data = pd.read_csv(summary_path)
    plot_revision_validation_heatmap(
        data=data,
        output_path=Path(args.output),
        title=args.title,
        use_latex=args.use_latex,
        latex_preamble=args.latex_preamble,
        show_annotations=args.show_annotations,
    )
    print("plot:", args.output)


if __name__ == "__main__":
    main()
