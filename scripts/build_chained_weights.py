"""Build chained CN conversion weights for a target vintage."""

from __future__ import annotations

import argparse
from pathlib import Path

from comext_harmonisation.estimation.chaining import (
    build_chained_weights_for_range,
    DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    DEFAULT_CHAINED_WEIGHTS_DIR,
)
from comext_harmonisation.weights import DEFAULT_WEIGHTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chained conversion weights for a target vintage."
    )
    parser.add_argument("--start-year", required=True, type=int, help="First origin year.")
    parser.add_argument("--end-year", required=True, type=int, help="Last origin year (inclusive).")
    parser.add_argument("--target-year", required=True, type=int, help="Target vintage year.")
    parser.add_argument(
        "--measure",
        default="BOTH",
        choices=["VALUE_EUR", "QUANTITY_KG", "BOTH"],
        help="Measures to chain.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=str(DEFAULT_WEIGHTS_DIR),
        help="Directory with adjacent weights.",
    )
    parser.add_argument(
        "--output-weights-dir",
        type=str,
        default=str(DEFAULT_CHAINED_WEIGHTS_DIR),
        help="Output directory for chained weights.",
    )
    parser.add_argument(
        "--output-diagnostics-dir",
        type=str,
        default=str(DEFAULT_CHAINED_DIAGNOSTICS_DIR),
        help="Output directory for chained diagnostics.",
    )
    parser.add_argument(
        "--finalize-weights",
        action="store_true",
        help="Clamp small weights and renormalize after chaining.",
    )
    parser.add_argument(
        "--threshold-abs",
        type=float,
        default=1e-3,
        help="Absolute threshold for clamping weights when finalizing.",
    )
    parser.add_argument(
        "--row-sum-tol",
        type=float,
        default=1e-6,
        help="Row sum tolerance for chained weights.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing weights during chaining (do not fail fast).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measures = ["VALUE_EUR", "QUANTITY_KG"] if args.measure == "BOTH" else [args.measure]
    outputs = build_chained_weights_for_range(
        start_year=args.start_year,
        end_year=args.end_year,
        target_year=args.target_year,
        measures=measures,
        weights_dir=Path(args.weights_dir),
        output_weights_dir=Path(args.output_weights_dir),
        output_diagnostics_dir=Path(args.output_diagnostics_dir),
        finalize_weights=args.finalize_weights,
        threshold_abs=args.threshold_abs,
        row_sum_tol=args.row_sum_tol,
        fail_on_missing=not args.allow_missing,
    )
    for output in outputs:
        print(
            f"{output.origin_year}->{output.target_year} {output.measure} "
            f"({output.direction}) rows={len(output.weights)}"
        )


if __name__ == "__main__":
    main()
