"""Build chained weights and apply them to annual data with wide outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from comext_harmonisation.application import apply_chained_weights_wide_for_range
from comext_harmonisation.estimation.chaining import (
    DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    DEFAULT_CHAINED_WEIGHTS_DIR,
)
from comext_harmonisation.weights import DEFAULT_WEIGHTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chained weights and apply them to annual data."
    )
    parser.add_argument("--start-year", required=True, type=int)
    parser.add_argument("--end-year", required=True, type=int)
    parser.add_argument("--target-year", required=True, type=int)
    parser.add_argument(
        "--measure",
        default="BOTH",
        choices=["VALUE_EUR", "QUANTITY_KG", "BOTH"],
        help="Measures to chain and apply.",
    )
    parser.add_argument(
        "--annual-base-dir",
        type=str,
        default="data/extracted_annual_no_confidential/products_like",
        help="Directory containing annual parquet files.",
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=str(DEFAULT_WEIGHTS_DIR),
        help="Directory with adjacent weights.",
    )
    parser.add_argument(
        "--chained-weights-dir",
        type=str,
        default=str(DEFAULT_CHAINED_WEIGHTS_DIR),
        help="Directory to store chained weights.",
    )
    parser.add_argument(
        "--chained-diagnostics-dir",
        type=str,
        default=str(DEFAULT_CHAINED_DIAGNOSTICS_DIR),
        help="Directory to store chained diagnostics.",
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="outputs/apply",
        help="Base directory for harmonised outputs.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Optional CSV path for application summary.",
    )
    parser.add_argument(
        "--finalize-weights",
        action="store_true",
        help="Finalize chained weights (clamp/renorm) before apply.",
    )
    parser.add_argument(
        "--threshold-abs",
        type=float,
        default=1e-3,
        help="Absolute threshold for weight finalization.",
    )
    parser.add_argument(
        "--row-sum-tol",
        type=float,
        default=1e-6,
        help="Row sum tolerance for chained weights.",
    )
    parser.add_argument(
        "--no-identity-for-missing",
        action="store_true",
        help="Fail if weights are missing instead of assuming identity.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process years even if outputs/summary already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measures = ["VALUE_EUR", "QUANTITY_KG"] if args.measure == "BOTH" else [args.measure]

    summary = apply_chained_weights_wide_for_range(
        start_year=args.start_year,
        end_year=args.end_year,
        target_year=args.target_year,
        measures=measures,
        annual_base_dir=Path(args.annual_base_dir),
        weights_dir=Path(args.weights_dir),
        output_chained_weights_dir=Path(args.chained_weights_dir),
        output_chained_diagnostics_dir=Path(args.chained_diagnostics_dir),
        output_base_dir=Path(args.output_base_dir),
        output_summary_path=Path(args.summary_path) if args.summary_path else None,
        finalize_weights=args.finalize_weights,
        threshold_abs=args.threshold_abs,
        row_sum_tol=args.row_sum_tol,
        assume_identity_for_missing=not args.no_identity_for_missing,
        skip_existing=not args.no_skip_existing,
    )

    print(summary)


if __name__ == "__main__":
    main()
