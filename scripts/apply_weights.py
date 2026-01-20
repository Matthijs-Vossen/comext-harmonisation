#!/usr/bin/env python3
"""Apply conversion weights to annual data for a single period."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply conversion weights to annual trade data for one period.",
    )
    parser.add_argument("--period", required=True, help="Concordance period, e.g. 20092010.")
    parser.add_argument(
        "--direction",
        default="a_to_b",
        choices=["a_to_b", "b_to_a"],
        help="Conversion direction.",
    )
    parser.add_argument(
        "--strategy",
        default="weights_split",
        choices=["weights_split", "weights_value", "weights_quantity"],
        help="Which weights to apply to which measures.",
    )
    parser.add_argument(
        "--no-identity-for-missing",
        action="store_true",
        help="Fail if any codes are missing weights instead of assuming identity.",
    )
    parser.add_argument(
        "--finalize-weights",
        action="store_true",
        help="Clamp tiny weights and renormalize before applying.",
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
        default=1e-9,
        help="Row-sum tolerance for finalized weights.",
    )
    parser.add_argument(
        "--annual-base-dir",
        default="data/extracted_annual_no_confidential/products_like",
        help="Directory containing annual parquet files.",
    )
    parser.add_argument(
        "--weights-dir",
        default="outputs/weights",
        help="Directory containing estimated weights.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/harmonised/annual",
        help="Base directory for harmonised outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from comext_harmonisation import apply_weights_to_annual_period

    diagnostics = apply_weights_to_annual_period(
        period=str(args.period),
        direction=args.direction,
        strategy=args.strategy,
        annual_base_dir=Path(args.annual_base_dir),
        weights_dir=Path(args.weights_dir),
        output_base_dir=Path(args.output_dir),
        assume_identity_for_missing=not args.no_identity_for_missing,
        finalize_weights=bool(args.finalize_weights),
        threshold_abs=float(args.threshold_abs),
        row_sum_tol=float(args.row_sum_tol),
    )

    print(
        "Applied",
        diagnostics.strategy,
        "for",
        diagnostics.period,
        "-> CN",
        diagnostics.target_year,
    )
    print("rows:", diagnostics.n_rows_input, "->", diagnostics.n_rows_output)


if __name__ == "__main__":
    main()
