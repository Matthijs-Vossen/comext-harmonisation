#!/usr/bin/env python3
"""Run the LT weight estimation pipeline for a single period."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LT weight estimation for one concordance period.",
    )
    parser.add_argument("--period", required=True, help="Concordance period, e.g. 20092010.")
    parser.add_argument(
        "--direction",
        default="a_to_b",
        choices=["a_to_b", "b_to_a"],
        help="Conversion direction.",
    )
    parser.add_argument(
        "--measure",
        default="BOTH",
        choices=["VALUE_EUR", "QUANTITY_KG", "BOTH"],
        help="Measure used for estimating weights.",
    )
    parser.add_argument(
        "--flow",
        default="1",
        help="Flow code to use when preparing shares (default: 1).",
    )
    parser.add_argument(
        "--concordance-path",
        default="data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
        help="Path to the CN concordance XLS file.",
    )
    parser.add_argument(
        "--concordance-sheet",
        default=None,
        help="Optional sheet name or index for the concordance XLS.",
    )
    parser.add_argument(
        "--annual-base-dir",
        default="data/extracted_annual_no_confidential/products_like",
        help="Directory containing annual parquet files.",
    )
    parser.add_argument(
        "--include-aggregate-codes",
        action="store_true",
        help="Include aggregate country codes during estimation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory (writes weights/diagnostics/summary.csv under it).",
    )
    parser.add_argument(
        "--no-fail-on-status",
        action="store_true",
        help="Do not raise if the solver reports a non-solved status.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from ..estimation.runner import (
        run_weight_estimation_for_period,
        run_weight_estimation_for_period_multi,
    )

    exclude_aggregate_codes = True
    if args.include_aggregate_codes:
        exclude_aggregate_codes = False

    if args.measure == "BOTH":
        results = run_weight_estimation_for_period_multi(
            period=str(args.period),
            direction=args.direction,
            measures=["VALUE_EUR", "QUANTITY_KG"],
            concordance_path=Path(args.concordance_path),
            concordance_sheet=args.concordance_sheet,
            annual_base_dir=Path(args.annual_base_dir),
            flow=str(args.flow),
            exclude_aggregate_codes=exclude_aggregate_codes,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            fail_on_status=not args.no_fail_on_status,
        )
        for outputs in results:
            print("weights:", outputs.weights_path)
            print("deterministic:", outputs.deterministic_path)
            print("diagnostics:", outputs.diagnostics_path)
        summary_path = (
            Path(args.output_dir) / "summary.csv"
            if args.output_dir
            else Path("outputs/weights/summary.csv")
        )
        print("summary csv:", summary_path)
    else:
        outputs = run_weight_estimation_for_period(
            period=str(args.period),
            direction=args.direction,
            measure=args.measure,
            concordance_path=Path(args.concordance_path),
            concordance_sheet=args.concordance_sheet,
            annual_base_dir=Path(args.annual_base_dir),
            flow=str(args.flow),
            exclude_aggregate_codes=exclude_aggregate_codes,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            fail_on_status=not args.no_fail_on_status,
        )

        print("weights:", outputs.weights_path)
        print("deterministic:", outputs.deterministic_path)
        print("diagnostics:", outputs.diagnostics_path)
        if outputs.summary_csv_path is not None:
            print("summary csv:", outputs.summary_csv_path)


if __name__ == "__main__":
    main()
