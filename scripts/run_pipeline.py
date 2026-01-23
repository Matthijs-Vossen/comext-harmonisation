"""Run end-to-end estimation → chaining → apply pipeline using YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end harmonisation pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a pipeline YAML config file.",
    )
    return parser.parse_args()


def _estimate_periods(start_year: int, end_year: int, target_year: int) -> list[tuple[str, str]]:
    periods: list[tuple[str, str]] = []
    if start_year < target_year:
        for year in range(start_year, target_year):
            periods.append((f"{year}{year + 1}", "a_to_b"))
    if end_year > target_year:
        for year in range(target_year + 1, end_year + 1):
            periods.append((f"{year - 1}{year}", "b_to_a"))
    return periods


def _weights_exist(weights_dir: Path, period: str, direction: str, measure: str) -> bool:
    measure_tag = measure.lower()
    base = weights_dir / period / direction / measure_tag
    return (base / "weights_ambiguous.csv").exists() and (base / "weights_deterministic.csv").exists()


def _combine_chain_diagnostics(
    *,
    combined_path: Path,
    diag_paths: list[Path],
) -> None:
    frames = [pd.read_csv(path) for path in diag_paths if path.exists()]
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)

def _print_config_summary(config) -> None:
    years = config.years
    print("Pipeline run settings:")
    print(f"- years: {years.start}-{years.end} -> CN{years.target}")
    print(f"- measures: {', '.join(config.measures)}")
    print(
        "- stages: "
        f"estimate={config.stages.estimate}, "
        f"chain={config.stages.chain}, "
        f"apply_annual={config.stages.apply_annual}, "
        f"apply_monthly={config.stages.apply_monthly}"
    )
    print(
        "- workers: "
        f"matrices={config.parallel.max_workers_matrices}, "
        f"solver={config.parallel.max_workers_solver}, "
        f"chain={config.parallel.max_workers_chain}, "
        f"apply={config.parallel.max_workers_apply}"
    )
    print(
        "- paths: "
        f"estimate={config.paths.estimate_weights_dir}, "
        f"chain={config.paths.chain_weights_dir}, "
        f"apply={config.paths.apply_output_dir}"
    )
    print()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from comext_harmonisation import (
        run_weight_estimation_for_period_multi,
        build_chained_weights_for_range,
        apply_chained_weights_wide_for_range,
        apply_chained_weights_wide_for_month_range,
    )
    from comext_harmonisation.pipeline_config import load_pipeline_config

    config = load_pipeline_config(Path(args.config))
    print()
    _print_config_summary(config)
    measures = config.measures
    start_year = config.years.start
    end_year = config.years.end
    target_year = config.years.target

    estimate_weights_dir = config.paths.estimate_weights_dir
    estimate_diagnostics_dir = config.paths.estimate_diagnostics_dir
    estimate_summary_path = config.paths.estimate_summary_path
    chain_weights_dir = config.paths.chain_weights_dir
    chain_diagnostics_dir = config.paths.chain_diagnostics_dir
    apply_output_dir = config.paths.apply_output_dir

    if config.stages.estimate:
        print("Stage: estimation")
        periods = _estimate_periods(start_year, end_year, target_year)
        skipped = 0
        processed = 0
        to_process = []
        for period, direction in periods:
            if config.estimation.skip_existing and all(
                _weights_exist(estimate_weights_dir, period, direction, measure) for measure in measures
            ):
                skipped += 1
                continue
            to_process.append((period, direction))
        iterator = to_process
        if to_process:
            iterator = tqdm(to_process, desc="Estimating periods")
        for period, direction in iterator:
            run_weight_estimation_for_period_multi(
                period=period,
                direction=direction,
                measures=measures,
                concordance_path=config.paths.concordance_path,
                concordance_sheet=config.paths.concordance_sheet,
                annual_base_dir=config.paths.annual_base_dir,
                flow=config.estimation.flow,
                exclude_aggregate_codes=not config.estimation.include_aggregate_codes,
                output_weights_dir=estimate_weights_dir,
                output_diagnostics_dir=estimate_diagnostics_dir,
                output_summary_path=estimate_summary_path,
                fail_on_status=config.estimation.fail_on_status,
                max_workers_matrices=config.parallel.max_workers_matrices,
                max_workers_solver=config.parallel.max_workers_solver,
            )
            processed += 1
        print(f"Estimation complete: processed={processed}, skipped={skipped}")
        print()

    chained_outputs = []
    if config.stages.chain:
        print("Stage: chaining")
        if config.parallel.max_workers_chain and config.parallel.max_workers_chain > 1 and len(measures) > 1:
            diag_paths: list[Path] = []

            def _chain_for_measure(measure: str) -> list:
                diag_dir = chain_diagnostics_dir / "_by_measure" / measure.lower()
                outputs = build_chained_weights_for_range(
                    start_year=start_year,
                    end_year=end_year,
                    target_year=target_year,
                    measures=[measure],
                    weights_dir=estimate_weights_dir,
                    output_weights_dir=chain_weights_dir,
                    output_diagnostics_dir=diag_dir,
                    finalize_weights=config.chaining.finalize_weights,
                    threshold_abs=config.chaining.threshold_abs,
                    row_sum_tol=config.chaining.row_sum_tol,
                    fail_on_missing=config.chaining.fail_on_missing,
                )
                diag_paths.append(diag_dir / f"CN{target_year}" / "diagnostics.csv")
                return outputs

            with ThreadPoolExecutor(max_workers=config.parallel.max_workers_chain) as executor:
                futures = {executor.submit(_chain_for_measure, measure): measure for measure in measures}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Chaining measures"):
                    chained_outputs.extend(future.result())

            combined_path = chain_diagnostics_dir / f"CN{target_year}" / "diagnostics.csv"
            _combine_chain_diagnostics(combined_path=combined_path, diag_paths=diag_paths)
            print(f"Chaining complete: measures={len(measures)} (parallel)")
        else:
            for measure in tqdm(measures, desc="Chaining measures"):
                chained_outputs.extend(
                    build_chained_weights_for_range(
                        start_year=start_year,
                        end_year=end_year,
                        target_year=target_year,
                        measures=[measure],
                        weights_dir=estimate_weights_dir,
                        output_weights_dir=chain_weights_dir,
                        output_diagnostics_dir=chain_diagnostics_dir,
                        finalize_weights=config.chaining.finalize_weights,
                        threshold_abs=config.chaining.threshold_abs,
                        row_sum_tol=config.chaining.row_sum_tol,
                        fail_on_missing=config.chaining.fail_on_missing,
                    )
                )
            print(f"Chaining complete: measures={len(measures)}")
        print()

    if config.stages.apply_annual:
        print("Stage: apply annual")
        skipped = 0
        processed = 0
        for year in range(start_year, end_year + 1):
            origin = str(year)
            output_path = (
                apply_output_dir / f"CN{target_year}" / "annual" / f"comext_{origin}_wide.parquet"
            )
            if config.apply.skip_existing and output_path.exists():
                skipped += 1
            else:
                processed += 1
        apply_chained_weights_wide_for_range(
            start_year=start_year,
            end_year=end_year,
            target_year=target_year,
            measures=measures,
            annual_base_dir=config.paths.annual_base_dir,
            weights_dir=estimate_weights_dir,
            output_chained_weights_dir=chain_weights_dir,
            output_chained_diagnostics_dir=chain_diagnostics_dir,
            output_base_dir=apply_output_dir,
            chained_outputs=chained_outputs if chained_outputs else None,
            finalize_weights=config.chaining.finalize_weights,
            threshold_abs=config.chaining.threshold_abs,
            row_sum_tol=config.chaining.row_sum_tol,
            assume_identity_for_missing=config.apply.assume_identity_for_missing,
            fail_on_missing=config.apply.fail_on_missing,
            skip_existing=config.apply.skip_existing,
            max_workers=config.parallel.max_workers_apply,
            show_progress=True,
            progress_desc="Applying annual",
        )
        print(f"Apply annual complete: processed={processed}, skipped={skipped}")
        print()

    if config.stages.apply_monthly:
        print("Stage: apply monthly")
        skipped = 0
        processed = 0
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                period = f"{year}{month:02d}"
                output_path = (
                    apply_output_dir
                    / f"CN{target_year}"
                    / "monthly"
                    / f"comext_{period}_wide.parquet"
                )
                if config.apply.skip_existing and output_path.exists():
                    skipped += 1
                else:
                    processed += 1
        apply_chained_weights_wide_for_month_range(
            start_year=start_year,
            end_year=end_year,
            target_year=target_year,
            measures=measures,
            monthly_base_dir=config.paths.monthly_base_dir,
            weights_dir=estimate_weights_dir,
            output_chained_weights_dir=chain_weights_dir,
            output_chained_diagnostics_dir=chain_diagnostics_dir,
            output_base_dir=apply_output_dir,
            chained_outputs=chained_outputs if chained_outputs else None,
            finalize_weights=config.chaining.finalize_weights,
            threshold_abs=config.chaining.threshold_abs,
            row_sum_tol=config.chaining.row_sum_tol,
            assume_identity_for_missing=config.apply.assume_identity_for_missing,
            fail_on_missing=config.apply.fail_on_missing,
            skip_existing=config.apply.skip_existing,
            max_workers=config.parallel.max_workers_apply,
            show_progress=True,
            progress_desc="Applying monthly",
        )
        print(f"Apply monthly complete: processed={processed}, skipped={skipped}")
        print()


if __name__ == "__main__":
    main()
