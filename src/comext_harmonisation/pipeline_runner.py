"""Pipeline orchestration for end-to-end estimation → chaining → apply runs."""

from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import shutil
from typing import Any, Callable, Iterable, Sequence

import pandas as pd
from tqdm import tqdm


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


def _partition_estimation_periods(
    *,
    periods: Sequence[tuple[str, str]],
    measures: Sequence[str],
    estimate_weights_dir: Path,
    skip_existing: bool,
) -> tuple[list[tuple[str, str]], int]:
    if not skip_existing:
        return list(periods), 0

    to_process: list[tuple[str, str]] = []
    skipped = 0
    for period, direction in periods:
        if all(
            _weights_exist(estimate_weights_dir, period, direction, measure) for measure in measures
        ):
            skipped += 1
            continue
        to_process.append((period, direction))
    return to_process, skipped


def _count_existing_outputs(paths: Iterable[Path], *, skip_existing: bool) -> tuple[int, int]:
    paths_list = list(paths)
    if not skip_existing:
        return len(paths_list), 0
    skipped = sum(1 for path in paths_list if path.exists())
    return len(paths_list) - skipped, skipped


def _annual_output_paths(
    *,
    start_year: int,
    end_year: int,
    target_year: int,
    apply_output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for year in range(start_year, end_year + 1):
        origin = str(year)
        paths.append(
            apply_output_dir / f"CN{target_year}" / "annual" / f"comext_{origin}_wide.parquet"
        )
    return paths


def _monthly_output_paths(
    *,
    start_year: int,
    end_year: int,
    target_year: int,
    apply_output_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            period = f"{year}{month:02d}"
            paths.append(
                apply_output_dir / f"CN{target_year}" / "monthly" / f"comext_{period}_wide.parquet"
            )
    return paths


def _build_chains_for_measure(
    *,
    chain_builder: Callable[..., list],
    measure: str,
    start_year: int,
    end_year: int,
    target_year: int,
    code_universe: dict[int, set[str]],
    estimate_weights_dir: Path,
    chain_weights_dir: Path,
    output_diagnostics_dir: Path,
    config,
    revised_codes_by_step,
) -> list:
    return chain_builder(
        start_year=start_year,
        end_year=end_year,
        target_year=target_year,
        measures=[measure],
        code_universe=code_universe,
        weights_dir=estimate_weights_dir,
        output_weights_dir=chain_weights_dir,
        output_diagnostics_dir=output_diagnostics_dir,
        finalize_weights=config.chaining.finalize_weights,
        neg_tol=config.chaining.neg_tol,
        pos_tol=config.chaining.pos_tol,
        row_sum_tol=config.chaining.row_sum_tol,
        fail_on_missing=config.chaining.fail_on_missing,
        revised_codes_by_step=revised_codes_by_step,
        strict_revised_link_validation=config.chaining.strict_revised_link_validation,
        write_unresolved_details=config.chaining.write_unresolved_details,
    )


def _append_apply_summary_sanity(summary: pd.DataFrame, sanity_items: list[tuple[str, str]]) -> None:
    totals = [
        ("sum_value_eur_input", "sum_value_eur_w_value"),
        ("sum_quantity_kg_input", "sum_quantity_kg_w_value"),
        ("sum_value_eur_input", "sum_value_eur_w_quantity"),
        ("sum_quantity_kg_input", "sum_quantity_kg_w_quantity"),
    ]
    for src, dst in totals:
        if src in summary.columns and dst in summary.columns:
            diff = (summary[dst] - summary[src]).abs()
            sanity_items.append((f"{src}->{dst}", f"max_abs={diff.max()} mean_abs={diff.mean()}"))
    if "n_missing_value" in summary.columns:
        sanity_items.append(("missing_value max", str(summary["n_missing_value"].max())))
    if "n_missing_quantity" in summary.columns:
        sanity_items.append(("missing_quantity max", str(summary["n_missing_quantity"].max())))
    if "n_rows_input" in summary.columns and "n_rows_output" in summary.columns:
        sanity_items.append(("output rows < input", str((summary["n_rows_output"] < summary["n_rows_input"]).sum())))
        sanity_items.append(("output rows > input", str((summary["n_rows_output"] > summary["n_rows_input"]).sum())))


def _run_apply_stage(
    *,
    section_key: str,
    total_key: str,
    total_count: int,
    processed: int,
    skipped: int,
    apply_fn: Callable[..., Any],
    apply_kwargs: dict[str, Any],
    log,
) -> None:
    _log_section(
        section_key,
        [
            (total_key, str(total_count)),
            (f"{total_key.split('_')[0]}_to_run", str(processed)),
            (f"{total_key.split('_')[0]}_skipped", str(skipped)),
        ],
        log,
    )
    apply_fn(**apply_kwargs)
    _log_section(
        f"{section_key}_complete",
        [("processed", str(processed)), ("skipped", str(skipped))],
        log,
    )


def _log_section(section_key: str, items: list[tuple[str, str]], write_line) -> None:
    write_line()
    write_line(f"[{section_key}]")
    if not items:
        return
    width = max(len(key) for key, _ in items)
    for key, value in items:
        write_line(f"{key.ljust(width)} = {value}")


def _print_config_summary(config, write_line, run_dir: Path, chain_dir: Path, apply_dir: Path) -> None:
    years = config.years
    items = [
        ("years", f"{years.start}-{years.end} -> CN{years.target}"),
        ("measures", ", ".join(config.measures)),
        (
            "stages",
            (
                f"estimate={config.stages.estimate}, "
                f"chain={config.stages.chain}, "
                f"apply_annual={config.stages.apply_annual}, "
                f"apply_monthly={config.stages.apply_monthly}"
            ),
        ),
        (
            "workers",
            (
                f"matrices={config.parallel.max_workers_matrices}, "
                f"solver={config.parallel.max_workers_solver}, "
                f"chain={config.parallel.max_workers_chain}, "
                f"apply={config.parallel.max_workers_apply}"
            ),
        ),
        (
            "paths",
            (
                f"estimate={config.paths.estimate_weights_dir}, "
                f"run={run_dir}, "
                f"chain={chain_dir}, "
                f"apply={apply_dir}"
            ),
        ),
    ]
    _log_section("pipeline_settings", items, write_line)


def run_pipeline_with_config(config, *, config_path: Path | None = None) -> Path:
    from comext_harmonisation import (
        run_weight_estimation_for_period_multi,
        build_chained_weights_for_range,
        build_code_universe_from_annual,
        apply_chained_weights_wide_for_range,
        apply_chained_weights_wide_for_month_range,
        read_concordance_xls,
        build_revised_code_index_from_concordance,
    )

    measures = config.measures
    start_year = config.years.start
    end_year = config.years.end
    target_year = config.years.target

    run_base_dir = config.paths.run_base_dir
    run_base_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y-%m-%dT%H-%M-%S_CN") + str(target_year)
    run_dir = run_base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    config_copy_path = run_dir / "config.yaml"
    if config_path is not None:
        shutil.copy2(config_path, config_copy_path)

    def _log(message: str = "") -> None:
        if message is None:
            message = ""
        try:
            tqdm.write(message)
        except Exception:
            print(message)
        with log_path.open("a") as handle:
            handle.write(f"{message}\n")

    chain_dir = run_dir / "chain"
    apply_dir = run_dir / "apply"
    _log_section(
        "run_metadata",
        [
            ("run_id", str(run_id)),
            ("config", str(config_copy_path)),
            ("run_dir", str(run_dir)),
        ],
        _log,
    )
    _print_config_summary(config, _log, run_dir, chain_dir, apply_dir)
    estimate_weights_dir = config.paths.estimate_weights_dir
    estimate_diagnostics_dir = config.paths.estimate_diagnostics_dir
    estimate_summary_path = config.paths.estimate_summary_path
    chain_weights_dir = run_dir / "chain"
    chain_diagnostics_dir = run_dir / "chain"
    apply_output_dir = run_dir / "apply"

    stage_stats: dict[str, int | str] = {}
    revised_codes_by_step = None
    strict_revised_validation = (
        config.chaining.strict_revised_link_validation
        or config.apply.strict_revised_link_validation
    )
    if strict_revised_validation:
        concordance_edges = read_concordance_xls(
            str(config.paths.concordance_path),
            sheet_name=config.paths.concordance_sheet,
        )
        revised_codes_by_step = build_revised_code_index_from_concordance(concordance_edges)

    if config.stages.estimate:
        periods = _estimate_periods(start_year, end_year, target_year)
        to_process, skipped = _partition_estimation_periods(
            periods=periods,
            measures=measures,
            estimate_weights_dir=estimate_weights_dir,
            skip_existing=config.estimation.skip_existing,
        )
        processed = 0
        _log_section(
            "stage_estimation",
            [
                ("periods_total", str(len(periods))),
                ("periods_to_run", str(len(to_process))),
                ("periods_skipped", str(skipped)),
            ],
            _log,
        )
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
        _log_section(
            "stage_estimation_complete",
            [("processed", str(processed)), ("skipped", str(skipped))],
            _log,
        )
        stage_stats["estimate_processed"] = processed
        stage_stats["estimate_skipped"] = skipped

    chained_outputs = []
    if config.stages.chain:
        code_universe = build_code_universe_from_annual(
            annual_base_dir=config.paths.annual_base_dir,
            years=range(start_year, end_year + 1),
        )
        if config.parallel.max_workers_chain and config.parallel.max_workers_chain > 1 and len(measures) > 1:
            diag_paths: list[Path] = []

            def _chain_for_measure(measure: str) -> list:
                diag_dir = chain_diagnostics_dir / "_by_measure" / measure.lower()
                outputs = _build_chains_for_measure(
                    chain_builder=build_chained_weights_for_range,
                    measure=measure,
                    start_year=start_year,
                    end_year=end_year,
                    target_year=target_year,
                    code_universe=code_universe,
                    estimate_weights_dir=estimate_weights_dir,
                    chain_weights_dir=chain_weights_dir,
                    output_diagnostics_dir=diag_dir,
                    config=config,
                    revised_codes_by_step=revised_codes_by_step,
                )
                diag_paths.append(diag_dir / f"CN{target_year}" / "diagnostics.csv")
                return outputs

            _log_section(
                "stage_chaining",
                [("measures", f"{len(measures)} (parallel)")],
                _log,
            )
            with ThreadPoolExecutor(max_workers=config.parallel.max_workers_chain) as executor:
                futures = {executor.submit(_chain_for_measure, measure): measure for measure in measures}
                for future in as_completed(futures):
                    chained_outputs.extend(future.result())

            combined_path = chain_diagnostics_dir / f"CN{target_year}" / "diagnostics.csv"
            _combine_chain_diagnostics(combined_path=combined_path, diag_paths=diag_paths)
            _log_section(
                "stage_chaining_complete",
                [("measures", f"{len(measures)} (parallel)")],
                _log,
            )
        else:
            _log_section(
                "stage_chaining",
                [("measures", str(len(measures)))],
                _log,
            )
            for measure in measures:
                chained_outputs.extend(
                    _build_chains_for_measure(
                        chain_builder=build_chained_weights_for_range,
                        measure=measure,
                        start_year=start_year,
                        end_year=end_year,
                        target_year=target_year,
                        code_universe=code_universe,
                        estimate_weights_dir=estimate_weights_dir,
                        chain_weights_dir=chain_weights_dir,
                        output_diagnostics_dir=chain_diagnostics_dir,
                        config=config,
                        revised_codes_by_step=revised_codes_by_step,
                    )
                )
            _log_section(
                "stage_chaining_complete",
                [("measures", str(len(measures)))],
                _log,
            )
        stage_stats["chain_measures"] = len(measures)

    if config.stages.apply_annual:
        processed, skipped = _count_existing_outputs(
            _annual_output_paths(
                start_year=start_year,
                end_year=end_year,
                target_year=target_year,
                apply_output_dir=apply_output_dir,
            ),
            skip_existing=config.apply.skip_existing,
        )
        _run_apply_stage(
            section_key="stage_apply_annual",
            total_key="years_total",
            total_count=end_year - start_year + 1,
            processed=processed,
            skipped=skipped,
            apply_fn=apply_chained_weights_wide_for_range,
            apply_kwargs={
                "start_year": start_year,
                "end_year": end_year,
                "target_year": target_year,
                "measures": measures,
                "annual_base_dir": config.paths.annual_base_dir,
                "weights_dir": estimate_weights_dir,
                "output_chained_weights_dir": chain_weights_dir,
                "output_chained_diagnostics_dir": chain_diagnostics_dir,
                "output_base_dir": apply_output_dir,
                "chained_outputs": chained_outputs if chained_outputs else None,
                "finalize_weights": config.chaining.finalize_weights,
                "neg_tol": config.chaining.neg_tol,
                "pos_tol": config.chaining.pos_tol,
                "row_sum_tol": config.chaining.row_sum_tol,
                "assume_identity_for_missing": config.apply.assume_identity_for_missing,
                "fail_on_missing": config.apply.fail_on_missing,
                "revised_codes_by_step": revised_codes_by_step,
                "strict_revised_link_validation": config.apply.strict_revised_link_validation,
                "write_unresolved_details": config.apply.write_unresolved_details,
                "skip_existing": config.apply.skip_existing,
                "max_workers": config.parallel.max_workers_apply,
                "show_progress": True,
                "progress_desc": "Applying annual",
            },
            log=_log,
        )
        stage_stats["apply_annual_processed"] = processed
        stage_stats["apply_annual_skipped"] = skipped

    if config.stages.apply_monthly:
        processed, skipped = _count_existing_outputs(
            _monthly_output_paths(
                start_year=start_year,
                end_year=end_year,
                target_year=target_year,
                apply_output_dir=apply_output_dir,
            ),
            skip_existing=config.apply.skip_existing,
        )
        _run_apply_stage(
            section_key="stage_apply_monthly",
            total_key="months_total",
            total_count=(end_year - start_year + 1) * 12,
            processed=processed,
            skipped=skipped,
            apply_fn=apply_chained_weights_wide_for_month_range,
            apply_kwargs={
                "start_year": start_year,
                "end_year": end_year,
                "target_year": target_year,
                "measures": measures,
                "monthly_base_dir": config.paths.monthly_base_dir,
                "weights_dir": estimate_weights_dir,
                "output_chained_weights_dir": chain_weights_dir,
                "output_chained_diagnostics_dir": chain_diagnostics_dir,
                "output_base_dir": apply_output_dir,
                "chained_outputs": chained_outputs if chained_outputs else None,
                "finalize_weights": config.chaining.finalize_weights,
                "neg_tol": config.chaining.neg_tol,
                "pos_tol": config.chaining.pos_tol,
                "row_sum_tol": config.chaining.row_sum_tol,
                "assume_identity_for_missing": config.apply.assume_identity_for_missing,
                "fail_on_missing": config.apply.fail_on_missing,
                "revised_codes_by_step": revised_codes_by_step,
                "strict_revised_link_validation": config.apply.strict_revised_link_validation,
                "write_unresolved_details": config.apply.write_unresolved_details,
                "skip_existing": config.apply.skip_existing,
                "max_workers": config.parallel.max_workers_apply,
                "show_progress": True,
                "progress_desc": "Applying monthly",
            },
            log=_log,
        )
        stage_stats["apply_monthly_processed"] = processed
        stage_stats["apply_monthly_skipped"] = skipped

    sanity_items: list[tuple[str, str]] = []
    if config.stages.chain:
        chain_diag_path = chain_diagnostics_dir / f"CN{target_year}" / "diagnostics.csv"
        if chain_diag_path.exists():
            chain_diag = pd.read_csv(chain_diag_path)
            max_row_sum_dev = chain_diag["max_row_sum_dev"].max()
            sanity_items.append(("chain max_row_sum_dev", str(max_row_sum_dev)))
        else:
            sanity_items.append(("chain diagnostics", "missing"))

    if config.stages.apply_monthly:
        monthly_summary = apply_output_dir / f"CN{target_year}" / "monthly" / "summary.csv"
        if monthly_summary.exists():
            summary = pd.read_csv(monthly_summary)
            if "origin_period" in summary.columns:
                summary["origin_period"] = summary["origin_period"].astype(str).str.zfill(6)
            _append_apply_summary_sanity(summary, sanity_items)
        else:
            sanity_items.append(("monthly summary", "missing"))

    if config.stages.apply_annual:
        annual_summary = apply_output_dir / f"CN{target_year}" / "summary.csv"
        if annual_summary.exists():
            summary = pd.read_csv(annual_summary)
            _append_apply_summary_sanity(summary, sanity_items)
        else:
            sanity_items.append(("annual summary", "missing"))

    if sanity_items:
        _log_section("sanity_checks", sanity_items, _log)

    index_path = run_base_dir / "index.csv"
    index_row = {
        "run_id": run_id,
        "timestamp_local": run_id.replace("run_", ""),
        "config_path": str(config_copy_path),
        "run_dir": str(run_dir),
        "start_year": start_year,
        "end_year": end_year,
        "target_year": target_year,
        "measures": ",".join(measures),
        "estimate_weights_dir": str(estimate_weights_dir),
        "chain_dir": str(chain_weights_dir),
        "apply_dir": str(apply_output_dir),
        "estimate_processed": stage_stats.get("estimate_processed", 0),
        "estimate_skipped": stage_stats.get("estimate_skipped", 0),
        "apply_annual_processed": stage_stats.get("apply_annual_processed", 0),
        "apply_annual_skipped": stage_stats.get("apply_annual_skipped", 0),
        "apply_monthly_processed": stage_stats.get("apply_monthly_processed", 0),
        "apply_monthly_skipped": stage_stats.get("apply_monthly_skipped", 0),
    }
    index_df = pd.DataFrame([index_row])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.exists()
    index_df.to_csv(index_path, mode="a", index=False, header=write_header)
    return run_dir


def run_pipeline_from_config_path(config_path: Path) -> Path:
    from .pipeline_config import load_pipeline_config

    config = load_pipeline_config(config_path)
    return run_pipeline_with_config(config, config_path=config_path)
