"""Public apply-stage entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .annual import WEIGHT_STRATEGIES, apply_weights_to_annual_period_impl
from .chained_wide import (
    MEASURE_COLUMNS,
    MONTHLY_DATA_DIR,
    apply_chained_weights_wide_for_month_range as _apply_chained_weights_wide_for_month_range_impl,
    apply_chained_weights_wide_for_range as _apply_chained_weights_wide_for_range_impl,
)
from ..weights.finalize import finalize_weights_table_impl
from ..chaining.engine import (
    ChainedWeightsOutput,
    DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    DEFAULT_CHAINED_WEIGHTS_DIR,
)
from ..estimation.shares import ANNUAL_DATA_DIR
from ..weights.schema import DEFAULT_WEIGHTS_DIR


@dataclass(frozen=True)
class ApplyDiagnostics:
    period: str
    direction: str
    strategy: str
    origin_year: str
    target_year: str
    n_rows_input: int
    n_rows_output: int
    n_codes_input: int
    n_codes_weighted: int
    n_codes_missing: int


@dataclass(frozen=True)
class ChainedApplySummary:
    origin_year: str
    target_year: str
    n_rows_input: int
    n_rows_output: int
    n_codes_input: int
    n_missing_value: int
    n_missing_quantity: int


def finalize_weights_table(
    weights: pd.DataFrame,
    *,
    neg_tol: float = 1e-6,
    pos_tol: float = 1e-10,
    row_sum_tol: float = 1e-9,
) -> pd.DataFrame:
    """Clamp small weights and renormalize rows by from_code."""
    return finalize_weights_table_impl(
        weights,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
    )


def apply_chained_weights_wide_for_range(
    *,
    start_year: int,
    end_year: int,
    target_year: int,
    measures: Sequence[str] = MEASURE_COLUMNS,
    annual_base_dir: Path = ANNUAL_DATA_DIR,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    output_chained_weights_dir: Path = DEFAULT_CHAINED_WEIGHTS_DIR,
    output_chained_diagnostics_dir: Path = DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    output_base_dir: Path = Path("outputs/apply"),
    output_summary_path: Path | None = None,
    chained_outputs: Sequence[ChainedWeightsOutput] | None = None,
    finalize_weights: bool = False,
    neg_tol: float = 1e-6,
    pos_tol: float = 1e-10,
    row_sum_tol: float = 1e-6,
    assume_identity_for_missing: bool = True,
    fail_on_missing: bool = True,
    revised_codes_by_step: dict[tuple[str, str], Iterable[str]] | None = None,
    strict_revised_link_validation: bool = False,
    write_unresolved_details: bool = False,
    skip_existing: bool = True,
    max_workers: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    return _apply_chained_weights_wide_for_range_impl(
        start_year=start_year,
        end_year=end_year,
        target_year=target_year,
        measures=measures,
        annual_base_dir=annual_base_dir,
        weights_dir=weights_dir,
        output_chained_weights_dir=output_chained_weights_dir,
        output_chained_diagnostics_dir=output_chained_diagnostics_dir,
        output_base_dir=output_base_dir,
        output_summary_path=output_summary_path,
        chained_outputs=chained_outputs,
        finalize_weights=finalize_weights,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
        assume_identity_for_missing=assume_identity_for_missing,
        fail_on_missing=fail_on_missing,
        revised_codes_by_step=revised_codes_by_step,
        strict_revised_link_validation=strict_revised_link_validation,
        write_unresolved_details=write_unresolved_details,
        skip_existing=skip_existing,
        max_workers=max_workers,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def apply_chained_weights_wide_for_month_range(
    *,
    start_year: int,
    end_year: int,
    target_year: int,
    measures: Sequence[str] = MEASURE_COLUMNS,
    monthly_base_dir: Path = MONTHLY_DATA_DIR,
    annual_base_dir: Path = ANNUAL_DATA_DIR,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    output_chained_weights_dir: Path = DEFAULT_CHAINED_WEIGHTS_DIR,
    output_chained_diagnostics_dir: Path = DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    output_base_dir: Path = Path("outputs/apply"),
    output_summary_path: Path | None = None,
    chained_outputs: Sequence[ChainedWeightsOutput] | None = None,
    finalize_weights: bool = False,
    neg_tol: float = 1e-6,
    pos_tol: float = 1e-10,
    row_sum_tol: float = 1e-6,
    assume_identity_for_missing: bool = True,
    fail_on_missing: bool = True,
    revised_codes_by_step: dict[tuple[str, str], Iterable[str]] | None = None,
    strict_revised_link_validation: bool = False,
    write_unresolved_details: bool = False,
    skip_existing: bool = True,
    max_workers: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    return _apply_chained_weights_wide_for_month_range_impl(
        start_year=start_year,
        end_year=end_year,
        target_year=target_year,
        measures=measures,
        monthly_base_dir=monthly_base_dir,
        annual_base_dir=annual_base_dir,
        weights_dir=weights_dir,
        output_chained_weights_dir=output_chained_weights_dir,
        output_chained_diagnostics_dir=output_chained_diagnostics_dir,
        output_base_dir=output_base_dir,
        output_summary_path=output_summary_path,
        chained_outputs=chained_outputs,
        finalize_weights=finalize_weights,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
        assume_identity_for_missing=assume_identity_for_missing,
        fail_on_missing=fail_on_missing,
        revised_codes_by_step=revised_codes_by_step,
        strict_revised_link_validation=strict_revised_link_validation,
        write_unresolved_details=write_unresolved_details,
        skip_existing=skip_existing,
        max_workers=max_workers,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def apply_weights_to_annual_period(
    *,
    period: str,
    direction: str = "a_to_b",
    strategy: str = "weights_split",
    annual_base_dir: Path = ANNUAL_DATA_DIR,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    output_base_dir: Path = Path("outputs/apply"),
    measure_columns: Sequence[str] = MEASURE_COLUMNS,
    fail_on_missing: bool = True,
    assume_identity_for_missing: bool = True,
    finalize_weights: bool = False,
    neg_tol: float = 1e-6,
    pos_tol: float = 1e-10,
    row_sum_tol: float = 1e-9,
) -> ApplyDiagnostics:
    row = apply_weights_to_annual_period_impl(
        period=period,
        direction=direction,
        strategy=strategy,
        annual_base_dir=annual_base_dir,
        weights_dir=weights_dir,
        output_base_dir=output_base_dir,
        measure_columns=measure_columns,
        fail_on_missing=fail_on_missing,
        assume_identity_for_missing=assume_identity_for_missing,
        finalize_weights=finalize_weights,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
    )
    return ApplyDiagnostics(
        period=str(row["period"]),
        direction=str(row["direction"]),
        strategy=str(row["strategy"]),
        origin_year=str(row["origin_year"]),
        target_year=str(row["target_year"]),
        n_rows_input=int(row["n_rows_input"]),
        n_rows_output=int(row["n_rows_output"]),
        n_codes_input=int(row["n_codes_input"]),
        n_codes_weighted=int(row["n_codes_weighted"]),
        n_codes_missing=int(row["n_codes_missing"]),
    )


__all__ = [
    "ApplyDiagnostics",
    "ChainedApplySummary",
    "WEIGHT_STRATEGIES",
    "apply_weights_to_annual_period",
    "apply_chained_weights_wide_for_range",
    "apply_chained_weights_wide_for_month_range",
    "finalize_weights_table",
]
