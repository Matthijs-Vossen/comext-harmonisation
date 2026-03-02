"""Chained wide apply engines for annual and monthly outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

from ..core.codes import chain_periods, normalize_codes
from ..core.diagnostics import append_detail_rows
from ..core.revised_links import normalize_revised_index
from ..weights.finalize import finalize_weights_table_impl
from ..chaining.engine import (
    ChainedWeightsOutput,
    DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    DEFAULT_CHAINED_WEIGHTS_DIR,
    build_chained_weights_for_range,
    build_code_universe_from_annual,
)
from ..estimation.shares import ANNUAL_DATA_DIR
from ..weights.schema import DEFAULT_WEIGHTS_DIR


MEASURE_COLUMNS = ("VALUE_EUR", "QUANTITY_KG")
MONTHLY_DATA_DIR = Path("data/extracted_no_confidential/products_like")


def _normalize_codes(series: pd.Series) -> pd.Series:
    return normalize_codes(series)


def _normalize_revised_index(
    revised_codes_by_step: Mapping[tuple[str, str], Iterable[str]] | None,
) -> dict[tuple[str, str], set[str]]:
    return normalize_revised_index(revised_codes_by_step)


def _chain_periods(origin_year: str | int, target_year: str | int) -> tuple[list[str], str]:
    return chain_periods(origin_year, target_year)


def _write_unresolved_details(rows: list[dict[str, object]], *, path: Path) -> None:
    append_detail_rows(
        rows,
        path=path,
        columns=[
            "origin_year",
            "target_year",
            "direction",
            "measure",
            "period",
            "step_index",
            "code",
            "reason",
            "count",
            "source",
        ],
    )


def _validate_chained_outputs_for_apply(
    *,
    chained_outputs: Sequence[ChainedWeightsOutput],
    target_year: int,
    measures: Sequence[str],
    strict_revised_link_validation: bool,
    revised_codes_by_step: Mapping[tuple[str, str], set[str]],
) -> list[dict[str, object]]:
    unresolved_rows: list[dict[str, object]] = []
    measure_set = {str(m).strip().upper() for m in measures}
    for output in chained_outputs:
        if output.origin_year == str(target_year):
            continue
        if output.measure not in measure_set:
            continue

        diagnostics = output.diagnostics if isinstance(output.diagnostics, pd.DataFrame) else pd.DataFrame()
        unresolved_total = 0
        if not diagnostics.empty and "n_unresolved_revised_total" in diagnostics.columns:
            unresolved_total = int(
                pd.to_numeric(
                    diagnostics["n_unresolved_revised_total"],
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            )
        if unresolved_total > 0:
            unresolved_rows.append(
                {
                    "origin_year": output.origin_year,
                    "target_year": str(target_year),
                    "direction": output.direction,
                    "measure": output.measure,
                    "period": "",
                    "step_index": -1,
                    "code": "",
                    "reason": "chain_diagnostics_unresolved_revised_total",
                    "count": unresolved_total,
                    "source": "chained_output_diagnostics",
                }
            )

        if not strict_revised_link_validation:
            continue
        periods, direction = _chain_periods(output.origin_year, target_year)
        if not periods:
            continue
        first_period = periods[0]
        revised_origin_codes = set(revised_codes_by_step.get((first_period, direction), set()))
        if not revised_origin_codes:
            continue
        if output.weights.empty:
            missing_revised = revised_origin_codes
        else:
            from_codes = set(_normalize_codes(output.weights["from_code"]).tolist())
            missing_revised = revised_origin_codes - from_codes
        for code in sorted(missing_revised):
            unresolved_rows.append(
                {
                    "origin_year": output.origin_year,
                    "target_year": str(target_year),
                    "direction": direction,
                    "measure": output.measure,
                    "period": first_period,
                    "step_index": 0,
                    "code": code,
                    "reason": "missing_revised_from_code_in_final_chain",
                    "count": 1,
                    "source": "final_chain_coverage",
                }
            )
    return unresolved_rows


def _prepare_weights(
    *,
    weights: pd.DataFrame,
    data_codes: set[str],
    assume_identity_for_missing: bool,
    fail_on_missing: bool,
) -> tuple[pd.DataFrame, int]:
    weights = weights[["from_code", "to_code", "weight"]].copy()
    weights["from_code"] = _normalize_codes(weights["from_code"])
    weights["to_code"] = _normalize_codes(weights["to_code"])
    missing_codes = data_codes - set(weights["from_code"])
    missing_count = len(missing_codes)
    if missing_codes and assume_identity_for_missing:
        identity = pd.DataFrame(
            {
                "from_code": list(missing_codes),
                "to_code": list(missing_codes),
                "weight": 1.0,
            }
        )
        weights = pd.concat([weights, identity], ignore_index=True)
        missing_codes = set()
    if missing_codes and fail_on_missing:
        sample = sorted(list(missing_codes))[:10]
        raise ValueError(f"Missing weights for {len(missing_codes)} codes; sample: {sample}")
    return weights, missing_count


def _apply_weights_to_frame(
    df: pd.DataFrame,
    *,
    weights: pd.DataFrame,
    measure_columns: Sequence[str],
    code_column: str = "PRODUCT_NC",
    fail_on_missing: bool = True,
    id_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    missing_cols = [col for col in measure_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required measure columns: {missing_cols}")

    data = df.copy()
    data[code_column] = _normalize_codes(data[code_column])

    weight_codes = set(weights["from_code"])
    data_codes = set(data[code_column].unique())
    missing_codes = data_codes - weight_codes
    if missing_codes and fail_on_missing:
        sample = sorted(list(missing_codes))[:10]
        raise ValueError(f"Missing weights for {len(missing_codes)} codes; sample: {sample}")

    merged = data.merge(weights, left_on=code_column, right_on="from_code", how="inner")
    for col in measure_columns:
        merged[col] = merged[col].astype(float) * merged["weight"].astype(float)

    merged[code_column] = merged["to_code"]
    merged = merged.drop(columns=["from_code", "to_code", "weight"])

    if id_columns is None:
        id_columns = [col for col in merged.columns if col not in measure_columns]
    result = merged.groupby(list(id_columns), as_index=False, sort=False)[
        list(measure_columns)
    ].sum()
    return result


def _apply_weights_wide(
    *,
    data: pd.DataFrame,
    weights_value: pd.DataFrame | None,
    weights_quantity: pd.DataFrame | None,
    assume_identity_for_missing: bool,
    fail_on_missing: bool,
) -> tuple[pd.DataFrame, int, int]:
    base = data.copy()
    base["PRODUCT_NC"] = _normalize_codes(base["PRODUCT_NC"])
    data_codes = set(base["PRODUCT_NC"].unique())
    id_cols = [col for col in base.columns if col not in MEASURE_COLUMNS]

    missing_value = 0
    missing_quantity = 0

    frames = []
    if weights_value is not None:
        prepared, missing_value = _prepare_weights(
            weights=weights_value,
            data_codes=data_codes,
            assume_identity_for_missing=assume_identity_for_missing,
            fail_on_missing=fail_on_missing,
        )
        converted = _apply_weights_to_frame(
            base,
            weights=prepared,
            measure_columns=["VALUE_EUR", "QUANTITY_KG"],
            fail_on_missing=False,
            id_columns=id_cols,
        )
        converted = converted.rename(
            columns={
                "VALUE_EUR": "VALUE_EUR_w_value",
                "QUANTITY_KG": "QUANTITY_KG_w_value",
            }
        )
        frames.append(converted)

    if weights_quantity is not None:
        prepared, missing_quantity = _prepare_weights(
            weights=weights_quantity,
            data_codes=data_codes,
            assume_identity_for_missing=assume_identity_for_missing,
            fail_on_missing=fail_on_missing,
        )
        converted = _apply_weights_to_frame(
            base,
            weights=prepared,
            measure_columns=["VALUE_EUR", "QUANTITY_KG"],
            fail_on_missing=False,
            id_columns=id_cols,
        )
        converted = converted.rename(
            columns={
                "VALUE_EUR": "VALUE_EUR_w_quantity",
                "QUANTITY_KG": "QUANTITY_KG_w_quantity",
            }
        )
        frames.append(converted)

    if not frames:
        raise ValueError("No weights provided for wide conversion output.")

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=id_cols, how="outer")

    for col in merged.columns:
        if col.endswith("_w_value") or col.endswith("_w_quantity"):
            merged[col] = merged[col].fillna(0.0)

    return merged, missing_value, missing_quantity


def _sum_or_zero(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(frame[column].sum())


def _resolve_chained_outputs_for_apply(
    *,
    start_year: int,
    end_year: int,
    target_year: int,
    measures: Sequence[str],
    annual_base_dir: Path,
    weights_dir: Path,
    output_chained_weights_dir: Path,
    output_chained_diagnostics_dir: Path,
    chained_outputs: Sequence[ChainedWeightsOutput] | None,
    finalize_weights: bool,
    neg_tol: float,
    pos_tol: float,
    row_sum_tol: float,
    fail_on_missing: bool,
    revised_index: Mapping[tuple[str, str], set[str]],
    strict_revised_link_validation: bool,
    write_unresolved_details: bool,
) -> Sequence[ChainedWeightsOutput]:
    if chained_outputs is not None:
        return chained_outputs
    code_universe = build_code_universe_from_annual(
        annual_base_dir=annual_base_dir,
        years=range(int(start_year), int(end_year) + 1),
    )
    return build_chained_weights_for_range(
        start_year=start_year,
        end_year=end_year,
        target_year=target_year,
        measures=measures,
        code_universe=code_universe,
        weights_dir=weights_dir,
        output_weights_dir=output_chained_weights_dir,
        output_diagnostics_dir=output_chained_diagnostics_dir,
        finalize_weights=finalize_weights,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
        fail_on_missing=fail_on_missing,
        revised_codes_by_step=revised_index,
        strict_revised_link_validation=strict_revised_link_validation,
        write_unresolved_details=write_unresolved_details,
    )


def _validate_or_raise_unresolved_for_apply(
    *,
    chained_outputs: Sequence[ChainedWeightsOutput],
    target_year: int,
    measures: Sequence[str],
    strict_revised_link_validation: bool,
    revised_index: Mapping[tuple[str, str], set[str]],
    output_base_dir: Path,
    write_unresolved_details: bool,
    fail_on_missing: bool,
) -> None:
    unresolved_path = output_base_dir / f"CN{target_year}" / "diagnostics" / "unresolved_details.csv"
    unresolved_rows = _validate_chained_outputs_for_apply(
        chained_outputs=chained_outputs,
        target_year=target_year,
        measures=measures,
        strict_revised_link_validation=strict_revised_link_validation,
        revised_codes_by_step=revised_index,
    )
    if unresolved_rows and write_unresolved_details:
        _write_unresolved_details(unresolved_rows, path=unresolved_path)
    if unresolved_rows and fail_on_missing:
        sample = [row["code"] for row in unresolved_rows if row.get("code")][:10]
        raise ValueError(
            "Unresolved revised links detected in chained outputs for apply; "
            f"sample codes: {sample}"
        )


def _finalize_chained_weights_by_year(
    *,
    chained_outputs: Sequence[ChainedWeightsOutput],
    neg_tol: float,
    pos_tol: float,
    row_sum_tol: float,
) -> dict[str, dict[str, pd.DataFrame]]:
    weights_by_year: dict[str, dict[str, pd.DataFrame]] = {}
    for output in chained_outputs:
        finalized = finalize_weights_table_impl(
            output.weights[["from_code", "to_code", "weight"]],
            neg_tol=neg_tol,
            pos_tol=pos_tol,
            row_sum_tol=row_sum_tol,
        )
        weights_by_year.setdefault(output.origin_year, {})[output.measure] = finalized
    return weights_by_year


def _load_existing_summary_rows(
    *,
    output_summary_path: Path,
    key_column: str,
    skip_existing: bool,
    zfill_width: int | None = None,
) -> tuple[set[str], list[dict[str, object]]]:
    existing_summary = pd.DataFrame()
    if skip_existing and output_summary_path.exists():
        existing_summary = pd.read_csv(output_summary_path)
        if key_column in existing_summary.columns:
            key_series = existing_summary[key_column].astype(str)
            if zfill_width is not None:
                key_series = key_series.str.zfill(zfill_width)
            existing_summary[key_column] = key_series
    existing_values = {str(value) for value in existing_summary.get(key_column, [])}
    rows = existing_summary.to_dict("records") if not existing_summary.empty else []
    return existing_values, rows


def _run_processing(
    *,
    items: Sequence[object],
    process_item: Callable[[object], dict[str, object]],
    max_workers: int | None,
    show_progress: bool,
    progress_desc: str,
) -> list[dict[str, object]]:
    if not items:
        return []

    show_progress = bool(show_progress)
    if max_workers is None or max_workers <= 1 or len(items) <= 1:
        iterator: Sequence[object] = items
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(items, desc=progress_desc, total=len(items))
        return [process_item(item) for item in iterator]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    progress = None
    if show_progress:
        from tqdm import tqdm

        progress = tqdm(total=len(items), desc=progress_desc)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return results


def _prepare_summary(
    *,
    rows: list[dict[str, object]],
    key_column: str,
    zfill_width: int | None = None,
) -> pd.DataFrame:
    summary = pd.DataFrame(rows)
    if key_column in summary.columns:
        key_series = summary[key_column].astype(str)
        if zfill_width is not None:
            key_series = key_series.str.zfill(zfill_width)
        summary[key_column] = key_series
        summary = summary.drop_duplicates(subset=[key_column], keep="last")
    return summary


def _build_identity_wide_output(data: pd.DataFrame, *, measures: Sequence[str]) -> pd.DataFrame:
    output = data.copy()
    output["PRODUCT_NC"] = _normalize_codes(output["PRODUCT_NC"])
    if "VALUE_EUR" in measures:
        output["VALUE_EUR_w_value"] = output["VALUE_EUR"]
        output["QUANTITY_KG_w_value"] = output["QUANTITY_KG"]
    if "QUANTITY_KG" in measures:
        output["VALUE_EUR_w_quantity"] = output["VALUE_EUR"]
        output["QUANTITY_KG_w_quantity"] = output["QUANTITY_KG"]
    drop_cols = ["VALUE_EUR", "QUANTITY_KG"]
    return output.drop(columns=[col for col in drop_cols if col in output.columns])


def _resolve_weights_for_origin(
    *,
    origin: str,
    target_year: int,
    measures: Sequence[str],
    weights_by_year: Mapping[str, Mapping[str, pd.DataFrame]],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    weights_for_year = weights_by_year.get(origin, {})
    weights_value = weights_for_year.get("VALUE_EUR")
    weights_quantity = weights_for_year.get("QUANTITY_KG")
    if "VALUE_EUR" in measures and weights_value is None:
        raise ValueError(f"Missing chained VALUE_EUR weights for {origin}->{target_year}")
    if "QUANTITY_KG" in measures and weights_quantity is None:
        raise ValueError(f"Missing chained QUANTITY_KG weights for {origin}->{target_year}")
    return (
        weights_value if "VALUE_EUR" in measures else None,
        weights_quantity if "QUANTITY_KG" in measures else None,
    )


def apply_chained_weights_wide_for_range(
    *,
    start_year: int,
    end_year: int,
    target_year: int,
    measures: Sequence[str] = ("VALUE_EUR", "QUANTITY_KG"),
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
    revised_codes_by_step: Mapping[tuple[str, str], Iterable[str]] | None = None,
    strict_revised_link_validation: bool = False,
    write_unresolved_details: bool = False,
    skip_existing: bool = True,
    max_workers: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    measures = [str(measure).strip().upper() for measure in measures]
    revised_index = _normalize_revised_index(revised_codes_by_step)
    chained_outputs = _resolve_chained_outputs_for_apply(
        start_year=start_year,
        end_year=end_year,
        target_year=target_year,
        measures=measures,
        annual_base_dir=annual_base_dir,
        weights_dir=weights_dir,
        output_chained_weights_dir=output_chained_weights_dir,
        output_chained_diagnostics_dir=output_chained_diagnostics_dir,
        chained_outputs=chained_outputs,
        finalize_weights=finalize_weights,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
        fail_on_missing=fail_on_missing,
        revised_index=revised_index,
        strict_revised_link_validation=strict_revised_link_validation,
        write_unresolved_details=write_unresolved_details,
    )
    _validate_or_raise_unresolved_for_apply(
        chained_outputs=chained_outputs,
        target_year=target_year,
        measures=measures,
        strict_revised_link_validation=strict_revised_link_validation,
        revised_index=revised_index,
        output_base_dir=output_base_dir,
        write_unresolved_details=write_unresolved_details,
        fail_on_missing=fail_on_missing,
    )

    weights_by_year = _finalize_chained_weights_by_year(
        chained_outputs=chained_outputs,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
    )

    if output_summary_path is None:
        output_summary_path = output_base_dir / f"CN{target_year}" / "summary.csv"
    existing_years, summary_rows = _load_existing_summary_rows(
        output_summary_path=output_summary_path,
        key_column="origin_year",
        skip_existing=skip_existing,
    )

    output_dir = output_base_dir / f"CN{target_year}" / "annual"
    output_dir.mkdir(parents=True, exist_ok=True)

    years_to_process: list[str] = []
    for year in range(int(start_year), int(end_year) + 1):
        origin = str(year)
        output_path = output_dir / f"comext_{origin}_wide.parquet"
        if skip_existing and output_path.exists() and origin in existing_years:
            continue
        years_to_process.append(origin)

    def _process_year(origin: str) -> dict[str, object]:
        data_path = annual_base_dir / f"comext_{origin}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing annual data file: {data_path}")
        data = pd.read_parquet(data_path)
        n_rows_input = len(data)
        n_codes_input = data["PRODUCT_NC"].astype(str).nunique()

        if origin == str(target_year):
            output = _build_identity_wide_output(data, measures=measures)
            missing_value = 0
            missing_quantity = 0
        else:
            weights_value, weights_quantity = _resolve_weights_for_origin(
                origin=origin,
                target_year=target_year,
                measures=measures,
                weights_by_year=weights_by_year,
            )
            output, missing_value, missing_quantity = _apply_weights_wide(
                data=data,
                weights_value=weights_value,
                weights_quantity=weights_quantity,
                assume_identity_for_missing=assume_identity_for_missing,
                fail_on_missing=fail_on_missing,
            )

        output_path = output_dir / f"comext_{origin}_wide.parquet"
        output.to_parquet(output_path, index=False)

        return {
            "origin_year": origin,
            "target_year": str(target_year),
            "n_rows_input": n_rows_input,
            "n_rows_output": len(output),
            "n_codes_input": n_codes_input,
            "n_missing_value": missing_value,
            "n_missing_quantity": missing_quantity,
            "sum_value_eur_input": float(data["VALUE_EUR"].sum()),
            "sum_quantity_kg_input": float(data["QUANTITY_KG"].sum()),
            "sum_value_eur_w_value": _sum_or_zero(output, "VALUE_EUR_w_value"),
            "sum_quantity_kg_w_value": _sum_or_zero(output, "QUANTITY_KG_w_value"),
            "sum_value_eur_w_quantity": _sum_or_zero(output, "VALUE_EUR_w_quantity"),
            "sum_quantity_kg_w_quantity": _sum_or_zero(output, "QUANTITY_KG_w_quantity"),
        }

    processed_rows = _run_processing(
        items=years_to_process,
        process_item=lambda item: _process_year(str(item)),
        max_workers=max_workers,
        show_progress=bool(show_progress and years_to_process),
        progress_desc=progress_desc or "Apply annual",
    )
    summary_rows.extend(processed_rows)

    summary = _prepare_summary(rows=summary_rows, key_column="origin_year")
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_summary_path, index=False)
    return summary


def apply_chained_weights_wide_for_month_range(
    *,
    start_year: int,
    end_year: int,
    target_year: int,
    measures: Sequence[str] = ("VALUE_EUR", "QUANTITY_KG"),
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
    revised_codes_by_step: Mapping[tuple[str, str], Iterable[str]] | None = None,
    strict_revised_link_validation: bool = False,
    write_unresolved_details: bool = False,
    skip_existing: bool = True,
    max_workers: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    measures = [str(measure).strip().upper() for measure in measures]
    revised_index = _normalize_revised_index(revised_codes_by_step)
    chained_outputs = _resolve_chained_outputs_for_apply(
        start_year=start_year,
        end_year=end_year,
        target_year=target_year,
        measures=measures,
        annual_base_dir=annual_base_dir,
        weights_dir=weights_dir,
        output_chained_weights_dir=output_chained_weights_dir,
        output_chained_diagnostics_dir=output_chained_diagnostics_dir,
        chained_outputs=chained_outputs,
        finalize_weights=finalize_weights,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
        fail_on_missing=fail_on_missing,
        revised_index=revised_index,
        strict_revised_link_validation=strict_revised_link_validation,
        write_unresolved_details=write_unresolved_details,
    )
    _validate_or_raise_unresolved_for_apply(
        chained_outputs=chained_outputs,
        target_year=target_year,
        measures=measures,
        strict_revised_link_validation=strict_revised_link_validation,
        revised_index=revised_index,
        output_base_dir=output_base_dir,
        write_unresolved_details=write_unresolved_details,
        fail_on_missing=fail_on_missing,
    )

    weights_by_year = _finalize_chained_weights_by_year(
        chained_outputs=chained_outputs,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
    )

    if output_summary_path is None:
        output_summary_path = output_base_dir / f"CN{target_year}" / "monthly" / "summary.csv"
    existing_periods, summary_rows = _load_existing_summary_rows(
        output_summary_path=output_summary_path,
        key_column="origin_period",
        skip_existing=skip_existing,
        zfill_width=6,
    )

    output_dir = output_base_dir / f"CN{target_year}" / "monthly"
    output_dir.mkdir(parents=True, exist_ok=True)

    periods_to_process: list[tuple[str, str, int]] = []
    for year in range(int(start_year), int(end_year) + 1):
        origin = str(year)
        for month in range(1, 13):
            period = f"{origin}{month:02d}"
            output_path = output_dir / f"comext_{period}_wide.parquet"
            if skip_existing and output_path.exists() and period in existing_periods:
                continue
            periods_to_process.append((period, origin, month))

    def _process_period(item: object) -> dict[str, object]:
        period, origin, month = item
        if not isinstance(period, str) or not isinstance(origin, str) or not isinstance(month, int):
            raise ValueError("Invalid period processing item.")
        data_path = monthly_base_dir / f"comext_{period}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing monthly data file: {data_path}")
        data = pd.read_parquet(data_path)
        n_rows_input = len(data)
        n_codes_input = data["PRODUCT_NC"].astype(str).nunique()

        if origin == str(target_year):
            output = _build_identity_wide_output(data, measures=measures)
            missing_value = 0
            missing_quantity = 0
        else:
            weights_value, weights_quantity = _resolve_weights_for_origin(
                origin=origin,
                target_year=target_year,
                measures=measures,
                weights_by_year=weights_by_year,
            )
            output, missing_value, missing_quantity = _apply_weights_wide(
                data=data,
                weights_value=weights_value,
                weights_quantity=weights_quantity,
                assume_identity_for_missing=assume_identity_for_missing,
                fail_on_missing=fail_on_missing,
            )

        output_path = output_dir / f"comext_{period}_wide.parquet"
        output.to_parquet(output_path, index=False)

        return {
            "origin_period": period,
            "origin_year": origin,
            "origin_month": month,
            "target_year": str(target_year),
            "n_rows_input": n_rows_input,
            "n_rows_output": len(output),
            "n_codes_input": n_codes_input,
            "n_missing_value": missing_value,
            "n_missing_quantity": missing_quantity,
            "sum_value_eur_input": float(data["VALUE_EUR"].sum()),
            "sum_quantity_kg_input": float(data["QUANTITY_KG"].sum()),
            "sum_value_eur_w_value": _sum_or_zero(output, "VALUE_EUR_w_value"),
            "sum_quantity_kg_w_value": _sum_or_zero(output, "QUANTITY_KG_w_value"),
            "sum_value_eur_w_quantity": _sum_or_zero(output, "VALUE_EUR_w_quantity"),
            "sum_quantity_kg_w_quantity": _sum_or_zero(output, "QUANTITY_KG_w_quantity"),
        }

    processed_rows = _run_processing(
        items=periods_to_process,
        process_item=_process_period,
        max_workers=max_workers,
        show_progress=bool(show_progress and periods_to_process),
        progress_desc=progress_desc or "Apply monthly",
    )
    summary_rows.extend(processed_rows)

    summary = _prepare_summary(rows=summary_rows, key_column="origin_period", zfill_width=6)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_summary_path, index=False)
    return summary
