"""Apply estimated conversion weights to trade data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .estimation.shares import ANNUAL_DATA_DIR
from .estimation.chaining import (
    build_chained_weights_for_range,
    ChainedWeightsOutput,
    DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    DEFAULT_CHAINED_WEIGHTS_DIR,
)
from .weights import DEFAULT_WEIGHTS_DIR, validate_weight_table


MEASURE_COLUMNS = ("VALUE_EUR", "QUANTITY_KG")
MONTHLY_DATA_DIR = Path("data/extracted_no_confidential/products_like")

WEIGHT_STRATEGIES: Mapping[str, Mapping[str, str]] = {
    "weights_value": {"VALUE_EUR": "VALUE_EUR", "QUANTITY_KG": "VALUE_EUR"},
    "weights_quantity": {"VALUE_EUR": "QUANTITY_KG", "QUANTITY_KG": "QUANTITY_KG"},
    "weights_split": {"VALUE_EUR": "VALUE_EUR", "QUANTITY_KG": "QUANTITY_KG"},
}


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
    threshold_abs: float = 1e-3,
    row_sum_tol: float = 1e-9,
) -> pd.DataFrame:
    """Clamp small weights and renormalize rows by from_code."""
    if threshold_abs < 0:
        raise ValueError("threshold_abs must be non-negative")

    required = {"from_code", "to_code", "weight"}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = weights.copy()
    df["from_code"] = _normalize_codes(df["from_code"])
    df["to_code"] = _normalize_codes(df["to_code"])
    df["weight"] = df["weight"].astype(float)

    too_negative = df["weight"] < -threshold_abs
    if too_negative.any():
        sample = df.loc[too_negative, ["from_code", "to_code", "weight"]].head(5).to_dict("records")
        raise ValueError(f"Found weights below -threshold_abs: {sample}")

    df.loc[df["weight"].abs() < threshold_abs, "weight"] = 0.0

    row_sums = (
        df.groupby("from_code", as_index=False, sort=False)["weight"]
        .sum()
        .rename(columns={"weight": "row_sum"})
    )
    zero_rows = row_sums[row_sums["row_sum"] <= 0]
    if not zero_rows.empty:
        sample = zero_rows["from_code"].head(5).tolist()
        raise ValueError(f"Rows with zero weight after thresholding: {sample}")

    df = df.merge(row_sums, on="from_code", how="left")
    df["weight"] = df["weight"] / df["row_sum"]
    df = df.drop(columns=["row_sum"])

    row_sums = df.groupby("from_code", sort=False)["weight"].sum()
    max_dev = float((row_sums - 1.0).abs().max()) if not row_sums.empty else 0.0
    if max_dev > row_sum_tol:
        raise ValueError(f"Row sums deviate from 1 by {max_dev} (tolerance {row_sum_tol})")

    df = df[df["weight"] > 0].reset_index(drop=True)
    validate_weight_table(df, schema="minimal", check_bounds=True, check_row_sums=True)
    return df


def _split_period(period: str) -> tuple[str, str]:
    period = str(period)
    if len(period) != 8 or not period.isdigit():
        raise ValueError(f"Invalid period '{period}'; expected 8-digit YYYYYYYY")
    return period[:4], period[4:]


def _normalize_codes(series: pd.Series) -> pd.Series:
    codes = series.astype(str).str.strip().str.replace(" ", "", regex=False)
    mask = codes.str.isdigit()
    codes = codes.where(~mask, codes.str.zfill(8))
    return codes


def _load_weights(
    *,
    period: str,
    direction: str,
    measure: str,
    weights_dir: Path,
    validate: bool = True,
) -> pd.DataFrame:
    measure_tag = measure.lower()
    weights_path = weights_dir / period / direction / measure_tag
    ambiguous_path = weights_path / "weights_ambiguous.csv"
    deterministic_path = weights_path / "weights_deterministic.csv"
    if not ambiguous_path.exists():
        raise FileNotFoundError(f"Missing weights file: {ambiguous_path}")
    if not deterministic_path.exists():
        raise FileNotFoundError(f"Missing weights file: {deterministic_path}")
    ambiguous = pd.read_csv(ambiguous_path)
    deterministic = pd.read_csv(deterministic_path)
    if not deterministic.empty and not ambiguous.empty:
        deterministic = deterministic.loc[
            ~deterministic["from_code"].isin(ambiguous["from_code"])
        ]
    frames = []
    for frame in (ambiguous, deterministic):
        if frame.empty:
            continue
        if frame.isna().all().all():
            continue
        frames.append(frame)
    if not frames:
        raise ValueError(f"No weights found for period {period} ({measure}).")
    weights = frames[0].copy() if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    if validate:
        validate_weight_table(weights, check_bounds=True, check_row_sums=True)
    weights = weights[["from_code", "to_code", "weight"]].copy()
    weights["from_code"] = _normalize_codes(weights["from_code"])
    weights["to_code"] = _normalize_codes(weights["to_code"])
    return weights


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
    threshold_abs: float = 1e-3,
    row_sum_tol: float = 1e-6,
    assume_identity_for_missing: bool = True,
    fail_on_missing: bool = True,
    skip_existing: bool = True,
    max_workers: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    measures = [str(measure).strip().upper() for measure in measures]
    if chained_outputs is None:
        chained_outputs = build_chained_weights_for_range(
            start_year=start_year,
            end_year=end_year,
            target_year=target_year,
            measures=measures,
            weights_dir=weights_dir,
            output_weights_dir=output_chained_weights_dir,
            output_diagnostics_dir=output_chained_diagnostics_dir,
            finalize_weights=finalize_weights,
            threshold_abs=threshold_abs,
            row_sum_tol=row_sum_tol,
            fail_on_missing=fail_on_missing,
        )

    weights_by_year: dict[str, dict[str, pd.DataFrame]] = {}
    for output in chained_outputs:
        weights_by_year.setdefault(output.origin_year, {})[output.measure] = output.weights

    existing_summary = pd.DataFrame()
    if output_summary_path is None:
        output_summary_path = output_base_dir / f"CN{target_year}" / "summary.csv"
    if skip_existing and output_summary_path.exists():
        existing_summary = pd.read_csv(output_summary_path)
        if "origin_year" in existing_summary.columns:
            existing_summary["origin_year"] = existing_summary["origin_year"].astype(str)
    existing_years = set(existing_summary.get("origin_year", []))
    summary_rows: list[dict[str, object]] = (
        existing_summary.to_dict("records") if not existing_summary.empty else []
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

        weights_value = None
        weights_quantity = None
        if origin != str(target_year):
            weights_for_year = weights_by_year.get(origin, {})
            weights_value = weights_for_year.get("VALUE_EUR")
            weights_quantity = weights_for_year.get("QUANTITY_KG")
            if "VALUE_EUR" in measures and weights_value is None:
                raise ValueError(f"Missing chained VALUE_EUR weights for {origin}->{target_year}")
            if "QUANTITY_KG" in measures and weights_quantity is None:
                raise ValueError(f"Missing chained QUANTITY_KG weights for {origin}->{target_year}")
        else:
            weights_value = None if "VALUE_EUR" not in measures else pd.DataFrame(
                columns=["from_code", "to_code", "weight"]
            )
            weights_quantity = None if "QUANTITY_KG" not in measures else pd.DataFrame(
                columns=["from_code", "to_code", "weight"]
            )

        if origin == str(target_year):
            output = data.copy()
            output["PRODUCT_NC"] = _normalize_codes(output["PRODUCT_NC"])
            if "VALUE_EUR" in measures:
                output["VALUE_EUR_w_value"] = output["VALUE_EUR"]
                output["QUANTITY_KG_w_value"] = output["QUANTITY_KG"]
            if "QUANTITY_KG" in measures:
                output["VALUE_EUR_w_quantity"] = output["VALUE_EUR"]
                output["QUANTITY_KG_w_quantity"] = output["QUANTITY_KG"]
            drop_cols = ["VALUE_EUR", "QUANTITY_KG"]
            output = output.drop(columns=[col for col in drop_cols if col in output.columns])
            missing_value = 0
            missing_quantity = 0
        else:
            output, missing_value, missing_quantity = _apply_weights_wide(
                data=data,
                weights_value=weights_value if "VALUE_EUR" in measures else None,
                weights_quantity=weights_quantity if "QUANTITY_KG" in measures else None,
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
            "sum_value_eur_w_value": float(output.get("VALUE_EUR_w_value", 0.0).sum()),
            "sum_quantity_kg_w_value": float(output.get("QUANTITY_KG_w_value", 0.0).sum()),
            "sum_value_eur_w_quantity": float(
                output.get("VALUE_EUR_w_quantity", 0.0).sum()
            ),
            "sum_quantity_kg_w_quantity": float(
                output.get("QUANTITY_KG_w_quantity", 0.0).sum()
            ),
        }

    show_progress = bool(show_progress and years_to_process)

    if max_workers is None or max_workers <= 1 or len(years_to_process) <= 1:
        iterator = years_to_process
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(
                years_to_process,
                desc=progress_desc or "Apply annual",
                total=len(years_to_process),
            )
        for origin in iterator:
            summary_rows.append(_process_year(origin))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        progress = None
        if show_progress:
            from tqdm import tqdm

            progress = tqdm(
                total=len(years_to_process),
                desc=progress_desc or "Apply annual",
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_year, origin): origin for origin in years_to_process}
            for future in as_completed(futures):
                summary_rows.append(future.result())
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()

    summary = pd.DataFrame(summary_rows)
    if "origin_year" in summary.columns:
        summary["origin_year"] = summary["origin_year"].astype(str)
        summary = summary.drop_duplicates(subset=["origin_year"], keep="last")
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
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    output_chained_weights_dir: Path = DEFAULT_CHAINED_WEIGHTS_DIR,
    output_chained_diagnostics_dir: Path = DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    output_base_dir: Path = Path("outputs/apply"),
    output_summary_path: Path | None = None,
    chained_outputs: Sequence[ChainedWeightsOutput] | None = None,
    finalize_weights: bool = False,
    threshold_abs: float = 1e-3,
    row_sum_tol: float = 1e-6,
    assume_identity_for_missing: bool = True,
    fail_on_missing: bool = True,
    skip_existing: bool = True,
    max_workers: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame:
    measures = [str(measure).strip().upper() for measure in measures]
    if chained_outputs is None:
        chained_outputs = build_chained_weights_for_range(
            start_year=start_year,
            end_year=end_year,
            target_year=target_year,
            measures=measures,
            weights_dir=weights_dir,
            output_weights_dir=output_chained_weights_dir,
            output_diagnostics_dir=output_chained_diagnostics_dir,
            finalize_weights=finalize_weights,
            threshold_abs=threshold_abs,
            row_sum_tol=row_sum_tol,
            fail_on_missing=fail_on_missing,
        )

    weights_by_year: dict[str, dict[str, pd.DataFrame]] = {}
    for output in chained_outputs:
        weights_by_year.setdefault(output.origin_year, {})[output.measure] = output.weights

    existing_summary = pd.DataFrame()
    if output_summary_path is None:
        output_summary_path = output_base_dir / f"CN{target_year}" / "monthly" / "summary.csv"
    if skip_existing and output_summary_path.exists():
        existing_summary = pd.read_csv(output_summary_path)
        if "origin_period" in existing_summary.columns:
            existing_summary["origin_period"] = (
                existing_summary["origin_period"].astype(str).str.zfill(6)
            )
    existing_periods = set(existing_summary.get("origin_period", []))
    summary_rows: list[dict[str, object]] = (
        existing_summary.to_dict("records") if not existing_summary.empty else []
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

    def _process_period(period: str, origin: str, month: int) -> dict[str, object]:
        data_path = monthly_base_dir / f"comext_{period}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing monthly data file: {data_path}")
        data = pd.read_parquet(data_path)
        n_rows_input = len(data)
        n_codes_input = data["PRODUCT_NC"].astype(str).nunique()

        weights_value = None
        weights_quantity = None
        if origin != str(target_year):
            weights_for_year = weights_by_year.get(origin, {})
            weights_value = weights_for_year.get("VALUE_EUR")
            weights_quantity = weights_for_year.get("QUANTITY_KG")
            if "VALUE_EUR" in measures and weights_value is None:
                raise ValueError(f"Missing chained VALUE_EUR weights for {origin}->{target_year}")
            if "QUANTITY_KG" in measures and weights_quantity is None:
                raise ValueError(
                    f"Missing chained QUANTITY_KG weights for {origin}->{target_year}"
                )
        else:
            weights_value = None if "VALUE_EUR" not in measures else pd.DataFrame(
                columns=["from_code", "to_code", "weight"]
            )
            weights_quantity = None if "QUANTITY_KG" not in measures else pd.DataFrame(
                columns=["from_code", "to_code", "weight"]
            )

        if origin == str(target_year):
            output = data.copy()
            output["PRODUCT_NC"] = _normalize_codes(output["PRODUCT_NC"])
            if "VALUE_EUR" in measures:
                output["VALUE_EUR_w_value"] = output["VALUE_EUR"]
                output["QUANTITY_KG_w_value"] = output["QUANTITY_KG"]
            if "QUANTITY_KG" in measures:
                output["VALUE_EUR_w_quantity"] = output["VALUE_EUR"]
                output["QUANTITY_KG_w_quantity"] = output["QUANTITY_KG"]
            drop_cols = ["VALUE_EUR", "QUANTITY_KG"]
            output = output.drop(columns=[col for col in drop_cols if col in output.columns])
            missing_value = 0
            missing_quantity = 0
        else:
            output, missing_value, missing_quantity = _apply_weights_wide(
                data=data,
                weights_value=weights_value if "VALUE_EUR" in measures else None,
                weights_quantity=weights_quantity if "QUANTITY_KG" in measures else None,
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
            "sum_value_eur_w_value": float(output.get("VALUE_EUR_w_value", 0.0).sum()),
            "sum_quantity_kg_w_value": float(output.get("QUANTITY_KG_w_value", 0.0).sum()),
            "sum_value_eur_w_quantity": float(
                output.get("VALUE_EUR_w_quantity", 0.0).sum()
            ),
            "sum_quantity_kg_w_quantity": float(
                output.get("QUANTITY_KG_w_quantity", 0.0).sum()
            ),
        }

    show_progress = bool(show_progress and periods_to_process)

    if max_workers is None or max_workers <= 1 or len(periods_to_process) <= 1:
        iterator = periods_to_process
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(
                periods_to_process,
                desc=progress_desc or "Apply monthly",
                total=len(periods_to_process),
            )
        for period, origin, month in iterator:
            summary_rows.append(_process_period(period, origin, month))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        progress = None
        if show_progress:
            from tqdm import tqdm

            progress = tqdm(
                total=len(periods_to_process),
                desc=progress_desc or "Apply monthly",
            )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_period, period, origin, month): period
                for period, origin, month in periods_to_process
            }
            for future in as_completed(futures):
                summary_rows.append(future.result())
                if progress is not None:
                    progress.update(1)
        if progress is not None:
            progress.close()

    summary = pd.DataFrame(summary_rows)
    if "origin_period" in summary.columns:
        summary["origin_period"] = summary["origin_period"].astype(str).str.zfill(6)
        summary = summary.drop_duplicates(subset=["origin_period"], keep="last")
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_summary_path, index=False)
    return summary


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

    df = df.copy()
    df[code_column] = _normalize_codes(df[code_column])

    weight_codes = set(weights["from_code"])
    data_codes = set(df[code_column].unique())
    missing_codes = data_codes - weight_codes
    if missing_codes and fail_on_missing:
        sample = sorted(list(missing_codes))[:10]
        raise ValueError(f"Missing weights for {len(missing_codes)} codes; sample: {sample}")

    merged = df.merge(weights, left_on=code_column, right_on="from_code", how="inner")
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
    threshold_abs: float = 1e-3,
    row_sum_tol: float = 1e-9,
) -> ApplyDiagnostics:
    """Apply estimated weights to annual data for a concordance period."""
    if strategy not in WEIGHT_STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    vintage_a_year, vintage_b_year = _split_period(period)
    if direction == "a_to_b":
        origin_year = vintage_a_year
        target_year = vintage_b_year
    elif direction == "b_to_a":
        origin_year = vintage_b_year
        target_year = vintage_a_year
    else:
        raise ValueError("direction must be 'a_to_b' or 'b_to_a'")

    data_path = annual_base_dir / f"comext_{origin_year}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing annual data file: {data_path}")

    data = pd.read_parquet(data_path)
    n_rows_input = len(data)
    data["PRODUCT_NC"] = _normalize_codes(data["PRODUCT_NC"])
    data_codes = set(data["PRODUCT_NC"].unique())

    strategy_map = WEIGHT_STRATEGIES[strategy]
    value_weight_measure = strategy_map["VALUE_EUR"]
    quantity_weight_measure = strategy_map["QUANTITY_KG"]

    missing_codes: set[str] = set()

    if value_weight_measure == quantity_weight_measure:
        weights = _load_weights(
            period=period,
            direction=direction,
            measure=value_weight_measure,
            weights_dir=weights_dir,
            validate=not finalize_weights,
        )
        if finalize_weights:
            weights = finalize_weights_table(
                weights, threshold_abs=threshold_abs, row_sum_tol=row_sum_tol
            )
        missing_codes = data_codes - set(weights["from_code"])
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
        converted = _apply_weights_to_frame(
            data,
            weights=weights,
            measure_columns=measure_columns,
            fail_on_missing=False,
        )
        n_codes_weighted = len(set(weights["from_code"]))
    else:
        weights_value = _load_weights(
            period=period,
            direction=direction,
            measure=value_weight_measure,
            weights_dir=weights_dir,
            validate=not finalize_weights,
        )
        weights_quantity = _load_weights(
            period=period,
            direction=direction,
            measure=quantity_weight_measure,
            weights_dir=weights_dir,
            validate=not finalize_weights,
        )
        if finalize_weights:
            weights_value = finalize_weights_table(
                weights_value, threshold_abs=threshold_abs, row_sum_tol=row_sum_tol
            )
            weights_quantity = finalize_weights_table(
                weights_quantity, threshold_abs=threshold_abs, row_sum_tol=row_sum_tol
            )
        missing_value = data_codes - set(weights_value["from_code"])
        missing_quantity = data_codes - set(weights_quantity["from_code"])
        if assume_identity_for_missing:
            if missing_value:
                identity = pd.DataFrame(
                    {
                        "from_code": list(missing_value),
                        "to_code": list(missing_value),
                        "weight": 1.0,
                    }
                )
                weights_value = pd.concat([weights_value, identity], ignore_index=True)
                missing_value = set()
            if missing_quantity:
                identity = pd.DataFrame(
                    {
                        "from_code": list(missing_quantity),
                        "to_code": list(missing_quantity),
                        "weight": 1.0,
                    }
                )
                weights_quantity = pd.concat([weights_quantity, identity], ignore_index=True)
                missing_quantity = set()
        missing_codes = missing_value | missing_quantity
        if missing_codes and fail_on_missing:
            sample = sorted(list(missing_codes))[:10]
            raise ValueError(f"Missing weights for {len(missing_codes)} codes; sample: {sample}")
        id_cols = [col for col in data.columns if col not in measure_columns]
        converted_value = _apply_weights_to_frame(
            data,
            weights=weights_value,
            measure_columns=["VALUE_EUR"],
            fail_on_missing=False,
            id_columns=id_cols,
        )
        converted_quantity = _apply_weights_to_frame(
            data,
            weights=weights_quantity,
            measure_columns=["QUANTITY_KG"],
            fail_on_missing=False,
            id_columns=id_cols,
        )
        converted = converted_value.merge(converted_quantity, on=id_cols, how="outer")
        for col in measure_columns:
            if col in converted.columns:
                converted[col] = converted[col].fillna(0.0)
        n_codes_weighted = max(len(set(weights_value["from_code"])), len(set(weights_quantity["from_code"])))

    output_dir = output_base_dir / f"CN{target_year}" / "annual"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"comext_{origin_year}_{strategy}.parquet"
    converted.to_parquet(output_path, index=False)

    diagnostics = ApplyDiagnostics(
        period=period,
        direction=direction,
        strategy=strategy,
        origin_year=origin_year,
        target_year=target_year,
        n_rows_input=n_rows_input,
        n_rows_output=len(converted),
        n_codes_input=len(data_codes),
        n_codes_weighted=n_codes_weighted,
        n_codes_missing=0 if fail_on_missing else len(missing_codes),
    )
    return diagnostics
