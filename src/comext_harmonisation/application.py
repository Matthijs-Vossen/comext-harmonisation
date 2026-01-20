"""Apply estimated conversion weights to trade data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .estimation.shares import ANNUAL_DATA_DIR
from .weights import DEFAULT_WEIGHTS_DIR, validate_weight_table


MEASURE_COLUMNS = ("VALUE_EUR", "QUANTITY_KG")

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

    row_sums = df.groupby("from_code", as_index=False)["weight"].sum().rename(columns={"weight": "row_sum"})
    zero_rows = row_sums[row_sums["row_sum"] <= 0]
    if not zero_rows.empty:
        sample = zero_rows["from_code"].head(5).tolist()
        raise ValueError(f"Rows with zero weight after thresholding: {sample}")

    df = df.merge(row_sums, on="from_code", how="left")
    df["weight"] = df["weight"] / df["row_sum"]
    df = df.drop(columns=["row_sum"])

    row_sums = df.groupby("from_code")["weight"].sum()
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
    ambiguous_path = weights_dir / f"weights_ambiguous_{period}_{direction}_{measure_tag}.csv"
    deterministic_path = weights_dir / f"weights_deterministic_{period}_{direction}_{measure_tag}.csv"
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
    if deterministic.empty:
        weights = ambiguous.copy()
    else:
        weights = pd.concat([ambiguous, deterministic], ignore_index=True)
    if weights.empty:
        raise ValueError(f"No weights found for period {period} ({measure}).")
    if validate:
        validate_weight_table(weights, check_bounds=True, check_row_sums=True)
    weights = weights[["from_code", "to_code", "weight"]].copy()
    weights["from_code"] = _normalize_codes(weights["from_code"])
    weights["to_code"] = _normalize_codes(weights["to_code"])
    return weights


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
    result = merged.groupby(list(id_columns), as_index=False)[list(measure_columns)].sum()
    return result


def apply_weights_to_annual_period(
    *,
    period: str,
    direction: str = "a_to_b",
    strategy: str = "weights_split",
    annual_base_dir: Path = ANNUAL_DATA_DIR,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    output_base_dir: Path = Path("outputs/harmonised/annual"),
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

    output_dir = output_base_dir / f"CN{target_year}"
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
