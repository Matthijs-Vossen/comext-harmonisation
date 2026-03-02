"""Annual apply implementation for adjacent-period conversions."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .._core.codes import normalize_codes
from .._core.weights_finalize import finalize_weights_table_impl
from .._core.weights_io import read_adjacent_weights
from ..estimation.shares import ANNUAL_DATA_DIR
from ..weights import DEFAULT_WEIGHTS_DIR


WEIGHT_STRATEGIES: Mapping[str, Mapping[str, str]] = {
    "weights_value": {"VALUE_EUR": "VALUE_EUR", "QUANTITY_KG": "VALUE_EUR"},
    "weights_quantity": {"VALUE_EUR": "QUANTITY_KG", "QUANTITY_KG": "QUANTITY_KG"},
    "weights_split": {"VALUE_EUR": "VALUE_EUR", "QUANTITY_KG": "QUANTITY_KG"},
}


def _split_period(period: str) -> tuple[str, str]:
    value = str(period)
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"Invalid period '{period}'; expected 8-digit YYYYYYYY")
    return value[:4], value[4:]


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
    data[code_column] = normalize_codes(data[code_column])

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


def _load_weights(
    *,
    period: str,
    direction: str,
    measure: str,
    weights_dir: Path,
) -> pd.DataFrame:
    return read_adjacent_weights(
        period=period,
        direction=direction,
        measure=measure,
        weights_dir=weights_dir,
        validate=False,
    )


def _append_identity_rows(weights: pd.DataFrame, missing_codes: set[str]) -> pd.DataFrame:
    if not missing_codes:
        return weights
    identity = pd.DataFrame(
        {
            "from_code": list(missing_codes),
            "to_code": list(missing_codes),
            "weight": 1.0,
        }
    )
    return pd.concat([weights, identity], ignore_index=True)


def _finalize_and_prepare_weights(
    *,
    weights: pd.DataFrame,
    data_codes: set[str],
    assume_identity_for_missing: bool,
    fail_on_missing: bool,
    neg_tol: float,
    pos_tol: float,
    row_sum_tol: float,
) -> tuple[pd.DataFrame, set[str]]:
    prepared = finalize_weights_table_impl(
        weights,
        neg_tol=neg_tol,
        pos_tol=pos_tol,
        row_sum_tol=row_sum_tol,
    )
    missing_codes = data_codes - set(prepared["from_code"])
    if missing_codes and assume_identity_for_missing:
        prepared = _append_identity_rows(prepared, missing_codes)
        missing_codes = set()
    if missing_codes and fail_on_missing:
        sample = sorted(list(missing_codes))[:10]
        raise ValueError(f"Missing weights for {len(missing_codes)} codes; sample: {sample}")
    return prepared, missing_codes


def apply_weights_to_annual_period_impl(
    *,
    period: str,
    direction: str = "a_to_b",
    strategy: str = "weights_split",
    annual_base_dir: Path = ANNUAL_DATA_DIR,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    output_base_dir: Path = Path("outputs/apply"),
    measure_columns: Sequence[str] = ("VALUE_EUR", "QUANTITY_KG"),
    fail_on_missing: bool = True,
    assume_identity_for_missing: bool = True,
    finalize_weights: bool = False,
    neg_tol: float = 1e-6,
    pos_tol: float = 1e-10,
    row_sum_tol: float = 1e-9,
) -> dict[str, object]:
    _ = finalize_weights  # kept for compatibility; behavior intentionally finalizes before use
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
    data["PRODUCT_NC"] = normalize_codes(data["PRODUCT_NC"])
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
        )
        weights, missing_codes = _finalize_and_prepare_weights(
            weights=weights,
            data_codes=data_codes,
            assume_identity_for_missing=assume_identity_for_missing,
            fail_on_missing=fail_on_missing,
            neg_tol=neg_tol,
            pos_tol=pos_tol,
            row_sum_tol=row_sum_tol,
        )
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
        )
        weights_quantity = _load_weights(
            period=period,
            direction=direction,
            measure=quantity_weight_measure,
            weights_dir=weights_dir,
        )
        weights_value, missing_value = _finalize_and_prepare_weights(
            weights=weights_value,
            data_codes=data_codes,
            assume_identity_for_missing=assume_identity_for_missing,
            fail_on_missing=False,
            neg_tol=neg_tol,
            pos_tol=pos_tol,
            row_sum_tol=row_sum_tol,
        )
        weights_quantity, missing_quantity = _finalize_and_prepare_weights(
            weights=weights_quantity,
            data_codes=data_codes,
            assume_identity_for_missing=assume_identity_for_missing,
            fail_on_missing=False,
            neg_tol=neg_tol,
            pos_tol=pos_tol,
            row_sum_tol=row_sum_tol,
        )
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
        n_codes_weighted = max(
            len(set(weights_value["from_code"])),
            len(set(weights_quantity["from_code"])),
        )

    output_dir = output_base_dir / f"CN{target_year}" / "annual"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"comext_{origin_year}_{strategy}.parquet"
    converted.to_parquet(output_path, index=False)

    return {
        "period": period,
        "direction": direction,
        "strategy": strategy,
        "origin_year": origin_year,
        "target_year": target_year,
        "n_rows_input": n_rows_input,
        "n_rows_output": len(converted),
        "n_codes_input": len(data_codes),
        "n_codes_weighted": n_codes_weighted,
        "n_codes_missing": 0 if fail_on_missing else len(missing_codes),
    }
