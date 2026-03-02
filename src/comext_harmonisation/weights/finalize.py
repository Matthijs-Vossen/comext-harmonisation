"""Shared weight finalization logic used before applying conversions."""

from __future__ import annotations

import pandas as pd

from .schema import validate_weight_table
from ..core.codes import normalize_codes


def finalize_weights_table_impl(
    weights: pd.DataFrame,
    *,
    neg_tol: float = 1e-6,
    pos_tol: float = 1e-10,
    row_sum_tol: float = 1e-9,
) -> pd.DataFrame:
    if neg_tol < 0 or pos_tol < 0:
        raise ValueError("neg_tol and pos_tol must be non-negative")

    required = {"from_code", "to_code", "weight"}
    missing = required.difference(weights.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = weights.copy()
    df["from_code"] = normalize_codes(df["from_code"])
    df["to_code"] = normalize_codes(df["to_code"])
    df["weight"] = df["weight"].astype(float)

    too_negative = df["weight"] < -neg_tol
    if too_negative.any():
        sample = df.loc[too_negative, ["from_code", "to_code", "weight"]].head(5).to_dict("records")
        raise ValueError(f"Found weights below -neg_tol: {sample}")

    if neg_tol > 0:
        df.loc[(df["weight"] < 0) & (df["weight"] >= -neg_tol), "weight"] = 0.0
    if pos_tol > 0:
        df.loc[(df["weight"] > 0) & (df["weight"] < pos_tol), "weight"] = 0.0

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
