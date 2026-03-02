"""Shared operations for chained weight composition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse as sp

from .codes import normalize_code_set


def max_row_sum_dev(weights: pd.DataFrame) -> float:
    if weights.empty:
        return 0.0
    row_sums = weights.groupby("from_code", sort=False)["weight"].sum()
    return float((row_sums - 1.0).abs().max())


def compose_weights(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    revised_mid_codes: set[str] | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    if left.empty:
        return pd.DataFrame(columns=["from_code", "to_code", "weight"]), set()

    left_to = set(left["to_code"])
    right_from = set(right["from_code"]) if not right.empty else set()
    common_mid = sorted(left_to & right_from)
    missing_mid = sorted(left_to - right_from)
    unresolved_revised_mid: set[str] = set()

    chained_parts: list[pd.DataFrame] = []

    if common_mid:
        from_codes = sorted(left["from_code"].unique())
        to_codes = sorted(right["to_code"].unique())
        mid_index = {code: idx for idx, code in enumerate(common_mid)}
        from_index = {code: idx for idx, code in enumerate(from_codes)}
        to_index = {code: idx for idx, code in enumerate(to_codes)}

        left_filtered = left[left["to_code"].isin(common_mid)]
        right_filtered = right[right["from_code"].isin(common_mid)]

        left_rows = left_filtered["from_code"].map(from_index).to_numpy(dtype=int)
        left_cols = left_filtered["to_code"].map(mid_index).to_numpy(dtype=int)
        left_data = left_filtered["weight"].to_numpy(dtype=float)
        left_mat = sp.coo_matrix(
            (left_data, (left_rows, left_cols)),
            shape=(len(from_codes), len(common_mid)),
        ).tocsr()

        right_rows = right_filtered["from_code"].map(mid_index).to_numpy(dtype=int)
        right_cols = right_filtered["to_code"].map(to_index).to_numpy(dtype=int)
        right_data = right_filtered["weight"].to_numpy(dtype=float)
        right_mat = sp.coo_matrix(
            (right_data, (right_rows, right_cols)),
            shape=(len(common_mid), len(to_codes)),
        ).tocsr()

        chained = left_mat @ right_mat
        if chained.nnz:
            chained = chained.tocoo()
            chained_parts.append(
                pd.DataFrame(
                    {
                        "from_code": np.take(from_codes, chained.row),
                        "to_code": np.take(to_codes, chained.col),
                        "weight": chained.data,
                    }
                )
            )

    if missing_mid:
        carry_mid = missing_mid
        if revised_mid_codes is not None:
            unresolved_revised_mid = set(missing_mid) & set(revised_mid_codes)
            carry_mid = sorted(set(missing_mid) - unresolved_revised_mid)
        if carry_mid:
            carry = left[left["to_code"].isin(carry_mid)]
            carry = carry.groupby(["from_code", "to_code"], as_index=False)["weight"].sum()
            chained_parts.append(carry)

    if not chained_parts:
        return (
            pd.DataFrame(columns=["from_code", "to_code", "weight"]),
            unresolved_revised_mid,
        )

    combined = pd.concat(chained_parts, ignore_index=True)
    return (
        combined.groupby(["from_code", "to_code"], as_index=False)["weight"].sum(),
        unresolved_revised_mid,
    )


def inject_step_identity_strict(
    *,
    step_weights: pd.DataFrame,
    universe_codes: set[str],
    revised_from_codes: set[str] | None,
) -> tuple[pd.DataFrame, set[str]]:
    if step_weights.empty:
        step = pd.DataFrame(columns=["from_code", "to_code", "weight"])
    else:
        step = step_weights[["from_code", "to_code", "weight"]].copy()
    missing = normalize_code_set(universe_codes) - set(step["from_code"])
    if not missing:
        return step, set()

    unresolved_revised = set()
    inject_codes = set(missing)
    if revised_from_codes is not None:
        unresolved_revised = set(missing) & set(revised_from_codes)
        inject_codes = set(missing) - unresolved_revised

    if inject_codes:
        identity = pd.DataFrame(
            {
                "from_code": sorted(inject_codes),
                "to_code": sorted(inject_codes),
                "weight": 1.0,
            }
        )
        step = pd.concat([step, identity], ignore_index=True)

    return step, unresolved_revised


def check_weight_bounds(weights: pd.DataFrame, *, bound_tol: float, context: str) -> None:
    min_weight = float(weights["weight"].min())
    max_weight = float(weights["weight"].max())
    if min_weight < -bound_tol or max_weight > 1.0 + bound_tol:
        raise ValueError(
            "Weights outside [0, 1] tolerance in "
            f"{context}: min={min_weight}, max={max_weight}, tol={bound_tol}"
        )

