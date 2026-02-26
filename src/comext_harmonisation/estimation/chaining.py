"""Chain adjacent CN conversion weights across multiple vintages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse as sp
from ..groups import build_concordance_groups
from ..weights import DEFAULT_WEIGHTS_DIR, validate_weight_table


DEFAULT_CHAINED_WEIGHTS_DIR = Path("outputs/chain")
DEFAULT_CHAINED_DIAGNOSTICS_DIR = Path("outputs/chain")
DEFAULT_ANNUAL_DATA_DIR = Path("data/extracted_annual_no_confidential/products_like")


@dataclass(frozen=True)
class ChainedWeightsOutput:
    origin_year: str
    target_year: str
    direction: str
    measure: str
    weights: pd.DataFrame
    diagnostics: pd.DataFrame
    weights_path: Path
    diagnostics_path: Path


class _UnresolvedRevisedLinksError(ValueError):
    def __init__(self, message: str, unresolved_rows: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.unresolved_rows = unresolved_rows


def _normalize_year(year: str | int) -> str:
    year_str = str(year)
    if len(year_str) != 4 or not year_str.isdigit():
        raise ValueError(f"Invalid year '{year}'; expected 4-digit year")
    return year_str


def _normalize_codes(series: pd.Series) -> pd.Series:
    codes = series.astype(str).str.strip().str.replace(" ", "", regex=False)
    mask = codes.str.isdigit()
    codes = codes.where(~mask, codes.str.zfill(8))
    return codes


def build_code_universe_from_annual(
    *, annual_base_dir: Path, years: Sequence[int]
) -> dict[int, set[str]]:
    universe: dict[int, set[str]] = {}
    for year in years:
        data_path = annual_base_dir / f"comext_{year}.parquet"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing annual data file: {data_path}")
        data = pd.read_parquet(data_path, columns=["PRODUCT_NC"])
        codes = _normalize_codes(data["PRODUCT_NC"]).dropna().unique().tolist()
        universe[int(year)] = set(codes)
    return universe


def _normalize_code_set(codes: set[str]) -> set[str]:
    if not codes:
        return set()
    series = pd.Series(list(codes))
    normalized = _normalize_codes(series)
    return set(normalized.tolist())


def _normalize_revised_index(
    revised_codes_by_step: Mapping[tuple[str, str], Iterable[str]] | None,
) -> dict[tuple[str, str], set[str]]:
    if revised_codes_by_step is None:
        return {}
    normalized: dict[tuple[str, str], set[str]] = {}
    for key, codes in revised_codes_by_step.items():
        if len(key) != 2:
            raise ValueError(
                "revised_codes_by_step keys must be (period, direction) tuples"
            )
        period, direction = key
        if direction not in {"a_to_b", "b_to_a"}:
            raise ValueError(
                "revised_codes_by_step direction must be 'a_to_b' or 'b_to_a'"
            )
        normalized[(str(period), direction)] = _normalize_code_set(set(codes))
    return normalized


def build_revised_code_index_from_concordance(
    concordance_edges: pd.DataFrame,
) -> dict[tuple[str, str], set[str]]:
    required = {"period", "vintage_a_code", "vintage_b_code"}
    missing = required.difference(concordance_edges.columns)
    if missing:
        raise ValueError(
            "Concordance dataframe missing required columns for revised-link index: "
            f"{sorted(missing)}"
        )

    edges = concordance_edges.copy()
    edges["period"] = edges["period"].astype(str).str.strip()
    edges["vintage_a_code"] = _normalize_codes(edges["vintage_a_code"])
    edges["vintage_b_code"] = _normalize_codes(edges["vintage_b_code"])

    if "vintage_a_year" not in edges.columns:
        edges["vintage_a_year"] = edges["period"].str[:4]
    if "vintage_b_year" not in edges.columns:
        edges["vintage_b_year"] = edges["period"].str[4:]

    edges = edges[
        ["period", "vintage_a_year", "vintage_b_year", "vintage_a_code", "vintage_b_code"]
    ].drop_duplicates()
    if edges.empty:
        return {}

    revised: dict[tuple[str, str], set[str]] = {}
    a_counts = (
        edges.groupby(["period", "vintage_a_code"], as_index=False)["vintage_b_code"]
        .nunique()
        .rename(columns={"vintage_b_code": "n_to"})
    )
    for row in a_counts.loc[a_counts["n_to"] > 1, ["period", "vintage_a_code"]].itertuples(
        index=False
    ):
        revised.setdefault((row.period, "a_to_b"), set()).add(row.vintage_a_code)

    b_counts = (
        edges.groupby(["period", "vintage_b_code"], as_index=False)["vintage_a_code"]
        .nunique()
        .rename(columns={"vintage_a_code": "n_to"})
    )
    for row in b_counts.loc[b_counts["n_to"] > 1, ["period", "vintage_b_code"]].itertuples(
        index=False
    ):
        revised.setdefault((row.period, "b_to_a"), set()).add(row.vintage_b_code)

    groups = build_concordance_groups(edges)
    ambiguous_a_groups = groups.group_summary.loc[
        groups.group_summary["a_to_b_ambiguous"], ["period", "group_id"]
    ]
    if not ambiguous_a_groups.empty:
        ambiguous_a_codes = (
            groups.edges.merge(ambiguous_a_groups, on=["period", "group_id"], how="inner")[
                ["period", "vintage_a_code"]
            ]
            .drop_duplicates()
            .itertuples(index=False)
        )
        for row in ambiguous_a_codes:
            revised.setdefault((row.period, "a_to_b"), set()).add(row.vintage_a_code)

    ambiguous_b_groups = groups.group_summary.loc[
        groups.group_summary["b_to_a_ambiguous"], ["period", "group_id"]
    ]
    if not ambiguous_b_groups.empty:
        ambiguous_b_codes = (
            groups.edges.merge(ambiguous_b_groups, on=["period", "group_id"], how="inner")[
                ["period", "vintage_b_code"]
            ]
            .drop_duplicates()
            .itertuples(index=False)
        )
        for row in ambiguous_b_codes:
            revised.setdefault((row.period, "b_to_a"), set()).add(row.vintage_b_code)

    return revised


def _unresolved_rows(
    *,
    origin_year: str,
    target_year: str,
    direction: str,
    measure: str,
    period: str,
    step_index: int,
    codes: set[str],
    reason: str,
) -> list[dict[str, object]]:
    return [
        {
            "origin_year": str(origin_year),
            "target_year": str(target_year),
            "direction": direction,
            "measure": measure,
            "period": period,
            "step_index": int(step_index),
            "code": code,
            "reason": reason,
        }
        for code in sorted(codes)
    ]


def _revised_codes_for_step(
    *,
    period: str,
    direction: str,
    strict_revised_link_validation: bool,
    revised_codes_by_step: Mapping[tuple[str, str], set[str]],
) -> set[str] | None:
    if not strict_revised_link_validation:
        return None
    return set(revised_codes_by_step.get((period, direction), set()))


def _chain_periods(origin_year: str, target_year: str) -> tuple[list[str], str]:
    origin = int(origin_year)
    target = int(target_year)
    if origin == target:
        return [], "identity"
    if origin < target:
        periods = [f"{year}{year + 1}" for year in range(origin, target)]
        return periods, "a_to_b"
    periods = [f"{year}{year + 1}" for year in range(origin - 1, target - 1, -1)]
    return periods, "b_to_a"


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
    weights = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    weights = weights[["from_code", "to_code", "weight"]].copy()
    weights["from_code"] = _normalize_codes(weights["from_code"])
    weights["to_code"] = _normalize_codes(weights["to_code"])
    weights["weight"] = weights["weight"].astype(float)
    if validate:
        validate_weight_table(weights, schema="minimal", check_bounds=True, check_row_sums=True)
    return weights


def _max_row_sum_dev(weights: pd.DataFrame) -> float:
    if weights.empty:
        return 0.0
    row_sums = weights.groupby("from_code", sort=False)["weight"].sum()
    return float((row_sums - 1.0).abs().max())


def _compose_weights(
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


def _inject_step_identity(
    step_weights: pd.DataFrame,
    *,
    universe_codes: set[str],
) -> tuple[pd.DataFrame, set[str]]:
    return _inject_step_identity_strict(
        step_weights=step_weights,
        universe_codes=universe_codes,
        revised_from_codes=None,
    )


def _inject_step_identity_strict(
    *,
    step_weights: pd.DataFrame,
    universe_codes: set[str],
    revised_from_codes: set[str] | None,
) -> tuple[pd.DataFrame, set[str]]:
    if step_weights.empty:
        step = pd.DataFrame(columns=["from_code", "to_code", "weight"])
    else:
        step = step_weights[["from_code", "to_code", "weight"]].copy()
    missing = _normalize_code_set(universe_codes) - set(step["from_code"])
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


def _check_weight_bounds(weights: pd.DataFrame, *, bound_tol: float, context: str) -> None:
    min_weight = float(weights["weight"].min())
    max_weight = float(weights["weight"].max())
    if min_weight < -bound_tol or max_weight > 1.0 + bound_tol:
        raise ValueError(
            "Weights outside [0, 1] tolerance in "
            f"{context}: min={min_weight}, max={max_weight}, tol={bound_tol}"
        )


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)


def _write_unresolved_details(
    rows: list[dict[str, object]],
    *,
    path: Path,
) -> None:
    if not rows:
        return
    columns = [
        "origin_year",
        "target_year",
        "direction",
        "measure",
        "period",
        "step_index",
        "code",
        "reason",
    ]
    details = pd.DataFrame(rows, columns=columns)
    _append_csv(details, path)


def chain_weights_for_year(
    *,
    origin_year: str | int,
    target_year: str | int,
    measure: str,
    code_universe: dict[int, set[str]],
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    finalize_weights: bool = False,
    neg_tol: float = 1e-6,
    pos_tol: float = 1e-10,
    row_sum_tol: float = 1e-6,
    fail_on_missing: bool = True,
    revised_codes_by_step: Mapping[tuple[str, str], Iterable[str]] | None = None,
    strict_revised_link_validation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    origin = _normalize_year(origin_year)
    target = _normalize_year(target_year)
    periods, direction = _chain_periods(origin, target)
    if not periods:
        raise ValueError("origin_year and target_year are identical; no chaining required")
    revised_index = _normalize_revised_index(revised_codes_by_step)

    diagnostics_rows: list[dict[str, object]] = []
    current: pd.DataFrame | None = None
    expected_from: set[str] | None = None

    for idx, period in enumerate(periods):
        step_weights = _load_weights(
            period=period,
            direction=direction,
            measure=measure,
            weights_dir=weights_dir,
            validate=False,
        )
        step_year = int(period[:4]) if direction == "a_to_b" else int(period[4:])
        if step_year not in code_universe:
            raise ValueError(f"Missing code universe for year {step_year}")
        revised_step_codes = _revised_codes_for_step(
            period=period,
            direction=direction,
            strict_revised_link_validation=strict_revised_link_validation,
            revised_codes_by_step=revised_index,
        )
        step_weights, unresolved_step_missing = _inject_step_identity_strict(
            step_weights=step_weights,
            universe_codes=code_universe[step_year],
            revised_from_codes=revised_step_codes,
        )
        if unresolved_step_missing and fail_on_missing:
            sample = sorted(unresolved_step_missing)[:10]
            raise ValueError(
                "Unresolved revised step links after identity injection "
                f"for {period} ({direction}, {measure}): {sample}"
            )
        _check_weight_bounds(
            step_weights,
            bound_tol=row_sum_tol,
            context=f"{period} {direction} {measure}",
        )
        if current is None:
            current = step_weights
            expected_from = set(current["from_code"])
            diagnostics_rows.append(
                {
                    "step_index": idx,
                    "period": period,
                    "direction": direction,
                    "n_rows_in": len(step_weights),
                    "n_rows_step": len(step_weights),
                    "n_rows_out": len(step_weights),
                    "n_from_codes_in": len(expected_from),
                    "n_from_codes_out": len(expected_from),
                    "n_missing_from_codes": 0,
                    "n_unresolved_revised_step_missing": len(unresolved_step_missing),
                    "n_unresolved_revised_missing_mid": 0,
                    "n_unresolved_revised_total": len(unresolved_step_missing),
                    "max_row_sum_dev": _max_row_sum_dev(current),
                }
            )
            continue

        chained, unresolved_missing_mid = _compose_weights(
            current,
            step_weights,
            revised_mid_codes=revised_step_codes,
        )
        if unresolved_missing_mid and fail_on_missing:
            sample = sorted(unresolved_missing_mid)[:10]
            raise ValueError(
                "Unresolved revised intermediate links during chaining "
                f"for {period} ({direction}, {measure}): {sample}"
            )

        missing_from = set()
        if expected_from is not None:
            missing_from = expected_from - set(chained["from_code"])
            if missing_from and fail_on_missing:
                sample = sorted(list(missing_from))[:10]
                raise ValueError(
                    f"Missing chained weights for {len(missing_from)} codes after {period}: {sample}"
                )

        max_dev = _max_row_sum_dev(chained)
        if max_dev > row_sum_tol and fail_on_missing:
            raise ValueError(
                f"Row sums deviate from 1 by {max_dev} after {period} (tol {row_sum_tol})"
            )

        diagnostics_rows.append(
            {
                "step_index": idx,
                "period": period,
                "direction": direction,
                "n_rows_in": len(current),
                "n_rows_step": len(step_weights),
                "n_rows_out": len(chained),
                "n_from_codes_in": len(expected_from or []),
                "n_from_codes_out": chained["from_code"].nunique(),
                "n_missing_from_codes": len(missing_from),
                "n_unresolved_revised_step_missing": len(unresolved_step_missing),
                "n_unresolved_revised_missing_mid": len(unresolved_missing_mid),
                "n_unresolved_revised_total": len(unresolved_step_missing)
                + len(unresolved_missing_mid),
                "max_row_sum_dev": max_dev,
            }
        )

        current = chained
        expected_from = set(current["from_code"])

    if current is None:
        raise ValueError("Failed to build chained weights; no periods loaded")

    if finalize_weights:
        from ..application import finalize_weights_table

        current = finalize_weights_table(
            current, neg_tol=neg_tol, pos_tol=pos_tol, row_sum_tol=row_sum_tol
        )

    diagnostics = pd.DataFrame(diagnostics_rows)
    return current, diagnostics, direction


def _build_forward_chains(
    *,
    start_year: int,
    target_year: int,
    measure: str,
    code_universe: dict[int, set[str]],
    weights_dir: Path,
    row_sum_tol: float,
    fail_on_missing: bool,
    revised_codes_by_step: Mapping[tuple[str, str], set[str]],
    strict_revised_link_validation: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    chains: dict[str, pd.DataFrame] = {}
    diagnostics_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    cumulative_next: pd.DataFrame | None = None

    for year in range(target_year - 1, start_year - 1, -1):
        period = f"{year}{year + 1}"
        step_weights = _load_weights(
            period=period,
            direction="a_to_b",
            measure=measure,
            weights_dir=weights_dir,
            validate=False,
        )
        if year not in code_universe:
            raise ValueError(f"Missing code universe for year {year}")
        revised_step_codes = _revised_codes_for_step(
            period=period,
            direction="a_to_b",
            strict_revised_link_validation=strict_revised_link_validation,
            revised_codes_by_step=revised_codes_by_step,
        )
        step_weights, unresolved_step_missing = _inject_step_identity_strict(
            step_weights=step_weights,
            universe_codes=code_universe[year],
            revised_from_codes=revised_step_codes,
        )
        step_unresolved_rows = _unresolved_rows(
            origin_year=str(year),
            target_year=str(target_year),
            direction="a_to_b",
            measure=measure,
            period=period,
            step_index=0,
            codes=unresolved_step_missing,
            reason="step_missing_revised",
        )
        unresolved_rows.extend(step_unresolved_rows)
        if unresolved_step_missing and fail_on_missing:
            sample = sorted(unresolved_step_missing)[:10]
            raise _UnresolvedRevisedLinksError(
                "Unresolved revised step links after identity injection "
                f"for {period} (a_to_b, {measure}): {sample}",
                step_unresolved_rows,
            )
        _check_weight_bounds(
            step_weights,
            bound_tol=row_sum_tol,
            context=f"{period} a_to_b {measure}",
        )
        expected_from = set(step_weights["from_code"])
        unresolved_missing_mid: set[str] = set()
        if cumulative_next is None:
            current = step_weights
        else:
            next_period = f"{year + 1}{year + 2}"
            revised_mid_codes = _revised_codes_for_step(
                period=next_period,
                direction="a_to_b",
                strict_revised_link_validation=strict_revised_link_validation,
                revised_codes_by_step=revised_codes_by_step,
            )
            current, unresolved_missing_mid = _compose_weights(
                step_weights,
                cumulative_next,
                revised_mid_codes=revised_mid_codes,
            )
            mid_unresolved_rows = _unresolved_rows(
                origin_year=str(year),
                target_year=str(target_year),
                direction="a_to_b",
                measure=measure,
                period=next_period,
                step_index=1,
                codes=unresolved_missing_mid,
                reason="missing_mid_revised",
            )
            unresolved_rows.extend(mid_unresolved_rows)
            if unresolved_missing_mid and fail_on_missing:
                sample = sorted(unresolved_missing_mid)[:10]
                raise _UnresolvedRevisedLinksError(
                    "Unresolved revised intermediate links during chaining "
                    f"for {next_period} (a_to_b, {measure}): {sample}",
                    mid_unresolved_rows,
                )
        missing_from = expected_from - set(current["from_code"])
        max_dev = _max_row_sum_dev(current)

        if missing_from and fail_on_missing:
            sample = sorted(list(missing_from))[:10]
            raise ValueError(
                f"Missing chained weights for {len(missing_from)} codes after {period}: {sample}"
            )
        if max_dev > row_sum_tol and fail_on_missing:
            raise ValueError(
                f"Row sums deviate from 1 by {max_dev} after {period} (tol {row_sum_tol})"
            )

        chains[str(year)] = current
        diagnostics_rows.append(
            {
                "origin_year": str(year),
                "target_year": str(target_year),
                "direction": "a_to_b",
                "measure": measure,
                "n_steps": target_year - year,
                "n_rows_final": len(current),
                "n_from_codes_final": current["from_code"].nunique(),
                "n_to_codes_final": current["to_code"].nunique(),
                "n_missing_from_codes": len(missing_from),
                "n_unresolved_revised_step_missing": len(unresolved_step_missing),
                "n_unresolved_revised_missing_mid": len(unresolved_missing_mid),
                "n_unresolved_revised_total": len(unresolved_step_missing)
                + len(unresolved_missing_mid),
                "max_row_sum_dev": max_dev,
            }
        )
        cumulative_next = current

    return chains, diagnostics_rows, unresolved_rows


def _build_backward_chains(
    *,
    end_year: int,
    target_year: int,
    measure: str,
    code_universe: dict[int, set[str]],
    weights_dir: Path,
    row_sum_tol: float,
    fail_on_missing: bool,
    revised_codes_by_step: Mapping[tuple[str, str], set[str]],
    strict_revised_link_validation: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    chains: dict[str, pd.DataFrame] = {}
    diagnostics_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    cumulative_prev: pd.DataFrame | None = None

    for year in range(target_year + 1, end_year + 1):
        period = f"{year - 1}{year}"
        step_weights = _load_weights(
            period=period,
            direction="b_to_a",
            measure=measure,
            weights_dir=weights_dir,
            validate=False,
        )
        if year not in code_universe:
            raise ValueError(f"Missing code universe for year {year}")
        revised_step_codes = _revised_codes_for_step(
            period=period,
            direction="b_to_a",
            strict_revised_link_validation=strict_revised_link_validation,
            revised_codes_by_step=revised_codes_by_step,
        )
        step_weights, unresolved_step_missing = _inject_step_identity_strict(
            step_weights=step_weights,
            universe_codes=code_universe[year],
            revised_from_codes=revised_step_codes,
        )
        step_unresolved_rows = _unresolved_rows(
            origin_year=str(year),
            target_year=str(target_year),
            direction="b_to_a",
            measure=measure,
            period=period,
            step_index=0,
            codes=unresolved_step_missing,
            reason="step_missing_revised",
        )
        unresolved_rows.extend(step_unresolved_rows)
        if unresolved_step_missing and fail_on_missing:
            sample = sorted(unresolved_step_missing)[:10]
            raise _UnresolvedRevisedLinksError(
                "Unresolved revised step links after identity injection "
                f"for {period} (b_to_a, {measure}): {sample}",
                step_unresolved_rows,
            )
        _check_weight_bounds(
            step_weights,
            bound_tol=row_sum_tol,
            context=f"{period} b_to_a {measure}",
        )
        expected_from = set(step_weights["from_code"])
        unresolved_missing_mid: set[str] = set()
        if cumulative_prev is None:
            current = step_weights
        else:
            previous_period = f"{year - 2}{year - 1}"
            revised_mid_codes = _revised_codes_for_step(
                period=previous_period,
                direction="b_to_a",
                strict_revised_link_validation=strict_revised_link_validation,
                revised_codes_by_step=revised_codes_by_step,
            )
            current, unresolved_missing_mid = _compose_weights(
                step_weights,
                cumulative_prev,
                revised_mid_codes=revised_mid_codes,
            )
            mid_unresolved_rows = _unresolved_rows(
                origin_year=str(year),
                target_year=str(target_year),
                direction="b_to_a",
                measure=measure,
                period=previous_period,
                step_index=1,
                codes=unresolved_missing_mid,
                reason="missing_mid_revised",
            )
            unresolved_rows.extend(mid_unresolved_rows)
            if unresolved_missing_mid and fail_on_missing:
                sample = sorted(unresolved_missing_mid)[:10]
                raise _UnresolvedRevisedLinksError(
                    "Unresolved revised intermediate links during chaining "
                    f"for {previous_period} (b_to_a, {measure}): {sample}",
                    mid_unresolved_rows,
                )
        missing_from = expected_from - set(current["from_code"])
        max_dev = _max_row_sum_dev(current)

        if missing_from and fail_on_missing:
            sample = sorted(list(missing_from))[:10]
            raise ValueError(
                f"Missing chained weights for {len(missing_from)} codes after {period}: {sample}"
            )
        if max_dev > row_sum_tol and fail_on_missing:
            raise ValueError(
                f"Row sums deviate from 1 by {max_dev} after {period} (tol {row_sum_tol})"
            )

        chains[str(year)] = current
        diagnostics_rows.append(
            {
                "origin_year": str(year),
                "target_year": str(target_year),
                "direction": "b_to_a",
                "measure": measure,
                "n_steps": year - target_year,
                "n_rows_final": len(current),
                "n_from_codes_final": current["from_code"].nunique(),
                "n_to_codes_final": current["to_code"].nunique(),
                "n_missing_from_codes": len(missing_from),
                "n_unresolved_revised_step_missing": len(unresolved_step_missing),
                "n_unresolved_revised_missing_mid": len(unresolved_missing_mid),
                "n_unresolved_revised_total": len(unresolved_step_missing)
                + len(unresolved_missing_mid),
                "max_row_sum_dev": max_dev,
            }
        )
        cumulative_prev = current

    return chains, diagnostics_rows, unresolved_rows


def build_chained_weights_for_range(
    *,
    start_year: str | int,
    end_year: str | int,
    target_year: str | int,
    measures: Sequence[str],
    code_universe: dict[int, set[str]],
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    output_weights_dir: Path = DEFAULT_CHAINED_WEIGHTS_DIR,
    output_diagnostics_dir: Path = DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    finalize_weights: bool = False,
    neg_tol: float = 1e-6,
    pos_tol: float = 1e-10,
    row_sum_tol: float = 1e-6,
    fail_on_missing: bool = True,
    revised_codes_by_step: Mapping[tuple[str, str], Iterable[str]] | None = None,
    strict_revised_link_validation: bool = False,
    write_unresolved_details: bool = False,
) -> list[ChainedWeightsOutput]:
    start = int(_normalize_year(start_year))
    end = int(_normalize_year(end_year))
    target = _normalize_year(target_year)
    if start > end:
        raise ValueError("start_year must be <= end_year")
    revised_index = _normalize_revised_index(revised_codes_by_step)
    unresolved_path = output_diagnostics_dir / f"CN{target}" / "unresolved_details.csv"

    outputs: list[ChainedWeightsOutput] = []
    for measure in measures:
        forward_chains: dict[str, pd.DataFrame] = {}
        forward_diagnostics: list[dict[str, object]] = []
        forward_unresolved: list[dict[str, object]] = []
        backward_chains: dict[str, pd.DataFrame] = {}
        backward_diagnostics: list[dict[str, object]] = []
        backward_unresolved: list[dict[str, object]] = []

        try:
            if start < int(target):
                (
                    forward_chains,
                    forward_diagnostics,
                    forward_unresolved,
                ) = _build_forward_chains(
                    start_year=start,
                    target_year=int(target),
                    measure=measure,
                    code_universe=code_universe,
                    weights_dir=weights_dir,
                    row_sum_tol=row_sum_tol,
                    fail_on_missing=fail_on_missing,
                    revised_codes_by_step=revised_index,
                    strict_revised_link_validation=strict_revised_link_validation,
                )
            if end > int(target):
                (
                    backward_chains,
                    backward_diagnostics,
                    backward_unresolved,
                ) = _build_backward_chains(
                    end_year=end,
                    target_year=int(target),
                    measure=measure,
                    code_universe=code_universe,
                    weights_dir=weights_dir,
                    row_sum_tol=row_sum_tol,
                    fail_on_missing=fail_on_missing,
                    revised_codes_by_step=revised_index,
                    strict_revised_link_validation=strict_revised_link_validation,
                )
        except _UnresolvedRevisedLinksError as exc:
            if write_unresolved_details:
                _write_unresolved_details(exc.unresolved_rows, path=unresolved_path)
            raise ValueError(str(exc)) from exc

        all_chains = {**forward_chains, **backward_chains}
        all_diagnostics = {
            row["origin_year"]: row for row in forward_diagnostics + backward_diagnostics
        }
        all_unresolved = forward_unresolved + backward_unresolved
        if write_unresolved_details:
            _write_unresolved_details(all_unresolved, path=unresolved_path)

        for origin, weights in all_chains.items():
            if finalize_weights:
                from ..application import finalize_weights_table

                weights = finalize_weights_table(
                    weights, neg_tol=neg_tol, pos_tol=pos_tol, row_sum_tol=row_sum_tol
                )

            diagnostics_row = dict(all_diagnostics.get(origin, {}))
            diagnostics_row.update(
                {
                    "n_rows_final": len(weights),
                    "n_from_codes_final": weights["from_code"].nunique(),
                    "n_to_codes_final": weights["to_code"].nunique(),
                    "max_row_sum_dev": _max_row_sum_dev(weights),
                    "finalize_weights": finalize_weights,
                    "neg_tol": neg_tol,
                    "pos_tol": pos_tol,
                    "row_sum_tol": row_sum_tol,
                }
            )
            diagnostics = pd.DataFrame([diagnostics_row])

            direction = diagnostics_row.get("direction", _chain_periods(origin, target)[1])
            weights_out = weights.copy()
            weights_out["from_vintage_year"] = origin
            weights_out["to_vintage_year"] = target
            weights_out["direction"] = direction
            weights_out["measure"] = measure
            weights_out = weights_out[
                ["from_vintage_year", "to_vintage_year", "direction", "measure", "from_code", "to_code", "weight"]
            ]

            measure_tag = measure.lower()
            weights_path = (
                output_weights_dir
                / f"CN{target}"
                / "weights"
                / origin
                / direction
                / measure_tag
                / "weights.csv"
            )
            diagnostics_path = output_diagnostics_dir / f"CN{target}" / "diagnostics.csv"
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            weights_out.to_csv(weights_path, index=False)
            _append_csv(diagnostics, diagnostics_path)

            outputs.append(
                ChainedWeightsOutput(
                    origin_year=origin,
                    target_year=target,
                    direction=direction,
                    measure=measure,
                    weights=weights_out,
                    diagnostics=diagnostics,
                    weights_path=weights_path,
                    diagnostics_path=diagnostics_path,
                )
            )

    return outputs
