"""Chain adjacent CN conversion weights across multiple vintages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse as sp
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


def _compose_weights(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty:
        return pd.DataFrame(columns=["from_code", "to_code", "weight"])
    if right.empty:
        return left.groupby(["from_code", "to_code"], as_index=False)["weight"].sum()

    left_to = set(left["to_code"])
    right_from = set(right["from_code"])
    common_mid = sorted(left_to & right_from)
    missing_mid = sorted(left_to - right_from)

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
        carry = left[left["to_code"].isin(missing_mid)]
        carry = carry.groupby(["from_code", "to_code"], as_index=False)["weight"].sum()
        chained_parts.append(carry)

    if not chained_parts:
        return pd.DataFrame(columns=["from_code", "to_code", "weight"])

    combined = pd.concat(chained_parts, ignore_index=True)
    return combined.groupby(["from_code", "to_code"], as_index=False)["weight"].sum()


def _inject_step_identity(
    step_weights: pd.DataFrame,
    *,
    universe_codes: set[str],
) -> pd.DataFrame:
    if step_weights.empty:
        return step_weights
    missing = _normalize_code_set(universe_codes) - set(step_weights["from_code"])
    if not missing:
        return step_weights
    identity = pd.DataFrame(
        {
            "from_code": list(missing),
            "to_code": list(missing),
            "weight": 1.0,
        }
    )
    return pd.concat([step_weights, identity], ignore_index=True)


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
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    origin = _normalize_year(origin_year)
    target = _normalize_year(target_year)
    periods, direction = _chain_periods(origin, target)
    if not periods:
        raise ValueError("origin_year and target_year are identical; no chaining required")

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
        step_weights = _inject_step_identity(
            step_weights, universe_codes=code_universe[step_year]
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
                    "max_row_sum_dev": _max_row_sum_dev(current),
                }
            )
            continue

        chained = _compose_weights(current, step_weights)

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
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    chains: dict[str, pd.DataFrame] = {}
    diagnostics_rows: list[dict[str, object]] = []
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
        step_weights = _inject_step_identity(
            step_weights, universe_codes=code_universe[year]
        )
        _check_weight_bounds(
            step_weights,
            bound_tol=row_sum_tol,
            context=f"{period} a_to_b {measure}",
        )
        expected_from = set(step_weights["from_code"])
        if cumulative_next is None:
            current = step_weights
        else:
            current = _compose_weights(step_weights, cumulative_next)
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
                "max_row_sum_dev": max_dev,
            }
        )
        cumulative_next = current

    return chains, diagnostics_rows


def _build_backward_chains(
    *,
    end_year: int,
    target_year: int,
    measure: str,
    code_universe: dict[int, set[str]],
    weights_dir: Path,
    row_sum_tol: float,
    fail_on_missing: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    chains: dict[str, pd.DataFrame] = {}
    diagnostics_rows: list[dict[str, object]] = []
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
        step_weights = _inject_step_identity(
            step_weights, universe_codes=code_universe[year]
        )
        _check_weight_bounds(
            step_weights,
            bound_tol=row_sum_tol,
            context=f"{period} b_to_a {measure}",
        )
        expected_from = set(step_weights["from_code"])
        if cumulative_prev is None:
            current = step_weights
        else:
            current = _compose_weights(step_weights, cumulative_prev)
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
                "max_row_sum_dev": max_dev,
            }
        )
        cumulative_prev = current

    return chains, diagnostics_rows


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
) -> list[ChainedWeightsOutput]:
    start = int(_normalize_year(start_year))
    end = int(_normalize_year(end_year))
    target = _normalize_year(target_year)
    if start > end:
        raise ValueError("start_year must be <= end_year")

    outputs: list[ChainedWeightsOutput] = []
    for measure in measures:
        forward_chains: dict[str, pd.DataFrame] = {}
        forward_diagnostics: list[dict[str, object]] = []
        backward_chains: dict[str, pd.DataFrame] = {}
        backward_diagnostics: list[dict[str, object]] = []

        if start < int(target):
            forward_chains, forward_diagnostics = _build_forward_chains(
                start_year=start,
                target_year=int(target),
                measure=measure,
                code_universe=code_universe,
                weights_dir=weights_dir,
                row_sum_tol=row_sum_tol,
                fail_on_missing=fail_on_missing,
            )
        if end > int(target):
            backward_chains, backward_diagnostics = _build_backward_chains(
                end_year=end,
                target_year=int(target),
                measure=measure,
                code_universe=code_universe,
                weights_dir=weights_dir,
                row_sum_tol=row_sum_tol,
                fail_on_missing=fail_on_missing,
            )

        all_chains = {**forward_chains, **backward_chains}
        all_diagnostics = {
            row["origin_year"]: row for row in forward_diagnostics + backward_diagnostics
        }

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
