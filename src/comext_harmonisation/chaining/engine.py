"""Chain adjacent CN conversion weights across multiple vintages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd
from ..core.codes import (
    chain_periods,
    normalize_codes,
    normalize_year,
)
from ..core.diagnostics import append_csv, append_detail_rows
from ..core.revised_links import normalize_revised_index
from ..weights.finalize import finalize_weights_table_impl
from ..weights.io import read_adjacent_weights
from ..concordance.groups import build_concordance_groups
from ..weights.schema import DEFAULT_WEIGHTS_DIR
from .composition import (
    check_weight_bounds,
    compose_weights,
    inject_step_identity_strict,
    max_row_sum_dev,
)


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
    return normalize_year(year)


def _normalize_codes(series: pd.Series) -> pd.Series:
    return normalize_codes(series)


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


def _normalize_revised_index(
    revised_codes_by_step: Mapping[tuple[str, str], Iterable[str]] | None,
) -> dict[tuple[str, str], set[str]]:
    return normalize_revised_index(revised_codes_by_step)


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
    return chain_periods(origin_year, target_year)


def _load_weights(
    *,
    period: str,
    direction: str,
    measure: str,
    weights_dir: Path,
    validate: bool = True,
) -> pd.DataFrame:
    return read_adjacent_weights(
        period=period,
        direction=direction,
        measure=measure,
        weights_dir=weights_dir,
        validate=validate,
    )


def _append_csv(df: pd.DataFrame, path: Path) -> None:
    append_csv(df, path)


def _write_unresolved_details(
    rows: list[dict[str, object]],
    *,
    path: Path,
) -> None:
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
        ],
    )


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
        step_weights, unresolved_step_missing = inject_step_identity_strict(
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
        check_weight_bounds(
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
                    "max_row_sum_dev": max_row_sum_dev(current),
                }
            )
            continue

        chained, unresolved_missing_mid = compose_weights(
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

        max_dev = max_row_sum_dev(chained)
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
        current = finalize_weights_table_impl(
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
    return _build_directional_chains(
        years=range(target_year - 1, start_year - 1, -1),
        target_year=target_year,
        direction="a_to_b",
        measure=measure,
        code_universe=code_universe,
        weights_dir=weights_dir,
        row_sum_tol=row_sum_tol,
        fail_on_missing=fail_on_missing,
        revised_codes_by_step=revised_codes_by_step,
        strict_revised_link_validation=strict_revised_link_validation,
        period_for_year=lambda year: f"{year}{year + 1}",
        mid_period_for_year=lambda year: f"{year + 1}{year + 2}",
        n_steps_for_year=lambda year: target_year - year,
    )


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
    return _build_directional_chains(
        years=range(target_year + 1, end_year + 1),
        target_year=target_year,
        direction="b_to_a",
        measure=measure,
        code_universe=code_universe,
        weights_dir=weights_dir,
        row_sum_tol=row_sum_tol,
        fail_on_missing=fail_on_missing,
        revised_codes_by_step=revised_codes_by_step,
        strict_revised_link_validation=strict_revised_link_validation,
        period_for_year=lambda year: f"{year - 1}{year}",
        mid_period_for_year=lambda year: f"{year - 2}{year - 1}",
        n_steps_for_year=lambda year: year - target_year,
    )


def _build_directional_chains(
    *,
    years: Iterable[int],
    target_year: int,
    direction: str,
    measure: str,
    code_universe: dict[int, set[str]],
    weights_dir: Path,
    row_sum_tol: float,
    fail_on_missing: bool,
    revised_codes_by_step: Mapping[tuple[str, str], set[str]],
    strict_revised_link_validation: bool,
    period_for_year: Callable[[int], str],
    mid_period_for_year: Callable[[int], str],
    n_steps_for_year: Callable[[int], int],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]], list[dict[str, object]]]:
    chains: dict[str, pd.DataFrame] = {}
    diagnostics_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    cumulative_next: pd.DataFrame | None = None

    for year in years:
        period = period_for_year(year)
        step_weights = _load_weights(
            period=period,
            direction=direction,
            measure=measure,
            weights_dir=weights_dir,
            validate=False,
        )
        if year not in code_universe:
            raise ValueError(f"Missing code universe for year {year}")
        revised_step_codes = _revised_codes_for_step(
            period=period,
            direction=direction,
            strict_revised_link_validation=strict_revised_link_validation,
            revised_codes_by_step=revised_codes_by_step,
        )
        step_weights, unresolved_step_missing = inject_step_identity_strict(
            step_weights=step_weights,
            universe_codes=code_universe[year],
            revised_from_codes=revised_step_codes,
        )
        step_unresolved_rows = _unresolved_rows(
            origin_year=str(year),
            target_year=str(target_year),
            direction=direction,
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
                f"for {period} ({direction}, {measure}): {sample}",
                step_unresolved_rows,
            )
        check_weight_bounds(
            step_weights,
            bound_tol=row_sum_tol,
            context=f"{period} {direction} {measure}",
        )
        expected_from = set(step_weights["from_code"])
        unresolved_missing_mid: set[str] = set()
        if cumulative_next is None:
            current = step_weights
        else:
            next_period = mid_period_for_year(year)
            revised_mid_codes = _revised_codes_for_step(
                period=next_period,
                direction=direction,
                strict_revised_link_validation=strict_revised_link_validation,
                revised_codes_by_step=revised_codes_by_step,
            )
            current, unresolved_missing_mid = compose_weights(
                step_weights,
                cumulative_next,
                revised_mid_codes=revised_mid_codes,
            )
            mid_unresolved_rows = _unresolved_rows(
                origin_year=str(year),
                target_year=str(target_year),
                direction=direction,
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
                    f"for {next_period} ({direction}, {measure}): {sample}",
                    mid_unresolved_rows,
                )
        missing_from = expected_from - set(current["from_code"])
        max_dev = max_row_sum_dev(current)

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
                "direction": direction,
                "measure": measure,
                "n_steps": n_steps_for_year(year),
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
                weights = finalize_weights_table_impl(
                    weights, neg_tol=neg_tol, pos_tol=pos_tol, row_sum_tol=row_sum_tol
                )

            diagnostics_row = dict(all_diagnostics.get(origin, {}))
            diagnostics_row.update(
                {
                    "n_rows_final": len(weights),
                    "n_from_codes_final": weights["from_code"].nunique(),
                    "n_to_codes_final": weights["to_code"].nunique(),
                    "max_row_sum_dev": max_row_sum_dev(weights),
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
