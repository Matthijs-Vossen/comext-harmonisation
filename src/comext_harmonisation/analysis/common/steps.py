"""Shared helpers for step-level analysis metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import entropy_weighted, trade_weighted_exposure
from .shares import normalize_codes


def chain_steps(base_year: int, target_year: int) -> list[dict[str, int | str]]:
    steps: list[dict[str, int | str]] = []
    if base_year == target_year:
        return steps
    if base_year < target_year:
        for year in range(base_year, target_year):
            steps.append(
                {
                    "period": f"{year}{year + 1}",
                    "direction": "a_to_b",
                    "source_year": year,
                    "target_year": year + 1,
                }
            )
    else:
        for year in range(base_year - 1, target_year - 1, -1):
            steps.append(
                {
                    "period": f"{year}{year + 1}",
                    "direction": "b_to_a",
                    "source_year": year + 1,
                    "target_year": year,
                }
            )
    return steps


def load_annual_totals(
    *,
    annual_base_dir: Path,
    year: int,
    measure: str,
    exclude_reporters: Iterable[str],
    exclude_partners: Iterable[str],
) -> pd.DataFrame:
    data_path = annual_base_dir / f"comext_{year}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing annual data file: {data_path}")
    cols = ["REPORTER", "PARTNER", "PRODUCT_NC", measure]
    data = pd.read_parquet(data_path, columns=cols)
    if exclude_reporters:
        data = data.loc[~data["REPORTER"].isin(exclude_reporters)]
    if exclude_partners:
        data = data.loc[~data["PARTNER"].isin(exclude_partners)]
    totals = (
        data.groupby("PRODUCT_NC", as_index=False, sort=False)[measure]
        .sum()
        .rename(columns={measure: "value"})
    )
    totals["PRODUCT_NC"] = normalize_codes(totals["PRODUCT_NC"])
    return totals


def sample_source_codes(
    *,
    sample_target_codes: set[str],
    weights_to_target: pd.DataFrame | None,
) -> set[str]:
    if not sample_target_codes:
        return set()
    if weights_to_target is None:
        return set(sample_target_codes)
    weights = weights_to_target[["from_code", "to_code"]].copy()
    weights["from_code"] = normalize_codes(weights["from_code"])
    weights["to_code"] = normalize_codes(weights["to_code"])
    matched = weights[weights["to_code"].isin(sample_target_codes)][
        "from_code"
    ].unique().tolist()
    return set(matched)


def load_step_weights(
    *,
    period: str,
    direction: str,
    measure: str,
    weights_dir: Path,
) -> pd.DataFrame:
    measure_tag = measure.lower()
    weights_path = weights_dir / period / direction / measure_tag / "weights_ambiguous.csv"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    weights = pd.read_csv(weights_path)
    if weights.empty:
        return pd.DataFrame(columns=["from_code", "to_code", "weight"])
    weights = weights[["from_code", "to_code", "weight"]].copy()
    weights["from_code"] = normalize_codes(weights["from_code"])
    weights["to_code"] = normalize_codes(weights["to_code"])
    weights["weight"] = weights["weight"].astype(float)
    return weights


def feasible_target_map(
    period_edges: pd.DataFrame,
    direction: str,
) -> dict[str, list[str]]:
    if period_edges.empty:
        return {}
    if direction == "a_to_b":
        grouped = period_edges.groupby("vintage_a_code", sort=False)[
            "vintage_b_code"
        ].unique()
    else:
        grouped = period_edges.groupby("vintage_b_code", sort=False)[
            "vintage_a_code"
        ].unique()
    mapping: dict[str, list[str]] = {}
    for code, targets in grouped.items():
        mapping[str(code)] = normalize_codes(pd.Series(list(targets))).tolist()
    return mapping


def compute_step_metrics(
    *,
    base_year: int,
    target_year: int,
    sample_target_codes: set[str],
    weights_by_year: dict[str, pd.DataFrame],
    groups,
    annual_base_dir: Path,
    measure: str,
    weights_dir: Path,
    weights_source: str,
    exclude_reporters: Iterable[str],
    exclude_partners: Iterable[str],
    compute_exposure: bool,
    compute_diffuseness: bool,
) -> list[dict[str, object]]:
    if not (compute_exposure or compute_diffuseness):
        return []
    step_rows_chain: list[dict[str, object]] = []
    for step_idx, step in enumerate(chain_steps(base_year, target_year), start=1):
        period = str(step["period"])
        direction = str(step["direction"])
        source_year = int(step["source_year"])
        weights_to_target = (
            weights_by_year.get(str(source_year)) if source_year != target_year else None
        )
        sample_source = sample_source_codes(
            sample_target_codes=sample_target_codes,
            weights_to_target=weights_to_target,
        )

        totals = load_annual_totals(
            annual_base_dir=annual_base_dir,
            year=source_year,
            measure=measure,
            exclude_reporters=exclude_reporters,
            exclude_partners=exclude_partners,
        )
        totals = totals.loc[totals["PRODUCT_NC"].isin(sample_source)]
        total_trade = float(totals["value"].sum())

        step_weights = load_step_weights(
            period=period,
            direction=direction,
            measure=weights_source,
            weights_dir=weights_dir,
        )
        period_edges = groups.edges.loc[groups.edges["period"] == period]
        feasible_map = feasible_target_map(period_edges, direction)
        ambiguous_sources = {
            code for code, targets in feasible_map.items() if len(targets) > 1
        }
        estimable_sources = set(step_weights["from_code"].unique().tolist())
        ambiguous_sources = ambiguous_sources & set(totals["PRODUCT_NC"].unique().tolist())

        exposure = float("nan")
        ambiguous_trade = float("nan")
        if compute_exposure:
            exposure, ambiguous_trade = trade_weighted_exposure(
                totals=totals, ambiguous_sources=ambiguous_sources
            )

        step_entropy = float("nan")
        if compute_diffuseness:
            step_entropy, ambiguous_trade_entropy = entropy_weighted(
                totals=totals,
                step_weights=step_weights,
                feasible_map=feasible_map,
                ambiguous_sources=ambiguous_sources,
                estimable_sources=estimable_sources,
            )
            if np.isnan(ambiguous_trade):
                ambiguous_trade = ambiguous_trade_entropy

        step_rows_chain.append(
            {
                "base_year": base_year,
                "compare_year": target_year,
                "target_year": target_year,
                "step_index": step_idx,
                "period": period,
                "direction": direction,
                "source_year": source_year,
                "total_trade_sample": total_trade,
                "ambiguous_trade": ambiguous_trade,
                "ambiguity_exposure": exposure,
                "diffuseness": step_entropy,
                "n_ambiguous_sources": int(len(ambiguous_sources)),
            }
        )
    return step_rows_chain
