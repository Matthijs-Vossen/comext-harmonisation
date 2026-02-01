"""Shared analysis metrics and registry."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

MetricFn = Callable[..., float]

_METRICS: dict[str, MetricFn] = {}


def register_metric(name: str) -> Callable[[MetricFn], MetricFn]:
    def decorator(func: MetricFn) -> MetricFn:
        _METRICS[name] = func
        return func

    return decorator


def get_metric(name: str) -> MetricFn:
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown metric '{name}'. Available: {sorted(_METRICS)}") from exc


def list_metrics() -> list[str]:
    return sorted(_METRICS)


@register_metric("r2_45")
def r2_45(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    y_bar = float(np.mean(y))
    sse = float(np.sum((y - x) ** 2))
    sst = float(np.sum((y - y_bar) ** 2))
    if sst == 0:
        return float("nan")
    return 1.0 - sse / sst


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute a weighted mean, ignoring non-finite entries and non-positive weights."""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float((weights[mask] * values[mask]).sum() / weights[mask].sum())


def trade_weighted_exposure(
    *,
    totals: pd.DataFrame,
    ambiguous_sources: Iterable[str],
) -> tuple[float, float]:
    """Return (exposure, ambiguous_trade) for a step."""
    if totals.empty:
        return float("nan"), 0.0
    total_trade = float(totals["value"].sum())
    if total_trade <= 0:
        return float("nan"), 0.0
    ambiguous_trade = float(
        totals.loc[totals["PRODUCT_NC"].isin(ambiguous_sources), "value"].sum()
    )
    return ambiguous_trade / total_trade, ambiguous_trade


def entropy_weighted(
    *,
    totals: pd.DataFrame,
    step_weights: pd.DataFrame,
    feasible_map: dict[str, list[str]],
    ambiguous_sources: Iterable[str],
    estimable_sources: Iterable[str],
) -> tuple[float, float]:
    """Compute trade-weighted normalized entropy and its trade weight."""
    if totals.empty:
        return float("nan"), 0.0
    ambiguous_sources = set(ambiguous_sources)
    estimable_sources = set(estimable_sources)
    entropy_sources = ambiguous_sources & estimable_sources
    if not entropy_sources:
        return float("nan"), 0.0
    ambiguous_trade = float(
        totals.loc[totals["PRODUCT_NC"].isin(entropy_sources), "value"].sum()
    )
    if ambiguous_trade <= 0:
        return float("nan"), 0.0

    entropy_rows: list[tuple[float, float]] = []
    for code in entropy_sources:
        weights_code = step_weights.loc[step_weights["from_code"] == code, "weight"]
        weights_code = weights_code[weights_code > 0]
        weights_sum = float(weights_code.sum())
        if weights_sum <= 0:
            raise ValueError(
                f"Entropy computation failed: row {code} has no positive weights after clipping."
            )
        probs = (weights_code / weights_sum).to_numpy()
        positive_probs = probs[probs > 0]
        k_est = len(feasible_map.get(code, []))
        if k_est <= 1:
            h_norm = 0.0
        else:
            h_val = float(-(positive_probs * np.log(positive_probs)).sum())
            h_norm = h_val / float(np.log(k_est))
        trade_val = float(totals.loc[totals["PRODUCT_NC"] == code, "value"].sum())
        entropy_rows.append((trade_val, h_norm))
    if not entropy_rows:
        return float("nan"), 0.0
    weights_trade = np.array([row[0] for row in entropy_rows], dtype=float)
    values = np.array([row[1] for row in entropy_rows], dtype=float)
    denom = float(weights_trade.sum())
    if denom <= 0:
        return float("nan"), 0.0
    return float((weights_trade * values).sum() / denom), ambiguous_trade
