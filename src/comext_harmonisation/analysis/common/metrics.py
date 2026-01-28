"""Shared analysis metrics and registry."""

from __future__ import annotations

from typing import Callable

import numpy as np

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
