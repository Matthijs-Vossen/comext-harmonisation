"""Shared I/O helpers for adjacent conversion weights."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import validate_weight_table
from ..core.codes import normalize_codes


def read_adjacent_weights(
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

    weights = frames[0].copy() if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    weights = weights[["from_code", "to_code", "weight"]].copy()
    weights["from_code"] = normalize_codes(weights["from_code"])
    weights["to_code"] = normalize_codes(weights["to_code"])
    weights["weight"] = weights["weight"].astype(float)

    if validate:
        validate_weight_table(
            weights,
            schema="minimal",
            check_bounds=True,
            check_row_sums=True,
        )
    return weights
