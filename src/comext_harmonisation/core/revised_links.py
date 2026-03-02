"""Shared revised-link normalization helpers."""

from __future__ import annotations

from typing import Iterable, Mapping

from .codes import normalize_code_set


def normalize_revised_index(
    revised_codes_by_step: Mapping[tuple[str, str], Iterable[str]] | None,
) -> dict[tuple[str, str], set[str]]:
    if revised_codes_by_step is None:
        return {}
    normalized: dict[tuple[str, str], set[str]] = {}
    for key, codes in revised_codes_by_step.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                "revised_codes_by_step keys must be (period, direction) tuples"
            )
        period, direction = key
        if direction not in {"a_to_b", "b_to_a"}:
            raise ValueError(
                "revised_codes_by_step direction must be 'a_to_b' or 'b_to_a'"
            )
        normalized[(str(period), direction)] = normalize_code_set(codes)
    return normalized

