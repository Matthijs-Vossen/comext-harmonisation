"""Shared sampling helpers for chain-based analysis runners."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ...concordance.mappings import get_ambiguous_group_summary
from .shares import normalize_codes
from .steps import chain_steps


class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def add(self, item: str) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: str) -> str:
        self.add(item)
        parent = self._parent[item]
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self._size[root_left] < self._size[root_right]:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        self._size[root_left] += self._size[root_right]


def ambiguous_edges_for_step(
    *,
    groups,
    period: str,
    direction: str,
) -> pd.DataFrame:
    summary = get_ambiguous_group_summary(groups, direction)
    summary_period = summary.loc[summary["period"] == period]
    if summary_period.empty:
        return groups.edges.iloc[0:0].copy()
    return groups.edges.merge(
        summary_period[["period", "group_id"]],
        on=["period", "group_id"],
        how="inner",
    )


def map_codes_to_target(
    codes: Iterable[str],
    weights: pd.DataFrame | None,
    *,
    preserve_unmapped: bool,
) -> set[str]:
    codes_list = [str(code) for code in codes]
    if not codes_list:
        return set()

    codes_series = normalize_codes(pd.Series(codes_list))
    if weights is None:
        return set(codes_series.tolist())

    mapped_weights = weights[["from_code", "to_code"]].copy()
    mapped_weights["from_code"] = normalize_codes(mapped_weights["from_code"])
    mapped_weights["to_code"] = normalize_codes(mapped_weights["to_code"])
    mapped = set(
        mapped_weights[mapped_weights["from_code"].isin(codes_series)]["to_code"]
        .astype(str)
        .tolist()
    )

    if preserve_unmapped:
        missing = set(codes_series.tolist()) - set(mapped_weights["from_code"].unique())
        mapped |= missing
    return mapped


def build_chain_group_map(
    *,
    groups,
    base_year: int,
    target_year: int,
    weights_by_year: dict[str, pd.DataFrame],
    preserve_unmapped: bool,
) -> tuple[pd.DataFrame, set[str], set[str]]:
    uf = UnionFind()
    sample_codes: set[str] = set()

    for step in chain_steps(base_year, target_year):
        period = str(step["period"])
        direction = str(step["direction"])
        step_edges = ambiguous_edges_for_step(groups=groups, period=period, direction=direction)
        if step_edges.empty:
            continue

        year_a = int(step_edges["vintage_a_year"].iloc[0])
        year_b = int(step_edges["vintage_b_year"].iloc[0])
        weights_a = weights_by_year.get(str(year_a)) if year_a != target_year else None
        weights_b = weights_by_year.get(str(year_b)) if year_b != target_year else None

        for _, edges in step_edges.groupby("group_id", sort=False):
            codes_a = set(edges["vintage_a_code"].tolist())
            codes_b = set(edges["vintage_b_code"].tolist())
            mapped = set()
            mapped |= map_codes_to_target(
                codes_a,
                weights_a,
                preserve_unmapped=preserve_unmapped,
            )
            mapped |= map_codes_to_target(
                codes_b,
                weights_b,
                preserve_unmapped=preserve_unmapped,
            )
            mapped = set(normalize_codes(pd.Series(list(mapped))).tolist())
            if not mapped:
                continue
            sample_codes |= mapped
            mapped_list = list(mapped)
            for code in mapped_list:
                uf.add(code)
            anchor = mapped_list[0]
            for code in mapped_list[1:]:
                uf.union(anchor, code)

    if not sample_codes:
        return (
            pd.DataFrame(columns=["target_code", "group_id"]),
            set(),
            set(),
        )

    components: dict[str, list[str]] = {}
    for code in sample_codes:
        root = uf.find(code)
        components.setdefault(root, []).append(code)

    component_rows: list[tuple[str, list[str]]] = []
    for codes in components.values():
        codes_sorted = sorted(codes)
        component_rows.append((codes_sorted[0], codes_sorted))
    component_rows.sort(key=lambda item: item[0])

    group_map_rows: list[dict[str, str]] = []
    group_ids: set[str] = set()
    for idx, (_, codes) in enumerate(component_rows, start=1):
        group_id = f"{base_year}to{target_year}_g{idx:06d}"
        group_ids.add(group_id)
        for code in codes:
            group_map_rows.append({"target_code": code, "group_id": group_id})

    return pd.DataFrame(group_map_rows), group_ids, sample_codes

