"""Build concordance connected components and ambiguity flags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class ConcordanceGroups:
    edges: pd.DataFrame
    vintage_a_nodes: pd.DataFrame
    vintage_b_nodes: pd.DataFrame
    group_summary: pd.DataFrame


class UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._size: Dict[Tuple[str, str], int] = {}

    def add(self, item: Tuple[str, str]) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: Tuple[str, str]) -> Tuple[str, str]:
        self.add(item)
        parent = self._parent[item]
        if parent != item:
            self._parent[item] = self.find(parent)
        return self._parent[item]

    def union(self, left: Tuple[str, str], right: Tuple[str, str]) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self._size[root_left] < self._size[root_right]:
            root_left, root_right = root_right, root_left
        self._parent[root_right] = root_left
        self._size[root_left] += self._size[root_right]


def _group_ids_for_period(period_edges: pd.DataFrame) -> pd.DataFrame:
    uf = UnionFind()
    for row in period_edges.itertuples(index=False):
        a_node = ("A", row.vintage_a_code)
        b_node = ("B", row.vintage_b_code)
        uf.union(a_node, b_node)

    edge_roots: List[Tuple[str, str]] = []
    for row in period_edges.itertuples(index=False):
        a_node = ("A", row.vintage_a_code)
        edge_roots.append(uf.find(a_node))

    component_data: Dict[Tuple[str, str], Dict[str, object]] = {}
    for root, row in zip(edge_roots, period_edges.itertuples(index=False)):
        data = component_data.setdefault(root, {"vintage_a": set(), "vintage_b": set()})
        data["vintage_a"].add(row.vintage_a_code)
        data["vintage_b"].add(row.vintage_b_code)

    components = []
    for root, data in component_data.items():
        vintage_a_codes = sorted(data["vintage_a"])
        vintage_b_codes = sorted(data["vintage_b"])
        components.append((vintage_a_codes[0], vintage_b_codes[0], root))
    components.sort()

    group_map: Dict[Tuple[str, str], str] = {}
    period = period_edges["period"].iloc[0]
    for idx, (_, _, root) in enumerate(components, start=1):
        group_map[root] = f"{period}_g{idx:06d}"

    group_ids = [group_map[root] for root in edge_roots]
    edges_with_group = period_edges.copy()
    edges_with_group["group_id"] = group_ids
    return edges_with_group


def build_concordance_groups(concordance_edges: pd.DataFrame) -> ConcordanceGroups:
    """Build connected components and ambiguity flags per concordance period."""
    required = {"period", "vintage_a_year", "vintage_b_year", "vintage_a_code", "vintage_b_code"}
    missing = required.difference(concordance_edges.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if concordance_edges.empty:
        edges = pd.DataFrame(
            columns=[
                "period",
                "vintage_a_year",
                "vintage_b_year",
                "vintage_a_code",
                "vintage_b_code",
                "group_id",
            ]
        )
        vintage_a_nodes = pd.DataFrame(
            columns=[
                "period",
                "vintage_a_year",
                "vintage_b_year",
                "vintage_a_code",
                "group_id",
                "n_vintage_b_links",
                "deterministic_a_to_b",
            ]
        )
        vintage_b_nodes = pd.DataFrame(
            columns=[
                "period",
                "vintage_a_year",
                "vintage_b_year",
                "vintage_b_code",
                "group_id",
                "n_vintage_a_links",
                "deterministic_b_to_a",
            ]
        )
        group_summary = pd.DataFrame(
            columns=[
                "period",
                "vintage_a_year",
                "vintage_b_year",
                "group_id",
                "n_vintage_a",
                "n_vintage_b",
                "n_edges",
                "a_to_b_ambiguous",
                "b_to_a_ambiguous",
            ]
        )
        return ConcordanceGroups(
            edges=edges,
            vintage_a_nodes=vintage_a_nodes,
            vintage_b_nodes=vintage_b_nodes,
            group_summary=group_summary,
        )

    concordance_edges = concordance_edges.drop_duplicates(
        subset=["period", "vintage_a_year", "vintage_b_year", "vintage_a_code", "vintage_b_code"]
    ).reset_index(drop=True)

    period_groups = []
    for period, period_edges in concordance_edges.groupby("period", sort=True):
        period_groups.append(_group_ids_for_period(period_edges))

    edges_with_group = pd.concat(period_groups, ignore_index=True)

    vintage_a_nodes = (
        edges_with_group.groupby(
            ["period", "vintage_a_year", "vintage_b_year", "vintage_a_code", "group_id"], as_index=False
        )
        .agg(n_vintage_b_links=("vintage_b_code", "nunique"))
        .sort_values(["period", "vintage_a_code"])
        .reset_index(drop=True)
    )
    vintage_a_nodes["deterministic_a_to_b"] = vintage_a_nodes["n_vintage_b_links"] == 1

    vintage_b_nodes = (
        edges_with_group.groupby(
            ["period", "vintage_a_year", "vintage_b_year", "vintage_b_code", "group_id"], as_index=False
        )
        .agg(n_vintage_a_links=("vintage_a_code", "nunique"))
        .sort_values(["period", "vintage_b_code"])
        .reset_index(drop=True)
    )
    vintage_b_nodes["deterministic_b_to_a"] = vintage_b_nodes["n_vintage_a_links"] == 1

    a_flags = (
        vintage_a_nodes.groupby(["period", "vintage_a_year", "vintage_b_year", "group_id"], as_index=False)
        .agg(max_n_vintage_b_links=("n_vintage_b_links", "max"))
    )
    b_flags = (
        vintage_b_nodes.groupby(["period", "vintage_a_year", "vintage_b_year", "group_id"], as_index=False)
        .agg(max_n_vintage_a_links=("n_vintage_a_links", "max"))
    )

    group_summary = (
        edges_with_group.groupby(["period", "vintage_a_year", "vintage_b_year", "group_id"], as_index=False)
        .agg(
            n_vintage_a=("vintage_a_code", "nunique"),
            n_vintage_b=("vintage_b_code", "nunique"),
            n_edges=("vintage_b_code", "count"),
        )
        .merge(a_flags, on=["period", "vintage_a_year", "vintage_b_year", "group_id"], how="left")
        .merge(b_flags, on=["period", "vintage_a_year", "vintage_b_year", "group_id"], how="left")
    )
    group_summary["a_to_b_ambiguous"] = group_summary["max_n_vintage_b_links"] > 1
    group_summary["b_to_a_ambiguous"] = group_summary["max_n_vintage_a_links"] > 1
    group_summary = group_summary.drop(columns=["max_n_vintage_b_links", "max_n_vintage_a_links"]).sort_values(
        ["period", "group_id"]
    )

    return ConcordanceGroups(
        edges=edges_with_group,
        vintage_a_nodes=vintage_a_nodes,
        vintage_b_nodes=vintage_b_nodes,
        group_summary=group_summary.reset_index(drop=True),
    )
