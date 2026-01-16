"""Helpers for selecting ambiguous groups and deterministic mappings."""

from __future__ import annotations

import pandas as pd

from .groups import ConcordanceGroups
from .weights import WEIGHT_COLUMNS


def _validate_direction(direction: str) -> str:
    if direction not in {"a_to_b", "b_to_a"}:
        raise ValueError("direction must be 'a_to_b' or 'b_to_a'")
    return direction


def get_ambiguous_group_summary(groups: ConcordanceGroups, direction: str) -> pd.DataFrame:
    """Return group_summary rows that require estimation for a direction."""
    direction = _validate_direction(direction)
    flag = "a_to_b_ambiguous" if direction == "a_to_b" else "b_to_a_ambiguous"
    return groups.group_summary.loc[groups.group_summary[flag]].reset_index(drop=True)


def get_ambiguous_edges(groups: ConcordanceGroups, direction: str) -> pd.DataFrame:
    """Return concordance edges that belong to ambiguous groups for a direction."""
    summary = get_ambiguous_group_summary(groups, direction)
    if summary.empty:
        return groups.edges.iloc[0:0].copy()
    return groups.edges.merge(summary[["period", "group_id"]], on=["period", "group_id"], how="inner")


def build_deterministic_mappings(groups: ConcordanceGroups, direction: str) -> pd.DataFrame:
    """Build deterministic (weight=1) mappings for a conversion direction."""
    direction = _validate_direction(direction)
    if direction == "a_to_b":
        deterministic_nodes = groups.vintage_a_nodes.loc[
            groups.vintage_a_nodes["deterministic_a_to_b"],
            ["period", "vintage_a_code", "group_id"],
        ]
        edges = groups.edges.merge(
            deterministic_nodes, on=["period", "vintage_a_code", "group_id"], how="inner"
        )
        mappings = edges[
            [
                "period",
                "vintage_a_year",
                "vintage_b_year",
                "vintage_a_code",
                "vintage_b_code",
                "group_id",
            ]
        ].rename(
            columns={
                "vintage_a_year": "from_vintage_year",
                "vintage_b_year": "to_vintage_year",
                "vintage_a_code": "from_code",
                "vintage_b_code": "to_code",
            }
        )
    else:
        deterministic_nodes = groups.vintage_b_nodes.loc[
            groups.vintage_b_nodes["deterministic_b_to_a"],
            ["period", "vintage_b_code", "group_id"],
        ]
        edges = groups.edges.merge(
            deterministic_nodes, on=["period", "vintage_b_code", "group_id"], how="inner"
        )
        mappings = edges[
            [
                "period",
                "vintage_b_year",
                "vintage_a_year",
                "vintage_b_code",
                "vintage_a_code",
                "group_id",
            ]
        ].rename(
            columns={
                "vintage_b_year": "from_vintage_year",
                "vintage_a_year": "to_vintage_year",
                "vintage_b_code": "from_code",
                "vintage_a_code": "to_code",
            }
        )

    mappings = mappings.drop_duplicates(
        subset=["period", "from_vintage_year", "to_vintage_year", "from_code", "to_code", "group_id"]
    ).reset_index(drop=True)
    mappings["weight"] = 1.0
    return mappings[WEIGHT_COLUMNS]
