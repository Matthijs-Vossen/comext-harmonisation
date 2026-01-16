import pandas as pd
import pytest

from comext_harmonisation.groups import build_concordance_groups
from comext_harmonisation.mappings import (
    build_deterministic_mappings,
    get_ambiguous_edges,
)


def test_get_ambiguous_edges_by_direction():
    df = pd.DataFrame(
        [
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000012",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000002",
                "vintage_b_code": "00000013",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000003",
                "vintage_b_code": "00000013",
            },
        ]
    )
    groups = build_concordance_groups(df)

    a_edges = get_ambiguous_edges(groups, "a_to_b")
    assert set(a_edges["vintage_a_code"].unique()) == {"00000001"}

    b_edges = get_ambiguous_edges(groups, "b_to_a")
    assert set(b_edges["vintage_b_code"].unique()) == {"00000013"}


def test_build_deterministic_mappings_both_directions():
    df = pd.DataFrame(
        [
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000012",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000002",
                "vintage_b_code": "00000012",
            },
        ]
    )
    groups = build_concordance_groups(df)

    a_to_b = build_deterministic_mappings(groups, "a_to_b")
    assert len(a_to_b) == 1
    row = a_to_b.iloc[0]
    assert row["from_code"] == "00000002"
    assert row["to_code"] == "00000012"
    assert row["weight"] == 1.0

    b_to_a = build_deterministic_mappings(groups, "b_to_a")
    assert len(b_to_a) == 1
    row = b_to_a.iloc[0]
    assert row["from_code"] == "00000011"
    assert row["to_code"] == "00000001"
    assert row["weight"] == 1.0


def test_get_ambiguous_edges_none():
    df = pd.DataFrame(
        [
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000002",
                "vintage_b_code": "00000012",
            },
        ]
    )
    groups = build_concordance_groups(df)

    a_edges = get_ambiguous_edges(groups, "a_to_b")
    b_edges = get_ambiguous_edges(groups, "b_to_a")
    assert a_edges.empty
    assert b_edges.empty


def test_build_deterministic_mappings_all_ambiguous():
    df = pd.DataFrame(
        [
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000012",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000002",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000002",
                "vintage_b_code": "00000012",
            },
        ]
    )
    groups = build_concordance_groups(df)

    a_to_b = build_deterministic_mappings(groups, "a_to_b")
    b_to_a = build_deterministic_mappings(groups, "b_to_a")
    assert a_to_b.empty
    assert b_to_a.empty


def test_build_deterministic_mappings_dedup_edges():
    df = pd.DataFrame(
        [
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
        ]
    )
    groups = build_concordance_groups(df)
    mappings = build_deterministic_mappings(groups, "a_to_b")
    assert len(mappings) == 1


def test_get_ambiguous_edges_multiple_periods():
    df = pd.DataFrame(
        [
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000012",
            },
            {
                "period": "20012002",
                "vintage_a_year": "2001",
                "vintage_b_year": "2002",
                "vintage_a_code": "00000002",
                "vintage_b_code": "00000013",
            },
            {
                "period": "20012002",
                "vintage_a_year": "2001",
                "vintage_b_year": "2002",
                "vintage_a_code": "00000003",
                "vintage_b_code": "00000013",
            },
        ]
    )
    groups = build_concordance_groups(df)

    a_edges = get_ambiguous_edges(groups, "a_to_b")
    assert set(a_edges["period"].unique()) == {"20002001"}

    b_edges = get_ambiguous_edges(groups, "b_to_a")
    assert set(b_edges["period"].unique()) == {"20012002"}


def test_invalid_direction_raises():
    df = pd.DataFrame(
        [
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
        ]
    )
    groups = build_concordance_groups(df)
    with pytest.raises(ValueError):
        get_ambiguous_edges(groups, "a_to_c")
    with pytest.raises(ValueError):
        build_deterministic_mappings(groups, "b_to_c")


def test_deterministic_mappings_from_ambiguous_group():
    df = pd.DataFrame(
        [
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000012",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000002",
                "vintage_b_code": "00000012",
            },
        ]
    )
    groups = build_concordance_groups(df)

    a_to_b = build_deterministic_mappings(groups, "a_to_b")
    assert set(a_to_b["from_code"]) == {"00000002"}

    b_to_a = build_deterministic_mappings(groups, "b_to_a")
    assert set(b_to_a["from_code"]) == {"00000011"}
