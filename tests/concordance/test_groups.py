import pandas as pd

from comext_harmonisation.concordance.groups import build_concordance_groups


# LT_REF: Sec3 correspondence graph/group structure
def test_build_concordance_groups_flags():
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
                "vintage_b_code": "00000014",
            },
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000004",
                "vintage_b_code": "00000014",
            },
        ]
    )

    result = build_concordance_groups(df)

    g1 = result.group_summary[
        (result.group_summary["period"] == "20002001")
        & (result.group_summary["group_id"] == "20002001_g000001")
    ].iloc[0]
    assert g1["n_vintage_a"] == 2
    assert g1["n_vintage_b"] == 3
    assert g1["n_edges"] == 4
    assert bool(g1["a_to_b_ambiguous"]) is True
    assert bool(g1["b_to_a_ambiguous"]) is True

    g2 = result.group_summary[
        (result.group_summary["period"] == "20002001")
        & (result.group_summary["group_id"] == "20002001_g000002")
    ].iloc[0]
    assert g2["n_vintage_a"] == 2
    assert g2["n_vintage_b"] == 1
    assert g2["n_edges"] == 2
    assert bool(g2["a_to_b_ambiguous"]) is False
    assert bool(g2["b_to_a_ambiguous"]) is True

    vintage_a_nodes = result.vintage_a_nodes
    row_o1 = vintage_a_nodes[
        (vintage_a_nodes["period"] == "20002001")
        & (vintage_a_nodes["vintage_a_code"] == "00000001")
    ].iloc[0]
    assert row_o1["n_vintage_b_links"] == 2
    assert bool(row_o1["deterministic_a_to_b"]) is False

    row_o2 = vintage_a_nodes[
        (vintage_a_nodes["period"] == "20002001")
        & (vintage_a_nodes["vintage_a_code"] == "00000002")
    ].iloc[0]
    assert row_o2["n_vintage_b_links"] == 2
    assert bool(row_o2["deterministic_a_to_b"]) is False

    vintage_b_nodes = result.vintage_b_nodes
    row_d2 = vintage_b_nodes[
        (vintage_b_nodes["period"] == "20002001")
        & (vintage_b_nodes["vintage_b_code"] == "00000012")
    ].iloc[0]
    assert row_d2["n_vintage_a_links"] == 2
    assert bool(row_d2["deterministic_b_to_a"]) is False

    edges = result.edges
    row_edge = edges[
        (edges["period"] == "20002001")
        & (edges["vintage_a_code"] == "00000003")
        & (edges["vintage_b_code"] == "00000014")
    ].iloc[0]
    assert row_edge["group_id"] == "20002001_g000002"


# LT_REF: Sec3 correspondence graph/group structure
def test_group_flags_pure_split_a_to_b_only():
    df = pd.DataFrame(
        [
            {
                "period": "20102011",
                "vintage_a_year": "2010",
                "vintage_b_year": "2011",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20102011",
                "vintage_a_year": "2010",
                "vintage_b_year": "2011",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000012",
            },
        ]
    )
    result = build_concordance_groups(df)
    group = result.group_summary.iloc[0]
    assert bool(group["a_to_b_ambiguous"]) is True
    assert bool(group["b_to_a_ambiguous"]) is False


# LT_REF: Sec3 correspondence graph/group structure
def test_group_flags_pure_merge_b_to_a_only():
    df = pd.DataFrame(
        [
            {
                "period": "20102011",
                "vintage_a_year": "2010",
                "vintage_b_year": "2011",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20102011",
                "vintage_a_year": "2010",
                "vintage_b_year": "2011",
                "vintage_a_code": "00000002",
                "vintage_b_code": "00000011",
            },
        ]
    )
    result = build_concordance_groups(df)
    group = result.group_summary.iloc[0]
    assert bool(group["a_to_b_ambiguous"]) is False
    assert bool(group["b_to_a_ambiguous"]) is True


# LT_REF: Sec3 correspondence graph/group structure
def test_group_id_stable_under_edge_ordering():
    base = [
        {
            "period": "20102011",
            "vintage_a_year": "2010",
            "vintage_b_year": "2011",
            "vintage_a_code": "00000001",
            "vintage_b_code": "00000011",
        },
        {
            "period": "20102011",
            "vintage_a_year": "2010",
            "vintage_b_year": "2011",
            "vintage_a_code": "00000002",
            "vintage_b_code": "00000012",
        },
        {
            "period": "20102011",
            "vintage_a_year": "2010",
            "vintage_b_year": "2011",
            "vintage_a_code": "00000001",
            "vintage_b_code": "00000012",
        },
    ]
    df_a = pd.DataFrame(base)
    df_b = pd.DataFrame(list(reversed(base)))
    result_a = build_concordance_groups(df_a)
    result_b = build_concordance_groups(df_b)
    edges_a = result_a.edges.sort_values(["vintage_a_code", "vintage_b_code"]).reset_index(drop=True)
    edges_b = result_b.edges.sort_values(["vintage_a_code", "vintage_b_code"]).reset_index(drop=True)
    assert edges_a["group_id"].tolist() == edges_b["group_id"].tolist()


# LT_REF: Sec3 correspondence graph/group structure
def test_group_edge_counts_include_duplicates():
    df = pd.DataFrame(
        [
            {
                "period": "20102011",
                "vintage_a_year": "2010",
                "vintage_b_year": "2011",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
            {
                "period": "20102011",
                "vintage_a_year": "2010",
                "vintage_b_year": "2011",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            },
        ]
    )
    result = build_concordance_groups(df)
    group = result.group_summary.iloc[0]
    assert group["n_edges"] == 1
