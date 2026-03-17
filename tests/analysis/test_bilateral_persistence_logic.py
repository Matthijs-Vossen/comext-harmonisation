import numpy as np
import pandas as pd

from comext_harmonisation.analysis.bilateral_persistence import runner as bp_runner
from comext_harmonisation.concordance.groups import build_concordance_groups


def test_strict_one_to_one_maps_keep_only_bijective_links() -> None:
    edges = pd.DataFrame(
        {
            "period": ["20042005"] * 5,
            "vintage_a_year": ["2004"] * 5,
            "vintage_b_year": ["2005"] * 5,
            "vintage_a_code": ["A", "C", "C", "F", "H"],
            "vintage_b_code": ["B", "D", "E", "G", "G"],
        }
    )
    groups = build_concordance_groups(edges)

    maps = bp_runner._strict_one_to_one_maps(groups)

    assert maps[("20042005", "a_to_b")].to_dict("records") == [
        {"from_code": "A", "to_code": "B"}
    ]
    assert maps[("20042005", "b_to_a")].to_dict("records") == [
        {"from_code": "B", "to_code": "A"}
    ]


def test_scaled_bilateral_flows_use_global_group_total() -> None:
    frame = pd.DataFrame(
        {
            "REPORTER": ["R1", "R2", "R1"],
            "PARTNER": ["P1", "P1", "P2"],
            "concept_code": ["B", "B", "C"],
            "value": [30.0, 20.0, 50.0],
        }
    )
    group_map = pd.DataFrame(
        {
            "concept_code": ["B", "C"],
            "group_id": ["g1", "g1"],
        }
    )

    scaled = bp_runner._positive_scaled_flows(frame=frame, group_map=group_map)

    assert np.isclose(float(scaled["group_total"].iloc[0]), 100.0)
    assert np.isclose(
        scaled.loc[
            (scaled["REPORTER"] == "R1") & (scaled["PARTNER"] == "P1") & (scaled["concept_code"] == "B"),
            "scaled_flow",
        ].iloc[0],
        0.30,
    )


def test_lt_no_constant_hc1_returns_expected_beta() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])

    beta, se = bp_runner._lt_no_constant_hc1(x, y)

    assert np.isclose(beta, 2.0)
    assert np.isclose(se, 0.0)


def test_non_bijective_codes_allow_pure_bijective_renames() -> None:
    edges = pd.DataFrame(
        {
            "vintage_a_code": ["A", "B", "B", "D"],
            "vintage_b_code": ["C", "E", "F", "G"],
        }
    )

    unstable_a, unstable_b = bp_runner._non_bijective_codes_from_edges(edges)

    assert unstable_a == {"B"}
    assert unstable_b == {"E", "F"}


def test_select_adjusted_group_ids_union_covers_both_ambiguity_directions() -> None:
    edges = pd.DataFrame(
        {
            "period": ["20062007"] * 4,
            "vintage_a_year": ["2006"] * 4,
            "vintage_b_year": ["2007"] * 4,
            "vintage_a_code": ["A1", "A1", "C1", "C2"],
            "vintage_b_code": ["B1", "B2", "D1", "D1"],
        }
    )
    groups = build_concordance_groups(edges)

    union_ids = bp_runner._select_adjusted_group_ids(
        groups,
        break_period="20062007",
        direction="union",
    )
    a_to_b_ids = bp_runner._select_adjusted_group_ids(
        groups,
        break_period="20062007",
        direction="a_to_b",
    )
    b_to_a_ids = bp_runner._select_adjusted_group_ids(
        groups,
        break_period="20062007",
        direction="b_to_a",
    )

    assert len(union_ids) == 2
    assert union_ids == (a_to_b_ids | b_to_a_ids)


def test_collect_contaminated_group_ids_ignores_bijective_renames_for_candidate_subset() -> None:
    edges = pd.DataFrame(
        {
            "period": ["20052006"] * 3 + ["20062007"] * 3,
            "vintage_a_year": ["2005"] * 3 + ["2006"] * 3,
            "vintage_b_year": ["2006"] * 3 + ["2007"] * 3,
            "vintage_a_code": ["P1", "P1", "R1", "A1", "A1", "A2"],
            "vintage_b_code": ["A1", "A2", "S1", "B1", "B2", "B2"],
        }
    )
    groups = build_concordance_groups(edges)
    one_to_one_maps = bp_runner._strict_one_to_one_maps(groups)
    revised_a_by_period, revised_b_by_period = bp_runner._period_revised_code_sets(groups)
    adjusted_group_ids = bp_runner._select_adjusted_group_ids(
        groups,
        break_period="20062007",
        direction="union",
    )

    contaminated = bp_runner._collect_contaminated_group_ids(
        groups=groups,
        break_period="20062007",
        filter_years=[2005],
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
        candidate_group_ids=adjusted_group_ids,
    )

    assert contaminated == adjusted_group_ids


def test_zero_completed_panel_keeps_entries_and_exits() -> None:
    lag_positive = pd.DataFrame(
        {
            "REPORTER": ["R1"],
            "PARTNER": ["P1"],
            "group_id": ["g1"],
            "concept_code": ["C1"],
            "value": [10.0],
            "group_total": [10.0],
            "scaled_flow": [1.0],
        }
    )
    year_positive = pd.DataFrame(
        {
            "REPORTER": ["R1"],
            "PARTNER": ["P1"],
            "group_id": ["g1"],
            "concept_code": ["C2"],
            "value": [20.0],
            "group_total": [20.0],
            "scaled_flow": [1.0],
        }
    )
    code_universe = pd.DataFrame(
        {
            "group_id": ["g1", "g1"],
            "concept_code": ["C1", "C2"],
        }
    )

    panel, diagnostics = bp_runner._panel_from_positive_flows(
        lag_positive=lag_positive,
        year_positive=year_positive,
        code_universe=code_universe,
    )

    assert len(panel) == 2
    assert diagnostics["n_positive_lag"] == 1
    assert diagnostics["n_positive_current"] == 1
    assert diagnostics["n_both_positive"] == 0
    assert panel.sort_values("concept_code")["scaled_flow_lag"].tolist() == [1.0, 0.0]
    assert panel.sort_values("concept_code")["scaled_flow_cur"].tolist() == [0.0, 1.0]


def test_build_deterministic_basis_group_map_adds_singletons_outside_break_groups() -> None:
    annual_by_year = {
        2006: pd.DataFrame(
            {
                "REPORTER": ["R1", "R1"],
                "PARTNER": ["P1", "P1"],
                "PRODUCT_NC": ["A1", "X1"],
                "value": [10.0, 20.0],
            }
        )
    }
    break_group_map = pd.DataFrame(
        {
            "concept_code": ["A1"],
            "group_id": ["g_break"],
        }
    )

    group_map, diagnostics = bp_runner._build_deterministic_basis_group_map(
        target_year=2006,
        annual_by_year=annual_by_year,
        break_group_map=break_group_map,
        contaminated_codes=set(),
    )

    assert diagnostics["n_linked_groups_pre_filter"] == 1
    assert diagnostics["n_singleton_groups_pre_filter"] == 1
    assert diagnostics["n_linked_groups"] == 1
    assert diagnostics["n_singleton_groups"] == 1
    assert group_map.sort_values("concept_code").to_dict("records") == [
        {"concept_code": "A1", "group_id": "g_break"},
        {"concept_code": "X1", "group_id": "singleton::2006::X1"},
    ]


def test_collect_contaminated_basis_codes_hits_linked_and_singleton_targets() -> None:
    edges = pd.DataFrame(
        {
            "period": ["20042005", "20042005", "20052006", "20052006", "20052006"],
            "vintage_a_year": ["2004", "2004", "2005", "2005", "2005"],
            "vintage_b_year": ["2005", "2005", "2006", "2006", "2006"],
            "vintage_a_code": ["P1", "P1", "A1", "A1", "S1"],
            "vintage_b_code": ["A1", "X1", "B1", "B2", "T1"],
        }
    )
    groups = build_concordance_groups(edges)
    one_to_one_maps = bp_runner._strict_one_to_one_maps(groups)
    revised_a_by_period, revised_b_by_period = bp_runner._period_revised_code_sets(groups)
    contaminated = bp_runner._collect_contaminated_basis_codes(
        break_a_year=2006,
        break_b_year=2007,
        target_year=2006,
        filter_years=[2004, 2005],
        one_to_one_maps=one_to_one_maps,
        revised_a_by_period=revised_a_by_period,
        revised_b_by_period=revised_b_by_period,
        groups=groups,
        candidate_codes={"A1", "X1", "T1"},
    )

    assert contaminated == {"X1"}
