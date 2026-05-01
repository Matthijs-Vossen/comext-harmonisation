import pandas as pd

from comext_harmonisation.concordance.groups import build_concordance_groups
from comext_harmonisation.concordance.mappings import get_ambiguous_group_summary
from comext_harmonisation.analysis.share_stability import runner as ss_runner
from comext_harmonisation.analysis.stress_test import runner as st_runner


def test_ambiguous_group_summary_direction_flags():
    edges = pd.DataFrame(
        {
            "period": ["20062007", "20062007", "20062007"],
            "vintage_a_year": ["2006", "2006", "2006"],
            "vintage_b_year": ["2007", "2007", "2007"],
            "vintage_a_code": ["A", "C", "C"],
            "vintage_b_code": ["A", "D", "E"],
        }
    )
    groups = build_concordance_groups(edges)

    summary_a = get_ambiguous_group_summary(groups, "a_to_b")
    summary_b = get_ambiguous_group_summary(groups, "b_to_a")

    assert len(summary_a) == 1
    assert len(summary_b) == 0


def test_unstable_codes_from_edges():
    edges = pd.DataFrame(
        {
            "vintage_a_code": ["A", "B", "D", "D"],
            "vintage_b_code": ["A", "C", "E", "F"],
        }
    )
    unstable_a, unstable_b = ss_runner._unstable_codes_from_edges(edges)

    assert unstable_a == {"B", "D"}
    assert unstable_b == {"C", "E", "F"}


def test_build_chain_group_map_sample_codes():
    edges = pd.DataFrame(
        {
            "period": ["20062007", "20062007"],
            "vintage_a_year": ["2006", "2006"],
            "vintage_b_year": ["2007", "2007"],
            "vintage_a_code": ["A", "A"],
            "vintage_b_code": ["B", "C"],
        }
    )
    groups = build_concordance_groups(edges)

    weights_by_year = {
        "2006": pd.DataFrame(
            {
                "from_code": ["A"],
                "to_code": ["B"],
                "weight": [1.0],
            }
        )
    }

    group_map, group_ids, sample_codes = st_runner._build_chain_group_map(
        groups=groups,
        base_year=2006,
        target_year=2007,
        weights_by_year=weights_by_year,
    )

    assert sample_codes == {"B", "C"}
    assert len(group_ids) == 1
    assert set(group_map["target_code"].tolist()) == {"B", "C"}


def test_complete_lineage_maps_allow_renames_and_drop_non_bijective_groups():
    edges = pd.DataFrame(
        {
            "period": ["20052006", "20052006", "20052006"],
            "vintage_a_year": ["2005", "2005", "2005"],
            "vintage_b_year": ["2006", "2006", "2006"],
            "vintage_a_code": ["A_OLD", "C_OLD", "D_OLD"],
            "vintage_b_code": ["A_NEW", "C_NEW", "C_NEW"],
        }
    )
    groups = build_concordance_groups(edges)
    one_to_one = ss_runner._strict_one_to_one_maps(groups)
    revised_a, revised_b = ss_runner._period_revised_code_sets(groups)
    base_map = pd.DataFrame(
        {
            "group_id": ["g_keep", "g_drop"],
            "target_code": ["A_NEW", "C_NEW"],
        }
    )

    maps_by_year, retained, diagnostics = ss_runner._complete_lineage_maps(
        base_map=base_map,
        base_year=2006,
        target_years=[2005, 2006],
        one_to_one_maps=one_to_one,
        revised_a_by_period=revised_a,
        revised_b_by_period=revised_b,
    )

    assert retained == {"g_keep"}
    assert diagnostics["n_groups_dropped_lineage"] == 1
    lineage_2005 = maps_by_year[2005]
    assert lineage_2005[["group_id", "lineage_code", "native_code"]].to_dict("records") == [
        {"group_id": "g_keep", "lineage_code": "A_NEW", "native_code": "A_OLD"}
    ]
