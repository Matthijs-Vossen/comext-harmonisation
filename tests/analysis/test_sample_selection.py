import pandas as pd

from comext_harmonisation.groups import build_concordance_groups
from comext_harmonisation.mappings import get_ambiguous_group_summary
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
