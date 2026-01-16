import numpy as np
import pandas as pd

from comext_harmonisation.estimation_matrices import build_group_matrices
from comext_harmonisation.estimation_shares import prepare_estimation_shares_from_frames
from comext_harmonisation.groups import build_concordance_groups


def _build_groups():
    edges = pd.DataFrame(
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
    return build_concordance_groups(edges)


def test_build_group_matrices_union_pairs_and_values():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "VALUE_EUR": 10.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "FR",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "VALUE_EUR": 5.0,
            },
        ]
    )

    shares = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )

    matrices = build_group_matrices(shares, groups=groups, dense=True)
    group = matrices["20002001_g000001"]

    assert group.codes_a == ["00000001", "00000002"]
    assert group.codes_b == ["00000011", "00000012"]
    assert group.matrix_a.shape == (2, 2)
    assert group.matrix_b.shape == (2, 2)

    pairs = list(zip(group.pairs["REPORTER"], group.pairs["PARTNER"]))
    assert pairs == [("NL", "BE"), ("NL", "FR")]

    dense_a = group.dense_a
    dense_b = group.dense_b
    assert dense_a is not None and dense_b is not None

    assert np.isclose(dense_a.loc[("NL", "BE"), "00000001"], 1.0)
    assert np.isclose(dense_a.loc[("NL", "FR"), "00000001"], 0.0)
    assert np.isclose(dense_a.loc[("NL", "BE"), "00000002"], 0.0)

    assert np.isclose(dense_b.loc[("NL", "FR"), "00000011"], 1.0)
    assert np.isclose(dense_b.loc[("NL", "BE"), "00000011"], 0.0)
    assert np.isclose(dense_b.loc[("NL", "FR"), "00000012"], 0.0)
