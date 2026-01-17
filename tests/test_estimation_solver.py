import numpy as np
import pandas as pd

from comext_harmonisation.estimation_matrices import build_group_matrices
from comext_harmonisation.estimation_shares import prepare_estimation_shares_from_frames
from comext_harmonisation.estimation_solver import estimate_weights
from comext_harmonisation.groups import build_concordance_groups


def _edge(vintage_a_code, vintage_b_code, period="20002001", vintage_a_year="2000", vintage_b_year="2001"):
    return {
        "period": period,
        "vintage_a_year": vintage_a_year,
        "vintage_b_year": vintage_b_year,
        "vintage_a_code": vintage_a_code,
        "vintage_b_code": vintage_b_code,
    }


def _trade_row(product, value, reporter="NL", partner="BE", trade_type="I", flow="1"):
    return {
        "REPORTER": reporter,
        "PARTNER": partner,
        "TRADE_TYPE": trade_type,
        "PRODUCT_NC": product,
        "FLOW": flow,
        "VALUE_EUR": value,
    }


def test_estimate_weights_simple_split():
    edges = pd.DataFrame(
        [
            _edge("00000001", "00000011"),
            _edge("00000001", "00000012"),
        ]
    )
    groups = build_concordance_groups(edges)

    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 60.0),
            _trade_row("00000001", 40.0, partner="FR"),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 60.0),
            _trade_row("00000011", 40.0, partner="FR"),
        ]
    )

    shares = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )
    matrices = build_group_matrices(shares, groups=groups, dense=False)

    weights, diagnostics = estimate_weights(
        estimation=shares,
        matrices=matrices,
        groups=groups,
        direction="a_to_b",
    )

    assert len(weights) == 2
    weights = weights.sort_values("to_code").reset_index(drop=True)
    assert np.isclose(weights.loc[0, "weight"], 1.0, atol=1e-6)
    assert np.isclose(weights.loc[1, "weight"], 0.0, atol=1e-6)

    assert diagnostics.loc[0, "status"].lower().startswith("solved")
    assert diagnostics.loc[0, "max_row_sum_dev"] < 1e-6


def test_estimate_weights_merge_b_to_a():
    edges = pd.DataFrame(
        [
            _edge("00000001", "00000011"),
            _edge("00000002", "00000011"),
        ]
    )
    groups = build_concordance_groups(edges)

    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 1.0),
            _trade_row("00000001", 1.0, partner="FR"),
            _trade_row("00000002", 3.0),
            _trade_row("00000002", 3.0, partner="FR"),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 1.0),
            _trade_row("00000011", 1.0, partner="FR"),
        ]
    )

    shares = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="b_to_a",
        data_a=data_a,
        data_b=data_b,
    )
    matrices = build_group_matrices(shares, groups=groups, dense=False)

    weights, diagnostics = estimate_weights(
        estimation=shares,
        matrices=matrices,
        groups=groups,
        direction="b_to_a",
    )

    weights = weights.sort_values("to_code").reset_index(drop=True)
    assert np.isclose(weights.loc[0, "weight"], 0.25, atol=1e-5)
    assert np.isclose(weights.loc[1, "weight"], 0.75, atol=1e-5)
    assert diagnostics.loc[0, "status"].lower().startswith("solved")


def test_estimate_weights_mixed_allowed_links():
    edges = pd.DataFrame(
        [
            _edge("00000001", "00000011"),
            _edge("00000001", "00000012"),
            _edge("00000002", "00000012"),
        ]
    )
    groups = build_concordance_groups(edges)

    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 2.0),
            _trade_row("00000002", 1.0),
            _trade_row("00000002", 1.0, partner="FR"),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 30.0),
            _trade_row("00000012", 45.0),
            _trade_row("00000012", 25.0, partner="FR"),
        ]
    )

    shares = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )
    matrices = build_group_matrices(shares, groups=groups, dense=False)

    weights, diagnostics = estimate_weights(
        estimation=shares,
        matrices=matrices,
        groups=groups,
        direction="a_to_b",
    )

    weights = weights.sort_values(["from_code", "to_code"]).reset_index(drop=True)
    assert np.isclose(weights.loc[0, "weight"], 0.6, atol=1e-5)
    assert np.isclose(weights.loc[1, "weight"], 0.4, atol=1e-5)
    assert np.isclose(weights.loc[2, "weight"], 1.0, atol=1e-5)
    assert diagnostics.loc[0, "status"].lower().startswith("solved")


def test_estimate_weights_complex_two_groups_no_deterministic_in_one():
    edges = pd.DataFrame(
        [
            _edge("00000001", "00000011"),
            _edge("00000001", "00000012"),
            _edge("00000002", "00000011"),
            _edge("00000002", "00000012"),
            _edge("00000002", "00000013"),
            _edge("00000003", "00000021"),
            _edge("00000003", "00000022"),
            _edge("00000004", "00000022"),
        ]
    )
    groups = build_concordance_groups(edges)

    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 50.0),
            _trade_row("00000001", 30.0, partner="FR"),
            _trade_row("00000001", 10.0, reporter="DE"),
            _trade_row("00000001", 10.0, reporter="DE", partner="FR"),
            _trade_row("00000002", 10.0),
            _trade_row("00000002", 20.0, partner="FR"),
            _trade_row("00000002", 30.0, reporter="DE"),
            _trade_row("00000002", 40.0, reporter="DE", partner="FR"),
            _trade_row("00000003", 20.0),
            _trade_row("00000003", 10.0, partner="FR"),
            _trade_row("00000003", 20.0, reporter="DE"),
            _trade_row("00000003", 50.0, reporter="DE", partner="FR"),
            _trade_row("00000004", 5.0),
            _trade_row("00000004", 5.0, partner="FR"),
            _trade_row("00000004", 10.0, reporter="DE"),
        ]
    )

    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 32.0),
            _trade_row("00000011", 22.0, partner="FR"),
            _trade_row("00000011", 12.0, reporter="DE"),
            _trade_row("00000011", 14.0, reporter="DE", partner="FR"),
            _trade_row("00000012", 23.0),
            _trade_row("00000012", 18.0, partner="FR"),
            _trade_row("00000012", 13.0, reporter="DE"),
            _trade_row("00000012", 16.0, reporter="DE", partner="FR"),
            _trade_row("00000013", 5.0),
            _trade_row("00000013", 10.0, partner="FR"),
            _trade_row("00000013", 15.0, reporter="DE"),
            _trade_row("00000013", 20.0, reporter="DE", partner="FR"),
            _trade_row("00000021", 14.0),
            _trade_row("00000021", 7.0, partner="FR"),
            _trade_row("00000021", 14.0, reporter="DE"),
            _trade_row("00000021", 35.0, reporter="DE", partner="FR"),
            _trade_row("00000022", 11.0),
            _trade_row("00000022", 8.0, partner="FR"),
            _trade_row("00000022", 16.0, reporter="DE"),
            _trade_row("00000022", 15.0, reporter="DE", partner="FR"),
        ]
    )

    shares = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )
    matrices = build_group_matrices(shares, groups=groups, dense=False)

    weights, diagnostics = estimate_weights(
        estimation=shares,
        matrices=matrices,
        groups=groups,
        direction="a_to_b",
    )

    assert len(weights) == 8
    expected = {
        ("00000001", "00000011"): 0.6,
        ("00000001", "00000012"): 0.4,
        ("00000002", "00000011"): 0.2,
        ("00000002", "00000012"): 0.3,
        ("00000002", "00000013"): 0.5,
        ("00000003", "00000021"): 0.7,
        ("00000003", "00000022"): 0.3,
        ("00000004", "00000022"): 1.0,
    }
    for row in weights.itertuples(index=False):
        key = (row.from_code, row.to_code)
        assert np.isclose(row.weight, expected[key], atol=1e-5)

    assert diagnostics["status"].str.lower().str.startswith("solved").all()
