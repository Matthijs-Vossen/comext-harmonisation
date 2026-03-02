import numpy as np
import pandas as pd

from comext_harmonisation.estimation.matrices import build_group_matrices
from comext_harmonisation.estimation.shares import prepare_estimation_shares_from_frames
from comext_harmonisation.concordance.groups import build_concordance_groups


TRADE_COLS = ["REPORTER", "PARTNER", "TRADE_TYPE", "PRODUCT_NC", "FLOW", "VALUE_EUR"]


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


def _build_groups():
    edges = pd.DataFrame(
        [
            _edge("00000001", "00000011"),
            _edge("00000001", "00000012"),
            _edge("00000002", "00000012"),
        ]
    )
    return build_concordance_groups(edges)


# LT_REF: Sec3 Eq1 (constrained least squares inputs)
def test_build_group_matrices_union_pairs_and_values():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 10.0),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 5.0, partner="FR"),
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


# LT_REF: Sec3 Eq1 (constrained least squares inputs)
def test_build_group_matrices_multiple_groups():
    edges = pd.DataFrame(
        [
            _edge("00000001", "00000011"),
            _edge("00000001", "00000012"),
            _edge("00000002", "00000012"),
            _edge("00000003", "00000013"),
            _edge("00000004", "00000014"),
            _edge("00000005", "00000014"),
        ]
    )
    groups = build_concordance_groups(edges)
    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 10.0),
            _trade_row("00000004", 20.0),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 15.0),
            _trade_row("00000014", 25.0),
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
    assert set(matrices.keys()) == {"20002001_g000001"}


# LT_REF: Sec3 Eq1 (constrained least squares inputs)
def test_build_group_matrices_skips_groups_with_skip_reason():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 10.0),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000013", 5.0),
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
    assert matrices == {}


# LT_REF: Sec3 Eq1 (constrained least squares inputs)
def test_build_group_matrices_empty_shares_returns_empty():
    groups = _build_groups()
    data = pd.DataFrame(columns=TRADE_COLS)
    shares = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data,
        data_b=data,
    )
    matrices = build_group_matrices(shares, groups=groups, dense=True)
    assert matrices == {}


# LT_REF: Sec3 Eq1 (constrained least squares inputs)
def test_build_group_matrices_ordering_is_stable():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            _trade_row("00000002", 20.0, partner="FR"),
            _trade_row("00000001", 10.0),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 5.0, partner="FR"),
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
    pairs = list(zip(group.pairs["REPORTER"], group.pairs["PARTNER"]))
    assert pairs == [("NL", "BE"), ("NL", "FR")]
    assert group.codes_a == ["00000001", "00000002"]
    assert group.codes_b == ["00000011", "00000012"]


# LT_REF: Sec3 Eq1 (constrained least squares inputs)
def test_build_group_matrices_includes_zero_columns_for_missing_codes():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 10.0),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 5.0),
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
    assert np.isclose(group.dense_a["00000002"].sum(), 0.0)
    assert np.isclose(group.dense_b["00000012"].sum(), 0.0)


# LT_REF: Sec3 Eq1 (constrained least squares inputs)
def test_build_group_matrices_sparse_matches_dense():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 10.0),
            _trade_row("00000002", 20.0, partner="FR"),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 5.0),
            _trade_row("00000012", 15.0, partner="FR"),
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
    assert np.allclose(group.matrix_a.toarray(), group.dense_a.values)
    assert np.allclose(group.matrix_b.toarray(), group.dense_b.values)
