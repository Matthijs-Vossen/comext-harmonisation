import numpy as np
import pandas as pd

from comext_harmonisation.estimation.matrices import build_group_matrices
from comext_harmonisation.estimation.shares import prepare_estimation_shares_from_frames
from comext_harmonisation.estimation.solver import estimate_weights
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


def _assert_lt_feasibility(
    weights: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    tol: float = 1e-6,
) -> None:
    assert not weights.empty

    # LT constraints: non-negativity and row-sum-to-one per from_code.
    assert (weights["weight"] >= -tol).all()
    row_sums = weights.groupby("from_code")["weight"].sum()
    assert np.allclose(row_sums.values, np.ones_like(row_sums.values), atol=tol)

    # Allowed links only: no (from_code, to_code) outside concordance edges.
    allowed = set(zip(edges["vintage_a_code"], edges["vintage_b_code"]))
    actual = set(zip(weights["from_code"], weights["to_code"]))
    assert actual.issubset(allowed)


def _objective_value(
    *,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    weights: pd.DataFrame,
    codes_a: list[str],
    codes_b: list[str],
) -> float:
    code_a_idx = {code: i for i, code in enumerate(codes_a)}
    code_b_idx = {code: i for i, code in enumerate(codes_b)}
    beta = np.zeros((len(codes_a), len(codes_b)), dtype=float)
    for row in weights.itertuples(index=False):
        beta[code_a_idx[row.from_code], code_b_idx[row.to_code]] = float(row.weight)
    residual = matrix_b - (matrix_a @ beta)
    return float(np.sum(residual**2))


# LT_REF: Sec3 Eq1 (objective + constraints)
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
    _assert_lt_feasibility(weights, edges)

    assert diagnostics.loc[0, "status"].lower().startswith("solved")
    assert diagnostics.loc[0, "max_row_sum_dev"] < 1e-6


# LT_REF: Sec3 Eq1 (objective + constraints)
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
    edges_b_to_a = edges.rename(
        columns={"vintage_b_code": "vintage_a_code", "vintage_a_code": "vintage_b_code"}
    )
    _assert_lt_feasibility(weights, edges_b_to_a)
    assert diagnostics.loc[0, "status"].lower().startswith("solved")


# LT_REF: Sec3 Eq1 (objective + constraints)
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
    _assert_lt_feasibility(weights, edges)
    assert diagnostics.loc[0, "status"].lower().startswith("solved")


# LT_REF: Sec3 Eq1 (objective + constraints)
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
    _assert_lt_feasibility(weights, edges)

    assert diagnostics["status"].str.lower().str.startswith("solved").all()


# LT_REF: Sec3 Eq1 (objective + constraints)
def test_estimate_weights_noisy_non_exact_case_matches_grid_optimum():
    edges = pd.DataFrame(
        [
            _edge("00000001", "00000011"),
            _edge("00000001", "00000012"),
            _edge("00000002", "00000011"),
            _edge("00000002", "00000012"),
        ]
    )
    groups = build_concordance_groups(edges)

    # Construct a deliberately non-exact system: pair totals differ across vintages.
    data_a = pd.DataFrame(
        [
            _trade_row("00000001", 80.0, partner="BE"),
            _trade_row("00000002", 20.0, partner="BE"),
            _trade_row("00000001", 20.0, partner="FR"),
            _trade_row("00000002", 80.0, partner="FR"),
        ]
    )
    data_b = pd.DataFrame(
        [
            _trade_row("00000011", 70.0, partner="BE"),
            _trade_row("00000012", 20.0, partner="BE"),
            _trade_row("00000011", 10.0, partner="FR"),
            _trade_row("00000012", 100.0, partner="FR"),
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

    weights, diagnostics = estimate_weights(
        estimation=shares,
        matrices=matrices,
        groups=groups,
        direction="a_to_b",
    )
    weights = weights.sort_values(["from_code", "to_code"]).reset_index(drop=True)
    _assert_lt_feasibility(weights, edges)
    assert diagnostics.loc[0, "status"].lower().startswith("solved")

    solver_obj = _objective_value(
        matrix_a=group.matrix_a.toarray(),
        matrix_b=group.matrix_b.toarray(),
        weights=weights,
        codes_a=group.codes_a,
        codes_b=group.codes_b,
    )
    assert solver_obj > 0.0

    # Independent coarse grid search over feasible rows:
    # beta[00000001,00000011]=a, beta[00000002,00000011]=b,
    # with row sums forcing complements on to_code 00000012.
    code_a_idx = {code: i for i, code in enumerate(group.codes_a)}
    code_b_idx = {code: i for i, code in enumerate(group.codes_b)}
    i_a1 = code_a_idx["00000001"]
    i_a2 = code_a_idx["00000002"]
    j_b1 = code_b_idx["00000011"]
    j_b2 = code_b_idx["00000012"]

    x = group.matrix_a.toarray()
    y = group.matrix_b.toarray()
    best_grid_obj = float("inf")
    grid = np.linspace(0.0, 1.0, 401)
    for a in grid:
        for b in grid:
            beta = np.zeros((2, 2), dtype=float)
            beta[i_a1, j_b1] = a
            beta[i_a1, j_b2] = 1.0 - a
            beta[i_a2, j_b1] = b
            beta[i_a2, j_b2] = 1.0 - b
            obj = float(np.sum((y - (x @ beta)) ** 2))
            if obj < best_grid_obj:
                best_grid_obj = obj

    assert solver_obj <= best_grid_obj + 1e-5
