import numpy as np
import pandas as pd

from comext_harmonisation.analysis.common.metrics import (
    entropy_weighted,
    mae,
    mae_weighted,
    r2_45_weighted,
    r2_45_weighted_symmetric,
    trade_weighted_exposure,
)


def test_trade_weighted_exposure_basic():
    totals = pd.DataFrame(
        {
            "PRODUCT_NC": ["A", "B", "C"],
            "value": [10.0, 30.0, 60.0],
        }
    )
    exposure, ambiguous_trade = trade_weighted_exposure(
        totals=totals, ambiguous_sources={"A", "B"}
    )
    assert np.isclose(ambiguous_trade, 40.0)
    assert np.isclose(exposure, 0.4)


def test_entropy_weighted_basic():
    totals = pd.DataFrame(
        {
            "PRODUCT_NC": ["A", "B"],
            "value": [100.0, 50.0],
        }
    )
    step_weights = pd.DataFrame(
        {
            "from_code": ["A", "A", "B", "B"],
            "to_code": ["x", "y", "x", "y"],
            "weight": [0.25, 0.75, 0.5, 0.5],
        }
    )
    feasible_map = {"A": ["x", "y"], "B": ["x", "y"]}
    ambiguous_sources = {"A", "B"}
    estimable_sources = {"A", "B"}

    h_val, ambiguous_trade = entropy_weighted(
        totals=totals,
        step_weights=step_weights,
        feasible_map=feasible_map,
        ambiguous_sources=ambiguous_sources,
        estimable_sources=estimable_sources,
    )

    h_a = -((0.25 * np.log(0.25)) + (0.75 * np.log(0.75))) / np.log(2.0)
    h_b = -((0.5 * np.log(0.5)) + (0.5 * np.log(0.5))) / np.log(2.0)
    expected = (100.0 * h_a + 50.0 * h_b) / 150.0

    assert np.isclose(ambiguous_trade, 150.0)
    assert np.isclose(h_val, expected)


def test_r2_45_weighted_basic():
    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([0.0, 2.0], dtype=float)
    weights = np.array([1.0, 1.0], dtype=float)
    r2_w = r2_45_weighted(x, y, weights)
    assert np.isclose(r2_w, 0.5)


def test_r2_45_weighted_symmetric_basic():
    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([0.0, 2.0], dtype=float)
    w_initial = np.array([2.0, 0.0], dtype=float)
    w_target = np.array([0.0, 2.0], dtype=float)
    r2_sym = r2_45_weighted_symmetric(x, y, w_initial, w_target)
    assert np.isclose(r2_sym, 0.5)


def test_mae_weighted_basic():
    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([0.0, 2.0], dtype=float)
    w_initial = np.array([2.0, 0.0], dtype=float)
    w_target = np.array([0.0, 2.0], dtype=float)
    mae = mae_weighted(x, y, w_initial, w_target)
    # symmetric weights -> [1,1], |y-x| -> [0,1]
    assert np.isclose(mae, 0.5)


def test_mae_basic():
    x = np.array([0.0, 1.0], dtype=float)
    y = np.array([0.0, 2.0], dtype=float)
    assert np.isclose(mae(x, y), 0.5)


def test_mae_and_mae_weighted_agree_for_equal_weights():
    x = np.array([0.1, 0.4, 0.9], dtype=float)
    y = np.array([0.3, 0.1, 0.6], dtype=float)
    weights = np.ones(3, dtype=float)
    assert np.isclose(mae(x, y), mae_weighted(x, y, weights, weights))
