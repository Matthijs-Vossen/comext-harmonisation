import numpy as np
import pandas as pd

from comext_harmonisation.analysis.common.metrics import (
    entropy_weighted,
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
