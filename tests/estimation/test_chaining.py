import pandas as pd
import pytest

from comext_harmonisation.estimation.chaining import (
    build_chained_weights_for_range,
    chain_weights_for_year,
)


def _write_weights(tmp_path, *, period, direction, measure, rows):
    measure_tag = measure.lower()
    weights_path = tmp_path / period / direction / measure_tag
    weights_path.mkdir(parents=True, exist_ok=True)
    ambiguous_path = weights_path / "weights_ambiguous.csv"
    deterministic_path = weights_path / "weights_deterministic.csv"
    pd.DataFrame(rows).to_csv(ambiguous_path, index=False)
    pd.DataFrame(columns=["from_code", "to_code", "weight"]).to_csv(deterministic_path, index=False)


def _compose(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = left.merge(
        right,
        left_on="to_code",
        right_on="from_code",
        how="inner",
        suffixes=("_left", "_right"),
    )
    merged["weight"] = merged["weight_left"] * merged["weight_right"]
    return (
        merged.groupby(["from_code_left", "to_code_right"], as_index=False)["weight"]
        .sum()
        .rename(columns={"from_code_left": "from_code", "to_code_right": "to_code"})
    )


def _get_output(outputs, *, origin_year, direction, measure):
    for output in outputs:
        if (
            output.origin_year == origin_year
            and output.direction == direction
            and output.measure == measure
        ):
            return output.weights
    raise AssertionError(f"Missing output for {origin_year} {direction} {measure}")


def test_chain_weights_forward(tmp_path):
    _write_weights(
        tmp_path,
        period="20092010",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.6},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.4},
            {"from_code": "00000002", "to_code": "00000012", "weight": 1.0},
        ],
    )
    _write_weights(
        tmp_path,
        period="20102011",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000011", "to_code": "00000021", "weight": 1.0},
            {"from_code": "00000012", "to_code": "00000021", "weight": 0.5},
            {"from_code": "00000012", "to_code": "00000022", "weight": 0.5},
        ],
    )

    weights, diagnostics, direction = chain_weights_for_year(
        origin_year="2009",
        target_year="2011",
        measure="VALUE_EUR",
        weights_dir=tmp_path,
        finalize_weights=False,
    )

    assert direction == "a_to_b"
    weights = weights.sort_values(["from_code", "to_code"]).reset_index(drop=True)
    expected = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000021", "weight": 0.8},
            {"from_code": "00000001", "to_code": "00000022", "weight": 0.2},
            {"from_code": "00000002", "to_code": "00000021", "weight": 0.5},
            {"from_code": "00000002", "to_code": "00000022", "weight": 0.5},
        ]
    ).sort_values(["from_code", "to_code"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(weights, expected)
    assert not diagnostics.empty


def test_chain_weights_missing_raises(tmp_path):
    _write_weights(
        tmp_path,
        period="20092010",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.6},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.4},
            {"from_code": "00000002", "to_code": "00000012", "weight": 1.0},
        ],
    )
    _write_weights(
        tmp_path,
        period="20102011",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000011", "to_code": "00000021", "weight": 1.0},
        ],
    )

    with pytest.raises(ValueError, match="Missing chained weights"):
        chain_weights_for_year(
            origin_year="2009",
            target_year="2011",
            measure="VALUE_EUR",
            weights_dir=tmp_path,
            finalize_weights=False,
            fail_on_missing=True,
        )


def test_chain_weights_backward(tmp_path):
    _write_weights(
        tmp_path,
        period="20112012",
        direction="b_to_a",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000011", "to_code": "00000001", "weight": 0.5},
            {"from_code": "00000011", "to_code": "00000002", "weight": 0.5},
            {"from_code": "00000012", "to_code": "00000002", "weight": 1.0},
        ],
    )
    _write_weights(
        tmp_path,
        period="20102011",
        direction="b_to_a",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000091", "weight": 1.0},
            {"from_code": "00000002", "to_code": "00000092", "weight": 1.0},
        ],
    )

    weights, _, direction = chain_weights_for_year(
        origin_year="2012",
        target_year="2010",
        measure="VALUE_EUR",
        weights_dir=tmp_path,
        finalize_weights=False,
    )

    assert direction == "b_to_a"
    weights = weights.sort_values(["from_code", "to_code"]).reset_index(drop=True)
    expected = pd.DataFrame(
        [
            {"from_code": "00000011", "to_code": "00000091", "weight": 0.5},
            {"from_code": "00000011", "to_code": "00000092", "weight": 0.5},
            {"from_code": "00000012", "to_code": "00000092", "weight": 1.0},
        ]
    ).sort_values(["from_code", "to_code"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(weights, expected)


def test_chain_identity_injection(tmp_path):
    _write_weights(
        tmp_path,
        period="20102011",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 1.0},
            {"from_code": "00000002", "to_code": "00000012", "weight": 1.0},
        ],
    )
    _write_weights(
        tmp_path,
        period="20112012",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000011", "to_code": "00000021", "weight": 1.0},
        ],
    )

    outputs = build_chained_weights_for_range(
        start_year=2010,
        end_year=2012,
        target_year=2012,
        measures=["VALUE_EUR"],
        weights_dir=tmp_path,
        output_weights_dir=tmp_path / "outw",
        output_diagnostics_dir=tmp_path / "outd",
        finalize_weights=False,
    )

    weights = _get_output(outputs, origin_year="2010", direction="a_to_b", measure="VALUE_EUR")
    weights = weights.sort_values(["from_code", "to_code"]).reset_index(drop=True)
    expected = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000021", "weight": 1.0},
            {"from_code": "00000002", "to_code": "00000012", "weight": 1.0},
        ]
    )
    pd.testing.assert_frame_equal(weights[["from_code", "to_code", "weight"]], expected)


def test_chain_caching_consistency(tmp_path):
    _write_weights(
        tmp_path,
        period="20102011",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.6},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.4},
        ],
    )
    _write_weights(
        tmp_path,
        period="20112012",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000011", "to_code": "00000021", "weight": 1.0},
            {"from_code": "00000012", "to_code": "00000022", "weight": 1.0},
        ],
    )

    outputs = build_chained_weights_for_range(
        start_year=2010,
        end_year=2012,
        target_year=2012,
        measures=["VALUE_EUR"],
        weights_dir=tmp_path,
        output_weights_dir=tmp_path / "outw",
        output_diagnostics_dir=tmp_path / "outd",
        finalize_weights=False,
    )

    chain_2011 = _get_output(outputs, origin_year="2011", direction="a_to_b", measure="VALUE_EUR")
    chain_2010 = _get_output(outputs, origin_year="2010", direction="a_to_b", measure="VALUE_EUR")

    step = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.6},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.4},
        ]
    )
    expected = _compose(step, chain_2011)
    chain_2010 = chain_2010[["from_code", "to_code", "weight"]].sort_values(
        ["from_code", "to_code"]
    )
    expected = expected.sort_values(["from_code", "to_code"])
    pd.testing.assert_frame_equal(chain_2010.reset_index(drop=True), expected.reset_index(drop=True))


def test_chain_finalization(tmp_path):
    _write_weights(
        tmp_path,
        period="20102011",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.9995},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.0005},
        ],
    )

    outputs = build_chained_weights_for_range(
        start_year=2010,
        end_year=2011,
        target_year=2011,
        measures=["VALUE_EUR"],
        weights_dir=tmp_path,
        output_weights_dir=tmp_path / "outw",
        output_diagnostics_dir=tmp_path / "outd",
        finalize_weights=True,
        threshold_abs=1e-3,
    )

    weights = _get_output(outputs, origin_year="2010", direction="a_to_b", measure="VALUE_EUR")
    weights = weights.sort_values(["from_code", "to_code"]).reset_index(drop=True)
    expected = pd.DataFrame([{"from_code": "00000001", "to_code": "00000011", "weight": 1.0}])
    pd.testing.assert_frame_equal(weights[["from_code", "to_code", "weight"]], expected)


def test_chain_multi_step_both_directions(tmp_path):
    _write_weights(
        tmp_path,
        period="20102011",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[{"from_code": "00000001", "to_code": "00000011", "weight": 1.0}],
    )
    _write_weights(
        tmp_path,
        period="20112012",
        direction="b_to_a",
        measure="VALUE_EUR",
        rows=[{"from_code": "00000021", "to_code": "00000011", "weight": 1.0}],
    )
    _write_weights(
        tmp_path,
        period="20122013",
        direction="b_to_a",
        measure="VALUE_EUR",
        rows=[{"from_code": "00000031", "to_code": "00000021", "weight": 1.0}],
    )

    outputs = build_chained_weights_for_range(
        start_year=2010,
        end_year=2013,
        target_year=2011,
        measures=["VALUE_EUR"],
        weights_dir=tmp_path,
        output_weights_dir=tmp_path / "outw",
        output_diagnostics_dir=tmp_path / "outd",
        finalize_weights=False,
    )

    weights_2010 = _get_output(outputs, origin_year="2010", direction="a_to_b", measure="VALUE_EUR")
    weights_2012 = _get_output(outputs, origin_year="2012", direction="b_to_a", measure="VALUE_EUR")
    weights_2013 = _get_output(outputs, origin_year="2013", direction="b_to_a", measure="VALUE_EUR")

    assert not weights_2010.empty
    assert not weights_2012.empty
    weights_2013 = weights_2013[["from_code", "to_code", "weight"]].reset_index(drop=True)
    expected = pd.DataFrame([{"from_code": "00000031", "to_code": "00000011", "weight": 1.0}])
    pd.testing.assert_frame_equal(weights_2013, expected)
