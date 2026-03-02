import pandas as pd
import pytest

from comext_harmonisation.chaining.engine import (
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


def _universe(mapping):
    return {int(year): set(codes) for year, codes in mapping.items()}


def _revised(mapping):
    return {
        (str(period), str(direction)): set(codes)
        for (period, direction), codes in mapping.items()
    }


# LT_REF: Sec3 chaining (non-adjacent weight composition)
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
        code_universe=_universe(
            {
                2009: ["00000001", "00000002"],
                2010: ["00000011", "00000012"],
            }
        ),
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


# LT_REF: Sec3 chaining (non-adjacent weight composition)
def test_chain_weights_missing_step_identity_preserved(tmp_path):
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

    weights, _, direction = chain_weights_for_year(
        origin_year="2009",
        target_year="2011",
        measure="VALUE_EUR",
        code_universe=_universe(
            {
                2009: ["00000001", "00000002"],
                2010: ["00000011", "00000012"],
            }
        ),
        weights_dir=tmp_path,
        finalize_weights=False,
        fail_on_missing=True,
    )

    assert direction == "a_to_b"
    weights = weights.sort_values(["from_code", "to_code"]).reset_index(drop=True)
    expected = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.4},
            {"from_code": "00000001", "to_code": "00000021", "weight": 0.6},
            {"from_code": "00000002", "to_code": "00000012", "weight": 1.0},
        ]
    ).sort_values(["from_code", "to_code"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(weights, expected)


# LT_REF: Sec3 chaining (non-adjacent weight composition)
def test_chain_weights_revised_step_missing_raises(tmp_path):
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

    with pytest.raises(ValueError, match="Unresolved revised step links"):
        chain_weights_for_year(
            origin_year="2009",
            target_year="2011",
            measure="VALUE_EUR",
            code_universe=_universe(
                {
                    2009: ["00000001", "00000002"],
                    2010: ["00000011", "00000012"],
                }
            ),
            weights_dir=tmp_path,
            finalize_weights=False,
            fail_on_missing=True,
            revised_codes_by_step=_revised(
                {
                    ("20102011", "a_to_b"): {"00000012"},
                }
            ),
            strict_revised_link_validation=True,
        )


# LT_REF: Sec3 chaining (non-adjacent weight composition)
def test_chain_weights_revised_missing_mid_raises(tmp_path):
    _write_weights(
        tmp_path,
        period="20042005",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.5},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.5},
        ],
    )
    _write_weights(
        tmp_path,
        period="20052006",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000011", "to_code": "00000021", "weight": 1.0},
        ],
    )

    with pytest.raises(ValueError, match="Unresolved revised intermediate links"):
        chain_weights_for_year(
            origin_year="2004",
            target_year="2006",
            measure="VALUE_EUR",
            code_universe=_universe(
                {
                    2004: ["00000001"],
                    2005: ["00000011"],
                }
            ),
            weights_dir=tmp_path,
            finalize_weights=False,
            fail_on_missing=True,
            revised_codes_by_step=_revised(
                {
                    ("20052006", "a_to_b"): {"00000012"},
                }
            ),
            strict_revised_link_validation=True,
        )


# LT_REF: Sec3 chaining (non-adjacent weight composition)
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
        code_universe=_universe(
            {
                2012: ["00000011", "00000012"],
                2011: ["00000001", "00000002"],
            }
        ),
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


# LT_REF: Sec3 chaining (non-adjacent weight composition)
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
        code_universe=_universe(
            {
                2010: ["00000001", "00000002"],
                2011: ["00000011", "00000012"],
                2012: ["00000021"],
            }
        ),
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


# LT_REF: Sec3 chaining (non-adjacent weight composition)
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
        code_universe=_universe(
            {
                2010: ["00000001"],
                2011: ["00000011", "00000012"],
                2012: ["00000021", "00000022"],
            }
        ),
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


# LT_REF: Sec3 chaining (non-adjacent weight composition)
def test_chain_finalization(tmp_path):
    _write_weights(
        tmp_path,
        period="20102011",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.99999999995},
            {"from_code": "00000001", "to_code": "00000012", "weight": 5e-11},
        ],
    )

    outputs = build_chained_weights_for_range(
        start_year=2010,
        end_year=2011,
        target_year=2011,
        measures=["VALUE_EUR"],
        code_universe=_universe(
            {
                2010: ["00000001"],
                2011: ["00000011", "00000012"],
            }
        ),
        weights_dir=tmp_path,
        output_weights_dir=tmp_path / "outw",
        output_diagnostics_dir=tmp_path / "outd",
        finalize_weights=True,
        neg_tol=1e-6,
        pos_tol=1e-10,
    )

    weights = _get_output(outputs, origin_year="2010", direction="a_to_b", measure="VALUE_EUR")
    weights = weights.sort_values(["from_code", "to_code"]).reset_index(drop=True)
    expected = pd.DataFrame([{"from_code": "00000001", "to_code": "00000011", "weight": 1.0}])
    pd.testing.assert_frame_equal(weights[["from_code", "to_code", "weight"]], expected)


# LT_REF: Sec3 chaining (non-adjacent weight composition)
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
        code_universe=_universe(
            {
                2010: ["00000001"],
                2011: ["00000011"],
                2012: ["00000021"],
                2013: ["00000031"],
            }
        ),
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


# LT_REF: Sec3 chaining (non-adjacent weight composition)
def test_chain_missing_identity_drops_forward_codes(tmp_path):
    _write_weights(
        tmp_path,
        period="20042005",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.5},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.5},
        ],
    )
    _write_weights(
        tmp_path,
        period="20052006",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000011", "to_code": "00000021", "weight": 0.5},
            {"from_code": "00000011", "to_code": "00000022", "weight": 0.5},
        ],
    )

    outputs = build_chained_weights_for_range(
        start_year=2004,
        end_year=2006,
        target_year=2006,
        measures=["VALUE_EUR"],
        code_universe=_universe(
            {
                2004: ["00000001"],
                2005: ["00000011", "00000012"],
                2006: ["00000021", "00000022"],
            }
        ),
        weights_dir=tmp_path,
        output_weights_dir=tmp_path / "outw",
        output_diagnostics_dir=tmp_path / "outd",
        finalize_weights=False,
    )

    weights = _get_output(outputs, origin_year="2004", direction="a_to_b", measure="VALUE_EUR")
    weights = weights[["from_code", "to_code", "weight"]].sort_values(
        ["from_code", "to_code"]
    )
    expected = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000021", "weight": 0.25},
            {"from_code": "00000001", "to_code": "00000022", "weight": 0.25},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.5},
        ]
    ).sort_values(["from_code", "to_code"])
    pd.testing.assert_frame_equal(
        weights.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


# LT_REF: Sec3 chaining (non-adjacent weight composition)
def test_chain_missing_identity_drops_backward_codes(tmp_path):
    _write_weights(
        tmp_path,
        period="20062007",
        direction="b_to_a",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000011", "to_code": "00000021", "weight": 0.5},
            {"from_code": "00000011", "to_code": "00000022", "weight": 0.5},
            {"from_code": "00000012", "to_code": "00000023", "weight": 1.0},
        ],
    )
    _write_weights(
        tmp_path,
        period="20072008",
        direction="b_to_a",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.5},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.5},
        ],
    )

    outputs = build_chained_weights_for_range(
        start_year=2006,
        end_year=2008,
        target_year=2006,
        measures=["VALUE_EUR"],
        code_universe=_universe(
            {
                2007: ["00000011", "00000012"],
                2008: ["00000001"],
                2006: ["00000021", "00000022", "00000023"],
            }
        ),
        weights_dir=tmp_path,
        output_weights_dir=tmp_path / "outw",
        output_diagnostics_dir=tmp_path / "outd",
        finalize_weights=False,
    )

    weights = _get_output(outputs, origin_year="2008", direction="b_to_a", measure="VALUE_EUR")
    weights = weights[["from_code", "to_code", "weight"]].sort_values(
        ["from_code", "to_code"]
    )
    expected = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000021", "weight": 0.25},
            {"from_code": "00000001", "to_code": "00000022", "weight": 0.25},
            {"from_code": "00000001", "to_code": "00000023", "weight": 0.5},
        ]
    ).sort_values(["from_code", "to_code"])
    pd.testing.assert_frame_equal(
        weights.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


# LT_REF: Sec3 chaining (non-adjacent weight composition)
def test_chain_missing_step_identity_preserves_universe_code(tmp_path):
    _write_weights(
        tmp_path,
        period="20062007",
        direction="b_to_a",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000012", "to_code": "00000023", "weight": 1.0},
        ],
    )
    _write_weights(
        tmp_path,
        period="20072008",
        direction="b_to_a",
        measure="VALUE_EUR",
        rows=[
            {"from_code": "00000001", "to_code": "00000011", "weight": 1.0},
        ],
    )

    outputs = build_chained_weights_for_range(
        start_year=2006,
        end_year=2008,
        target_year=2006,
        measures=["VALUE_EUR"],
        code_universe=_universe(
            {
                2007: ["00000012"],
                2008: ["00000001", "00000012"],
                2006: ["00000023"],
            }
        ),
        weights_dir=tmp_path,
        output_weights_dir=tmp_path / "outw",
        output_diagnostics_dir=tmp_path / "outd",
        finalize_weights=False,
    )

    weights = _get_output(outputs, origin_year="2008", direction="b_to_a", measure="VALUE_EUR")
    weights = weights[["from_code", "to_code", "weight"]].sort_values(
        ["from_code", "to_code"]
    )
    expected = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000011", "weight": 1.0},
            {"from_code": "00000012", "to_code": "00000023", "weight": 1.0},
        ]
    ).sort_values(["from_code", "to_code"])
    pd.testing.assert_frame_equal(
        weights.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


# LT_REF: Sec3 chaining (non-adjacent weight composition)
def test_chain_unresolved_diagnostics_and_details_written(tmp_path):
    _write_weights(
        tmp_path,
        period="20092010",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[{"from_code": "00000001", "to_code": "00000011", "weight": 1.0}],
    )
    _write_weights(
        tmp_path,
        period="20102011",
        direction="a_to_b",
        measure="VALUE_EUR",
        rows=[{"from_code": "00000011", "to_code": "00000021", "weight": 1.0}],
    )

    outputs = build_chained_weights_for_range(
        start_year=2009,
        end_year=2011,
        target_year=2011,
        measures=["VALUE_EUR"],
        code_universe=_universe(
            {
                2009: ["00000001"],
                2010: ["00000011", "00000012"],
                2011: ["00000021"],
            }
        ),
        weights_dir=tmp_path,
        output_weights_dir=tmp_path / "outw",
        output_diagnostics_dir=tmp_path / "outd",
        finalize_weights=False,
        fail_on_missing=False,
        revised_codes_by_step=_revised(
            {
                ("20102011", "a_to_b"): {"00000012"},
            }
        ),
        strict_revised_link_validation=True,
        write_unresolved_details=True,
    )

    diag_2010 = None
    for output in outputs:
        if output.origin_year == "2010":
            diag_2010 = output.diagnostics.iloc[0]
            break
    assert diag_2010 is not None
    assert "n_unresolved_revised_step_missing" in diag_2010.index
    assert "n_unresolved_revised_missing_mid" in diag_2010.index
    assert "n_unresolved_revised_total" in diag_2010.index
    assert int(diag_2010["n_unresolved_revised_step_missing"]) == 1
    assert int(diag_2010["n_unresolved_revised_total"]) == 1

    unresolved_path = tmp_path / "outd" / "CN2011" / "unresolved_details.csv"
    unresolved = pd.read_csv(unresolved_path, dtype={"code": str})
    unresolved["code"] = unresolved["code"].str.zfill(8)
    assert "code" in unresolved.columns
    assert "reason" in unresolved.columns
    assert ((unresolved["code"] == "00000012") & (unresolved["reason"] == "step_missing_revised")).any()
