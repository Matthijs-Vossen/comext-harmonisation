import pandas as pd
import pytest

from comext_harmonisation.application import (
    apply_chained_weights_wide_for_range,
    apply_weights_to_annual_period,
    finalize_weights_table,
)
from comext_harmonisation.estimation.chaining import ChainedWeightsOutput
from comext_harmonisation.weights import WEIGHT_COLUMNS


def _write_weights(tmp_path, *, period, direction, measure_tag, rows):
    weights_dir = tmp_path / "weights"
    weights_path = weights_dir / period / direction / measure_tag
    weights_path.mkdir(parents=True, exist_ok=True)
    amb_path = weights_path / "weights_ambiguous.csv"
    det_path = weights_path / "weights_deterministic.csv"
    df = pd.DataFrame(rows)[WEIGHT_COLUMNS]
    df.to_csv(amb_path, index=False)
    pd.DataFrame(columns=WEIGHT_COLUMNS).to_csv(det_path, index=False)
    return weights_dir


def _write_annual(tmp_path, year, rows):
    annual_dir = tmp_path / "annual"
    annual_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(annual_dir / f"comext_{year}.parquet", index=False)
    return annual_dir


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_apply_weights_annual_value_strategy(tmp_path):
    period = "20092010"
    direction = "a_to_b"
    annual_dir = _write_annual(
        tmp_path,
        "2009",
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "PERIOD": 2009,
                "VALUE_EUR": 100.0,
                "QUANTITY_KG": 10.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "PERIOD": 2009,
                "VALUE_EUR": 50.0,
                "QUANTITY_KG": 5.0,
            },
        ],
    )
    weights_dir = _write_weights(
        tmp_path,
        period=period,
        direction=direction,
        measure_tag="value_eur",
        rows=[
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000001",
                "to_code": "00000011",
                "group_id": "g1",
                "weight": 1.0,
            },
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000002",
                "to_code": "00000021",
                "group_id": "g2",
                "weight": 0.6,
            },
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000002",
                "to_code": "00000022",
                "group_id": "g2",
                "weight": 0.4,
            },
        ],
    )

    diagnostics = apply_weights_to_annual_period(
        period=period,
        direction=direction,
        strategy="weights_value",
        annual_base_dir=annual_dir,
        weights_dir=weights_dir,
        output_base_dir=tmp_path / "out",
        finalize_weights=True,
        assume_identity_for_missing=False,
    )

    output_path = tmp_path / "out" / "CN2010" / "annual" / "comext_2009_weights_value.parquet"
    result = pd.read_parquet(output_path).sort_values(["PRODUCT_NC"]).reset_index(drop=True)

    assert diagnostics.n_rows_input == 2
    assert diagnostics.n_rows_output == 3
    assert result.loc[0, "PRODUCT_NC"] == "00000011"
    assert result.loc[0, "VALUE_EUR"] == 100.0
    assert result.loc[0, "QUANTITY_KG"] == 10.0
    assert result.loc[1, "VALUE_EUR"] == 30.0
    assert result.loc[1, "QUANTITY_KG"] == 3.0
    assert result.loc[2, "VALUE_EUR"] == 20.0
    assert result.loc[2, "QUANTITY_KG"] == 2.0


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_apply_weights_annual_split_strategy(tmp_path):
    period = "20092010"
    direction = "a_to_b"
    annual_dir = _write_annual(
        tmp_path,
        "2009",
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "PERIOD": 2009,
                "VALUE_EUR": 50.0,
                "QUANTITY_KG": 5.0,
            },
        ],
    )
    _write_weights(
        tmp_path,
        period=period,
        direction=direction,
        measure_tag="value_eur",
        rows=[
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000002",
                "to_code": "00000021",
                "group_id": "g2",
                "weight": 0.6,
            },
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000002",
                "to_code": "00000022",
                "group_id": "g2",
                "weight": 0.4,
            },
        ],
    )
    weights_dir = _write_weights(
        tmp_path,
        period=period,
        direction=direction,
        measure_tag="quantity_kg",
        rows=[
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000002",
                "to_code": "00000021",
                "group_id": "g2",
                "weight": 0.2,
            },
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000002",
                "to_code": "00000022",
                "group_id": "g2",
                "weight": 0.8,
            },
        ],
    )

    apply_weights_to_annual_period(
        period=period,
        direction=direction,
        strategy="weights_split",
        annual_base_dir=annual_dir,
        weights_dir=weights_dir,
        output_base_dir=tmp_path / "out",
        finalize_weights=True,
        assume_identity_for_missing=False,
    )

    output_path = tmp_path / "out" / "CN2010" / "annual" / "comext_2009_weights_split.parquet"
    result = pd.read_parquet(output_path).sort_values(["PRODUCT_NC"]).reset_index(drop=True)
    assert result.loc[0, "VALUE_EUR"] == 30.0
    assert result.loc[1, "VALUE_EUR"] == 20.0
    assert result.loc[0, "QUANTITY_KG"] == 1.0
    assert result.loc[1, "QUANTITY_KG"] == 4.0


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_apply_weights_annual_missing_weights_raises(tmp_path):
    period = "20092010"
    direction = "a_to_b"
    annual_dir = _write_annual(
        tmp_path,
        "2009",
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "PERIOD": 2009,
                "VALUE_EUR": 100.0,
                "QUANTITY_KG": 10.0,
            },
        ],
    )
    weights_dir = _write_weights(
        tmp_path,
        period=period,
        direction=direction,
        measure_tag="value_eur",
        rows=[
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000002",
                "to_code": "00000021",
                "group_id": "g2",
                "weight": 1.0,
            },
        ],
    )

    with pytest.raises(ValueError):
        apply_weights_to_annual_period(
            period=period,
            direction=direction,
            strategy="weights_value",
            annual_base_dir=annual_dir,
            weights_dir=weights_dir,
            output_base_dir=tmp_path / "out",
            finalize_weights=True,
            assume_identity_for_missing=False,
        )


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_apply_weights_annual_missing_weights_identity_default(tmp_path):
    period = "20092010"
    direction = "a_to_b"
    annual_dir = _write_annual(
        tmp_path,
        "2009",
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "PERIOD": 2009,
                "VALUE_EUR": 100.0,
                "QUANTITY_KG": 10.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "PERIOD": 2009,
                "VALUE_EUR": 50.0,
                "QUANTITY_KG": 5.0,
            },
        ],
    )
    weights_dir = _write_weights(
        tmp_path,
        period=period,
        direction=direction,
        measure_tag="value_eur",
        rows=[
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000001",
                "to_code": "00000011",
                "group_id": "g1",
                "weight": 1.0,
            },
        ],
    )

    apply_weights_to_annual_period(
        period=period,
        direction=direction,
        strategy="weights_value",
        annual_base_dir=annual_dir,
        weights_dir=weights_dir,
        output_base_dir=tmp_path / "out",
        assume_identity_for_missing=True,
    )

    output_path = tmp_path / "out" / "CN2010" / "annual" / "comext_2009_weights_value.parquet"
    result = pd.read_parquet(output_path).sort_values(["PRODUCT_NC"]).reset_index(drop=True)
    assert result.loc[0, "PRODUCT_NC"] == "00000002"
    assert result.loc[1, "PRODUCT_NC"] == "00000011"
    assert result.loc[0, "VALUE_EUR"] == 50.0
    assert result.loc[0, "QUANTITY_KG"] == 5.0


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_finalize_weights_table_clamps_and_renormalizes():
    weights = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.999},
            {"from_code": "00000001", "to_code": "00000012", "weight": 5e-11},
            {"from_code": "00000002", "to_code": "00000021", "weight": 0.5},
            {"from_code": "00000002", "to_code": "00000022", "weight": 0.5},
            {"from_code": "00000002", "to_code": "00000023", "weight": -0.0000005},
        ]
    )

    finalized = finalize_weights_table(weights, neg_tol=1e-6, pos_tol=1e-10, row_sum_tol=1e-9)
    sums = finalized.groupby("from_code")["weight"].sum()
    assert pytest.approx(sums.loc["00000001"], abs=1e-12) == 1.0
    assert pytest.approx(sums.loc["00000002"], abs=1e-12) == 1.0
    assert (finalized["weight"] > 0).all()
    assert ("00000012" not in finalized["to_code"].tolist())
    assert ("00000023" not in finalized["to_code"].tolist())


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_finalize_weights_table_raises_on_large_negative():
    weights = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.7},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.3},
            {"from_code": "00000001", "to_code": "00000013", "weight": -1e-5},
        ]
    )

    with pytest.raises(ValueError, match="below -neg_tol"):
        finalize_weights_table(weights, neg_tol=1e-6, pos_tol=0.0, row_sum_tol=1e-9)


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_finalize_weights_table_clamps_small_negative():
    weights = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.6},
            {"from_code": "00000001", "to_code": "00000012", "weight": 0.4},
            {"from_code": "00000001", "to_code": "00000013", "weight": -5e-7},
        ]
    )

    finalized = finalize_weights_table(weights, neg_tol=1e-6, pos_tol=0.0, row_sum_tol=1e-9)
    sums = finalized.groupby("from_code")["weight"].sum()
    assert pytest.approx(sums.loc["00000001"], abs=1e-12) == 1.0
    assert (finalized["weight"] >= 0).all()
    assert ("00000013" not in finalized["to_code"].tolist())


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_apply_chained_wide_finalizes_before_use(tmp_path):
    annual_dir = _write_annual(
        tmp_path,
        "2009",
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "PERIOD": 2009,
                "VALUE_EUR": 100.0,
                "QUANTITY_KG": 10.0,
            },
        ],
    )

    weights = pd.DataFrame(
        [
            {"from_code": "00000001", "to_code": "00000011", "weight": 0.99999999995},
            {"from_code": "00000001", "to_code": "00000012", "weight": 5e-11},
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "origin_year": "2009",
                "target_year": "2010",
                "direction": "a_to_b",
                "measure": "VALUE_EUR",
                "n_unresolved_revised_step_missing": 0,
                "n_unresolved_revised_missing_mid": 0,
                "n_unresolved_revised_total": 0,
            }
        ]
    )
    chained_outputs = [
        ChainedWeightsOutput(
            origin_year="2009",
            target_year="2010",
            direction="a_to_b",
            measure="VALUE_EUR",
            weights=weights,
            diagnostics=diagnostics,
            weights_path=tmp_path / "chain" / "weights.csv",
            diagnostics_path=tmp_path / "chain" / "diagnostics.csv",
        )
    ]

    apply_chained_weights_wide_for_range(
        start_year=2009,
        end_year=2009,
        target_year=2010,
        measures=["VALUE_EUR"],
        annual_base_dir=annual_dir,
        output_base_dir=tmp_path / "out",
        chained_outputs=chained_outputs,
        fail_on_missing=True,
        strict_revised_link_validation=True,
    )

    output_path = tmp_path / "out" / "CN2010" / "annual" / "comext_2009_wide.parquet"
    result = pd.read_parquet(output_path).sort_values("PRODUCT_NC").reset_index(drop=True)
    assert len(result) == 1
    assert result.loc[0, "PRODUCT_NC"] == "00000011"
    assert result.loc[0, "VALUE_EUR_w_value"] == 100.0
    assert result.loc[0, "QUANTITY_KG_w_value"] == 10.0


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_apply_chained_wide_external_unresolved_raises_and_writes_details(tmp_path):
    annual_dir = _write_annual(
        tmp_path,
        "2009",
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "PERIOD": 2009,
                "VALUE_EUR": 100.0,
                "QUANTITY_KG": 10.0,
            },
        ],
    )

    weights = pd.DataFrame([{"from_code": "00000001", "to_code": "00000011", "weight": 1.0}])
    diagnostics = pd.DataFrame(
        [
            {
                "origin_year": "2009",
                "target_year": "2010",
                "direction": "a_to_b",
                "measure": "VALUE_EUR",
                "n_unresolved_revised_step_missing": 1,
                "n_unresolved_revised_missing_mid": 0,
                "n_unresolved_revised_total": 1,
            }
        ]
    )
    chained_outputs = [
        ChainedWeightsOutput(
            origin_year="2009",
            target_year="2010",
            direction="a_to_b",
            measure="VALUE_EUR",
            weights=weights,
            diagnostics=diagnostics,
            weights_path=tmp_path / "chain" / "weights.csv",
            diagnostics_path=tmp_path / "chain" / "diagnostics.csv",
        )
    ]

    with pytest.raises(ValueError, match="Unresolved revised links detected"):
        apply_chained_weights_wide_for_range(
            start_year=2009,
            end_year=2009,
            target_year=2010,
            measures=["VALUE_EUR"],
            annual_base_dir=annual_dir,
            output_base_dir=tmp_path / "out",
            chained_outputs=chained_outputs,
            fail_on_missing=True,
            strict_revised_link_validation=True,
            write_unresolved_details=True,
        )

    unresolved_path = tmp_path / "out" / "CN2010" / "diagnostics" / "unresolved_details.csv"
    details = pd.read_csv(unresolved_path)
    assert "reason" in details.columns
    assert (details["reason"] == "chain_diagnostics_unresolved_revised_total").any()


# LT_REF: Sec3 application equation x_hat = sum_k x * beta
def test_apply_weights_annual_finalizes_even_without_flag(tmp_path):
    period = "20092010"
    direction = "a_to_b"
    annual_dir = _write_annual(
        tmp_path,
        "2009",
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "PERIOD": 2009,
                "VALUE_EUR": 50.0,
                "QUANTITY_KG": 5.0,
            },
        ],
    )
    weights_dir = _write_weights(
        tmp_path,
        period=period,
        direction=direction,
        measure_tag="value_eur",
        rows=[
            {
                "period": period,
                "from_vintage_year": "2009",
                "to_vintage_year": "2010",
                "from_code": "00000002",
                "to_code": "00000021",
                "group_id": "g2",
                "weight": 1.00000000005,
            },
        ],
    )

    diagnostics = apply_weights_to_annual_period(
        period=period,
        direction=direction,
        strategy="weights_value",
        annual_base_dir=annual_dir,
        weights_dir=weights_dir,
        output_base_dir=tmp_path / "out",
        finalize_weights=False,
        assume_identity_for_missing=False,
    )

    output_path = tmp_path / "out" / "CN2010" / "annual" / "comext_2009_weights_value.parquet"
    result = pd.read_parquet(output_path)
    assert diagnostics.n_rows_output == 1
    assert result.loc[0, "PRODUCT_NC"] == "00000021"
    assert result.loc[0, "VALUE_EUR"] == 50.0
