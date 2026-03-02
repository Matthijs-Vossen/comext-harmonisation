import pandas as pd
import pytest

from comext_harmonisation.estimation.shares import (
    prepare_estimation_shares_for_period,
    prepare_estimation_shares_from_frames,
)
from comext_harmonisation.concordance.groups import build_concordance_groups


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
            {
                "period": "20002001",
                "vintage_a_year": "2000",
                "vintage_b_year": "2001",
                "vintage_a_code": "00000003",
                "vintage_b_code": "00000013",
            },
        ]
    )
    return build_concordance_groups(edges)


def _build_merge_groups():
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
                "vintage_a_code": "00000002",
                "vintage_b_code": "00000011",
            },
        ]
    )
    return build_concordance_groups(edges)


def _build_multi_groups():
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
                "vintage_b_code": "00000013",
            },
        ]
    )
    return build_concordance_groups(edges)


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_zero_group_skipped():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "VALUE_EUR": 100.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "VALUE_EUR": 50.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000013",
                "FLOW": "1",
                "VALUE_EUR": 10.0,
            },
        ]
    )

    result = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )

    assert result.shares_a.empty
    assert result.shares_b.empty
    assert len(result.skipped_groups) == 1
    assert result.skipped_groups.iloc[0]["skip_reason"] == "zero_total_b"


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_excludes_codes():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            {
                "REPORTER": "QP",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": 1,
                "VALUE_EUR": 100.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": 1,
                "VALUE_EUR": 50.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": 1,
                "VALUE_EUR": 75.0,
            },
            {
                "REPORTER": "QP",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000012",
                "FLOW": 1,
                "VALUE_EUR": 25.0,
            },
        ]
    )

    result = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )

    assert set(result.shares_a["REPORTER"].unique()) == {"NL"}
    assert set(result.shares_b["REPORTER"].unique()) == {"NL"}


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_direction_filtering():
    groups = _build_merge_groups()
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
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "VALUE_EUR": 20.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "VALUE_EUR": 30.0,
            },
        ]
    )

    result_a_to_b = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )
    assert result_a_to_b.shares_a.empty
    assert result_a_to_b.shares_b.empty

    result_b_to_a = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="b_to_a",
        data_a=data_a,
        data_b=data_b,
    )
    assert not result_b_to_a.shares_a.empty
    assert not result_b_to_a.shares_b.empty


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_excludes_aggregate_codes_toggle():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            {
                "REPORTER": "XA",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "VALUE_EUR": 10.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "VALUE_EUR": 20.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "XA",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "VALUE_EUR": 15.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000012",
                "FLOW": "1",
                "VALUE_EUR": 25.0,
            },
        ]
    )

    keep_aggregates = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
        exclude_aggregate_codes=False,
        exclude_codes=set(),
    )
    assert set(keep_aggregates.shares_a["REPORTER"].unique()) == {"XA", "NL"}

    drop_aggregates = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
        exclude_aggregate_codes=True,
        exclude_codes=set(),
    )
    assert set(drop_aggregates.shares_a["REPORTER"].unique()) == {"NL"}


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_flow_filtering():
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
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "2",
                "VALUE_EUR": 100.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "VALUE_EUR": 5.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000012",
                "FLOW": "1",
                "VALUE_EUR": 5.0,
            },
        ]
    )

    result = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
        flow="1",
    )

    totals = result.group_totals.iloc[0]
    assert totals["total_value_eur_a"] == 10.0


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_quantity_kg_measure():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "QUANTITY_KG": 12.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "QUANTITY_KG": 7.0,
            },
        ]
    )

    result = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
        measure="QUANTITY_KG",
    )

    totals = result.group_totals.iloc[0]
    assert totals["total_quantity_kg_a"] == 12.0
    assert totals["total_quantity_kg_b"] == 7.0


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_share_sums_to_one():
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
            {
                "REPORTER": "NL",
                "PARTNER": "FR",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "VALUE_EUR": 30.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "VALUE_EUR": 15.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "FR",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000012",
                "FLOW": "1",
                "VALUE_EUR": 25.0,
            },
        ]
    )

    result = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )

    sums_a = result.shares_a.groupby("group_id")["share"].sum().tolist()
    sums_b = result.shares_b.groupby("group_id")["share"].sum().tolist()
    assert all(abs(val - 1.0) < 1e-9 for val in sums_a)
    assert all(abs(val - 1.0) < 1e-9 for val in sums_b)


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_values_match_expected():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000001",
                "FLOW": "1",
                "VALUE_EUR": 100.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "FR",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "VALUE_EUR": 300.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "VALUE_EUR": 200.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "FR",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000012",
                "FLOW": "1",
                "VALUE_EUR": 100.0,
            },
        ]
    )

    result = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )

    shares_a = result.shares_a.set_index(["REPORTER", "PARTNER", "vintage_a_code"])["share"]
    assert abs(shares_a.loc[("NL", "BE", "00000001")] - 0.25) < 1e-9
    assert abs(shares_a.loc[("NL", "FR", "00000002")] - 0.75) < 1e-9

    shares_b = result.shares_b.set_index(["REPORTER", "PARTNER", "vintage_b_code"])["share"]
    assert abs(shares_b.loc[("NL", "BE", "00000011")] - (2.0 / 3.0)) < 1e-9
    assert abs(shares_b.loc[("NL", "FR", "00000012")] - (1.0 / 3.0)) < 1e-9


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_filters_to_ambiguous_groups():
    groups = _build_multi_groups()
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
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000002",
                "FLOW": "1",
                "VALUE_EUR": 20.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "VALUE_EUR": 15.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000012",
                "FLOW": "1",
                "VALUE_EUR": 25.0,
            },
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000013",
                "FLOW": "1",
                "VALUE_EUR": 5.0,
            },
        ]
    )

    result = prepare_estimation_shares_from_frames(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        data_a=data_a,
        data_b=data_b,
    )

    assert set(result.shares_a["group_id"].unique()) == {"20002001_g000001"}


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_invalid_period_raises():
    groups = _build_groups()
    data = pd.DataFrame(
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
    with pytest.raises(ValueError):
        prepare_estimation_shares_from_frames(
            period="2000",
            groups=groups,
            direction="a_to_b",
            data_a=data,
            data_b=data,
        )


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_missing_columns_raises():
    groups = _build_groups()
    data_a = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "FLOW": "1",
                "VALUE_EUR": 10.0,
            },
        ]
    )
    data_b = pd.DataFrame(
        [
            {
                "REPORTER": "NL",
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "VALUE_EUR": 10.0,
            },
        ]
    )
    with pytest.raises(KeyError):
        prepare_estimation_shares_from_frames(
            period="20002001",
            groups=groups,
            direction="a_to_b",
            data_a=data_a,
            data_b=data_b,
        )


# LT_REF: Sec3 scaling of bilateral product shares
def test_prepare_estimation_shares_for_period_reads_parquet(tmp_path):
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
                "PARTNER": "BE",
                "TRADE_TYPE": "I",
                "PRODUCT_NC": "00000011",
                "FLOW": "1",
                "VALUE_EUR": 10.0,
            },
        ]
    )
    data_a.to_parquet(tmp_path / "comext_2000.parquet", index=False)
    data_b.to_parquet(tmp_path / "comext_2001.parquet", index=False)

    result = prepare_estimation_shares_for_period(
        period="20002001",
        groups=groups,
        direction="a_to_b",
        base_dir=tmp_path,
    )

    assert not result.shares_a.empty
    assert not result.shares_b.empty
