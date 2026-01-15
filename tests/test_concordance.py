import pandas as pd
import pytest

from comext_harmonisation.concordance import parse_concordance_df, _normalize_code, _normalize_period


def test_normalize_code_numeric():
    assert _normalize_code(2012011) == "02012011"


def test_normalize_code_string_with_leading_zeros():
    assert _normalize_code("00123456") == "00123456"


def test_normalize_code_non_integer_float_raises():
    with pytest.raises(ValueError):
        _normalize_code(1234.56)


def test_normalize_period_valid():
    period = _normalize_period("19881989")
    assert period.period == "19881989"
    assert period.origin_year == "1988"
    assert period.dest_year == "1989"


def test_parse_concordance_df_dedup_and_format():
    df = pd.DataFrame(
        {
            "Period": [19881989, 19881989, 20012002],
            "Origin code": [2012011, 2012011, 12012013],
            "Destination code": [2012021, 2012021, 12012023],
        }
    )
    parsed = parse_concordance_df(df)
    assert list(parsed.columns) == [
        "period",
        "origin_year",
        "dest_year",
        "origin_code",
        "dest_code",
    ]
    assert len(parsed) == 2
    row = parsed.iloc[0]
    assert row["origin_code"] == "02012011"
    assert row["dest_code"] == "02012021"
    assert row["period"] == "19881989"
    row2 = parsed.iloc[1]
    assert row2["origin_code"] == "12012013"
    assert row2["dest_code"] == "12012023"
    assert row2["period"] == "20012002"
