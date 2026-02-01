from pathlib import Path

import numpy as np
import pandas as pd

from comext_harmonisation.analysis.common.metrics import weighted_mean
from comext_harmonisation.analysis.common.steps import compute_step_metrics


class DummyGroups:
    def __init__(self, edges: pd.DataFrame) -> None:
        self.edges = edges


def test_compute_step_metrics_two_step_chain(tmp_path: Path) -> None:
    annual_dir = tmp_path / "annual"
    annual_dir.mkdir()

    df_2000 = pd.DataFrame(
        {
            "REPORTER": ["X", "X"],
            "PARTNER": ["Y", "Y"],
            "PRODUCT_NC": ["A", "B"],
            "VALUE_EUR": [100.0, 50.0],
        }
    )
    df_2001 = pd.DataFrame(
        {
            "REPORTER": ["X", "X"],
            "PARTNER": ["Y", "Y"],
            "PRODUCT_NC": ["C", "D"],
            "VALUE_EUR": [80.0, 70.0],
        }
    )
    df_2002 = pd.DataFrame(
        {
            "REPORTER": ["X", "X"],
            "PARTNER": ["Y", "Y"],
            "PRODUCT_NC": ["E", "F"],
            "VALUE_EUR": [90.0, 60.0],
        }
    )

    df_2000.to_parquet(annual_dir / "comext_2000.parquet", index=False)
    df_2001.to_parquet(annual_dir / "comext_2001.parquet", index=False)
    df_2002.to_parquet(annual_dir / "comext_2002.parquet", index=False)

    edges = pd.DataFrame(
        {
            "period": ["20002001", "20002001", "20002001", "20012002", "20012002"],
            "vintage_a_code": ["A", "A", "B", "C", "D"],
            "vintage_b_code": ["C", "D", "D", "E", "F"],
        }
    )
    groups = DummyGroups(edges)

    weights_dir = tmp_path / "weights"
    for period, direction in [("20002001", "a_to_b"), ("20012002", "a_to_b")]:
        (weights_dir / period / direction / "value_eur").mkdir(parents=True)

    pd.DataFrame(
        {
            "from_code": ["A", "A", "B"],
            "to_code": ["C", "D", "D"],
            "weight": [0.25, 0.75, 1.0],
        }
    ).to_csv(weights_dir / "20002001" / "a_to_b" / "value_eur" / "weights_ambiguous.csv", index=False)
    pd.DataFrame(
        {
            "from_code": ["C", "D"],
            "to_code": ["E", "F"],
            "weight": [1.0, 1.0],
        }
    ).to_csv(weights_dir / "20012002" / "a_to_b" / "value_eur" / "weights_ambiguous.csv", index=False)

    weights_by_year = {
        "2000": pd.DataFrame({"from_code": ["A", "B"], "to_code": ["E", "F"], "weight": [1.0, 1.0]}),
        "2001": pd.DataFrame({"from_code": ["C", "D"], "to_code": ["E", "F"], "weight": [1.0, 1.0]}),
    }

    step_rows = compute_step_metrics(
        base_year=2000,
        target_year=2002,
        sample_target_codes={"E", "F"},
        weights_by_year=weights_by_year,
        groups=groups,
        annual_base_dir=annual_dir,
        measure="VALUE_EUR",
        weights_dir=weights_dir,
        weights_source="VALUE_EUR",
        exclude_reporters=[],
        exclude_partners=[],
        compute_exposure=True,
        compute_diffuseness=True,
    )

    steps_df = pd.DataFrame(step_rows).sort_values("step_index")
    assert len(steps_df) == 2

    exposure_step1 = steps_df.loc[steps_df["step_index"] == 1, "ambiguity_exposure"].iloc[0]
    assert np.isclose(exposure_step1, 100.0 / 150.0)

    exposure_step2 = steps_df.loc[steps_df["step_index"] == 2, "ambiguity_exposure"].iloc[0]
    assert np.isclose(exposure_step2, 0.0)
    ambiguous_trade_step1 = steps_df.loc[steps_df["step_index"] == 1, "ambiguous_trade"].iloc[0]
    assert np.isclose(ambiguous_trade_step1, 100.0)

    exp_vals = steps_df["ambiguity_exposure"].to_numpy(dtype=float)
    exp_w = steps_df["total_trade_sample"].to_numpy(dtype=float)
    weighted = weighted_mean(exp_vals, exp_w)
    expected = (150.0 * (100.0 / 150.0) + 150.0 * 0.0) / (150.0 + 150.0)
    assert np.isclose(weighted, expected)

    step1_entropy = steps_df.loc[steps_df["step_index"] == 1, "diffuseness"].iloc[0]
    step2_entropy = steps_df.loc[steps_df["step_index"] == 2, "diffuseness"].iloc[0]
    assert np.isnan(step2_entropy)

    h_a = -((0.25 * np.log(0.25)) + (0.75 * np.log(0.75))) / np.log(2.0)
    assert np.isclose(step1_entropy, h_a)

    diff_vals = steps_df["diffuseness"].to_numpy(dtype=float)
    diff_w = steps_df["ambiguous_trade"].to_numpy(dtype=float)
    diff_weighted = weighted_mean(diff_vals, diff_w)
    assert np.isclose(diff_weighted, h_a)
