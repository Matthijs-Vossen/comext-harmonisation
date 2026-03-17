from pathlib import Path

import pandas as pd

from comext_harmonisation.analysis.bilateral_persistence import runner as bp_runner
from comext_harmonisation.analysis.config import (
    BilateralPersistenceBreakConfig,
    BilateralPersistenceConfig,
    BilateralPersistenceFilterConfig,
    BilateralPersistenceFlowConfig,
    BilateralPersistenceMeasureConfig,
    BilateralPersistenceOutputConfig,
    BilateralPersistencePathsConfig,
    BilateralPersistenceSampleConfig,
    BilateralPersistenceYearsConfig,
)
from comext_harmonisation.concordance.groups import build_concordance_groups


def _make_config(tmp_path: Path) -> BilateralPersistenceConfig:
    output_dir = tmp_path / "outputs"
    return BilateralPersistenceConfig(
        years=BilateralPersistenceYearsConfig(columns=[2005, 2006, 2008, 2009]),
        break_config=BilateralPersistenceBreakConfig(period="20062007", direction="union"),
        measures=BilateralPersistenceMeasureConfig(analysis_measure="VALUE_EUR"),
        flow=BilateralPersistenceFlowConfig(flow_code="1"),
        paths=BilateralPersistencePathsConfig(
            concordance_path=Path("dummy.xls"),
            concordance_sheet=None,
            annual_base_dir=tmp_path / "annual",
            output_dir=output_dir,
        ),
        sample=BilateralPersistenceSampleConfig(exclude_reporters=[], exclude_partners=[]),
        adjusted_filter=BilateralPersistenceFilterConfig(years=[2004, 2005, 2007, 2008]),
        output=BilateralPersistenceOutputConfig(
            table_csv=output_dir / "table.csv",
            table_tex=output_dir / "table.tex",
            details_csv=output_dir / "regression_details.csv",
            sample_diagnostics_csv=output_dir / "sample_diagnostics.csv",
        ),
    )


def test_bilateral_persistence_runner_writes_expected_outputs(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    edges = pd.DataFrame(
        {
            "period": (
                ["20042005"] * 3
                + ["20052006"] * 3
                + ["20062007"] * 4
                + ["20072008"] * 3
                + ["20082009"] * 3
            ),
            "vintage_a_year": (
                ["2004"] * 3
                + ["2005"] * 3
                + ["2006"] * 4
                + ["2007"] * 3
                + ["2008"] * 3
            ),
            "vintage_b_year": (
                ["2005"] * 3
                + ["2006"] * 3
                + ["2007"] * 4
                + ["2008"] * 3
                + ["2009"] * 3
            ),
            "vintage_a_code": [
                "Q1",
                "R1",
                "W1",
                "P1",
                "S1",
                "X1",
                "A1",
                "A1",
                "A2",
                "X1",
                "B1",
                "B2",
                "Y1",
                "C1",
                "C2",
                "Z1",
            ],
            "vintage_b_code": [
                "P1",
                "S1",
                "X1",
                "A1",
                "T1",
                "X1",
                "B1",
                "B2",
                "B2",
                "Y1",
                "C1",
                "C2",
                "Z1",
                "D1",
                "D2",
                "V1",
            ],
        }
    )
    groups = build_concordance_groups(edges)
    monkeypatch.setattr(bp_runner, "load_concordance_groups", lambda **_: groups)

    annual_by_year = {
        2004: pd.DataFrame(
            {
                "REPORTER": ["R1", "R2", "R1", "R1"],
                "PARTNER": ["P", "P", "P", "P"],
                "PRODUCT_NC": ["Q1", "R1", "W1", "K1"],
                "value": [10.0, 5.0, 8.0, 4.0],
            }
        ),
        2005: pd.DataFrame(
            {
                "REPORTER": ["R1", "R2", "R1", "R1"],
                "PARTNER": ["P", "P", "P", "P"],
                "PRODUCT_NC": ["P1", "S1", "X1", "K1"],
                "value": [12.0, 6.0, 9.0, 5.0],
            }
        ),
        2006: pd.DataFrame(
            {
                "REPORTER": ["R1", "R1", "R2", "R2", "R1", "R1"],
                "PARTNER": ["P", "P", "P", "P", "P", "P"],
                "PRODUCT_NC": ["A1", "A2", "A1", "A2", "X1", "K1"],
                "value": [15.0, 25.0, 7.0, 13.0, 11.0, 6.0],
            }
        ),
        2007: pd.DataFrame(
            {
                "REPORTER": ["R1", "R2", "R1", "R1"],
                "PARTNER": ["P", "P", "P", "P"],
                "PRODUCT_NC": ["B1", "B2", "Y1", "K1"],
                "value": [14.0, 14.0, 10.0, 7.0],
            }
        ),
        2008: pd.DataFrame(
            {
                "REPORTER": ["R1", "R2", "R1", "R1"],
                "PARTNER": ["P", "P", "P", "P"],
                "PRODUCT_NC": ["C1", "C2", "Z1", "K1"],
                "value": [13.0, 15.0, 9.0, 8.0],
            }
        ),
        2009: pd.DataFrame(
            {
                "REPORTER": ["R1", "R2", "R1", "R1"],
                "PARTNER": ["P", "P", "P", "P"],
                "PRODUCT_NC": ["D1", "D2", "V1", "K1"],
                "value": [12.0, 16.0, 8.0, 9.0],
            }
        ),
    }

    monkeypatch.setattr(
        bp_runner,
        "_load_native_bilateral_year",
        lambda **kwargs: annual_by_year[int(kwargs["year"])],
    )

    outputs = bp_runner.run_bilateral_persistence_analysis(config)

    assert Path(outputs["table_csv"]).exists()
    assert Path(outputs["details_csv"]).exists()
    assert Path(outputs["sample_diagnostics_csv"]).exists()
    assert config.output.table_tex.exists()

    details = pd.read_csv(outputs["details_csv"])
    assert set(details["row_key"]) == {
        bp_runner.ROW_DETERMINISTIC_ALL,
        bp_runner.ROW_ALL,
        bp_runner.ROW_ADJUSTED,
    }
    assert set(details["year"]) == {2005, 2006, 2008, 2009}
    assert set(details["basis_year"]) == {2006, 2007}
    assert set(details["sample_basis"]) == {
        "break_filtered_deterministic_all",
        "break_filtered_break_groups",
        "break_filtered_adjusted",
    }
    det_details = details.loc[details["row_key"] == bp_runner.ROW_DETERMINISTIC_ALL].sort_values("year")
    assert det_details["basis_year"].tolist() == [2006, 2006, 2007, 2007]
    assert det_details["coef"].notna().all()

    diagnostics = pd.read_csv(outputs["sample_diagnostics_csv"])
    det_groups = (
        diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_DETERMINISTIC_ALL]
        .sort_values("year")["n_groups"]
        .to_numpy()
    )
    broad_groups = (
        diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_ALL]
        .sort_values("year")["n_groups"]
        .to_numpy()
    )
    adjusted_groups = (
        diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_ADJUSTED]
        .sort_values("year")["n_groups"]
        .to_numpy()
    )
    assert (det_groups > broad_groups).all()
    assert (broad_groups >= adjusted_groups).all()
    assert (
        diagnostics.loc[
            diagnostics["row_key"] == bp_runner.ROW_DETERMINISTIC_ALL, "n_groups_pre_filter"
        ]
        .to_numpy()
        >= diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_DETERMINISTIC_ALL, "n_groups"].to_numpy()
    ).all()
    assert (
        diagnostics.loc[
            diagnostics["row_key"] == bp_runner.ROW_DETERMINISTIC_ALL, "n_groups_pre_filter"
        ].to_numpy()
        == diagnostics.loc[
            diagnostics["row_key"] == bp_runner.ROW_DETERMINISTIC_ALL, "n_linked_groups_pre_filter"
        ].to_numpy()
        + diagnostics.loc[
            diagnostics["row_key"] == bp_runner.ROW_DETERMINISTIC_ALL, "n_singleton_groups_pre_filter"
        ].to_numpy()
    ).all()
    assert (
        diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_ALL, "n_groups_pre_filter"]
        .to_numpy()
        > diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_ADJUSTED, "n_groups_pre_filter"]
        .to_numpy()
    ).all()
    assert (
        diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_ALL, "n_groups_contaminated"]
        .to_numpy()
        >= diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_ADJUSTED, "n_groups_contaminated"]
        .to_numpy()
    ).all()
    assert (
        diagnostics["n_cells"] >= diagnostics["n_both_positive"]
    ).all()
    assert (
        diagnostics["n_cells"] > diagnostics["n_positive_lag"]
    ).any()
    assert (
        diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_ALL, "n_groups"]
        < diagnostics.loc[diagnostics["row_key"] == bp_runner.ROW_ALL, "n_concepts"]
    ).all()
    assert (
        diagnostics.loc[
            diagnostics["row_key"] == bp_runner.ROW_DETERMINISTIC_ALL, "n_singleton_groups"
        ]
        > 0
    ).all()
    assert (
        diagnostics.loc[diagnostics["row_key"] != bp_runner.ROW_DETERMINISTIC_ALL, "n_singleton_groups"]
        == 0
    ).all()
    assert set(diagnostics["basis_year"]) == {2006, 2007}
    assert set(diagnostics["sample_basis"]) == {
        "break_filtered_deterministic_all",
        "break_filtered_break_groups",
        "break_filtered_adjusted",
    }

    table = pd.read_csv(outputs["table_csv"])
    assert table["row_label"].tolist() == [
        bp_runner.ROW_LABELS[bp_runner.ROW_DETERMINISTIC_ALL],
        bp_runner.ROW_LABELS[bp_runner.ROW_ALL],
        bp_runner.ROW_LABELS[bp_runner.ROW_ADJUSTED],
    ]
