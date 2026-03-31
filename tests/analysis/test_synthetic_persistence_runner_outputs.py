from pathlib import Path

import pandas as pd

from comext_harmonisation.analysis.config import (
    SyntheticPersistenceCandidatesConfig,
    SyntheticPersistenceChainingConfig,
    SyntheticPersistenceConfig,
    SyntheticPersistenceFlowConfig,
    SyntheticPersistenceMeasureConfig,
    SyntheticPersistencePathsConfig,
    SyntheticPersistencePlotConfig,
    SyntheticPersistenceSampleConfig,
    SyntheticPersistenceYearsConfig,
)
from comext_harmonisation.analysis.synthetic_persistence import runner as sp_runner


def _make_config(tmp_path: Path) -> SyntheticPersistenceConfig:
    output_dir = tmp_path / "outputs"
    return SyntheticPersistenceConfig(
        years=SyntheticPersistenceYearsConfig(
            start=2000,
            end=2002,
            prehistory_anchor=2002,
            afterlife_anchor=2000,
        ),
        measures=SyntheticPersistenceMeasureConfig(
            weights_source="VALUE_EUR",
            analysis_measure="VALUE_EUR",
        ),
        flow=SyntheticPersistenceFlowConfig(mode="imports_only", flow_code="1"),
        candidates=SyntheticPersistenceCandidatesConfig(
            prehistory=["11111111"],
            afterlife=["22222222", "33333333"],
            display_labels={
                "11111111": "Modern widgets",
                "22222222": "Legacy widgets",
                "33333333": "Unused label",
            },
        ),
        paths=SyntheticPersistencePathsConfig(
            concordance_path=Path("dummy.xls"),
            concordance_sheet=None,
            annual_base_dir=tmp_path / "annual",
            weights_dir=tmp_path / "weights",
            output_dir=output_dir,
        ),
        chaining=SyntheticPersistenceChainingConfig(
            finalize_weights=True,
            neg_tol=1e-6,
            pos_tol=1e-10,
            row_sum_tol=1e-6,
            fail_on_missing=True,
            strict_revised_link_validation=False,
            write_unresolved_details=False,
        ),
        sample=SyntheticPersistenceSampleConfig(exclude_reporters=[], exclude_partners=[]),
        plot=SyntheticPersistencePlotConfig(
            summary_output_path=output_dir / "qualitative_summary.png",
            use_latex=False,
            latex_preamble="",
            line_width=1.0,
            point_size=3.0,
            y_axis_unit="percent",
        ),
    )


def test_synthetic_persistence_runner_writes_expected_outputs(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    totals_by_year = {
        2000: pd.DataFrame({"code": ["22222222", "00000010"], "value": [50.0, 50.0]}),
        2001: pd.DataFrame({"code": ["00000010", "00000020"], "value": [80.0, 20.0]}),
        2002: pd.DataFrame({"code": ["11111111", "00000030"], "value": [100.0, 50.0]}),
    }
    total_trade_by_year = {year: float(df["value"].sum()) for year, df in totals_by_year.items()}
    observed_years_by_code = {
        "11111111": [2002],
        "22222222": [2000],
        "00000010": [2000, 2001],
        "00000020": [2001],
        "00000030": [2002],
    }

    monkeypatch.setattr(
        sp_runner,
        "_build_totals_cache",
        lambda _cfg: (totals_by_year, total_trade_by_year, observed_years_by_code),
    )
    monkeypatch.setattr(
        sp_runner,
        "_concordance_timing_by_code",
        lambda **_: (
            {"11111111": 2002, "22222222": 1999, "33333333": 2025},
            {"22222222": 2000},
        ),
    )
    monkeypatch.setattr(sp_runner, "build_code_universe_from_annual", lambda **_: {})

    pre_weights = {
        2000: pd.DataFrame(
            {
                "from_code": ["00000010"],
                "to_code": ["11111111"],
                "weight": [1.0],
            }
        ),
        2001: pd.DataFrame(
            {
                "from_code": ["00000020"],
                "to_code": ["11111111"],
                "weight": [1.0],
            }
        ),
    }
    after_weights = {
        2001: pd.DataFrame(
            {
                "from_code": ["00000010"],
                "to_code": ["22222222"],
                "weight": [0.5],
            }
        ),
        2002: pd.DataFrame(
            {
                "from_code": ["00000030"],
                "to_code": ["22222222"],
                "weight": [0.2],
            }
        ),
    }

    def _stub_build_chain_weights(*, target_year: int, **_kwargs):
        if target_year == 2002:
            return pre_weights
        if target_year == 2000:
            return after_weights
        raise AssertionError("unexpected target year")

    monkeypatch.setattr(sp_runner, "_build_chain_weights", _stub_build_chain_weights)

    def _stub_plot(**kwargs):
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"plot")

    monkeypatch.setattr(sp_runner, "_plot_summary", _stub_plot)

    outputs = sp_runner.run_synthetic_persistence_analysis(config)

    expected_files = [
        "code_catalog_csv",
        "candidate_series_csv",
        "code_evidence_csv",
    ]
    for key in expected_files:
        path = outputs[key]
        assert path.exists(), f"missing output: {path}"

    assert outputs["output_plot"].exists()
    assert "output_plot_examples" not in outputs
    assert "output_plot_basket_composition" not in outputs

    catalog = pd.read_csv(outputs["code_catalog_csv"])
    assert {"set_name", "code", "included_in_series", "exclusion_reason"}.issubset(catalog.columns)
    catalog["code"] = catalog["code"].astype(str).str.zfill(8)
    catalog_labels = dict(zip(catalog["code"], catalog["label"]))
    assert catalog_labels["11111111"] == "Modern widgets"
    assert catalog_labels["22222222"] == "Legacy widgets"

    series = pd.read_csv(outputs["candidate_series_csv"])
    series["code"] = series["code"].astype(str).str.zfill(8)
    assert {
        "dimension",
        "set_name",
        "code",
        "label",
        "year",
        "value_conv",
        "share_conv",
        "is_synthetic_window",
        "is_inlife_window",
    }.issubset(series.columns)
    assert set(series.loc[series["code"] == "11111111", "label"]) == {"Modern widgets"}
    assert set(series.loc[series["code"] == "22222222", "label"]) == {"Legacy widgets"}

    evidence = pd.read_csv(outputs["code_evidence_csv"])
    assert {
        "dimension",
        "code",
        "synthetic_peak_share",
        "synthetic_cumulative_share",
        "cumulative_ratio_synth_to_inlife",
    }.issubset(evidence.columns)
