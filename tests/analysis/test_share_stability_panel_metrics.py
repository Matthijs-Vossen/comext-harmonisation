from pathlib import Path

import numpy as np
import pandas as pd

from comext_harmonisation.analysis.share_stability import runner as ss_runner
from comext_harmonisation.analysis.config import (
    AnalysisPlotConfig,
    ShareStabilityBreakConfig,
    ShareStabilityChainingConfig,
    ShareStabilityConfig,
    ShareStabilityFilterConfig,
    ShareStabilityMeasureConfig,
    ShareStabilityPathsConfig,
    ShareStabilitySampleConfig,
    ShareStabilityYearsConfig,
)
from comext_harmonisation.concordance.groups import build_concordance_groups


def test_share_stability_panel_local_metrics(monkeypatch, tmp_path: Path) -> None:
    edges = pd.DataFrame(
        {
            "period": ["20062007", "20062007"],
            "vintage_a_year": ["2006", "2006"],
            "vintage_b_year": ["2007", "2007"],
            "vintage_a_code": ["A", "A"],
            "vintage_b_code": ["B", "C"],
        }
    )
    groups = build_concordance_groups(edges)

    monkeypatch.setattr(ss_runner, "load_concordance_groups", lambda **_: groups)
    monkeypatch.setattr(ss_runner, "build_code_universe_from_annual", lambda **_: {})
    monkeypatch.setattr(ss_runner, "build_chained_weights_for_range", lambda **_: [])

    def stub_year_shares(**_kwargs):
        df = pd.DataFrame(
            {
                "group_id": ["20062007_g000001"],
                "product_code": ["A"],
                "share": [0.5],
            }
        )
        return {2006: df, 2007: df, 2008: df}

    monkeypatch.setattr(ss_runner, "build_year_shares_from_totals", stub_year_shares)
    monkeypatch.setattr(
        ss_runner,
        "load_annual_totals",
        lambda **_kwargs: pd.DataFrame({"PRODUCT_NC": ["A"], "value": [1.0]}),
    )
    monkeypatch.setattr(ss_runner, "plot_share_panels", lambda **_kwargs: None)

    def stub_compute_step_metrics(**kwargs):
        if kwargs["base_year"] == 2007:
            return [
                {
                    "step_index": 1,
                    "ambiguity_exposure": 0.2,
                    "diffuseness": 0.4,
                    "total_trade_sample": 100.0,
                    "ambiguous_trade": 50.0,
                }
            ]
        return []

    monkeypatch.setattr(ss_runner, "compute_step_metrics", stub_compute_step_metrics)

    config = ShareStabilityConfig(
        years=ShareStabilityYearsConfig(start=2006, end=2008, target=2006),
        break_config=ShareStabilityBreakConfig(period="20062007", direction="b_to_a"),
        measures=ShareStabilityMeasureConfig(weights_source="VALUE_EUR", analysis_measure="VALUE_EUR"),
        metrics=["r2_45", "exposure_weighted", "diffuseness_weighted"],
        paths=ShareStabilityPathsConfig(
            concordance_path=Path("dummy.xlsx"),
            concordance_sheet=None,
            annual_base_dir=tmp_path / "annual",
            weights_dir=tmp_path / "weights",
            output_dir=tmp_path / "out",
        ),
        chaining=ShareStabilityChainingConfig(
            finalize_weights=True,
            neg_tol=1e-6,
            pos_tol=1e-10,
            row_sum_tol=1e-6,
        ),
        sample=ShareStabilitySampleConfig(exclude_reporters=[], exclude_partners=[]),
        stability_filter=ShareStabilityFilterConfig(enabled=False, years=[]),
        plot=AnalysisPlotConfig(
            output_path=tmp_path / "plot.png",
            title=None,
            point_alpha=0.3,
            point_size=8.0,
            axis_padding=0.04,
            point_color="black",
            use_latex=False,
            latex_preamble="",
        ),
    )

    outputs = ss_runner.run_share_stability_analysis(config)
    summary = pd.read_csv(outputs["summary_csv"])

    row_2006 = summary.loc[summary["year_t"] == 2006].iloc[0]
    row_2007 = summary.loc[summary["year_t"] == 2007].iloc[0]

    assert np.isclose(row_2006["ambiguity_exposure_weighted"], 0.2)
    assert np.isclose(row_2006["diffuseness_weighted"], 0.4)
    assert np.isnan(row_2007["ambiguity_exposure_weighted"])
    assert np.isnan(row_2007["diffuseness_weighted"])
