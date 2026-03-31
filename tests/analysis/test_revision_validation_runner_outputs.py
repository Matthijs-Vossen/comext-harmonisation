from pathlib import Path

import numpy as np
import pandas as pd

from comext_harmonisation.analysis.config import (
    RevisionValidationBreakConfig,
    RevisionValidationChainingConfig,
    RevisionValidationConfig,
    RevisionValidationFlowConfig,
    RevisionValidationMeasureConfig,
    RevisionValidationOutputConfig,
    RevisionValidationPathsConfig,
    RevisionValidationPlotConfig,
    RevisionValidationRunConfig,
    RevisionValidationSampleConfig,
    RevisionValidationYearsConfig,
)
from comext_harmonisation.analysis.revision_validation import runner as rv_runner


def _make_config(tmp_path: Path) -> RevisionValidationConfig:
    output_dir = tmp_path / "outputs"
    return RevisionValidationConfig(
        years=RevisionValidationYearsConfig(min_year=2000, max_year=2005),
        break_config=RevisionValidationBreakConfig(direction="b_to_a"),
        measures=RevisionValidationMeasureConfig(
            weights_source="VALUE_EUR",
            analysis_measure="VALUE_EUR",
        ),
        flow=RevisionValidationFlowConfig(flow_code="1"),
        paths=RevisionValidationPathsConfig(
            concordance_path=Path("dummy.xls"),
            concordance_sheet=None,
            annual_base_dir=tmp_path / "annual",
            weights_dir=tmp_path / "weights",
            output_dir=output_dir,
        ),
        chaining=RevisionValidationChainingConfig(
            finalize_weights=True,
            neg_tol=1e-6,
            pos_tol=1e-10,
            row_sum_tol=1e-6,
        ),
        sample=RevisionValidationSampleConfig(exclude_reporters=[], exclude_partners=[]),
        run=RevisionValidationRunConfig(n_bins=20, seed=7, max_workers=2),
        output=RevisionValidationOutputConfig(
            summary_csv=output_dir / "summary.csv",
            panel_details_csv=output_dir / "panel_details.csv",
            link_summary_csv=output_dir / "link_summary.csv",
        ),
        plot=RevisionValidationPlotConfig(
            output_path=output_dir / "revision_validation_heatmap.png",
            title="Revision validation",
            use_latex=False,
            latex_preamble="",
            show_annotations=False,
        ),
    )


def test_enumerate_internal_target_years_skips_edge_windows() -> None:
    assert rv_runner._enumerate_internal_target_years(min_year=2000, max_year=2005) == [2001, 2002, 2003]


def test_summarize_panel_metrics_uses_immediate_placebos() -> None:
    panel_details = pd.DataFrame(
        {
            "panel_label": ["pre_immediate", "break", "post_immediate"],
            "r2_45": [0.90, 0.80, 0.92],
            "mae": [0.08, 0.12, 0.06],
            "mae_weighted": [0.09, 0.13, 0.07],
            "n_points": [11, 12, 13],
        }
    )

    summary = rv_runner._summarize_panel_metrics(panel_details)

    assert np.isclose(summary["placebo_baseline_fit"], 0.91)
    assert np.isclose(summary["break_penalty"], 0.11)
    assert np.isclose(summary["non_revised_mae"], 0.07)
    assert np.isclose(summary["break_mae"], 0.12)
    assert np.isclose(summary["break_year_mae"], 0.12)
    assert np.isclose(summary["excess_break_mae"], 0.05)
    assert np.isclose(summary["non_revised_wmae"], 0.08)
    assert np.isclose(summary["excess_break_wmae"], 0.05)
    assert summary["n_points_break"] == 12


def test_importance_weighted_mean_instability_uses_importance_product() -> None:
    link_summary = pd.DataFrame(
        {
            "max_minus_min": [0.10, 0.30, 0.80],
            "importance_product": [0.80, 0.15, 0.05],
        }
    )

    result = rv_runner._importance_weighted_mean_instability(link_summary)

    assert np.isclose(result, 0.165)


def test_revision_validation_runner_writes_expected_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    config = _make_config(tmp_path)
    plot_calls: list[dict[str, object]] = []
    monkeypatch.setattr(rv_runner, "load_concordance_groups", lambda **_: object())
    monkeypatch.setattr(
        rv_runner,
        "get_ambiguous_group_summary",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "period": ["20012002", "20022003", "20032004"],
                "group_id": ["g0", "g1", "g2"],
            }
        ),
    )
    monkeypatch.setattr(rv_runner, "build_code_universe_from_annual", lambda **_: {})
    monkeypatch.setattr(
        rv_runner,
        "load_annual_totals",
        lambda **_: pd.DataFrame({"PRODUCT_NC": ["A"], "value": [1.0]}),
    )
    monkeypatch.setattr(
        rv_runner,
        "_load_annual_frame",
        lambda **_: pd.DataFrame(
            {
                "REPORTER": ["R1"],
                "PARTNER": ["P1"],
                "TRADE_TYPE": ["TOTAL"],
                "PRODUCT_NC": ["A"],
                "FLOW": ["1"],
                "VALUE_EUR": [1.0],
            }
        ),
    )

    def _stub_panels(*, period: str, target_year: int, **_kwargs):
        return pd.DataFrame(
            {
                "period": [period] * 3,
                "target_year": [target_year] * 3,
                "panel_label": [
                    "pre_immediate",
                    "break",
                    "post_immediate",
                ],
                "year_t": [target_year - 1, target_year, target_year + 1],
                "year_t1": [target_year, target_year + 1, target_year + 2],
                "r2_45": [0.91, 0.84, 0.92],
                "mae": [0.08, 0.12, 0.06],
                "mae_weighted": [0.09, 0.13, 0.07],
                "n_points": [11, 12, 13],
                "n_groups": [1, 1, 1],
                "n_codes": [2, 2, 2],
                "n_focal_groups_before_filter": [1, 1, 1],
                "n_focal_groups_after_filter": [1, 1, 1],
            }
        )

    def _stub_sampling(*, period: str, target_year: int, **_kwargs):
        return (
            {
                "n_observations": 25,
                "n_groups": 2,
                "n_links": 3,
                "coverage_complete_share": 1.0,
                "max_missing_run_count": 0,
                "instability_p50": 0.05,
                "instability_p90": 0.12,
                "instability_importance_weighted_mean": 0.08,
            },
            pd.DataFrame(
                {
                    "period": [period] * 2,
                    "target_year": [target_year] * 2,
                    "group_id": ["g1", "g1"],
                    "from_code": ["B1", "B2"],
                    "to_code": ["A1", "A2"],
                    "importance_product": [0.2, 0.3],
                    "max_minus_min": [0.05, 0.10],
                    "coverage_complete": [True, True],
                    "missing_run_count": [0, 0],
                }
            ),
        )

    monkeypatch.setattr(rv_runner, "_compute_panel_details_for_period", _stub_panels)
    monkeypatch.setattr(rv_runner, "_compute_sampling_robustness_for_period", _stub_sampling)
    def _stub_plot(**kwargs):
        plot_calls.append(kwargs)
        Path(kwargs["output_path"]).write_text("plot")

    monkeypatch.setattr(rv_runner, "plot_revision_validation_heatmap", _stub_plot)

    outputs = rv_runner.run_revision_validation_analysis(config)

    assert Path(outputs["output_plot"]).exists()
    assert Path(outputs["summary_csv"]).exists()
    assert Path(outputs["panel_details_csv"]).exists()
    assert Path(outputs["link_summary_csv"]).exists()

    summary = pd.read_csv(outputs["summary_csv"])
    assert len(summary) == 3
    assert list(summary["period"].astype(str)) == ["20012002", "20022003", "20032004"]
    assert np.isclose(summary.loc[0, "placebo_baseline_fit"], 0.915)
    assert np.isclose(summary.loc[0, "break_penalty"], 0.075)
    assert np.isclose(summary.loc[0, "non_revised_mae"], 0.07)
    assert np.isclose(summary.loc[0, "break_mae"], 0.12)
    assert np.isclose(summary.loc[0, "break_year_mae"], 0.12)
    assert np.isclose(summary.loc[0, "excess_break_mae"], 0.05)
    assert np.isclose(summary.loc[0, "non_revised_wmae"], 0.08)
    assert np.isclose(summary.loc[0, "break_wmae"], 0.13)
    assert np.isclose(summary.loc[0, "excess_break_wmae"], 0.05)

    panel_details = pd.read_csv(outputs["panel_details_csv"])
    assert len(panel_details) == 9
    assert set(panel_details["panel_label"]) == {
        "pre_immediate",
        "break",
        "post_immediate",
    }
    assert {"mae", "mae_weighted", "r2_45"} <= set(panel_details.columns)

    link_summary = pd.read_csv(outputs["link_summary_csv"])
    assert set(link_summary["period"].astype(str)) == {"20012002", "20022003", "20032004"}
    assert len(plot_calls) == 1
    assert plot_calls[0]["show_annotations"] is False


def test_revision_validation_runner_soft_skips_empty_group_revision(
    monkeypatch, tmp_path: Path
) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(rv_runner, "load_concordance_groups", lambda **_: object())
    monkeypatch.setattr(
        rv_runner,
        "get_ambiguous_group_summary",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "period": ["20022003", "20032004"],
                "group_id": ["g1", "g2"],
            }
        ),
    )
    monkeypatch.setattr(rv_runner, "build_code_universe_from_annual", lambda **_: {})
    monkeypatch.setattr(
        rv_runner,
        "load_annual_totals",
        lambda **_: pd.DataFrame({"PRODUCT_NC": ["A"], "value": [1.0]}),
    )
    monkeypatch.setattr(
        rv_runner,
        "_load_annual_frame",
        lambda **_: pd.DataFrame(
            {
                "REPORTER": ["R1"],
                "PARTNER": ["P1"],
                "TRADE_TYPE": ["TOTAL"],
                "PRODUCT_NC": ["A"],
                "FLOW": ["1"],
                "VALUE_EUR": [1.0],
            }
        ),
    )

    def _stub_panels(*, period: str, target_year: int, **_kwargs):
        return pd.DataFrame(
            {
                "period": [period] * 3,
                "target_year": [target_year] * 3,
                "panel_label": ["pre_immediate", "break", "post_immediate"],
                "year_t": [target_year - 1, target_year, target_year + 1],
                "year_t1": [target_year, target_year + 1, target_year + 2],
                "r2_45": [0.91, 0.84, 0.92],
                "mae": [0.08, 0.12, 0.06],
                "mae_weighted": [0.09, 0.13, 0.07],
                "n_points": [11, 12, 13],
                "n_groups": [1, 1, 1],
                "n_codes": [2, 2, 2],
                "n_focal_groups_before_filter": [1, 1, 1],
                "n_focal_groups_after_filter": [1, 1, 1],
            }
        )

    def _stub_sampling(*, period: str, target_year: int, **_kwargs):
        return (
            {
                "n_observations": 25,
                "n_groups": 2,
                "n_links": 3,
                "coverage_complete_share": 1.0,
                "max_missing_run_count": 0,
                "instability_p50": 0.05,
                "instability_p90": 0.12,
                "instability_importance_weighted_mean": 0.08,
            },
            pd.DataFrame(
                {
                    "period": [period] * 2,
                    "target_year": [target_year] * 2,
                    "group_id": ["g1", "g1"],
                    "from_code": ["B1", "B2"],
                    "to_code": ["A1", "A2"],
                    "importance_product": [0.2, 0.3],
                    "max_minus_min": [0.05, 0.10],
                    "coverage_complete": [True, True],
                    "missing_run_count": [0, 0],
                }
            ),
        )

    monkeypatch.setattr(rv_runner, "_compute_panel_details_for_period", _stub_panels)
    monkeypatch.setattr(rv_runner, "_compute_sampling_robustness_for_period", _stub_sampling)
    monkeypatch.setattr(
        rv_runner,
        "plot_revision_validation_heatmap",
        lambda **kwargs: Path(kwargs["output_path"]).write_text("plot"),
    )

    outputs = rv_runner.run_revision_validation_analysis(config)

    summary = pd.read_csv(outputs["summary_csv"])
    assert list(summary["period"].astype(str)) == ["20012002", "20022003", "20032004"]

    skipped = summary.loc[summary["period"].astype(str) == "20012002"].iloc[0]
    assert skipped["status"] == "skipped"
    assert skipped["skip_reason"] == "no_ambiguous_groups"
    assert pd.isna(skipped["break_fit"])
    assert pd.isna(skipped["break_mae"])
    assert pd.isna(skipped["non_revised_wmae"])
    assert pd.isna(skipped["instability_p50"])

    ok_rows = summary.loc[summary["status"] == "ok"]
    assert set(ok_rows["period"].astype(str)) == {"20022003", "20032004"}

    panel_details = pd.read_csv(outputs["panel_details_csv"])
    assert set(panel_details["period"].astype(str)) == {"20022003", "20032004"}

    link_summary = pd.read_csv(outputs["link_summary_csv"])
    assert set(link_summary["period"].astype(str)) == {"20022003", "20032004"}
