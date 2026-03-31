from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

from comext_harmonisation.analysis.config import (
    SamplingRobustnessBreakConfig,
    SamplingRobustnessConfig,
    SamplingRobustnessFlowConfig,
    SamplingRobustnessMeasureConfig,
    SamplingRobustnessOutputConfig,
    SamplingRobustnessPathsConfig,
    SamplingRobustnessPlotConfig,
    SamplingRobustnessRunConfig,
    SamplingRobustnessSampleConfig,
)
from comext_harmonisation.analysis.sampling_robustness import runner as sr_runner
from comext_harmonisation.concordance.groups import build_concordance_groups


def _make_groups():
    edges = pd.DataFrame(
        {
            "period": ["20062007"] * 5,
            "vintage_a_year": ["2006"] * 5,
            "vintage_b_year": ["2007"] * 5,
            "vintage_a_code": ["A1", "A2", "A3", "C1", "C2"],
            "vintage_b_code": ["B1", "B1", "B2", "D1", "D1"],
        }
    )
    return build_concordance_groups(edges)


def _make_config(
    tmp_path: Path,
    *,
    exclude_reporters: Optional[list[str]] = None,
    exclude_partners: Optional[list[str]] = None,
    n_bins: int = 2,
    seed: int = 7,
) -> SamplingRobustnessConfig:
    output_dir = tmp_path / "outputs"
    return SamplingRobustnessConfig(
        break_config=SamplingRobustnessBreakConfig(period="20062007", direction="b_to_a"),
        measures=SamplingRobustnessMeasureConfig(estimation_measure="VALUE_EUR"),
        flow=SamplingRobustnessFlowConfig(flow_code="1"),
        paths=SamplingRobustnessPathsConfig(
            concordance_path=Path("dummy.xls"),
            concordance_sheet=None,
            annual_base_dir=tmp_path / "annual",
            output_dir=output_dir,
        ),
        sample=SamplingRobustnessSampleConfig(
            exclude_reporters=exclude_reporters or [],
            exclude_partners=exclude_partners or [],
        ),
        run=SamplingRobustnessRunConfig(n_bins=n_bins, seed=seed),
        output=SamplingRobustnessOutputConfig(
            subsample_weights_csv=output_dir / "subsample_weights.csv",
            link_summary_csv=output_dir / "link_summary.csv",
            summary_csv=output_dir / "summary.csv",
            bin_assignments_csv=output_dir / "bin_assignments.csv",
        ),
        plot=SamplingRobustnessPlotConfig(
            output_path=output_dir / "sampling_robustness.png",
            title="Toy robustness",
            point_alpha=0.4,
            point_size=5.0,
            point_color="black",
            histogram_bins=8,
            use_latex=False,
            latex_preamble="",
        ),
    )


def _write_annual_parquets(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    year_2006 = pd.DataFrame(
        {
            "REPORTER": [
                "R1",
                "R1",
                "R1",
                "R1",
                "R2",
                "R2",
                "R2",
                "R2",
                "R3",
                "R3",
                "R3",
                "R3",
                "R4",
                "R4",
                "R4",
                "R4",
                "R5",
                "R5",
                "R5",
                "R5",
                "R6",
                "R6",
                "R6",
                "R6",
            ],
            "PARTNER": ["P1"] * 24,
            "TRADE_TYPE": ["TOTAL"] * 24,
            "PRODUCT_NC": [
                "A1",
                "A2",
                "A3",
                "C1",
                "A1",
                "A2",
                "A3",
                "C1",
                "A1",
                "A2",
                "A3",
                "C2",
                "A1",
                "A2",
                "A3",
                "C2",
                "A1",
                "A2",
                "A3",
                "C1",
                "A1",
                "A2",
                "A3",
                "C2",
            ],
            "FLOW": ["1"] * 24,
            "VALUE_EUR": [
                90.0,
                10.0,
                20.0,
                70.0,
                80.0,
                20.0,
                10.0,
                65.0,
                20.0,
                80.0,
                30.0,
                70.0,
                10.0,
                90.0,
                40.0,
                80.0,
                60.0,
                40.0,
                15.0,
                55.0,
                40.0,
                60.0,
                25.0,
                55.0,
            ],
        }
    )
    year_2007 = pd.DataFrame(
        {
            "REPORTER": [
                "R1",
                "R1",
                "R1",
                "R2",
                "R2",
                "R2",
                "R3",
                "R3",
                "R3",
                "R4",
                "R4",
                "R4",
                "R5",
                "R5",
                "R5",
                "R6",
                "R6",
                "R6",
            ],
            "PARTNER": ["P1"] * 18,
            "TRADE_TYPE": ["TOTAL"] * 18,
            "PRODUCT_NC": [
                "B1",
                "B2",
                "D1",
                "B1",
                "B2",
                "D1",
                "B1",
                "B2",
                "D1",
                "B1",
                "B2",
                "D1",
                "B1",
                "B2",
                "D1",
                "B1",
                "B2",
                "D1",
            ],
            "FLOW": ["1"] * 18,
            "VALUE_EUR": [
                100.0,
                20.0,
                100.0,
                90.0,
                10.0,
                100.0,
                40.0,
                30.0,
                100.0,
                30.0,
                40.0,
                100.0,
                70.0,
                15.0,
                100.0,
                60.0,
                25.0,
                100.0,
            ],
        }
    )
    year_2006.to_parquet(base_dir / "comext_2006.parquet", index=False)
    year_2007.to_parquet(base_dir / "comext_2007.parquet", index=False)


def test_sampling_robustness_runner_writes_expected_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    config = _make_config(tmp_path)
    _write_annual_parquets(config.paths.annual_base_dir)

    groups = _make_groups()
    monkeypatch.setattr(sr_runner, "load_concordance_groups", lambda **_: groups)

    plot_calls: list[dict[str, object]] = []

    def _stub_plot(**kwargs):
        plot_calls.append(kwargs)
        Path(kwargs["output_path"]).write_text("plot")

    monkeypatch.setattr(sr_runner, "plot_sampling_robustness_panels", _stub_plot)

    outputs = sr_runner.run_sampling_robustness_analysis(config)

    assert Path(outputs["output_plot"]).exists()
    assert Path(outputs["summary_csv"]).exists()
    assert Path(outputs["link_summary_csv"]).exists()
    assert Path(outputs["subsample_weights_csv"]).exists()
    assert Path(outputs["bin_assignments_csv"]).exists()

    link_summary = pd.read_csv(outputs["link_summary_csv"])
    assert len(link_summary) == 4
    assert set(link_summary["from_code"]) == {"B1", "D1"}
    assert "B2" not in set(link_summary["from_code"])
    assert link_summary["coverage_complete"].all()
    assert link_summary["missing_run_count"].eq(0).all()
    assert link_summary["importance_product"].between(0.0, 1.0).all()

    run_weights = pd.read_csv(outputs["subsample_weights_csv"])
    assert set(run_weights["run_id"]) == {0, 1}
    assert set(run_weights["omitted_bin"]) == {0, 1}

    assignments = pd.read_csv(outputs["bin_assignments_csv"])
    assert len(assignments) == 12
    assert assignments["bin_id"].value_counts().to_dict() == {0: 6, 1: 6}

    summary = pd.read_csv(outputs["summary_csv"])
    assert summary.loc[0, "n_links"] == 4
    assert summary.loc[0, "n_bins"] == 2
    assert summary.loc[0, "coverage_complete_share"] == 1.0

    assert len(plot_calls) == 1
    assert plot_calls[0]["output_path"] == config.plot.output_path
    assert plot_calls[0]["histogram_bins"] == config.plot.histogram_bins


def test_sampling_robustness_runner_applies_sample_exclusions(
    monkeypatch, tmp_path: Path
) -> None:
    config = _make_config(tmp_path, exclude_reporters=["R6"])
    _write_annual_parquets(config.paths.annual_base_dir)

    groups = _make_groups()
    monkeypatch.setattr(sr_runner, "load_concordance_groups", lambda **_: groups)
    monkeypatch.setattr(
        sr_runner,
        "plot_sampling_robustness_panels",
        lambda **kwargs: Path(kwargs["output_path"]).write_text("plot"),
    )

    outputs = sr_runner.run_sampling_robustness_analysis(config)

    assignments = pd.read_csv(outputs["bin_assignments_csv"])
    assert "R6" not in set(assignments["REPORTER"])
    assert len(assignments) == 10
    assert assignments["bin_id"].value_counts().to_dict() == {0: 5, 1: 5}

    summary = pd.read_csv(outputs["summary_csv"])
    assert summary.loc[0, "n_observations"] == 10


def test_filter_holdout_rows_is_group_specific() -> None:
    group_map = pd.DataFrame(
        {
            "PRODUCT_NC": ["A1", "C1"],
            "group_id": ["g1", "g2"],
        }
    )
    raw = pd.DataFrame(
        {
            "REPORTER": ["R1", "R1"],
            "PARTNER": ["P1", "P1"],
            "TRADE_TYPE": ["TOTAL", "TOTAL"],
            "PRODUCT_NC": ["A1", "C1"],
            "FLOW": ["1", "1"],
            "VALUE_EUR": [10.0, 20.0],
        }
    )

    annotated = sr_runner._annotate_for_holdout(raw, group_map=group_map)
    filtered = sr_runner._filter_holdout_rows(
        annotated,
        holdout_keys={"g1|R1|P1"},
        measure="VALUE_EUR",
    )

    assert len(filtered) == 1
    assert filtered.iloc[0]["PRODUCT_NC"] == "C1"


def test_reported_ambiguous_weights_excludes_deterministic_links() -> None:
    weights = pd.DataFrame(
        {
            "group_id": ["g1", "g1", "g1", "g2"],
            "from_code": ["B1", "B1", "B2", "D1"],
            "to_code": ["A1", "A2", "A3", "C1"],
            "weight": [0.6, 0.4, 1.0, 1.0],
        }
    )

    reported = sr_runner._reported_ambiguous_weights(weights)

    assert set(reported["from_code"]) == {"B1"}
    assert set(reported["to_code"]) == {"A1", "A2"}
    assert "B2" not in set(reported["from_code"])
    assert "D1" not in set(reported["from_code"])


def test_sampling_robustness_runner_is_reproducible_for_fixed_seed(
    monkeypatch, tmp_path: Path
) -> None:
    groups = _make_groups()
    monkeypatch.setattr(sr_runner, "load_concordance_groups", lambda **_: groups)
    monkeypatch.setattr(
        sr_runner,
        "plot_sampling_robustness_panels",
        lambda **kwargs: Path(kwargs["output_path"]).write_text("plot"),
    )

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    config1 = _make_config(run1, seed=11)
    config2 = _make_config(run2, seed=11)
    _write_annual_parquets(config1.paths.annual_base_dir)
    _write_annual_parquets(config2.paths.annual_base_dir)

    outputs1 = sr_runner.run_sampling_robustness_analysis(config1)
    outputs2 = sr_runner.run_sampling_robustness_analysis(config2)

    assignments1 = pd.read_csv(outputs1["bin_assignments_csv"])
    assignments2 = pd.read_csv(outputs2["bin_assignments_csv"])
    pd.testing.assert_frame_equal(assignments1, assignments2)

    summary1 = pd.read_csv(outputs1["summary_csv"])
    summary2 = pd.read_csv(outputs2["summary_csv"])
    pd.testing.assert_frame_equal(summary1, summary2)

    link_summary1 = pd.read_csv(outputs1["link_summary_csv"])
    link_summary2 = pd.read_csv(outputs2["link_summary_csv"])
    pd.testing.assert_frame_equal(link_summary1, link_summary2)


def test_sampling_robustness_runner_rejects_more_bins_than_observations(
    monkeypatch, tmp_path: Path
) -> None:
    config = _make_config(tmp_path, n_bins=13)
    _write_annual_parquets(config.paths.annual_base_dir)
    groups = _make_groups()
    monkeypatch.setattr(sr_runner, "load_concordance_groups", lambda **_: groups)

    with pytest.raises(ValueError, match="must not exceed the number of estimation observations"):
        sr_runner.run_sampling_robustness_analysis(config)
