from pathlib import Path

import pandas as pd

from comext_harmonisation.analysis.chain_length import runner as cl_runner
from comext_harmonisation.analysis.config import (
    AnalysisPlotConfig,
    ChainLengthChainingConfig,
    ChainLengthConfig,
    ChainLengthMeasureConfig,
    ChainLengthPathsConfig,
    ChainLengthSampleConfig,
    ChainLengthYearsConfig,
)


class _DummyGroups:
    def __init__(self) -> None:
        self.edges = pd.DataFrame(
            {
                "period": ["20002001", "20012002"],
            }
        )


def _make_config(tmp_path: Path, *, output_name: str, point_size: float, point_color: str) -> ChainLengthConfig:
    output_dir = tmp_path / "out"
    return ChainLengthConfig(
        years=ChainLengthYearsConfig(
            min_year=2000,
            max_year=2002,
            backward_anchor=2000,
            forward_anchor=2002,
        ),
        measures=ChainLengthMeasureConfig(weights_source="VALUE_EUR", analysis_measure="VALUE_EUR"),
        metrics=["mae_weighted", "mae_weighted_step", "diffuse_exposure"],
        paths=ChainLengthPathsConfig(
            concordance_path=Path("dummy.xlsx"),
            concordance_sheet=None,
            annual_base_dir=tmp_path / "annual",
            weights_dir=tmp_path / "weights",
            output_dir=output_dir,
        ),
        chaining=ChainLengthChainingConfig(
            finalize_weights=True,
            neg_tol=1e-6,
            pos_tol=1e-10,
            row_sum_tol=1e-6,
        ),
        sample=ChainLengthSampleConfig(
            exclude_reporters=[],
            exclude_partners=[],
            sample_mode="per_chain",
        ),
        plot=AnalysisPlotConfig(
            output_path=output_dir / output_name,
            title="Custom title",
            point_alpha=0.3,
            point_size=point_size,
            axis_padding=0.02,
            point_color=point_color,
            use_latex=False,
            latex_preamble="",
        ),
    )


def _patch_runner_for_plot_semantics(monkeypatch, delta_calls: list[dict[str, object]]) -> None:
    monkeypatch.setattr(cl_runner, "load_concordance_groups", lambda **_: _DummyGroups())
    monkeypatch.setattr(cl_runner, "build_code_universe_from_annual", lambda **_: {})
    monkeypatch.setattr(cl_runner, "build_chained_weights_for_range", lambda **_: [])
    monkeypatch.setattr(
        cl_runner,
        "load_annual_totals",
        lambda **_kwargs: pd.DataFrame({"PRODUCT_NC": ["00000001"], "value": [1.0]}),
    )
    monkeypatch.setattr(
        cl_runner,
        "load_step_weights",
        lambda **_kwargs: pd.DataFrame(
            {"from_code": ["00000001"], "to_code": ["00000001"], "weight": [1.0]}
        ),
    )
    monkeypatch.setattr(cl_runner, "feasible_target_map", lambda *_args, **_kwargs: {})

    def _stub_compute_chain_point(*, base_year: int, target_year: int, **_kwargs):
        length = abs(base_year - target_year)
        mae_step = 0.1 * length + (0.01 if target_year == 2000 else 0.02)
        diffuse_exposure = 0.2 * length + (0.03 if target_year == 2000 else 0.04)
        step_rows = [
            {
                "step_index": 1,
                "ambiguity_exposure": 0.2,
                "diffuseness": 0.4,
                "diffuse_exposure": diffuse_exposure,
                "mae_weighted_step": mae_step,
            }
        ]
        return 0.9, 0.05 * length, mae_step, 0.2, 0.4, diffuse_exposure, 5, step_rows

    monkeypatch.setattr(cl_runner, "_compute_chain_point", _stub_compute_chain_point)

    def _stub_delta_plot(**kwargs):
        delta_calls.append(kwargs)

    monkeypatch.setattr(cl_runner, "plot_chain_length_delta_panels", _stub_delta_plot)

    def _legacy_plot_should_not_be_called(**_kwargs):
        raise AssertionError("plot_chain_length_panels should not be called in delta-only mode")

    monkeypatch.setattr(
        cl_runner,
        "plot_chain_length_panels",
        _legacy_plot_should_not_be_called,
        raising=False,
    )


def test_chain_length_runner_delta_only_primary_output(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        output_name="custom_primary.png",
        point_size=3.0,
        point_color="black",
    )
    delta_calls: list[dict[str, object]] = []
    _patch_runner_for_plot_semantics(monkeypatch, delta_calls)

    outputs = cl_runner.run_chain_length_analysis(config)

    assert len(delta_calls) == 1
    assert delta_calls[0]["output_path"] == config.plot.output_path
    assert delta_calls[0]["metrics"] == config.metrics
    assert outputs["output_plot"] == config.plot.output_path
    assert outputs["output_plot_delta"] == outputs["output_plot"]
    assert (config.paths.output_dir / "summary.csv").exists()
    assert (config.paths.output_dir / "step_metrics.csv").exists()


def test_chain_length_runner_delta_plot_style_passthrough(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        output_name="exact_filename_no_suffix.png",
        point_size=4.5,
        point_color="teal",
    )
    delta_calls: list[dict[str, object]] = []
    _patch_runner_for_plot_semantics(monkeypatch, delta_calls)

    outputs = cl_runner.run_chain_length_analysis(config)

    assert outputs["output_plot"] == config.plot.output_path
    assert len(delta_calls) == 1
    kwargs = delta_calls[0]
    assert kwargs["output_path"] == config.plot.output_path
    assert kwargs["title"] == config.plot.title
    assert kwargs["point_color"] == config.plot.point_color
    assert kwargs["point_size"] == max(1.0, config.plot.point_size - 1.0)
    assert kwargs["use_latex"] == config.plot.use_latex
    assert kwargs["latex_preamble"] == config.plot.latex_preamble
    assert kwargs["metrics"] == config.metrics
