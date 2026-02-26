from pathlib import Path

import pandas as pd
import pytest

from comext_harmonisation.analysis.common.plotting import plot_chain_length_delta_panels


def _delta_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "direction": "backward",
                "anchor_year": 2000,
                "chain_length": 1,
                "mae_weighted": 0.2,
                "mae_weighted_step": 0.1,
                "diffuse_exposure": 0.05,
            },
            {
                "direction": "backward",
                "anchor_year": 2000,
                "chain_length": 2,
                "mae_weighted": 0.25,
                "mae_weighted_step": 0.12,
                "diffuse_exposure": 0.07,
            },
            {
                "direction": "forward",
                "anchor_year": 2002,
                "chain_length": 1,
                "mae_weighted": 0.3,
                "mae_weighted_step": 0.11,
                "diffuse_exposure": 0.08,
            },
            {
                "direction": "forward",
                "anchor_year": 2002,
                "chain_length": 2,
                "mae_weighted": 0.35,
                "mae_weighted_step": 0.13,
                "diffuse_exposure": 0.1,
            },
        ]
    )


def test_chain_length_delta_plot_uses_selected_metrics(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    captured: dict[str, object] = {}
    original_subplots = plt.subplots

    def _capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", _capture_subplots)
    output_path = tmp_path / "delta.png"
    plot_chain_length_delta_panels(
        data=_delta_data(),
        output_path=output_path,
        title="test",
        point_color="black",
        point_size=3.0,
        use_latex=False,
        latex_preamble="",
        metrics=["mae_weighted_step", "diffuse_exposure"],
    )
    axes = captured["axes"]
    assert axes.shape[0] == 2


def test_chain_length_delta_plot_labels_weighted_d(monkeypatch, tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    captured: dict[str, object] = {}
    original_subplots = plt.subplots

    def _capture_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured["axes"] = axes
        return fig, axes

    monkeypatch.setattr(plt, "subplots", _capture_subplots)
    output_path = tmp_path / "delta_label.png"
    plot_chain_length_delta_panels(
        data=_delta_data(),
        output_path=output_path,
        title="test",
        point_color="black",
        point_size=3.0,
        use_latex=False,
        latex_preamble="",
        spearman_by_direction={"backward": 0.5, "forward": 0.3},
        metrics=["diffuse_exposure"],
    )
    axes = captured["axes"]
    assert axes[0][0].get_ylabel() == r"$\mathrm{w}D_\ell$"
    assert r"\mathrm{w}D_\ell" in axes[0][0].get_title()


def test_chain_length_delta_plot_rejects_invalid_metric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown chain-length delta metric"):
        plot_chain_length_delta_panels(
            data=_delta_data(),
            output_path=tmp_path / "bad.png",
            title=None,
            point_color="black",
            point_size=3.0,
            use_latex=False,
            latex_preamble="",
            metrics=["mae_weighted", "r2_45_weighted_symmetric"],
        )
