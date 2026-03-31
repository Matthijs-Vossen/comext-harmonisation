from pathlib import Path

import pandas as pd

from comext_harmonisation.analysis.common.plotting import plot_revision_validation_heatmap


def _sample_revision_validation_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["20012002", "20022003"],
            "non_revised_mae": [0.10, 0.20],
            "break_year_mae": [0.12, float("nan")],
            "excess_break_mae": [0.02, float("nan")],
            "n_points_break": [12.0, float("nan")],
            "instability_p50": [0.05, 0.06],
            "instability_p90": [0.10, 0.11],
            "instability_importance_weighted_mean": [0.07, 0.08],
        }
    )


def test_revision_validation_heatmap_defaults_to_no_annotations(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.pyplot as plt

    captured: dict[str, object] = {}
    original_figure = plt.figure

    def _capture_figure(*args, **kwargs):
        fig = original_figure(*args, **kwargs)
        captured["fig"] = fig
        return fig

    monkeypatch.setattr(plt, "figure", _capture_figure)
    output_path = tmp_path / "revision_validation.png"
    plot_revision_validation_heatmap(
        data=_sample_revision_validation_summary(),
        output_path=output_path,
        title=None,
        use_latex=False,
        latex_preamble="",
    )

    fig = captured["fig"]
    axes = fig.axes
    top_ax = axes[0]
    middle_ax = axes[1]
    bottom_ax = axes[2]
    scale_axes = axes[3:]
    assert [tick.get_text() for tick in top_ax.get_yticklabels()] == [
        "Non-revised MAE",
        "Break-year MAE",
        "Excess break MAE",
    ]
    assert [tick.get_text() for tick in middle_ax.get_yticklabels()] == [
        "Median instability",
        "Importance-weighted instability",
    ]
    assert [tick.get_text() for tick in bottom_ax.get_yticklabels()] == [
        "Observations",
    ]
    assert top_ax.get_title() == "Local persistence"
    assert middle_ax.get_title() == "Weight robustness"
    assert bottom_ax.get_title() == "Break-year sample size"
    width, height = fig.get_size_inches()
    assert width < 10.0
    assert height < 6.0
    texts = [text.get_text() for text in top_ax.texts + middle_ax.texts + bottom_ax.texts]
    assert "0.120" not in texts
    assert "12" not in texts
    assert texts.count("nan") == 0
    assert len(scale_axes) == 3
    scale_texts = [text.get_text() for ax in scale_axes for text in ax.texts]
    assert "0.100" in scale_texts
    assert "0.200" in scale_texts
    assert "12" in scale_texts


def test_revision_validation_heatmap_can_show_annotations(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.pyplot as plt

    captured: dict[str, object] = {}
    original_figure = plt.figure

    def _capture_figure(*args, **kwargs):
        fig = original_figure(*args, **kwargs)
        captured["fig"] = fig
        return fig

    monkeypatch.setattr(plt, "figure", _capture_figure)
    output_path = tmp_path / "revision_validation_annotated.png"
    plot_revision_validation_heatmap(
        data=_sample_revision_validation_summary(),
        output_path=output_path,
        title=None,
        use_latex=False,
        latex_preamble="",
        show_annotations=True,
    )

    fig = captured["fig"]
    axes = fig.axes
    top_ax = axes[0]
    middle_ax = axes[1]
    bottom_ax = axes[2]
    scale_axes = axes[3:]
    width, height = fig.get_size_inches()
    assert width < 10.0
    assert height < 6.0
    texts = [text.get_text() for text in top_ax.texts + middle_ax.texts + bottom_ax.texts]
    assert "0.120" in texts
    assert "12" in texts
    assert "12.000" not in texts
    assert texts.count("nan") == 0
    scale_texts = [text.get_text() for ax in scale_axes for text in ax.texts]
    assert "0.100" in scale_texts
    assert "0.200" in scale_texts
    assert "12" in scale_texts


def test_revision_validation_heatmap_size_is_stable_across_annotation_modes(
    monkeypatch, tmp_path: Path
) -> None:
    import matplotlib.pyplot as plt

    figures: list[object] = []
    original_figure = plt.figure

    def _capture_figure(*args, **kwargs):
        fig = original_figure(*args, **kwargs)
        figures.append(fig)
        return fig

    monkeypatch.setattr(plt, "figure", _capture_figure)

    plot_revision_validation_heatmap(
        data=_sample_revision_validation_summary(),
        output_path=tmp_path / "revision_validation_no_annotations.png",
        title=None,
        use_latex=False,
        latex_preamble="",
    )
    plot_revision_validation_heatmap(
        data=_sample_revision_validation_summary(),
        output_path=tmp_path / "revision_validation_with_annotations.png",
        title=None,
        use_latex=False,
        latex_preamble="",
        show_annotations=True,
    )

    assert len(figures) == 2
    size_without_annotations = tuple(figures[0].get_size_inches())
    size_with_annotations = tuple(figures[1].get_size_inches())
    assert size_without_annotations == size_with_annotations
