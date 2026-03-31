from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


def test_plot_revision_validation_from_summary_script(tmp_path: Path, monkeypatch, capsys) -> None:
    summary = pd.DataFrame(
        [
            {
                "period": "20012002",
                "non_revised_mae": 0.10,
                "break_year_mae": 0.12,
                "excess_break_mae": 0.02,
                "instability_p50": 0.05,
                "instability_p90": 0.10,
                "instability_importance_weighted_mean": 0.07,
            },
            {
                "period": "20022003",
                "non_revised_mae": 0.20,
                "break_year_mae": 0.25,
                "excess_break_mae": 0.05,
                "instability_p50": 0.06,
                "instability_p90": 0.11,
                "instability_importance_weighted_mean": 0.08,
            },
        ]
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)
    output_path = tmp_path / "plot.png"

    from comext_harmonisation.analysis.common import plotting as plotting_module

    calls: list[dict[str, object]] = []

    def _stub_plot(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(plotting_module, "plot_revision_validation_heatmap", _stub_plot)

    from comext_harmonisation.cli import (
        plot_revision_validation_from_summary as plot_revision_validation_cli,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_revision_validation_from_summary.py",
            "--summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--title",
            "Revision validation",
        ],
    )

    plot_revision_validation_cli.main()
    captured = capsys.readouterr().out.strip().splitlines()

    assert len(calls) == 1
    assert calls[0]["output_path"] == output_path
    assert calls[0]["title"] == "Revision validation"
    assert calls[0]["show_annotations"] is False
    plot_lines = [line for line in captured if line.startswith("plot:")]
    assert len(plot_lines) == 1
    assert plot_lines[0].endswith(str(output_path))


def test_plot_revision_validation_from_summary_script_can_enable_annotations(
    tmp_path: Path, monkeypatch
) -> None:
    summary = pd.DataFrame(
        [
            {
                "period": "20012002",
                "non_revised_mae": 0.10,
                "break_year_mae": 0.12,
                "excess_break_mae": 0.02,
                "instability_p50": 0.05,
                "instability_p90": 0.10,
                "instability_importance_weighted_mean": 0.07,
            }
        ]
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    from comext_harmonisation.analysis.common import plotting as plotting_module

    calls: list[dict[str, object]] = []

    def _stub_plot(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(plotting_module, "plot_revision_validation_heatmap", _stub_plot)

    from comext_harmonisation.cli import (
        plot_revision_validation_from_summary as plot_revision_validation_cli,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_revision_validation_from_summary.py",
            "--summary",
            str(summary_path),
            "--show-annotations",
        ],
    )

    plot_revision_validation_cli.main()

    assert len(calls) == 1
    assert calls[0]["show_annotations"] is True


def test_plot_revision_validation_from_summary_script_missing_summary(tmp_path: Path, monkeypatch) -> None:
    missing_path = tmp_path / "missing.csv"

    from comext_harmonisation.cli import (
        plot_revision_validation_from_summary as plot_revision_validation_cli,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_revision_validation_from_summary.py",
            "--summary",
            str(missing_path),
        ],
    )

    with pytest.raises(FileNotFoundError, match="Missing summary CSV"):
        plot_revision_validation_cli.main()
