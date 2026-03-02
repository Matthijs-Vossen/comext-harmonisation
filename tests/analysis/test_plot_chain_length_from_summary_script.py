from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def test_plot_chain_length_from_summary_script_delta_only(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    summary = pd.DataFrame(
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
                "direction": "forward",
                "anchor_year": 2002,
                "chain_length": 1,
                "mae_weighted": 0.3,
                "mae_weighted_step": 0.11,
                "diffuse_exposure": 0.08,
            },
        ]
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)
    output_path = tmp_path / "plot.png"

    from comext_harmonisation.analysis.common import plotting as plotting_module

    calls: list[dict[str, object]] = []

    def _stub_delta_plot(**kwargs):
        calls.append(kwargs)

    def _legacy_plot_should_not_run(**_kwargs):
        raise AssertionError("Legacy chain-length panel plot should not be called")

    monkeypatch.setattr(plotting_module, "plot_chain_length_delta_panels", _stub_delta_plot)
    monkeypatch.setattr(
        plotting_module,
        "plot_chain_length_panels",
        _legacy_plot_should_not_run,
        raising=False,
    )

    from comext_harmonisation.cli import plot_chain_length_from_summary as plot_chain_length_cli

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_chain_length_from_summary.py",
            "--summary",
            str(summary_path),
            "--output",
            str(output_path),
            "--point-color",
            "black",
            "--point-size",
            "3",
        ],
    )

    plot_chain_length_cli.main()
    captured = capsys.readouterr().out.strip().splitlines()

    assert len(calls) == 1
    assert calls[0]["output_path"] == output_path
    plot_lines = [line for line in captured if line.startswith("plot:")]
    assert len(plot_lines) == 1
    assert plot_lines[0].endswith(str(output_path))
