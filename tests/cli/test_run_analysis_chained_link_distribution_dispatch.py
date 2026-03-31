from __future__ import annotations

import sys
from pathlib import Path

import comext_harmonisation.analysis as analysis_module
from comext_harmonisation.cli import run_analysis


def test_run_analysis_dispatches_chained_link_distribution(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg_path = tmp_path / "chained.yaml"
    cfg_path.write_text("analysis_type: chained_link_distribution\n")

    sentinel_cfg = object()

    def _stub_loader(path: Path):
        assert path == cfg_path
        return sentinel_cfg

    def _stub_runner(cfg):
        assert cfg is sentinel_cfg
        return {
            "output_plot": tmp_path / "plot.png",
            "output_plot_bars": tmp_path / "plot_bars.png",
            "summary_csv": tmp_path / "summary.csv",
        }

    monkeypatch.setattr(analysis_module, "load_chained_link_distribution_config", _stub_loader)
    monkeypatch.setattr(
        analysis_module,
        "run_chained_link_distribution_analysis",
        _stub_runner,
    )
    monkeypatch.setattr(sys, "argv", ["run_analysis.py", "--config", str(cfg_path)])

    run_analysis.main()
    out = capsys.readouterr().out
    assert "plot:" in out
    assert "plot_bars:" in out
    assert "summary:" in out
