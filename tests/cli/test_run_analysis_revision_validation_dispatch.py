from __future__ import annotations

import sys
from pathlib import Path

import comext_harmonisation.analysis as analysis_module
from comext_harmonisation.cli import run_analysis


def test_run_analysis_dispatches_revision_validation(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg_path = tmp_path / "revision_validation.yaml"
    cfg_path.write_text("analysis_type: revision_validation\n")

    sentinel_cfg = object()

    def _stub_loader(path: Path):
        assert path == cfg_path
        return sentinel_cfg

    def _stub_runner(cfg):
        assert cfg is sentinel_cfg
        return {
            "output_plot": tmp_path / "plot.png",
            "summary_csv": tmp_path / "summary.csv",
            "panel_details_csv": tmp_path / "panel_details.csv",
            "link_summary_csv": tmp_path / "link_summary.csv",
        }

    monkeypatch.setattr(analysis_module, "load_revision_validation_config", _stub_loader)
    monkeypatch.setattr(
        analysis_module,
        "run_revision_validation_analysis",
        _stub_runner,
    )
    monkeypatch.setattr(sys, "argv", ["run_analysis.py", "--config", str(cfg_path)])

    run_analysis.main()
    out = capsys.readouterr().out
    assert "plot:" in out
    assert "summary:" in out
    assert "panel_details:" in out
    assert "link_summary:" in out
