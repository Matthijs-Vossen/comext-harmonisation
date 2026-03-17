from __future__ import annotations

import sys
from pathlib import Path

import comext_harmonisation.analysis as analysis_module
from comext_harmonisation.cli import run_analysis


def test_run_analysis_dispatches_bilateral_persistence(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg_path = tmp_path / "bilateral.yaml"
    cfg_path.write_text("analysis_type: bilateral_persistence\n")

    sentinel_cfg = object()

    def _stub_loader(path: Path):
        assert path == cfg_path
        return sentinel_cfg

    def _stub_runner(cfg):
        assert cfg is sentinel_cfg
        return {
            "table_csv": tmp_path / "table.csv",
            "details_csv": tmp_path / "details.csv",
        }

    monkeypatch.setattr(analysis_module, "load_bilateral_persistence_config", _stub_loader)
    monkeypatch.setattr(
        analysis_module,
        "run_bilateral_persistence_analysis",
        _stub_runner,
    )
    monkeypatch.setattr(sys, "argv", ["run_analysis.py", "--config", str(cfg_path)])

    run_analysis.main()
    out = capsys.readouterr().out
    assert "table:" in out
    assert "details:" in out
