from __future__ import annotations

import sys
from pathlib import Path

import comext_harmonisation.analysis as analysis_module
from comext_harmonisation.cli import run_analysis


def test_run_analysis_dispatches_synthetic_persistence(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg_path = tmp_path / "synthetic.yaml"
    cfg_path.write_text("analysis_type: synthetic_persistence\n")

    sentinel_cfg = object()

    def _stub_loader(path: Path):
        assert path == cfg_path
        return sentinel_cfg

    def _stub_runner(cfg):
        assert cfg is sentinel_cfg
        return {
            "output_plot": tmp_path / "plot.png",
            "code_evidence_csv": tmp_path / "code_evidence.csv",
        }

    monkeypatch.setattr(analysis_module, "load_synthetic_persistence_config", _stub_loader)
    monkeypatch.setattr(
        analysis_module,
        "run_synthetic_persistence_analysis",
        _stub_runner,
    )
    monkeypatch.setattr(sys, "argv", ["run_analysis.py", "--config", str(cfg_path)])

    run_analysis.main()
    out = capsys.readouterr().out
    assert "plot:" in out
    assert "evidence:" in out
