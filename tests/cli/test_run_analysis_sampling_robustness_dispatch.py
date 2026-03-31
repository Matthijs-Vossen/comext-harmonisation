from __future__ import annotations

import sys
from pathlib import Path

import comext_harmonisation.analysis as analysis_module
from comext_harmonisation.cli import run_analysis


def test_run_analysis_dispatches_sampling_robustness(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg_path = tmp_path / "sampling.yaml"
    cfg_path.write_text("analysis_type: sampling_robustness\n")

    sentinel_cfg = object()

    def _stub_loader(path: Path):
        assert path == cfg_path
        return sentinel_cfg

    def _stub_runner(cfg):
        assert cfg is sentinel_cfg
        return {
            "output_plot": tmp_path / "plot.png",
            "summary_csv": tmp_path / "summary.csv",
        }

    monkeypatch.setattr(analysis_module, "load_sampling_robustness_config", _stub_loader)
    monkeypatch.setattr(
        analysis_module,
        "run_sampling_robustness_analysis",
        _stub_runner,
    )
    monkeypatch.setattr(sys, "argv", ["run_analysis.py", "--config", str(cfg_path)])

    run_analysis.main()
    out = capsys.readouterr().out
    assert "plot:" in out
    assert "summary:" in out
