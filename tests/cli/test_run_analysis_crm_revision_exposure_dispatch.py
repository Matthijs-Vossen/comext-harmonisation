import sys
from pathlib import Path
from types import SimpleNamespace

from comext_harmonisation.cli import run_analysis


def test_run_analysis_dispatches_crm_revision_exposure(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("analysis_type: crm_revision_exposure\n")

    captured: dict[str, object] = {}

    def _stub_loader(path: Path) -> object:
        captured["config_path"] = path
        return SimpleNamespace(output=SimpleNamespace(summary_csv=Path("summary.csv")))

    def _stub_runner(config: object) -> dict[str, Path]:
        captured["config"] = config
        return {
            "output_plot": Path("fig.png"),
            "summary_csv": Path("summary.csv"),
            "code_exposure_csv": Path("code_exposure.csv"),
            "benchmark_summary_csv": Path("benchmark_summary.csv"),
        }

    import comext_harmonisation.analysis as analysis_module

    monkeypatch.setattr(
        analysis_module, "load_crm_revision_exposure_config", _stub_loader
    )
    monkeypatch.setattr(
        analysis_module,
        "run_crm_revision_exposure_analysis",
        _stub_runner,
    )
    monkeypatch.setattr(sys, "argv", ["run_analysis.py", "--config", str(cfg_path)])

    run_analysis.main()

    out = capsys.readouterr().out
    assert "plot: fig.png" in out
    assert "summary: summary.csv" in out
    assert "code_exposure: code_exposure.csv" in out
    assert "benchmark_summary: benchmark_summary.csv" in out
    assert captured["config_path"] == cfg_path
