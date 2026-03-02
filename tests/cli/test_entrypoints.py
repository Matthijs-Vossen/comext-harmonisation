from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


EXPECTED_SCRIPTS = {
    "comext-run-pipeline": "comext_harmonisation.cli.run_pipeline:main",
    "comext-run-estimation": "comext_harmonisation.cli.run_estimation:main",
    "comext-run-analysis": "comext_harmonisation.cli.run_analysis:main",
    "comext-plot-chain-length": "comext_harmonisation.cli.plot_chain_length_from_summary:main",
}


def _load_project_scripts() -> dict[str, str]:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    lines = pyproject.read_text().splitlines()

    scripts: dict[str, str] = {}
    in_scripts = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_scripts = line == "[project.scripts]"
            continue
        if not in_scripts:
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        scripts[key.strip()] = value.strip().strip('"').strip("'")
    return scripts


@pytest.mark.parametrize(
    ("module_name", "argv0"),
    [
        ("comext_harmonisation.cli.run_pipeline", "run_pipeline.py"),
        ("comext_harmonisation.cli.run_estimation", "run_estimation.py"),
        ("comext_harmonisation.cli.run_analysis", "run_analysis.py"),
        ("comext_harmonisation.cli.plot_chain_length_from_summary", "plot_chain_length_from_summary.py"),
    ],
)
def test_cli_modules_expose_help(monkeypatch: pytest.MonkeyPatch, module_name: str, argv0: str) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [argv0, "--help"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0


def test_console_scripts_mapping_in_pyproject() -> None:
    scripts = _load_project_scripts()
    assert scripts == EXPECTED_SCRIPTS
