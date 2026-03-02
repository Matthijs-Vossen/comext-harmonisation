from __future__ import annotations

from pathlib import Path

import pytest

from comext_harmonisation.pipeline.config import load_pipeline_config


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).resolve().parents[2] / "configs" / "pipeline").glob("*.yaml")),
)
def test_all_pipeline_configs_load(config_path: Path) -> None:
    cfg = load_pipeline_config(config_path)
    assert cfg.years.start <= cfg.years.end
