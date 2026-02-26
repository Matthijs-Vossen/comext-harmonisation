from pathlib import Path

import pytest

from comext_harmonisation.analysis.config import load_chain_length_config


def _base_yaml(metrics_block: str) -> str:
    return f"""
analysis_type: chain_length
years:
  min_year: 2000
  max_year: 2002
  backward_anchor: 2000
  forward_anchor: 2002
measures:
  weights_source: VALUE_EUR
  analysis_measure: VALUE_EUR
{metrics_block}
paths:
  annual_base_dir: data/extracted_annual_no_confidential/products_like
  weights_dir: outputs/weights/adjacent
  output_dir: outputs/analysis/chain_length_test
plot:
  output_path: outputs/analysis/chain_length_test/chain_length_delta.png
"""


def test_chain_length_config_valid_metrics(tmp_path: Path) -> None:
    cfg_path = tmp_path / "chain_length.yaml"
    cfg_path.write_text(
        _base_yaml('metrics: ["mae_weighted", "mae_weighted_step", "diffuse_exposure"]')
    )
    config = load_chain_length_config(cfg_path)
    assert list(config.metrics) == ["mae_weighted", "mae_weighted_step", "diffuse_exposure"]


def test_chain_length_config_defaults_to_delta_metrics(tmp_path: Path) -> None:
    cfg_path = tmp_path / "chain_length_default.yaml"
    cfg_path.write_text(_base_yaml(""))
    config = load_chain_length_config(cfg_path)
    assert list(config.metrics) == ["mae_weighted", "mae_weighted_step", "diffuse_exposure"]


def test_chain_length_config_rejects_invalid_metrics(tmp_path: Path) -> None:
    cfg_path = tmp_path / "chain_length_invalid.yaml"
    cfg_path.write_text(_base_yaml('metrics: ["mae_weighted", "r2_45_weighted_symmetric"]'))
    with pytest.raises(ValueError, match="Invalid chain_length metrics"):
        load_chain_length_config(cfg_path)
