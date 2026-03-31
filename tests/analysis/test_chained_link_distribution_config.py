from pathlib import Path

import pytest

from comext_harmonisation.analysis.config import load_chained_link_distribution_config


def _base_yaml(extra: str = "") -> str:
    return f"""
analysis_type: chained_link_distribution
years:
  backward_anchor: 2002
  forward_anchor: 2000
paths:
  annual_base_dir: data/extracted_annual_no_confidential/products_like
  output_dir: outputs/analysis/chained_link_distribution_test
{extra}
"""


def test_chained_link_distribution_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "chained.yaml"
    cfg_path.write_text(_base_yaml())

    cfg = load_chained_link_distribution_config(cfg_path)

    assert cfg.scope.mode == "observed_universe_implied_identities"
    assert cfg.plot.output_path.name == "chained_link_distribution.png"
    assert cfg.plot.bar_output_path.name == "chained_link_distribution_bars.png"
    assert cfg.output.summary_csv.name == "summary.csv"


def test_chained_link_distribution_config_rejects_invalid_anchor_order(tmp_path: Path) -> None:
    cfg_path = tmp_path / "chained_bad_years.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
years:
  backward_anchor: 2000
  forward_anchor: 2002
"""
        )
    )

    with pytest.raises(ValueError, match="backward_anchor"):
        load_chained_link_distribution_config(cfg_path)
