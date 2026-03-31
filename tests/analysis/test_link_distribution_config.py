from pathlib import Path

import pytest

from comext_harmonisation.analysis.config import load_link_distribution_config


def _base_yaml(extra: str = "") -> str:
    return f"""
analysis_type: link_distribution
paths:
  output_dir: outputs/analysis/link_distribution_test
{extra}
"""


def test_link_distribution_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "link_distribution.yaml"
    cfg_path.write_text(_base_yaml())

    cfg = load_link_distribution_config(cfg_path)

    assert cfg.scope.mode == "revised_only"
    assert cfg.paths.annual_base_dir.name == "products_like"
    assert cfg.output.summary_csv.name == "summary.csv"
    assert cfg.output.focal_codes_csv.name == "focal_codes.csv"


def test_link_distribution_config_accepts_observed_universe_scope(tmp_path: Path) -> None:
    cfg_path = tmp_path / "link_distribution_observed.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
scope:
  mode: observed_universe_implied_identities
"""
        )
    )

    cfg = load_link_distribution_config(cfg_path)

    assert cfg.scope.mode == "observed_universe_implied_identities"


def test_link_distribution_config_rejects_unknown_scope(tmp_path: Path) -> None:
    cfg_path = tmp_path / "link_distribution_bad_scope.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
scope:
  mode: full_universe
"""
        )
    )

    with pytest.raises(ValueError, match="scope.mode"):
        load_link_distribution_config(cfg_path)
