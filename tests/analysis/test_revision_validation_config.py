from pathlib import Path

import pytest

from comext_harmonisation.analysis.config import load_revision_validation_config


def _base_yaml(extra: str = "") -> str:
    return f"""
analysis_type: revision_validation
paths:
  annual_base_dir: data/extracted_annual_no_confidential/products_like
  output_dir: outputs/analysis/revision_validation_test
{extra}
"""


def test_revision_validation_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "revision_validation.yaml"
    cfg_path.write_text(_base_yaml())

    cfg = load_revision_validation_config(cfg_path)

    assert cfg.break_config.direction == "b_to_a"
    assert cfg.measures.weights_source == "VALUE_EUR"
    assert cfg.measures.analysis_measure == "VALUE_EUR"
    assert cfg.run.n_bins == 20
    assert cfg.run.max_workers == 8
    assert cfg.output.panel_details_csv.name == "panel_details.csv"
    assert cfg.plot.show_annotations is False


def test_revision_validation_config_rejects_non_backward_direction(tmp_path: Path) -> None:
    cfg_path = tmp_path / "revision_validation_bad_direction.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
break:
  direction: a_to_b
"""
        )
    )

    with pytest.raises(ValueError, match="break.direction"):
        load_revision_validation_config(cfg_path)


def test_revision_validation_config_rejects_small_bin_count(tmp_path: Path) -> None:
    cfg_path = tmp_path / "revision_validation_bad_bins.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
run:
  n_bins: 1
"""
        )
    )

    with pytest.raises(ValueError, match="n_bins"):
        load_revision_validation_config(cfg_path)


def test_revision_validation_config_rejects_small_max_workers(tmp_path: Path) -> None:
    cfg_path = tmp_path / "revision_validation_bad_workers.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
run:
  max_workers: 0
"""
        )
    )

    with pytest.raises(ValueError, match="max_workers"):
        load_revision_validation_config(cfg_path)
