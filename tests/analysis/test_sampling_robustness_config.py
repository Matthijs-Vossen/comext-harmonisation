from pathlib import Path

import pytest

from comext_harmonisation.analysis.config import load_sampling_robustness_config


def _base_yaml(extra: str = "") -> str:
    return f"""
analysis_type: sampling_robustness
break:
  period: "20062007"
paths:
  annual_base_dir: data/extracted_annual_no_confidential/products_like
  output_dir: outputs/analysis/sampling_robustness_test
{extra}
"""


def test_sampling_robustness_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "sampling.yaml"
    cfg_path.write_text(_base_yaml())

    cfg = load_sampling_robustness_config(cfg_path)

    assert cfg.break_config.direction == "b_to_a"
    assert cfg.flow.flow_code == "1"
    assert cfg.measures.estimation_measure == "VALUE_EUR"
    assert cfg.run.n_bins == 20
    assert cfg.output.link_summary_csv.name == "link_summary.csv"


def test_sampling_robustness_config_rejects_invalid_direction(tmp_path: Path) -> None:
    cfg_path = tmp_path / "sampling_bad_direction.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
break:
  period: "20062007"
  direction: union
"""
        )
    )

    with pytest.raises(ValueError, match="break.direction"):
        load_sampling_robustness_config(cfg_path)


def test_sampling_robustness_config_rejects_small_bin_count(tmp_path: Path) -> None:
    cfg_path = tmp_path / "sampling_bad_bins.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
run:
  n_bins: 1
"""
        )
    )

    with pytest.raises(ValueError, match="n_bins"):
        load_sampling_robustness_config(cfg_path)
