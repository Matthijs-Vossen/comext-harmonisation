from pathlib import Path

import pytest

from comext_harmonisation.analysis.config import load_synthetic_persistence_config


def _base_yaml(extra: str = "") -> str:
    return f"""
analysis_type: synthetic_persistence
years:
  start: 1988
  end: 2023
  prehistory_anchor: 2023
  afterlife_anchor: 1988
paths:
  annual_base_dir: data/extracted_annual_no_confidential/products_like
  weights_dir: outputs/weights/adjacent
  output_dir: outputs/analysis/synthetic_persistence_test
{extra}
"""


def test_synthetic_persistence_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "synthetic.yaml"
    cfg_path.write_text(_base_yaml())
    cfg = load_synthetic_persistence_config(cfg_path)

    assert cfg.flow.mode == "imports_only"
    assert cfg.flow.flow_code == "1"
    assert cfg.plot.y_axis_unit == "percent"
    assert cfg.plot.font_scale == 1.0
    assert cfg.plot.section_title_scale == 1.0
    assert cfg.years.prehistory_anchor == 2023
    assert cfg.years.afterlife_anchor == 1988
    assert len(cfg.candidates.afterlife) == 7
    assert cfg.candidates.display_labels == {}


def test_synthetic_persistence_config_loads_plot_font_scale(tmp_path: Path) -> None:
    cfg_path = tmp_path / "synthetic_font_scale.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
plot:
  font_scale: 1.35
"""
        )
    )

    cfg = load_synthetic_persistence_config(cfg_path)
    assert cfg.plot.font_scale == 1.35


def test_synthetic_persistence_config_loads_section_title_scale(tmp_path: Path) -> None:
    cfg_path = tmp_path / "synthetic_section_title_scale.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
plot:
  section_title_scale: 0.9
"""
        )
    )

    cfg = load_synthetic_persistence_config(cfg_path)
    assert cfg.plot.section_title_scale == 0.9


def test_synthetic_persistence_config_loads_optional_display_labels(tmp_path: Path) -> None:
    cfg_path = tmp_path / "synthetic_labels.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
candidates:
  prehistory: [85171300]
  afterlife: [85211031]
  display_labels:
    85171300: Smartphones
    85211031: Tape camcorders
"""
        )
    )

    cfg = load_synthetic_persistence_config(cfg_path)
    assert cfg.candidates.display_labels == {
        "85171300": "Smartphones",
        "85211031": "Tape camcorders",
    }


def test_synthetic_persistence_config_rejects_unknown_flow_mode(tmp_path: Path) -> None:
    cfg_path = tmp_path / "synthetic_bad_flow.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
flow:
  mode: all_flows
"""
        )
    )

    with pytest.raises(ValueError, match="flow.mode"):
        load_synthetic_persistence_config(cfg_path)


def test_synthetic_persistence_config_rejects_legacy_thresholds(tmp_path: Path) -> None:
    cfg_path = tmp_path / "synthetic_bad_thresholds.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
thresholds:
  share_nontrivial: 0.0001
"""
        )
    )

    with pytest.raises(ValueError, match="thresholds are deprecated"):
        load_synthetic_persistence_config(cfg_path)


def test_synthetic_persistence_config_rejects_legacy_afterlife_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "synthetic_bad_candidates.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
candidates:
  prehistory: [85171300]
  afterlife_semantic: [85281011]
"""
        )
    )

    with pytest.raises(ValueError, match="legacy candidate keys"):
        load_synthetic_persistence_config(cfg_path)


def test_synthetic_persistence_config_accepts_legacy_selection_and_plot_keys(tmp_path: Path) -> None:
    cfg_path = tmp_path / "synthetic_legacy_keys.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
selection:
  top_n_per_dimension: 2
  ranking_metric: synthetic_peak_share
plot:
  examples_output_path: outputs/analysis/synthetic_persistence_qualitative/example_trajectories.png
  basket_composition_output_path: outputs/analysis/synthetic_persistence_qualitative/basket_composition.png
"""
        )
    )

    cfg = load_synthetic_persistence_config(cfg_path)
    assert cfg.plot.summary_output_path.name == "qualitative_summary.png"
    assert not hasattr(cfg.plot, "examples_output_path")
    assert not hasattr(cfg.plot, "basket_composition_output_path")


def test_synthetic_persistence_config_rejects_invalid_plot_axis_unit(tmp_path: Path) -> None:
    cfg_path = tmp_path / "synthetic_bad_axis_unit.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
plot:
  y_axis_unit: basis_points
"""
        )
    )

    with pytest.raises(ValueError, match="y_axis_unit"):
        load_synthetic_persistence_config(cfg_path)
