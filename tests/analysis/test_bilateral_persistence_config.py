from pathlib import Path

import pytest

from comext_harmonisation.analysis.config import load_bilateral_persistence_config


def _base_yaml(extra: str = "") -> str:
    return f"""
analysis_type: bilateral_persistence
break:
  period: "20062007"
paths:
  annual_base_dir: data/extracted_annual_no_confidential/products_like
  output_dir: outputs/analysis/bilateral_persistence_test
{extra}
"""


def test_bilateral_persistence_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bilateral.yaml"
    cfg_path.write_text(_base_yaml())

    cfg = load_bilateral_persistence_config(cfg_path)

    assert cfg.break_config.direction == "union"
    assert cfg.flow.flow_code == "1"
    assert cfg.measures.analysis_measure == "VALUE_EUR"
    assert cfg.years.columns == [2005, 2006, 2008, 2009]
    assert cfg.aggregation_levels == ["bilateral"]
    assert cfg.output.table_csv.name == "table.csv"
    assert cfg.output.aggregation_table_csv.name == "aggregation_table.csv"


def test_bilateral_persistence_config_loads_aggregation_levels(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bilateral_aggregation.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
aggregation:
  levels: [bilateral, importer, aggregate]
"""
        )
    )

    cfg = load_bilateral_persistence_config(cfg_path)

    assert cfg.aggregation_levels == ["bilateral", "importer", "aggregate"]


def test_bilateral_persistence_config_rejects_empty_columns(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bilateral_bad_columns.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
years:
  columns: []
"""
        )
    )

    with pytest.raises(ValueError, match="years.columns"):
        load_bilateral_persistence_config(cfg_path)


def test_bilateral_persistence_config_rejects_invalid_direction(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bilateral_bad_direction.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
break:
  period: "20062007"
  direction: sideways
"""
        )
    )

    with pytest.raises(ValueError, match="break.direction"):
        load_bilateral_persistence_config(cfg_path)


def test_bilateral_persistence_config_rejects_invalid_aggregation_level(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "bilateral_bad_aggregation.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
aggregation:
  levels: [bilateral, partner]
"""
        )
    )

    with pytest.raises(ValueError, match="aggregation.levels"):
        load_bilateral_persistence_config(cfg_path)
