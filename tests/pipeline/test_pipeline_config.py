from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from comext_harmonisation.pipeline.config import load_pipeline_config


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


# LT_REF: Sec3 operational enforcement via config
def test_load_pipeline_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pipeline_defaults.yaml"
    _write_yaml(
        cfg_path,
        {
            "years": {
                "start": 2010,
                "end": 2011,
                "target": 2011,
            }
        },
    )

    cfg = load_pipeline_config(cfg_path)

    assert cfg.years.start == 2010
    assert cfg.years.end == 2011
    assert cfg.years.target == 2011
    assert cfg.measures == ["VALUE_EUR", "QUANTITY_KG"]

    assert cfg.stages.estimate is True
    assert cfg.stages.chain is True
    assert cfg.stages.apply_annual is True
    assert cfg.stages.apply_monthly is False

    assert cfg.estimation.flow == "1"
    assert cfg.estimation.include_aggregate_codes is False
    assert cfg.estimation.fail_on_status is True
    assert cfg.estimation.skip_existing is True

    assert cfg.chaining.fail_on_missing is True
    assert cfg.chaining.strict_revised_link_validation is True
    assert cfg.chaining.write_unresolved_details is True

    assert cfg.apply.fail_on_missing is True
    assert cfg.apply.strict_revised_link_validation is True
    assert cfg.apply.write_unresolved_details is True


# LT_REF: Sec3 operational enforcement via config
def test_load_pipeline_config_normalizes_and_casts(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pipeline_casts.yaml"
    _write_yaml(
        cfg_path,
        {
            "years": {
                "start": "2010",
                "end": "2012",
                "target": "2011",
            },
            "measures": "both",
            "stages": {
                "estimate": False,
                "chain": True,
                "apply_annual": False,
                "apply_monthly": True,
            },
            "estimation": {
                "flow": 2,
                "include_aggregate_codes": True,
                "fail_on_status": False,
                "skip_existing": False,
            },
            "chaining": {
                "finalize_weights": True,
                "neg_tol": "1e-5",
                "pos_tol": "1e-11",
                "row_sum_tol": "1e-4",
                "fail_on_missing": False,
                "strict_revised_link_validation": False,
                "write_unresolved_details": False,
            },
            "apply": {
                "skip_existing": False,
                "assume_identity_for_missing": False,
                "fail_on_missing": False,
                "strict_revised_link_validation": False,
                "write_unresolved_details": False,
            },
        },
    )

    cfg = load_pipeline_config(cfg_path)

    assert cfg.years.start == 2010
    assert cfg.years.end == 2012
    assert cfg.years.target == 2011
    assert cfg.measures == ["VALUE_EUR", "QUANTITY_KG"]

    assert cfg.stages.estimate is False
    assert cfg.stages.chain is True
    assert cfg.stages.apply_annual is False
    assert cfg.stages.apply_monthly is True

    assert cfg.estimation.flow == "2"
    assert cfg.estimation.include_aggregate_codes is True
    assert cfg.estimation.fail_on_status is False
    assert cfg.estimation.skip_existing is False

    assert cfg.chaining.finalize_weights is True
    assert cfg.chaining.neg_tol == pytest.approx(1e-5)
    assert cfg.chaining.pos_tol == pytest.approx(1e-11)
    assert cfg.chaining.row_sum_tol == pytest.approx(1e-4)
    assert cfg.chaining.fail_on_missing is False
    assert cfg.chaining.strict_revised_link_validation is False
    assert cfg.chaining.write_unresolved_details is False

    assert cfg.apply.skip_existing is False
    assert cfg.apply.assume_identity_for_missing is False
    assert cfg.apply.fail_on_missing is False
    assert cfg.apply.strict_revised_link_validation is False
    assert cfg.apply.write_unresolved_details is False


# LT_REF: Sec3 operational enforcement via config
def test_load_pipeline_config_missing_required_year_field_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pipeline_missing_year.yaml"
    _write_yaml(
        cfg_path,
        {
            "years": {
                "start": 2010,
                "end": 2012,
            }
        },
    )

    with pytest.raises(ValueError, match="years.start, years.end, and years.target"):
        load_pipeline_config(cfg_path)


# LT_REF: Sec3 operational enforcement via config
def test_load_pipeline_config_invalid_year_type_raises(tmp_path: Path) -> None:
    cfg_path = tmp_path / "pipeline_invalid_year_type.yaml"
    _write_yaml(
        cfg_path,
        {
            "years": {
                "start": "not-an-int",
                "end": 2012,
                "target": 2011,
            }
        },
    )

    with pytest.raises(ValueError):
        load_pipeline_config(cfg_path)
