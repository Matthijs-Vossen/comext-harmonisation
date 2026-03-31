from pathlib import Path

import pytest

from comext_harmonisation.analysis.config import load_crm_revision_exposure_config


def _base_yaml(extra: str = "") -> str:
    return f"""
analysis_type: crm_revision_exposure
paths:
  annual_base_dir: data/extracted_annual_no_confidential/products_like
  crm_codes_path: data/crm_allocations/annex2_norm.csv
  output_dir: outputs/analysis/crm_revision_exposure_test
{extra}
"""


def test_crm_revision_exposure_config_defaults(tmp_path: Path) -> None:
    cfg_path = tmp_path / "crm_revision_exposure.yaml"
    cfg_path.write_text(_base_yaml())

    cfg = load_crm_revision_exposure_config(cfg_path)

    assert cfg.scope.mode == "observed_universe_implied_identities"
    assert cfg.years.anchor_year == 2023
    assert list(cfg.years.benchmark_backward_years) == [2022, 2017, 2007, 1988]
    assert list(cfg.years.benchmark_forward_years) == [2024]
    assert cfg.plot.output_path.name == "crm_revision_exposure.png"
    assert cfg.output.summary_csv.name == "summary.csv"
    assert cfg.output.code_exposure_csv.name == "code_exposure.csv"
    assert cfg.output.benchmark_summary_csv.name == "benchmark_summary.csv"


def test_crm_revision_exposure_config_rejects_invalid_backward_benchmark(tmp_path: Path) -> None:
    cfg_path = tmp_path / "crm_revision_exposure_bad.yaml"
    cfg_path.write_text(
        _base_yaml(
            """
years:
  anchor_year: 2023
  backward_end_year: 1988
  forward_end_year: 2024
  benchmark_backward_years: [2023]
"""
        )
    )

    with pytest.raises(ValueError, match="benchmark_backward_years"):
        load_crm_revision_exposure_config(cfg_path)
