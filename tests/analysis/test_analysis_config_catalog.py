from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from comext_harmonisation.analysis import (
    load_bilateral_persistence_config,
    load_crm_revision_exposure_config,
    load_chained_link_distribution_config,
    load_chain_length_config,
    load_link_distribution_config,
    load_revision_validation_config,
    load_sampling_robustness_config,
    load_share_stability_config,
    load_synthetic_persistence_config,
    load_stress_config,
)


@pytest.mark.parametrize(
    "config_path",
    sorted((Path(__file__).resolve().parents[2] / "configs" / "analysis").glob("*.yaml")),
)
def test_all_analysis_configs_load(config_path: Path) -> None:
    data = yaml.safe_load(config_path.read_text()) or {}
    analysis_type = str(data.get("analysis_type", "")).strip().lower()
    if analysis_type == "chain_length":
        cfg = load_chain_length_config(config_path)
        assert cfg.years.min_year <= cfg.years.max_year
        return
    if analysis_type == "share_stability":
        cfg = load_share_stability_config(config_path)
        assert cfg.years.start <= cfg.years.end
        return
    if analysis_type == "stress_test":
        cfg = load_stress_config(config_path)
        assert len(cfg.years.chains) > 0
        return
    if analysis_type == "synthetic_persistence":
        cfg = load_synthetic_persistence_config(config_path)
        assert cfg.years.start <= cfg.years.end
        return
    if analysis_type == "link_distribution":
        cfg = load_link_distribution_config(config_path)
        assert cfg.scope.mode in {"revised_only", "observed_universe_implied_identities"}
        return
    if analysis_type == "chained_link_distribution":
        cfg = load_chained_link_distribution_config(config_path)
        assert cfg.years.backward_anchor > cfg.years.forward_anchor
        return
    if analysis_type == "crm_revision_exposure":
        cfg = load_crm_revision_exposure_config(config_path)
        assert cfg.years.backward_end_year <= cfg.years.anchor_year
        assert cfg.years.forward_end_year >= cfg.years.anchor_year
        return
    if analysis_type == "bilateral_persistence":
        cfg = load_bilateral_persistence_config(config_path)
        assert len(cfg.years.columns) > 0
        return
    if analysis_type == "sampling_robustness":
        cfg = load_sampling_robustness_config(config_path)
        assert cfg.run.n_bins >= 2
        return
    if analysis_type == "revision_validation":
        cfg = load_revision_validation_config(config_path)
        assert cfg.years.min_year <= cfg.years.max_year
        return
    raise AssertionError(f"Unknown analysis_type '{analysis_type}' in {config_path}")
