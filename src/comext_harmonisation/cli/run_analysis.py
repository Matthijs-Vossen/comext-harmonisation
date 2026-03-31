#!/usr/bin/env python3
"""Run analysis based on a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    data = yaml.safe_load(config_path.read_text()) or {}
    analysis_type = data.get("analysis_type")
    if not analysis_type:
        raise ValueError("Config must include analysis_type.")
    analysis_type = str(analysis_type).strip().lower()

    from ..analysis import (
        load_share_stability_config,
        load_stress_config,
        load_chain_length_config,
        load_synthetic_persistence_config,
        load_link_distribution_config,
        load_chained_link_distribution_config,
        load_crm_revision_exposure_config,
        load_bilateral_persistence_config,
        load_sampling_robustness_config,
        load_revision_validation_config,
        run_share_stability_analysis,
        run_stress_test_analysis,
        run_chain_length_analysis,
        run_synthetic_persistence_analysis,
        run_link_distribution_analysis,
        run_chained_link_distribution_analysis,
        run_crm_revision_exposure_analysis,
        run_bilateral_persistence_analysis,
        run_sampling_robustness_analysis,
        run_revision_validation_analysis,
    )

    if analysis_type == "share_stability":
        config = load_share_stability_config(config_path)
        outputs = run_share_stability_analysis(config)
        print("plot:", outputs["output_plot"])
        print("summary:", outputs["summary_csv"])
        return
    if analysis_type == "stress_test":
        config = load_stress_config(config_path)
        outputs = run_stress_test_analysis(config)
        print("plot:", outputs["output_plot"])
        print("summary:", outputs["summary_csv"])
        return
    if analysis_type == "chain_length":
        config = load_chain_length_config(config_path)
        outputs = run_chain_length_analysis(config)
        print("plot:", outputs["output_plot"])
        print("summary:", outputs["summary_csv"])
        return
    if analysis_type == "synthetic_persistence":
        config = load_synthetic_persistence_config(config_path)
        outputs = run_synthetic_persistence_analysis(config)
        print("plot:", outputs["output_plot"])
        print("evidence:", outputs["code_evidence_csv"])
        return
    if analysis_type == "link_distribution":
        config = load_link_distribution_config(config_path)
        outputs = run_link_distribution_analysis(config)
        print("summary:", outputs["summary_csv"])
        print("focal_codes:", outputs["focal_codes_csv"])
        return
    if analysis_type == "chained_link_distribution":
        config = load_chained_link_distribution_config(config_path)
        outputs = run_chained_link_distribution_analysis(config)
        print("plot:", outputs["output_plot"])
        print("plot_bars:", outputs["output_plot_bars"])
        print("summary:", outputs["summary_csv"])
        return
    if analysis_type == "crm_revision_exposure":
        config = load_crm_revision_exposure_config(config_path)
        outputs = run_crm_revision_exposure_analysis(config)
        print("plot:", outputs["output_plot"])
        print("summary:", outputs["summary_csv"])
        print("code_exposure:", outputs["code_exposure_csv"])
        print("benchmark_summary:", outputs["benchmark_summary_csv"])
        return
    if analysis_type == "bilateral_persistence":
        config = load_bilateral_persistence_config(config_path)
        outputs = run_bilateral_persistence_analysis(config)
        print("table:", outputs["table_csv"])
        print("details:", outputs["details_csv"])
        return
    if analysis_type == "sampling_robustness":
        config = load_sampling_robustness_config(config_path)
        outputs = run_sampling_robustness_analysis(config)
        print("plot:", outputs["output_plot"])
        print("summary:", outputs["summary_csv"])
        return
    if analysis_type == "revision_validation":
        config = load_revision_validation_config(config_path)
        outputs = run_revision_validation_analysis(config)
        print("plot:", outputs["output_plot"])
        print("summary:", outputs["summary_csv"])
        print("panel_details:", outputs["panel_details_csv"])
        print("link_summary:", outputs["link_summary_csv"])
        return
    raise ValueError(f"Unknown analysis_type '{analysis_type}'")


if __name__ == "__main__":
    main()
