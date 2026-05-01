from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from comext_harmonisation.cli import run_pipeline as run_pipeline_cli


def test_demo_cn_revision_pipeline_runs_end_to_end(monkeypatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    demo_config = (
        repo_root / "examples" / "demo_cn_revision" / "configs" / "demo_pipeline.yaml"
    )
    config = yaml.safe_load(demo_config.read_text())
    config["paths"].update(
        {
            "concordance_path": str(
                repo_root
                / "data"
                / "concordances"
                / "CN_concordances_1988_2025_XLS_FORMAT.xls"
            ),
            "annual_base_dir": str(
                repo_root / "examples" / "demo_cn_revision" / "data" / "annual"
            ),
            "monthly_base_dir": str(tmp_path / "monthly"),
            "estimate_weights_dir": str(tmp_path / "weights" / "adjacent"),
            "estimate_diagnostics_dir": str(tmp_path / "weights" / "diagnostics"),
            "estimate_summary_path": str(tmp_path / "weights" / "summary.csv"),
            "run_base_dir": str(tmp_path / "runs"),
        }
    )
    config_path = tmp_path / "demo_pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    monkeypatch.setattr(
        run_pipeline_cli, "_parse_args", lambda: argparse.Namespace(config=config_path)
    )

    run_pipeline_cli.main()

    weights = pd.read_csv(
        tmp_path
        / "weights"
        / "adjacent"
        / "20072008"
        / "a_to_b"
        / "value_eur"
        / "weights_ambiguous.csv"
    )
    split = weights.loc[weights["from_code"].astype(str).str.zfill(8) == "57032019"]
    assert set(split["to_code"].astype(str).str.zfill(8)) == {"57032012", "57032018"}
    assert split["weight"].between(0.0, 1.0).all()
    assert abs(split["weight"].sum() - 1.0) < 1e-9

    deterministic = pd.read_csv(
        tmp_path
        / "weights"
        / "adjacent"
        / "20072008"
        / "a_to_b"
        / "value_eur"
        / "weights_deterministic.csv"
    )
    deterministic_pairs = set(
        zip(
            deterministic["from_code"].astype(str).str.zfill(8),
            deterministic["to_code"].astype(str).str.zfill(8),
            strict=True,
        )
    )
    assert ("29163600", "29161950") in deterministic_pairs
    assert ("90241091", "90241090") in deterministic_pairs

    run_index = pd.read_csv(tmp_path / "runs" / "index.csv")
    run_dir = Path(run_index.loc[0, "run_dir"])
    summary = pd.read_csv(run_dir / "apply" / "CN2008" / "summary.csv")
    assert set(summary["origin_year"].astype(str)) == {"2007", "2008"}
    assert summary["n_missing_value"].max() == 0
    assert summary["n_missing_quantity"].max() == 0
    assert (
        summary["sum_value_eur_w_value"] - summary["sum_value_eur_input"]
    ).abs().max() < 1e-9
    assert (
        summary["sum_quantity_kg_w_quantity"] - summary["sum_quantity_kg_input"]
    ).abs().max() < 1e-9
