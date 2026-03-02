from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from comext_harmonisation import apply as apply_module
from comext_harmonisation.cli import run_pipeline as run_pipeline_cli
from comext_harmonisation.chaining import engine as chaining_engine
from comext_harmonisation.concordance import io as concordance_io
from comext_harmonisation.estimation import runner as estimation_runner


def _write_pipeline_config(
    path: Path,
    *,
    years: tuple[int, int, int],
    measures: list[str] | None = None,
    stages: dict | None = None,
    estimation: dict | None = None,
    chaining: dict | None = None,
    apply: dict | None = None,
) -> None:
    start, end, target = years
    base = path.parent
    cfg = {
        "years": {"start": start, "end": end, "target": target},
        "measures": measures or ["VALUE_EUR"],
        "stages": {
            "estimate": True,
            "chain": True,
            "apply_annual": True,
            "apply_monthly": False,
        },
        "paths": {
            "concordance_path": str(base / "concordance.xls"),
            "concordance_sheet": None,
            "annual_base_dir": str(base / "annual"),
            "monthly_base_dir": str(base / "monthly"),
            "estimate_weights_dir": str(base / "weights"),
            "estimate_diagnostics_dir": str(base / "diagnostics"),
            "estimate_summary_path": str(base / "summary.csv"),
            "run_base_dir": str(base / "runs"),
        },
        "estimation": {
            "flow": "1",
            "include_aggregate_codes": False,
            "fail_on_status": True,
            "skip_existing": False,
        },
        "chaining": {
            "finalize_weights": False,
            "neg_tol": 1e-6,
            "pos_tol": 1e-10,
            "row_sum_tol": 1e-6,
            "fail_on_missing": True,
            "strict_revised_link_validation": False,
            "write_unresolved_details": False,
        },
        "apply": {
            "skip_existing": False,
            "assume_identity_for_missing": True,
            "fail_on_missing": True,
            "strict_revised_link_validation": False,
            "write_unresolved_details": False,
        },
        "parallel": {
            "max_workers_matrices": None,
            "max_workers_solver": None,
            "max_workers_chain": None,
            "max_workers_apply": None,
        },
    }
    if stages:
        cfg["stages"].update(stages)
    if estimation:
        cfg["estimation"].update(estimation)
    if chaining:
        cfg["chaining"].update(chaining)
    if apply:
        cfg["apply"].update(apply)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


# LT_REF: Sec3 operational sequencing estimate->chain->apply
def test_run_pipeline_stage_gating_estimate_only(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "cfg_estimate_only.yaml"
    _write_pipeline_config(
        cfg_path,
        years=(2010, 2011, 2011),
        stages={"estimate": True, "chain": False, "apply_annual": False, "apply_monthly": False},
        chaining={"strict_revised_link_validation": False},
        apply={"strict_revised_link_validation": False},
    )

    module = run_pipeline_cli
    monkeypatch.setattr(module, "_parse_args", lambda: argparse.Namespace(config=str(cfg_path)))

    estimation_calls: list[dict] = []

    def _stub_estimation(**kwargs):
        estimation_calls.append(kwargs)
        return []

    monkeypatch.setattr(estimation_runner, "run_weight_estimation_for_period_multi", _stub_estimation)
    monkeypatch.setattr(
        chaining_engine,
        "build_chained_weights_for_range",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("chain should not run")),
    )
    monkeypatch.setattr(
        apply_module,
        "apply_chained_weights_wide_for_range",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("annual apply should not run")),
    )
    monkeypatch.setattr(
        apply_module,
        "apply_chained_weights_wide_for_month_range",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("monthly apply should not run")),
    )
    monkeypatch.setattr(
        concordance_io,
        "read_concordance_xls",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("concordance read should not run")),
    )
    monkeypatch.setattr(
        chaining_engine,
        "build_revised_code_index_from_concordance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("revised index build should not run")),
    )

    module.main()

    assert len(estimation_calls) == 1
    assert estimation_calls[0]["period"] == "20102011"
    assert estimation_calls[0]["direction"] == "a_to_b"
    assert estimation_calls[0]["flow"] == "1"

    index = pd.read_csv(tmp_path / "runs" / "index.csv")
    assert len(index) == 1
    assert int(index.loc[0, "estimate_processed"]) == 1
    assert int(index.loc[0, "estimate_skipped"]) == 0


# LT_REF: Sec3 operational sequencing estimate->chain->apply
def test_run_pipeline_estimation_skip_existing(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "cfg_skip_existing.yaml"
    _write_pipeline_config(
        cfg_path,
        years=(2010, 2012, 2012),
        stages={"estimate": True, "chain": False, "apply_annual": False, "apply_monthly": False},
        estimation={"skip_existing": True},
    )

    weights_dir = tmp_path / "weights"
    for period in ["20102011", "20112012"]:
        base = weights_dir / period / "a_to_b" / "value_eur"
        base.mkdir(parents=True, exist_ok=True)
        (base / "weights_ambiguous.csv").write_text("from_code,to_code,weight\n")
        (base / "weights_deterministic.csv").write_text("from_code,to_code,weight\n")

    module = run_pipeline_cli
    monkeypatch.setattr(module, "_parse_args", lambda: argparse.Namespace(config=str(cfg_path)))

    estimation_calls: list[dict] = []
    monkeypatch.setattr(
        estimation_runner, "run_weight_estimation_for_period_multi", lambda **kwargs: estimation_calls.append(kwargs)
    )
    monkeypatch.setattr(
        chaining_engine,
        "build_chained_weights_for_range",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("chain should not run")),
    )
    monkeypatch.setattr(
        apply_module,
        "apply_chained_weights_wide_for_range",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("annual apply should not run")),
    )
    monkeypatch.setattr(
        apply_module,
        "apply_chained_weights_wide_for_month_range",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("monthly apply should not run")),
    )

    module.main()

    assert estimation_calls == []
    index = pd.read_csv(tmp_path / "runs" / "index.csv")
    assert int(index.loc[0, "estimate_processed"]) == 0
    assert int(index.loc[0, "estimate_skipped"]) == 2


# LT_REF: Sec3 operational sequencing estimate->chain->apply
def test_run_pipeline_strict_revised_validation_wiring(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "cfg_strict_chain_apply.yaml"
    _write_pipeline_config(
        cfg_path,
        years=(2010, 2011, 2011),
        stages={"estimate": False, "chain": True, "apply_annual": True, "apply_monthly": False},
        chaining={"strict_revised_link_validation": True, "write_unresolved_details": True},
        apply={"strict_revised_link_validation": True, "write_unresolved_details": True, "skip_existing": True},
    )

    module = run_pipeline_cli
    monkeypatch.setattr(module, "_parse_args", lambda: argparse.Namespace(config=str(cfg_path)))

    calls: dict[str, list[dict]] = {
        "read_concordance": [],
        "build_revised_index": [],
        "chain": [],
        "apply_annual": [],
    }

    revised_index = {("20102011", "a_to_b"): {"00000011"}}
    concordance_edges = pd.DataFrame(
        [
            {
                "period": "20102011",
                "vintage_a_year": "2010",
                "vintage_b_year": "2011",
                "vintage_a_code": "00000001",
                "vintage_b_code": "00000011",
            }
        ]
    )

    def _stub_read_concordance(*args, **kwargs):
        calls["read_concordance"].append({"args": args, "kwargs": kwargs})
        return concordance_edges

    def _stub_build_revised_index(edges):
        calls["build_revised_index"].append({"edges": edges})
        return revised_index

    def _stub_chain(**kwargs):
        calls["chain"].append(kwargs)
        return ["CHAINED_OUTPUT"]

    def _stub_apply_annual(**kwargs):
        calls["apply_annual"].append(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(concordance_io, "read_concordance_xls", _stub_read_concordance)
    monkeypatch.setattr(chaining_engine, "build_revised_code_index_from_concordance", _stub_build_revised_index)
    monkeypatch.setattr(
        estimation_runner,
        "run_weight_estimation_for_period_multi",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("estimate should not run")),
    )
    monkeypatch.setattr(chaining_engine, "build_code_universe_from_annual", lambda **_kwargs: {})
    monkeypatch.setattr(chaining_engine, "build_chained_weights_for_range", _stub_chain)
    monkeypatch.setattr(apply_module, "apply_chained_weights_wide_for_range", _stub_apply_annual)
    monkeypatch.setattr(
        apply_module,
        "apply_chained_weights_wide_for_month_range",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("monthly apply should not run")),
    )

    module.main()

    assert len(calls["read_concordance"]) == 1
    assert len(calls["build_revised_index"]) == 1
    assert len(calls["chain"]) == 1
    assert len(calls["apply_annual"]) == 1

    assert calls["chain"][0]["revised_codes_by_step"] == revised_index
    assert calls["chain"][0]["strict_revised_link_validation"] is True
    assert calls["chain"][0]["write_unresolved_details"] is True

    assert calls["apply_annual"][0]["revised_codes_by_step"] == revised_index
    assert calls["apply_annual"][0]["strict_revised_link_validation"] is True
    assert calls["apply_annual"][0]["write_unresolved_details"] is True
    assert calls["apply_annual"][0]["skip_existing"] is True
    assert calls["apply_annual"][0]["chained_outputs"] == ["CHAINED_OUTPUT"]
