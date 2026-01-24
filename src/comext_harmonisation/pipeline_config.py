"""Pipeline configuration loader for end-to-end runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


def _merge(defaults: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(defaults)
    if overrides:
        data.update(overrides)
    return data


def _normalize_measures(value: Any) -> list[str]:
    if value is None:
        return ["VALUE_EUR", "QUANTITY_KG"]
    if isinstance(value, str):
        val = value.strip().upper()
        if val == "BOTH":
            return ["VALUE_EUR", "QUANTITY_KG"]
        return [val]
    return [str(item).strip().upper() for item in value]


@dataclass(frozen=True)
class YearsConfig:
    start: int
    end: int
    target: int


@dataclass(frozen=True)
class StagesConfig:
    estimate: bool
    chain: bool
    apply_annual: bool
    apply_monthly: bool


@dataclass(frozen=True)
class PathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    monthly_base_dir: Path
    estimate_weights_dir: Path
    estimate_diagnostics_dir: Path
    estimate_summary_path: Path
    run_base_dir: Path


@dataclass(frozen=True)
class EstimationConfig:
    flow: str
    include_aggregate_codes: bool
    fail_on_status: bool
    skip_existing: bool


@dataclass(frozen=True)
class ChainingConfig:
    finalize_weights: bool
    neg_tol: float
    pos_tol: float
    row_sum_tol: float
    fail_on_missing: bool


@dataclass(frozen=True)
class ApplyConfig:
    skip_existing: bool
    assume_identity_for_missing: bool
    fail_on_missing: bool


@dataclass(frozen=True)
class ParallelConfig:
    max_workers_matrices: int | None
    max_workers_solver: int | None
    max_workers_chain: int | None
    max_workers_apply: int | None


@dataclass(frozen=True)
class PipelineConfig:
    years: YearsConfig
    measures: list[str]
    stages: StagesConfig
    paths: PathsConfig
    estimation: EstimationConfig
    chaining: ChainingConfig
    apply: ApplyConfig
    parallel: ParallelConfig


def load_pipeline_config(path: Path) -> PipelineConfig:
    data = yaml.safe_load(path.read_text()) or {}

    years = _merge({"start": None, "end": None, "target": None}, data.get("years"))
    if years["start"] is None or years["end"] is None or years["target"] is None:
        raise ValueError("Config must include years.start, years.end, and years.target.")

    stages = _merge(
        {
            "estimate": True,
            "chain": True,
            "apply_annual": True,
            "apply_monthly": False,
        },
        data.get("stages"),
    )
    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "monthly_base_dir": "data/extracted_no_confidential/products_like",
            "estimate_weights_dir": "outputs/weights/adjacent",
            "estimate_diagnostics_dir": "outputs/weights/diagnostics",
            "estimate_summary_path": "outputs/weights/summary.csv",
            "run_base_dir": "outputs/runs",
        },
        data.get("paths"),
    )
    estimation = _merge(
        {
            "flow": "1",
            "include_aggregate_codes": False,
            "fail_on_status": True,
            "skip_existing": True,
        },
        data.get("estimation"),
    )
    chaining = _merge(
        {
            "finalize_weights": False,
            "neg_tol": 1e-6,
            "pos_tol": 1e-10,
            "row_sum_tol": 1e-6,
            "fail_on_missing": True,
        },
        data.get("chaining"),
    )
    apply = _merge(
        {
            "skip_existing": True,
            "assume_identity_for_missing": True,
            "fail_on_missing": True,
        },
        data.get("apply"),
    )
    parallel = _merge(
        {
            "max_workers_matrices": None,
            "max_workers_solver": None,
            "max_workers_chain": None,
            "max_workers_apply": None,
        },
        data.get("parallel"),
    )

    return PipelineConfig(
        years=YearsConfig(
            start=int(years["start"]),
            end=int(years["end"]),
            target=int(years["target"]),
        ),
        measures=_normalize_measures(data.get("measures")),
        stages=StagesConfig(
            estimate=bool(stages["estimate"]),
            chain=bool(stages["chain"]),
            apply_annual=bool(stages["apply_annual"]),
            apply_monthly=bool(stages["apply_monthly"]),
        ),
        paths=PathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            monthly_base_dir=Path(paths["monthly_base_dir"]),
            estimate_weights_dir=Path(paths["estimate_weights_dir"]),
            estimate_diagnostics_dir=Path(paths["estimate_diagnostics_dir"]),
            estimate_summary_path=Path(paths["estimate_summary_path"]),
            run_base_dir=Path(paths["run_base_dir"]),
        ),
        estimation=EstimationConfig(
            flow=str(estimation["flow"]),
            include_aggregate_codes=bool(estimation["include_aggregate_codes"]),
            fail_on_status=bool(estimation["fail_on_status"]),
            skip_existing=bool(estimation["skip_existing"]),
        ),
        chaining=ChainingConfig(
            finalize_weights=bool(chaining["finalize_weights"]),
            neg_tol=float(chaining["neg_tol"]),
            pos_tol=float(chaining["pos_tol"]),
            row_sum_tol=float(chaining["row_sum_tol"]),
            fail_on_missing=bool(chaining["fail_on_missing"]),
        ),
        apply=ApplyConfig(
            skip_existing=bool(apply["skip_existing"]),
            assume_identity_for_missing=bool(apply["assume_identity_for_missing"]),
            fail_on_missing=bool(apply["fail_on_missing"]),
        ),
        parallel=ParallelConfig(
            max_workers_matrices=parallel["max_workers_matrices"],
            max_workers_solver=parallel["max_workers_solver"],
            max_workers_chain=parallel["max_workers_chain"],
            max_workers_apply=parallel["max_workers_apply"],
        ),
    )
