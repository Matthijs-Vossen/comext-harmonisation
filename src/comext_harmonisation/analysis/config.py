"""Config loader for Figure-3 style analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def _merge(defaults: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(defaults)
    if overrides:
        data.update(overrides)
    return data


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


@dataclass(frozen=True)
class Fig3YearsConfig:
    start: int
    end: int
    target: int


@dataclass(frozen=True)
class Fig3BreakConfig:
    period: str
    direction: str


@dataclass(frozen=True)
class Fig3MeasureConfig:
    weights_source: str
    analysis_measure: str


@dataclass(frozen=True)
class Fig3PathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    weights_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class Fig3ChainingConfig:
    finalize_weights: bool
    neg_tol: float
    pos_tol: float
    row_sum_tol: float


@dataclass(frozen=True)
class Fig3SampleConfig:
    exclude_reporters: Sequence[str]
    exclude_partners: Sequence[str]


@dataclass(frozen=True)
class Fig3PlotConfig:
    output_path: Path
    point_alpha: float
    point_size: float
    axis_padding: float
    point_color: str
    use_latex: bool
    latex_preamble: str


@dataclass(frozen=True)
class Fig3StabilityFilterConfig:
    enabled: bool
    years: Sequence[int]


@dataclass(frozen=True)
class Fig3Config:
    years: Fig3YearsConfig
    break_config: Fig3BreakConfig
    measures: Fig3MeasureConfig
    paths: Fig3PathsConfig
    chaining: Fig3ChainingConfig
    sample: Fig3SampleConfig
    stability_filter: Fig3StabilityFilterConfig
    plot: Fig3PlotConfig


def load_fig3_config(path: Path) -> Fig3Config:
    data = yaml.safe_load(path.read_text()) or {}

    years = _merge({"start": None, "end": None, "target": None}, data.get("years"))
    if years["start"] is None or years["end"] is None or years["target"] is None:
        raise ValueError("Config must include years.start, years.end, and years.target.")

    break_config = _merge({"period": None, "direction": "b_to_a"}, data.get("break"))
    if break_config["period"] is None:
        raise ValueError("Config must include break.period")

    measures = _merge(
        {"weights_source": "VALUE_EUR", "analysis_measure": "VALUE_EUR"},
        data.get("measures"),
    )

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "weights_dir": "outputs/weights/adjacent",
            "output_dir": "outputs/analysis/fig3",
        },
        data.get("paths"),
    )

    chaining = _merge(
        {
            "finalize_weights": True,
            "neg_tol": 1e-6,
            "pos_tol": 1e-10,
            "row_sum_tol": 1e-6,
        },
        data.get("chaining"),
    )

    sample = _merge(
        {
            "exclude_reporters": [],
            "exclude_partners": [],
        },
        data.get("sample"),
    )

    plot = _merge(
        {
            "output_path": "outputs/analysis/fig3/fig3.png",
            "point_alpha": 0.5,
            "point_size": 8.0,
            "axis_padding": 0.02,
            "point_color": "gray",
            "use_latex": False,
            "latex_preamble": r"\\usepackage{newtxtext,newtxmath}",
        },
        data.get("plot"),
    )
    stability_filter = _merge(
        {
            "enabled": False,
            "years": [],
        },
        data.get("stability_filter"),
    )

    return Fig3Config(
        years=Fig3YearsConfig(
            start=int(years["start"]),
            end=int(years["end"]),
            target=int(years["target"]),
        ),
        break_config=Fig3BreakConfig(
            period=str(break_config["period"]),
            direction=str(break_config["direction"]),
        ),
        measures=Fig3MeasureConfig(
            weights_source=str(measures["weights_source"]).upper(),
            analysis_measure=str(measures["analysis_measure"]).upper(),
        ),
        paths=Fig3PathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            weights_dir=Path(paths["weights_dir"]),
            output_dir=Path(paths["output_dir"]),
        ),
        chaining=Fig3ChainingConfig(
            finalize_weights=bool(chaining["finalize_weights"]),
            neg_tol=float(chaining["neg_tol"]),
            pos_tol=float(chaining["pos_tol"]),
            row_sum_tol=float(chaining["row_sum_tol"]),
        ),
        sample=Fig3SampleConfig(
            exclude_reporters=_normalize_list(sample.get("exclude_reporters")),
            exclude_partners=_normalize_list(sample.get("exclude_partners")),
        ),
        stability_filter=Fig3StabilityFilterConfig(
            enabled=bool(stability_filter["enabled"]),
            years=[int(year) for year in _normalize_list(stability_filter.get("years"))],
        ),
        plot=Fig3PlotConfig(
            output_path=Path(plot["output_path"]),
            point_alpha=float(plot["point_alpha"]),
            point_size=float(plot["point_size"]),
            axis_padding=float(plot["axis_padding"]),
            point_color=str(plot["point_color"]),
            use_latex=bool(plot["use_latex"]),
            latex_preamble=str(plot["latex_preamble"]),
        ),
    )
