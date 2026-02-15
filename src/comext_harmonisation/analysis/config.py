"""Config loader for share-stability and stress-test analyses."""

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
class ShareStabilityYearsConfig:
    start: int
    end: int
    target: int


@dataclass(frozen=True)
class ShareStabilityBreakConfig:
    period: str
    direction: str


@dataclass(frozen=True)
class ShareStabilityMeasureConfig:
    weights_source: str
    analysis_measure: str


@dataclass(frozen=True)
class ShareStabilityPathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    weights_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class ShareStabilityChainingConfig:
    finalize_weights: bool
    neg_tol: float
    pos_tol: float
    row_sum_tol: float


@dataclass(frozen=True)
class ShareStabilitySampleConfig:
    exclude_reporters: Sequence[str]
    exclude_partners: Sequence[str]


@dataclass(frozen=True)
class AnalysisPlotConfig:
    output_path: Path
    title: str | None
    point_alpha: float
    point_size: float
    axis_padding: float
    point_color: str
    use_latex: bool
    latex_preamble: str


@dataclass(frozen=True)
class ShareStabilityFilterConfig:
    enabled: bool
    years: Sequence[int]


@dataclass(frozen=True)
class ShareStabilityConfig:
    years: ShareStabilityYearsConfig
    break_config: ShareStabilityBreakConfig
    measures: ShareStabilityMeasureConfig
    metrics: Sequence[str]
    paths: ShareStabilityPathsConfig
    chaining: ShareStabilityChainingConfig
    sample: ShareStabilitySampleConfig
    stability_filter: ShareStabilityFilterConfig
    plot: AnalysisPlotConfig


@dataclass(frozen=True)
class StressChainSpec:
    base_year: int
    compare_year: int


@dataclass(frozen=True)
class StressYearsConfig:
    target: int
    chains: Sequence[StressChainSpec]


@dataclass(frozen=True)
class StressMeasureConfig:
    weights_source: str
    analysis_measure: str


@dataclass(frozen=True)
class StressPathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    weights_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class StressChainingConfig:
    finalize_weights: bool
    neg_tol: float
    pos_tol: float
    row_sum_tol: float


@dataclass(frozen=True)
class StressSampleConfig:
    exclude_reporters: Sequence[str]
    exclude_partners: Sequence[str]


@dataclass(frozen=True)
class StressConfig:
    years: StressYearsConfig
    measures: StressMeasureConfig
    metrics: Sequence[str]
    paths: StressPathsConfig
    chaining: StressChainingConfig
    sample: StressSampleConfig
    plot: AnalysisPlotConfig


@dataclass(frozen=True)
class ChainLengthYearsConfig:
    min_year: int
    max_year: int
    backward_anchor: int
    forward_anchor: int


@dataclass(frozen=True)
class ChainLengthMeasureConfig:
    weights_source: str
    analysis_measure: str


@dataclass(frozen=True)
class ChainLengthPathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    weights_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class ChainLengthChainingConfig:
    finalize_weights: bool
    neg_tol: float
    pos_tol: float
    row_sum_tol: float


@dataclass(frozen=True)
class ChainLengthSampleConfig:
    exclude_reporters: Sequence[str]
    exclude_partners: Sequence[str]
    sample_mode: str


@dataclass(frozen=True)
class ChainLengthConfig:
    years: ChainLengthYearsConfig
    measures: ChainLengthMeasureConfig
    metrics: Sequence[str]
    paths: ChainLengthPathsConfig
    chaining: ChainLengthChainingConfig
    sample: ChainLengthSampleConfig
    plot: AnalysisPlotConfig


def load_share_stability_config(path: Path) -> ShareStabilityConfig:
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
    metric_names = _normalize_list(data.get("metrics"))
    if not metric_names:
        metric_names = ["r2_45"]

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "weights_dir": "outputs/weights/adjacent",
            "output_dir": "outputs/analysis/share_stability",
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
            "sample_mode": "per_chain",
        },
        data.get("sample"),
    )

    plot = _merge(
        {
            "output_path": "outputs/analysis/share_stability/share_stability.png",
            "title": None,
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

    return ShareStabilityConfig(
        years=ShareStabilityYearsConfig(
            start=int(years["start"]),
            end=int(years["end"]),
            target=int(years["target"]),
        ),
        break_config=ShareStabilityBreakConfig(
            period=str(break_config["period"]),
            direction=str(break_config["direction"]),
        ),
        measures=ShareStabilityMeasureConfig(
            weights_source=str(measures["weights_source"]).upper(),
            analysis_measure=str(measures["analysis_measure"]).upper(),
        ),
        metrics=[name.lower() for name in metric_names],
        paths=ShareStabilityPathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            weights_dir=Path(paths["weights_dir"]),
            output_dir=Path(paths["output_dir"]),
        ),
        chaining=ShareStabilityChainingConfig(
            finalize_weights=bool(chaining["finalize_weights"]),
            neg_tol=float(chaining["neg_tol"]),
            pos_tol=float(chaining["pos_tol"]),
            row_sum_tol=float(chaining["row_sum_tol"]),
        ),
        sample=ShareStabilitySampleConfig(
            exclude_reporters=_normalize_list(sample.get("exclude_reporters")),
            exclude_partners=_normalize_list(sample.get("exclude_partners")),
        ),
        stability_filter=ShareStabilityFilterConfig(
            enabled=bool(stability_filter["enabled"]),
            years=[int(year) for year in _normalize_list(stability_filter.get("years"))],
        ),
        plot=AnalysisPlotConfig(
            output_path=Path(plot["output_path"]),
            title=plot.get("title"),
            point_alpha=float(plot["point_alpha"]),
            point_size=float(plot["point_size"]),
            axis_padding=float(plot["axis_padding"]),
            point_color=str(plot["point_color"]),
            use_latex=bool(plot["use_latex"]),
            latex_preamble=str(plot["latex_preamble"]),
        ),
    )


def load_stress_config(path: Path) -> StressConfig:
    data = yaml.safe_load(path.read_text()) or {}

    years = _merge({"target": None, "chains": []}, data.get("years"))
    if years["target"] is None or not years["chains"]:
        raise ValueError("Config must include years.target and years.chains.")

    chains = []
    for chain in years["chains"]:
        base = chain.get("base_year")
        compare = chain.get("compare_year", years["target"])
        if base is None:
            raise ValueError("Each chain must include base_year.")
        chains.append(StressChainSpec(base_year=int(base), compare_year=int(compare)))

    measures = _merge(
        {"weights_source": "VALUE_EUR", "analysis_measure": "VALUE_EUR"},
        data.get("measures"),
    )
    metric_names = _normalize_list(data.get("metrics"))
    if not metric_names:
        metric_names = [
            "r2_45",
            "diffuse_exposure",
        ]

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "weights_dir": "outputs/weights/adjacent",
            "output_dir": "outputs/analysis/stress_test",
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
            "output_path": "outputs/analysis/stress_test/fig.png",
            "title": None,
            "point_alpha": 0.4,
            "point_size": 8.0,
            "axis_padding": 0.04,
            "point_color": "black",
            "use_latex": True,
            "latex_preamble": r"\\usepackage{newtxtext,newtxmath}",
        },
        data.get("plot"),
    )

    return StressConfig(
        years=StressYearsConfig(
            target=int(years["target"]),
            chains=chains,
        ),
        measures=StressMeasureConfig(
            weights_source=str(measures["weights_source"]).upper(),
            analysis_measure=str(measures["analysis_measure"]).upper(),
        ),
        metrics=[name.lower() for name in metric_names],
        paths=StressPathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            weights_dir=Path(paths["weights_dir"]),
            output_dir=Path(paths["output_dir"]),
        ),
        chaining=StressChainingConfig(
            finalize_weights=bool(chaining["finalize_weights"]),
            neg_tol=float(chaining["neg_tol"]),
            pos_tol=float(chaining["pos_tol"]),
            row_sum_tol=float(chaining["row_sum_tol"]),
        ),
        sample=StressSampleConfig(
            exclude_reporters=_normalize_list(sample.get("exclude_reporters")),
            exclude_partners=_normalize_list(sample.get("exclude_partners")),
        ),
        plot=AnalysisPlotConfig(
            output_path=Path(plot["output_path"]),
            title=plot.get("title"),
            point_alpha=float(plot["point_alpha"]),
            point_size=float(plot["point_size"]),
            axis_padding=float(plot["axis_padding"]),
            point_color=str(plot["point_color"]),
            use_latex=bool(plot["use_latex"]),
            latex_preamble=str(plot["latex_preamble"]),
        ),
    )


def load_chain_length_config(path: Path) -> ChainLengthConfig:
    data = yaml.safe_load(path.read_text()) or {}

    years = _merge(
        {"min_year": None, "max_year": None, "backward_anchor": None, "forward_anchor": None},
        data.get("years"),
    )
    if (
        years["min_year"] is None
        or years["max_year"] is None
        or years["backward_anchor"] is None
        or years["forward_anchor"] is None
    ):
        raise ValueError(
            "Config must include years.min_year, years.max_year, years.backward_anchor, and years.forward_anchor."
        )

    measures = _merge(
        {"weights_source": "VALUE_EUR", "analysis_measure": "VALUE_EUR"},
        data.get("measures"),
    )
    metric_names = _normalize_list(data.get("metrics"))
    if not metric_names:
        metric_names = [
            "r2_45_weighted_symmetric",
            "exposure_weighted",
            "diffuseness_weighted",
        ]

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "weights_dir": "outputs/weights/adjacent",
            "output_dir": "outputs/analysis/chain_length",
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
            "output_path": "outputs/analysis/chain_length/chain_length.png",
            "title": None,
            "point_alpha": 0.5,
            "point_size": 8.0,
            "axis_padding": 0.02,
            "point_color": "gray",
            "use_latex": True,
            "latex_preamble": r"\\usepackage{newtxtext,newtxmath}",
        },
        data.get("plot"),
    )

    return ChainLengthConfig(
        years=ChainLengthYearsConfig(
            min_year=int(years["min_year"]),
            max_year=int(years["max_year"]),
            backward_anchor=int(years["backward_anchor"]),
            forward_anchor=int(years["forward_anchor"]),
        ),
        measures=ChainLengthMeasureConfig(
            weights_source=str(measures["weights_source"]).upper(),
            analysis_measure=str(measures["analysis_measure"]).upper(),
        ),
        metrics=[name.lower() for name in metric_names],
        paths=ChainLengthPathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            weights_dir=Path(paths["weights_dir"]),
            output_dir=Path(paths["output_dir"]),
        ),
        chaining=ChainLengthChainingConfig(
            finalize_weights=bool(chaining["finalize_weights"]),
            neg_tol=float(chaining["neg_tol"]),
            pos_tol=float(chaining["pos_tol"]),
            row_sum_tol=float(chaining["row_sum_tol"]),
        ),
        sample=ChainLengthSampleConfig(
            exclude_reporters=_normalize_list(sample.get("exclude_reporters")),
            exclude_partners=_normalize_list(sample.get("exclude_partners")),
            sample_mode=str(sample.get("sample_mode", "per_chain")).lower(),
        ),
        plot=AnalysisPlotConfig(
            output_path=Path(plot["output_path"]),
            title=plot.get("title"),
            point_alpha=float(plot["point_alpha"]),
            point_size=float(plot["point_size"]),
            axis_padding=float(plot["axis_padding"]),
            point_color=str(plot["point_color"]),
            use_latex=bool(plot["use_latex"]),
            latex_preamble=str(plot["latex_preamble"]),
        ),
    )
