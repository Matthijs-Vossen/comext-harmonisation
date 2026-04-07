"""Config loader for analysis scenarios."""

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


@dataclass(frozen=True)
class SyntheticPersistenceYearsConfig:
    start: int
    end: int
    prehistory_anchor: int
    afterlife_anchor: int


@dataclass(frozen=True)
class SyntheticPersistenceMeasureConfig:
    weights_source: str
    analysis_measure: str


@dataclass(frozen=True)
class SyntheticPersistenceFlowConfig:
    mode: str
    flow_code: str


@dataclass(frozen=True)
class SyntheticPersistenceCandidatesConfig:
    prehistory: Sequence[str]
    afterlife: Sequence[str]
    display_labels: Mapping[str, str]


@dataclass(frozen=True)
class SyntheticPersistencePathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    weights_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class SyntheticPersistenceChainingConfig:
    finalize_weights: bool
    neg_tol: float
    pos_tol: float
    row_sum_tol: float
    fail_on_missing: bool
    strict_revised_link_validation: bool
    write_unresolved_details: bool


@dataclass(frozen=True)
class SyntheticPersistenceSampleConfig:
    exclude_reporters: Sequence[str]
    exclude_partners: Sequence[str]


@dataclass(frozen=True)
class SyntheticPersistencePlotConfig:
    summary_output_path: Path
    use_latex: bool
    latex_preamble: str
    line_width: float
    point_size: float
    font_scale: float
    section_title_scale: float
    y_axis_unit: str


@dataclass(frozen=True)
class SyntheticPersistenceConfig:
    years: SyntheticPersistenceYearsConfig
    measures: SyntheticPersistenceMeasureConfig
    flow: SyntheticPersistenceFlowConfig
    candidates: SyntheticPersistenceCandidatesConfig
    paths: SyntheticPersistencePathsConfig
    chaining: SyntheticPersistenceChainingConfig
    sample: SyntheticPersistenceSampleConfig
    plot: SyntheticPersistencePlotConfig


@dataclass(frozen=True)
class LinkDistributionPathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class LinkDistributionScopeConfig:
    mode: str


@dataclass(frozen=True)
class LinkDistributionOutputConfig:
    summary_csv: Path
    focal_codes_csv: Path


@dataclass(frozen=True)
class LinkDistributionConfig:
    paths: LinkDistributionPathsConfig
    scope: LinkDistributionScopeConfig
    output: LinkDistributionOutputConfig


@dataclass(frozen=True)
class ChainedLinkDistributionYearsConfig:
    backward_anchor: int
    forward_anchor: int


@dataclass(frozen=True)
class ChainedLinkDistributionPathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class ChainedLinkDistributionScopeConfig:
    mode: str


@dataclass(frozen=True)
class ChainedLinkDistributionOutputConfig:
    summary_csv: Path


@dataclass(frozen=True)
class ChainedLinkDistributionPlotConfig:
    output_path: Path
    bar_output_path: Path
    title: str | None
    use_latex: bool
    latex_preamble: str


@dataclass(frozen=True)
class ChainedLinkDistributionConfig:
    years: ChainedLinkDistributionYearsConfig
    paths: ChainedLinkDistributionPathsConfig
    scope: ChainedLinkDistributionScopeConfig
    output: ChainedLinkDistributionOutputConfig
    plot: ChainedLinkDistributionPlotConfig


@dataclass(frozen=True)
class CrmRevisionExposureYearsConfig:
    anchor_year: int
    backward_end_year: int
    forward_end_year: int
    benchmark_backward_years: Sequence[int]
    benchmark_forward_years: Sequence[int]


@dataclass(frozen=True)
class CrmRevisionExposurePathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    crm_codes_path: Path
    output_dir: Path


@dataclass(frozen=True)
class CrmRevisionExposureScopeConfig:
    mode: str


@dataclass(frozen=True)
class CrmRevisionExposureOutputConfig:
    summary_csv: Path
    code_exposure_csv: Path
    benchmark_summary_csv: Path


@dataclass(frozen=True)
class CrmRevisionExposurePlotConfig:
    output_path: Path
    title: str | None
    use_latex: bool
    latex_preamble: str


@dataclass(frozen=True)
class CrmRevisionExposureConfig:
    years: CrmRevisionExposureYearsConfig
    paths: CrmRevisionExposurePathsConfig
    scope: CrmRevisionExposureScopeConfig
    output: CrmRevisionExposureOutputConfig
    plot: CrmRevisionExposurePlotConfig


@dataclass(frozen=True)
class BilateralPersistenceYearsConfig:
    columns: Sequence[int]


@dataclass(frozen=True)
class BilateralPersistenceBreakConfig:
    period: str
    direction: str


@dataclass(frozen=True)
class BilateralPersistenceMeasureConfig:
    analysis_measure: str


@dataclass(frozen=True)
class BilateralPersistenceFlowConfig:
    flow_code: str


@dataclass(frozen=True)
class BilateralPersistencePathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class BilateralPersistenceSampleConfig:
    exclude_reporters: Sequence[str]
    exclude_partners: Sequence[str]


@dataclass(frozen=True)
class BilateralPersistenceFilterConfig:
    years: Sequence[int]


@dataclass(frozen=True)
class BilateralPersistenceOutputConfig:
    table_csv: Path
    table_tex: Path
    details_csv: Path
    sample_diagnostics_csv: Path


@dataclass(frozen=True)
class BilateralPersistenceConfig:
    years: BilateralPersistenceYearsConfig
    break_config: BilateralPersistenceBreakConfig
    measures: BilateralPersistenceMeasureConfig
    flow: BilateralPersistenceFlowConfig
    paths: BilateralPersistencePathsConfig
    sample: BilateralPersistenceSampleConfig
    adjusted_filter: BilateralPersistenceFilterConfig
    output: BilateralPersistenceOutputConfig


@dataclass(frozen=True)
class SamplingRobustnessBreakConfig:
    period: str
    direction: str


@dataclass(frozen=True)
class SamplingRobustnessMeasureConfig:
    estimation_measure: str


@dataclass(frozen=True)
class SamplingRobustnessFlowConfig:
    flow_code: str


@dataclass(frozen=True)
class SamplingRobustnessPathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class SamplingRobustnessSampleConfig:
    exclude_reporters: Sequence[str]
    exclude_partners: Sequence[str]


@dataclass(frozen=True)
class SamplingRobustnessRunConfig:
    n_bins: int
    seed: int


@dataclass(frozen=True)
class SamplingRobustnessOutputConfig:
    subsample_weights_csv: Path
    link_summary_csv: Path
    summary_csv: Path
    bin_assignments_csv: Path


@dataclass(frozen=True)
class SamplingRobustnessPlotConfig:
    output_path: Path
    title: str | None
    point_alpha: float
    point_size: float
    point_color: str
    histogram_bins: int
    use_latex: bool
    latex_preamble: str


@dataclass(frozen=True)
class SamplingRobustnessConfig:
    break_config: SamplingRobustnessBreakConfig
    measures: SamplingRobustnessMeasureConfig
    flow: SamplingRobustnessFlowConfig
    paths: SamplingRobustnessPathsConfig
    sample: SamplingRobustnessSampleConfig
    run: SamplingRobustnessRunConfig
    output: SamplingRobustnessOutputConfig
    plot: SamplingRobustnessPlotConfig


@dataclass(frozen=True)
class RevisionValidationYearsConfig:
    min_year: int
    max_year: int


@dataclass(frozen=True)
class RevisionValidationBreakConfig:
    direction: str


@dataclass(frozen=True)
class RevisionValidationMeasureConfig:
    weights_source: str
    analysis_measure: str


@dataclass(frozen=True)
class RevisionValidationFlowConfig:
    flow_code: str


@dataclass(frozen=True)
class RevisionValidationPathsConfig:
    concordance_path: Path
    concordance_sheet: str | int | None
    annual_base_dir: Path
    weights_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class RevisionValidationChainingConfig:
    finalize_weights: bool
    neg_tol: float
    pos_tol: float
    row_sum_tol: float


@dataclass(frozen=True)
class RevisionValidationSampleConfig:
    exclude_reporters: Sequence[str]
    exclude_partners: Sequence[str]


@dataclass(frozen=True)
class RevisionValidationRunConfig:
    n_bins: int
    seed: int
    max_workers: int


@dataclass(frozen=True)
class RevisionValidationOutputConfig:
    summary_csv: Path
    panel_details_csv: Path
    link_summary_csv: Path


@dataclass(frozen=True)
class RevisionValidationPlotConfig:
    output_path: Path
    title: str | None
    use_latex: bool
    latex_preamble: str
    show_annotations: bool


@dataclass(frozen=True)
class RevisionValidationConfig:
    years: RevisionValidationYearsConfig
    break_config: RevisionValidationBreakConfig
    measures: RevisionValidationMeasureConfig
    flow: RevisionValidationFlowConfig
    paths: RevisionValidationPathsConfig
    chaining: RevisionValidationChainingConfig
    sample: RevisionValidationSampleConfig
    run: RevisionValidationRunConfig
    output: RevisionValidationOutputConfig
    plot: RevisionValidationPlotConfig


CHAIN_LENGTH_DELTA_METRICS = (
    "mae_weighted",
    "mae_weighted_step",
    "diffuse_exposure",
)


def load_share_stability_config(path: Path) -> ShareStabilityConfig:
    data = yaml.safe_load(path.read_text()) or {}

    years = _merge({"start": None, "end": None, "target": None}, data.get("years"))
    if years["start"] is None or years["end"] is None or years["target"] is None:
        raise ValueError("Config must include years.start, years.end, and years.target.")

    break_config = _merge({"period": None, "direction": "union"}, data.get("break"))
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
        metric_names = list(CHAIN_LENGTH_DELTA_METRICS)
    metrics = [name.lower() for name in metric_names]
    invalid_metrics = sorted(set(metrics) - set(CHAIN_LENGTH_DELTA_METRICS))
    if invalid_metrics:
        raise ValueError(
            "Invalid chain_length metrics: "
            f"{invalid_metrics}. Allowed: {list(CHAIN_LENGTH_DELTA_METRICS)}"
        )

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
            "output_path": "outputs/analysis/chain_length/chain_length_delta.png",
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
        metrics=metrics,
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


def _normalize_code_list(value: Any) -> list[str]:
    result: list[str] = []
    for raw in _normalize_list(value):
        code = str(raw).strip()
        if not code:
            continue
        if code.isdigit():
            code = code.zfill(8)
        result.append(code)
    return result


def _normalize_code_label_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("synthetic_persistence: candidates.display_labels must be a mapping")

    result: dict[str, str] = {}
    for raw_code, raw_label in value.items():
        code = str(raw_code).strip()
        if not code:
            continue
        if code.isdigit():
            code = code.zfill(8)
        label = str(raw_label).strip()
        if not label:
            continue
        result[code] = label
    return result


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def load_synthetic_persistence_config(path: Path) -> SyntheticPersistenceConfig:
    data = yaml.safe_load(path.read_text()) or {}
    if "thresholds" in data:
        raise ValueError(
            "synthetic_persistence: thresholds are deprecated in qualitative mode; "
            "remove the thresholds block from config."
        )

    candidate_block = data.get("candidates") or {}
    legacy_afterlife_keys = [
        key
        for key in ("afterlife_semantic", "afterlife_high_signal", "afterlife_original")
        if key in candidate_block
    ]
    if legacy_afterlife_keys:
        keys = ", ".join(sorted(legacy_afterlife_keys))
        raise ValueError(
            "synthetic_persistence: legacy candidate keys detected "
            f"({keys}); use candidates.afterlife for qualitative analysis."
        )

    years = _merge(
        {
            "start": 1988,
            "end": 2023,
            "prehistory_anchor": 2023,
            "afterlife_anchor": 1988,
        },
        data.get("years"),
    )
    start = int(years["start"])
    end = int(years["end"])
    prehistory_anchor = int(years["prehistory_anchor"])
    afterlife_anchor = int(years["afterlife_anchor"])
    if start > end:
        raise ValueError("synthetic_persistence: years.start must be <= years.end")
    if not (start <= prehistory_anchor <= end):
        raise ValueError(
            "synthetic_persistence: years.prehistory_anchor must be inside [years.start, years.end]"
        )
    if not (start <= afterlife_anchor <= end):
        raise ValueError(
            "synthetic_persistence: years.afterlife_anchor must be inside [years.start, years.end]"
        )

    measures = _merge(
        {"weights_source": "VALUE_EUR", "analysis_measure": "VALUE_EUR"},
        data.get("measures"),
    )

    flow = _merge({"mode": "imports_only", "flow_code": "1"}, data.get("flow"))
    flow_mode = str(flow.get("mode", "imports_only")).strip().lower()
    if flow_mode != "imports_only":
        raise ValueError(
            "synthetic_persistence: flow.mode must be 'imports_only' for this analysis."
        )

    candidates = _merge(
        {
            "prehistory": [
                "85171300",
                "88062210",
                "85241100",
                "85414100",
                "85414300",
                "85235200",
            ],
            "afterlife": [
                "85281011",
                "85282010",
                "85401130",
                "85211039",
                "85281079",
                "85401190",
                "85211031",
            ],
        },
        data.get("candidates"),
    )
    # Legacy key kept for backward compatibility; ignored in qualitative mode.
    _ = data.get("selection")

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "weights_dir": "outputs/weights/adjacent",
            "output_dir": "outputs/analysis/synthetic_persistence_qualitative",
        },
        data.get("paths"),
    )

    chaining = _merge(
        {
            "finalize_weights": True,
            "neg_tol": 1e-6,
            "pos_tol": 1e-10,
            "row_sum_tol": 1e-6,
            "fail_on_missing": True,
            "strict_revised_link_validation": False,
            "write_unresolved_details": False,
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
            "summary_output_path": (
                "outputs/analysis/synthetic_persistence_qualitative/qualitative_summary.png"
            ),
            "use_latex": True,
            "latex_preamble": r"\\usepackage{newtxtext,newtxmath}",
            "line_width": 1.0,
            "point_size": 3.0,
            "font_scale": 1.0,
            "section_title_scale": 1.0,
            "y_axis_unit": "percent",
        },
        data.get("plot"),
    )
    y_axis_unit = str(plot["y_axis_unit"]).strip().lower()
    if y_axis_unit not in {"percent", "share"}:
        raise ValueError("synthetic_persistence: plot.y_axis_unit must be one of ['percent', 'share']")

    return SyntheticPersistenceConfig(
        years=SyntheticPersistenceYearsConfig(
            start=start,
            end=end,
            prehistory_anchor=prehistory_anchor,
            afterlife_anchor=afterlife_anchor,
        ),
        measures=SyntheticPersistenceMeasureConfig(
            weights_source=str(measures["weights_source"]).upper(),
            analysis_measure=str(measures["analysis_measure"]).upper(),
        ),
        flow=SyntheticPersistenceFlowConfig(
            mode=flow_mode,
            flow_code=str(flow.get("flow_code", "1")).strip(),
        ),
        candidates=SyntheticPersistenceCandidatesConfig(
            prehistory=_dedupe_preserve_order(_normalize_code_list(candidates.get("prehistory"))),
            afterlife=_dedupe_preserve_order(_normalize_code_list(candidates.get("afterlife"))),
            display_labels=_normalize_code_label_map(candidates.get("display_labels")),
        ),
        paths=SyntheticPersistencePathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            weights_dir=Path(paths["weights_dir"]),
            output_dir=Path(paths["output_dir"]),
        ),
        chaining=SyntheticPersistenceChainingConfig(
            finalize_weights=bool(chaining["finalize_weights"]),
            neg_tol=float(chaining["neg_tol"]),
            pos_tol=float(chaining["pos_tol"]),
            row_sum_tol=float(chaining["row_sum_tol"]),
            fail_on_missing=bool(chaining["fail_on_missing"]),
            strict_revised_link_validation=bool(chaining["strict_revised_link_validation"]),
            write_unresolved_details=bool(chaining["write_unresolved_details"]),
        ),
        sample=SyntheticPersistenceSampleConfig(
            exclude_reporters=_normalize_list(sample.get("exclude_reporters")),
            exclude_partners=_normalize_list(sample.get("exclude_partners")),
        ),
        plot=SyntheticPersistencePlotConfig(
            summary_output_path=Path(plot["summary_output_path"]),
            use_latex=bool(plot["use_latex"]),
            latex_preamble=str(plot["latex_preamble"]),
            line_width=float(plot["line_width"]),
            point_size=float(plot["point_size"]),
            font_scale=float(plot["font_scale"]),
            section_title_scale=float(plot["section_title_scale"]),
            y_axis_unit=y_axis_unit,
        ),
    )


def load_link_distribution_config(path: Path) -> LinkDistributionConfig:
    data = yaml.safe_load(path.read_text()) or {}

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "output_dir": "outputs/analysis/link_distribution",
        },
        data.get("paths"),
    )
    scope = _merge({"mode": "revised_only"}, data.get("scope"))
    scope_mode = str(scope["mode"]).strip().lower()
    if scope_mode not in {"revised_only", "observed_universe_implied_identities"}:
        raise ValueError(
            "link_distribution: scope.mode must be one of "
            "['revised_only', 'observed_universe_implied_identities']."
        )

    output_dir = Path(paths["output_dir"])
    output = _merge(
        {
            "summary_csv": str(output_dir / "summary.csv"),
            "focal_codes_csv": str(output_dir / "focal_codes.csv"),
        },
        data.get("output"),
    )

    return LinkDistributionConfig(
        paths=LinkDistributionPathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            output_dir=output_dir,
        ),
        scope=LinkDistributionScopeConfig(mode=scope_mode),
        output=LinkDistributionOutputConfig(
            summary_csv=Path(output["summary_csv"]),
            focal_codes_csv=Path(output["focal_codes_csv"]),
        ),
    )


def load_chained_link_distribution_config(path: Path) -> ChainedLinkDistributionConfig:
    data = yaml.safe_load(path.read_text()) or {}

    years = _merge(
        {
            "backward_anchor": 2024,
            "forward_anchor": 1988,
        },
        data.get("years"),
    )
    backward_anchor = int(years["backward_anchor"])
    forward_anchor = int(years["forward_anchor"])
    if backward_anchor <= forward_anchor:
        raise ValueError(
            "chained_link_distribution: years.backward_anchor must be > years.forward_anchor"
        )

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "output_dir": "outputs/analysis/chained_link_distribution",
        },
        data.get("paths"),
    )

    scope = _merge({"mode": "observed_universe_implied_identities"}, data.get("scope"))
    scope_mode = str(scope["mode"]).strip().lower()
    if scope_mode not in {"revised_only", "observed_universe_implied_identities"}:
        raise ValueError(
            "chained_link_distribution: scope.mode must be one of "
            "['revised_only', 'observed_universe_implied_identities']."
        )

    output_dir = Path(paths["output_dir"])
    output = _merge(
        {
            "summary_csv": str(output_dir / "summary.csv"),
        },
        data.get("output"),
    )
    plot = _merge(
        {
            "output_path": str(output_dir / "chained_link_distribution.png"),
            "bar_output_path": str(output_dir / "chained_link_distribution_bars.png"),
            "title": None,
            "use_latex": True,
            "latex_preamble": r"\\usepackage{newtxtext,newtxmath}",
        },
        data.get("plot"),
    )

    return ChainedLinkDistributionConfig(
        years=ChainedLinkDistributionYearsConfig(
            backward_anchor=backward_anchor,
            forward_anchor=forward_anchor,
        ),
        paths=ChainedLinkDistributionPathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            output_dir=output_dir,
        ),
        scope=ChainedLinkDistributionScopeConfig(mode=scope_mode),
        output=ChainedLinkDistributionOutputConfig(
            summary_csv=Path(output["summary_csv"]),
        ),
        plot=ChainedLinkDistributionPlotConfig(
            output_path=Path(plot["output_path"]),
            bar_output_path=Path(plot["bar_output_path"]),
            title=plot.get("title"),
            use_latex=bool(plot["use_latex"]),
            latex_preamble=str(plot["latex_preamble"]),
        ),
    )


def load_crm_revision_exposure_config(path: Path) -> CrmRevisionExposureConfig:
    data = yaml.safe_load(path.read_text()) or {}

    years = _merge(
        {
            "anchor_year": 2023,
            "backward_end_year": 1988,
            "forward_end_year": 2024,
            "benchmark_backward_years": [2022, 2017, 2007, 1988],
            "benchmark_forward_years": [2024],
        },
        data.get("years"),
    )
    anchor_year = int(years["anchor_year"])
    backward_end_year = int(years["backward_end_year"])
    forward_end_year = int(years["forward_end_year"])
    if backward_end_year > anchor_year:
        raise ValueError(
            "crm_revision_exposure: years.backward_end_year must be <= years.anchor_year"
        )
    if forward_end_year < anchor_year:
        raise ValueError(
            "crm_revision_exposure: years.forward_end_year must be >= years.anchor_year"
        )
    if backward_end_year == anchor_year and forward_end_year == anchor_year:
        raise ValueError(
            "crm_revision_exposure: at least one of backward_end_year or forward_end_year "
            "must differ from anchor_year"
        )

    benchmark_backward_years = [int(year) for year in years["benchmark_backward_years"]]
    benchmark_forward_years = [int(year) for year in years["benchmark_forward_years"]]
    for year in benchmark_backward_years:
        if year < backward_end_year or year > anchor_year or year == anchor_year:
            raise ValueError(
                "crm_revision_exposure: years.benchmark_backward_years must fall within "
                "[backward_end_year, anchor_year) and exclude anchor_year"
            )
    for year in benchmark_forward_years:
        if year < anchor_year or year > forward_end_year or year == anchor_year:
            raise ValueError(
                "crm_revision_exposure: years.benchmark_forward_years must fall within "
                "(anchor_year, forward_end_year] and exclude anchor_year"
            )

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "crm_codes_path": "data/crm_allocations/annex2_norm.csv",
            "output_dir": "outputs/analysis/crm_revision_exposure_2023",
        },
        data.get("paths"),
    )

    scope = _merge({"mode": "observed_universe_implied_identities"}, data.get("scope"))
    scope_mode = str(scope["mode"]).strip().lower()
    if scope_mode not in {"revised_only", "observed_universe_implied_identities"}:
        raise ValueError(
            "crm_revision_exposure: scope.mode must be one of "
            "['revised_only', 'observed_universe_implied_identities']."
        )

    output_dir = Path(paths["output_dir"])
    output = _merge(
        {
            "summary_csv": str(output_dir / "summary.csv"),
            "code_exposure_csv": str(output_dir / "code_exposure.csv"),
            "benchmark_summary_csv": str(output_dir / "benchmark_summary.csv"),
        },
        data.get("output"),
    )
    plot = _merge(
        {
            "output_path": str(output_dir / "crm_revision_exposure.png"),
            "title": None,
            "use_latex": True,
            "latex_preamble": r"\\usepackage{newtxtext,newtxmath}",
        },
        data.get("plot"),
    )

    return CrmRevisionExposureConfig(
        years=CrmRevisionExposureYearsConfig(
            anchor_year=anchor_year,
            backward_end_year=backward_end_year,
            forward_end_year=forward_end_year,
            benchmark_backward_years=tuple(benchmark_backward_years),
            benchmark_forward_years=tuple(benchmark_forward_years),
        ),
        paths=CrmRevisionExposurePathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            crm_codes_path=Path(paths["crm_codes_path"]),
            output_dir=output_dir,
        ),
        scope=CrmRevisionExposureScopeConfig(mode=scope_mode),
        output=CrmRevisionExposureOutputConfig(
            summary_csv=Path(output["summary_csv"]),
            code_exposure_csv=Path(output["code_exposure_csv"]),
            benchmark_summary_csv=Path(output["benchmark_summary_csv"]),
        ),
        plot=CrmRevisionExposurePlotConfig(
            output_path=Path(plot["output_path"]),
            title=plot.get("title"),
            use_latex=bool(plot["use_latex"]),
            latex_preamble=str(plot["latex_preamble"]),
        ),
    )


def load_bilateral_persistence_config(path: Path) -> BilateralPersistenceConfig:
    data = yaml.safe_load(path.read_text()) or {}

    years = _merge(
        {
            "columns": [2005, 2006, 2008, 2009],
        },
        data.get("years"),
    )
    columns = [int(year) for year in years.get("columns", [])]
    if not columns:
        raise ValueError("bilateral_persistence: years.columns must not be empty")
    if sorted(columns) != columns:
        raise ValueError("bilateral_persistence: years.columns must be sorted ascending")
    if len(set(columns)) != len(columns):
        raise ValueError("bilateral_persistence: years.columns must be unique")

    break_config = _merge({"period": None, "direction": "union"}, data.get("break"))
    period = str(break_config.get("period") or "").strip()
    if not period:
        raise ValueError("bilateral_persistence: break.period is required")
    if len(period) != 8 or not period.isdigit():
        raise ValueError("bilateral_persistence: break.period must be an 8-digit period")
    direction = str(break_config.get("direction", "union")).strip().lower()
    if direction not in {"a_to_b", "b_to_a", "union"}:
        raise ValueError(
            "bilateral_persistence: break.direction must be 'a_to_b', 'b_to_a', or 'union'"
        )

    measures = _merge({"analysis_measure": "VALUE_EUR"}, data.get("measures"))
    analysis_measure = str(measures.get("analysis_measure", "VALUE_EUR")).strip().upper()
    if analysis_measure not in {"VALUE_EUR", "QUANTITY_KG"}:
        raise ValueError("bilateral_persistence: measures.analysis_measure must be VALUE_EUR or QUANTITY_KG")

    flow = _merge({"flow_code": "1"}, data.get("flow"))
    flow_code = str(flow.get("flow_code", "1")).strip()

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "output_dir": "outputs/analysis/bilateral_persistence_cn2007_raw",
        },
        data.get("paths"),
    )
    output_dir = Path(paths["output_dir"])

    sample = _merge(
        {
            "exclude_reporters": [],
            "exclude_partners": [],
        },
        data.get("sample"),
    )
    adjusted_filter = _merge({"years": [2004, 2005, 2007, 2008]}, data.get("adjusted_filter"))
    filter_years = [int(year) for year in _normalize_list(adjusted_filter.get("years"))]

    outputs = _merge(
        {
            "table_csv": str(output_dir / "table.csv"),
            "table_tex": str(output_dir / "table.tex"),
            "details_csv": str(output_dir / "regression_details.csv"),
            "sample_diagnostics_csv": str(output_dir / "sample_diagnostics.csv"),
        },
        data.get("output"),
    )

    return BilateralPersistenceConfig(
        years=BilateralPersistenceYearsConfig(columns=columns),
        break_config=BilateralPersistenceBreakConfig(period=period, direction=direction),
        measures=BilateralPersistenceMeasureConfig(analysis_measure=analysis_measure),
        flow=BilateralPersistenceFlowConfig(flow_code=flow_code),
        paths=BilateralPersistencePathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            output_dir=output_dir,
        ),
        sample=BilateralPersistenceSampleConfig(
            exclude_reporters=_normalize_list(sample.get("exclude_reporters")),
            exclude_partners=_normalize_list(sample.get("exclude_partners")),
        ),
        adjusted_filter=BilateralPersistenceFilterConfig(years=filter_years),
        output=BilateralPersistenceOutputConfig(
            table_csv=Path(outputs["table_csv"]),
            table_tex=Path(outputs["table_tex"]),
            details_csv=Path(outputs["details_csv"]),
            sample_diagnostics_csv=Path(outputs["sample_diagnostics_csv"]),
        ),
    )


def load_sampling_robustness_config(path: Path) -> SamplingRobustnessConfig:
    data = yaml.safe_load(path.read_text()) or {}

    break_config = _merge({"period": None, "direction": "b_to_a"}, data.get("break"))
    period = str(break_config.get("period") or "").strip()
    if not period:
        raise ValueError("sampling_robustness: break.period is required")
    if len(period) != 8 or not period.isdigit():
        raise ValueError("sampling_robustness: break.period must be an 8-digit period")
    direction = str(break_config.get("direction", "b_to_a")).strip().lower()
    if direction not in {"a_to_b", "b_to_a"}:
        raise ValueError("sampling_robustness: break.direction must be 'a_to_b' or 'b_to_a'")

    measures = _merge({"estimation_measure": "VALUE_EUR"}, data.get("measures"))
    estimation_measure = str(measures.get("estimation_measure", "VALUE_EUR")).strip().upper()
    if estimation_measure not in {"VALUE_EUR", "QUANTITY_KG"}:
        raise ValueError(
            "sampling_robustness: measures.estimation_measure must be VALUE_EUR or QUANTITY_KG"
        )

    flow = _merge({"flow_code": "1"}, data.get("flow"))
    flow_code = str(flow.get("flow_code", "1")).strip()

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "output_dir": "outputs/analysis/sampling_robustness_cn2007",
        },
        data.get("paths"),
    )
    output_dir = Path(paths["output_dir"])

    sample = _merge(
        {
            "exclude_reporters": [],
            "exclude_partners": [],
        },
        data.get("sample"),
    )

    run = _merge({"n_bins": 20, "seed": 20260317}, data.get("run"))
    n_bins = int(run.get("n_bins", 20))
    seed = int(run.get("seed", 20260317))
    if n_bins < 2:
        raise ValueError("sampling_robustness: run.n_bins must be at least 2")

    outputs = _merge(
        {
            "subsample_weights_csv": str(output_dir / "subsample_weights.csv"),
            "link_summary_csv": str(output_dir / "link_summary.csv"),
            "summary_csv": str(output_dir / "summary.csv"),
            "bin_assignments_csv": str(output_dir / "bin_assignments.csv"),
        },
        data.get("output"),
    )

    plot = _merge(
        {
            "output_path": str(output_dir / "sampling_robustness.png"),
            "title": None,
            "point_alpha": 0.45,
            "point_size": 8.0,
            "point_color": "black",
            "histogram_bins": 20,
            "use_latex": True,
            "latex_preamble": r"\\usepackage{newtxtext,newtxmath}",
        },
        data.get("plot"),
    )
    histogram_bins = int(plot.get("histogram_bins", 20))
    if histogram_bins < 1:
        raise ValueError("sampling_robustness: plot.histogram_bins must be positive")

    return SamplingRobustnessConfig(
        break_config=SamplingRobustnessBreakConfig(period=period, direction=direction),
        measures=SamplingRobustnessMeasureConfig(estimation_measure=estimation_measure),
        flow=SamplingRobustnessFlowConfig(flow_code=flow_code),
        paths=SamplingRobustnessPathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            output_dir=output_dir,
        ),
        sample=SamplingRobustnessSampleConfig(
            exclude_reporters=_normalize_list(sample.get("exclude_reporters")),
            exclude_partners=_normalize_list(sample.get("exclude_partners")),
        ),
        run=SamplingRobustnessRunConfig(
            n_bins=n_bins,
            seed=seed,
        ),
        output=SamplingRobustnessOutputConfig(
            subsample_weights_csv=Path(outputs["subsample_weights_csv"]),
            link_summary_csv=Path(outputs["link_summary_csv"]),
            summary_csv=Path(outputs["summary_csv"]),
            bin_assignments_csv=Path(outputs["bin_assignments_csv"]),
        ),
        plot=SamplingRobustnessPlotConfig(
            output_path=Path(plot["output_path"]),
            title=plot.get("title"),
            point_alpha=float(plot["point_alpha"]),
            point_size=float(plot["point_size"]),
            point_color=str(plot["point_color"]),
            histogram_bins=histogram_bins,
            use_latex=bool(plot["use_latex"]),
            latex_preamble=str(plot["latex_preamble"]),
        ),
    )


def load_revision_validation_config(path: Path) -> RevisionValidationConfig:
    data = yaml.safe_load(path.read_text()) or {}

    years = _merge(
        {
            "min_year": 1988,
            "max_year": 2024,
        },
        data.get("years"),
    )
    min_year = int(years.get("min_year", 1988))
    max_year = int(years.get("max_year", 2024))
    if min_year >= max_year:
        raise ValueError("revision_validation: years.min_year must be less than years.max_year")
    if (max_year - min_year) < 4:
        raise ValueError(
            "revision_validation: years range must span at least 5 annual vintages"
        )

    break_config = _merge({"direction": "b_to_a"}, data.get("break"))
    direction = str(break_config.get("direction", "b_to_a")).strip().lower()
    if direction != "b_to_a":
        raise ValueError("revision_validation: break.direction must be 'b_to_a'")

    measures = _merge(
        {
            "weights_source": "VALUE_EUR",
            "analysis_measure": "VALUE_EUR",
        },
        data.get("measures"),
    )
    weights_source = str(measures.get("weights_source", "VALUE_EUR")).strip().upper()
    analysis_measure = str(measures.get("analysis_measure", "VALUE_EUR")).strip().upper()
    valid_measures = {"VALUE_EUR", "QUANTITY_KG"}
    if weights_source not in valid_measures:
        raise ValueError(
            "revision_validation: measures.weights_source must be VALUE_EUR or QUANTITY_KG"
        )
    if analysis_measure not in valid_measures:
        raise ValueError(
            "revision_validation: measures.analysis_measure must be VALUE_EUR or QUANTITY_KG"
        )

    flow = _merge({"flow_code": "1"}, data.get("flow"))
    flow_code = str(flow.get("flow_code", "1")).strip()

    paths = _merge(
        {
            "concordance_path": "data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls",
            "concordance_sheet": None,
            "annual_base_dir": "data/extracted_annual_no_confidential/products_like",
            "weights_dir": "outputs/weights/adjacent",
            "output_dir": "outputs/analysis/revision_validation",
        },
        data.get("paths"),
    )
    output_dir = Path(paths["output_dir"])

    chaining = _merge(
        {
            "finalize_weights": True,
            "neg_tol": 0.000001,
            "pos_tol": 0.0000000001,
            "row_sum_tol": 0.000001,
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

    run = _merge({"n_bins": 20, "seed": 20260317, "max_workers": 8}, data.get("run"))
    n_bins = int(run.get("n_bins", 20))
    seed = int(run.get("seed", 20260317))
    max_workers = int(run.get("max_workers", 8))
    if n_bins < 2:
        raise ValueError("revision_validation: run.n_bins must be at least 2")
    if max_workers < 1:
        raise ValueError("revision_validation: run.max_workers must be at least 1")

    outputs = _merge(
        {
            "summary_csv": str(output_dir / "summary.csv"),
            "panel_details_csv": str(output_dir / "panel_details.csv"),
            "link_summary_csv": str(output_dir / "link_summary.csv"),
        },
        data.get("output"),
    )

    plot = _merge(
        {
            "output_path": str(output_dir / "revision_validation_heatmap.png"),
            "title": None,
            "use_latex": True,
            "latex_preamble": r"\\usepackage{newtxtext,newtxmath}",
            "show_annotations": False,
        },
        data.get("plot"),
    )

    return RevisionValidationConfig(
        years=RevisionValidationYearsConfig(min_year=min_year, max_year=max_year),
        break_config=RevisionValidationBreakConfig(direction=direction),
        measures=RevisionValidationMeasureConfig(
            weights_source=weights_source,
            analysis_measure=analysis_measure,
        ),
        flow=RevisionValidationFlowConfig(flow_code=flow_code),
        paths=RevisionValidationPathsConfig(
            concordance_path=Path(paths["concordance_path"]),
            concordance_sheet=paths["concordance_sheet"],
            annual_base_dir=Path(paths["annual_base_dir"]),
            weights_dir=Path(paths["weights_dir"]),
            output_dir=output_dir,
        ),
        chaining=RevisionValidationChainingConfig(
            finalize_weights=bool(chaining["finalize_weights"]),
            neg_tol=float(chaining["neg_tol"]),
            pos_tol=float(chaining["pos_tol"]),
            row_sum_tol=float(chaining["row_sum_tol"]),
        ),
        sample=RevisionValidationSampleConfig(
            exclude_reporters=_normalize_list(sample.get("exclude_reporters")),
            exclude_partners=_normalize_list(sample.get("exclude_partners")),
        ),
        run=RevisionValidationRunConfig(
            n_bins=n_bins,
            seed=seed,
            max_workers=max_workers,
        ),
        output=RevisionValidationOutputConfig(
            summary_csv=Path(outputs["summary_csv"]),
            panel_details_csv=Path(outputs["panel_details_csv"]),
            link_summary_csv=Path(outputs["link_summary_csv"]),
        ),
        plot=RevisionValidationPlotConfig(
            output_path=Path(plot["output_path"]),
            title=plot.get("title"),
            use_latex=bool(plot["use_latex"]),
            latex_preamble=str(plot["latex_preamble"]),
            show_annotations=bool(plot["show_annotations"]),
        ),
    )
