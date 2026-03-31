"""CRM-focused structural revision exposure for CN2023 anchor codes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...chaining.composition import compose_weights
from ...chaining.engine import build_code_universe_from_annual
from ...concordance.groups import build_concordance_groups
from ...core.codes import normalize_code_set
from ...estimation.runner import load_concordance_groups
from ..common.plotting import plot_crm_revision_exposure_panels
from ..common.steps import chain_steps
from ..config import CrmRevisionExposureConfig


RELATIONSHIP_ORDER = ["1:1", "m:1", "1:n", "m:n"]
SUMMARY_METRICS = [
    "remained_strict_1_to_1",
    "ever_non_1_to_1_step",
    "ever_unknown_weight_step",
]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _classify_relationship(n_focal: int, n_other: int) -> str:
    if n_focal == 1 and n_other == 1:
        return "1:1"
    if n_focal > 1 and n_other == 1:
        return "m:1"
    if n_focal == 1 and n_other > 1:
        return "1:n"
    return "m:n"


def _unknown_relationships_for_direction(direction: str) -> set[str]:
    if direction == "a_to_b":
        return {"1:n", "m:n"}
    if direction == "b_to_a":
        return {"m:1", "m:n"}
    raise ValueError(f"Unsupported direction '{direction}'")


def _step_edges(
    *,
    groups,
    period: str,
    direction: str,
    code_universe: dict[int, set[str]],
    scope_mode: str,
    required_source_codes: set[str] | None = None,
) -> pd.DataFrame:
    edges = groups.edges.loc[groups.edges["period"].astype(str) == str(period)].copy()
    if edges.empty:
        raise ValueError(f"Missing concordance edges for period '{period}'")

    if direction == "a_to_b":
        source_year = int(str(edges["vintage_a_year"].iloc[0]))
        target_year = int(str(edges["vintage_b_year"].iloc[0]))
        revised_source = normalize_code_set(edges["vintage_a_code"].tolist())
        revised_target = normalize_code_set(edges["vintage_b_code"].tolist())
        step = edges.rename(
            columns={"vintage_a_code": "from_code", "vintage_b_code": "to_code"}
        )[["from_code", "to_code"]]
    elif direction == "b_to_a":
        source_year = int(str(edges["vintage_b_year"].iloc[0]))
        target_year = int(str(edges["vintage_a_year"].iloc[0]))
        revised_source = normalize_code_set(edges["vintage_b_code"].tolist())
        revised_target = normalize_code_set(edges["vintage_a_code"].tolist())
        step = edges.rename(
            columns={"vintage_b_code": "from_code", "vintage_a_code": "to_code"}
        )[["from_code", "to_code"]]
    else:
        raise ValueError(f"Unsupported direction '{direction}'")

    step = step.drop_duplicates().reset_index(drop=True)
    if scope_mode == "observed_universe_implied_identities":
        identity_codes = (
            code_universe[source_year] & code_universe[target_year]
        ) - revised_source - revised_target
        if identity_codes:
            identity = pd.DataFrame(
                {
                    "from_code": sorted(identity_codes),
                    "to_code": sorted(identity_codes),
                }
            )
            step = pd.concat([step, identity], ignore_index=True)
        if required_source_codes:
            missing_source_codes = set(required_source_codes) - set(step["from_code"].tolist())
            if missing_source_codes:
                carry_identity = pd.DataFrame(
                    {
                        "from_code": sorted(missing_source_codes),
                        "to_code": sorted(missing_source_codes),
                    }
                )
                step = pd.concat([step, carry_identity], ignore_index=True)

    step = step.drop_duplicates().reset_index(drop=True)
    step["weight"] = 1.0
    return step


def _binarize_relation(df: pd.DataFrame) -> pd.DataFrame:
    relation = df[["from_code", "to_code"]].drop_duplicates().reset_index(drop=True)
    relation["weight"] = 1.0
    return relation


def _identity_relation(anchor_codes: set[str]) -> pd.DataFrame:
    relation = pd.DataFrame(
        {
            "from_code": sorted(anchor_codes),
            "to_code": sorted(anchor_codes),
            "weight": 1.0,
        }
    )
    return relation


def _load_crm_codes(path: Path) -> set[str]:
    data = pd.read_csv(path, dtype=str)
    if "cn_code_2023" not in data.columns:
        raise ValueError("CRM allocations file must include a 'cn_code_2023' column")
    return set(normalize_code_set(data["cn_code_2023"].dropna().tolist()))


def _anchor_codes(
    *,
    groups,
    anchor_year: int,
    backward_steps: list[dict[str, int | str]],
    forward_steps: list[dict[str, int | str]],
    code_universe: dict[int, set[str]],
    scope_mode: str,
) -> set[str]:
    if scope_mode == "observed_universe_implied_identities":
        return set(code_universe[anchor_year])
    if scope_mode != "revised_only":
        raise ValueError(f"Unsupported scope mode '{scope_mode}'")

    anchor_codes: set[str] = set()
    for steps in (backward_steps[:1], forward_steps[:1]):
        if not steps:
            continue
        step = steps[0]
        period = str(step["period"])
        direction = str(step["direction"])
        edges = groups.edges.loc[groups.edges["period"].astype(str) == period]
        if direction == "a_to_b":
            anchor_codes |= normalize_code_set(edges["vintage_a_code"].tolist())
        else:
            anchor_codes |= normalize_code_set(edges["vintage_b_code"].tolist())
    if not anchor_codes:
        raise ValueError("Unable to derive revised-only anchor code universe")
    return anchor_codes


def _step_source_status(
    *,
    step_relation: pd.DataFrame,
    source_year: int,
    target_year: int,
    direction: str,
) -> pd.DataFrame:
    edges = step_relation[["from_code", "to_code"]].drop_duplicates().rename(
        columns={"from_code": "vintage_a_code", "to_code": "vintage_b_code"}
    )
    edges["period"] = f"{source_year}{target_year}"
    edges["vintage_a_year"] = str(source_year)
    edges["vintage_b_year"] = str(target_year)
    step_groups = build_concordance_groups(
        edges[["period", "vintage_a_year", "vintage_b_year", "vintage_a_code", "vintage_b_code"]]
    )
    source_nodes = step_groups.vintage_a_nodes[
        ["period", "vintage_a_code", "group_id"]
    ].drop_duplicates()
    source_nodes = source_nodes.merge(
        step_groups.group_summary[["period", "group_id", "n_vintage_a", "n_vintage_b"]],
        on=["period", "group_id"],
        how="inner",
    )
    source_nodes["step_relationship"] = [
        _classify_relationship(int(n_a), int(n_b))
        for n_a, n_b in zip(
            source_nodes["n_vintage_a"].tolist(),
            source_nodes["n_vintage_b"].tolist(),
        )
    ]
    unknown_relationships = _unknown_relationships_for_direction(direction)
    source_nodes["touched_non_1_to_1_this_step"] = source_nodes["step_relationship"] != "1:1"
    source_nodes["touched_unknown_weight_this_step"] = source_nodes["step_relationship"].isin(
        unknown_relationships
    )
    return source_nodes[
        [
            "vintage_a_code",
            "step_relationship",
            "touched_non_1_to_1_this_step",
            "touched_unknown_weight_this_step",
        ]
    ].rename(columns={"vintage_a_code": "source_code"})


def _final_relationships(
    *,
    relation: pd.DataFrame,
    anchor_year: int,
    compare_year: int,
) -> pd.DataFrame:
    edges = relation[["from_code", "to_code"]].drop_duplicates().rename(
        columns={"from_code": "vintage_a_code", "to_code": "vintage_b_code"}
    )
    edges["period"] = f"{anchor_year}{compare_year}"
    edges["vintage_a_year"] = str(anchor_year)
    edges["vintage_b_year"] = str(compare_year)
    relation_groups = build_concordance_groups(
        edges[["period", "vintage_a_year", "vintage_b_year", "vintage_a_code", "vintage_b_code"]]
    )
    anchor_nodes = relation_groups.vintage_a_nodes[
        ["period", "vintage_a_code", "group_id"]
    ].drop_duplicates()
    anchor_nodes = anchor_nodes.merge(
        relation_groups.group_summary[["period", "group_id", "n_vintage_a", "n_vintage_b"]],
        on=["period", "group_id"],
        how="inner",
    )
    anchor_nodes["final_relationship"] = [
        _classify_relationship(int(n_a), int(n_b))
        for n_a, n_b in zip(
            anchor_nodes["n_vintage_a"].tolist(),
            anchor_nodes["n_vintage_b"].tolist(),
        )
    ]
    return anchor_nodes[["vintage_a_code", "final_relationship"]].rename(
        columns={"vintage_a_code": "anchor_code"}
    )


def _panel_code_exposure(
    *,
    groups,
    panel_direction: str,
    steps: list[dict[str, int | str]],
    anchor_year: int,
    anchor_codes: set[str],
    crm_codes: set[str],
    code_universe: dict[int, set[str]],
    scope_mode: str,
) -> pd.DataFrame:
    current = _identity_relation(anchor_codes)
    status = pd.DataFrame({"anchor_code": sorted(anchor_codes)})
    status["ever_non_1_to_1_step"] = False
    status["ever_unknown_weight_step"] = False
    status["is_crm_code"] = status["anchor_code"].isin(crm_codes)
    rows: list[pd.DataFrame] = []

    for step in steps:
        period = str(step["period"])
        direction = str(step["direction"])
        source_year = int(step["source_year"])
        compare_year = int(step["target_year"])
        step_relation = _step_edges(
            groups=groups,
            period=period,
            direction=direction,
            code_universe=code_universe,
            scope_mode=scope_mode,
            required_source_codes=set(current["to_code"].drop_duplicates().tolist()),
        )
        step_status = _step_source_status(
            step_relation=step_relation,
            source_year=source_year,
            target_year=compare_year,
            direction=direction,
        )
        anchor_source = current[["from_code", "to_code"]].drop_duplicates().rename(
            columns={"from_code": "anchor_code", "to_code": "source_code"}
        )
        anchor_step = anchor_source.merge(step_status, on="source_code", how="left")
        if anchor_step["step_relationship"].isna().any():
            missing_codes = sorted(
                anchor_step.loc[anchor_step["step_relationship"].isna(), "source_code"]
                .drop_duplicates()
                .tolist()
            )
            raise ValueError(
                "Missing step relationship classification for source codes: "
                + ", ".join(missing_codes[:10])
            )
        anchor_touch = (
            anchor_step.groupby("anchor_code", as_index=False)
            .agg(
                touched_non_1_to_1_this_step=("touched_non_1_to_1_this_step", "max"),
                touched_unknown_weight_this_step=("touched_unknown_weight_this_step", "max"),
            )
        )

        status = status.merge(anchor_touch, on="anchor_code", how="left")
        status["touched_non_1_to_1_this_step"] = (
            status["touched_non_1_to_1_this_step"].fillna(False).astype(bool)
        )
        status["touched_unknown_weight_this_step"] = (
            status["touched_unknown_weight_this_step"].fillna(False).astype(bool)
        )
        status["ever_non_1_to_1_step"] = (
            status["ever_non_1_to_1_step"] | status["touched_non_1_to_1_this_step"]
        )
        status["ever_unknown_weight_step"] = (
            status["ever_unknown_weight_step"] | status["touched_unknown_weight_this_step"]
        )

        current, _ = compose_weights(current, step_relation)
        current = _binarize_relation(current)
        final_relationships = _final_relationships(
            relation=current,
            anchor_year=anchor_year,
            compare_year=compare_year,
        )
        year_rows = status.merge(final_relationships, on="anchor_code", how="inner")
        year_rows["analysis_type"] = "crm_revision_exposure"
        year_rows["scope_mode"] = scope_mode
        year_rows["panel_direction"] = panel_direction
        year_rows["anchor_year"] = int(anchor_year)
        year_rows["compare_year"] = int(compare_year)
        year_rows["chain_length"] = abs(int(anchor_year) - int(compare_year))
        year_rows["remained_strict_1_to_1"] = ~year_rows["ever_non_1_to_1_step"]
        rows.append(
            year_rows[
                [
                    "analysis_type",
                    "scope_mode",
                    "panel_direction",
                    "anchor_year",
                    "compare_year",
                    "chain_length",
                    "anchor_code",
                    "is_crm_code",
                    "final_relationship",
                    "touched_non_1_to_1_this_step",
                    "touched_unknown_weight_this_step",
                    "ever_non_1_to_1_step",
                    "ever_unknown_weight_step",
                    "remained_strict_1_to_1",
                ]
            ].copy()
        )
        status = status[
            [
                "anchor_code",
                "is_crm_code",
                "ever_non_1_to_1_step",
                "ever_unknown_weight_step",
            ]
        ].copy()

    if not rows:
        return pd.DataFrame(
            columns=[
                "analysis_type",
                "scope_mode",
                "panel_direction",
                "anchor_year",
                "compare_year",
                "chain_length",
                "anchor_code",
                "is_crm_code",
                "final_relationship",
                "touched_non_1_to_1_this_step",
                "touched_unknown_weight_this_step",
                "ever_non_1_to_1_step",
                "ever_unknown_weight_step",
                "remained_strict_1_to_1",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _summarize(code_exposure: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = [
        "analysis_type",
        "scope_mode",
        "panel_direction",
        "anchor_year",
        "compare_year",
        "chain_length",
    ]
    for keys, group in code_exposure.groupby(group_cols, sort=True):
        key_map = dict(zip(group_cols, keys))
        for population, pop_df in {
            "all_anchor_codes": group,
            "crm_anchor_codes": group.loc[group["is_crm_code"]],
        }.items():
            total_codes = int(len(pop_df))
            for metric in SUMMARY_METRICS:
                n_codes = int(pop_df[metric].sum()) if total_codes else 0
                rows.append(
                    {
                        **key_map,
                        "population": population,
                        "metric": metric,
                        "n_codes": n_codes,
                        "total_codes": total_codes,
                        "share_codes": (
                            n_codes / float(total_codes) if total_codes else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["panel_direction", "compare_year", "population", "metric"]
    ).reset_index(drop=True)


def run_crm_revision_exposure_analysis(
    config: CrmRevisionExposureConfig,
) -> dict[str, Path]:
    groups = load_concordance_groups(
        concordance_path=config.paths.concordance_path,
        sheet_name=config.paths.concordance_sheet,
    )
    years = list(
        range(config.years.backward_end_year, config.years.forward_end_year + 1)
    )
    code_universe = build_code_universe_from_annual(
        annual_base_dir=config.paths.annual_base_dir,
        years=years,
    )
    crm_codes = _load_crm_codes(config.paths.crm_codes_path)

    backward_steps = chain_steps(
        config.years.anchor_year,
        config.years.backward_end_year,
    )
    forward_steps = chain_steps(
        config.years.anchor_year,
        config.years.forward_end_year,
    )
    anchor_codes = _anchor_codes(
        groups=groups,
        anchor_year=config.years.anchor_year,
        backward_steps=backward_steps,
        forward_steps=forward_steps,
        code_universe=code_universe,
        scope_mode=config.scope.mode,
    )
    code_rows = [
        _panel_code_exposure(
            groups=groups,
            panel_direction="backward",
            steps=backward_steps,
            anchor_year=config.years.anchor_year,
            anchor_codes=anchor_codes,
            crm_codes=crm_codes,
            code_universe=code_universe,
            scope_mode=config.scope.mode,
        )
    ]
    if forward_steps:
        code_rows.append(
            _panel_code_exposure(
                groups=groups,
                panel_direction="forward",
                steps=forward_steps,
                anchor_year=config.years.anchor_year,
                anchor_codes=anchor_codes,
                crm_codes=crm_codes,
                code_universe=code_universe,
                scope_mode=config.scope.mode,
            )
        )
    code_exposure = pd.concat(code_rows, ignore_index=True)
    summary = _summarize(code_exposure)
    benchmark = summary.loc[
        (
            (summary["panel_direction"] == "backward")
            & (summary["compare_year"].isin(config.years.benchmark_backward_years))
        )
        | (
            (summary["panel_direction"] == "forward")
            & (summary["compare_year"].isin(config.years.benchmark_forward_years))
        )
    ].copy()
    benchmark = benchmark.sort_values(
        ["panel_direction", "compare_year", "population", "metric"]
    ).reset_index(drop=True)

    _write_csv(code_exposure, config.output.code_exposure_csv)
    _write_csv(summary, config.output.summary_csv)
    _write_csv(benchmark, config.output.benchmark_summary_csv)
    plot_crm_revision_exposure_panels(
        data=summary,
        output_path=config.plot.output_path,
        title=config.plot.title,
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
    )

    return {
        "output_plot": config.plot.output_path,
        "summary_csv": config.output.summary_csv,
        "code_exposure_csv": config.output.code_exposure_csv,
        "benchmark_summary_csv": config.output.benchmark_summary_csv,
    }
