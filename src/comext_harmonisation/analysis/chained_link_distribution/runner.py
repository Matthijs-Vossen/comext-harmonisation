"""Chained structural CN link-distribution analysis in the spirit of LT Table 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...chaining.composition import compose_weights
from ...chaining.engine import build_code_universe_from_annual
from ...concordance.groups import build_concordance_groups
from ...core.codes import normalize_code_set
from ...estimation.runner import load_concordance_groups
from ..common.plotting import (
    plot_chained_link_distribution_bar_panels,
    plot_chained_link_distribution_panels,
)
from ..common.steps import chain_steps
from ..config import ChainedLinkDistributionConfig


RELATIONSHIP_ORDER = ["1:1", "m:1", "1:n", "m:n"]


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


def _step_edges(
    *,
    groups,
    period: str,
    direction: str,
    code_universe: dict[int, set[str]],
    scope_mode: str,
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

    step = step.drop_duplicates().reset_index(drop=True)
    step["weight"] = 1.0
    return step


def _binarize_relation(df: pd.DataFrame) -> pd.DataFrame:
    relation = df[["from_code", "to_code"]].drop_duplicates().reset_index(drop=True)
    relation["weight"] = 1.0
    return relation


def _anchor_codes_for_panel(
    *,
    groups,
    panel_steps: list[dict[str, int | str]],
    code_universe: dict[int, set[str]],
    anchor_year: int,
    scope_mode: str,
) -> set[str]:
    if scope_mode == "observed_universe_implied_identities":
        return set(code_universe[anchor_year])
    if scope_mode != "revised_only":
        raise ValueError(f"Unsupported scope mode '{scope_mode}'")

    first_step = panel_steps[0]
    period = str(first_step["period"])
    direction = str(first_step["direction"])
    edges = groups.edges.loc[groups.edges["period"].astype(str) == period]
    if direction == "a_to_b":
        return normalize_code_set(edges["vintage_a_code"].tolist())
    return normalize_code_set(edges["vintage_b_code"].tolist())


def _relation_summary_for_year(
    *,
    relation: pd.DataFrame,
    panel_direction: str,
    anchor_year: int,
    compare_year: int,
    scope_mode: str,
    total_anchor_codes: int,
) -> pd.DataFrame:
    period = f"{anchor_year}{compare_year}"
    edges = relation[["from_code", "to_code"]].drop_duplicates().rename(
        columns={"from_code": "vintage_a_code", "to_code": "vintage_b_code"}
    )
    edges["period"] = period
    edges["vintage_a_year"] = str(anchor_year)
    edges["vintage_b_year"] = str(compare_year)
    groups = build_concordance_groups(
        edges[["period", "vintage_a_year", "vintage_b_year", "vintage_a_code", "vintage_b_code"]]
    )
    anchor_nodes = groups.vintage_a_nodes[
        ["period", "vintage_a_code", "group_id"]
    ].drop_duplicates()
    anchor_nodes = anchor_nodes.merge(
        groups.group_summary[["period", "group_id", "n_vintage_a", "n_vintage_b"]],
        on=["period", "group_id"],
        how="inner",
    )
    anchor_nodes = anchor_nodes.drop_duplicates(subset=["vintage_a_code"]).reset_index(drop=True)

    if int(anchor_nodes["vintage_a_code"].nunique()) != int(total_anchor_codes):
        raise ValueError(
            "Anchor denominator drifted during chained link-distribution analysis: "
            f"anchor_year={anchor_year}, compare_year={compare_year}, "
            f"expected={total_anchor_codes}, observed={anchor_nodes['vintage_a_code'].nunique()}"
        )

    anchor_nodes["relationship"] = [
        _classify_relationship(int(n_a), int(n_b))
        for n_a, n_b in zip(
            anchor_nodes["n_vintage_a"].tolist(),
            anchor_nodes["n_vintage_b"].tolist(),
        )
    ]
    counts = (
        anchor_nodes.groupby("relationship", as_index=False)
        .agg(n_anchor_codes=("vintage_a_code", "nunique"))
    )
    counts = counts.set_index("relationship")["n_anchor_codes"].to_dict()

    rows = []
    for relationship in RELATIONSHIP_ORDER:
        n_anchor_codes = int(counts.get(relationship, 0))
        rows.append(
            {
                "panel_direction": panel_direction,
                "anchor_year": int(anchor_year),
                "compare_year": int(compare_year),
                "chain_length": abs(int(anchor_year) - int(compare_year)),
                "scope_mode": scope_mode,
                "relationship": relationship,
                "n_anchor_codes": n_anchor_codes,
                "total_anchor_codes": int(total_anchor_codes),
                "share_anchor_codes": n_anchor_codes / float(total_anchor_codes),
            }
        )
    return pd.DataFrame(rows)


def _identity_relation(anchor_codes: set[str]) -> pd.DataFrame:
    relation = pd.DataFrame(
        {
            "from_code": sorted(anchor_codes),
            "to_code": sorted(anchor_codes),
            "weight": 1.0,
        }
    )
    return relation


def _panel_summary(
    *,
    groups,
    panel_direction: str,
    anchor_year: int,
    final_year: int,
    code_universe: dict[int, set[str]],
    scope_mode: str,
) -> pd.DataFrame:
    steps = chain_steps(anchor_year, final_year)
    anchor_codes = _anchor_codes_for_panel(
        groups=groups,
        panel_steps=steps,
        code_universe=code_universe,
        anchor_year=anchor_year,
        scope_mode=scope_mode,
    )
    current = _identity_relation(anchor_codes)
    total_anchor_codes = len(anchor_codes)
    rows: list[pd.DataFrame] = []

    for step in steps:
        period = str(step["period"])
        direction = str(step["direction"])
        compare_year = int(step["target_year"])
        step_relation = _step_edges(
            groups=groups,
            period=period,
            direction=direction,
            code_universe=code_universe,
            scope_mode=scope_mode,
        )
        current, _ = compose_weights(current, step_relation)
        current = _binarize_relation(current)
        rows.append(
            _relation_summary_for_year(
                relation=current,
                panel_direction=panel_direction,
                anchor_year=anchor_year,
                compare_year=compare_year,
                scope_mode=scope_mode,
                total_anchor_codes=total_anchor_codes,
            )
        )

    return pd.concat(rows, ignore_index=True)


def run_chained_link_distribution_analysis(
    config: ChainedLinkDistributionConfig,
) -> dict[str, Path]:
    groups = load_concordance_groups(
        concordance_path=config.paths.concordance_path,
        sheet_name=config.paths.concordance_sheet,
    )
    years = list(range(config.years.forward_anchor, config.years.backward_anchor + 1))
    code_universe = build_code_universe_from_annual(
        annual_base_dir=config.paths.annual_base_dir,
        years=years,
    )

    backward = _panel_summary(
        groups=groups,
        panel_direction="backward",
        anchor_year=config.years.backward_anchor,
        final_year=config.years.forward_anchor,
        code_universe=code_universe,
        scope_mode=config.scope.mode,
    )
    forward = _panel_summary(
        groups=groups,
        panel_direction="forward",
        anchor_year=config.years.forward_anchor,
        final_year=config.years.backward_anchor,
        code_universe=code_universe,
        scope_mode=config.scope.mode,
    )
    summary = pd.concat([backward, forward], ignore_index=True).sort_values(
        ["panel_direction", "compare_year", "relationship"]
    ).reset_index(drop=True)

    _write_csv(summary, config.output.summary_csv)
    plot_chained_link_distribution_panels(
        data=summary,
        output_path=config.plot.output_path,
        title=config.plot.title,
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
    )
    plot_chained_link_distribution_bar_panels(
        data=summary,
        output_path=config.plot.bar_output_path,
        title=config.plot.title,
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
    )

    return {
        "output_plot": config.plot.output_path,
        "output_plot_bars": config.plot.bar_output_path,
        "summary_csv": config.output.summary_csv,
    }
