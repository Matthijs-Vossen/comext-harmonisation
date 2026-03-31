"""Adjacent CN link-distribution analysis in the spirit of LT Table 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...chaining.engine import build_code_universe_from_annual
from ...estimation.runner import load_concordance_groups
from ..config import LinkDistributionConfig


RELATIONSHIP_ORDER = ["1:1", "m:1", "1:n", "m:n"]
FOCAL_SIDE_ORDER = ["vintage_a", "vintage_b"]
DIRECTION_LABELS = {
    "vintage_a": "a_to_b",
    "vintage_b": "b_to_a",
}


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _validate_adjacent_periods(periods: pd.Series) -> None:
    normalized = periods.astype(str).str.strip().drop_duplicates()
    for period in normalized:
        if len(period) != 8 or not period.isdigit():
            raise ValueError(f"Invalid period '{period}'")
        year_a = int(period[:4])
        year_b = int(period[4:])
        if year_b != year_a + 1:
            raise ValueError(
                "link_distribution: all concordance periods must be adjacent annual breaks; "
                f"found '{period}'."
            )


def _classify_relationship(n_focal: int, n_other: int) -> str:
    if n_focal == 1 and n_other == 1:
        return "1:1"
    if n_focal > 1 and n_other == 1:
        return "m:1"
    if n_focal == 1 and n_other > 1:
        return "1:n"
    return "m:n"


def _relationship_has_unknown_weight(relationship: str, direction: str) -> bool:
    if direction == "a_to_b":
        return relationship in {"1:n", "m:n"}
    if direction == "b_to_a":
        return relationship in {"m:1", "m:n"}
    raise ValueError(f"Unsupported direction '{direction}'")


def _scope_label(scope_mode: str) -> str:
    if scope_mode == "revised_only":
        return "revised_only"
    if scope_mode == "observed_universe_implied_identities":
        return "observed_universe_implied_identities"
    raise ValueError(f"Unsupported scope mode '{scope_mode}'")


def _available_periods_for_observed_universe(groups, *, annual_base_dir: Path) -> set[str]:
    period_meta = groups.edges[
        ["period", "vintage_a_year", "vintage_b_year"]
    ].drop_duplicates()
    available_periods: set[str] = set()
    for row in period_meta.itertuples(index=False):
        period = str(row.period)
        vintage_a_year = int(str(row.vintage_a_year))
        vintage_b_year = int(str(row.vintage_b_year))
        path_a = annual_base_dir / f"comext_{vintage_a_year}.parquet"
        path_b = annual_base_dir / f"comext_{vintage_b_year}.parquet"
        if path_a.exists() and path_b.exists():
            available_periods.add(period)
    return available_periods


def _build_revised_only_focal_code_rows(groups, *, scope_mode: str) -> pd.DataFrame:
    group_sizes = groups.group_summary[
        ["period", "group_id", "n_vintage_a", "n_vintage_b"]
    ].drop_duplicates()
    scope_label = _scope_label(scope_mode)

    focal_frames: list[pd.DataFrame] = []
    for focal_side, code_column, year_column, other_year_column in (
        ("vintage_a", "vintage_a_code", "vintage_a_year", "vintage_b_year"),
        ("vintage_b", "vintage_b_code", "vintage_b_year", "vintage_a_year"),
    ):
        focal = groups.edges[
            ["period", "group_id", year_column, other_year_column, code_column]
        ].drop_duplicates()
        focal = focal.merge(group_sizes, on=["period", "group_id"], how="inner")
        focal = focal.rename(
            columns={
                code_column: "focal_code",
                year_column: "focal_year",
                other_year_column: "other_year",
            }
        )
        focal["period"] = focal["period"].astype(str)
        focal["focal_year"] = focal["focal_year"].astype(str)
        focal["other_year"] = focal["other_year"].astype(str)
        if focal_side == "vintage_a":
            focal["n_focal_codes"] = focal["n_vintage_a"]
            focal["n_other_codes"] = focal["n_vintage_b"]
        else:
            focal["n_focal_codes"] = focal["n_vintage_b"]
            focal["n_other_codes"] = focal["n_vintage_a"]

        focal["analysis_type"] = "link_distribution"
        focal["scope_mode"] = scope_mode
        focal["scope_label"] = scope_label
        focal["focal_side"] = focal_side
        focal["direction"] = DIRECTION_LABELS[focal_side]
        focal["relationship"] = [
            _classify_relationship(int(n_focal), int(n_other))
            for n_focal, n_other in zip(
                focal["n_focal_codes"].tolist(), focal["n_other_codes"].tolist()
            )
        ]
        focal["unknown_conversion_weight"] = [
            _relationship_has_unknown_weight(relationship, DIRECTION_LABELS[focal_side])
            for relationship in focal["relationship"].tolist()
        ]
        focal_frames.append(
            focal[
                [
                    "analysis_type",
                    "scope_mode",
                    "scope_label",
                    "period",
                    "focal_side",
                    "direction",
                    "focal_year",
                    "other_year",
                    "group_id",
                    "focal_code",
                    "n_focal_codes",
                    "n_other_codes",
                    "relationship",
                    "unknown_conversion_weight",
                ]
            ]
        )

    result = pd.concat(focal_frames, ignore_index=True)
    return result.sort_values(
        ["period", "focal_side", "focal_code", "group_id"]
    ).reset_index(drop=True)


def _build_observed_identity_rows(
    groups,
    *,
    annual_base_dir: Path,
    scope_mode: str,
    available_periods: set[str],
) -> pd.DataFrame:
    period_meta = groups.edges[
        ["period", "vintage_a_year", "vintage_b_year"]
    ].drop_duplicates()
    period_meta = period_meta.loc[period_meta["period"].astype(str).isin(available_periods)].reset_index(
        drop=True
    )
    if period_meta.empty:
        return pd.DataFrame(
            columns=[
                "analysis_type",
                "scope_mode",
                "scope_label",
                "period",
                "focal_side",
                "direction",
                "focal_year",
                "other_year",
                "group_id",
                "focal_code",
                "n_focal_codes",
                "n_other_codes",
                "relationship",
                "unknown_conversion_weight",
            ]
        )
    years = sorted(
        {
            int(year)
            for year in pd.concat(
                [period_meta["vintage_a_year"], period_meta["vintage_b_year"]],
                ignore_index=True,
            )
            .astype(str)
            .tolist()
        }
    )
    code_universe = build_code_universe_from_annual(
        annual_base_dir=annual_base_dir,
        years=years,
    )

    revised_a = (
        groups.edges.groupby("period", sort=False)["vintage_a_code"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    revised_b = (
        groups.edges.groupby("period", sort=False)["vintage_b_code"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )

    rows: list[dict[str, object]] = []
    scope_label = _scope_label(scope_mode)
    for row in period_meta.itertuples(index=False):
        period = str(row.period)
        vintage_a_year = int(str(row.vintage_a_year))
        vintage_b_year = int(str(row.vintage_b_year))
        unchanged_codes = (
            (code_universe[vintage_a_year] & code_universe[vintage_b_year])
            - revised_a.get(period, set())
            - revised_b.get(period, set())
        )
        for code in sorted(unchanged_codes):
            rows.append(
                {
                    "analysis_type": "link_distribution",
                    "scope_mode": scope_mode,
                    "scope_label": scope_label,
                    "period": period,
                    "focal_side": "vintage_a",
                    "direction": "a_to_b",
                    "focal_year": str(vintage_a_year),
                    "other_year": str(vintage_b_year),
                    "group_id": f"{period}_identity_{code}",
                    "focal_code": code,
                    "n_focal_codes": 1,
                    "n_other_codes": 1,
                    "relationship": "1:1",
                    "unknown_conversion_weight": False,
                }
            )
            rows.append(
                {
                    "analysis_type": "link_distribution",
                    "scope_mode": scope_mode,
                    "scope_label": scope_label,
                    "period": period,
                    "focal_side": "vintage_b",
                    "direction": "b_to_a",
                    "focal_year": str(vintage_b_year),
                    "other_year": str(vintage_a_year),
                    "group_id": f"{period}_identity_{code}",
                    "focal_code": code,
                    "n_focal_codes": 1,
                    "n_other_codes": 1,
                    "relationship": "1:1",
                    "unknown_conversion_weight": False,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "analysis_type",
                "scope_mode",
                "scope_label",
                "period",
                "focal_side",
                "direction",
                "focal_year",
                "other_year",
                "group_id",
                "focal_code",
                "n_focal_codes",
                "n_other_codes",
                "relationship",
                "unknown_conversion_weight",
            ]
        )

    return pd.DataFrame(rows)


def _build_focal_code_rows(
    groups,
    *,
    scope_mode: str,
    annual_base_dir: Path,
) -> pd.DataFrame:
    focal_codes = _build_revised_only_focal_code_rows(groups, scope_mode=scope_mode)
    if scope_mode == "revised_only":
        return focal_codes

    available_periods = _available_periods_for_observed_universe(
        groups,
        annual_base_dir=annual_base_dir,
    )
    focal_codes = focal_codes.loc[focal_codes["period"].astype(str).isin(available_periods)].reset_index(
        drop=True
    )
    identity_rows = _build_observed_identity_rows(
        groups,
        annual_base_dir=annual_base_dir,
        scope_mode=scope_mode,
        available_periods=available_periods,
    )
    combined = pd.concat([focal_codes, identity_rows], ignore_index=True)
    return combined.sort_values(
        ["period", "focal_side", "focal_code", "group_id"]
    ).reset_index(drop=True)


def _build_summary(focal_codes: pd.DataFrame) -> pd.DataFrame:
    counts = (
        focal_codes.groupby(
            [
                "analysis_type",
                "scope_mode",
                "scope_label",
                "period",
                "focal_side",
                "direction",
                "focal_year",
                "other_year",
                "relationship",
                "unknown_conversion_weight",
            ],
            as_index=False,
            sort=False,
        )
        .agg(n_focal_codes=("focal_code", "count"))
    )
    totals = (
        focal_codes.groupby(
            [
                "analysis_type",
                "scope_mode",
                "scope_label",
                "period",
                "focal_side",
                "direction",
                "focal_year",
                "other_year",
            ],
            as_index=False,
            sort=False,
        )
        .agg(total_focal_codes=("focal_code", "count"))
    )
    summary = counts.merge(
        totals,
        on=[
            "analysis_type",
            "scope_mode",
            "scope_label",
            "period",
            "focal_side",
            "direction",
            "focal_year",
            "other_year",
        ],
        how="left",
    )
    summary["share_focal_codes"] = summary["n_focal_codes"] / summary["total_focal_codes"]
    relationship_rank = {name: idx for idx, name in enumerate(RELATIONSHIP_ORDER)}
    focal_rank = {name: idx for idx, name in enumerate(FOCAL_SIDE_ORDER)}
    summary["_relationship_rank"] = summary["relationship"].map(relationship_rank).fillna(999)
    summary["_focal_rank"] = summary["focal_side"].map(focal_rank).fillna(999)
    summary = summary.sort_values(
        ["period", "_focal_rank", "_relationship_rank"]
    ).drop(columns=["_relationship_rank", "_focal_rank"])
    return summary.reset_index(drop=True)


def run_link_distribution_analysis(config: LinkDistributionConfig) -> dict[str, str]:
    groups = load_concordance_groups(
        concordance_path=config.paths.concordance_path,
        sheet_name=config.paths.concordance_sheet,
    )
    _validate_adjacent_periods(groups.edges["period"])

    focal_codes = _build_focal_code_rows(
        groups,
        scope_mode=config.scope.mode,
        annual_base_dir=config.paths.annual_base_dir,
    )
    summary = _build_summary(focal_codes)

    _write_csv(summary, config.output.summary_csv)
    _write_csv(focal_codes, config.output.focal_codes_csv)

    return {
        "summary_csv": str(config.output.summary_csv),
        "focal_codes_csv": str(config.output.focal_codes_csv),
    }
