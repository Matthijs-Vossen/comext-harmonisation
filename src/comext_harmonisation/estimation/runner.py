"""Run LT weight estimation for a concordance period and persist outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..concordance import read_concordance_xls
from .matrices import GroupMatrices, build_group_matrices
from .shares import ANNUAL_DATA_DIR, EstimationShares, prepare_estimation_shares_for_period
from .solver import estimate_weights
from ..groups import ConcordanceGroups, build_concordance_groups
from ..mappings import build_deterministic_mappings, get_ambiguous_group_summary
from ..weights import DEFAULT_WEIGHTS_DIR, empty_weight_table


DEFAULT_CONCORDANCE_PATH = Path("data/concordances/CN_concordances_1988_2025_XLS_FORMAT.xls")
DEFAULT_DIAGNOSTICS_DIR = Path("outputs/diagnostics")
DEFAULT_SUMMARIES_DIR = Path("outputs/summaries")


@dataclass(frozen=True)
class RunnerOutputs:
    period: str
    direction: str
    weights_ambiguous: pd.DataFrame
    weights_deterministic: pd.DataFrame
    diagnostics: pd.DataFrame
    summary: pd.DataFrame
    weights_path: Path
    deterministic_path: Path
    diagnostics_path: Path
    summary_csv_path: Path
    summary_txt_path: Path


def load_concordance_groups(
    *,
    concordance_path: Path = DEFAULT_CONCORDANCE_PATH,
    sheet_name: str | int | None = None,
) -> ConcordanceGroups:
    """Load the CN concordance and build group metadata."""
    concordance = read_concordance_xls(str(concordance_path), sheet_name=sheet_name)
    return build_concordance_groups(concordance)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_summary_text(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        path.write_text("", encoding="utf-8")
        return
    row = summary.iloc[0].to_dict()
    lines = [f"{key}: {value}" for key, value in row.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sort_weights(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(["period", "from_code", "to_code", "group_id"]).reset_index(drop=True)


def _build_group_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    groups: ConcordanceGroups,
    matrices: dict[str, GroupMatrices],
    period: str,
) -> pd.DataFrame:
    desired = [
        "group_id",
        "period",
        "status",
        "objective",
        "n_vars",
        "n_obs",
        "n_pairs",
        "n_codes_a",
        "n_codes_b",
        "n_edges",
        "min_weight",
        "max_weight",
        "max_row_sum_dev",
    ]
    if diagnostics.empty:
        return pd.DataFrame(columns=desired)

    group_summary = groups.group_summary.loc[groups.group_summary["period"] == period].copy()
    group_summary = group_summary.rename(
        columns={"n_vintage_a": "n_codes_a", "n_vintage_b": "n_codes_b"}
    )

    pair_rows = [
        {"group_id": group_id, "n_pairs": len(group.pairs)} for group_id, group in matrices.items()
    ]
    pairs_df = pd.DataFrame(pair_rows)

    diagnostics = diagnostics.merge(
        group_summary[["period", "group_id", "n_codes_a", "n_codes_b", "n_edges"]],
        on=["period", "group_id"],
        how="left",
    ).merge(pairs_df, on="group_id", how="left")

    diagnostics["n_pairs"] = diagnostics["n_pairs"].fillna(diagnostics["n_obs"]).astype(int)

    ordered = [col for col in desired if col in diagnostics.columns]
    return diagnostics[ordered]


def _build_run_summary(
    *,
    period: str,
    direction: str,
    groups: ConcordanceGroups,
    estimation: EstimationShares,
    diagnostics: pd.DataFrame,
    weights_ambiguous: pd.DataFrame,
    weights_deterministic: pd.DataFrame,
    started_at: datetime,
    ended_at: datetime,
) -> pd.DataFrame:
    ambiguous = get_ambiguous_group_summary(groups, direction)
    ambiguous = ambiguous.loc[ambiguous["period"] == period]
    n_groups_total = len(ambiguous)
    n_groups_with_data = len(estimation.group_totals.loc[estimation.group_totals["skip_reason"] == ""])

    if diagnostics.empty:
        solved = 0
        total_pairs = 0
        total_obs = 0
        max_row_sum_dev_min = 0.0
        max_row_sum_dev_max = 0.0
        max_row_sum_dev_mean = 0.0
    else:
        status = diagnostics["status"].str.lower()
        solved = int(status.str.startswith("solved").sum())
        total_pairs = int(diagnostics["n_pairs"].sum())
        total_obs = int(diagnostics["n_obs"].sum())
        max_row_sum_dev_min = float(diagnostics["max_row_sum_dev"].min())
        max_row_sum_dev_max = float(diagnostics["max_row_sum_dev"].max())
        max_row_sum_dev_mean = float(diagnostics["max_row_sum_dev"].mean())

    summary = {
        "period": period,
        "direction": direction,
        "n_groups_total": n_groups_total,
        "n_groups_with_data": n_groups_with_data,
        "n_groups_solved": solved,
        "n_groups_failed": n_groups_with_data - solved,
        "n_groups_skipped": n_groups_total - n_groups_with_data,
        "n_weight_rows_ambiguous": len(weights_ambiguous),
        "n_weight_rows_deterministic": len(weights_deterministic),
        "total_pairs": total_pairs,
        "total_obs": total_obs,
        "max_row_sum_dev_min": max_row_sum_dev_min,
        "max_row_sum_dev_max": max_row_sum_dev_max,
        "max_row_sum_dev_mean": max_row_sum_dev_mean,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": (ended_at - started_at).total_seconds(),
    }
    return pd.DataFrame([summary])


def run_weight_estimation_for_period(
    *,
    period: str,
    direction: str = "a_to_b",
    concordance_path: Path = DEFAULT_CONCORDANCE_PATH,
    concordance_sheet: str | int | None = None,
    annual_base_dir: Path = ANNUAL_DATA_DIR,
    flow: str = "1",
    exclude_codes: Optional[Iterable[str]] = None,
    exclude_aggregate_codes: bool = True,
    output_weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    output_diagnostics_dir: Path = DEFAULT_DIAGNOSTICS_DIR,
    output_summaries_dir: Path = DEFAULT_SUMMARIES_DIR,
    output_dir: Optional[Path] = None,
    fail_on_status: bool = True,
) -> RunnerOutputs:
    """Run the full estimation pipeline for one concordance period."""
    started_at = datetime.now(timezone.utc)

    groups = load_concordance_groups(
        concordance_path=concordance_path,
        sheet_name=concordance_sheet,
    )

    estimation = prepare_estimation_shares_for_period(
        period=period,
        groups=groups,
        direction=direction,
        base_dir=annual_base_dir,
        flow=flow,
        exclude_codes=exclude_codes,
        exclude_aggregate_codes=exclude_aggregate_codes,
    )

    matrices = build_group_matrices(estimation, groups=groups, dense=False)
    weights_ambiguous, diagnostics = estimate_weights(
        estimation=estimation,
        matrices=matrices,
        groups=groups,
        direction=direction,
    )

    weights_ambiguous = _sort_weights(
        weights_ambiguous if not weights_ambiguous.empty else empty_weight_table()
    )
    weights_deterministic = _sort_weights(
        build_deterministic_mappings(groups, direction).loc[lambda df: df["period"] == period]
    )
    diagnostics = _build_group_diagnostics(
        diagnostics,
        groups=groups,
        matrices=matrices,
        period=period,
    )

    if fail_on_status and not diagnostics.empty:
        unsolved = diagnostics.loc[~diagnostics["status"].str.lower().str.startswith("solved")]
        if not unsolved.empty:
            failures = unsolved[["group_id", "status"]].to_dict(orient="records")
            raise RuntimeError(f"Solver failed for period {period}: {failures}")

    ended_at = datetime.now(timezone.utc)
    summary = _build_run_summary(
        period=period,
        direction=direction,
        groups=groups,
        estimation=estimation,
        diagnostics=diagnostics,
        weights_ambiguous=weights_ambiguous,
        weights_deterministic=weights_deterministic,
        started_at=started_at,
        ended_at=ended_at,
    )

    if output_dir is not None:
        output_weights_dir = output_dir / "weights"
        output_diagnostics_dir = output_dir / "diagnostics"
        output_summaries_dir = output_dir / "summaries"

    weights_path = output_weights_dir / f"weights_ambiguous_{period}_{direction}.csv"
    deterministic_path = output_weights_dir / f"weights_deterministic_{period}_{direction}.csv"
    diagnostics_path = output_diagnostics_dir / f"diagnostics_{period}_{direction}.csv"
    summary_csv_path = output_summaries_dir / f"run_summary_{period}_{direction}.csv"
    summary_txt_path = output_summaries_dir / f"run_summary_{period}_{direction}.txt"

    _write_csv(weights_ambiguous, weights_path)
    _write_csv(weights_deterministic, deterministic_path)
    _write_csv(diagnostics, diagnostics_path)
    _write_csv(summary, summary_csv_path)
    _write_summary_text(summary, summary_txt_path)

    return RunnerOutputs(
        period=period,
        direction=direction,
        weights_ambiguous=weights_ambiguous,
        weights_deterministic=weights_deterministic,
        diagnostics=diagnostics,
        summary=summary,
        weights_path=weights_path,
        deterministic_path=deterministic_path,
        diagnostics_path=diagnostics_path,
        summary_csv_path=summary_csv_path,
        summary_txt_path=summary_txt_path,
    )
