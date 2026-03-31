"""Leave-one-bin-out sampling robustness for adjacent LT weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common.shares import filter_partners
from ..common.plotting import plot_sampling_robustness_panels
from ..common.progress import progress
from ..config import SamplingRobustnessConfig
from ...estimation.matrices import build_group_matrices
from ...estimation.runner import load_concordance_groups
from ...estimation.shares import prepare_estimation_shares_from_frames
from ...estimation.solver import estimate_weights


RAW_COLUMNS = ["REPORTER", "PARTNER", "TRADE_TYPE", "PRODUCT_NC", "FLOW"]


def _load_annual_frame(
    *,
    annual_base_dir: Path,
    year: int,
    measure: str,
    exclude_reporters: tuple[str, ...] | list[str],
    exclude_partners: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    path = annual_base_dir / f"comext_{year}.parquet"
    columns = RAW_COLUMNS + [measure]
    data = pd.read_parquet(path, columns=columns)
    return filter_partners(
        data,
        exclude_reporters=exclude_reporters,
        exclude_partners=exclude_partners,
    )


def _sorted_weights(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(["group_id", "from_code", "to_code"]).reset_index(drop=True)


def _reported_ambiguous_weights(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    counts = (
        df.groupby(["group_id", "from_code"], as_index=False, sort=False)
        .size()
        .rename(columns={"size": "n_targets"})
    )
    reported = df.merge(counts, on=["group_id", "from_code"], how="left")
    reported = reported[reported["n_targets"] > 1].copy()
    return reported.drop(columns=["n_targets"]).reset_index(drop=True)


def _require_solved(diagnostics: pd.DataFrame, *, label: str) -> None:
    if diagnostics.empty:
        raise RuntimeError(f"{label}: estimation returned no diagnostics")
    unsolved = diagnostics.loc[~diagnostics["status"].str.lower().str.startswith("solved")]
    if unsolved.empty:
        return
    failures = unsolved[["group_id", "status"]].to_dict(orient="records")
    raise RuntimeError(f"{label}: solver failed for one or more groups: {failures}")


def _estimate_ambiguous_weights(
    *,
    groups,
    period: str,
    direction: str,
    measure: str,
    flow_code: str,
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
) -> tuple:
    estimation = prepare_estimation_shares_from_frames(
        period=period,
        groups=groups,
        direction=direction,
        data_a=data_a,
        data_b=data_b,
        measure=measure,
        flow=flow_code,
        exclude_aggregate_codes=True,
    )
    matrices = build_group_matrices(estimation, groups=groups, dense=False, max_workers=1)
    weights, diagnostics = estimate_weights(
        estimation=estimation,
        matrices=matrices,
        groups=groups,
        direction=direction,
        max_workers=1,
    )
    _require_solved(diagnostics, label=f"sampling_robustness {period} {direction}")
    return estimation, matrices, _sorted_weights(weights), diagnostics


def _group_maps_for_period(*, groups, period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges_period = groups.edges[groups.edges["period"] == period]
    if edges_period.empty:
        raise ValueError(f"No concordance edges found for period {period}")
    a_map = (
        edges_period[["vintage_a_code", "group_id"]]
        .drop_duplicates()
        .rename(columns={"vintage_a_code": "PRODUCT_NC"})
    )
    b_map = (
        edges_period[["vintage_b_code", "group_id"]]
        .drop_duplicates()
        .rename(columns={"vintage_b_code": "PRODUCT_NC"})
    )
    return a_map, b_map


def _build_bin_assignments(*, matrices: dict[str, object], n_bins: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for group_id, matrix in matrices.items():
        pairs = matrix.pairs.copy()
        pairs["group_id"] = group_id
        rows.append(pairs[["group_id", "REPORTER", "PARTNER"]])
    if not rows:
        raise ValueError("sampling_robustness: no estimation pairs available for bin assignment")
    assignments = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["group_id", "REPORTER", "PARTNER"])
        .reset_index(drop=True)
    )
    n_rows = len(assignments)
    if n_bins > n_rows:
        raise ValueError(
            "sampling_robustness: run.n_bins must not exceed the number of estimation observations"
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_rows)
    bin_ids = np.empty(n_rows, dtype=int)
    bin_ids[shuffled] = np.arange(n_rows) % n_bins
    assignments["bin_id"] = bin_ids.astype(int)
    assignments["holdout_key"] = (
        assignments["group_id"].astype(str)
        + "|"
        + assignments["REPORTER"].astype(str)
        + "|"
        + assignments["PARTNER"].astype(str)
    )
    return assignments


def _annotate_for_holdout(df: pd.DataFrame, *, group_map: pd.DataFrame) -> pd.DataFrame:
    annotated = df.merge(group_map, on="PRODUCT_NC", how="left")
    annotated["group_id"] = annotated["group_id"].astype("string")
    annotated["holdout_key"] = (
        annotated["group_id"].fillna("")
        + "|"
        + annotated["REPORTER"].astype(str)
        + "|"
        + annotated["PARTNER"].astype(str)
    )
    return annotated


def _filter_holdout_rows(
    df: pd.DataFrame,
    *,
    holdout_keys: set[str],
    measure: str,
) -> pd.DataFrame:
    keep_mask = ~df["holdout_key"].isin(holdout_keys)
    return df.loc[keep_mask, RAW_COLUMNS + [measure]].copy()


def _product_group_shares(shares: pd.DataFrame, *, code_col: str, value_col: str) -> pd.DataFrame:
    return (
        shares.groupby(["group_id", code_col], as_index=False, sort=False)["share"]
        .sum()
        .rename(columns={code_col: value_col, "share": f"{value_col}_group_share"})
    )


def _importance_frame(*, estimation, direction: str, full_weights: pd.DataFrame) -> pd.DataFrame:
    if direction == "a_to_b":
        source = _product_group_shares(
            estimation.shares_a,
            code_col="vintage_a_code",
            value_col="from_code",
        )
        target = _product_group_shares(
            estimation.shares_b,
            code_col="vintage_b_code",
            value_col="to_code",
        )
    else:
        source = _product_group_shares(
            estimation.shares_b,
            code_col="vintage_b_code",
            value_col="from_code",
        )
        target = _product_group_shares(
            estimation.shares_a,
            code_col="vintage_a_code",
            value_col="to_code",
        )
    importance = full_weights[["group_id", "from_code", "to_code"]].drop_duplicates()
    importance = importance.merge(source, on=["group_id", "from_code"], how="left")
    importance = importance.merge(target, on=["group_id", "to_code"], how="left")
    importance["from_code_group_share"] = (
        pd.to_numeric(importance["from_code_group_share"], errors="coerce").fillna(0.0)
    )
    importance["to_code_group_share"] = (
        pd.to_numeric(importance["to_code_group_share"], errors="coerce").fillna(0.0)
    )
    importance["importance_product"] = (
        importance["from_code_group_share"] * importance["to_code_group_share"]
    )
    return importance


def _build_link_summary(
    *,
    full_weights: pd.DataFrame,
    run_weights: pd.DataFrame,
    importance: pd.DataFrame,
    n_runs: int,
) -> pd.DataFrame:
    group_cols = ["group_id", "from_code", "to_code"]
    baseline = full_weights[group_cols + ["weight"]].rename(columns={"weight": "full_weight"})
    aggregates = (
        run_weights.groupby(group_cols, as_index=False, sort=False)
        .agg(
            min_weight=("weight", "min"),
            max_weight=("weight", "max"),
            mean_weight=("weight", "mean"),
            std_weight=("weight", "std"),
            n_runs_present=("run_id", "nunique"),
        )
    )
    summary = baseline.merge(aggregates, on=group_cols, how="left")
    summary = summary.merge(importance, on=group_cols, how="left")
    summary["missing_run_count"] = n_runs - summary["n_runs_present"].fillna(0).astype(int)
    summary["coverage_complete"] = summary["missing_run_count"] == 0
    summary["max_minus_min"] = summary["max_weight"] - summary["min_weight"]
    summary["std_weight"] = pd.to_numeric(summary["std_weight"], errors="coerce").fillna(0.0)
    summary = summary.sort_values(
        ["max_minus_min", "importance_product", "group_id", "from_code", "to_code"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    return summary


def _build_summary_row(
    *,
    config: SamplingRobustnessConfig,
    assignments: pd.DataFrame,
    link_summary: pd.DataFrame,
) -> pd.DataFrame:
    instability = link_summary["max_minus_min"].astype(float)
    summary = {
        "period": config.break_config.period,
        "direction": config.break_config.direction,
        "measure": config.measures.estimation_measure,
        "flow_code": config.flow.flow_code,
        "n_bins": config.run.n_bins,
        "seed": config.run.seed,
        "n_observations": len(assignments),
        "n_groups": int(assignments["group_id"].nunique()),
        "n_links": len(link_summary),
        "coverage_complete_share": float(link_summary["coverage_complete"].mean()),
        "max_missing_run_count": int(link_summary["missing_run_count"].max()),
        "instability_min": float(instability.min()),
        "instability_p50": float(instability.quantile(0.50)),
        "instability_p75": float(instability.quantile(0.75)),
        "instability_p90": float(instability.quantile(0.90)),
        "instability_p95": float(instability.quantile(0.95)),
        "instability_max": float(instability.max()),
        "share_instability_le_0_05": float((instability <= 0.05).mean()),
        "share_instability_le_0_10": float((instability <= 0.10).mean()),
        "share_instability_gt_0_20": float((instability > 0.20).mean()),
    }
    return pd.DataFrame([summary])


def run_sampling_robustness_analysis(config: SamplingRobustnessConfig) -> dict[str, object]:
    period = config.break_config.period
    vintage_a_year = int(period[:4])
    vintage_b_year = int(period[4:])
    measure = config.measures.estimation_measure

    groups = load_concordance_groups(
        concordance_path=config.paths.concordance_path,
        sheet_name=config.paths.concordance_sheet,
    )
    raw_a = _load_annual_frame(
        annual_base_dir=config.paths.annual_base_dir,
        year=vintage_a_year,
        measure=measure,
        exclude_reporters=config.sample.exclude_reporters,
        exclude_partners=config.sample.exclude_partners,
    )
    raw_b = _load_annual_frame(
        annual_base_dir=config.paths.annual_base_dir,
        year=vintage_b_year,
        measure=measure,
        exclude_reporters=config.sample.exclude_reporters,
        exclude_partners=config.sample.exclude_partners,
    )

    full_estimation, full_matrices, full_weights, _ = _estimate_ambiguous_weights(
        groups=groups,
        period=period,
        direction=config.break_config.direction,
        measure=measure,
        flow_code=config.flow.flow_code,
        data_a=raw_a,
        data_b=raw_b,
    )
    full_weights = _reported_ambiguous_weights(full_weights)
    if full_weights.empty:
        raise ValueError(
            f"sampling_robustness: no ambiguous weights estimated for {period} {config.break_config.direction}"
        )

    assignments = _build_bin_assignments(
        matrices=full_matrices,
        n_bins=config.run.n_bins,
        seed=config.run.seed,
    )
    a_map, b_map = _group_maps_for_period(groups=groups, period=period)
    raw_a_annotated = _annotate_for_holdout(raw_a, group_map=a_map)
    raw_b_annotated = _annotate_for_holdout(raw_b, group_map=b_map)

    run_frames: list[pd.DataFrame] = []
    for omitted_bin in progress(
        range(config.run.n_bins),
        desc="sampling_robustness bins",
        total=config.run.n_bins,
    ):
        holdout_keys = set(
            assignments.loc[assignments["bin_id"] == omitted_bin, "holdout_key"].astype(str).tolist()
        )
        subsample_a = _filter_holdout_rows(
            raw_a_annotated,
            holdout_keys=holdout_keys,
            measure=measure,
        )
        subsample_b = _filter_holdout_rows(
            raw_b_annotated,
            holdout_keys=holdout_keys,
            measure=measure,
        )
        _, _, run_weights, _ = _estimate_ambiguous_weights(
            groups=groups,
            period=period,
            direction=config.break_config.direction,
            measure=measure,
            flow_code=config.flow.flow_code,
            data_a=subsample_a,
            data_b=subsample_b,
        )
        run_weights = _reported_ambiguous_weights(run_weights).copy()
        run_weights["run_id"] = int(omitted_bin)
        run_weights["omitted_bin"] = int(omitted_bin)
        run_frames.append(run_weights)

    run_weights = pd.concat(run_frames, ignore_index=True)
    importance = _importance_frame(
        estimation=full_estimation,
        direction=config.break_config.direction,
        full_weights=full_weights,
    )
    link_summary = _build_link_summary(
        full_weights=full_weights,
        run_weights=run_weights,
        importance=importance,
        n_runs=config.run.n_bins,
    )
    summary = _build_summary_row(
        config=config,
        assignments=assignments,
        link_summary=link_summary,
    )

    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(config.output.bin_assignments_csv, index=False)
    run_weights.to_csv(config.output.subsample_weights_csv, index=False)
    link_summary.to_csv(config.output.link_summary_csv, index=False)
    summary.to_csv(config.output.summary_csv, index=False)

    plot_sampling_robustness_panels(
        data=link_summary,
        output_path=config.plot.output_path,
        title=config.plot.title,
        point_alpha=config.plot.point_alpha,
        point_size=config.plot.point_size,
        point_color=config.plot.point_color,
        histogram_bins=config.plot.histogram_bins,
        use_latex=config.plot.use_latex,
        latex_preamble=config.plot.latex_preamble,
    )

    return {
        "output_plot": config.plot.output_path,
        "summary_csv": config.output.summary_csv,
        "link_summary_csv": config.output.link_summary_csv,
        "subsample_weights_csv": config.output.subsample_weights_csv,
        "bin_assignments_csv": config.output.bin_assignments_csv,
    }
