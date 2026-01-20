"""Comext harmonisation tools."""

from .concordance import read_concordance_xls, parse_concordance_df
from .groups import build_concordance_groups, ConcordanceGroups
from .mappings import (
    get_ambiguous_group_summary,
    get_ambiguous_edges,
    build_deterministic_mappings,
)
from .weights import empty_weight_table, validate_weight_table, WEIGHT_COLUMNS, DEFAULT_WEIGHTS_DIR
from .application import (
    apply_weights_to_annual_period,
    ApplyDiagnostics,
    finalize_weights_table,
    WEIGHT_STRATEGIES,
)
from .estimation import (
    EstimationShares,
    prepare_estimation_shares_for_period,
    prepare_estimation_shares_from_frames,
    EXCLUDE_CODES_DEFAULT,
    AGGREGATE_CODES,
    build_group_matrices,
    GroupMatrices,
    estimate_weights,
    SolverDiagnostics,
    RunnerOutputs,
    load_concordance_groups,
    run_weight_estimation_for_period,
    run_weight_estimation_for_period_multi,
    DEFAULT_CONCORDANCE_PATH,
    DEFAULT_DIAGNOSTICS_DIR,
    DEFAULT_SUMMARIES_DIR,
)

__all__ = [
    "read_concordance_xls",
    "parse_concordance_df",
    "build_concordance_groups",
    "ConcordanceGroups",
    "get_ambiguous_group_summary",
    "get_ambiguous_edges",
    "build_deterministic_mappings",
    "empty_weight_table",
    "validate_weight_table",
    "WEIGHT_COLUMNS",
    "DEFAULT_WEIGHTS_DIR",
    "apply_weights_to_annual_period",
    "ApplyDiagnostics",
    "finalize_weights_table",
    "WEIGHT_STRATEGIES",
    "EstimationShares",
    "prepare_estimation_shares_for_period",
    "prepare_estimation_shares_from_frames",
    "EXCLUDE_CODES_DEFAULT",
    "AGGREGATE_CODES",
    "build_group_matrices",
    "GroupMatrices",
    "estimate_weights",
    "SolverDiagnostics",
    "RunnerOutputs",
    "load_concordance_groups",
    "run_weight_estimation_for_period",
    "run_weight_estimation_for_period_multi",
    "DEFAULT_CONCORDANCE_PATH",
    "DEFAULT_DIAGNOSTICS_DIR",
    "DEFAULT_SUMMARIES_DIR",
]
