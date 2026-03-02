"""Chaining engine for composing adjacent weights across years."""

from .engine import (
    ChainedWeightsOutput,
    DEFAULT_ANNUAL_DATA_DIR,
    DEFAULT_CHAINED_DIAGNOSTICS_DIR,
    DEFAULT_CHAINED_WEIGHTS_DIR,
    build_chained_weights_for_range,
    build_code_universe_from_annual,
    build_revised_code_index_from_concordance,
    chain_weights_for_year,
)

__all__ = [
    "ChainedWeightsOutput",
    "DEFAULT_ANNUAL_DATA_DIR",
    "DEFAULT_CHAINED_DIAGNOSTICS_DIR",
    "DEFAULT_CHAINED_WEIGHTS_DIR",
    "build_chained_weights_for_range",
    "build_code_universe_from_annual",
    "build_revised_code_index_from_concordance",
    "chain_weights_for_year",
]
