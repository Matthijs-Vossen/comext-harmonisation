import comext_harmonisation as ch
from comext_harmonisation.pipeline_config import load_pipeline_config


def test_public_api_contains_core_pipeline_entrypoints() -> None:
    required = [
        "run_weight_estimation_for_period",
        "run_weight_estimation_for_period_multi",
        "chain_weights_for_year",
        "build_chained_weights_for_range",
        "apply_weights_to_annual_period",
        "apply_chained_weights_wide_for_range",
        "apply_chained_weights_wide_for_month_range",
        "finalize_weights_table",
    ]
    for name in required:
        assert hasattr(ch, name), f"Missing attribute: {name}"
        assert name in ch.__all__, f"Missing __all__ export: {name}"


def test_public_api_exports_are_unique() -> None:
    assert len(ch.__all__) == len(set(ch.__all__)), "Duplicate entries in comext_harmonisation.__all__"


def test_pipeline_config_loader_is_stable_import_path() -> None:
    assert callable(load_pipeline_config)
