import comext_harmonisation as ch
from comext_harmonisation.pipeline.config import load_pipeline_config


def test_top_level_exports_namespace_modules() -> None:
    required = [
        "analysis",
        "apply",
        "chaining",
        "concordance",
        "core",
        "estimation",
        "pipeline",
        "weights",
    ]
    for name in required:
        assert name in ch.__all__, f"Missing __all__ entry: {name}"


def test_top_level_exports_are_unique() -> None:
    assert len(ch.__all__) == len(set(ch.__all__)), "Duplicate entries in comext_harmonisation.__all__"


def test_pipeline_config_loader_namespace() -> None:
    assert callable(load_pipeline_config)
