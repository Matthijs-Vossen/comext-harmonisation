"""Run end-to-end estimation -> chaining -> apply pipeline using YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the end-to-end harmonisation pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a pipeline YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from ..pipeline.runner import run_pipeline_from_config_path

    run_pipeline_from_config_path(Path(args.config))


if __name__ == "__main__":
    main()
