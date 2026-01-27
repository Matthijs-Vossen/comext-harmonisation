#!/usr/bin/env python3
"""Run Figure-3 style analysis using a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Figure-3 style analysis.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from comext_harmonisation.analysis import load_fig3_config, run_fig3_analysis

    config = load_fig3_config(Path(args.config))
    outputs = run_fig3_analysis(config)
    print("plot:", outputs["output_plot"])
    print("summary:", outputs["summary_csv"])


if __name__ == "__main__":
    main()
