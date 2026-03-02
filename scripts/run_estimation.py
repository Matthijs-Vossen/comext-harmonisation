#!/usr/bin/env python3
"""Compatibility wrapper for estimation CLI."""

from __future__ import annotations

def main() -> None:
    from comext_harmonisation.cli.run_estimation import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
