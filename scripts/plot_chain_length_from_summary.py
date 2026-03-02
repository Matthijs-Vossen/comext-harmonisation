#!/usr/bin/env python3
"""Compatibility wrapper for chain-length plotting CLI."""

from __future__ import annotations

def main() -> None:
    from comext_harmonisation.cli.plot_chain_length_from_summary import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
