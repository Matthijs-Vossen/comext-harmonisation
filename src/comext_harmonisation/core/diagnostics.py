"""Shared diagnostics persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def append_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    write_header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=write_header)


def append_detail_rows(
    rows: list[dict[str, object]],
    *,
    path: Path,
    columns: Sequence[str],
) -> None:
    if not rows:
        return
    append_csv(pd.DataFrame(rows, columns=list(columns)), path)

