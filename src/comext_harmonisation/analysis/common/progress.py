"""Progress-bar helper with a no-op fallback."""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def progress(iterable: Iterable[T], *, desc: str | None = None, total: int | None = None) -> Iterator[T]:
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return iter(iterable)
    return tqdm(iterable, desc=desc, total=total)
