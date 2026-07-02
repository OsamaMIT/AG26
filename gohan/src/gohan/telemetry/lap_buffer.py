"""Small rolling buffer for lap and episode records."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Generic, Iterable, TypeVar

T = TypeVar("T")


@dataclass
class LapBuffer(Generic[T]):
    """Bounded rolling buffer."""

    maxlen: int = 10000
    _rows: Deque[T] = field(init=False)

    def __post_init__(self) -> None:
        self._rows = deque(maxlen=self.maxlen)

    def append(self, item: T) -> None:
        self._rows.append(item)

    def clear(self) -> None:
        self._rows.clear()

    def rows(self) -> list[T]:
        return list(self._rows)

    def extend(self, items: Iterable[T]) -> None:
        for item in items:
            self.append(item)
