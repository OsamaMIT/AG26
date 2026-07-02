"""Simple running normalizer for future telemetry preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RunningNormalizer:
    """Online mean/variance normalizer."""

    shape: tuple[int, ...]
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        self.count = 0
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.m2 = np.zeros(self.shape, dtype=np.float64)

    def update(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float64)
        self.count += 1
        delta = arr - self.mean
        self.mean += delta / self.count
        delta2 = arr - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.shape, dtype=np.float64)
        return self.m2 / (self.count - 1)

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return ((np.asarray(value, dtype=np.float64) - self.mean) / np.sqrt(self.variance + self.epsilon)).astype(
            np.float32
        )
