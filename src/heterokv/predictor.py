"""Explicit profile inputs; no default hardware timings."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    worker_id: str
    prompt_grid: tuple[int, ...]
    prefill_ms: tuple[float, ...]
    context_grid: tuple[int, ...]
    decode_ms: tuple[float, ...]
    queue_ms_per_token: float
    contention_per_sequence: float = 0.0

    def __post_init__(self):
        for grid, values in [(self.prompt_grid, self.prefill_ms),
                             (self.context_grid, self.decode_ms)]:
            if not grid or len(grid) != len(values) or any(v < 0 for v in values):
                raise ValueError("invalid profile axes")
            if any(b <= a for a, b in zip(grid, grid[1:])):
                raise ValueError("profile axes must strictly increase")
        if self.queue_ms_per_token < 0 or self.contention_per_sequence < 0:
            raise ValueError("negative cost")

    def predict(self, prompt: int, context: int, queued: int, active: int):
        if not self.prompt_grid[0] <= prompt <= self.prompt_grid[-1]:
            raise ValueError("prefill outside profile coverage")
        if not self.context_grid[0] <= context <= self.context_grid[-1]:
            raise ValueError("decode outside profile coverage")
        return (float(np.interp(prompt, self.prompt_grid, self.prefill_ms)),
                float(np.interp(context, self.context_grid, self.decode_ms))
                * (1 + active * self.contention_per_sequence), queued * self.queue_ms_per_token)


@dataclass(frozen=True)
class Link:
    setup_ms: float
    bytes_per_second: float
    first_token: str = "decode"

    def __post_init__(self):
        if self.setup_ms < 0 or self.bytes_per_second <= 0:
            raise ValueError("invalid transfer profile")
        if self.first_token not in {"prefill", "decode"}:
            raise ValueError("unknown first-token protocol")

    def cost(self, size: int) -> float:
        return self.setup_ms + 1000 * size / self.bytes_per_second


class RegressionPredictor:
    """Optional component estimator; callers supply only arrival-time features."""

    def __init__(self, method: str = "linear"):
        if method == "linear":
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(positive=True)
        elif method == "xgboost":
            from xgboost import XGBRegressor
            self.model = XGBRegressor(n_estimators=100, max_depth=3, random_state=101, n_jobs=1)
        else:
            raise ValueError(method)

    def fit(self, features, targets, groups, validation_groups):
        if set(groups) & set(validation_groups):
            raise ValueError("training and validation run groups overlap")
        self.model.fit(features, targets)
        return self

    def predict(self, features):
        return np.maximum(0, self.model.predict(features))


def errors(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    absolute = np.abs(actual - predicted)
    percentage = 100 * absolute / np.maximum(np.abs(actual), 1.0)
    return {"median_ape": float(np.median(percentage)),
            "p90_ape": float(np.percentile(percentage, 90)),
            "median_absolute_ms": float(np.median(absolute)),
            "underprediction_fraction": float(np.mean(predicted < actual))}

