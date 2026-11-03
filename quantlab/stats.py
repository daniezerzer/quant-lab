"""Statistical tools: how much of a backtest result is real, and how much is luck?

A Sharpe ratio measured on a few years of data is an ESTIMATE with a margin of
error, not a fact. This module measures that margin with a bootstrap:

    1. Take the actual daily returns.
    2. Build thousands of alternative histories by re-sampling them.
    3. Compute the Sharpe ratio of each alternative history.
    4. The spread of those numbers shows how uncertain the real estimate is.

Returns are not independent day to day (calm and stormy periods cluster), so we
resample in BLOCKS of consecutive days, not single days. That keeps the clustering.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
_CHUNK = 500  # resamples per batch, keeps memory use small


@dataclass
class Interval:
    estimate: float   # Sharpe on the real data
    low: float        # lower end of the confidence interval
    high: float       # upper end

    def contains_zero(self) -> bool:
        return self.low <= 0 <= self.high


@dataclass
class DifferenceInterval(Interval):
    prob_a_better: float = float("nan")  # share of resamples where A's Sharpe beats B's


def _bootstrap_sharpes(arrays: list[np.ndarray], n_boot: int, block: int, seed: int) -> np.ndarray:
    """Sharpe ratios of `n_boot` block-resampled histories, for each input series.

    All series are resampled with the SAME dates, so comparisons between them stay
    paired: if a resample happens to pick a crash, it picks it for every series.
    Returns an array of shape (len(arrays), n_boot).
    """
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All series must have the same length")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)  # ceil division
    out = np.empty((len(arrays), n_boot))
    done = 0
    while done < n_boot:
        m = min(_CHUNK, n_boot - done)
        starts = rng.integers(0, n, size=(m, n_blocks))
        idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n  # wrap around the end
        idx = idx.reshape(m, n_blocks * block)[:, :n]
        for k, arr in enumerate(arrays):
            sample = arr[idx]
            sd = sample.std(axis=1, ddof=1)
            mean = sample.mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[k, done:done + m] = np.where(sd > 0, mean / sd * np.sqrt(TRADING_DAYS), np.nan)
        done += m
    return out


def _sharpe(arr: np.ndarray) -> float:
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def bootstrap_sharpe(returns: pd.Series, n_boot: int = 2000, block: int = 21,
                     level: float = 0.95, seed: int = 0) -> Interval:
    """Confidence interval for one strategy's Sharpe ratio."""
    arr = returns.to_numpy(dtype=float)
    sharpes = _bootstrap_sharpes([arr], n_boot, block, seed)[0]
    tail = (1 - level) / 2
    return Interval(
        estimate=_sharpe(arr),
        low=float(np.nanquantile(sharpes, tail)),
        high=float(np.nanquantile(sharpes, 1 - tail)),
    )


def bootstrap_sharpe_difference(a: pd.Series, b: pd.Series, n_boot: int = 2000, block: int = 21,
                                level: float = 0.95, seed: int = 0) -> DifferenceInterval:
    """Confidence interval for (Sharpe of A) minus (Sharpe of B), on the same dates.

    If the interval contains zero, the data cannot tell the two strategies apart.
    """
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    arr_a = both.iloc[:, 0].to_numpy(dtype=float)
    arr_b = both.iloc[:, 1].to_numpy(dtype=float)
    sharpes = _bootstrap_sharpes([arr_a, arr_b], n_boot, block, seed)
    diff = sharpes[0] - sharpes[1]
    tail = (1 - level) / 2
    return DifferenceInterval(
        estimate=_sharpe(arr_a) - _sharpe(arr_b),
        low=float(np.nanquantile(diff, tail)),
        high=float(np.nanquantile(diff, 1 - tail)),
        prob_a_better=float(np.nanmean(diff > 0)),
    )
