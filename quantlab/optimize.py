"""Parameter sweeps and out-of-sample testing.

The trap this module exists to expose:

    If you try 30 different parameter settings and pick the best one, you have
    not found a good strategy. You have found the setting that happened to fit
    the past. The honest question is: does the winner STILL win on data it never saw?

So we split history in two:
    TRAIN  = the period we are allowed to pick parameters from
    TEST   = a later period, touched only once, to judge the pick
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab import metrics, strategies
from quantlab.backtest import run_backtest


def sma_returns(prices: pd.Series, fast: int, slow: int, cost_bps: float = 5.0) -> pd.Series:
    """Net daily returns of an SMA crossover with the given windows."""
    signal = strategies.sma_crossover(prices, fast, slow)
    return run_backtest(prices, signal, cost_bps).returns


def sweep_sma(prices: pd.Series, fast_grid, slow_grid, cost_bps: float = 5.0,
              start=None, end=None) -> pd.DataFrame:
    """Sharpe ratio for every (fast, slow) pair, judged over [start, end].

    Rows are fast windows, columns are slow windows. Pairs where fast >= slow
    make no sense and are left as NaN.

    The signals are always computed on the FULL price history (a moving average
    needs a warm-up), and only the returns are cut to the period. That is not
    look-ahead: each day's signal still only uses earlier prices.
    """
    grid = pd.DataFrame(np.nan, index=list(fast_grid), columns=list(slow_grid), dtype=float)
    for fast, slow in itertools.product(fast_grid, slow_grid):
        if fast >= slow:
            continue
        returns = sma_returns(prices, fast, slow, cost_bps).loc[start:end]
        grid.loc[fast, slow] = metrics.sharpe_ratio(returns)
    return grid


@dataclass
class SplitResult:
    split_date: pd.Timestamp
    best_fast: int
    best_slow: int
    train_grid: pd.DataFrame
    test_grid: pd.DataFrame
    train_sharpe: float           # Sharpe of the chosen pair on TRAIN (looks great by construction)
    test_sharpe: float            # Sharpe of the same pair on unseen TEST data (the honest number)
    test_rank: int                # where the chosen pair ranks among all pairs on TEST (1 = best)
    n_combos: int
    rank_corr: float              # do good TRAIN settings stay good on TEST? (1 = yes, 0 = no link)
    chosen_test_returns: pd.Series
    default_test_returns: pd.Series
    buy_hold_test_returns: pd.Series


def train_test_sma(prices: pd.Series, split_date, fast_grid, slow_grid,
                   cost_bps: float = 5.0, default: tuple[int, int] = (50, 200)) -> SplitResult:
    """Pick the best SMA windows on data before `split_date`, then test them after it."""
    split = pd.Timestamp(split_date)
    train_end = split - pd.Timedelta(days=1)
    # Start judging only once the longest moving average exists, so every pair
    # is compared over exactly the same days.
    warmup = prices.index[max(slow_grid)]

    train_grid = sweep_sma(prices, fast_grid, slow_grid, cost_bps, start=warmup, end=train_end)
    test_grid = sweep_sma(prices, fast_grid, slow_grid, cost_bps, start=split)

    train_scores = train_grid.stack().dropna()
    test_scores = test_grid.stack().dropna()

    best = train_scores.idxmax()
    best_fast, best_slow = int(best[0]), int(best[1])

    return SplitResult(
        split_date=split,
        best_fast=best_fast,
        best_slow=best_slow,
        train_grid=train_grid,
        test_grid=test_grid,
        train_sharpe=float(train_scores[best]),
        test_sharpe=float(test_scores[best]),
        test_rank=int(test_scores.rank(ascending=False)[best]),
        n_combos=len(train_scores),
        rank_corr=float(train_scores.rank().corr(test_scores.rank())),
        chosen_test_returns=sma_returns(prices, best_fast, best_slow, cost_bps).loc[split:],
        default_test_returns=sma_returns(prices, default[0], default[1], cost_bps).loc[split:],
        buy_hold_test_returns=run_backtest(
            prices, strategies.buy_and_hold(prices), cost_bps
        ).returns.loc[split:],
    )
