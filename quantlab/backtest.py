"""A small, honest backtesting engine.

The one idea that matters most here is TIMING:

    A signal computed from today's closing price can only be traded TOMORROW.

If you let a strategy see today's close and also earn today's return, it is
cheating: it "knew" the move before it happened. That is look-ahead bias, and
it is the most common reason a backtest looks amazing and then fails in real life.
This engine prevents it by shifting the signal forward one day (`signal.shift(1)`).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab import metrics


@dataclass
class BacktestResult:
    returns: pd.Series        # net daily strategy returns (after costs)
    gross_returns: pd.Series  # daily returns before costs
    position: pd.Series       # position actually held each day (0 = cash, 1 = fully invested)
    turnover: pd.Series       # how much the position changed each day
    cost_bps: float

    @property
    def equity(self) -> pd.Series:
        """Growth of 1 unit of capital."""
        return (1 + self.returns).cumprod()

    @property
    def n_trades(self) -> int:
        return int((self.turnover > 0).sum())

    def summary(self) -> dict[str, float]:
        stats = metrics.summary(self.returns)
        stats["trades"] = float(self.n_trades)
        stats["time_in_market"] = float((self.position != 0).mean())
        return stats


def run_backtest(prices: pd.Series, signal: pd.Series, cost_bps: float = 5.0) -> BacktestResult:
    """Run a strategy over historical prices.

    prices:   adjusted close prices indexed by date.
    signal:   target position decided at each day's close (0 = cash, 1 = long).
              Must only use information available up to and including that day.
    cost_bps: cost per unit of position change, in basis points (1 bp = 0.01%).
              5 bps is a rough, optimistic figure for liquid ETFs. Real costs
              (spread, slippage, fees) are often higher.
    """
    asset_returns = metrics.simple_returns(prices)

    # The crucial line: yesterday's signal decides today's position.
    position = (
        signal.reindex(prices.index).shift(1).reindex(asset_returns.index).fillna(0.0)
    )

    # Every change in position costs money. The first day counts as a change from 0.
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover * cost_bps / 10_000

    gross = position * asset_returns
    net = gross - costs

    return BacktestResult(
        returns=net,
        gross_returns=gross,
        position=position,
        turnover=turnover,
        cost_bps=cost_bps,
    )
