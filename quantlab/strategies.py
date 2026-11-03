"""Trading strategies.

A strategy is just a function: prices in, signal out.

    signal = 1  -> be invested at tomorrow's open/close
    signal = 0  -> be in cash

The signal for day t may ONLY use prices up to and including day t.
(The backtest engine delays it by one day, so it is traded tomorrow.)
"""
import pandas as pd


def buy_and_hold(prices: pd.Series) -> pd.Series:
    """Always invested. This is the benchmark every strategy has to beat."""
    return pd.Series(1.0, index=prices.index)


def sma_crossover(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Trend-following rule: be invested (1) when the fast moving average of
    the price is ABOVE the slow moving average, otherwise be in cash (0).

    Early on there isn't enough history, so the averages are NaN. Comparing
    with NaN gives False, which means "stay in cash".
    """
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma > slow_ma).astype(float)
