"""Core performance metrics.

Every function here is something an interviewer might ask you to explain, so each
docstring says what it measures and why it matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices: pd.Series) -> pd.Series:
    """Day-over-day percentage change: (P_t / P_{t-1}) - 1."""
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """ln(P_t / P_{t-1}). Log returns add up over time, which makes maths easier."""
    return np.log(prices / prices.shift(1)).dropna()


def annualised_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Geometric (compounded) average yearly return."""
    growth = (1 + returns).prod()
    years = len(returns) / periods
    return float(growth ** (1 / years) - 1)


def annualised_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Standard deviation of returns scaled to a year (multiply by sqrt(periods)).

    Volatility is the basic measure of risk: how much returns wobble.
    """
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                 periods: int = TRADING_DAYS) -> float:
    """Return earned per unit of risk: (mean excess return / volatility), annualised.

    Higher is better. Around 1 is decent, above 2 is excellent (and suspicious in
    a backtest, since it often means overfitting).
    """
    excess = returns - risk_free_rate / periods
    vol = excess.std(ddof=1)
    if vol == 0:
        return float("nan")
    return float(excess.mean() / vol * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """How far the equity curve sits below its previous peak, at each date."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return equity / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss. This is what actually hurts investors emotionally."""
    return float(drawdown_series(returns).min())


def summary(returns: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    """Bundle the headline numbers into one dict."""
    return {
        "annual_return": annualised_return(returns),
        "annual_volatility": annualised_volatility(returns),
        "sharpe": sharpe_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
    }
