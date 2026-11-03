import numpy as np
import pandas as pd

from quantlab import strategies


def make_prices(values):
    idx = pd.date_range("2015-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_buy_and_hold_is_always_invested():
    prices = make_prices([1, 2, 3])
    assert (strategies.buy_and_hold(prices) == 1.0).all()


def test_sma_signal_is_only_zero_or_one():
    rng = np.random.default_rng(1)
    prices = make_prices(100 + rng.normal(0, 1, 400).cumsum())
    signal = strategies.sma_crossover(prices, fast=10, slow=30)
    assert set(signal.unique()) <= {0.0, 1.0}
    assert signal.index.equals(prices.index)


def test_sma_stays_in_cash_until_enough_history():
    prices = make_prices(np.linspace(100, 200, 100))
    signal = strategies.sma_crossover(prices, fast=5, slow=20)
    assert (signal.iloc[:19] == 0.0).all()


def test_sma_invested_in_steady_uptrend():
    prices = make_prices(np.linspace(100, 200, 100))
    signal = strategies.sma_crossover(prices, fast=5, slow=20)
    assert signal.iloc[-1] == 1.0


def test_sma_in_cash_in_steady_downtrend():
    prices = make_prices(np.linspace(200, 100, 100))
    signal = strategies.sma_crossover(prices, fast=5, slow=20)
    assert signal.iloc[-1] == 0.0


def test_sma_has_no_lookahead():
    """Changing FUTURE prices must never change PAST signals."""
    rng = np.random.default_rng(2)
    prices = make_prices(100 + rng.normal(0, 1, 300).cumsum())
    full = strategies.sma_crossover(prices, fast=10, slow=30)
    truncated = strategies.sma_crossover(prices.iloc[:200], fast=10, slow=30)
    pd.testing.assert_series_equal(full.iloc[:200], truncated)
