import pandas as pd
import pytest

from quantlab.backtest import run_backtest


def make_prices(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_buy_and_hold_matches_price_growth_with_no_costs():
    prices = make_prices([100, 110, 121, 133.1])
    signal = pd.Series(1.0, index=prices.index)
    result = run_backtest(prices, signal, cost_bps=0)
    # Position is 1 from day 2 onward, so we earn every return after the first price.
    assert result.equity.iloc[-1] == pytest.approx(133.1 / 100)


def test_signal_is_delayed_one_day():
    # Price jumps 10% on day 2 (index 2). Signal switches on AT day 2's close.
    # We can't have caught that jump, so we must not earn it.
    prices = make_prices([100, 100, 110, 110, 110])
    signal = pd.Series([0, 0, 1, 1, 1], index=prices.index, dtype=float)
    result = run_backtest(prices, signal, cost_bps=0)
    assert result.equity.iloc[-1] == pytest.approx(1.0)


def test_costs_are_charged_on_position_changes():
    prices = make_prices([100, 100, 100, 100])
    signal = pd.Series([1, 1, 1, 1], index=prices.index, dtype=float)
    result = run_backtest(prices, signal, cost_bps=100)  # 1% per trade
    # Flat prices, one entry trade (0 -> 1) on day 2: lose exactly 1%.
    assert result.equity.iloc[-1] == pytest.approx(0.99)
    assert result.n_trades == 1


def test_round_trip_costs_twice():
    prices = make_prices([100, 100, 100, 100, 100])
    signal = pd.Series([1, 1, 0, 0, 0], index=prices.index, dtype=float)
    result = run_backtest(prices, signal, cost_bps=100)
    assert result.n_trades == 2  # in, then out
    assert result.equity.iloc[-1] == pytest.approx(0.99 * 0.99)


def test_position_is_never_known_before_signal():
    prices = make_prices([100, 101, 102, 103, 104])
    signal = pd.Series([0, 0, 1, 1, 1], index=prices.index, dtype=float)
    result = run_backtest(prices, signal, cost_bps=0)
    # Signal turns on at index 2, so the first held position is on index 3.
    assert result.position.loc[prices.index[2]] == 0.0
    assert result.position.loc[prices.index[3]] == 1.0
