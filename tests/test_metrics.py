import numpy as np
import pandas as pd
import pytest

from quantlab import metrics


def make_prices(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_simple_returns():
    prices = make_prices([100, 110, 99])
    r = metrics.simple_returns(prices)
    assert r.iloc[0] == pytest.approx(0.10)
    assert r.iloc[1] == pytest.approx(-0.10)


def test_log_returns_add_up():
    prices = make_prices([100, 110, 121])
    total = metrics.log_returns(prices).sum()
    assert total == pytest.approx(np.log(121 / 100))


def test_max_drawdown_known_case():
    # Up 10%, then down 50%: equity 1.1 -> 0.55, drawdown = -50%
    returns = pd.Series([0.10, -0.50])
    assert metrics.max_drawdown(returns) == pytest.approx(-0.50)


def test_max_drawdown_never_positive():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    assert metrics.max_drawdown(returns) <= 0


def test_volatility_zero_for_constant_returns():
    returns = pd.Series([0.01] * 50)
    assert metrics.annualised_volatility(returns) == pytest.approx(0.0, abs=1e-12)


def test_sharpe_scales_with_mean():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0.0005, 0.01, 1000))
    assert metrics.sharpe_ratio(base + 0.001) > metrics.sharpe_ratio(base)


def test_annualised_return_one_year():
    # 252 days of identical returns compounding to +10% in one year
    daily = 1.10 ** (1 / 252) - 1
    returns = pd.Series([daily] * 252)
    assert metrics.annualised_return(returns) == pytest.approx(0.10)
