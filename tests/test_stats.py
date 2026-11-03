import numpy as np
import pandas as pd
import pytest

from quantlab import metrics, stats


def returns(mean, sd, n, seed):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(mean, sd, n), index=idx)


def test_interval_is_ordered_and_brackets_estimate():
    r = returns(0.0005, 0.01, 1500, seed=1)
    ci = stats.bootstrap_sharpe(r, n_boot=500)
    assert ci.low < ci.estimate < ci.high


def test_estimate_matches_metrics_sharpe():
    r = returns(0.0005, 0.01, 1500, seed=1)
    ci = stats.bootstrap_sharpe(r, n_boot=200)
    assert ci.estimate == pytest.approx(metrics.sharpe_ratio(r))


def test_same_seed_gives_same_answer():
    r = returns(0.0005, 0.01, 800, seed=2)
    a = stats.bootstrap_sharpe(r, n_boot=300, seed=5)
    b = stats.bootstrap_sharpe(r, n_boot=300, seed=5)
    assert (a.low, a.high) == (b.low, b.high)


def test_more_data_gives_a_narrower_interval():
    short = stats.bootstrap_sharpe(returns(0.0005, 0.01, 500, seed=3), n_boot=500)
    long = stats.bootstrap_sharpe(returns(0.0005, 0.01, 5000, seed=3), n_boot=500)
    assert (long.high - long.low) < (short.high - short.low)


def test_strong_edge_is_clearly_above_zero():
    r = returns(0.003, 0.005, 1500, seed=4)   # Sharpe around 9
    assert stats.bootstrap_sharpe(r, n_boot=500).low > 0


def test_pure_noise_cannot_be_told_from_zero():
    r = returns(0.0, 0.01, 1000, seed=6)
    assert stats.bootstrap_sharpe(r, n_boot=500).contains_zero()


def test_difference_of_identical_series_is_zero():
    r = returns(0.0004, 0.01, 800, seed=7)
    d = stats.bootstrap_sharpe_difference(r, r, n_boot=300)
    assert d.estimate == pytest.approx(0.0)
    assert d.low == pytest.approx(0.0)
    assert d.high == pytest.approx(0.0)


def test_difference_is_signed_correctly():
    a = returns(0.002, 0.01, 1500, seed=8)
    b = a - 0.002    # same noise, much lower mean
    d = stats.bootstrap_sharpe_difference(a, b, n_boot=500)
    assert d.estimate > 0
    assert d.prob_a_better > 0.95
