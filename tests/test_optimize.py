import numpy as np
import pandas as pd

from quantlab import optimize

FAST = [5, 10, 20]
SLOW = [30, 60, 90]


def random_walk(n=900, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(100 * np.exp(rng.normal(0.0003, 0.01, n).cumsum()), index=idx)


def test_sweep_shape_and_invalid_pairs_are_nan():
    prices = random_walk()
    grid = optimize.sweep_sma(prices, [5, 30, 60], [30, 60, 90])
    assert grid.shape == (3, 3)
    assert np.isnan(grid.loc[30, 30])      # fast == slow
    assert np.isnan(grid.loc[60, 30])      # fast > slow
    assert not np.isnan(grid.loc[5, 30])


def test_best_params_come_from_train_only():
    prices = random_walk()
    res = optimize.train_test_sma(prices, "2017-06-01", FAST, SLOW)
    assert (res.best_fast, res.best_slow) == res.train_grid.stack().idxmax()


def test_test_period_starts_at_split():
    prices = random_walk()
    res = optimize.train_test_sma(prices, "2017-06-01", FAST, SLOW)
    assert res.chosen_test_returns.index[0] >= pd.Timestamp("2017-06-01")
    assert res.buy_hold_test_returns.index[0] >= pd.Timestamp("2017-06-01")


def test_changing_test_prices_cannot_change_the_train_results():
    """The core no-cheating check: the future must not leak into the training grid."""
    prices = random_walk()
    split = pd.Timestamp("2017-06-01")
    altered = prices.copy()
    altered.loc[split:] = altered.loc[split:] * 3.0

    a = optimize.train_test_sma(prices, split, FAST, SLOW)
    b = optimize.train_test_sma(altered, split, FAST, SLOW)

    pd.testing.assert_frame_equal(a.train_grid, b.train_grid)
    assert (a.best_fast, a.best_slow) == (b.best_fast, b.best_slow)


def test_rank_is_within_range():
    prices = random_walk()
    res = optimize.train_test_sma(prices, "2017-06-01", FAST, SLOW)
    assert 1 <= res.test_rank <= res.n_combos
    assert -1.0 <= res.rank_corr <= 1.0
