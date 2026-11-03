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


def test_walk_forward_windows_do_not_overlap_and_move_forward():
    prices = random_walk(n=3000)
    wf = optimize.walk_forward_sma(prices, FAST, SLOW, train_years=3, test_years=1)
    starts = wf.choices["test_start"]
    assert starts.is_monotonic_increasing
    assert (wf.choices["test_end"] >= starts).all()
    assert wf.oos_returns.index.is_unique


def test_walk_forward_has_no_lookahead():
    """Changing prices from date D onward must not change any choice made for a window
    that starts on or before D, because those choices only look at earlier data."""
    prices = random_walk(n=3000)
    wf_a = optimize.walk_forward_sma(prices, FAST, SLOW, train_years=3, test_years=1)
    cut = wf_a.choices["test_start"].iloc[4]
    altered = prices.copy()
    altered.loc[cut:] = altered.loc[cut:] * 2.5
    wf_b = optimize.walk_forward_sma(altered, FAST, SLOW, train_years=3, test_years=1)
    early_a = wf_a.choices[wf_a.choices["test_start"] <= cut][["test_start", "fast", "slow"]]
    early_b = wf_b.choices[wf_b.choices["test_start"] <= cut][["test_start", "fast", "slow"]]
    pd.testing.assert_frame_equal(early_a.reset_index(drop=True), early_b.reset_index(drop=True))


def test_walk_forward_oos_matches_chosen_pair_in_each_window():
    prices = random_walk(n=3000)
    wf = optimize.walk_forward_sma(prices, FAST, SLOW, train_years=3, test_years=1)
    row = wf.choices.iloc[2]
    expected = optimize.sma_returns(prices, int(row["fast"]), int(row["slow"])).loc[
        row["test_start"]:row["test_end"]]
    actual = wf.oos_returns.loc[row["test_start"]:row["test_end"]]
    pd.testing.assert_series_equal(actual, expected)


def test_rank_is_within_range():
    prices = random_walk()
    res = optimize.train_test_sma(prices, "2017-06-01", FAST, SLOW)
    assert 1 <= res.test_rank <= res.n_combos
    assert -1.0 <= res.rank_corr <= 1.0
