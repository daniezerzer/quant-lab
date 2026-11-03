# quant-lab

A small research toolkit for testing trading ideas honestly: returns, risk metrics,
a backtesting engine with transaction costs and no look-ahead bias, and
out-of-sample validation of parameter choices.

Built as a learning project. The aim is not to find a money machine. It is to
understand why most apparent edges disappear once they are tested properly.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (cmd): .venv\Scripts\activate.bat
pip install -r requirements.txt
python -m pytest                 # all tests should pass
```

## Run

```bash
python scripts/explore.py        # market data, basic risk stats       -> docs/explore.png
python scripts/compare.py        # SMA 50/200 vs buy & hold, with costs -> docs/compare.png
python scripts/sweep.py          # parameter sweep + out-of-sample test -> docs/sweep.png
```

On Windows, `run_all.bat` runs the tests and all analysis scripts and saves the output to `results.txt`.

## Method

- **No look-ahead bias.** A signal computed from today's close is only traded tomorrow
  (`signal.shift(1)` in `quantlab/backtest.py`). Tests prove a strategy cannot earn a move
  that happens on the day it decides to buy.
- **Costs.** Every change of position is charged `cost_bps` (default 5 bps).
- **Out-of-sample testing.** Parameters are chosen on data before a split date and judged on
  data after it. A test checks that changing future prices can never change the training results.
- **Metrics.** Annualised return, volatility, Sharpe ratio, maximum drawdown.

## Findings

### SMA 50/200 vs buy & hold (SPY, 2005 onwards, 5 bps per trade)

| | Return/yr | Volatility | Sharpe | Max drawdown | Trades | In market |
|---|---|---|---|---|---|---|
| Buy & hold | 10.8% | 18.9% | 0.64 | -55.2% | 1 | 100% |
| SMA 50/200 | 8.9% | 13.5% | 0.70 | -33.7% | 21 | 76% |

The trend rule gave up about 2% a year of return in exchange for roughly a third less
volatility and a much shallower worst drawdown. The Sharpe difference is small enough to be
noise. Most of the benefit comes from avoiding the 2008 crash, which is one episode.

### Parameter sweep and out-of-sample test

29 fast/slow SMA combinations were tried on 2006-2015 (train). The best pair on train was
30/200 (Sharpe 0.77). All results below are on 2016 onwards, data the choice never saw.
Chart: `docs/sweep.png`.

| Test period (2016 onwards) | Return/yr | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
| SMA 30/200 (picked on train) | 9.5% | 13.8% | 0.73 | -33.3% |
| SMA 50/200 (textbook) | 9.5% | 14.9% | 0.68 | -33.7% |
| Buy & hold | 14.7% | 17.7% | 0.87 | -33.7% |

- **The parameter choice carried no usable information.** The train winner ranked 17th of 29
  on unseen data, and the rank correlation between train and test Sharpe was -0.31: settings
  that looked best before 2016 did not do better after it. The pair that won on train, 30/200,
  was only marginally different from the untuned textbook 50/200.
- **The trend filter lagged in a strong bull market.** Buy & hold beat every SMA variant on
  both return and Sharpe over 2016-2026.
- **No drawdown protection in the test period.** The 2020 crash was too fast for a 200-day
  filter: all three had a maximum drawdown of about -33%. Trend following helped in the slow
  2008 decline but not in a sudden crash.
- **Limits.** One asset, one split, 29 combinations, few trades, and a period that was
  unusually strong for US equities. Next: walk-forward validation and confidence intervals
  on the Sharpe ratio.

## Roadmap

- [x] Data loader with caching
- [x] Core metrics (returns, volatility, Sharpe, drawdown) with unit tests
- [x] Backtesting engine (no look-ahead bias, transaction costs)
- [x] Strategy: SMA crossover vs buy & hold
- [x] Parameter sweep and train/test validation
- [ ] More strategies: momentum, mean reversion
- [ ] Walk-forward validation
- [ ] Portfolio optimisation and Monte Carlo risk simulation
- [ ] Write-up of findings

## Layout

```
quantlab/     the library (data, metrics, backtest, strategies, optimize)
tests/        pytest unit tests
scripts/      runnable experiments
docs/         charts produced by the scripts
data/         cached price data (git-ignored)
```

## Caveats

Backtests are not predictions. Results here use one asset, free data and simple cost
assumptions, and ignore taxes. Nothing here is financial advice.
