"""Step 2: compare strategies against buy-and-hold, with trading costs.

Run from the quant-lab folder:
    python scripts/compare.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from quantlab import strategies
from quantlab.backtest import run_backtest
from quantlab.data import load_prices

TICKER = "SPY"
COST_BPS = 5

prices = load_prices(TICKER, start="2005-01-01")

runs = {"Buy & hold": run_backtest(prices, strategies.buy_and_hold(prices), COST_BPS)}

try:
    runs["SMA 50/200"] = run_backtest(prices, strategies.sma_crossover(prices, 50, 200), COST_BPS)
except NotImplementedError:
    print("sma_crossover isn't written yet. Finish the exercise in quantlab/strategies.py,")
    print("then run this again. Showing buy & hold only for now.\n")

print(f"{TICKER}, costs = {COST_BPS} bps per trade\n")
header = f"{'':12}{'return':>9}{'vol':>9}{'sharpe':>9}{'max DD':>9}{'trades':>9}{'in mkt':>9}"
print(header)
for name, res in runs.items():
    s = res.summary()
    print(f"{name:12}{s['annual_return']:9.1%}{s['annual_volatility']:9.1%}"
          f"{s['sharpe']:9.2f}{s['max_drawdown']:9.1%}{s['trades']:9.0f}{s['time_in_market']:9.0%}")

fig, ax = plt.subplots(figsize=(10, 5))
for name, res in runs.items():
    res.equity.plot(ax=ax, label=name)
ax.set_title(f"{TICKER}: growth of 1 unit (after costs)")
ax.legend()
plt.tight_layout()
out_dir = Path(__file__).resolve().parent.parent / "docs"
out_dir.mkdir(exist_ok=True)
out = out_dir / "compare.png"
plt.savefig(out, dpi=120)
print(f"\nSaved chart to {out}")
