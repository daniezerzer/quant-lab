"""Step 1 exploration: look at real market data and compute the basic risk numbers.

Run from the quant-lab folder:
    python scripts/explore.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from quantlab.data import load_prices
from quantlab import metrics

TICKERS = ["SPY", "QQQ", "GLD"]  # US stocks, US tech, gold

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for ticker in TICKERS:
    prices = load_prices(ticker, start="2012-01-01")
    returns = metrics.simple_returns(prices)
    stats = metrics.summary(returns)
    print(f"\n{ticker}")
    for name, value in stats.items():
        print(f"  {name:>18}: {value:8.3f}")

    (prices / prices.iloc[0]).plot(ax=axes[0], label=ticker)
    metrics.drawdown_series(returns).plot(ax=axes[1], label=ticker)

axes[0].set_title("Growth of 1 unit invested")
axes[0].legend()
axes[1].set_title("Drawdown from previous peak")
axes[1].legend()
plt.tight_layout()
out_dir = Path(__file__).resolve().parent.parent / "docs"
out_dir.mkdir(exist_ok=True)
out = out_dir / "explore.png"
plt.savefig(out, dpi=120)
print(f"\nSaved chart to {out}")
