"""Step 3: try many SMA settings, then test the winner on data it has never seen.

Run from the quant-lab folder:
    python scripts/sweep.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from quantlab import metrics
from quantlab.data import load_prices
from quantlab.optimize import train_test_sma

TICKER = "SPY"
COST_BPS = 5
SPLIT_DATE = "2016-01-01"          # before: pick parameters. after: judge them.
FAST_GRID = [5, 10, 20, 30, 50, 100]
SLOW_GRID = [100, 150, 200, 250, 300]

prices = load_prices(TICKER, start="2005-01-01")
res = train_test_sma(prices, SPLIT_DATE, FAST_GRID, SLOW_GRID, COST_BPS)

print(f"{TICKER}, costs = {COST_BPS} bps, split at {SPLIT_DATE}")
print(f"Tried {res.n_combos} (fast, slow) combinations on the TRAIN period.\n")
print(f"Best on TRAIN: fast={res.best_fast}, slow={res.best_slow}")
print(f"  Sharpe on TRAIN (what we picked it for): {res.train_sharpe:5.2f}")
print(f"  Sharpe on TEST  (unseen data):           {res.test_sharpe:5.2f}")
print(f"  Rank on TEST among all combos:           {res.test_rank} of {res.n_combos}")
print(f"  Rank correlation train vs test:          {res.rank_corr:5.2f}")
print("  (1 = good train settings stay good, 0 = no link, so train performance is mostly luck)\n")

print("TEST period results")
header = f"{'':26}{'return':>9}{'vol':>9}{'sharpe':>9}{'max DD':>9}"
print(header)
rows = {
    f"SMA {res.best_fast}/{res.best_slow} (picked)": res.chosen_test_returns,
    "SMA 50/200 (textbook)": res.default_test_returns,
    "Buy & hold": res.buy_hold_test_returns,
}
for name, r in rows.items():
    s = metrics.summary(r)
    print(f"{name:26}{s['annual_return']:9.1%}{s['annual_volatility']:9.1%}"
          f"{s['sharpe']:9.2f}{s['max_drawdown']:9.1%}")

# ---- charts -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), layout="constrained")
vmin = float(np.nanmin([res.train_grid.values, res.test_grid.values]))
vmax = float(np.nanmax([res.train_grid.values, res.test_grid.values]))
mid = (vmin + vmax) / 2

for ax, grid, title in (
    (axes[0], res.train_grid, f"TRAIN Sharpe (up to {SPLIT_DATE})"),
    (axes[1], res.test_grid, f"TEST Sharpe (from {SPLIT_DATE})"),
):
    im = ax.imshow(grid.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("slow window (days)")
    ax.set_ylabel("fast window (days)")
    ax.set_title(title)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if v > mid else "white")
    bi = list(grid.index).index(res.best_fast)
    bj = list(grid.columns).index(res.best_slow)
    ax.add_patch(Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
fig.colorbar(im, ax=axes[:2], shrink=0.8, label="Sharpe ratio")

for name, r in rows.items():
    (1 + r).cumprod().plot(ax=axes[2], label=name)
axes[2].set_title("TEST period: growth of 1 unit (after costs)")
axes[2].legend()

out_dir = ROOT / "docs"
out_dir.mkdir(exist_ok=True)
out = out_dir / "sweep.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved chart to {out}  (red box = the pair picked on TRAIN)")
