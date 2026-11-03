"""Step 4: how much of the result is real? Confidence intervals and walk-forward testing.

Run from the quant-lab folder:
    python scripts/robustness.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from quantlab import metrics
from quantlab.data import load_prices
from quantlab.optimize import train_test_sma, walk_forward_sma
from quantlab.stats import bootstrap_sharpe, bootstrap_sharpe_difference

TICKER = "SPY"
COST_BPS = 5
SPLIT_DATE = "2016-01-01"
FAST_GRID = [5, 10, 20, 30, 50, 100]
SLOW_GRID = [100, 150, 200, 250, 300]
N_BOOT = 2000       # bootstrap resamples
BLOCK = 21          # about one trading month per block

prices = load_prices(TICKER, start="2005-01-01")


def show_ci(name, ci):
    print(f"  {name:28} Sharpe {ci.estimate:5.2f}   95% CI [{ci.low:5.2f}, {ci.high:5.2f}]")


def show_diff(name, d):
    verdict = "cannot tell apart" if d.contains_zero() else "clear difference"
    print(f"  {name:28} diff {d.estimate:+5.2f}   95% CI [{d.low:+5.2f}, {d.high:+5.2f}]"
          f"   A better in {d.prob_a_better:4.0%} of resamples -> {verdict}")


# ---- Part 1: single split, with error bars --------------------------------------
split = train_test_sma(prices, SPLIT_DATE, FAST_GRID, SLOW_GRID, COST_BPS)
picked_name = f"SMA {split.best_fast}/{split.best_slow} (picked)"
sp_rows = {
    picked_name: split.chosen_test_returns,
    "SMA 50/200 (textbook)": split.default_test_returns,
    "Buy & hold": split.buy_hold_test_returns,
}
print(f"{TICKER}, costs = {COST_BPS} bps. Bootstrap: {N_BOOT} resamples, {BLOCK}-day blocks.\n")
print(f"PART 1: unseen data from {SPLIT_DATE}")
print(" Sharpe ratio with 95% confidence interval:")
sp_ci = {name: bootstrap_sharpe(r, N_BOOT, BLOCK) for name, r in sp_rows.items()}
for name, ci in sp_ci.items():
    show_ci(name, ci)
print(" Is one really better than another? (A minus B, same dates)")
show_diff(f"{picked_name} - B&H",
          bootstrap_sharpe_difference(sp_rows[picked_name], sp_rows["Buy & hold"], N_BOOT, BLOCK))
show_diff("SMA 50/200 - B&H",
          bootstrap_sharpe_difference(sp_rows["SMA 50/200 (textbook)"], sp_rows["Buy & hold"], N_BOOT, BLOCK))
show_diff(f"{picked_name} - 50/200",
          bootstrap_sharpe_difference(sp_rows[picked_name], sp_rows["SMA 50/200 (textbook)"], N_BOOT, BLOCK))

# ---- Part 2: walk-forward ---------------------------------------------------------
wf = walk_forward_sma(prices, FAST_GRID, SLOW_GRID, train_years=5, test_years=1, cost_bps=COST_BPS)
start, end = wf.oos_returns.index[0].date(), wf.oos_returns.index[-1].date()
print(f"\nPART 2: walk-forward, re-picking the pair every year from the previous 5 years")
print(f" Out-of-sample period: {start} to {end} ({len(wf.choices)} yearly windows)\n")
print(" Pair chosen each year:")
for _, row in wf.choices.iterrows():
    print(f"  {row['test_start'].year}: SMA {int(row['fast']):>3}/{int(row['slow']):<3}"
          f"  train Sharpe {row['train_sharpe']:5.2f}  ->  that year's Sharpe {row['test_sharpe']:5.2f}")
n_changes = int(((wf.choices["fast"].diff() != 0) | (wf.choices["slow"].diff() != 0)).sum() - 1)
print(f"  The chosen pair changed {n_changes} times in {len(wf.choices) - 1} opportunities.\n")

wf_rows = {
    "Walk-forward SMA": wf.oos_returns,
    "SMA 50/200 (textbook)": wf.default_returns,
    "Buy & hold": wf.buy_hold_returns,
}
header = f"{'':26}{'return':>9}{'vol':>9}{'sharpe':>9}{'max DD':>9}"
print(" " + header)
for name, r in wf_rows.items():
    s = metrics.summary(r)
    print(f"  {name:24}{s['annual_return']:9.1%}{s['annual_volatility']:9.1%}"
          f"{s['sharpe']:9.2f}{s['max_drawdown']:9.1%}")

print("\n Sharpe ratio with 95% confidence interval:")
wf_ci = {name: bootstrap_sharpe(r, N_BOOT, BLOCK) for name, r in wf_rows.items()}
for name, ci in wf_ci.items():
    show_ci(name, ci)
print(" Is one really better than another? (A minus B, same dates)")
show_diff("Walk-forward - B&H",
          bootstrap_sharpe_difference(wf_rows["Walk-forward SMA"], wf_rows["Buy & hold"], N_BOOT, BLOCK))
show_diff("SMA 50/200 - B&H",
          bootstrap_sharpe_difference(wf_rows["SMA 50/200 (textbook)"], wf_rows["Buy & hold"], N_BOOT, BLOCK))
show_diff("Walk-forward - 50/200",
          bootstrap_sharpe_difference(wf_rows["Walk-forward SMA"], wf_rows["SMA 50/200 (textbook)"], N_BOOT, BLOCK))

# ---- chart ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), layout="constrained")

ax = axes[0]
labels, cis = [], []
for name, ci in sp_ci.items():
    labels.append(f"{SPLIT_DATE[:4]}+: {name}")
    cis.append(ci)
for name, ci in wf_ci.items():
    labels.append(f"Walk-fwd: {name}")
    cis.append(ci)
ys = list(range(len(labels)))[::-1]
for y, ci in zip(ys, cis):
    ax.errorbar(ci.estimate, y, xerr=[[ci.estimate - ci.low], [ci.high - ci.estimate]],
                fmt="o", capsize=4, color="tab:blue")
ax.axvline(0, color="grey", ls="--", lw=1)
ax.set_yticks(ys, labels)
ax.set_xlabel("Sharpe ratio (dot = estimate, bar = 95% confidence interval)")
ax.set_title("How uncertain is each Sharpe?")

for name, r in wf_rows.items():
    (1 + r).cumprod().plot(ax=axes[1], label=name)
axes[1].set_title(f"Walk-forward out-of-sample: growth of 1 unit")
axes[1].legend()

ax = axes[2]
ax.step(wf.choices["test_start"], wf.choices["fast"], where="post", marker="o", label="fast window")
ax.step(wf.choices["test_start"], wf.choices["slow"], where="post", marker="o", label="slow window")
ax.set_title("Parameters chosen each year")
ax.set_ylabel("days")
ax.legend()

out_dir = ROOT / "docs"
out_dir.mkdir(exist_ok=True)
out = out_dir / "robustness.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved chart to {out}")
