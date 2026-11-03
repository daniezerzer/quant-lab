"""Price data loading with a simple on-disk cache.

Why cache? Free data sources rate-limit you, and reproducible research needs the
same data every time you re-run an experiment.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(ticker: str, start: str = "2010-01-01", end: str | None = None,
                refresh: bool = False) -> pd.Series:
    """Return adjusted close prices for `ticker` as a Series indexed by date.

    Adjusted close accounts for dividends and splits, so returns computed from it
    reflect what an investor would actually have earned.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end or 'latest'}.csv"

    if cache_file.exists() and not refresh:
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df["Close"].rename(ticker)

    import yfinance as yf  # imported here so tests don't need network libs

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker!r}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a 1-column frame
        close = close.iloc[:, 0]
    close = close.dropna().rename("Close")
    close.to_frame().to_csv(cache_file)
    return close.rename(ticker)
