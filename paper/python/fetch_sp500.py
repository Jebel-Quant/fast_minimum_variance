"""Fetch S&P 500 constituent returns for the last 5 years via yfinance."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "yfinance",
#     "pandas",
#     "numpy",
#     "lxml",
#     "requests",
#     "pyarrow",
# ]
# ///

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── 1. S&P 500 tickers from Wikipedia ─────────────────────────────────────────

resp = requests.get(
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    headers={"User-Agent": "Mozilla/5.0 (research script)"},
    timeout=30,
)
resp.raise_for_status()
tables = pd.read_html(io.StringIO(resp.text))
tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
print(f"Found {len(tickers)} tickers in the S&P 500 index")

# ── 2. Download 5 years of adjusted close prices ──────────────────────────────

raw = yf.download(
    tickers,
    period="5y",
    auto_adjust=True,
    progress=True,
    threads=True,
)["Close"]

print(f"\nRaw download: {raw.shape[0]} trading days x {raw.shape[1]} tickers")

# ── 3. Clean: keep tickers with <5% missing days, forward-fill short gaps ─────

threshold = 0.05
missing_frac = raw.isna().mean()
keep = missing_frac[missing_frac <= threshold].index
raw = raw[keep].ffill()

print(f"After cleaning: {raw.shape[0]} days x {raw.shape[1]} assets")

# ── 4. Log returns ─────────────────────────────────────────────────────────────

log_returns = np.log(raw / raw.shift(1)).dropna()
print(f"Return matrix shape: {log_returns.shape}  (T={log_returns.shape[0]}, N={log_returns.shape[1]})")

# ── 5. Save ───────────────────────────────────────────────────────────────────
file = Path(__file__).parent.parent / "data" / "sp500_returns.parquet"

log_returns.to_parquet(file)
print("\nSaved data/sp500_returns.parquet")
print(f"Date range: {log_returns.index[0].date()} → {log_returns.index[-1].date()}")
print(f"Assets: {log_returns.shape[1]}")
print(f"Trading days: {log_returns.shape[0]}")
