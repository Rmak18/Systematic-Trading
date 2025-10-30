
import os
from datetime import datetime

import numpy as np
import pandas as pd

DATA_FILE = "data/prices_daily.csv"   
OUTPUT_DIR = "output/signals"
TICKERS = ["EWJ", "FEZ", "SPY"]       

def load_prices(path, tickers):
    df = pd.read_csv(path)
    # Find the date column (expect "time")
    date_col = "time" if "time" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # Keep only the tickers we asked for that exist in the file
    cols = [t for t in tickers if t in df.columns]
    if not cols:
        raise ValueError("None of the requested tickers were found in the CSV.")
    prices = df[cols].astype(float)
    # Drop duplicate timestamps if any (keep the last)
    prices = prices[~prices.index.duplicated(keep="last")]
    print(f"Loaded {len(prices)} rows from {prices.index.min().date()} to {prices.index.max().date()}")
    print("Missing values per ticker:")
    print(prices.isna().sum())
    return prices
def build_signal_12m_minus1(prices):
    # 1) month-end prices
    me = prices.resample("M").last()

    # 2) monthly log returns
    mret = np.log(me / me.shift(1))

    # 3) 12m sum, then skip the most recent month
    mom12 = mret.rolling(12).sum()
    sig_me = mom12.shift(1)

    # 4) move to first business day of next month
    sig_bms = sig_me.copy()
    sig_bms.index = sig_bms.index + pd.offsets.BMonthBegin(1)

    # 5) holiday-safe daily alignment
    idx = prices.index.union(sig_bms.index)   
    daily = sig_bms.reindex(idx).ffill()             
    daily = daily.reindex(prices.index)            

    return daily

def to_long_table(daily, tickers):
    daily = daily.copy()
    daily.index.name = "date"

    out = daily.reset_index().melt(
        id_vars="date",
        value_vars=list(daily.columns),
        var_name="ticker",
        value_name="signal_12m"
    )
    out = out[out["ticker"].isin(tickers)]
    out["signal_type"] = "12m_minus_1"
    out["calculation_date"] = datetime.now().strftime("%Y-%m-%d")
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out

def save_signals(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp1 = os.path.join(out_dir, f"signals_12m_minus_1_{ts}.csv")
    fp2 = os.path.join(out_dir, "signals_12m_minus_1_latest.csv")
    df.to_csv(fp1, index=False)
    df.to_csv(fp2, index=False)
    print("Saved:", fp1)
    print("Saved:", fp2)


def main():
    prices = load_prices(DATA_FILE, TICKERS)
    signals_daily = build_signal_12m_minus1(prices)
    signals_long = to_long_table(signals_daily, TICKERS)
    print("Rows in output:", len(signals_long))
    save_signals(signals_long, OUTPUT_DIR)
    return signals_long

if __name__ == "__main__":
    _signals = main()
