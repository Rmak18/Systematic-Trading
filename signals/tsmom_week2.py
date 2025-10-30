import os
from datetime import datetime
import numpy as np
import pandas as pd

DATA_FILE = "data/prices_daily.csv"
OUTPUT_DIR = "output/signals"
TICKERS = ["EWJ", "FEZ", "SPY"]
SMOOTH_DAYS = 63  # ~3 months

def load_prices(path, tickers):
    df = pd.read_csv(path)
    date_col = "time" if "time" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    keep = [t for t in tickers if t in df.columns]
    if not keep:
        raise ValueError("None of the requested tickers were found in the CSV.")
    px = df[keep].astype(float)
    return px[~px.index.duplicated(keep="last")]

def monthly_log_returns(prices):
    me = prices.resample("ME").last()
    return np.log(me / me.shift(1))

def mom_excl_recent(prices, months):
    mret = monthly_log_returns(prices)
    mom = mret.rolling(months).sum().shift(1)
    mom.index = mom.index + pd.offsets.BMonthBegin(1)
    idx = prices.index.union(mom.index)
    return mom.reindex(idx).ffill().reindex(prices.index)

def build_signals(prices):
    s3 = mom_excl_recent(prices, 3)
    s6 = mom_excl_recent(prices, 6)
    s12 = mom_excl_recent(prices, 12)
    avg = (s3 + s6 + s12) / 3.0
    smoothed = avg.rolling(SMOOTH_DAYS, min_periods=1).mean()
    out = pd.concat(
        {"signal_3m": s3, "signal_6m": s6, "signal_12m": s12, "signal_multi": smoothed},
        axis=1
    )
    return out

def to_long(df, tickers):
    df = df.copy()
    df.index.name = "date"
    wide = {}
    for c in ["signal_3m","signal_6m","signal_12m","signal_multi"]:
        tmp = df[c].reset_index().melt("date", var_name="ticker", value_name=c)
        wide[c] = tmp
    out = wide["signal_3m"]
    for c in ["signal_6m","signal_12m","signal_multi"]:
        out = out.merge(wide[c], on=["date","ticker"], how="inner")
    out = out[out["ticker"].isin(tickers)].sort_values(["date","ticker"]).reset_index(drop=True)
    out["signal_type"] = "multi_horizon_avg_smoothed"
    out["calculation_date"] = datetime.now().strftime("%Y-%m-%d")
    return out

def save(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p1 = os.path.join(out_dir, f"signals_multi_horizon_{ts}.csv")
    p2 = os.path.join(out_dir, "signals_multi_horizon_latest.csv")
    df.to_csv(p1, index=False)
    df.to_csv(p2, index=False)
    print("Saved:", p1)
    print("Saved:", p2)

def main():
    prices = load_prices(DATA_FILE, TICKERS)
    sigs = build_signals(prices)
    out = to_long(sigs, TICKERS)
    save(out, OUTPUT_DIR)
    print(f"Rows exported: {len(out):,}")
    return out

if __name__ == "__main__":
    _ = main()
