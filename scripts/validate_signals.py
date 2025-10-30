import sys, os
from datetime import datetime
import numpy as np
import pandas as pd

DATA_FILE = "data/prices_daily.csv"
W1_FILE = "output/signals/signals_12m_minus_1_latest.csv"
W2_FILE = "output/signals/signals_multi_horizon_latest.csv"
TICKERS = ["EWJ", "FEZ", "SPY"]
SMOOTH_DAYS = 63
TOL = 1e-9

def load_prices(path, tickers):
    df = pd.read_csv(path)
    date_col = "time" if "time" in df.columns else "date"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    keep = [t for t in tickers if t in df.columns]
    if not keep:
        raise SystemExit("FATAL: none of the requested tickers found in prices CSV.")
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

def build_ref(prices):
    s3 = mom_excl_recent(prices, 3)
    s6 = mom_excl_recent(prices, 6)
    s12 = mom_excl_recent(prices, 12)
    avg = (s3 + s6 + s12) / 3.0
    smoothed = avg.rolling(SMOOTH_DAYS, min_periods=1).mean()
    ref = pd.concat({"signal_3m": s3, "signal_6m": s6, "signal_12m": s12, "signal_multi": smoothed}, axis=1)
    return ref

def pivot_w2(long_df):
    req = {"date","ticker","signal_3m","signal_6m","signal_12m","signal_multi"}
    miss = req - set(long_df.columns)
    if miss:
        raise SystemExit(f"FATAL: W2 file missing columns: {sorted(miss)}")
    w = long_df.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w.pivot(index="date", columns="ticker", values=["signal_3m","signal_6m","signal_12m","signal_multi"])
    return w.sort_index()

def pivot_w1(long_df):
    req = {"date","ticker","signal_12m"}
    miss = req - set(long_df.columns)
    if miss:
        raise SystemExit(f"FATAL: W1 file missing columns: {sorted(miss)}")
    w = long_df.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w.pivot(index="date", columns="ticker", values="signal_12m").sort_index()
    return w

def warmup_cutoffs(w2_long):
    cuts = {}
    for t in TICKERS:
        sub = w2_long[w2_long["ticker"] == t]
        # first date where all signals are non-null for that ticker
        nonnull = sub.dropna(subset=["signal_3m","signal_6m","signal_12m","signal_multi"])
        cuts[t] = nonnull["date"].min() if not nonnull.empty else pd.Timestamp.max
    return cuts

def mae(a, b):
    d = (a - b).abs()
    return float(d.stack().mean()) if isinstance(d, pd.DataFrame) else float(d.mean())

def compare_blocks(ref_block, got_block, label, start_dates_by_ticker):
    ok = True
    for t in TICKERS:
        start = start_dates_by_ticker[t]
        rb = ref_block[t]
        gb = got_block[t]
        mask = (rb.index >= start) & (gb.index >= start)
        if mask.any():
            m = mae(rb[mask], gb[mask])
            status = "PASS" if m <= TOL else "FAIL"
            print(f"[{status}] {label} {t}  MAE={m:.3e}  n={mask.sum()}")
            ok &= (status == "PASS")
    return ok

def main():
    if not os.path.exists(DATA_FILE):
        raise SystemExit("FATAL: data/prices_daily.csv not found.")
    px = load_prices(DATA_FILE, TICKERS)
    ref = build_ref(px)

    if not os.path.exists(W2_FILE):
        raise SystemExit("FATAL: Week-2 file not found: " + W2_FILE)
    w2_long = pd.read_csv(W2_FILE, parse_dates=["date"])
    # allow NaNs during warmup, but not after
    cuts = warmup_cutoffs(w2_long)
    for t in TICKERS:
        start = cuts[t]
        if pd.isna(start):
            continue
        late = w2_long[(w2_long["ticker"] == t) & (w2_long["date"] >= start)]
        if late[["signal_3m","signal_6m","signal_12m","signal_multi"]].isna().any().any():
            raise AssertionError(f"W2: NaNs found for {t} after warm-up start {start.date()}")

    w2 = pivot_w2(w2_long)

    ok = True
    for sig in ["signal_3m","signal_6m","signal_12m","signal_multi"]:
        ok &= compare_blocks(ref[sig][TICKERS], w2[sig][TICKERS], f"W2 {sig}", cuts)

    if os.path.exists(W1_FILE):
        w1_long = pd.read_csv(W1_FILE, parse_dates=["date"])
        w1 = pivot_w1(w1_long)
        # Week-1 comparison only after the 12m warmup for each ticker
        cuts12 = {t: cuts[t] for t in TICKERS}
        ok &= compare_blocks(ref["signal_12m"][TICKERS], w1[TICKERS], "W1 signal_12m", cuts12)
    else:
        print("[WARN] Week-1 file not found; skipping W1 comparison.")

    print("\nSummary:", "ALL CHECKS PASSED ✅" if ok else "SOME CHECKS FAILED ❌")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
