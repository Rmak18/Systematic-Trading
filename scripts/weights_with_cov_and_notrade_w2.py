import pandas as pd
import numpy as np
from pathlib import Path

PRICES_CSV = Path("Systematic-Trading-main/data/prices_daily.csv")
SIGNALS_CSV = Path("Systematic-Trading-main/output/signals/signals_12m_minus_1_latest.csv")
SIGNALS_CSV_MH = Path("Systematic-Trading-main/output/signals/signals_multi_horizon_latest.csv")

OUT_VOL = Path("Systematic-Trading-main/output/vol_ewma_annualized.csv")
OUT_BEFORE = Path("Systematic-Trading-main/output/weights_before_caps.csv")
OUT_AFTER = Path("Systematic-Trading-main/output/weights_after_caps.csv")

TARGET_VOL    = 0.10
PER_FUND_CAP  = 0.25
COM           = 60
ALPHA         = 1.0 / (1.0 + COM)
ANNUAL_DAYS   = 252
EPS           = 1e-12
SPAN_COV      = 60
NO_TRADE_BAND = 0.01

def load_prices_wide(path):
    df = pd.read_csv(path)
    date_col = "time" if "time" in df.columns else "date"
    if date_col not in df.columns:
        raise ValueError("Prices file must have a 'date' or 'time' column.")
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    return df.set_index("date").sort_index()

def compute_ewma_vol_annual(prices, alpha, annual_days):
    rets = prices.pct_change().fillna(0.0)
    ew_var_daily = (rets ** 2).ewm(alpha=alpha, adjust=False).mean()
    vol_annual = np.sqrt(annual_days * ew_var_daily)
    return vol_annual

def load_signals_long_to_wide(path):
    s = pd.read_csv(path)
    date_col = "time" if "time" in s.columns else "date"
    if date_col not in s.columns or not {"ticker", "signal_12m"}.issubset(s.columns):
        raise ValueError("Signals file must have 'date' (or 'time'), 'ticker', and 'signal_12m'.")
    s = s.rename(columns={date_col: "date"})
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    wide = s.pivot_table(index="date", columns="ticker", values="signal_12m", aggfunc="last")
    return wide.sort_index()

def align_signals_and_vol(signals_wide, vol_annual):
    common = sorted(set(signals_wide.columns) & set(vol_annual.columns))
    if not common:
        raise ValueError("No common tickers between signals and prices/volatility.")
    sig = signals_wide[common].copy()
    vol = vol_annual[common].reindex(sig.index).ffill()
    return sig, vol

def compute_weights(signals, vol_daily, target_vol):
    vol_safe = vol_daily.clip(lower=EPS)
    w_pre = signals / vol_safe
    pre_sq_sum = (w_pre.pow(2)).sum(axis=1).clip(lower=EPS)
    scale = target_vol / np.sqrt(pre_sq_sum)
    w_target = w_pre.mul(scale, axis=0)
    zero_mask = (signals.abs().sum(axis=1) < EPS)
    w_target[zero_mask] = 0.0
    return w_target

def apply_per_fund_cap(weights, cap):
    return weights.clip(lower=-cap, upper=cap)

def main():
    prices = load_prices_wide(PRICES_CSV)
    vol_ewma_annual = compute_ewma_vol_annual(prices, alpha=ALPHA, annual_days=ANNUAL_DAYS)
    vol_ewma_annual.to_csv(OUT_VOL)

    signals_wide = load_signals_long_to_wide(SIGNALS_CSV)
    signals, vol_aligned = align_signals_and_vol(signals_wide, vol_ewma_annual)

    w_before_caps = compute_weights(signals, vol_aligned / np.sqrt(ANNUAL_DAYS), target_vol=TARGET_VOL)
    w_after_caps  = apply_per_fund_cap(w_before_caps, cap=PER_FUND_CAP)

    w_before_caps.reset_index().to_csv(OUT_BEFORE, index=False)
    w_after_caps.reset_index().to_csv(OUT_AFTER,  index=False)

    OUT = Path("Systematic-Trading-main/output/weights")
    OUT.mkdir(parents=True, exist_ok=True)
    OUT_BEFORE_W2 = OUT / "weights_before_caps_W2.csv"
    OUT_AFTER_W2  = OUT / "weights_after_caps_notrade_W2.csv"

    signals = load_signals_long_to_wide(SIGNALS_CSV_MH)
    signals, vol = align_signals_and_vol(signals, vol_ewma_annual)

    rets = prices.pct_change().reindex(index=signals.index, columns=signals.columns).fillna(0.0)
    mean_ew = rets.ewm(alpha=2.0/(SPAN_COV+1.0), adjust=False).mean()
    resid   = (rets - mean_ew).fillna(0.0)

    cov_all = resid.ewm(alpha=2.0/(SPAN_COV+1.0), adjust=False).cov(pairwise=True)

    vol_daily = vol[signals.columns].reindex(signals.index).ffill() / np.sqrt(ANNUAL_DAYS)
    prev = pd.Series(0.0, index=signals.columns)

    rows_bef, rows_aft = [], []
    for dt in signals.index:
        Sigma_df = cov_all.xs(dt, level=0).reindex(index=signals.columns, columns=signals.columns).fillna(0.0)
        Sigma_t  = Sigma_df.values

        w0 = (signals.loc[dt] / vol_daily.loc[dt]).fillna(0.0).values
        temp = Sigma_t @ w0
        var_p = w0 @ temp

        if var_p > EPS:
            scaling = (TARGET_VOL/np.sqrt(ANNUAL_DAYS)) / np.sqrt(var_p)
            w_scaled = w0 * scaling
        else:
            w_scaled = np.zeros_like(w0)

        w_before = pd.Series(w_scaled, index=signals.columns, name=dt)
        w_capped = apply_per_fund_cap(w_before, PER_FUND_CAP)

        w_after = pd.Series(index=signals.columns, name=dt)
        for ticker in signals.columns:
            diff = abs(w_capped[ticker] - prev[ticker])
            w_after[ticker] = w_capped[ticker] if diff >= NO_TRADE_BAND else prev[ticker]

        rows_bef.append(w_before)
        rows_aft.append(w_after)
        prev = w_after

    pd.DataFrame(rows_bef).reset_index().rename(columns={'index':'date'}).to_csv(OUT_BEFORE_W2, index=False)
    pd.DataFrame(rows_aft).reset_index().rename(columns={'index':'date'}).to_csv(OUT_AFTER_W2,  index=False)

if __name__ == "__main__":
    main()

