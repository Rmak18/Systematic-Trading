# %%
import pandas as pd
import numpy as np
from pathlib import Path


PRICES_CSV = Path("Systematic-Trading-main/data/prices_daily.csv")
SIGNALS_CSV = Path("Systematic-Trading-main/output/signals/signals_12m_minus_1_latest.csv")

OUT_VOL = Path("Systematic-Trading-main/output/vol_ewma_annualized.csv")
OUT_BEFORE = Path("Systematic-Trading-main/output/weights_before_caps.csv")
OUT_AFTER = Path("Systematic-Trading-main/output/weights_after_caps.csv")

# Vol targeting + risk model
TARGET_VOL = 0.10        # 10% annualized portfolio volatility
PER_FUND_CAP = 0.25      # ±25% per-fund cap
COM = 60                 # EWMA center-of-mass ≈ 60 days
ALPHA = 1.0 / (1.0 + COM)
ANNUAL_DAYS = 252   
EPS = 1e-12    


def load_prices_wide(path):
    df = pd.read_csv(path)
    # normalize date column name
    date_col = "time" if "time" in df.columns else "date"                                                       
    if date_col not in df.columns:
        raise ValueError("Prices file must have a 'date' or 'time' column.")
    df = df.rename(columns={date_col: "date"})
    # sort dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").dropna(subset=["date"]).reset_index(drop = True)
    # set index
    return df.set_index("date").sort_index()


def compute_ewma_vol_annual(prices, alpha, annual_days):
    rets = prices.pct_change().fillna(0.0)  # start EWMA without a gap
    ew_var_daily = (rets ** 2).ewm(alpha = alpha, adjust = False).mean()
    vol_annual = np.sqrt(annual_days * ew_var_daily)
    return vol_annual


def load_signals_long_to_wide(path):
    s = pd.read_csv(path)
    date_col = "time" if "time" in s.columns else "date"
    if date_col not in s.columns:
        raise ValueError("Signals file must have 'date' or 'time' + 'ticker' + 'signal_12m'.")
    if not {"ticker", "signal_12m"}.issubset(s.columns):
        raise ValueError("Signals file must include columns: 'ticker' and 'signal_12m'.")

    s = s.rename(columns={date_col: "date"})
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)

    wide = s.pivot_table(index = "date", columns = "ticker", values = "signal_12m", aggfunc = "last")
    return wide.sort_index()


def align_signals_and_vol(signals_wide, vol_annual):
    common = sorted(set(signals_wide.columns) & set(vol_annual.columns))
    if not common:
        raise ValueError("No common tickers between signals and prices/volatility.")

    sig = signals_wide[common].copy()
    vol = vol_annual[common].reindex(sig.index).ffill()
    return sig, vol


def compute_weights(signals, vol, target_vol):
    vol_safe = vol.clip(lower=EPS)
    w_pre = signals / vol_safe

    pre_sq_sum = (w_pre.pow(2)).sum(axis=1).clip(lower=EPS)
    scale = target_vol / np.sqrt(pre_sq_sum)

    w_target = w_pre.mul(scale, axis=0)

    zero_mask = (signals.abs().sum(axis=1) < EPS)
    w_target[zero_mask] = 0.0

    return w_target



def apply_per_fund_cap(weights,cap):
    return weights.clip(lower = -cap, upper = cap)



def main():
    prices = load_prices_wide(PRICES_CSV)
    vol_ewma_annual = compute_ewma_vol_annual(prices, alpha = ALPHA, annual_days = ANNUAL_DAYS)
    vol_ewma_annual.to_csv(OUT_VOL) 

    signals_wide = load_signals_long_to_wide(SIGNALS_CSV)

    signals, vol_aligned = align_signals_and_vol(signals_wide, vol_ewma_annual)

    w_before_caps = compute_weights(signals, vol_aligned, target_vol=TARGET_VOL)

    # Per-fund cap ±25%
    w_after_caps = apply_per_fund_cap(w_before_caps, cap = PER_FUND_CAP)

    # Outputs
    w_before_caps.reset_index().to_csv(OUT_BEFORE, index = False)
    w_after_caps.reset_index().to_csv(OUT_AFTER,  index = False)

if __name__ == "__main__":
    main()



# %%

