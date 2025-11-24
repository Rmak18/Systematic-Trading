from AlgorithmImports import *
from regimes import scale
from neel import neel_scale
import math
import numpy as np
from datetime import timedelta

class TimeSeriesMomentumStrategy(QCAlgorithm):
    """
    Time Series Momentum Strategy for SPY

    Rules:
    - Calculate the return over the previous 12 months
    - If the 12-month return is positive, go long SPY for the next month
    - If the 12-month return is negative or zero, hold no position (stay in cash)
    - Rebalance monthly
    - Allocation is scaled each month by BOTH a regime scale and a neel scale
    - Sharpe ratio is computed with MONTHLY returns and MONTHLY volatility
    """

    def Initialize(self):
        # Backtest span and cash
        self.SetStartDate(2007, 5, 1)
        self.SetEndDate(2025, 8, 31)
        self.SetCash(100000)

        # Asset
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.gold = self.AddEquity("GLD", Resolution.DAILY).Symbol

        # Fees & benchmark
        self.Securities[self.spy].FeeModel = ConstantFeeModel(1)
        self.Securities[self.gold].FeeModel = ConstantFeeModel(1)
        self.SetBenchmark("SPY")

        # Monthly scheduling:
        # 1) Track prior month's return at month-end close (true month boundary)
        self.Schedule.On(
            self.DateRules.MonthEnd(self.spy),
            self.TimeRules.BeforeMarketClose(self.spy, 0),
            self.TrackMonthlyReturn
        )
        # 2) Rebalance at next month start
        self.Schedule.On(
            self.DateRules.MonthStart(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 30),
            self.Rebalance
        )

        # State
        self.scale_dict = scale
        self.scale_dict_neel = neel_scale

        # Monthly performance tracking
        self.monthly_returns = []
        self._last_month_equity = None

        # Warmup to compute 12M momentum
        self.SetWarmUp(timedelta(days=365))

    # ----- helpers for scales -----
    def GetScaleFactor(self, current_date):
        """Regime scale: use exact date if present, else most recent prior date; default 1.0."""
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str in self.scale_dict:
            return float(self.scale_dict[date_str])
        # find most recent prior
        prior = [d for d in self.scale_dict.keys() if d <= date_str]
        if prior:
            return float(self.scale_dict[sorted(prior)[-1]])
        return 1.0

    def GetScaleFactorNeel(self, current_date):
        """Neel scale: use exact date if present, else most recent prior date; default 1.0."""
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str in self.scale_dict_neel:
            return float(self.scale_dict_neel[date_str])
        prior = [d for d in self.scale_dict_neel.keys() if d <= date_str]
        if prior:
            return float(self.scale_dict_neel[sorted(prior)[-1]])
        return 1.0

    # ----- strategy logic -----
    def Rebalance(self):
        """
        Monthly rebalancing:
        - Compute 12M momentum
        - If positive, set target allocation scaled by BOTH regime scales
        - Else go to cash
        """
        # Need ~12 months of data
        history = self.History(self.spy, 252, Resolution.Daily)
        if history.empty or len(history) < 252:
            self.Debug(f"Not enough historical data: {len(history)} bars")
            return

        price_12m_ago = float(history['close'].iloc[0])
        current_price = float(history['close'].iloc[-1])
        twelve_month_return = (current_price - price_12m_ago) / price_12m_ago

        # Both scaling factors
        s_regime = self.GetScaleFactor(self.Time)
        s_neel = self.GetScaleFactorNeel(self.Time)

        # Default: no position
        target_allocation = 0.0

        if twelve_month_return > 0:
            # Base long signal with BOTH scales applied
            target_allocation = 1.0 * s_regime * s_neel
            # Clamp to [0, 1]
            target_allocation = max(0.0, min(1.0, target_allocation))

        # Only rebalance if different by > ~1%
        current_allocation = 0.0
        if self.Portfolio.TotalPortfolioValue > 0:
            current_allocation = self.Portfolio[self.spy].HoldingsValue / self.Portfolio.TotalPortfolioValue

        if abs(current_allocation - target_allocation) > 0.01:
            if target_allocation > 0:
                self.set_holdings(self.spy, target_allocation)
                if 1 - target_allocation < 0.01:
                    self.liquidate(self.gold)
                else:
                    self.set_holdings(self.gold, 1 - target_allocation)
            else:
                self.liquidate(self.spy)
                self.set_holdings(self.gold, 1)

    def TrackMonthlyReturn(self):
        """
        Called at month-end just before close.
        Records the portfolio's monthly return based on changes in TotalPortfolioValue.
        """
        equity = float(self.Portfolio.TotalPortfolioValue)
        if self._last_month_equity is None:
            # Initialize the baseline at first month-end encountered
            self._last_month_equity = equity
            return

        if self._last_month_equity > 0:
            r = (equity / self._last_month_equity) - 1.0
            self.monthly_returns.append(r)
            # Optional: plot or debug
            # self.Debug(f"[{self.Time:%Y-%m-%d}] Monthly Return: {r:.2%} (n={len(self.monthly_returns)})")

        # reset baseline for next month
        self._last_month_equity = equity

    def OnEndOfAlgorithm(self):
        """
        Compute Sharpe ratio using MONTHLY returns and MONTHLY volatility.
        Also prints an annualized Sharpe (monthly Sharpe * sqrt(12)) for context.
        Risk-free set to 0% on monthly basis (you can wire in a monthly RF series if desired).
        """
        n = len(self.monthly_returns)
        if n == 0:
            self.Log("No monthly returns collected; Sharpe cannot be computed.")
            return

        rets = np.array(self.monthly_returns, dtype=float)
        mu = float(np.mean(rets))                         # average monthly return
        sigma = float(np.std(rets, ddof=1)) if n > 1 else float(np.std(rets, ddof=0))  # monthly vol

        if sigma == 0:
            self.Log("Monthly volatility is zero; Sharpe undefined.")
            return

        sharpe_monthly = mu / sigma
        sharpe_annualized = sharpe_monthly * math.sqrt(12.0)

        # Nicely formatted summary
        self.Log("==== Monthly Performance (Strategy) ====")
        self.Log(f"Samples (months): {n}")
        self.Log(f"Avg Monthly Return: {mu:.4%}")
        self.Log(f"Monthly Volatility: {sigma:.4%}")
        self.Log(f"Sharpe (Monthly basis): {sharpe_monthly:.3f}")
        self.Log(f"Sharpe (Annualized from monthly): {sharpe_annualized:.3f}")

    def OnData(self, data):
        # All logic handled via scheduled functions
        pass
