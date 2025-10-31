from AlgorithmImports import *
import pandas as pd

class TwoBpsFeeModel(FeeModel):
    """A simple 2 bps fee on trade notional."""
    def GetOrderFee(self, parameters: OrderFeeParameters) -> OrderFee:
        order = parameters.Order
        security = parameters.Security

        notional = abs(order.Quantity) * security.Price
        fee = notional * 0.0002 

        currency = security.QuoteCurrency.Symbol
        return OrderFee(CashAmount(fee, currency))

class MonthlyTSMOMSpy(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2000, 1, 1)
        self.SetCash(10_000)
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        self.SetSecurityInitializer(lambda sec: sec.SetFeeModel(TwoBpsFeeModel()))
        
        self.SetWarmUp(252, Resolution.Daily)
        
        self.Schedule.On(
            self.DateRules.MonthStart(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 5),
            self.Rebalance
        )
        
        self.nextRebalance = None

    def Rebalance(self):
        if self.IsWarmingUp:
            return
        
        if self.nextRebalance and self.Time < self.nextRebalance:
            return

        hist = self.History(self.spy, 252, Resolution.Daily)
        if hist.empty:
            return
        
        try:
            closes = hist.loc[self.spy]['close']
        except Exception:
            closes = hist['close']
        
        if len(closes) < 2:
            return
        
        twelve_month_return = closes.iloc[-1] / closes.iloc[0] - 1.0
        
        if twelve_month_return > 0:
            self.SetHoldings(self.spy, 1.0)
        else:
            self.Liquidate(self.spy)
        
        self.nextRebalance = self.Time + timedelta(days=28)

    def OnSecuritiesChanged(self, changes: SecurityChanges):
        for sec in changes.AddedSecurities:
            sec.SetFeeModel(TwoBpsFeeModel())