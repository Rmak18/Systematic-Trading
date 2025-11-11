import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import sys
import os

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
gates_path = os.path.join(repo_root, 'gates')
sys.path.insert(0, gates_path)

from oil_state_gate_hmm import OilStateGateHMM, prepare_oil_data


class MomentumBacktester:
    """Compare momentum strategy with and without oil gate - DEBUG VERSION."""
    
    def __init__(self, equity_prices_csv: str, oil_prices_csv: str,
                 tickers: list = ['EWJ', 'FEZ', 'SPY'],
                 target_vol: float = 0.10,
                 per_fund_cap: float = 0.25,
                 transaction_cost: float = 0.0002,
                 train_end: str = "2018-12-31"):
        
        self.tickers = tickers
        self.target_vol = target_vol
        self.per_fund_cap = per_fund_cap
        self.transaction_cost = transaction_cost
        self.train_end = train_end
        
        # Load data
        print("Loading data...")
        self.equity_prices = self.load_equity_prices(equity_prices_csv)
        self.oil_hmm, self.oil_gate_daily = self.load_oil_gate(oil_prices_csv)
        
        print(f"Equity data: {len(self.equity_prices)} days from {self.equity_prices.index[0].date()} to {self.equity_prices.index[-1].date()}")
        print(f"Oil gate: {len(self.oil_gate_daily)} days from {self.oil_gate_daily.index[0].date()} to {self.oil_gate_daily.index[-1].date()}")
    
    def load_equity_prices(self, csv_path: str) -> pd.DataFrame:
        """Load equity prices for tickers."""
        df = pd.read_csv(csv_path)
        date_col = "time" if "time" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        # Keep only requested tickers
        available = [t for t in self.tickers if t in df.columns]
        if not available:
            raise ValueError(f"None of the tickers {self.tickers} found in {csv_path}")
        
        prices = df[available].astype(float)
        prices = prices[~prices.index.duplicated(keep='last')]
        
        return prices
    
    def load_oil_gate(self, csv_path: str) -> Tuple[OilStateGateHMM, pd.Series]:
        """Load oil data and train HMM to generate gate."""
        print("\nTraining Oil HMM...")
        
        oil_df = pd.read_csv(csv_path)
        date_col = "time" if "time" in oil_df.columns else "date"
        oil_df[date_col] = pd.to_datetime(oil_df[date_col])
        oil_df = oil_df.set_index(date_col).sort_index()
        
        # Prepare features
        use_features = ['oil_return', 'slope', 'vol_20d', 'return_20d']
        features_df, features_array, feature_names = prepare_oil_data(oil_df, use_features)
        
        # Split train/test
        train_mask = features_df.index <= self.train_end
        train_features = features_array[train_mask]
        
        # Train HMM
        hmm_model = OilStateGateHMM(n_states=2, random_state=42, n_iter=100)
        hmm_model.fit(train_features, feature_names=feature_names, verbose=False)
        
        print(f"  Bear state mean return: {hmm_model.model.means_[hmm_model.bear_state, 0]:.6f}")
        print(f"  Bull state mean return: {hmm_model.model.means_[hmm_model.bull_state, 0]:.6f}")
        
        # Generate gate for full period
        gate = hmm_model.get_continuous_gate(features_array, sensitivity=1.0, smooth_window=5)
        gate_series = pd.Series(gate, index=features_df.index, name='oil_gate')
        
        return hmm_model, gate_series
    
    def calculate_momentum_signals(self, prices: pd.DataFrame, 
                                   smooth_days: int = 63) -> pd.DataFrame:
        """
        Calculate multi-horizon momentum signals.
        
        Following tsmom_week2.py logic:
        - 3m, 6m, 12m momentum
        - Skip most recent month
        - Average and smooth with 63-day window
        """
        # Monthly end prices
        monthly_end = prices.resample('ME').last()
        
        # Monthly log returns
        monthly_returns = np.log(monthly_end / monthly_end.shift(1))
        
        # Calculate momentum for each horizon
        mom_3m = monthly_returns.rolling(3).sum().shift(1)
        mom_6m = monthly_returns.rolling(6).sum().shift(1)
        mom_12m = monthly_returns.rolling(12).sum().shift(1)
        
        # Shift to first business day of next month
        mom_3m.index = mom_3m.index + pd.offsets.BMonthBegin(1)
        mom_6m.index = mom_6m.index + pd.offsets.BMonthBegin(1)
        mom_12m.index = mom_12m.index + pd.offsets.BMonthBegin(1)
        
        # Forward fill to daily
        idx = prices.index.union(mom_3m.index).union(mom_6m.index).union(mom_12m.index)
        
        mom_3m_daily = mom_3m.reindex(idx).ffill().reindex(prices.index)
        mom_6m_daily = mom_6m.reindex(idx).ffill().reindex(prices.index)
        mom_12m_daily = mom_12m.reindex(idx).ffill().reindex(prices.index)
        
        # Average
        avg_momentum = (mom_3m_daily + mom_6m_daily + mom_12m_daily) / 3.0
        
        # Smooth with rolling window
        smoothed = avg_momentum.rolling(smooth_days, min_periods=1).mean()
        
        return smoothed
    
    def calculate_volatility(self, prices: pd.DataFrame, 
                           ewma_com: int = 60) -> pd.DataFrame:
        """Calculate EWMA volatility (annualized)."""
        returns = prices.pct_change().fillna(0.0)
        alpha = 1.0 / (1.0 + ewma_com)
        
        ewma_var = (returns ** 2).ewm(alpha=alpha, adjust=False).mean()
        vol_annual = np.sqrt(252 * ewma_var)
        
        return vol_annual
    
    def calculate_weights(self, signals: pd.DataFrame, 
                         volatility: pd.DataFrame) -> pd.DataFrame:
        """
        Convert signals to portfolio weights with volatility targeting.
        
        Following vol_targeting_and_caps.py logic:
        1. Base weights = signal / volatility
        2. Scale to hit target portfolio vol
        3. Apply per-fund caps
        """
        # Base weights (inverse vol)
        vol_safe = volatility.clip(lower=1e-12)
        w_pre = signals / vol_safe
        
        # Scale to hit target vol
        pre_sq_sum = (w_pre ** 2).sum(axis=1).clip(lower=1e-12)
        scale = self.target_vol / np.sqrt(pre_sq_sum)
        w_target = w_pre.mul(scale, axis=0)
        
        # Zero out if no signals
        zero_mask = (signals.abs().sum(axis=1) < 1e-12)
        w_target[zero_mask] = 0.0
        
        # Apply per-fund caps
        w_final = w_target.clip(lower=-self.per_fund_cap, upper=self.per_fund_cap)
        
        return w_final
    
    def backtest_strategy(self, weights: pd.DataFrame, 
                         prices: pd.DataFrame,
                         name: str) -> Dict:
        """
        Backtest a strategy given weights.
        
        Returns daily portfolio returns and metrics.
        """
        # Calculate returns
        returns = prices.pct_change()
        
        # Align weights and returns (weights at t, returns from t to t+1)
        weights_aligned = weights.shift(1).fillna(0)
        
        # Portfolio returns
        portfolio_returns = (weights_aligned * returns).sum(axis=1)
        
        # Transaction costs
        weight_changes = weights.diff().abs().sum(axis=1)
        tc = weight_changes * self.transaction_cost
        portfolio_returns = portfolio_returns - tc
        
        # Calculate metrics
        metrics = self.calculate_metrics(portfolio_returns, weights, name)
        
        return {
            'returns': portfolio_returns,
            'weights': weights,
            'metrics': metrics,
            'cumulative': (1 + portfolio_returns).cumprod()
        }
    
    def calculate_metrics(self, returns: pd.Series, weights: pd.DataFrame, 
                         name: str) -> Dict:
        """Calculate comprehensive performance metrics."""
        # Drop NaN
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {'name': name, 'sharpe': 0, 'annual_return': 0, 'annual_vol': 0, 
                   'max_dd': 0, 'total_return': 0, 'calmar': 0}
        
        # Basic stats
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        # Calmar
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        # Turnover
        weight_changes = weights.diff().abs().sum(axis=1)
        annual_turnover = weight_changes.mean() * 252
        
        return {
            'name': name,
            'sharpe': sharpe,
            'sortino': self.calculate_sortino(returns),
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'max_dd': max_dd,
            'calmar': calmar,
            'total_return': total_return,
            'annual_turnover': annual_turnover
        }
    
    def calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0
        downside_vol = downside_returns.std() * np.sqrt(252)
        annual_return = returns.mean() * 252
        return annual_return / downside_vol if downside_vol > 0 else 0
    
    def run_comparison(self) -> Tuple[Dict, Dict]:
        """Run both strategies and compare - WITH EXTENSIVE DEBUGGING."""
        print("\n" + "="*70)
        print("CALCULATING MOMENTUM SIGNALS")
        print("="*70)
        
        # Calculate momentum signals
        momentum_signals = self.calculate_momentum_signals(self.equity_prices)
        
        print("\n" + "="*70)
        print("DEBUG: MOMENTUM SIGNALS")
        print("="*70)
        print(f"Shape: {momentum_signals.shape}")
        print(f"Date range: {momentum_signals.index[0].date()} to {momentum_signals.index[-1].date()}")
        print(f"Total values: {momentum_signals.size}")
        print(f"NaN count: {momentum_signals.isna().sum().sum()}")
        print(f"Non-NaN count: {(~momentum_signals.isna()).sum().sum()}")
        print(f"Zero count: {(momentum_signals == 0).sum().sum()}")
        print(f"Non-zero count: {((momentum_signals != 0) & (~momentum_signals.isna())).sum().sum()}")
        print(f"\nFirst valid index: {momentum_signals.apply(pd.Series.first_valid_index).min()}")
        print(f"Last valid index: {momentum_signals.apply(pd.Series.last_valid_index).max()}")
        print(f"\nSample values (first 20 rows):")
        print(momentum_signals.head(20))
        print(f"\nSample values (around 2004-01):")
        print(momentum_signals.loc['2004-01-01':'2004-01-31'])
        
        # Calculate volatility
        volatility = self.calculate_volatility(self.equity_prices)
        
        print("\n" + "="*70)
        print("DEBUG: VOLATILITY")
        print("="*70)
        print(f"Shape: {volatility.shape}")
        print(f"NaN count: {volatility.isna().sum().sum()}")
        print(f"First valid index: {volatility.apply(pd.Series.first_valid_index).min()}")
        
        # Align oil gate with equity data
        oil_gate_aligned = self.oil_gate_daily.reindex(self.equity_prices.index).ffill()
        
        print("\n" + "="*70)
        print("DEBUG: OIL GATE (BEFORE ALIGNMENT)")
        print("="*70)
        print(f"Shape: {self.oil_gate_daily.shape}")
        print(f"Date range: {self.oil_gate_daily.index[0].date()} to {self.oil_gate_daily.index[-1].date()}")
        print(f"First valid date: {self.oil_gate_daily.first_valid_index().date()}")
        print(f"Last valid date: {self.oil_gate_daily.last_valid_index().date()}")
        print(f"NaN count: {self.oil_gate_daily.isna().sum()}")
        print(f"Value range: [{self.oil_gate_daily.min():.3f}, {self.oil_gate_daily.max():.3f}]")
        print(f"Mean: {self.oil_gate_daily.mean():.3f}")
        print(f"\nFirst 10 values:")
        print(self.oil_gate_daily.head(10))
        
        print("\n" + "="*70)
        print("DEBUG: OIL GATE (AFTER ALIGNMENT)")
        print("="*70)
        print(f"Shape: {oil_gate_aligned.shape}")
        print(f"Date range: {oil_gate_aligned.index[0].date()} to {oil_gate_aligned.index[-1].date()}")
        print(f"NaN count: {oil_gate_aligned.isna().sum()}")
        print(f"First NaN positions:")
        nan_mask = oil_gate_aligned.isna()
        if nan_mask.any():
            nan_dates = oil_gate_aligned[nan_mask].index[:10]
            print(f"  First 10 NaN dates: {[d.date() for d in nan_dates]}")
        print(f"First valid index: {oil_gate_aligned.first_valid_index()}")
        print(f"Value range: [{oil_gate_aligned.min():.3f}, {oil_gate_aligned.max():.3f}]")
        print(f"Mean (excluding NaN): {oil_gate_aligned.mean():.3f}")
        print(f"\nFirst 30 values:")
        print(oil_gate_aligned.head(30))
        print(f"\nValues around 2004-01:")
        print(oil_gate_aligned.loc['2004-01-01':'2004-01-31'])
        
        print("\n" + "="*70)
        print("STRATEGY 1: MOMENTUM ONLY (Baseline)")
        print("="*70)
        
        # Strategy 1: Momentum only
        weights_momentum = self.calculate_weights(momentum_signals, volatility)
        
        print("\n" + "="*70)
        print("DEBUG: WEIGHTS (MOMENTUM ONLY)")
        print("="*70)
        print(f"Shape: {weights_momentum.shape}")
        print(f"NaN count: {weights_momentum.isna().sum().sum()}")
        print(f"Zero count: {(weights_momentum == 0).sum().sum()}")
        print(f"Non-zero count: {((weights_momentum != 0) & (~weights_momentum.isna())).sum().sum()}")
        print(f"Weight range: [{weights_momentum.min().min():.6f}, {weights_momentum.max().max():.6f}]")
        print(f"\nFirst 30 rows:")
        print(weights_momentum.head(30))
        
        results_momentum = self.backtest_strategy(weights_momentum, self.equity_prices, 
                                                  "Momentum Only")
        
        print(f"\nPortfolio returns: {len(results_momentum['returns'])} days")
        
        print("\n" + "="*70)
        print("STRATEGY 2: MOMENTUM + OIL GATE")
        print("="*70)
        
        # Strategy 2: Momentum + Oil Gate
        # Apply gate as multiplier to signals
        print("\n" + "="*70)
        print("DEBUG: APPLYING GATE TO SIGNALS")
        print("="*70)
        print(f"Momentum signals shape: {momentum_signals.shape}")
        print(f"Oil gate aligned shape: {oil_gate_aligned.shape}")
        print(f"Momentum signals index: {momentum_signals.index[0]} to {momentum_signals.index[-1]}")
        print(f"Oil gate aligned index: {oil_gate_aligned.index[0]} to {oil_gate_aligned.index[-1]}")
        
        gated_signals = momentum_signals.mul(oil_gate_aligned, axis=0)
        
        print("\n" + "="*70)
        print("DEBUG: GATED SIGNALS (AFTER MULTIPLICATION)")
        print("="*70)
        print(f"Shape: {gated_signals.shape}")
        print(f"Total values: {gated_signals.size}")
        print(f"NaN count: {gated_signals.isna().sum().sum()}")
        print(f"Non-NaN count: {(~gated_signals.isna()).sum().sum()}")
        print(f"Zero count (including NaN as zero): {((gated_signals == 0) | gated_signals.isna()).sum().sum()}")
        print(f"Non-zero count: {((gated_signals != 0) & (~gated_signals.isna())).sum().sum()}")
        print(f"\nFirst valid index: {gated_signals.apply(pd.Series.first_valid_index).min()}")
        print(f"Last valid index: {gated_signals.apply(pd.Series.last_valid_index).max()}")
        print(f"\nFirst 30 rows:")
        print(gated_signals.head(30))
        print(f"\nValues around 2004-01:")
        print(gated_signals.loc['2004-01-01':'2004-01-31'])
        
        # Check where both are valid
        print("\n" + "="*70)
        print("DEBUG: OVERLAP ANALYSIS")
        print("="*70)
        mom_valid = ~momentum_signals.isna().any(axis=1)
        gate_valid = ~oil_gate_aligned.isna()
        both_valid = mom_valid & gate_valid
        print(f"Dates where momentum is valid: {mom_valid.sum()}")
        print(f"Dates where gate is valid: {gate_valid.sum()}")
        print(f"Dates where BOTH are valid: {both_valid.sum()}")
        if both_valid.any():
            print(f"First date with both valid: {momentum_signals.index[both_valid][0].date()}")
            print(f"Last date with both valid: {momentum_signals.index[both_valid][-1].date()}")
        
        weights_gated = self.calculate_weights(gated_signals, volatility)
        
        print("\n" + "="*70)
        print("DEBUG: WEIGHTS (GATED)")
        print("="*70)
        print(f"Shape: {weights_gated.shape}")
        print(f"NaN count: {weights_gated.isna().sum().sum()}")
        print(f"Zero count: {(weights_gated == 0).sum().sum()}")
        print(f"Non-zero count: {((weights_gated != 0) & (~weights_gated.isna())).sum().sum()}")
        print(f"Weight range: [{weights_gated.min().min():.6f}, {weights_gated.max().max():.6f}]")
        print(f"\nFirst 30 rows:")
        print(weights_gated.head(30))
        print(f"\nCompare weights side by side (first valid momentum date):")
        first_mom_valid = momentum_signals.apply(pd.Series.first_valid_index).min()
        print(f"\nAround {first_mom_valid.date()}:")
        compare_df = pd.DataFrame({
            'mom_signal': momentum_signals.loc[first_mom_valid],
            'gate': oil_gate_aligned.loc[first_mom_valid] if first_mom_valid in oil_gate_aligned.index else np.nan,
            'gated_signal': gated_signals.loc[first_mom_valid],
            'weight_mom': weights_momentum.loc[first_mom_valid],
            'weight_gated': weights_gated.loc[first_mom_valid]
        })
        print(compare_df)
        
        results_gated = self.backtest_strategy(weights_gated, self.equity_prices,
                                              "Momentum + Oil Gate")
        
        print(f"\nPortfolio returns: {len(results_gated['returns'])} days")
        
        print("\n" + "="*70)
        print("DEBUG: PORTFOLIO RETURNS COMPARISON")
        print("="*70)
        ret_mom = results_momentum['returns'].dropna()
        ret_gated = results_gated['returns'].dropna()
        print(f"Momentum returns - Length: {len(ret_mom)}, Non-zero: {(ret_mom != 0).sum()}")
        print(f"Gated returns - Length: {len(ret_gated)}, Non-zero: {(ret_gated != 0).sum()}")
        
        return results_momentum, results_gated
    
    def print_comparison(self, results_momentum: Dict, results_gated: Dict):
        """Print detailed comparison."""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        
        m1 = results_momentum['metrics']
        m2 = results_gated['metrics']
        
        print(f"\n{'Metric':<25s} {'Momentum Only':>18s} {'Mom + Oil Gate':>18s} {'Difference':>15s}")
        print("-"*80)
        
        metrics = [
            ('Sharpe Ratio', 'sharpe', '.3f'),
            ('Sortino Ratio', 'sortino', '.3f'),
            ('Annual Return', 'annual_return', '.2%'),
            ('Annual Volatility', 'annual_vol', '.2%'),
            ('Max Drawdown', 'max_dd', '.2%'),
            ('Calmar Ratio', 'calmar', '.3f'),
            ('Total Return', 'total_return', '.2%'),
            ('Annual Turnover', 'annual_turnover', '.2f'),
        ]
        
        for label, key, fmt in metrics:
            v1 = m1[key]
            v2 = m2[key]
            diff = v2 - v1
            
            # Format values
            if '%' in fmt:
                v1_str = f"{v1:{fmt}}"
                v2_str = f"{v2:{fmt}}"
                diff_str = f"{diff:+{fmt}}"
            else:
                v1_str = f"{v1:{fmt}}"
                v2_str = f"{v2:{fmt}}"
                diff_str = f"{diff:+{fmt}}"
            
            print(f"{label:<25s} {v1_str:>18s} {v2_str:>18s} {diff_str:>15s}")
    
    def plot_comparison(self, results_momentum: Dict, results_gated: Dict,
                       output_file: str = "momentum_comparison.png"):
        """Plot equity curves and metrics."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Align series
        dates = results_momentum['cumulative'].index
        cum_mom = results_momentum['cumulative']
        cum_gated = results_gated['cumulative']
        
        # Panel 1: Cumulative returns
        ax1 = axes[0]
        ax1.plot(dates, cum_mom, linewidth=2, label='Momentum Only', color='blue')
        ax1.plot(dates, cum_gated, linewidth=2, label='Momentum + Oil Gate', color='green')
        ax1.axvline(pd.Timestamp(self.train_end), color='red', linestyle='--', 
                   linewidth=1, alpha=0.5, label='Train/Test Split')
        ax1.set_ylabel('Cumulative Return')
        ax1.set_title('Portfolio Cumulative Returns')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Rolling Sharpe (1-year)
        ax2 = axes[1]
        window = 252
        
        ret_mom = results_momentum['returns']
        ret_gated = results_gated['returns']
        
        sharpe_mom = (ret_mom.rolling(window).mean() * 252) / (ret_mom.rolling(window).std() * np.sqrt(252))
        sharpe_gated = (ret_gated.rolling(window).mean() * 252) / (ret_gated.rolling(window).std() * np.sqrt(252))
        
        ax2.plot(dates, sharpe_mom, linewidth=1.5, label='Momentum Only', color='blue', alpha=0.7)
        ax2.plot(dates, sharpe_gated, linewidth=1.5, label='Momentum + Oil Gate', color='green', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(pd.Timestamp(self.train_end), color='red', linestyle='--', 
                   linewidth=1, alpha=0.5)
        ax2.set_ylabel('Rolling Sharpe (1Y)')
        ax2.set_title('Rolling 1-Year Sharpe Ratio')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Oil gate value
        ax3 = axes[2]
        oil_gate_aligned = self.oil_gate_daily.reindex(dates).ffill()
        ax3.plot(dates, oil_gate_aligned, linewidth=1.5, color='orange', label='Oil Gate Value')
        ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax3.axvline(pd.Timestamp(self.train_end), color='red', linestyle='--', 
                   linewidth=1, alpha=0.5)
        ax3.fill_between(dates, 0, oil_gate_aligned, alpha=0.2, color='orange')
        ax3.set_ylabel('Gate Value (0-1)')
        ax3.set_xlabel('Date')
        ax3.set_title('Oil State Gate (Applied to Momentum Signals)')
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"\nPlot saved: {output_file}")
        
        return fig


def run_momentum_oil_comparison(equity_csv: str = "data/prices_daily.csv",
                                oil_csv: str = "data/oil_prices.csv",
                                train_end: str = "2018-12-31",
                                output_plot: str = "momentum_oil_comparison_debug.png"):
    """Main function to run comparison - DEBUG VERSION."""
    
    print("="*70)
    print("MOMENTUM vs MOMENTUM+OIL GATE BACKTEST (DEBUG MODE)")
    print("="*70)
    print(f"Equities: EWJ, FEZ, SPY")
    print(f"Target Vol: 10%")
    print(f"Per-fund cap: ±25%")
    print(f"Transaction cost: 2 bps")
    print(f"Train/test split: {train_end}")
    print("="*70)
    
    # Initialize backtester
    backtester = MomentumBacktester(
        equity_prices_csv=equity_csv,
        oil_prices_csv=oil_csv,
        train_end=train_end
    )
    
    # Run comparison
    results_momentum, results_gated = backtester.run_comparison()
    
    # Print results
    backtester.print_comparison(results_momentum, results_gated)
    
    # Plot
    backtester.plot_comparison(results_momentum, results_gated, output_plot)
    
    return backtester, results_momentum, results_gated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare Momentum vs Momentum+Oil Gate strategies - DEBUG VERSION"
    )
    parser.add_argument("--equity_csv", default="data/prices_daily.csv",
                       help="CSV with equity prices")
    parser.add_argument("--oil_csv", default="data/oil_prices.csv",
                       help="CSV with oil prices")
    parser.add_argument("--train_end", default="2018-12-31",
                       help="Train/test split date")
    parser.add_argument("--output_plot", default="momentum_oil_comparison_debug.png",
                       help="Output plot filename")
    args = parser.parse_args()
    
    run_momentum_oil_comparison(
        args.equity_csv,
        args.oil_csv,
        args.train_end,
        args.output_plot
    )