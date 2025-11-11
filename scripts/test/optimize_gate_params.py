import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os
from itertools import product

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
gates_path = os.path.join(repo_root, 'gates')
sys.path.insert(0, gates_path)

from oil_state_gate_hmm import OilStateGateHMM, prepare_oil_data


class GateOptimizer:
    """Optimize oil gate parameters with train/validation/test split."""
    
    def __init__(self, equity_csv: str, oil_csv: str,
                 tickers: list = ['EWJ', 'FEZ', 'SPY'],
                 target_vol: float = 0.10,
                 per_fund_cap: float = 0.25,
                 transaction_cost: float = 0.0002,
                 train_end: str = "2018-12-31",
                 val_end: str = "2021-12-31"):
        
        self.tickers = tickers
        self.target_vol = target_vol
        self.per_fund_cap = per_fund_cap
        self.transaction_cost = transaction_cost
        self.train_end = train_end
        self.val_end = val_end
        
        print("="*70)
        print("INITIALIZING GATE PARAMETER OPTIMIZER")
        print("="*70)
        print(f"Tickers: {tickers}")
        print(f"Train period: 2003 - {train_end}")
        print(f"Validation period: 2019 - {val_end}")
        print(f"Test period: 2022 - 2025")
        
        # Load data
        self.equity_prices = self._load_equity_prices(equity_csv)
        self.oil_hmm, self.oil_features_df, self.oil_features_array = self._load_and_train_hmm(oil_csv)
        
        # Calculate base momentum and volatility once
        self.momentum_signals = self._calculate_momentum_signals(self.equity_prices)
        self.volatility = self._calculate_volatility(self.equity_prices)
        
    def _load_equity_prices(self, csv_path: str) -> pd.DataFrame:
        """Load equity prices."""
        df = pd.read_csv(csv_path)
        date_col = "time" if "time" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        
        available = [t for t in self.tickers if t in df.columns]
        if not available:
            raise ValueError(f"None of the tickers {self.tickers} found")
        
        prices = df[available].astype(float)
        prices = prices[~prices.index.duplicated(keep='last')]
        
        print(f"\nEquity data: {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}")
        return prices
    
    def _load_and_train_hmm(self, csv_path: str) -> Tuple[OilStateGateHMM, pd.DataFrame, np.ndarray]:
        """Load oil data and train HMM ONCE."""
        print("\nTraining Oil HMM (one time)...")
        
        oil_df = pd.read_csv(csv_path)
        date_col = "time" if "time" in oil_df.columns else "date"
        oil_df[date_col] = pd.to_datetime(oil_df[date_col])
        oil_df = oil_df.set_index(date_col).sort_index()
        
        # Prepare features
        use_features = ['oil_return', 'slope', 'vol_20d', 'return_20d']
        features_df, features_array, feature_names = prepare_oil_data(oil_df, use_features)
        
        # Train HMM on train period only
        train_mask = features_df.index <= self.train_end
        train_features = features_array[train_mask]
        
        hmm_model = OilStateGateHMM(n_states=2, random_state=42, n_iter=100)
        hmm_model.fit(train_features, feature_names=feature_names, verbose=False)
        
        print(f"  Bear state mean return: {hmm_model.model.means_[hmm_model.bear_state, 0]:.6f}")
        print(f"  Bull state mean return: {hmm_model.model.means_[hmm_model.bull_state, 0]:.6f}")
        print(f"  Oil features: {len(features_df)} days")
        
        return hmm_model, features_df, features_array
    
    def _calculate_momentum_signals(self, prices: pd.DataFrame, smooth_days: int = 63) -> pd.DataFrame:
        """Calculate multi-horizon momentum signals."""
        monthly_end = prices.resample('ME').last()
        monthly_returns = np.log(monthly_end / monthly_end.shift(1))
        
        mom_3m = monthly_returns.rolling(3).sum().shift(1)
        mom_6m = monthly_returns.rolling(6).sum().shift(1)
        mom_12m = monthly_returns.rolling(12).sum().shift(1)
        
        mom_3m.index = mom_3m.index + pd.offsets.BMonthBegin(1)
        mom_6m.index = mom_6m.index + pd.offsets.BMonthBegin(1)
        mom_12m.index = mom_12m.index + pd.offsets.BMonthBegin(1)
        
        idx = prices.index.union(mom_3m.index).union(mom_6m.index).union(mom_12m.index)
        
        mom_3m_daily = mom_3m.reindex(idx).ffill().reindex(prices.index)
        mom_6m_daily = mom_6m.reindex(idx).ffill().reindex(prices.index)
        mom_12m_daily = mom_12m.reindex(idx).ffill().reindex(prices.index)
        
        avg_momentum = (mom_3m_daily + mom_6m_daily + mom_12m_daily) / 3.0
        smoothed = avg_momentum.rolling(smooth_days, min_periods=1).mean()
        
        return smoothed
    
    def _calculate_volatility(self, prices: pd.DataFrame, ewma_com: int = 60) -> pd.DataFrame:
        """Calculate EWMA volatility."""
        returns = prices.pct_change().fillna(0.0)
        alpha = 1.0 / (1.0 + ewma_com)
        ewma_var = (returns ** 2).ewm(alpha=alpha, adjust=False).mean()
        vol_annual = np.sqrt(252 * ewma_var)
        return vol_annual
    
    def generate_gate(self, sensitivity: float, smooth_window: int) -> pd.Series:
        """Generate oil gate with given parameters."""
        gate = self.oil_hmm.get_continuous_gate(
            self.oil_features_array, 
            sensitivity=sensitivity, 
            smooth_window=smooth_window
        )
        gate_series = pd.Series(gate, index=self.oil_features_df.index, name='oil_gate')
        
        # Align with equity prices
        gate_normalized = gate_series.copy()
        gate_normalized.index = pd.to_datetime(gate_normalized.index).normalize()
        equity_dates_norm = pd.to_datetime(self.equity_prices.index).normalize()
        gate_temp = gate_normalized.reindex(equity_dates_norm).bfill().ffill()
        gate_aligned = pd.Series(gate_temp.values, index=self.equity_prices.index, name='oil_gate')
        
        return gate_aligned
    
    def _calculate_weights(self, signals: pd.DataFrame, volatility: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to portfolio weights."""
        vol_safe = volatility.clip(lower=1e-12)
        w_pre = signals / vol_safe
        
        pre_sq_sum = (w_pre ** 2).sum(axis=1).clip(lower=1e-12)
        scale = self.target_vol / np.sqrt(pre_sq_sum)
        w_target = w_pre.mul(scale, axis=0)
        
        zero_mask = (signals.abs().sum(axis=1) < 1e-12)
        w_target[zero_mask] = 0.0
        
        w_final = w_target.clip(lower=-self.per_fund_cap, upper=self.per_fund_cap)
        return w_final
    
    def backtest_with_gate(self, gate_aligned: pd.Series, period_start: str = None, 
                          period_end: str = None) -> Dict:
        """Backtest momentum + gate strategy for a given period."""
        # Apply gate to momentum
        gated_signals = self.momentum_signals.mul(gate_aligned, axis=0)
        
        # Calculate weights
        weights = self._calculate_weights(gated_signals, self.volatility)
        
        # Filter to period
        if period_start:
            weights = weights[weights.index > period_start]
        if period_end:
            weights = weights[weights.index <= period_end]
        
        prices_period = self.equity_prices.loc[weights.index]
        
        # Calculate returns
        returns = prices_period.pct_change()
        weights_aligned = weights.shift(1).fillna(0)
        portfolio_returns = (weights_aligned * returns).sum(axis=1)
        
        # Transaction costs
        weight_changes = weights.diff().abs().sum(axis=1)
        tc = weight_changes * self.transaction_cost
        portfolio_returns = portfolio_returns - tc
        
        # Calculate metrics
        returns_clean = portfolio_returns.dropna()
        
        if len(returns_clean) == 0:
            return {'sharpe': 0, 'sortino': 0, 'annual_return': 0, 'annual_vol': 0,
                   'max_dd': 0, 'calmar': 0, 'annual_turnover': 0, 'days': 0}
        
        total_return = (1 + returns_clean).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns_clean)) - 1
        annual_vol = returns_clean.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Sortino
        downside_returns = returns_clean[returns_clean < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino = (returns_clean.mean() * 252) / downside_vol if downside_vol > 0 else 0
        else:
            sortino = 0
        
        # Drawdown
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        # Calmar
        calmar = annual_return / max_dd if max_dd > 0 else 0
        
        # Turnover
        annual_turnover = weight_changes.mean() * 252
        
        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'max_dd': max_dd,
            'calmar': calmar,
            'annual_turnover': annual_turnover,
            'total_return': total_return,
            'days': len(returns_clean)
        }
    
    def backtest_baseline(self, period_start: str = None, period_end: str = None) -> Dict:
        """Backtest momentum-only (no gate) strategy."""
        weights = self._calculate_weights(self.momentum_signals, self.volatility)
        
        if period_start:
            weights = weights[weights.index > period_start]
        if period_end:
            weights = weights[weights.index <= period_end]
        
        prices_period = self.equity_prices.loc[weights.index]
        
        returns = prices_period.pct_change()
        weights_aligned = weights.shift(1).fillna(0)
        portfolio_returns = (weights_aligned * returns).sum(axis=1)
        
        weight_changes = weights.diff().abs().sum(axis=1)
        tc = weight_changes * self.transaction_cost
        portfolio_returns = portfolio_returns - tc
        
        returns_clean = portfolio_returns.dropna()
        
        if len(returns_clean) == 0:
            return {'sharpe': 0, 'annual_return': 0, 'annual_vol': 0, 'max_dd': 0}
        
        total_return = (1 + returns_clean).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(returns_clean)) - 1
        annual_vol = returns_clean.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(drawdown.min())
        
        annual_turnover = weight_changes.mean() * 252
        
        return {
            'sharpe': sharpe,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'max_dd': max_dd,
            'annual_turnover': annual_turnover,
            'total_return': total_return,
            'days': len(returns_clean)
        }
    
    def run_optimization(self, sensitivity_levels: List[float], 
                        smooth_windows: List[int]) -> pd.DataFrame:
        """Run optimization grid search."""
        print("\n" + "="*70)
        print("RUNNING PARAMETER GRID SEARCH")
        print("="*70)
        print(f"Sensitivity levels: {sensitivity_levels}")
        print(f"Smoothing windows: {smooth_windows}")
        print(f"Total configurations: {len(sensitivity_levels) * len(smooth_windows)}")
        
        # Calculate baseline performance
        print("\n" + "="*70)
        print("BASELINE: MOMENTUM ONLY (No Gate)")
        print("="*70)
        
        baseline_train = self.backtest_baseline(period_end=self.train_end)
        baseline_val = self.backtest_baseline(period_start=self.train_end, period_end=self.val_end)
        baseline_test = self.backtest_baseline(period_start=self.val_end)
        
        print(f"Train (2003-2018): Sharpe={baseline_train['sharpe']:.3f}, DD={baseline_train['max_dd']:.2%}")
        print(f"Val (2019-2021):   Sharpe={baseline_val['sharpe']:.3f}, DD={baseline_val['max_dd']:.2%}")
        print(f"Test (2022-2025):  Sharpe={baseline_test['sharpe']:.3f}, DD={baseline_test['max_dd']:.2%}")
        
        # Test all configurations
        results = []
        
        for sensitivity, smooth in product(sensitivity_levels, smooth_windows):
            print(f"\nTesting: sensitivity={sensitivity}, smooth_window={smooth}")
            
            # Generate gate with these parameters
            gate = self.generate_gate(sensitivity, smooth)
            
            # Backtest on all periods
            train_metrics = self.backtest_with_gate(gate, period_end=self.train_end)
            val_metrics = self.backtest_with_gate(gate, period_start=self.train_end, 
                                                 period_end=self.val_end)
            test_metrics = self.backtest_with_gate(gate, period_start=self.val_end)
            
            print(f"  Train: Sharpe={train_metrics['sharpe']:.3f}, Turnover={train_metrics['annual_turnover']:.2f}")
            print(f"  Val:   Sharpe={val_metrics['sharpe']:.3f}, Turnover={val_metrics['annual_turnover']:.2f}")
            print(f"  Test:  Sharpe={test_metrics['sharpe']:.3f}, Turnover={test_metrics['annual_turnover']:.2f}")
            
            results.append({
                'sensitivity': sensitivity,
                'smooth_window': smooth,
                'train_sharpe': train_metrics['sharpe'],
                'train_return': train_metrics['annual_return'],
                'train_vol': train_metrics['annual_vol'],
                'train_dd': train_metrics['max_dd'],
                'train_turnover': train_metrics['annual_turnover'],
                'val_sharpe': val_metrics['sharpe'],
                'val_return': val_metrics['annual_return'],
                'val_vol': val_metrics['annual_vol'],
                'val_dd': val_metrics['max_dd'],
                'val_turnover': val_metrics['annual_turnover'],
                'test_sharpe': test_metrics['sharpe'],
                'test_return': test_metrics['annual_return'],
                'test_vol': test_metrics['annual_vol'],
                'test_dd': test_metrics['max_dd'],
                'test_turnover': test_metrics['annual_turnover'],
            })
        
        results_df = pd.DataFrame(results)
        
        # Add baseline row
        baseline_row = {
            'sensitivity': 'BASELINE',
            'smooth_window': 'N/A',
            'train_sharpe': baseline_train['sharpe'],
            'train_return': baseline_train['annual_return'],
            'train_vol': baseline_train['annual_vol'],
            'train_dd': baseline_train['max_dd'],
            'train_turnover': baseline_train['annual_turnover'],
            'val_sharpe': baseline_val['sharpe'],
            'val_return': baseline_val['annual_return'],
            'val_vol': baseline_val['annual_vol'],
            'val_dd': baseline_val['max_dd'],
            'val_turnover': baseline_val['annual_turnover'],
            'test_sharpe': baseline_test['sharpe'],
            'test_return': baseline_test['annual_return'],
            'test_vol': baseline_test['annual_vol'],
            'test_dd': baseline_test['max_dd'],
            'test_turnover': baseline_test['annual_turnover'],
        }
        
        results_df = pd.concat([pd.DataFrame([baseline_row]), results_df], ignore_index=True)
        
        return results_df
    
    def analyze_results(self, results_df: pd.DataFrame, output_dir: str = "output"):
        """Analyze and visualize optimization results."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS ANALYSIS")
        print("="*70)
        
        # Separate baseline from configurations
        baseline = results_df.iloc[0]
        configs = results_df.iloc[1:].copy()
        
        # Sort by validation Sharpe (this is our selection criterion)
        configs_sorted = configs.sort_values('val_sharpe', ascending=False)
        
        print("\n" + "="*70)
        print("TOP 5 CONFIGURATIONS (by Validation Sharpe)")
        print("="*70)
        print("\nRank | Sens | Smooth | Val Sharpe | Test Sharpe | Val DD | Test DD | Turnover")
        print("-" * 85)
        
        for i, (_, row) in enumerate(configs_sorted.head(5).iterrows(), 1):
            print(f"{i:4d} | {row['sensitivity']:4.2f} | {int(row['smooth_window']):6d} | "
                  f"{row['val_sharpe']:10.3f} | {row['test_sharpe']:11.3f} | "
                  f"{row['val_dd']:6.2%} | {row['test_dd']:7.2%} | {row['val_turnover']:8.2f}")
        
        print("\n" + "="*70)
        print("BASELINE COMPARISON")
        print("="*70)
        print(f"\nBaseline (Momentum Only):")
        print(f"  Val:  Sharpe={baseline['val_sharpe']:.3f}, DD={baseline['val_dd']:.2%}, Turnover={baseline['val_turnover']:.2f}")
        print(f"  Test: Sharpe={baseline['test_sharpe']:.3f}, DD={baseline['test_dd']:.2%}, Turnover={baseline['test_turnover']:.2f}")
        
        best_config = configs_sorted.iloc[0]
        print(f"\nBest Config (sensitivity={best_config['sensitivity']}, smooth={int(best_config['smooth_window'])}):")
        print(f"  Val:  Sharpe={best_config['val_sharpe']:.3f}, DD={best_config['val_dd']:.2%}, Turnover={best_config['val_turnover']:.2f}")
        print(f"  Test: Sharpe={best_config['test_sharpe']:.3f}, DD={best_config['test_dd']:.2%}, Turnover={best_config['test_turnover']:.2f}")
        
        val_sharpe_improvement = best_config['val_sharpe'] - baseline['val_sharpe']
        test_sharpe_improvement = best_config['test_sharpe'] - baseline['test_sharpe']
        
        print(f"\nImprovements:")
        print(f"  Validation Sharpe: {val_sharpe_improvement:+.3f} ({val_sharpe_improvement/baseline['val_sharpe']*100:+.1f}%)")
        print(f"  Test Sharpe: {test_sharpe_improvement:+.3f} ({test_sharpe_improvement/baseline['test_sharpe']*100:+.1f}%)")
        
        # Save detailed results
        results_df.to_csv(f"{output_dir}/gate_optimization_results.csv", index=False)
        print(f"\nDetailed results saved to: {output_dir}/gate_optimization_results.csv")
        
        # Create visualization
        self.plot_optimization_results(configs, baseline, output_dir)
        
        return best_config
    
    def plot_optimization_results(self, configs: pd.DataFrame, baseline: pd.Series, 
                                  output_dir: str):
        """Create visualization of optimization results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Validation Sharpe Heatmap
        ax1 = axes[0, 0]
        pivot_val = configs.pivot(index='smooth_window', columns='sensitivity', values='val_sharpe')
        im1 = ax1.imshow(pivot_val.values, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(range(len(pivot_val.columns)))
        ax1.set_xticklabels([f'{x:.2f}' for x in pivot_val.columns])
        ax1.set_yticks(range(len(pivot_val.index)))
        ax1.set_yticklabels([int(x) for x in pivot_val.index])
        ax1.set_xlabel('Sensitivity')
        ax1.set_ylabel('Smooth Window')
        ax1.set_title('Validation Sharpe Ratio Heatmap')
        plt.colorbar(im1, ax=ax1)
        
        # Add baseline line
        ax1.axhline(y=-0.5, color='red', linestyle='--', linewidth=2, 
                   label=f'Baseline: {baseline["val_sharpe"]:.3f}')
        ax1.legend(loc='upper right')
        
        # Panel 2: Test Sharpe Heatmap
        ax2 = axes[0, 1]
        pivot_test = configs.pivot(index='smooth_window', columns='sensitivity', values='test_sharpe')
        im2 = ax2.imshow(pivot_test.values, cmap='RdYlGn', aspect='auto')
        ax2.set_xticks(range(len(pivot_test.columns)))
        ax2.set_xticklabels([f'{x:.2f}' for x in pivot_test.columns])
        ax2.set_yticks(range(len(pivot_test.index)))
        ax2.set_yticklabels([int(x) for x in pivot_test.index])
        ax2.set_xlabel('Sensitivity')
        ax2.set_ylabel('Smooth Window')
        ax2.set_title('Test Sharpe Ratio Heatmap')
        plt.colorbar(im2, ax=ax2)
        
        # Panel 3: Turnover vs Sharpe (Validation)
        ax3 = axes[1, 0]
        ax3.scatter(configs['val_turnover'], configs['val_sharpe'], 
                   c=configs['smooth_window'], cmap='viridis', s=100, alpha=0.6)
        ax3.axhline(y=baseline['val_sharpe'], color='red', linestyle='--', 
                   linewidth=1, label='Baseline Sharpe')
        ax3.axvline(x=baseline['val_turnover'], color='red', linestyle='--', 
                   linewidth=1, label='Baseline Turnover')
        ax3.set_xlabel('Annual Turnover')
        ax3.set_ylabel('Validation Sharpe')
        ax3.set_title('Turnover vs Sharpe (Validation Period)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        cbar3 = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar3.set_label('Smooth Window')
        
        # Panel 4: Train vs Test Sharpe (overfitting check)
        ax4 = axes[1, 1]
        ax4.scatter(configs['val_sharpe'], configs['test_sharpe'], 
                   c=configs['sensitivity'], cmap='plasma', s=100, alpha=0.6)
        
        # Add diagonal line (perfect generalization)
        min_sharpe = min(configs['val_sharpe'].min(), configs['test_sharpe'].min())
        max_sharpe = max(configs['val_sharpe'].max(), configs['test_sharpe'].max())
        ax4.plot([min_sharpe, max_sharpe], [min_sharpe, max_sharpe], 
                'k--', linewidth=1, alpha=0.5, label='Perfect Generalization')
        
        ax4.set_xlabel('Validation Sharpe')
        ax4.set_ylabel('Test Sharpe')
        ax4.set_title('Validation vs Test Performance (Overfitting Check)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        cbar4 = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar4.set_label('Sensitivity')
        
        plt.tight_layout()
        output_path = f"{output_dir}/gate_optimization_heatmaps.png"
        plt.savefig(output_path, dpi=150)
        print(f"Optimization plots saved to: {output_path}")
        plt.close()


def run_gate_optimization(equity_csv: str = "data/prices_daily.csv",
                         oil_csv: str = "data/oil_prices.csv",
                         output_dir: str = "output"):
    """Main optimization workflow."""
    
    # Initialize optimizer
    optimizer = GateOptimizer(
        equity_csv=equity_csv,
        oil_csv=oil_csv,
        train_end="2018-12-31",
        val_end="2021-12-31"
    )
    
    # Define parameter grid (focused and practical)
    sensitivity_levels = [0.75, 1.0, 1.25]  # Modest range around current
    smooth_windows = [5, 10, 20, 30]        # Address turnover
    
    # Run optimization
    results_df = optimizer.run_optimization(sensitivity_levels, smooth_windows)
    
    # Analyze results
    best_config = optimizer.analyze_results(results_df, output_dir)
    
    # Save recommended parameters
    recommended_params = {
        'sensitivity': float(best_config['sensitivity']),
        'smooth_window': int(best_config['smooth_window']),
        'validation_sharpe': float(best_config['val_sharpe']),
        'test_sharpe': float(best_config['test_sharpe']),
        'validation_drawdown': float(best_config['val_dd']),
        'test_drawdown': float(best_config['test_dd']),
        'validation_turnover': float(best_config['val_turnover']),
        'test_turnover': float(best_config['test_turnover'])
    }
    
    import json
    params_file = f"{output_dir}/gate_recommended_params.json"
    with open(params_file, 'w') as f:
        json.dump(recommended_params, f, indent=2)
    
    print(f"\n✅ Recommended parameters saved to: {params_file}")
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"\nRecommended Parameters:")
    print(f"  sensitivity: {recommended_params['sensitivity']}")
    print(f"  smooth_window: {recommended_params['smooth_window']}")
    print(f"\nNext steps:")
    print("  1. Review optimization plots in output/")
    print("  2. Update momentum_oil_backtest.py with recommended parameters")
    print("  3. Run final full-period backtest with these parameters")
    
    return optimizer, results_df, recommended_params


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize oil gate parameters")
    parser.add_argument("--equity_csv", default="data/prices_daily.csv",
                       help="CSV with equity prices")
    parser.add_argument("--oil_csv", default="data/oil_prices.csv",
                       help="CSV with oil prices")
    parser.add_argument("--output_dir", default="output",
                       help="Output directory for results")
    args = parser.parse_args()
    
    optimizer, results_df, recommended_params = run_gate_optimization(
        args.equity_csv,
        args.oil_csv,
        args.output_dir
    )