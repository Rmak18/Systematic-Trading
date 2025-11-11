import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add gates directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
gates_path = os.path.join(repo_root, 'gates')
sys.path.insert(0, gates_path)

from oil_state_gate_hmm import OilStateGateHMM, prepare_oil_data


@dataclass
class TradingMetrics:
    """Comprehensive trading performance metrics."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: float  # days in drawdown
    total_return: float
    annual_return: float
    annual_vol: float
    win_rate: float
    profit_factor: float
    turnover: float  # trades per year
    avg_holding_period: float  # days
    
    def to_dict(self):
        return {
            'Sharpe': self.sharpe_ratio,
            'Sortino': self.sortino_ratio,
            'Calmar': self.calmar_ratio,
            'Max_DD': self.max_drawdown,
            'Avg_DD': self.avg_drawdown,
            'DD_Duration': self.drawdown_duration,
            'Total_Return': self.total_return,
            'Annual_Return': self.annual_return,
            'Annual_Vol': self.annual_vol,
            'Win_Rate': self.win_rate,
            'Profit_Factor': self.profit_factor,
            'Turnover': self.turnover,
            'Avg_Hold_Days': self.avg_holding_period
        }


class TradingBacktester:
    """Realistic backtester with transaction costs and proper position management."""
    
    def __init__(self, transaction_cost: float = 0.001, slippage: float = 0.0005):
        """
        Args:
            transaction_cost: Proportional cost per trade (0.001 = 0.1% = 10 bps)
            slippage: Additional cost from market impact (0.0005 = 5 bps)
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.total_cost = transaction_cost + slippage
    
    def generate_positions(self, gate_signal: np.ndarray, 
                          entry_threshold: float = 0.7,
                          exit_threshold: float = 0.3,
                          use_smoothed: bool = True) -> np.ndarray:
        """
        Generate trading positions from gate signal.
        
        Args:
            gate_signal: Continuous gate values (0=bear, 1=bull)
            entry_threshold: Enter long when gate > this
            exit_threshold: Exit when gate < this
            use_smoothed: Whether signal is already smoothed
            
        Returns:
            positions: Array of positions (-1, 0, 1)
        """
        positions = np.zeros(len(gate_signal))
        current_position = 0
        
        for i in range(len(gate_signal)):
            if current_position == 0:  # Flat
                if gate_signal[i] > entry_threshold:
                    current_position = 1  # Enter long
                elif gate_signal[i] < (1 - entry_threshold):
                    current_position = -1  # Enter short
            
            elif current_position == 1:  # Long
                if gate_signal[i] < exit_threshold:
                    current_position = 0  # Exit
            
            elif current_position == -1:  # Short
                if gate_signal[i] > (1 - exit_threshold):
                    current_position = 0  # Exit
            
            positions[i] = current_position
        
        return positions
    
    def calculate_returns(self, positions: np.ndarray, 
                         asset_returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate strategy returns including transaction costs.
        
        Returns:
            strategy_returns: Daily returns after costs
            cumulative_returns: Cumulative wealth
        """
        # Detect position changes
        position_changes = np.diff(np.concatenate([[0], positions]))
        trades = np.abs(position_changes)
        
        # Calculate gross returns
        gross_returns = positions[:-1] * asset_returns[1:]
        
        # Apply transaction costs on trades
        costs = trades[1:] * self.total_cost
        net_returns = gross_returns - costs
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + net_returns)
        
        return net_returns, cumulative_returns
    
    def calculate_metrics(self, returns: np.ndarray, 
                         positions: np.ndarray,
                         trading_days: int = 252) -> TradingMetrics:
        """Calculate comprehensive trading metrics."""
        
        if len(returns) == 0 or np.all(returns == 0):
            return TradingMetrics(
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                max_drawdown=0, avg_drawdown=0, drawdown_duration=0,
                total_return=0, annual_return=0, annual_vol=0,
                win_rate=0, profit_factor=0, turnover=0, avg_holding_period=0
            )
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (trading_days / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(trading_days)
        
        # Risk-adjusted returns
        sharpe = (annual_return / annual_vol) if annual_vol > 0 else 0
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 1e-10
        sortino = (annual_return / downside_vol) if downside_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = abs(drawdown.min())
        avg_dd = abs(drawdown[drawdown < 0].mean()) if np.any(drawdown < 0) else 0
        
        # Drawdown duration (days in drawdown)
        in_drawdown = drawdown < -0.01  # More than 1% drawdown
        dd_duration = in_drawdown.sum()
        
        # Calmar ratio
        calmar = (annual_return / max_dd) if max_dd > 0 else 0
        
        # Win rate and profit factor
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        
        win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
        
        gross_profit = winning_days.sum() if len(winning_days) > 0 else 0
        gross_loss = abs(losing_days.sum()) if len(losing_days) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Turnover analysis
        position_changes = np.diff(np.concatenate([[0], positions]))
        num_trades = np.abs(position_changes).sum()
        years = len(returns) / trading_days
        turnover = num_trades / years if years > 0 else 0
        
        # Average holding period
        trades_mask = np.abs(position_changes) > 0
        if trades_mask.sum() > 1:
            trade_indices = np.where(trades_mask)[0]
            holding_periods = np.diff(trade_indices)
            avg_holding = holding_periods.mean() if len(holding_periods) > 0 else 0
        else:
            avg_holding = len(positions)
        
        return TradingMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            drawdown_duration=dd_duration,
            total_return=total_return,
            annual_return=annual_return,
            annual_vol=annual_vol,
            win_rate=win_rate,
            profit_factor=profit_factor,
            turnover=turnover,
            avg_holding_period=avg_holding
        )


class TradingPerformanceValidator:
    """Validate HMM model based on actual trading performance."""
    
    def __init__(self, oil_csv: str, train_end: str = "2018-12-31",
                 transaction_cost: float = 0.001, slippage: float = 0.0005):
        self.oil_csv = oil_csv
        self.train_end = train_end
        self.backtester = TradingBacktester(transaction_cost, slippage)
        self.results = {}
        
    def load_data(self, use_features: List[str] = None):
        """Load and prepare data."""
        if use_features is None:
            use_features = ['oil_return', 'slope', 'vol_20d', 'return_20d']
        
        oil_df = pd.read_csv(self.oil_csv)
        date_col = "time" if "time" in oil_df.columns else "date"
        oil_df[date_col] = pd.to_datetime(oil_df[date_col])
        oil_df = oil_df.set_index(date_col).sort_index()
        
        features_df, features_array, feature_names = prepare_oil_data(oil_df, use_features)
        
        # Add price returns for backtesting
        features_df['price_return'] = features_df['oil_return']
        
        return features_df, features_array, feature_names
    
    def test_seed_trading_performance(self, n_seeds: int = 20, 
                                     entry_threshold: float = 0.7,
                                     exit_threshold: float = 0.3,
                                     smooth_window: int = 5):
        """
        TEST 1: Does random seed affect out-of-sample trading performance?
        This is the critical test my first validation missed.
        """
        print("\n" + "="*70)
        print("TEST 1: SEED IMPACT ON TRADING PERFORMANCE")
        print("="*70)
        print(f"Testing {n_seeds} seeds on OUT-OF-SAMPLE trading results...")
        print(f"Transaction cost: {self.backtester.total_cost*100:.2f}% per trade")
        print(f"Entry threshold: {entry_threshold}, Exit: {exit_threshold}")
        
        features_df, features_array, feature_names = self.load_data()
        
        train_mask = features_df.index <= self.train_end
        train_features = features_array[train_mask]
        test_features = features_array[~train_mask]
        test_returns = features_df['price_return'].values[~train_mask]
        
        seed_results = []
        
        for seed in range(n_seeds):
            # Train model
            hmm = OilStateGateHMM(n_states=2, random_state=seed, n_iter=100)
            hmm.fit(train_features, feature_names=feature_names, verbose=False)
            
            # Generate test signals
            gate = hmm.get_continuous_gate(test_features, sensitivity=1.0, 
                                          smooth_window=smooth_window)
            
            # Generate positions
            positions = self.backtester.generate_positions(gate, entry_threshold, exit_threshold)
            
            # Calculate returns
            strategy_returns, _ = self.backtester.calculate_returns(positions, test_returns)
            
            # Calculate metrics
            metrics = self.backtester.calculate_metrics(strategy_returns, positions[:-1])
            
            result = {'seed': seed}
            result.update(metrics.to_dict())
            seed_results.append(result)
            
            if seed % 5 == 0:
                print(f"  Seed {seed:2d}: Sharpe={metrics.sharpe_ratio:.3f}, "
                      f"Return={metrics.annual_return*100:+.1f}%, "
                      f"MaxDD={metrics.max_drawdown*100:.1f}%")
        
        seed_df = pd.DataFrame(seed_results)
        
        print("\n" + "="*70)
        print("TRADING PERFORMANCE STATISTICS ACROSS SEEDS:")
        print("="*70)
        
        key_metrics = ['Sharpe', 'Sortino', 'Annual_Return', 'Max_DD', 'Turnover']
        summary = seed_df[key_metrics].describe()
        print(summary.to_string())
        
        print("\n" + "="*70)
        print("SEED SENSITIVITY ANALYSIS:")
        print("="*70)
        
        sharpe_range = seed_df['Sharpe'].max() - seed_df['Sharpe'].min()
        sharpe_std = seed_df['Sharpe'].std()
        return_range = seed_df['Annual_Return'].max() - seed_df['Annual_Return'].min()
        
        print(f"Sharpe ratio range: [{seed_df['Sharpe'].min():.3f}, {seed_df['Sharpe'].max():.3f}]")
        print(f"Sharpe ratio std: {sharpe_std:.3f}")
        print(f"Annual return range: [{seed_df['Annual_Return'].min()*100:.1f}%, "
              f"{seed_df['Annual_Return'].max()*100:.1f}%]")
        print(f"Return range span: {return_range*100:.1f}%")
        
        # Verdict based on TRADING performance
        if sharpe_std < 0.1 and return_range < 0.05:
            print("\n✅ VERDICT: Seed has MINIMAL impact on trading performance")
            print("   → Seed optimization NOT needed")
            print("   → Any seed will give similar trading results")
        elif sharpe_std < 0.2 and return_range < 0.10:
            print("\n✓ VERDICT: Seed has MODERATE impact")
            print("   → Some variation but probably acceptable")
            print("   → Could use ensemble to reduce variance")
        else:
            print("\n⚠️  VERDICT: Seed has SIGNIFICANT impact on trading performance")
            print(f"   → Sharpe varies by {sharpe_range:.2f} ({sharpe_std/seed_df['Sharpe'].mean()*100:.1f}% CV)")
            print(f"   → Returns vary by {return_range*100:.1f}%")
            print("   → PSO or ensemble approach RECOMMENDED")
        
        self.results['seed_trading'] = seed_df
        return seed_df
    
    def test_regime_robustness(self, seed: int = 42, n_regimes: int = 4):
        """
        TEST 2: Does the strategy work in different market regimes?
        Tests performance in bull/bear/volatile/calm periods.
        """
        print("\n" + "="*70)
        print("TEST 2: MARKET REGIME ROBUSTNESS")
        print("="*70)
        
        features_df, features_array, feature_names = self.load_data()
        
        train_mask = features_df.index <= self.train_end
        train_features = features_array[train_mask]
        
        # Train model
        hmm = OilStateGateHMM(n_states=2, random_state=seed, n_iter=100)
        hmm.fit(train_features, feature_names=feature_names, verbose=False)
        
        # Get test period
        test_df = features_df[~train_mask].copy()
        test_features = features_array[~train_mask]
        
        # Generate gate and positions
        gate = hmm.get_continuous_gate(test_features, sensitivity=1.0, smooth_window=5)
        positions = self.backtester.generate_positions(gate, 0.7, 0.3)
        
        test_df['gate'] = gate
        test_df['position'] = positions
        
        # Classify market regimes
        test_df['vol_regime'] = pd.qcut(test_df['vol_20d'], q=2, labels=['Low_Vol', 'High_Vol'])
        test_df['trend_regime'] = pd.qcut(test_df['return_20d'], q=2, labels=['Down_Trend', 'Up_Trend'])
        
        # Combined regime
        test_df['market_regime'] = (test_df['vol_regime'].astype(str) + '_' + 
                                    test_df['trend_regime'].astype(str))
        
        print("\nTesting performance in different market conditions:\n")
        
        regime_results = []
        
        for regime in test_df['market_regime'].unique():
            regime_mask = test_df['market_regime'] == regime
            regime_data = test_df[regime_mask]
            
            if len(regime_data) < 20:
                continue
            
            returns = regime_data['price_return'].values
            pos = regime_data['position'].values
            
            strategy_returns, _ = self.backtester.calculate_returns(pos, returns)
            metrics = self.backtester.calculate_metrics(strategy_returns, pos[:-1])
            
            result = {'regime': regime, 'days': len(regime_data)}
            result.update(metrics.to_dict())
            regime_results.append(result)
            
            print(f"{regime:25s} ({len(regime_data):4d} days): "
                  f"Sharpe={metrics.sharpe_ratio:+.2f}, "
                  f"Return={metrics.annual_return*100:+6.1f}%, "
                  f"MaxDD={metrics.max_drawdown*100:5.1f}%")
        
        regime_df = pd.DataFrame(regime_results)
        
        print("\n" + "="*70)
        print("REGIME ROBUSTNESS VERDICT:")
        print("="*70)
        
        sharpe_range = regime_df['Sharpe'].max() - regime_df['Sharpe'].min()
        positive_sharpes = (regime_df['Sharpe'] > 0).sum()
        
        if positive_sharpes == len(regime_df) and sharpe_range < 1.0:
            print("✅ STRATEGY IS ROBUST across all market regimes")
            print("   → Consistent positive Sharpe in all conditions")
        elif positive_sharpes >= len(regime_df) * 0.75:
            print("✓ STRATEGY WORKS in most regimes")
            print(f"   → Positive Sharpe in {positive_sharpes}/{len(regime_df)} regimes")
        else:
            print("⚠️  STRATEGY IS REGIME-DEPENDENT")
            print(f"   → Only {positive_sharpes}/{len(regime_df)} regimes profitable")
            print("   → May need regime-adaptive position sizing")
        
        self.results['regime_robustness'] = regime_df
        return regime_df
    
    def test_parameter_sensitivity(self, seed: int = 42):
        """
        TEST 3: How sensitive is performance to strategy parameters?
        Tests different entry/exit thresholds and smoothing windows.
        """
        print("\n" + "="*70)
        print("TEST 3: PARAMETER SENSITIVITY")
        print("="*70)
        print("Testing different entry/exit thresholds and smoothing windows...\n")
        
        features_df, features_array, feature_names = self.load_data()
        
        train_mask = features_df.index <= self.train_end
        train_features = features_array[train_mask]
        test_features = features_array[~train_mask]
        test_returns = features_df['price_return'].values[~train_mask]
        
        # Train model once
        hmm = OilStateGateHMM(n_states=2, random_state=seed, n_iter=100)
        hmm.fit(train_features, feature_names=feature_names, verbose=False)
        
        # Test parameter combinations
        thresholds = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]
        smooth_windows = [1, 3, 5, 10, 20]
        
        param_results = []
        
        print("Testing threshold combinations:")
        for entry_thresh, exit_thresh in thresholds:
            for smooth in smooth_windows:
                gate = hmm.get_continuous_gate(test_features, sensitivity=1.0, 
                                              smooth_window=smooth)
                positions = self.backtester.generate_positions(gate, entry_thresh, exit_thresh)
                strategy_returns, _ = self.backtester.calculate_returns(positions, test_returns)
                metrics = self.backtester.calculate_metrics(strategy_returns, positions[:-1])
                
                param_results.append({
                    'entry_threshold': entry_thresh,
                    'exit_threshold': exit_thresh,
                    'smooth_window': smooth,
                    **metrics.to_dict()
                })
        
        param_df = pd.DataFrame(param_results)
        
        # Find best configuration
        best_idx = param_df['Sharpe'].idxmax()
        best_params = param_df.loc[best_idx]
        
        print(f"\nBest configuration:")
        print(f"  Entry threshold: {best_params['entry_threshold']:.1f}")
        print(f"  Exit threshold: {best_params['exit_threshold']:.1f}")
        print(f"  Smooth window: {int(best_params['smooth_window'])} days")
        print(f"  Sharpe: {best_params['Sharpe']:.3f}")
        print(f"  Annual return: {best_params['Annual_Return']*100:.1f}%")
        print(f"  Max drawdown: {best_params['Max_DD']*100:.1f}%")
        
        # Worst configuration
        worst_idx = param_df['Sharpe'].idxmin()
        worst_params = param_df.loc[worst_idx]
        
        print(f"\nWorst configuration:")
        print(f"  Entry threshold: {worst_params['entry_threshold']:.1f}")
        print(f"  Exit threshold: {worst_params['exit_threshold']:.1f}")
        print(f"  Smooth window: {int(worst_params['smooth_window'])} days")
        print(f"  Sharpe: {worst_params['Sharpe']:.3f}")
        
        sharpe_range = param_df['Sharpe'].max() - param_df['Sharpe'].min()
        
        print("\n" + "="*70)
        print("PARAMETER SENSITIVITY VERDICT:")
        print("="*70)
        print(f"Sharpe range: {sharpe_range:.3f}")
        
        if sharpe_range < 0.3:
            print("✅ LOW parameter sensitivity")
            print("   → Strategy is robust to parameter choices")
        elif sharpe_range < 0.5:
            print("✓ MODERATE parameter sensitivity")
            print("   → Need to choose parameters carefully")
        else:
            print("⚠️  HIGH parameter sensitivity")
            print("   → Performance heavily depends on parameters")
            print("   → Should optimize thresholds, not just HMM")
        
        self.results['parameter_sensitivity'] = param_df
        return param_df
    
    def test_transaction_cost_impact(self, seed: int = 42):
        """
        TEST 4: How much do transaction costs affect performance?
        Critical for understanding real-world viability.
        """
        print("\n" + "="*70)
        print("TEST 4: TRANSACTION COST IMPACT")
        print("="*70)
        
        features_df, features_array, feature_names = self.load_data()
        
        train_mask = features_df.index <= self.train_end
        train_features = features_array[train_mask]
        test_features = features_array[~train_mask]
        test_returns = features_df['price_return'].values[~train_mask]
        
        # Train model
        hmm = OilStateGateHMM(n_states=2, random_state=seed, n_iter=100)
        hmm.fit(train_features, feature_names=feature_names, verbose=False)
        
        # Generate gate and positions
        gate = hmm.get_continuous_gate(test_features, sensitivity=1.0, smooth_window=5)
        positions = self.backtester.generate_positions(gate, 0.7, 0.3)
        
        # Test different cost levels
        cost_levels = [0, 0.0005, 0.001, 0.002, 0.005, 0.01]  # 0 to 1%
        
        cost_results = []
        
        print("\nTesting different transaction cost levels:\n")
        
        for cost in cost_levels:
            backtester = TradingBacktester(transaction_cost=cost, slippage=0)
            strategy_returns, _ = backtester.calculate_returns(positions, test_returns)
            metrics = backtester.calculate_metrics(strategy_returns, positions[:-1])
            
            cost_results.append({
                'cost_bps': cost * 10000,  # Convert to basis points
                **metrics.to_dict()
            })
            
            print(f"Cost: {cost*10000:5.0f} bps ({cost*100:.2f}%): "
                  f"Sharpe={metrics.sharpe_ratio:+.3f}, "
                  f"Return={metrics.annual_return*100:+6.1f}%, "
                  f"Turnover={metrics.turnover:.1f}/year")
        
        cost_df = pd.DataFrame(cost_results)
        
        # Calculate break-even cost
        positive_sharpe = cost_df[cost_df['Sharpe'] > 0]
        if len(positive_sharpe) > 0:
            breakeven_cost = positive_sharpe['cost_bps'].max()
        else:
            breakeven_cost = 0
        
        print("\n" + "="*70)
        print("TRANSACTION COST VERDICT:")
        print("="*70)
        print(f"Break-even cost: ~{breakeven_cost:.0f} bps ({breakeven_cost/100:.2f}%)")
        
        sharpe_at_zero = cost_df[cost_df['cost_bps'] == 0]['Sharpe'].values[0]
        sharpe_at_10bps = cost_df[cost_df['cost_bps'] == 10]['Sharpe'].values[0]
        sharpe_degradation = sharpe_at_zero - sharpe_at_10bps
        
        if breakeven_cost > 50:
            print("✅ STRATEGY IS ROBUST to transaction costs")
            print(f"   → Can tolerate >{breakeven_cost:.0f} bps before unprofitable")
            print("   → Viable for real trading")
        elif breakeven_cost > 20:
            print("✓ MODERATE cost tolerance")
            print(f"   → Profitable up to ~{breakeven_cost:.0f} bps")
            print("   → Viable for low-cost execution")
        else:
            print("⚠️  STRATEGY IS FRAGILE to costs")
            print(f"   → Only profitable below {breakeven_cost:.0f} bps")
            print("   → May not be viable in real trading")
        
        self.results['transaction_cost'] = cost_df
        return cost_df
    
    def generate_comprehensive_report(self, save_path: str = "trading_validation_report.txt"):
        """Generate comprehensive trading validation report."""
        print("\n" + "="*70)
        print("COMPREHENSIVE TRADING VALIDATION REPORT")
        print("="*70)
        
        report = []
        report.append("="*70)
        report.append("TRADING PERFORMANCE VALIDATION REPORT")
        report.append("="*70)
        report.append("")
        
        # Overall verdict
        needs_optimization = False
        can_trade = True
        issues = []
        
        if 'seed_trading' in self.results:
            seed_df = self.results['seed_trading']
            sharpe_std = seed_df['Sharpe'].std()
            if sharpe_std > 0.2:
                needs_optimization = True
                issues.append(f"High seed sensitivity (Sharpe std={sharpe_std:.3f})")
        
        if 'regime_robustness' in self.results:
            regime_df = self.results['regime_robustness']
            positive_regimes = (regime_df['Sharpe'] > 0).sum()
            if positive_regimes < len(regime_df) * 0.75:
                issues.append(f"Only profitable in {positive_regimes}/{len(regime_df)} regimes")
        
        if 'transaction_cost' in self.results:
            cost_df = self.results['transaction_cost']
            positive_at_10bps = cost_df[cost_df['cost_bps'] == 10]['Sharpe'].values[0] > 0
            if not positive_at_10bps:
                can_trade = False
                issues.append("Strategy unprofitable at realistic transaction costs (10 bps)")
        
        report.append("OVERALL VERDICT:")
        report.append("-"*70)
        
        if not can_trade:
            report.append("❌ STRATEGY NOT VIABLE FOR LIVE TRADING")
            report.append("")
            report.append("Critical issues:")
            for issue in issues:
                report.append(f"  - {issue}")
            report.append("")
            report.append("Recommendation: Focus on reducing turnover or improving signal quality")
        
        elif needs_optimization:
            report.append("⚠️  STRATEGY VIABLE BUT OPTIMIZATION RECOMMENDED")
            report.append("")
            report.append("Issues to address:")
            for issue in issues:
                report.append(f"  - {issue}")
            report.append("")
            report.append("Recommendations:")
            report.append("  1. Use ensemble approach to reduce seed sensitivity")
            report.append("  2. Implement regime-adaptive position sizing")
            report.append("  3. Consider PSO for parameter optimization")
            report.append("  4. Add filters for unfavorable market conditions")
        
        else:
            report.append("✅ STRATEGY IS VIABLE FOR LIVE TRADING")
            report.append("")
            report.append("Key strengths:")
            report.append("  - Consistent performance across random seeds")
            report.append("  - Robust across different market regimes")
            report.append("  - Profitable after realistic transaction costs")
            report.append("")
            report.append("Recommendation: Proceed to live testing with proper risk management")
        
        report.append("")
        report.append("="*70)
        report.append("DETAILED METRICS SUMMARY")
        report.append("="*70)
        
        if 'seed_trading' in self.results:
            seed_df = self.results['seed_trading']
            report.append("")
            report.append("Seed Performance Range:")
            report.append(f"  Sharpe: {seed_df['Sharpe'].min():.3f} to {seed_df['Sharpe'].max():.3f}")
            report.append(f"  Annual Return: {seed_df['Annual_Return'].min()*100:.1f}% to {seed_df['Annual_Return'].max()*100:.1f}%")
            report.append(f"  Max Drawdown: {seed_df['Max_DD'].min()*100:.1f}% to {seed_df['Max_DD'].max()*100:.1f}%")
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nDetailed report saved to: {save_path}")
        
        return needs_optimization, can_trade


def run_full_trading_validation(oil_csv: str, 
                                train_end: str = "2018-12-31",
                                transaction_cost: float = 0.001,
                                slippage: float = 0.0005):
    """Run complete trading-focused validation suite."""
    
    validator = TradingPerformanceValidator(oil_csv, train_end, 
                                           transaction_cost, slippage)
    
    print("="*70)
    print("TRADING PERFORMANCE VALIDATION SUITE")
    print("="*70)
    print(f"Transaction costs: {(transaction_cost + slippage)*100:.3f}% per trade")
    print(f"Train period: Up to {train_end}")
    print(f"Test period: After {train_end}")
    print("="*70)
    
    # Run all tests
    validator.test_seed_trading_performance(n_seeds=20)
    validator.test_regime_robustness()
    validator.test_parameter_sensitivity()
    validator.test_transaction_cost_impact()
    
    # Generate final report
    needs_opt, can_trade = validator.generate_comprehensive_report()
    
    return validator, needs_opt, can_trade


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate HMM based on real trading performance"
    )
    parser.add_argument("--oil_csv", required=True, help="Path to oil prices CSV")
    parser.add_argument("--train_end", default="2018-12-31", help="Training end date")
    parser.add_argument("--transaction_cost", type=float, default=0.001, 
                       help="Transaction cost (default: 0.1%)")
    parser.add_argument("--slippage", type=float, default=0.0005,
                       help="Slippage (default: 0.05%)")
    args = parser.parse_args()
    
    run_full_trading_validation(
        args.oil_csv, 
        args.train_end,
        args.transaction_cost,
        args.slippage
    )