import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import sys
import os
from dataclasses import dataclass
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Add gates directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(current_dir))
gates_path = os.path.join(repo_root, 'gates')
sys.path.insert(0, gates_path)

from oil_state_gate_hmm import OilStateGateHMM, prepare_oil_data


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    entry_threshold: float
    exit_threshold: float
    smooth_window: int
    sensitivity: float
    
    def __repr__(self):
        return (f"TradingConfig(entry={self.entry_threshold:.2f}, "
                f"exit={self.exit_threshold:.2f}, "
                f"smooth={self.smooth_window}, "
                f"sens={self.sensitivity:.2f})")


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation."""
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    annual_return: float
    annual_vol: float
    max_drawdown: float
    avg_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    turnover: float
    avg_holding_days: float
    
    def meets_constraints(self, max_dd_constraint: float = 0.30) -> bool:
        """Check if metrics meet constraints."""
        return self.max_drawdown <= max_dd_constraint
    
    def to_dict(self):
        return {
            'sharpe': self.sharpe_ratio,
            'sortino': self.sortino_ratio,
            'calmar': self.calmar_ratio,
            'annual_return': self.annual_return,
            'annual_vol': self.annual_vol,
            'max_dd': self.max_drawdown,
            'avg_dd': self.avg_drawdown,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'turnover': self.turnover,
            'avg_holding_days': self.avg_holding_days
        }


class TradingBacktester:
    """Backtester for strategy evaluation."""
    
    def __init__(self, transaction_cost: float = 0.001, slippage: float = 0.0005):
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.total_cost = transaction_cost + slippage
    
    def generate_positions(self, gate_signal: np.ndarray, 
                          config: TradingConfig) -> np.ndarray:
        """Generate trading positions from gate signal."""
        positions = np.zeros(len(gate_signal))
        current_position = 0
        
        for i in range(len(gate_signal)):
            if current_position == 0:  # Flat
                if gate_signal[i] > config.entry_threshold:
                    current_position = 1  # Enter long
                elif gate_signal[i] < (1 - config.entry_threshold):
                    current_position = -1  # Enter short
            
            elif current_position == 1:  # Long
                if gate_signal[i] < config.exit_threshold:
                    current_position = 0  # Exit
            
            elif current_position == -1:  # Short
                if gate_signal[i] > (1 - config.exit_threshold):
                    current_position = 0  # Exit
            
            positions[i] = current_position
        
        return positions
    
    def calculate_returns(self, positions: np.ndarray, 
                         asset_returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate strategy returns including transaction costs."""
        position_changes = np.diff(np.concatenate([[0], positions]))
        trades = np.abs(position_changes)
        
        gross_returns = positions[:-1] * asset_returns[1:]
        costs = trades[1:] * self.total_cost
        net_returns = gross_returns - costs
        
        cumulative_returns = np.cumprod(1 + net_returns)
        
        return net_returns, cumulative_returns
    
    def calculate_metrics(self, returns: np.ndarray, 
                         positions: np.ndarray,
                         trading_days: int = 252) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if len(returns) == 0 or np.all(returns == 0):
            return PerformanceMetrics(
                sharpe_ratio=-999, sortino_ratio=-999, calmar_ratio=-999,
                annual_return=0, annual_vol=0, max_drawdown=1.0,
                avg_drawdown=0, total_return=0, win_rate=0,
                profit_factor=0, turnover=0, avg_holding_days=0
            )
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (trading_days / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(trading_days)
        
        # Risk-adjusted returns
        sharpe = (annual_return / annual_vol) if annual_vol > 0 else -999
        
        # Sortino
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else 1e-10
        sortino = (annual_return / downside_vol) if downside_vol > 0 else -999
        
        # Drawdown analysis
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = abs(drawdown.min())
        avg_dd = abs(drawdown[drawdown < 0].mean()) if np.any(drawdown < 0) else 0
        
        # Calmar
        calmar = (annual_return / max_dd) if max_dd > 0 else -999
        
        # Win rate and profit factor
        winning_days = returns[returns > 0]
        losing_days = returns[returns < 0]
        win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
        
        gross_profit = winning_days.sum() if len(winning_days) > 0 else 0
        gross_loss = abs(losing_days.sum()) if len(losing_days) > 0 else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Turnover
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
        
        return PerformanceMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            annual_return=annual_return,
            annual_vol=annual_vol,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            turnover=turnover,
            avg_holding_days=avg_holding
        )


class TradingParameterOptimizer:
    """PSO-based optimizer for trading parameters."""
    
    def __init__(self, oil_csv: str, train_end: str = "2018-12-31",
                 transaction_cost: float = 0.001, slippage: float = 0.0005,
                 max_dd_constraint: float = 0.30):
        self.oil_csv = oil_csv
        self.train_end = train_end
        self.backtester = TradingBacktester(transaction_cost, slippage)
        self.max_dd_constraint = max_dd_constraint
        
        # Load data and train HMM once
        self._load_and_prepare()
        
    def _load_and_prepare(self):
        """Load data and train HMM."""
        print("Loading data and training HMM...")
        
        oil_df = pd.read_csv(self.oil_csv)
        date_col = "time" if "time" in oil_df.columns else "date"
        oil_df[date_col] = pd.to_datetime(oil_df[date_col])
        oil_df = oil_df.set_index(date_col).sort_index()
        
        use_features = ['oil_return', 'slope', 'vol_20d', 'return_20d']
        features_df, features_array, feature_names = prepare_oil_data(oil_df, use_features)
        features_df['price_return'] = features_df['oil_return']
        
        # Split data
        train_mask = features_df.index <= self.train_end
        self.train_features = features_array[train_mask]
        self.test_features = features_array[~train_mask]
        self.test_returns = features_df['price_return'].values[~train_mask]
        self.test_dates = features_df.index[~train_mask]
        
        # Train HMM once
        print("Training HMM on historical data...")
        self.hmm = OilStateGateHMM(n_states=2, random_state=42, n_iter=100)
        self.hmm.fit(self.train_features, feature_names=feature_names, verbose=True)
        
        print(f"Data loaded: {len(self.test_returns)} test days from {self.test_dates[0].date()} to {self.test_dates[-1].date()}")
    
    def evaluate_config(self, config: TradingConfig, verbose: bool = False) -> PerformanceMetrics:
        """Evaluate a trading configuration."""
        # Generate gate signal
        gate = self.hmm.get_continuous_gate(
            self.test_features, 
            sensitivity=config.sensitivity,
            smooth_window=config.smooth_window
        )
        
        # Generate positions
        positions = self.backtester.generate_positions(gate, config)
        
        # Calculate returns
        strategy_returns, _ = self.backtester.calculate_returns(positions, self.test_returns)
        
        # Calculate metrics
        metrics = self.backtester.calculate_metrics(strategy_returns, positions[:-1])
        
        if verbose:
            print(f"{config} -> Sharpe: {metrics.sharpe_ratio:.3f}, MaxDD: {metrics.max_drawdown*100:.1f}%")
        
        return metrics
    
    def grid_search(self, 
                   entry_thresholds: List[float] = None,
                   exit_thresholds: List[float] = None,
                   smooth_windows: List[int] = None,
                   sensitivities: List[float] = None) -> Tuple[TradingConfig, PerformanceMetrics, pd.DataFrame]:
        """Grid search over parameter space."""
        
        if entry_thresholds is None:
            entry_thresholds = [0.5, 0.6, 0.7, 0.8]
        if exit_thresholds is None:
            exit_thresholds = [0.2, 0.3, 0.4, 0.5]
        if smooth_windows is None:
            smooth_windows = [3, 5, 10, 15, 20]
        if sensitivities is None:
            sensitivities = [0.8, 1.0, 1.2]
        
        print("\n" + "="*70)
        print("GRID SEARCH OPTIMIZATION")
        print("="*70)
        print(f"Parameter space:")
        print(f"  Entry thresholds: {entry_thresholds}")
        print(f"  Exit thresholds: {exit_thresholds}")
        print(f"  Smooth windows: {smooth_windows}")
        print(f"  Sensitivities: {sensitivities}")
        
        total_configs = (len(entry_thresholds) * len(exit_thresholds) * 
                        len(smooth_windows) * len(sensitivities))
        print(f"\nTotal configurations to test: {total_configs}")
        print(f"Max DD constraint: {self.max_dd_constraint*100:.0f}%\n")
        
        results = []
        best_sharpe = -999
        best_config = None
        best_metrics = None
        
        valid_configs = 0
        
        for i, entry in enumerate(entry_thresholds):
            for exit in exit_thresholds:
                if exit >= entry:  # Skip invalid combinations
                    continue
                    
                for smooth in smooth_windows:
                    for sens in sensitivities:
                        config = TradingConfig(entry, exit, smooth, sens)
                        metrics = self.evaluate_config(config)
                        
                        # Check constraints
                        meets_constraints = metrics.meets_constraints(self.max_dd_constraint)
                        
                        result = {
                            'entry_threshold': entry,
                            'exit_threshold': exit,
                            'smooth_window': smooth,
                            'sensitivity': sens,
                            'meets_constraints': meets_constraints,
                            **metrics.to_dict()
                        }
                        results.append(result)
                        
                        # Track best
                        if meets_constraints:
                            valid_configs += 1
                            if metrics.sharpe_ratio > best_sharpe:
                                best_sharpe = metrics.sharpe_ratio
                                best_config = config
                                best_metrics = metrics
        
        results_df = pd.DataFrame(results)
        
        print("="*70)
        print("GRID SEARCH RESULTS")
        print("="*70)
        print(f"Configurations meeting constraints: {valid_configs}/{total_configs}")
        
        if best_config:
            print(f"\nBest configuration (meets {self.max_dd_constraint*100:.0f}% DD constraint):")
            print(f"  {best_config}")
            print(f"\nPerformance:")
            print(f"  Sharpe ratio: {best_metrics.sharpe_ratio:.3f}")
            print(f"  Annual return: {best_metrics.annual_return*100:.1f}%")
            print(f"  Annual volatility: {best_metrics.annual_vol*100:.1f}%")
            print(f"  Max drawdown: {best_metrics.max_drawdown*100:.1f}%")
            print(f"  Sortino ratio: {best_metrics.sortino_ratio:.3f}")
            print(f"  Calmar ratio: {best_metrics.calmar_ratio:.3f}")
            print(f"  Turnover: {best_metrics.turnover:.1f} trades/year")
            print(f"  Win rate: {best_metrics.win_rate*100:.1f}%")
        else:
            print(f"\n⚠️  No configuration met the {self.max_dd_constraint*100:.0f}% max drawdown constraint!")
            print(f"   Finding best unconstrained configuration instead...\n")
            
            # Find best by Sharpe without constraint
            best_idx_unconstrained = results_df['sharpe'].idxmax()
            best_row = results_df.loc[best_idx_unconstrained]
            
            best_config = TradingConfig(
                entry_threshold=best_row['entry_threshold'],
                exit_threshold=best_row['exit_threshold'],
                smooth_window=int(best_row['smooth_window']),
                sensitivity=best_row['sensitivity']
            )
            
            print(f"Best unconstrained configuration:")
            print(f"  {best_config}")
            print(f"\nPerformance:")
            print(f"  Sharpe ratio: {best_row['sharpe']:.3f}")
            print(f"  Annual return: {best_row['annual_return']*100:.1f}%")
            print(f"  Max drawdown: {best_row['max_dd']*100:.1f}% ⚠️  (exceeds {self.max_dd_constraint*100:.0f}% constraint)")
            print(f"  Sortino ratio: {best_row['sortino']:.3f}")
            print(f"  Turnover: {best_row['turnover']:.1f} trades/year")
            
            # Re-evaluate to get full metrics
            best_metrics = self.evaluate_config(best_config)
        
        return best_config, best_metrics, results_df
    
    def pso_optimize(self, n_particles: int = 30, n_iterations: int = 50,
                    w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> Tuple[TradingConfig, PerformanceMetrics, List]:
        """
        Particle Swarm Optimization for trading parameters.
        
        Args:
            n_particles: Number of particles
            n_iterations: Number of iterations
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        print("\n" + "="*70)
        print("PSO OPTIMIZATION")
        print("="*70)
        print(f"Particles: {n_particles}, Iterations: {n_iterations}")
        print(f"Inertia: {w}, Cognitive: {c1}, Social: {c2}")
        print(f"Max DD constraint: {self.max_dd_constraint*100:.0f}%\n")
        
        # Parameter bounds: [entry, exit, smooth, sensitivity]
        bounds_low = np.array([0.5, 0.2, 3, 0.5])
        bounds_high = np.array([0.9, 0.6, 30, 1.5])
        
        # Initialize particles
        particles = np.random.uniform(bounds_low, bounds_high, (n_particles, 4))
        velocities = np.random.uniform(-0.1, 0.1, (n_particles, 4))
        
        # Personal best
        p_best_positions = particles.copy()
        p_best_scores = np.full(n_particles, -999.0)
        
        # Global best
        g_best_position = None
        g_best_score = -999.0
        g_best_metrics = None
        
        history = []
        
        for iteration in range(n_iterations):
            print(f"Iteration {iteration+1}/{n_iterations}...", end=" ")
            
            valid_particles = 0
            iteration_best_sharpe = -999
            
            for i, particle in enumerate(particles):
                # Decode particle
                entry = np.clip(particle[0], bounds_low[0], bounds_high[0])
                exit = np.clip(particle[1], bounds_low[1], min(bounds_high[1], entry - 0.05))
                smooth = int(np.clip(particle[2], bounds_low[2], bounds_high[2]))
                sens = np.clip(particle[3], bounds_low[3], bounds_high[3])
                
                config = TradingConfig(entry, exit, smooth, sens)
                metrics = self.evaluate_config(config)
                
                # Fitness with constraint handling
                if metrics.meets_constraints(self.max_dd_constraint):
                    fitness = metrics.sharpe_ratio
                    valid_particles += 1
                else:
                    # Penalty for violating constraint
                    penalty = (metrics.max_drawdown - self.max_dd_constraint) * 10
                    fitness = metrics.sharpe_ratio - penalty
                
                # Update personal best
                if fitness > p_best_scores[i]:
                    p_best_scores[i] = fitness
                    p_best_positions[i] = particle.copy()
                
                # Update global best (only if constraints met)
                if metrics.meets_constraints(self.max_dd_constraint):
                    if metrics.sharpe_ratio > g_best_score:
                        g_best_score = metrics.sharpe_ratio
                        g_best_position = particle.copy()
                        g_best_metrics = metrics
                        iteration_best_sharpe = metrics.sharpe_ratio
            
            print(f"Valid: {valid_particles}/{n_particles}, Best Sharpe: {iteration_best_sharpe:.3f}")
            
            history.append({
                'iteration': iteration + 1,
                'best_sharpe': g_best_score,
                'valid_particles': valid_particles
            })
            
            # Update velocities and positions
            if g_best_position is not None:
                for i in range(n_particles):
                    r1, r2 = np.random.rand(2)
                    
                    velocities[i] = (w * velocities[i] + 
                                   c1 * r1 * (p_best_positions[i] - particles[i]) +
                                   c2 * r2 * (g_best_position - particles[i]))
                    
                    particles[i] = particles[i] + velocities[i]
                    
                    # Enforce bounds
                    particles[i] = np.clip(particles[i], bounds_low, bounds_high)
            else:
                # No valid solution yet, just use personal best
                for i in range(n_particles):
                    r1 = np.random.rand()
                    velocities[i] = w * velocities[i] + c1 * r1 * (p_best_positions[i] - particles[i])
                    particles[i] = particles[i] + velocities[i]
                    particles[i] = np.clip(particles[i], bounds_low, bounds_high)
        
        # Convert best to config
        if g_best_position is not None:
            best_config = TradingConfig(
                entry_threshold=g_best_position[0],
                exit_threshold=g_best_position[1],
                smooth_window=int(g_best_position[2]),
                sensitivity=g_best_position[3]
            )
            
            print("\n" + "="*70)
            print("PSO OPTIMIZATION RESULTS")
            print("="*70)
            print(f"Best configuration:")
            print(f"  {best_config}")
            print(f"\nPerformance:")
            print(f"  Sharpe ratio: {g_best_metrics.sharpe_ratio:.3f}")
            print(f"  Annual return: {g_best_metrics.annual_return*100:.1f}%")
            print(f"  Annual volatility: {g_best_metrics.annual_vol*100:.1f}%")
            print(f"  Max drawdown: {g_best_metrics.max_drawdown*100:.1f}%")
            print(f"  Sortino ratio: {g_best_metrics.sortino_ratio:.3f}")
            print(f"  Calmar ratio: {g_best_metrics.calmar_ratio:.3f}")
            print(f"  Turnover: {g_best_metrics.turnover:.1f} trades/year")
            print(f"  Win rate: {g_best_metrics.win_rate*100:.1f}%")
        else:
            print("\n⚠️  PSO did not find a configuration meeting constraints!")
            best_config = None
        
        return best_config, g_best_metrics, history
    
    def compare_configurations(self, configs: List[TradingConfig], 
                              labels: List[str] = None) -> pd.DataFrame:
        """Compare multiple configurations."""
        if labels is None:
            labels = [f"Config {i+1}" for i in range(len(configs))]
        
        results = []
        for config, label in zip(configs, labels):
            metrics = self.evaluate_config(config, verbose=True)
            result = {'config_name': label, **config.__dict__, **metrics.to_dict()}
            results.append(result)
        
        return pd.DataFrame(results)
    
    def save_results(self, config: TradingConfig, metrics: PerformanceMetrics,
                    results_df: pd.DataFrame = None, method: str = "grid",
                    output_dir: str = "output"):
        """Save optimization results."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best configuration
        config_dict = {
            'method': method,
            'timestamp': timestamp,
            'config': config.__dict__,
            'metrics': metrics.to_dict(),
            'max_dd_constraint': self.max_dd_constraint
        }
        
        config_file = os.path.join(output_dir, f"optimal_trading_params_{method}_{timestamp}.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"\nSaved optimal configuration to: {config_file}")
        
        # Save all results if available
        if results_df is not None:
            results_file = os.path.join(output_dir, f"param_optimization_results_{method}_{timestamp}.csv")
            results_df.to_csv(results_file, index=False)
            print(f"Saved detailed results to: {results_file}")


def run_parameter_optimization(oil_csv: str, 
                               train_end: str = "2018-12-31",
                               method: str = "both",
                               max_dd_constraint: float = 0.30,
                               transaction_cost: float = 0.001,
                               slippage: float = 0.0005):
    """
    Run trading parameter optimization.
    
    Args:
        method: "grid", "pso", or "both"
    """
    optimizer = TradingParameterOptimizer(
        oil_csv, train_end, transaction_cost, slippage, max_dd_constraint
    )
    
    results = {}
    
    if method in ["grid", "both"]:
        print("\n" + "="*70)
        print("RUNNING GRID SEARCH")
        print("="*70)
        best_config_grid, best_metrics_grid, results_df_grid = optimizer.grid_search()
        results['grid'] = (best_config_grid, best_metrics_grid, results_df_grid)
        
        if best_config_grid:
            optimizer.save_results(best_config_grid, best_metrics_grid, 
                                 results_df_grid, method="grid")
    
    if method in ["pso", "both"]:
        print("\n" + "="*70)
        print("RUNNING PSO OPTIMIZATION")
        print("="*70)
        best_config_pso, best_metrics_pso, history_pso = optimizer.pso_optimize()
        results['pso'] = (best_config_pso, best_metrics_pso, history_pso)
        
        if best_config_pso:
            optimizer.save_results(best_config_pso, best_metrics_pso, 
                                 method="pso")
    
    # Compare methods if both run
    if method == "both" and results.get('grid') and results.get('pso'):
        print("\n" + "="*70)
        print("COMPARISON: GRID SEARCH vs PSO")
        print("="*70)
        
        configs = [results['grid'][0], results['pso'][0]]
        labels = ['Grid Search', 'PSO']
        comparison_df = optimizer.compare_configurations(configs, labels)
        print(comparison_df.to_string(index=False))
    
    return optimizer, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize trading parameters using Grid Search or PSO"
    )
    parser.add_argument("--oil_csv", required=True, help="Path to oil prices CSV")
    parser.add_argument("--train_end", default="2018-12-31", help="Training end date")
    parser.add_argument("--method", choices=['grid', 'pso', 'both'], default='both',
                       help="Optimization method")
    parser.add_argument("--max_dd", type=float, default=0.30,
                       help="Maximum drawdown constraint (default: 30%)")
    parser.add_argument("--transaction_cost", type=float, default=0.001,
                       help="Transaction cost (default: 0.1%)")
    parser.add_argument("--slippage", type=float, default=0.0005,
                       help="Slippage (default: 0.05%)")
    args = parser.parse_args()
    
    run_parameter_optimization(
        args.oil_csv,
        args.train_end,
        args.method,
        args.max_dd,
        args.transaction_cost,
        args.slippage
    )