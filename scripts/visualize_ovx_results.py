"""
Visualize OVX Gate Results

Generates comprehensive visualizations for paper and analysis.
Includes Formula, HMM Original, and HMM Optimized gates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_csv_flexible(path):
    """Load CSV handling both 'time' and 'date' columns."""
    df = pd.read_csv(path)
    date_col = 'time' if 'time' in df.columns else 'date'
    if date_col not in df.columns:
        raise ValueError(f"No date column in {path}")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'date'})
    df = df.set_index('date').sort_index()
    return df


def plot_ovx_timeseries(
    ovx_df: pd.DataFrame,
    output_path: str
):
    """Plot OVX levels over time with regime shading."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot OVX
    ax.plot(ovx_df.index, ovx_df['ovx_close'], 
            label='OVX', linewidth=1.5, color='darkblue')
    
    # Add horizontal reference lines
    ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Baseline (30)')
    ax.axhline(y=40, color='orange', linestyle='--', alpha=0.5, label='Elevated (40)')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='High (50)')
    
    # Shade high volatility periods (OVX > 50)
    high_vol = ovx_df['ovx_close'] > 50
    for start, end in get_continuous_periods(high_vol):
        ax.axvspan(start, end, alpha=0.2, color='red')
    
    # Mark train/val/test splits
    ax.axvline(pd.Timestamp('2015-12-31'), color='black', 
               linestyle=':', alpha=0.7, linewidth=2)
    ax.axvline(pd.Timestamp('2018-12-31'), color='black', 
               linestyle=':', alpha=0.7, linewidth=2)
    
    # Add period labels
    ax.text(pd.Timestamp('2011-06-01'), ax.get_ylim()[1]*0.95, 'Train',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    ax.text(pd.Timestamp('2017-06-01'), ax.get_ylim()[1]*0.95, 'Val',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(pd.Timestamp('2022-06-01'), ax.get_ylim()[1]*0.95, 'Test',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('OVX Level', fontsize=12)
    ax.set_title('CBOE Oil Volatility Index (OVX) Time Series', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_gate_comparison(
    gates_formula: pd.DataFrame,
    gates_hmm: pd.DataFrame,
    gates_hmm_optimized: pd.DataFrame,
    output_path: str
):
    """Compare formula vs HMM original vs HMM optimized gates."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    
    # Formula gates
    ax = axes[0]
    ax.plot(gates_formula.index, gates_formula['gate_formula_no_scaleup'],
            label='Conservative (no scale-up)', linewidth=1.5, color='blue')
    ax.plot(gates_formula.index, gates_formula['gate_formula_with_scaleup'],
            label='Aggressive (with scale-up)', linewidth=1.5, color='red', alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Gate Value', fontsize=11)
    ax.set_title('Formula-Based OVX Gates', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.6)
    
    # HMM gates (original)
    ax = axes[1]
    ax.plot(gates_hmm.index, gates_hmm['gate_hmm_no_scaleup'],
            label='Conservative (no scale-up)', linewidth=1.5, color='blue')
    ax.plot(gates_hmm.index, gates_hmm['gate_hmm_with_scaleup'],
            label='Aggressive (with scale-up)', linewidth=1.5, color='red', alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Gate Value', fontsize=11)
    ax.set_title('HMM-Based OVX Gates (Original - Posterior Blending)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.6)
    
    # HMM gates (optimized)
    ax = axes[2]
    ax.plot(gates_hmm_optimized.index, gates_hmm_optimized['gate_hmm_no_scaleup'],
            label='Conservative (no scale-up)', linewidth=1.5, color='blue')
    ax.plot(gates_hmm_optimized.index, gates_hmm_optimized['gate_hmm_with_scaleup'],
            label='Aggressive (with scale-up)', linewidth=1.5, color='red', alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Gate Value', fontsize=11)
    ax.set_title('HMM-Based OVX Gates (Optimized - Hard Regime + Literature Powers)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def calculate_portfolio_returns(weights, prices):
    """Calculate portfolio returns."""
    asset_returns = prices.pct_change()
    common_dates = weights.index.intersection(asset_returns.index)
    weights = weights.loc[common_dates]
    asset_returns = asset_returns.loc[common_dates]
    portfolio_returns = (weights * asset_returns).sum(axis=1)
    return portfolio_returns


def plot_cumulative_returns(
    returns_dict: dict,
    output_path: str,
    test_start: str = '2019-01-01'
):
    """Plot cumulative returns for all strategies (test period only)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = {
        'Baseline': 'black',
        'Formula_Conservative': 'blue',
        'Formula_Aggressive': 'darkblue',
        'HMM_Conservative': 'green',
        'HMM_Aggressive': 'darkgreen',
        'HMM_Optimized_Conservative': 'purple',
        'HMM_Optimized_Aggressive': 'darkviolet'
    }
    
    for name, returns in returns_dict.items():
        # Filter to test period
        test_returns = returns.loc[test_start:]
        
        if len(test_returns) == 0:
            continue
        
        # Calculate cumulative returns
        cum_returns = (1 + test_returns).cumprod()
        
        # Plot
        ax.plot(cum_returns.index, cum_returns.values,
                label=name.replace('_', ' '),
                linewidth=2 if name == 'Baseline' else 1.5,
                color=colors.get(name, 'gray'),
                alpha=1.0 if name == 'Baseline' else 0.8)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title('Cumulative Returns - Test Period (2019-2025)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_performance_metrics(
    results_df: pd.DataFrame,
    output_path: str
):
    """Bar chart comparing key metrics on test period."""
    # Filter to test period
    test_results = results_df[results_df['period'] == 'Test'].copy()
    
    if len(test_results) == 0:
        print("  ✗ No test period results found")
        return
    
    # Prepare data
    test_results['strategy'] = test_results['strategy'].str.replace('_', ' ')
    
    # Create subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sharpe Ratio
    ax = axes[0, 0]
    test_results.plot(x='strategy', y='sharpe', kind='bar', ax=ax, legend=False, color='steelblue')
    ax.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=10)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Annual Return
    ax = axes[0, 1]
    test_results['annual_return_pct'] = test_results['annual_return'] * 100
    test_results.plot(x='strategy', y='annual_return_pct', kind='bar', ax=ax, legend=False, color='green')
    ax.set_title('Annual Return', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Return (%)', fontsize=10)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Max Drawdown
    ax = axes[1, 0]
    test_results['max_dd_pct'] = test_results['max_drawdown'] * 100
    test_results.plot(x='strategy', y='max_dd_pct', kind='bar', ax=ax, legend=False, color='red')
    ax.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Drawdown (%)', fontsize=10)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Sortino Ratio
    ax = axes[1, 1]
    test_results.plot(x='strategy', y='sortino', kind='bar', ax=ax, legend=False, color='purple')
    ax.set_title('Sortino Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sortino Ratio', fontsize=10)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Performance Metrics - Test Period (2019-2025)', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_drawdowns(
    returns_dict: dict,
    output_path: str,
    test_start: str = '2019-01-01'
):
    """Plot underwater (drawdown) chart for test period."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = {
        'Baseline': 'black',
        'Formula_Conservative': 'blue',
        'Formula_Aggressive': 'darkblue',
        'HMM_Conservative': 'green',
        'HMM_Aggressive': 'darkgreen',
        'HMM_Optimized_Conservative': 'purple',
        'HMM_Optimized_Aggressive': 'darkviolet'
    }
    
    for name, returns in returns_dict.items():
        # Filter to test period
        test_returns = returns.loc[test_start:]
        
        if len(test_returns) == 0:
            continue
        
        # Calculate drawdown
        cum_returns = (1 + test_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns / running_max - 1) * 100
        
        # Plot
        ax.plot(drawdown.index, drawdown.values,
                label=name.replace('_', ' '),
                linewidth=2 if name == 'Baseline' else 1.5,
                color=colors.get(name, 'gray'),
                alpha=1.0 if name == 'Baseline' else 0.7)
    
    ax.fill_between(ax.get_xlim(), 0, ax.get_ylim()[0], alpha=0.1, color='red')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Drawdown Comparison - Test Period (2019-2025)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def get_continuous_periods(boolean_series):
    """Get start and end dates of continuous True periods."""
    periods = []
    in_period = False
    start = None
    
    for date, value in boolean_series.items():
        if value and not in_period:
            start = date
            in_period = True
        elif not value and in_period:
            periods.append((start, date))
            in_period = False
    
    if in_period:
        periods.append((start, boolean_series.index[-1]))
    
    return periods


def main(
    ovx_csv: str = "data/ovx_daily.csv",
    gates_formula_csv: str = "output/oil_signals/gates_ovx_formula.csv",
    gates_hmm_csv: str = "output/oil_signals/gates_ovx_hmm.csv",
    gates_hmm_optimized_csv: str = "output/oil_signals/gates_ovx_hmm_stable.csv",
    backtest_csv: str = "output/backtest_ovx_comparison.csv",
    prices_csv: str = "data/prices_daily.csv",
    weights_baseline_csv: str = "output/signals/weights_after_caps.csv",
    output_dir: str = "output/figures"
):
    """Generate all visualizations."""
    print("="*70)
    print("GENERATING OVX GATE VISUALIZATIONS")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. OVX Time Series
    print("\n1. Plotting OVX time series...")
    try:
        ovx_df = load_csv_flexible(ovx_csv)
        plot_ovx_timeseries(ovx_df, output_path / 'ovx_timeseries.png')
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 2. Gate Comparison
    print("\n2. Plotting gate comparison (including HMM optimized)...")
    try:
        gates_formula = load_csv_flexible(gates_formula_csv)
        gates_hmm = load_csv_flexible(gates_hmm_csv)
        gates_hmm_optimized = load_csv_flexible(gates_hmm_optimized_csv)
        plot_gate_comparison(gates_formula, gates_hmm, gates_hmm_optimized,
                           output_path / 'gates_comparison.png')
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 3-5. Performance plots (need returns)
    print("\n3-5. Plotting performance metrics...")
    try:
        # Load prices
        prices_df = load_csv_flexible(prices_csv)
        
        # Calculate returns for each strategy
        strategies = {
            'Baseline': weights_baseline_csv,
            'Formula_Conservative': 'output/signals/weights_with_ovx_formula_no_scaleup.csv',
            'Formula_Aggressive': 'output/signals/weights_with_ovx_formula_with_scaleup.csv',
            'HMM_Conservative': 'output/signals/weights_with_ovx_hmm_no_scaleup.csv',
            'HMM_Aggressive': 'output/signals/weights_with_ovx_hmm_with_scaleup.csv',
            'HMM_Optimized_Conservative': 'output/signals/weights_with_ovx_hmm_optimized_no_scaleup.csv',
            'HMM_Optimized_Aggressive': 'output/signals/weights_with_ovx_hmm_optimized_with_scaleup.csv'
        }
        
        returns_dict = {}
        for name, weights_file in strategies.items():
            try:
                weights = load_csv_flexible(weights_file)
                portfolio_returns = calculate_portfolio_returns(weights, prices_df)
                returns_dict[name] = portfolio_returns
            except:
                pass
        
        if len(returns_dict) > 0:
            plot_cumulative_returns(returns_dict, output_path / 'cumulative_returns.png')
            plot_drawdowns(returns_dict, output_path / 'drawdowns.png')
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 4. Performance metrics bar chart
    print("\n4. Plotting performance metrics...")
    try:
        results_df = pd.read_csv(backtest_csv)
        plot_performance_metrics(results_df, output_path / 'performance_metrics.png')
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n" + "="*70)
    print(f"✓ ALL VISUALIZATIONS SAVED TO: {output_dir}/")
    print("="*70)
    print("\nGenerated files:")
    print("  - ovx_timeseries.png")
    print("  - gates_comparison.png (includes HMM optimized)")
    print("  - cumulative_returns.png (includes HMM optimized)")
    print("  - drawdowns.png (includes HMM optimized)")
    print("  - performance_metrics.png (includes HMM optimized)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize OVX gate results")
    parser.add_argument('--ovx_csv', default='data/ovx_daily.csv')
    parser.add_argument('--gates_formula_csv', default='output/oil_signals/gates_ovx_formula.csv')
    parser.add_argument('--gates_hmm_csv', default='output/oil_signals/gates_ovx_hmm.csv')
    parser.add_argument('--gates_hmm_optimized_csv', default='output/oil_signals/gates_ovx_hmm_stable.csv')
    parser.add_argument('--backtest_csv', default='output/backtest_ovx_comparison.csv')
    parser.add_argument('--prices_csv', default='data/prices_daily.csv')
    parser.add_argument('--weights_baseline_csv', default='output/signals/weights_after_caps.csv')
    parser.add_argument('--output_dir', default='output/figures')
    
    args = parser.parse_args()
    main(**vars(args))