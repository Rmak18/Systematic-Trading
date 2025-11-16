"""
Backtest OVX Gate Strategies

Compares strategies:
1. Baseline (no OVX)
2. Formula Conservative
3. Formula Aggressive  
4. HMM Conservative (Original)
5. HMM Aggressive (Original)
6. HMM Proper Conservative (New Method)
7. HMM Proper Aggressive (New Method)

With proper train/val/test evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, Tuple
from scipy import stats


def load_data_with_flexible_date_col(csv_path, parse_dates=True):
    """Load CSV with flexible date column name (handles 'time' or 'date')."""
    df = pd.read_csv(csv_path)
    
    # Find date column
    date_col = None
    if 'time' in df.columns:
        date_col = 'time'
    elif 'date' in df.columns:
        date_col = 'date'
    else:
        raise ValueError(f"No date column found in {csv_path}. Expected 'time' or 'date'. Found: {list(df.columns)}")
    
    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Rename to 'date' for consistency
    df = df.rename(columns={date_col: 'date'})
    df = df.set_index('date').sort_index()
    
    return df


def calculate_portfolio_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.Series:
    """
    Calculate portfolio returns from weights and prices.
    
    Parameters:
    -----------
    weights : pd.DataFrame
        Daily portfolio weights (date × assets)
    prices : pd.DataFrame
        Daily asset prices (date × assets)
    
    Returns:
    --------
    returns : pd.Series
        Daily portfolio returns
    """
    # Calculate asset returns
    asset_returns = prices.pct_change()
    
    # Align weights and returns
    common_dates = weights.index.intersection(asset_returns.index)
    weights = weights.loc[common_dates]
    asset_returns = asset_returns.loc[common_dates]
    
    # Calculate portfolio return each day
    portfolio_returns = (weights * asset_returns).sum(axis=1)
    
    return portfolio_returns


def calculate_metrics(returns: pd.Series, period_name: str = "") -> Dict:
    """
    Calculate performance metrics for a return series.
    
    Returns dict with Sharpe, returns, volatility, drawdown, etc.
    """
    # Remove NaN
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {
            'period': period_name,
            'days': 0,
            'sharpe': np.nan,
            'annual_return': np.nan,
            'annual_vol': np.nan,
            'max_drawdown': np.nan,
            'sortino': np.nan,
            'calmar': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan
        }
    
    # Annual factor
    annual_factor = np.sqrt(252)
    
    # Basic stats
    mean_daily = returns.mean()
    std_daily = returns.std()
    
    # Annualized
    annual_return = mean_daily * 252
    annual_vol = std_daily * annual_factor
    
    # Sharpe
    sharpe = (annual_return / annual_vol) if annual_vol > 0 else 0
    
    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative / running_max - 1)
    max_drawdown = drawdown.min()
    
    # Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else std_daily
    sortino = (annual_return / (downside_std * annual_factor)) if downside_std > 0 else 0
    
    # Calmar
    calmar = (annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0
    
    # Higher moments
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    return {
        'period': period_name,
        'days': len(returns),
        'sharpe': sharpe,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'max_drawdown': max_drawdown,
        'sortino': sortino,
        'calmar': calmar,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'total_return': (1 + returns).prod() - 1
    }


def calculate_turnover(weights: pd.DataFrame) -> float:
    """
    Calculate annualized turnover.
    
    Turnover = sum of absolute weight changes / days * 252
    """
    weight_changes = weights.diff().abs().sum(axis=1)
    daily_turnover = weight_changes.mean()
    annual_turnover = daily_turnover * 252
    
    return annual_turnover


def compare_strategies(
    baseline_returns: pd.Series,
    treatment_returns: pd.Series,
    strategy_name: str
):
    """
    Statistical comparison between baseline and treatment.
    
    Returns:
    - T-test for mean difference
    - Sharpe ratio difference with confidence interval
    """
    # T-test on returns
    t_stat, p_value = stats.ttest_ind(treatment_returns, baseline_returns)
    
    # Bootstrap Sharpe difference
    n_bootstrap = 1000
    sharpe_diffs = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        baseline_sample = baseline_returns.sample(len(baseline_returns), replace=True)
        treatment_sample = treatment_returns.sample(len(treatment_returns), replace=True)
        
        # Calculate Sharpes
        baseline_sharpe = (baseline_sample.mean() / baseline_sample.std()) * np.sqrt(252)
        treatment_sharpe = (treatment_sample.mean() / treatment_sample.std()) * np.sqrt(252)
        
        sharpe_diffs.append(treatment_sharpe - baseline_sharpe)
    
    sharpe_diff_ci = np.percentile(sharpe_diffs, [2.5, 97.5])
    
    return {
        'strategy': strategy_name,
        't_stat': t_stat,
        'p_value': p_value,
        'sharpe_diff_ci_lower': sharpe_diff_ci[0],
        'sharpe_diff_ci_upper': sharpe_diff_ci[1]
    }


def main(
    prices_csv: str = "data/prices_daily.csv",
    baseline_weights_csv: str = "output/signals/weights_after_caps.csv",
    output_csv: str = "output/backtest_ovx_comparison.csv"
):
    """
    Run comprehensive backtest comparison.
    """
    print("="*70)
    print("OVX GATE BACKTEST COMPARISON")
    print("="*70)
    
    # Load prices
    print(f"\nLoading price data from: {prices_csv}")
    try:
        prices_df = load_data_with_flexible_date_col(prices_csv)
        print(f"✓ Loaded {len(prices_df)} days of price data")
        print(f"  Date range: {prices_df.index.min().date()} to {prices_df.index.max().date()}")
        print(f"  Assets: {list(prices_df.columns)}")
    except Exception as e:
        print(f"✗ Error loading prices: {e}")
        return
    
    # Define strategies to test
    strategies = {
        'Baseline': baseline_weights_csv,
        'Formula_Conservative': 'output/signals/weights_with_ovx_formula_no_scaleup.csv',
        'Formula_Aggressive': 'output/signals/weights_with_ovx_formula_with_scaleup.csv',
        'HMM_Conservative': 'output/signals/weights_with_ovx_hmm_no_scaleup.csv',
        'HMM_Aggressive': 'output/signals/weights_with_ovx_hmm_with_scaleup.csv',
        'HMM_Proper_Conservative': 'output/signals/weights_with_ovx_hmm_proper_no_scaleup.csv',
        'HMM_Proper_Aggressive': 'output/signals/weights_with_ovx_hmm_proper_with_scaleup.csv'
    }
    
    # Calculate returns for each strategy
    strategy_returns = {}
    strategy_weights = {}
    
    print("\n" + "="*70)
    print("CALCULATING STRATEGY RETURNS")
    print("="*70)
    
    for name, weights_file in strategies.items():
        try:
            print(f"\n{name}:")
            weights = load_data_with_flexible_date_col(weights_file)
            
            # Calculate returns
            returns = calculate_portfolio_returns(weights, prices_df)
            
            strategy_returns[name] = returns
            strategy_weights[name] = weights
            
            print(f"  ✓ Calculated returns for {len(returns)} days")
            
        except FileNotFoundError:
            print(f"  ✗ File not found: {weights_file}")
            print(f"  Skipping {name}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  Skipping {name}")
    
    if len(strategy_returns) == 0:
        print("\n✗ No strategies could be loaded!")
        return
    
    # Define evaluation periods
    periods = {
        'Full': ('2007-01-01', '2025-12-31'),
        'Train': ('2007-01-01', '2015-12-31'),
        'Validation': ('2016-01-01', '2018-12-31'),
        'Test': ('2019-01-01', '2025-12-31')
    }
    
    # Calculate metrics for each strategy × period
    all_results = []
    
    print("\n" + "="*70)
    print("PERFORMANCE METRICS BY PERIOD")
    print("="*70)
    
    for period_name, (start, end) in periods.items():
        print(f"\n{period_name} Period ({start} to {end}):")
        print("-" * 70)
        
        for strategy_name, returns in strategy_returns.items():
            # Filter period
            period_returns = returns.loc[start:end]
            
            if len(period_returns) == 0:
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(period_returns, period_name)
            metrics['strategy'] = strategy_name
            
            all_results.append(metrics)
            
            # Print summary
            print(f"  {strategy_name:25s} | "
                  f"Sharpe: {metrics['sharpe']:6.3f} | "
                  f"Return: {metrics['annual_return']*100:6.2f}% | "
                  f"MaxDD: {metrics['max_drawdown']*100:6.2f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Calculate turnover
    print("\n" + "="*70)
    print("TURNOVER ANALYSIS")
    print("="*70)
    
    turnover_results = []
    for name, weights in strategy_weights.items():
        turnover = calculate_turnover(weights)
        turnover_results.append({
            'strategy': name,
            'annual_turnover': turnover
        })
        print(f"  {name:25s} | Turnover: {turnover:.2f}")
    
    # Statistical tests (on test period)
    print("\n" + "="*70)
    print("STATISTICAL TESTS (Test Period)")
    print("="*70)
    
    if 'Baseline' in strategy_returns:
        baseline_test = strategy_returns['Baseline'].loc['2019-01-01':'2025-12-31']
        
        test_results = []
        for name, returns in strategy_returns.items():
            if name == 'Baseline':
                continue
            
            treatment_test = returns.loc['2019-01-01':'2025-12-31']
            
            if len(treatment_test) > 0 and len(baseline_test) > 0:
                test_result = compare_strategies(
                    baseline_test,
                    treatment_test,
                    name
                )
                test_results.append(test_result)
                
                print(f"\n{name} vs Baseline:")
                print(f"  P-value: {test_result['p_value']:.4f}")
                print(f"  Sharpe diff 95% CI: [{test_result['sharpe_diff_ci_lower']:.3f}, "
                      f"{test_result['sharpe_diff_ci_upper']:.3f}]")
    
    # Save results
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print(f"✓ Results saved to: {output_csv}")
    
    # Print key comparison: HMM Original vs HMM Proper
    if 'HMM_Conservative' in strategy_returns and 'HMM_Proper_Conservative' in strategy_returns:
        print("\n" + "="*70)
        print("HMM COMPARISON: Original vs Proper (Test Period)")
        print("="*70)
        
        test_results = results_df[results_df['period'] == 'Test']
        
        hmm_orig = test_results[test_results['strategy'] == 'HMM_Conservative'].iloc[0]
        hmm_proper = test_results[test_results['strategy'] == 'HMM_Proper_Conservative'].iloc[0]
        
        print("\nOriginal HMM (Posterior Blending):")
        print(f"  Sharpe: {hmm_orig['sharpe']:.3f}")
        print(f"  Return: {hmm_orig['annual_return']*100:.2f}%")
        print(f"  Max DD: {hmm_orig['max_drawdown']*100:.2f}%")
        
        print("\nProper HMM (Constrained + Regularized):")
        print(f"  Sharpe: {hmm_proper['sharpe']:.3f}")
        print(f"  Return: {hmm_proper['annual_return']*100:.2f}%")
        print(f"  Max DD: {hmm_proper['max_drawdown']*100:.2f}%")
        
        sharpe_improvement = ((hmm_proper['sharpe'] - hmm_orig['sharpe']) / hmm_orig['sharpe']) * 100
        print(f"\nImprovement: {sharpe_improvement:+.1f}% Sharpe")
    
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest OVX gate strategies"
    )
    parser.add_argument(
        '--prices_csv',
        default='data/prices_daily.csv',
        help='Path to prices CSV'
    )
    parser.add_argument(
        '--baseline_weights_csv',
        default='output/signals/weights_after_caps.csv',
        help='Path to baseline weights'
    )
    parser.add_argument(
        '--output_csv',
        default='output/backtest_ovx_comparison.csv',
        help='Path for output results'
    )
    
    args = parser.parse_args()
    
    main(
        prices_csv=args.prices_csv,
        baseline_weights_csv=args.baseline_weights_csv,
        output_csv=args.output_csv
    )