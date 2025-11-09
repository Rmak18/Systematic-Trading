import argparse
import json
import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm


class OilStateGateHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 100, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.bear_state = None
        self.bull_state = None
        self.feature_names = None
        
    def fit(self, features: np.ndarray, feature_names: List[str] = None, verbose: bool = False):
        """
        Fit HMM on multiple features.
        
        Args:
            features: (n_samples, n_features) array
            feature_names: list of feature names for reporting
            verbose: print training details
        """
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]
        
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=verbose
        )
        
        self.model.fit(features)
        
        # Identify bear/bull states based on mean return (first feature)
        means = self.model.means_[:, 0]
        self.bear_state = np.argmin(means)
        self.bull_state = np.argmax(means)
        
        if verbose:
            print(f"\nHMM Converged: {self.model.monitor_.converged}")
            print(f"Final log-likelihood: {self.model.score(features):.2f}")
            self._print_state_characteristics()
    
    def _print_state_characteristics(self):
        """Print detailed state characteristics."""
        print("\n" + "="*60)
        print("STATE CHARACTERISTICS")
        print("="*60)
        
        for state_id in range(self.n_states):
            state_name = 'Bear' if state_id == self.bear_state else 'Bull'
            print(f"\nState {state_id} ({state_name}):")
            print(f"  Means: {dict(zip(self.feature_names, self.model.means_[state_id]))}")
            
            # Get standard deviations from covariance matrix
            stds = np.sqrt(np.diag(self.model.covars_[state_id]))
            print(f"  Std Devs: {dict(zip(self.feature_names, stds))}")
        
        print(f"\n{'Transition Matrix:':20s}")
        print(f"{'':20s} {'→ Bear':>10s} {'→ Bull':>10s}")
        print(f"{'Bear →':20s} {self.model.transmat_[self.bear_state, self.bear_state]:10.3f} {self.model.transmat_[self.bear_state, self.bull_state]:10.3f}")
        print(f"{'Bull →':20s} {self.model.transmat_[self.bull_state, self.bear_state]:10.3f} {self.model.transmat_[self.bull_state, self.bull_state]:10.3f}")
        
        print(f"\nRegime Persistence:")
        print(f"  Bear market stays bear: {self.model.transmat_[self.bear_state, self.bear_state]:.1%}")
        print(f"  Bull market stays bull: {self.model.transmat_[self.bull_state, self.bull_state]:.1%}")
    
    def predict_states(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        return self.model.predict(features)
    
    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        return self.model.predict_proba(features)
    
    def get_continuous_gate(self, features: np.ndarray, sensitivity: float = 1.0, 
                          smooth_window: Optional[int] = None) -> np.ndarray:
        """
        Get continuous gate value (0=bear, 1=bull).
        
        Args:
            features: input features
            sensitivity: >1 makes gate more aggressive, <1 more conservative
            smooth_window: if provided, apply rolling average smoothing
        """
        posteriors = self.predict_probabilities(features)
        prob_bull = posteriors[:, self.bull_state]
        
        if sensitivity != 1.0:
            prob_bull = self._apply_sensitivity(prob_bull, sensitivity)
        
        if smooth_window:
            prob_bull = pd.Series(prob_bull).rolling(
                smooth_window, center=True, min_periods=1
            ).mean().values
        
        return prob_bull
    
    def _apply_sensitivity(self, prob_bull: np.ndarray, sensitivity: float) -> np.ndarray:
        """Apply sensitivity adjustment via logit transformation."""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        prob_bull = np.clip(prob_bull, eps, 1 - eps)
        
        logit = np.log(prob_bull / (1 - prob_bull))
        logit_adjusted = logit * sensitivity
        prob_adjusted = 1 / (1 + np.exp(-logit_adjusted))
        
        return np.clip(prob_adjusted, 0.0, 1.0)
    
    def to_dict(self):
        return {
            "n_states": self.n_states,
            "bear_state": int(self.bear_state),
            "bull_state": int(self.bull_state),
            "feature_names": self.feature_names,
            "means": self.model.means_.tolist(),
            "covars": [cov.tolist() for cov in self.model.covars_],
            "transmat": self.model.transmat_.tolist(),
            "startprob": self.model.startprob_.tolist()
        }


def prepare_oil_data(oil_df: pd.DataFrame, use_features: List[str] = None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Prepare oil data with multiple features for HMM.
    
    Returns:
        features_df: DataFrame with all features
        features_array: numpy array of selected features for HMM
        feature_names: list of feature names used
    """
    features = pd.DataFrame(index=oil_df.index)
    
    features['near_month'] = oil_df['near_month']
    features['next_month'] = oil_df['next_month']
    features['oil_return'] = np.log(oil_df['near_month'] / oil_df['near_month'].shift(1))
    features['slope'] = (oil_df['next_month'] - oil_df['near_month']) / oil_df['near_month']
    
    # Rolling statistics
    for window in [5, 10, 20]:
        features[f'return_{window}d'] = features['oil_return'].rolling(window).sum()
        features[f'vol_{window}d'] = features['oil_return'].rolling(window).std()
    
    features = features.dropna()
    
    # Select features for HMM
    if use_features is None:
        use_features = ['oil_return', 'slope', 'vol_20d', 'return_20d']
    
    features_array = features[use_features].values
    
    return features, features_array, use_features


def train_test_split_by_date(df: pd.DataFrame, train_end: str = "2018-12-31"):
    train_mask = df.index <= train_end
    train_df = df[train_mask]
    test_df = df[~train_mask]
    return train_df, test_df


def calculate_regime_metrics(df: pd.DataFrame, regime_col: str = 'Regime', 
                            return_col: str = 'oil_return') -> pd.DataFrame:
    """Calculate trading metrics for each regime."""
    metrics = []
    
    for state_id in df[regime_col].unique():
        mask = df[regime_col] == state_id
        returns = df.loc[mask, return_col]
        
        if len(returns) == 0:
            continue
        
        cumulative_return = (1 + returns).prod() - 1
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics.append({
            'State': state_id,
            'Days': len(returns),
            'Pct_Days': 100 * len(returns) / len(df),
            'Mean_Daily_Return': returns.mean(),
            'Annualized_Return': annualized_return,
            'Annualized_Vol': annualized_vol,
            'Sharpe': sharpe,
            'Max_Drawdown': max_drawdown,
            'Cumulative_Return': cumulative_return
        })
    
    return pd.DataFrame(metrics)


def run_sensitivity_analysis(hmm_model: OilStateGateHMM, 
                            features_array: np.ndarray,
                            sensitivity_levels: list) -> pd.DataFrame:
    results = []
    
    for sensitivity in sensitivity_levels:
        gate = hmm_model.get_continuous_gate(features_array, sensitivity=sensitivity)
        
        mean_gate = gate.mean()
        median_gate = np.median(gate)
        pct_low = 100 * (gate < 0.3).sum() / len(gate)
        pct_high = 100 * (gate > 0.7).sum() / len(gate)
        pct_medium = 100 - pct_low - pct_high
        
        results.append({
            'sensitivity': sensitivity,
            'mean_gate': mean_gate,
            'median_gate': median_gate,
            'pct_low_exposure': pct_low,
            'pct_medium_exposure': pct_medium,
            'pct_high_exposure': pct_high
        })
    
    return pd.DataFrame(results)


def plot_hmm_results(oil_df: pd.DataFrame, 
                     results_df: pd.DataFrame,
                     train_end: str,
                     smooth_window: Optional[int] = None,
                     out_png: Optional[str] = None):
    
    df = oil_df.join(results_df[['Regime', 'gate_continuous', 'gate_smoothed', 'prob_bull']], how='inner')
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    ax1, ax2, ax3, ax4 = axes
    
    # Panel 1: Oil prices with regime background
    ax1.plot(df.index, df['near_month'], linewidth=1.5, color='black', label='Near Month Oil Price')
    ax1.axvline(pd.Timestamp(train_end), color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    
    regimes = df['Regime'].values
    for state_id in np.unique(regimes):
        mask = (regimes == state_id)
        if mask.sum() > 0:
            color = 'red' if state_id == 0 else 'green'
            ax1.fill_between(df.index, 0, df['near_month'].max() * 1.1,
                           where=mask, alpha=0.15, color=color)
    
    ax1.set_ylabel('Oil Price ($/barrel)')
    ax1.set_title('Oil Prices with HMM-Identified Regimes (Bear=Red, Bull=Green)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Futures curve slope
    slope = (df['next_month'] - df['near_month']) / df['near_month']
    ax2.plot(df.index, slope, linewidth=1, color='blue', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(pd.Timestamp(train_end), color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel('Slope (Contango/Backwardation)')
    ax2.set_title('Oil Futures Curve Slope')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Posterior probability
    ax3.plot(df.index, df['prob_bull'], linewidth=1.5, color='blue', alpha=0.7, label='P(Bull State)')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(pd.Timestamp(train_end), color='red', linestyle='--', linewidth=2)
    ax3.fill_between(df.index, 0, df['prob_bull'], alpha=0.2, color='blue')
    ax3.set_ylabel('Probability')
    ax3.set_title('Probability of Bull State (HMM Posterior)')
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Continuous gate (raw and smoothed)
    ax4.plot(df.index, df['gate_continuous'], linewidth=1, color='gray', 
             alpha=0.5, label='Raw Gate')
    ax4.plot(df.index, df['gate_smoothed'], linewidth=2, color='black', 
             label=f'Smoothed Gate (window={smooth_window})')
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(pd.Timestamp(train_end), color='red', linestyle='--', linewidth=2)
    ax4.fill_between(df.index, 0, df['gate_smoothed'], alpha=0.3, color='green')
    ax4.set_ylabel('Gate Value (0-1)')
    ax4.set_xlabel('Date')
    ax4.set_title(f'Oil State Gate (Continuous, Sensitivity=1.0)')
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=150)
    return fig


def run_oil_hmm_gate(oil_csv: str,
                     train_end: str = "2018-12-31",
                     out_gates_csv: str = "oil_gates_hmm.csv",
                     out_sensitivity_csv: str = "oil_sensitivity_analysis.csv",
                     out_hmm_json: str = "oil_hmm_params.json",
                     out_metrics_csv: str = "oil_regime_metrics.csv",
                     out_plot_png: Optional[str] = "oil_hmm_plot.png",
                     n_states: int = 2,
                     n_iter: int = 100,
                     use_features: List[str] = None,
                     smooth_window: int = 5,
                     sensitivity_levels: list = [0.5, 0.75, 1.0, 1.25, 1.5]):
    
    # Load data
    oil_df = pd.read_csv(oil_csv)
    date_col = "time" if "time" in oil_df.columns else "date"
    oil_df[date_col] = pd.to_datetime(oil_df[date_col])
    oil_df = oil_df.set_index(date_col).sort_index()
    
    print(f"Loaded oil data: {len(oil_df)} days")
    print(f"Date range: {oil_df.index.min().date()} to {oil_df.index.max().date()}")
    
    # Prepare features
    if use_features is None:
        use_features = ['oil_return', 'slope', 'vol_20d', 'return_20d']
    
    print(f"\nUsing features for HMM: {use_features}")
    
    features_df, features_array, feature_names = prepare_oil_data(oil_df, use_features)
    
    # Split train/test
    train_mask = features_df.index <= train_end
    train_features = features_array[train_mask]
    
    train_dates = features_df.index[train_mask]
    test_dates = features_df.index[~train_mask]
    
    print(f"\nTrain period: {train_dates.min().date()} to {train_dates.max().date()} ({len(train_features)} days)")
    print(f"Test period:  {test_dates.min().date()} to {test_dates.max().date()} ({len(test_dates)} days)")
    
    # Train HMM
    print("\n" + "="*60)
    print("TRAINING HMM ON MULTI-FEATURE DATA")
    print("="*60)
    hmm_model = OilStateGateHMM(n_states=n_states, n_iter=n_iter)
    hmm_model.fit(train_features, feature_names=feature_names, verbose=True)
    
    # Predict on full dataset
    print("\nPredicting states on full dataset...")
    states = hmm_model.predict_states(features_array)
    posteriors = hmm_model.predict_probabilities(features_array)
    gate_continuous = hmm_model.get_continuous_gate(features_array, sensitivity=1.0)
    gate_smoothed = hmm_model.get_continuous_gate(features_array, sensitivity=1.0, 
                                                   smooth_window=smooth_window)
    
    results = pd.DataFrame({
        'Regime': states,
        'prob_bear': posteriors[:, hmm_model.bear_state],
        'prob_bull': posteriors[:, hmm_model.bull_state],
        'gate_continuous': gate_continuous,
        'gate_smoothed': gate_smoothed,
        'oil_return': features_df['oil_return'].values,
        'slope': features_df['slope'].values
    }, index=features_df.index)
    
    # Calculate regime metrics
    print("\n" + "="*60)
    print("REGIME METRICS (Full Dataset)")
    print("="*60)
    full_metrics = calculate_regime_metrics(results)
    for _, row in full_metrics.iterrows():
        state_name = 'Bear' if row['State'] == hmm_model.bear_state else 'Bull'
        print(f"\nState {int(row['State'])} ({state_name}):")
        print(f"  Days: {int(row['Days'])} ({row['Pct_Days']:.1f}%)")
        print(f"  Mean daily return: {row['Mean_Daily_Return']:+.4%}")
        print(f"  Annualized return: {row['Annualized_Return']:+.2%}")
        print(f"  Annualized vol: {row['Annualized_Vol']:.2%}")
        print(f"  Sharpe ratio: {row['Sharpe']:.2f}")
        print(f"  Max drawdown: {row['Max_Drawdown']:.2%}")
        print(f"  Cumulative return: {row['Cumulative_Return']:+.2%}")
    
    # Test period metrics
    print("\n" + "="*60)
    print("REGIME METRICS (Test Period 2019-2025)")
    print("="*60)
    test_results = results[results.index > train_end]
    test_metrics = calculate_regime_metrics(test_results)
    for _, row in test_metrics.iterrows():
        state_name = 'Bear' if row['State'] == hmm_model.bear_state else 'Bull'
        print(f"\nState {int(row['State'])} ({state_name}):")
        print(f"  Days: {int(row['Days'])} ({row['Pct_Days']:.1f}%)")
        print(f"  Mean daily return: {row['Mean_Daily_Return']:+.4%}")
        print(f"  Annualized return: {row['Annualized_Return']:+.2%}")
        print(f"  Annualized vol: {row['Annualized_Vol']:.2%}")
        print(f"  Sharpe ratio: {row['Sharpe']:.2f}")
        print(f"  Max drawdown: {row['Max_Drawdown']:.2%}")
        print(f"  Cumulative return: {row['Cumulative_Return']:+.2%}")
    
    print(f"\nContinuous gate statistics (test period):")
    print(f"  Mean: {test_results['gate_smoothed'].mean():.3f}")
    print(f"  Median: {test_results['gate_smoothed'].median():.3f}")
    print(f"  Days with gate < 0.3: {(test_results['gate_smoothed'] < 0.3).sum()} ({100*(test_results['gate_smoothed'] < 0.3).sum()/len(test_results):.1f}%)")
    print(f"  Days with gate > 0.7: {(test_results['gate_smoothed'] > 0.7).sum()} ({100*(test_results['gate_smoothed'] > 0.7).sum()/len(test_results):.1f}%)")
    
    # Sensitivity analysis
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS (Test Period)")
    print("="*60)
    test_features = features_array[~train_mask]
    sensitivity_df = run_sensitivity_analysis(hmm_model, test_features, sensitivity_levels)
    print(sensitivity_df.to_string(index=False))
    
    # Save outputs
    results_out = results[['Regime', 'prob_bull', 'gate_continuous', 'gate_smoothed', 
                           'oil_return', 'slope']].copy()
    results_out.index.name = 'Date'
    results_out.to_csv(out_gates_csv)
    print(f"\nSaved gates: {out_gates_csv}")
    
    sensitivity_df.to_csv(out_sensitivity_csv, index=False)
    print(f"Saved sensitivity analysis: {out_sensitivity_csv}")
    
    # Combine train and test metrics
    full_metrics['Period'] = 'Full'
    test_metrics['Period'] = 'Test'
    combined_metrics = pd.concat([full_metrics, test_metrics], ignore_index=True)
    combined_metrics.to_csv(out_metrics_csv, index=False)
    print(f"Saved regime metrics: {out_metrics_csv}")
    
    with open(out_hmm_json, "w") as f:
        json.dump(hmm_model.to_dict(), f, indent=2)
    print(f"Saved HMM parameters: {out_hmm_json}")
    
    # Plot
    try:
        plot_hmm_results(oil_df, results, train_end, smooth_window, out_plot_png)
        print(f"Saved plot: {out_plot_png}")
    except Exception as e:
        print(f"Plotting error: {e}")
    
    return hmm_model, results, sensitivity_df, combined_metrics


def cli():
    parser = argparse.ArgumentParser(description="Enhanced multi-feature oil state gate HMM")
    parser.add_argument("--oil_csv", required=True, help="CSV with oil prices")
    parser.add_argument("--train_end", default="2018-12-31", help="End date for training period")
    parser.add_argument("--out_gates_csv", default="oil_gates_hmm.csv", help="Output gates")
    parser.add_argument("--out_sensitivity_csv", default="oil_sensitivity_analysis.csv", help="Sensitivity analysis")
    parser.add_argument("--out_hmm_json", default="oil_hmm_params.json", help="HMM parameters")
    parser.add_argument("--out_metrics_csv", default="oil_regime_metrics.csv", help="Regime performance metrics")
    parser.add_argument("--plot_png", default="oil_hmm_plot.png", help="Output plot")
    parser.add_argument("--n_states", type=int, default=2, help="Number of HMM states")
    parser.add_argument("--n_iter", type=int, default=100, help="Max HMM iterations")
    parser.add_argument("--smooth_window", type=int, default=5, help="Smoothing window for gate")
    parser.add_argument("--features", nargs='+', default=None, 
                       help="Features to use (default: oil_return slope vol_20d return_20d)")
    args = parser.parse_args()
    
    run_oil_hmm_gate(
        oil_csv=args.oil_csv,
        train_end=args.train_end,
        out_gates_csv=args.out_gates_csv,
        out_sensitivity_csv=args.out_sensitivity_csv,
        out_hmm_json=args.out_hmm_json,
        out_metrics_csv=args.out_metrics_csv,
        out_plot_png=args.plot_png,
        n_states=args.n_states,
        n_iter=args.n_iter,
        use_features=args.features,
        smooth_window=args.smooth_window
    )


if __name__ == "__main__":
    cli()