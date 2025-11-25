"""
Stable 2-State HMM for OVX

Fixes the regime instability issue by:
1. Using only 2 states (calm/stress)
2. Simpler feature set (just OVX level + 20-day change)
3. Enforcing minimum regime persistence
4. Exponential smoothing of features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle
from typing import Tuple, Optional, Dict
from hmmlearn import hmm
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


class OVX_HMM_Stable:
    """
    Stable 2-state HMM for OVX regime detection.
    
    Simplifications for stability:
    - Only 2 states (calm/stress)
    - Fewer features (level + 20-day change only)
    - Exponential smoothing
    - Hard assignment
    - Minimum persistence constraint
    """
    
    def __init__(
        self,
        n_states: int = 2,  # Only 2 states
        n_iter: int = 200,
        random_state: int = 42,
        smoothing_alpha: float = 0.1  # Exponential smoothing
    ):
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        self.smoothing_alpha = smoothing_alpha
        
        # Use diagonal covariance for stability
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=random_state,
            verbose=False,
            # Add transition matrix constraints for stability
            params="stmc",  # Update all parameters
            init_params="mc"  # Initialize means and covariances from data
        )
        
        self.scaler = RobustScaler()
        self.regime_powers = None
        self.regime_order = None
        self.is_fitted = False
        self.feature_names = None
    
    def prepare_features(self, ovx_data: pd.DataFrame) -> np.ndarray:
        """
        Simplified feature engineering for stability.
        
        Only 2 features:
        1. OVX level (smoothed)
        2. 20-day change (smoothed)
        """
        features = pd.DataFrame(index=ovx_data.index)
        
        # Feature 1: Exponentially smoothed OVX level
        features['ovx_smooth'] = ovx_data['ovx_close'].ewm(
            alpha=self.smoothing_alpha, 
            adjust=False
        ).mean()
        
        # Feature 2: Smoothed 20-day change
        ovx_change = ovx_data['ovx_close'].diff(20)
        features['ovx_change_20d_smooth'] = ovx_change.ewm(
            alpha=self.smoothing_alpha,
            adjust=False
        ).mean()
        
        # Fill NaN
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        self.feature_names = features.columns.tolist()
        
        return features.values
    
    def fit(self, train_data: pd.DataFrame, verbose: bool = True):
        """Fit HMM with stability checks."""
        if verbose:
            print("\nFitting 2-state HMM with stability constraints...")
            print(f"  Training samples: {len(train_data)}")
            print(f"  Smoothing alpha: {self.smoothing_alpha}")
        
        features = self.prepare_features(train_data)
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit HMM
        self.model.fit(features_scaled)
        
        # Check convergence
        if verbose and hasattr(self.model, 'monitor_'):
            converged = self.model.monitor_.converged
            print(f"  Converged: {converged}")
        
        # Identify regime order
        self._identify_regime_order(train_data)
        
        # Check stability
        self._check_regime_stability(train_data, verbose)
        
        # Enforce minimum persistence in transition matrix
        self._enforce_persistence(verbose)
        
        self.is_fitted = True
        
        if verbose:
            print("  ✓ HMM training complete")
            self._print_regime_summary(train_data)
    
    def _identify_regime_order(self, train_data: pd.DataFrame):
        """Identify which state is calm vs stress."""
        features = self.prepare_features(train_data)
        features_scaled = self.scaler.transform(features)
        states = self.model.predict(features_scaled)
        
        # Calculate mean OVX per state
        mean_ovx_per_state = {}
        for state in range(self.n_states):
            mask = states == state
            mean_ovx_per_state[state] = train_data.loc[mask, 'ovx_close'].mean()
        
        # Order states by mean OVX
        ordered_states = sorted(mean_ovx_per_state.items(), key=lambda x: x[1])
        
        # Map to regime names (only 2 states)
        self.regime_order = {
            'calm': ordered_states[0][0],    # Lower OVX
            'stress': ordered_states[1][0]   # Higher OVX
        }
    
    def _enforce_persistence(self, verbose: bool = True):
        """
        Enforce minimum regime persistence.
        
        Modifies transition matrix to ensure regimes don't flip too often.
        """
        # Get current transition matrix
        trans_mat = self.model.transmat_
        
        # Enforce minimum stay probability of 90%
        # (This means average regime duration = 10 days)
        min_stay_prob = 0.90
        
        for i in range(self.n_states):
            if trans_mat[i, i] < min_stay_prob:
                # Set stay probability
                trans_mat[i, i] = min_stay_prob
                
                # Distribute remaining probability to other states
                remaining = 1.0 - min_stay_prob
                for j in range(self.n_states):
                    if j != i:
                        trans_mat[i, j] = remaining / (self.n_states - 1)
        
        # Normalize rows to sum to 1
        trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
        
        # Update model
        self.model.transmat_ = trans_mat
        
        if verbose:
            print(f"\n  Enforced Minimum Persistence:")
            print(f"    Calm stays calm:   {trans_mat[self.regime_order['calm'], self.regime_order['calm']]:.1%}")
            print(f"    Stress stays stress: {trans_mat[self.regime_order['stress'], self.regime_order['stress']]:.1%}")
    
    def _check_regime_stability(self, train_data: pd.DataFrame, verbose: bool = True):
        """Check regime persistence."""
        features = self.prepare_features(train_data)
        features_scaled = self.scaler.transform(features)
        states = self.model.predict(features_scaled)
        
        transitions = (states[1:] != states[:-1]).sum()
        total_days = len(states)
        transition_rate = transitions / total_days
        
        if verbose:
            print(f"\n  Pre-Enforcement Regime Stability:")
            print(f"    Transition rate: {transition_rate:.1%}")
            
            if transition_rate < 0.02:
                print(f"    ✓ Very stable")
            elif transition_rate > 0.10:
                print(f"    ⚠️  Unstable (will enforce persistence)")
            else:
                print(f"    ✓ Good stability")
    
    def _print_regime_summary(self, train_data: pd.DataFrame):
        """Print regime summary."""
        features = self.prepare_features(train_data)
        features_scaled = self.scaler.transform(features)
        states = self.model.predict(features_scaled)
        
        print("\n  Regime Summary:")
        for regime_name, state_id in self.regime_order.items():
            mask = states == state_id
            regime_data = train_data.loc[mask, 'ovx_close']
            
            print(f"    {regime_name.capitalize()} (State {state_id}):")
            print(f"      Days: {mask.sum()} ({mask.mean()*100:.1f}%)")
            print(f"      Mean OVX: {regime_data.mean():.2f}")
            print(f"      Range: [{regime_data.min():.1f}, {regime_data.max():.1f}]")
    
    def predict_regimes_hard(self, data: pd.DataFrame) -> np.ndarray:
        """Hard regime assignment with enforced persistence."""
        features = self.prepare_features(data)
        features_scaled = self.scaler.transform(features)
        
        # Use Viterbi algorithm (respects transition probabilities)
        states = self.model.predict(features_scaled)
        
        return states
    
    def predict_probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """Get posterior probabilities."""
        features = self.prepare_features(data)
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)
    
    def optimize_regime_powers(
        self,
        val_ovx: pd.DataFrame,
        val_returns: pd.Series,
        baseline_ovx: float = 30.0,
        power_range: Tuple[float, float] = (0.2, 0.8),
        n_trials: int = 20,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Optimize powers with 2-state model."""
        if verbose:
            print("\nOptimizing 2-state regime powers...")
            print(f"  Validation days: {len(val_ovx)}")
        
        states = self.predict_regimes_hard(val_ovx)
        
        # Show regime distribution
        for regime_name, state_id in self.regime_order.items():
            mask = states == state_id
            if verbose:
                print(f"  {regime_name}: {mask.sum()} days ({mask.mean()*100:.1f}%)")
        
        # Grid search for optimal powers
        powers_to_test = np.linspace(power_range[0], power_range[1], n_trials)
        
        best_sharpe = -np.inf
        best_powers = None
        
        tested = 0
        
        if verbose:
            print(f"\n  Testing {n_trials**2} power combinations...")
        
        for calm_p in powers_to_test:
            for stress_p in powers_to_test:
                # Constraint: calm >= stress (more aggressive in calm periods)
                if calm_p < stress_p:
                    continue
                
                # Minimum gap
                if calm_p - stress_p < 0.15:
                    continue
                
                powers = {
                    'calm': calm_p,
                    'stress': stress_p
                }
                
                try:
                    sharpe = self._calculate_validation_sharpe(
                        val_ovx, val_returns, powers, baseline_ovx
                    )
                    
                    tested += 1
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_powers = powers
                except:
                    continue
        
        if best_powers is None:
            if verbose:
                print("  ⚠️  Optimization failed, using defaults")
            best_powers = {
                'calm': 0.7,
                'stress': 0.3
            }
            best_sharpe = 0.0
        
        if verbose:
            print(f"    Tested: {tested} combinations")
            print(f"    Best validation Sharpe: {best_sharpe:.3f}")
            print(f"    Optimal powers:")
            for regime, power in best_powers.items():
                print(f"      {regime}: {power:.3f}")
        
        self.regime_powers = best_powers
        return best_powers
    
    def _calculate_validation_sharpe(
        self,
        val_ovx: pd.DataFrame,
        val_returns: pd.Series,
        powers: Dict[str, float],
        baseline_ovx: float
    ) -> float:
        """Calculate Sharpe with given powers."""
        states = self.predict_regimes_hard(val_ovx)
        
        gates = np.zeros(len(val_ovx))
        
        for i, state in enumerate(states):
            regime_name = None
            for name, regime_state in self.regime_order.items():
                if regime_state == state:
                    regime_name = name
                    break
            
            if regime_name is None or regime_name not in powers:
                gates[i] = 1.0
                continue
            
            power = powers[regime_name]
            ovx_value = val_ovx['ovx_close'].iloc[i]
            gates[i] = (baseline_ovx / ovx_value) ** power
        
        gates = np.clip(gates, 0.5, 1.0)
        
        gate_series = pd.Series(gates, index=val_ovx.index)
        common_idx = gate_series.index.intersection(val_returns.index)
        
        if len(common_idx) == 0:
            return 0.0
        
        gate_series = gate_series.loc[common_idx]
        val_returns_aligned = val_returns.loc[common_idx]
        
        scaled_returns = val_returns_aligned * gate_series.values
        
        if len(scaled_returns) < 20 or scaled_returns.std() == 0:
            return 0.0
        
        sharpe = (scaled_returns.mean() / scaled_returns.std()) * np.sqrt(252)
        return sharpe
    
    def calculate_gates_hard_assignment(
        self,
        ovx_data: pd.DataFrame,
        baseline_ovx: float = 30.0,
        allow_scaleup: bool = False,
        smooth_window: int = 5
    ) -> pd.Series:
        """Calculate gates using hard assignment."""
        if not self.is_fitted or self.regime_powers is None:
            raise ValueError("Model not fitted or powers not optimized")
        
        states = self.predict_regimes_hard(ovx_data)
        
        gates = np.zeros(len(ovx_data))
        
        for i, state in enumerate(states):
            regime_name = None
            for name, regime_state in self.regime_order.items():
                if regime_state == state:
                    regime_name = name
                    break
            
            if regime_name is None or regime_name not in self.regime_powers:
                power = 0.5
            else:
                power = self.regime_powers[regime_name]
            
            ovx_value = ovx_data['ovx_close'].iloc[i]
            gates[i] = (baseline_ovx / ovx_value) ** power
        
        min_gate = 0.5
        max_gate = 1.5 if allow_scaleup else 1.0
        
        gates = np.clip(gates, min_gate, max_gate)
        
        gate_series = pd.Series(gates, index=ovx_data.index)
        smoothed = gate_series.rolling(smooth_window, min_periods=1).mean()
        
        return smoothed


def main(
    ovx_csv: str = "data/ovx_daily.csv",
    returns_csv: str = "output/validation_returns.csv",
    output_csv: str = "output/oil_signals/gates_ovx_hmm_stable.csv",
    model_path: str = "models/ovx_hmm_stable.pkl",
    smoothing_alpha: float = 0.1
):
    """Train stable 2-state HMM."""
    print("="*70)
    print("STABLE 2-STATE HMM FOR OVX")
    print("="*70)
    
    # Load OVX data
    print(f"\nLoading OVX data from: {ovx_csv}")
    ovx_df = pd.read_csv(ovx_csv)
    date_col = 'time' if 'time' in ovx_df.columns else 'date'
    ovx_df[date_col] = pd.to_datetime(ovx_df[date_col])
    ovx_df = ovx_df.rename(columns={date_col: 'date'})
    ovx_df = ovx_df.set_index('date').sort_index()
    
    print(f"Loaded {len(ovx_df)} days of OVX data")
    
    # Split data
    train_end = '2015-12-31'
    val_start = '2016-01-01'
    val_end = '2018-12-31'
    
    train_ovx = ovx_df.loc[:train_end]
    val_ovx = ovx_df.loc[val_start:val_end]
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_ovx)} days")
    print(f"  Val:   {len(val_ovx)} days")
    
    # Load validation returns
    print(f"\nLoading validation returns from: {returns_csv}")
    try:
        val_returns = pd.read_csv(returns_csv, index_col=0, parse_dates=True)
        if isinstance(val_returns, pd.DataFrame):
            val_returns = val_returns.iloc[:, 0]
        
        print(f"Loaded {len(val_returns)} days")
        
    except FileNotFoundError:
        print("  ⚠️  Validation returns file not found!")
        return
    
    # Initialize HMM
    hmm_model = OVX_HMM_Stable(
        n_states=2,
        n_iter=200,
        smoothing_alpha=smoothing_alpha
    )
    
    # Train
    hmm_model.fit(train_ovx, verbose=True)
    
    # Optimize powers
    hmm_model.optimize_regime_powers(val_ovx, val_returns, verbose=True)
    
    # Generate gates
    print("\nGenerating gates...")
    gates_conservative = hmm_model.calculate_gates_hard_assignment(
        ovx_df, allow_scaleup=False
    )
    gates_aggressive = hmm_model.calculate_gates_hard_assignment(
        ovx_df, allow_scaleup=True
    )
    
    # Get states
    states = hmm_model.predict_regimes_hard(ovx_df)
    posteriors = hmm_model.predict_probabilities(ovx_df)
    
    # Create output
    results = pd.DataFrame({
        'ovx_level': ovx_df['ovx_close'],
        'gate_hmm_no_scaleup': gates_conservative,
        'gate_hmm_with_scaleup': gates_aggressive,
        'state': states
    })
    
    # Add posteriors
    for regime_name, state_id in hmm_model.regime_order.items():
        results[f'prob_{regime_name}'] = posteriors[:, state_id]
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path)
    
    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(hmm_model, f)
    
    print(f"\n  ✓ Gates saved to: {output_csv}")
    print(f"  ✓ Model saved to: {model_path}")
    
    # Statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"\nOptimized Powers:")
    for regime, power in hmm_model.regime_powers.items():
        print(f"  {regime:7s}: {power:.3f}")
    
    print(f"\nGate Statistics:")
    print(f"  Mean:  {gates_conservative.mean():.3f}")
    print(f"  Std:   {gates_conservative.std():.3f}")
    print(f"  Range: [{gates_conservative.min():.3f}, {gates_conservative.max():.3f}]")
    
    # Check final stability
    print(f"\nFinal Regime Stability:")
    transitions = (states[1:] != states[:-1]).sum()
    transition_rate = transitions / len(states)
    print(f"  Transition rate: {transition_rate:.1%}")
    
    # Crisis behavior
    print(f"\nCrisis Behavior:")
    for name, start, end in [
        ('COVID 2020', '2020-02-01', '2020-04-30'),
        ('Ukraine 2022', '2022-02-15', '2022-04-15')
    ]:
        try:
            mask = (results.index >= start) & (results.index <= end)
            crisis_data = results[mask]
            print(f"  {name}:")
            print(f"    Mean OVX: {crisis_data['ovx_level'].mean():.1f}")
            print(f"    Mean gate: {crisis_data['gate_hmm_no_scaleup'].mean():.3f}")
            print(f"    De-risking: {(1 - crisis_data['gate_hmm_no_scaleup'].mean())*100:.1f}%")
        except:
            pass
    
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ovx_csv', default='data/ovx_daily.csv')
    parser.add_argument('--returns_csv', default='output/validation_returns.csv')
    parser.add_argument('--output_csv', default='output/oil_signals/gates_ovx_hmm_stable.csv')
    parser.add_argument('--model_path', default='models/ovx_hmm_stable.pkl')
    parser.add_argument('--smoothing_alpha', type=float, default=0.1)
    
    args = parser.parse_args()
    main(**vars(args))
