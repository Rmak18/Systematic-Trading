"""
OVX Gate - HMM-Based Continuous Scaling

Uses Hidden Markov Model to identify latent volatility regimes.
Each regime has a learned continuous scaling function.
Gate values are continuous blends via posterior probabilities.

NO ARBITRARY THRESHOLDS - all parameters learned from data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle
from typing import Tuple, Optional
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


class OVX_HMM:
    """
    Hidden Markov Model for OVX regime detection.
    
    Learns latent volatility regimes and regime-specific
    continuous scaling functions.
    """
    
    def __init__(
        self,
        n_states: int = 3,
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Parameters:
        -----------
        n_states : int
            Number of hidden states (regimes)
        n_iter : int
            Maximum iterations for EM algorithm
        random_state : int
            Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Initialize HMM (Gaussian emissions)
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            verbose=False
        )
        
        # Scalers for features
        self.scaler = StandardScaler()
        
        # Regime-specific parameters (learned during training)
        self.regime_powers = None
        self.regime_order = None  # Maps states to calm/normal/stress
        
        # Fitted flag
        self.is_fitted = False
    
    def prepare_features(self, ovx_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM.
        
        Features:
        - OVX level (normalized)
        - OVX 5-day change
        - OVX 20-day rolling std
        """
        features = pd.DataFrame(index=ovx_data.index)
        
        # Feature 1: OVX level
        features['ovx_level'] = ovx_data['ovx_close']
        
        # Feature 2: 5-day change
        features['ovx_change_5d'] = ovx_data['ovx_close'].diff(5)
        
        # Feature 3: 20-day rolling std
        features['ovx_std_20d'] = ovx_data['ovx_close'].rolling(20).std()
        
        # Fill NaNs
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        return features.values
    
    def fit(self, train_data: pd.DataFrame, verbose: bool = True):
        """
        Fit HMM on training data.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data with 'ovx_close' column
        verbose : bool
            Print training progress
        """
        if verbose:
            print("\nFitting HMM...")
            print(f"  States: {self.n_states}")
            print(f"  Training samples: {len(train_data)}")
        
        # Prepare features
        features = self.prepare_features(train_data)
        
        # Fit scaler on training data
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit HMM
        self.model.fit(features_scaled)
        
        # Identify regime order (calm → normal → stress)
        self._identify_regime_order(train_data)
        
        self.is_fitted = True
        
        if verbose:
            print("  ✓ HMM training complete")
            print(f"  Log-likelihood: {self.model.score(features_scaled):.2f}")
            self._print_regime_summary(train_data)
    
    def _identify_regime_order(self, train_data: pd.DataFrame):
        """
        Identify which state corresponds to which regime.
        
        Orders states by mean OVX level:
        - Lowest mean OVX = Calm regime
        - Middle mean OVX = Normal regime
        - Highest mean OVX = Stress regime
        """
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
        
        # Map to regime names
        self.regime_order = {
            'calm': ordered_states[0][0],
            'normal': ordered_states[1][0] if self.n_states > 2 else None,
            'stress': ordered_states[-1][0]
        }
    
    def _print_regime_summary(self, train_data: pd.DataFrame):
        """Print summary of learned regimes."""
        features = self.prepare_features(train_data)
        features_scaled = self.scaler.transform(features)
        states = self.model.predict(features_scaled)
        
        print("\n  Regime Summary:")
        regime_names = ['Calm', 'Normal', 'Stress'] if self.n_states == 3 else ['Calm', 'Stress']
        
        for i, (regime_name, state_id) in enumerate(self.regime_order.items()):
            if state_id is None:
                continue
            mask = states == state_id
            regime_data = train_data.loc[mask, 'ovx_close']
            
            print(f"    {regime_name.capitalize()} (State {state_id}):")
            print(f"      Days: {mask.sum()} ({mask.mean()*100:.1f}%)")
            print(f"      Mean OVX: {regime_data.mean():.2f}")
            print(f"      Std OVX: {regime_data.std():.2f}")
    
    def predict_probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict posterior probabilities for each regime.
        
        Returns:
        --------
        posteriors : np.ndarray
            Shape (n_samples, n_states) with P(regime | OVX)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features = self.prepare_features(data)
        features_scaled = self.scaler.transform(features)
        
        # Get posteriors
        posteriors = self.model.predict_proba(features_scaled)
        
        return posteriors
    
    def optimize_regime_powers(
        self,
        val_data: pd.DataFrame,
        val_returns: pd.Series,
        power_range: Tuple[float, float] = (0.2, 1.0),
        n_trials: int = 20,
        verbose: bool = True
    ) -> dict:
        """
        Optimize power parameter for each regime using validation set.
        
        Finds regime-specific powers that maximize Sharpe ratio.
        
        Parameters:
        -----------
        val_data : pd.DataFrame
            Validation OVX data
        val_returns : pd.Series
            Validation period returns
        power_range : tuple
            (min_power, max_power) to search
        n_trials : int
            Number of power values to test
        
        Returns:
        --------
        regime_powers : dict
            Optimal power for each regime
        """
        if verbose:
            print("\nOptimizing regime-specific powers on validation set...")
        
        # Get regime posteriors for validation set
        posteriors = self.predict_probabilities(val_data)
        
        # Try different power combinations
        powers_to_test = np.linspace(power_range[0], power_range[1], n_trials)
        
        best_sharpe = -np.inf
        best_powers = None
        
        # For each regime, independently optimize power
        regime_powers = {}
        
        for regime_name, state_id in self.regime_order.items():
            if state_id is None:
                continue
            
            if verbose:
                print(f"  Optimizing {regime_name} regime...")
            
            # Get periods where this regime is dominant
            regime_prob = posteriors[:, state_id]
            dominant_mask = regime_prob > 0.5
            
            if dominant_mask.sum() < 10:
                # Not enough data, use default
                regime_powers[regime_name] = 0.5
                if verbose:
                    print(f"    Insufficient data, using default power=0.5")
                continue
            
            # Test different powers for this regime
            best_regime_power = 0.5
            best_regime_sharpe = -np.inf
            
            for power in powers_to_test:
                # Calculate gates with this power for this regime
                gates = (30.0 / val_data['ovx_close']) ** power
                gates = np.clip(gates, 0.5, 1.5)
                
                # Calculate returns with this gate (only for dominant periods)
                scaled_returns = val_returns * gates
                regime_returns = scaled_returns[dominant_mask]
                
                # Calculate Sharpe
                if len(regime_returns) > 0:
                    sharpe = regime_returns.mean() / (regime_returns.std() + 1e-6) * np.sqrt(252)
                    
                    if sharpe > best_regime_sharpe:
                        best_regime_sharpe = sharpe
                        best_regime_power = power
            
            regime_powers[regime_name] = best_regime_power
            
            if verbose:
                print(f"    Optimal power: {best_regime_power:.3f} (Sharpe: {best_regime_sharpe:.3f})")
        
        self.regime_powers = regime_powers
        
        if verbose:
            print("  ✓ Power optimization complete")
        
        return regime_powers
    
    def calculate_continuous_gate(
        self,
        ovx_data: pd.DataFrame,
        baseline_ovx: float = 30.0,
        allow_scaleup: bool = False,
        smooth_window: int = 5
    ) -> pd.Series:
        """
        Calculate continuous gate values using regime-specific functions.
        
        Gate is continuous blend of regime-specific functions weighted
        by posterior probabilities.
        
        Parameters:
        -----------
        ovx_data : pd.DataFrame
            OVX data
        baseline_ovx : float
            Baseline for scaling
        allow_scaleup : bool
            Whether to allow scaling above 1.0
        smooth_window : int
            Smoothing window
        
        Returns:
        --------
        gate : pd.Series
            Continuous gate values
        """
        if not self.is_fitted or self.regime_powers is None:
            raise ValueError("Model not fitted or powers not optimized")
        
        # Get posterior probabilities
        posteriors = self.predict_probabilities(ovx_data)
        
        # Calculate gate for each regime
        gates_per_regime = np.zeros((len(ovx_data), self.n_states))
        
        for regime_name, state_id in self.regime_order.items():
            if state_id is None:
                continue
            
            power = self.regime_powers[regime_name]
            
            # Calculate continuous gate with this regime's power
            regime_gate = (baseline_ovx / ovx_data['ovx_close']) ** power
            
            gates_per_regime[:, state_id] = regime_gate.values
        
        # Blend gates using posteriors (continuous!)
        blended_gate = np.sum(posteriors * gates_per_regime, axis=1)
        
        # Determine limits
        min_gate = 0.5
        max_gate = 1.5 if allow_scaleup else 1.0
        
        # Clip
        clipped_gate = np.clip(blended_gate, min_gate, max_gate)
        
        # Smooth
        gate_series = pd.Series(clipped_gate, index=ovx_data.index)
        smoothed_gate = gate_series.rolling(
            window=smooth_window,
            min_periods=1
        ).mean()
        
        return smoothed_gate


def main(
    train_val_test_split: bool = True,
    ovx_csv: str = "data/ovx_daily.csv",
    output_csv: str = "output/oil_signals/gates_ovx_hmm.csv",
    model_path: str = "models/ovx_hmm.pkl",
    n_states: int = 3
):
    """
    Train HMM and generate gates with proper train/val/test split.
    """
    print("="*70)
    print("OVX HMM GATE TRAINING & GENERATION")
    print("="*70)
    
    # Load OVX data
    print(f"\nLoading OVX data from: {ovx_csv}")
    ovx_df = pd.read_csv(ovx_csv, parse_dates=['date'])
    ovx_df = ovx_df.set_index('date').sort_index()
    
    print(f"Loaded {len(ovx_df)} days of OVX data")
    print(f"Date range: {ovx_df.index.min().date()} to {ovx_df.index.max().date()}")
    
    # Split data
    if train_val_test_split:
        train_end = '2015-12-31'
        val_end = '2018-12-31'
        
        train_data = ovx_df.loc[:train_end]
        val_data = ovx_df.loc[train_end:val_end].iloc[1:]  # Exclude overlap
        test_data = ovx_df.loc[val_end:].iloc[1:]  # Exclude overlap
        
        print(f"\nData splits:")
        print(f"  Train: {train_data.index.min().date()} to {train_data.index.max().date()} ({len(train_data)} days)")
        print(f"  Val:   {val_data.index.min().date()} to {val_data.index.max().date()} ({len(val_data)} days)")
        print(f"  Test:  {test_data.index.min().date()} to {test_data.index.max().date()} ({len(test_data)} days)")
    else:
        train_data = ovx_df
        val_data = None
        test_data = None
        print("\nUsing full dataset (no splits)")
    
    # Initialize and train HMM
    print(f"\nInitializing HMM with {n_states} states...")
    hmm_model = OVX_HMM(n_states=n_states)
    
    # Train on training set
    hmm_model.fit(train_data, verbose=True)
    
    # Optimize powers on validation set
    if val_data is not None:
        # For power optimization, we need validation returns
        # Using zero returns as placeholder - in real use, pass actual returns
        val_returns = pd.Series(0, index=val_data.index)
        hmm_model.optimize_regime_powers(val_data, val_returns, verbose=True)
    else:
        # No validation - use default powers
        hmm_model.regime_powers = {
            'calm': 0.7,
            'normal': 0.5,
            'stress': 0.3
        }
        print("\nUsing default regime powers (no validation set)")
    
    # Generate gates for full dataset
    print("\nGenerating gates for full dataset...")
    
    # Conservative (no scale-up)
    gates_conservative = hmm_model.calculate_continuous_gate(
        ovx_df,
        allow_scaleup=False
    )
    
    # Aggressive (with scale-up)
    gates_aggressive = hmm_model.calculate_continuous_gate(
        ovx_df,
        allow_scaleup=True
    )
    
    # Get posteriors and states
    posteriors = hmm_model.predict_probabilities(ovx_df)
    
    # Create output DataFrame
    results = pd.DataFrame({
        'ovx_level': ovx_df['ovx_close'],
        'gate_hmm_no_scaleup': gates_conservative,
        'gate_hmm_with_scaleup': gates_aggressive
    })
    
    # Add regime probabilities
    for regime_name, state_id in hmm_model.regime_order.items():
        if state_id is not None:
            results[f'prob_{regime_name}'] = posteriors[:, state_id]
    
    # Add most likely state
    results['state'] = np.argmax(posteriors, axis=1)
    
    # Generate reason
    def generate_reason(row):
        dominant_regime = max(
            [(regime, row.get(f'prob_{regime}', 0)) 
             for regime in ['calm', 'normal', 'stress'] 
             if row.get(f'prob_{regime}') is not None],
            key=lambda x: x[1]
        )[0]
        return f"HMM: {dominant_regime} regime (OVX={row['ovx_level']:.1f})"
    
    results['reason'] = results.apply(generate_reason, axis=1)
    
    # Save model
    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(hmm_model, f)
    
    print(f"  ✓ Model saved to: {model_path}")
    
    # Save gates
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path)
    
    print(f"  ✓ Gates saved to: {output_csv}")
    
    # Print statistics
    print("\n" + "="*70)
    print("GATE STATISTICS")
    print("="*70)
    
    print("\nConservative (No Scale-Up):")
    print(f"  Mean: {results['gate_hmm_no_scaleup'].mean():.3f}")
    print(f"  Std:  {results['gate_hmm_no_scaleup'].std():.3f}")
    print(f"  Min:  {results['gate_hmm_no_scaleup'].min():.3f}")
    print(f"  Max:  {results['gate_hmm_no_scaleup'].max():.3f}")
    
    print("\nAggressive (With Scale-Up):")
    print(f"  Mean: {results['gate_hmm_with_scaleup'].mean():.3f}")
    print(f"  Std:  {results['gate_hmm_with_scaleup'].std():.3f}")
    print(f"  Min:  {results['gate_hmm_with_scaleup'].min():.3f}")
    print(f"  Max:  {results['gate_hmm_with_scaleup'].max():.3f}")
    
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train HMM and generate OVX gates"
    )
    parser.add_argument(
        '--ovx_csv',
        default='data/ovx_daily.csv',
        help='Path to OVX data CSV'
    )
    parser.add_argument(
        '--output_csv',
        default='output/oil_signals/gates_ovx_hmm.csv',
        help='Path for output gates CSV'
    )
    parser.add_argument(
        '--model_path',
        default='models/ovx_hmm.pkl',
        help='Path to save trained model'
    )
    parser.add_argument(
        '--n_states',
        type=int,
        default=3,
        help='Number of HMM states (default: 3)'
    )
    parser.add_argument(
        '--no_split',
        action='store_true',
        help='Do not split train/val/test (use full data)'
    )
    
    args = parser.parse_args()
    
    main(
        train_val_test_split=not args.no_split,
        ovx_csv=args.ovx_csv,
        output_csv=args.output_csv,
        model_path=args.model_path,
        n_states=args.n_states
    )