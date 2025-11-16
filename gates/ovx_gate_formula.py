"""
OVX Gate - Formula-Based Continuous Scaling

Pure continuous transformation of OVX to gate values.
No arbitrary thresholds - mathematically principled approach.

Based on:
- Barroso & Santa-Clara (2015) volatility scaling
- Power-law transformation with smooth clipping
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


class FormulaOVXGate:
    """
    Formula-based continuous OVX gate.
    
    Gate = (baseline_ovx / current_ovx)^power
    
    No discrete states or arbitrary thresholds.
    Pure continuous mathematical transformation.
    """
    
    def __init__(
        self,
        baseline_ovx: float = 30.0,
        power: float = 0.5,
        min_gate: float = 0.5,
        max_gate_no_scaleup: float = 1.0,
        max_gate_with_scaleup: float = 1.5,
        smooth_window: int = 5
    ):
        """
        Parameters:
        -----------
        baseline_ovx : float
            Baseline OVX level (historical mean ~30)
        power : float
            Power for transformation (0.5 = square root, following literature)
        min_gate : float
            Minimum gate value (maximum de-risking)
        max_gate_no_scaleup : float
            Max gate for conservative version (no leverage)
        max_gate_with_scaleup : float
            Max gate for aggressive version (allow leverage)
        smooth_window : int
            Window for smoothing gate values
        """
        self.baseline_ovx = baseline_ovx
        self.power = power
        self.min_gate = min_gate
        self.max_gate_no_scaleup = max_gate_no_scaleup
        self.max_gate_with_scaleup = max_gate_with_scaleup
        self.smooth_window = smooth_window
    
    def calculate_raw_gate(self, ovx: pd.Series) -> pd.Series:
        """
        Calculate raw gate value before clipping.
        
        Formula: (baseline / ovx)^power
        """
        return (self.baseline_ovx / ovx) ** self.power
    
    def smooth_clip(
        self, 
        values: pd.Series, 
        min_val: float, 
        max_val: float,
        smoothness: float = 0.05
    ) -> pd.Series:
        """
        Smooth clipping using logistic function.
        
        Avoids hard boundaries - gradual approach to limits.
        """
        # Normalize to [0, 1]
        normalized = (values - min_val) / (max_val - min_val)
        
        # Apply soft clipping
        clipped = 1 / (1 + np.exp(-(normalized - 0.5) / smoothness))
        
        # Map back to [min_val, max_val]
        return min_val + clipped * (max_val - min_val)
    
    def calculate_gate(
        self, 
        ovx: pd.Series, 
        allow_scaleup: bool = False
    ) -> pd.DataFrame:
        """
        Calculate gate values from OVX levels.
        
        Returns DataFrame with raw and clipped gates.
        """
        # Raw gate (no limits)
        raw_gate = self.calculate_raw_gate(ovx)
        
        # Determine max gate based on policy
        max_gate = (self.max_gate_with_scaleup if allow_scaleup 
                   else self.max_gate_no_scaleup)
        
        # Smooth clipping
        clipped_gate = self.smooth_clip(raw_gate, self.min_gate, max_gate)
        
        # Apply smoothing
        smoothed_gate = clipped_gate.rolling(
            window=self.smooth_window,
            min_periods=1,
            center=False
        ).mean()
        
        return pd.DataFrame({
            'raw_gate': raw_gate,
            'clipped_gate': clipped_gate,
            'smoothed_gate': smoothed_gate
        }, index=ovx.index)
    
    def generate_reason(self, ovx_level: float, gate_value: float) -> str:
        """Generate human-readable reason for gate value."""
        if ovx_level < 25:
            regime = "Very low oil volatility"
        elif ovx_level < 35:
            regime = "Low-normal oil volatility"
        elif ovx_level < 45:
            regime = "Elevated oil volatility"
        elif ovx_level < 60:
            regime = "High oil volatility"
        else:
            regime = "Extreme oil volatility"
        
        if gate_value > 1.1:
            action = "scaled up"
        elif gate_value > 0.95:
            action = "near neutral"
        elif gate_value > 0.7:
            action = "moderately reduced"
        else:
            action = "significantly reduced"
        
        return f"{regime} (OVX={ovx_level:.1f}) - exposure {action}"


def main(
    ovx_csv: str = "data/ovx_daily.csv",
    output_csv: str = "output/oil_signals/gates_ovx_formula.csv",
    baseline_ovx: float = 30.0,
    power: float = 0.5
):
    """
    Generate formula-based OVX gates.
    
    Parameters:
    -----------
    ovx_csv : str
        Path to OVX data CSV
    output_csv : str
        Path for output gates CSV
    baseline_ovx : float
        Baseline OVX level
    power : float
        Power for transformation
    """
    print("="*70)
    print("OVX FORMULA GATE CALCULATION")
    print("="*70)
    
    # Load OVX data
    print(f"\nLoading OVX data from: {ovx_csv}")
    ovx_df = pd.read_csv(ovx_csv, parse_dates=['date'])
    ovx_df = ovx_df.set_index('date').sort_index()
    
    print(f"Loaded {len(ovx_df)} days of OVX data")
    print(f"Date range: {ovx_df.index.min().date()} to {ovx_df.index.max().date()}")
    print(f"\nOVX statistics:")
    print(f"  Mean: {ovx_df['ovx_close'].mean():.2f}")
    print(f"  Std:  {ovx_df['ovx_close'].std():.2f}")
    print(f"  Min:  {ovx_df['ovx_close'].min():.2f}")
    print(f"  Max:  {ovx_df['ovx_close'].max():.2f}")
    
    # Initialize gate calculator
    print(f"\nInitializing formula gate:")
    print(f"  Baseline OVX: {baseline_ovx}")
    print(f"  Power: {power}")
    print(f"  Formula: gate = ({baseline_ovx} / OVX)^{power}")
    
    gate_calc = FormulaOVXGate(
        baseline_ovx=baseline_ovx,
        power=power
    )
    
    # Calculate gates - conservative version (no scale up)
    print("\nCalculating gates (no scale-up version)...")
    gates_conservative = gate_calc.calculate_gate(
        ovx_df['ovx_close'], 
        allow_scaleup=False
    )
    
    # Calculate gates - aggressive version (with scale up)
    print("Calculating gates (with scale-up version)...")
    gates_aggressive = gate_calc.calculate_gate(
        ovx_df['ovx_close'], 
        allow_scaleup=True
    )
    
    # Combine results
    results = pd.DataFrame({
        'ovx_level': ovx_df['ovx_close'],
        'gate_formula_no_scaleup': gates_conservative['smoothed_gate'],
        'gate_formula_with_scaleup': gates_aggressive['smoothed_gate'],
        'raw_gate': gates_conservative['raw_gate']
    })
    
    # Generate reasons
    results['reason'] = [
        gate_calc.generate_reason(row['ovx_level'], row['gate_formula_no_scaleup'])
        for _, row in results.iterrows()
    ]
    
    # Statistics
    print("\n" + "="*70)
    print("GATE STATISTICS")
    print("="*70)
    
    print("\nConservative (No Scale-Up):")
    print(f"  Mean: {results['gate_formula_no_scaleup'].mean():.3f}")
    print(f"  Std:  {results['gate_formula_no_scaleup'].std():.3f}")
    print(f"  Min:  {results['gate_formula_no_scaleup'].min():.3f}")
    print(f"  Max:  {results['gate_formula_no_scaleup'].max():.3f}")
    print(f"  % at max (1.0): {(results['gate_formula_no_scaleup'] >= 0.99).mean()*100:.1f}%")
    print(f"  % at min (0.5): {(results['gate_formula_no_scaleup'] <= 0.51).mean()*100:.1f}%")
    
    print("\nAggressive (With Scale-Up):")
    print(f"  Mean: {results['gate_formula_with_scaleup'].mean():.3f}")
    print(f"  Std:  {results['gate_formula_with_scaleup'].std():.3f}")
    print(f"  Min:  {results['gate_formula_with_scaleup'].min():.3f}")
    print(f"  Max:  {results['gate_formula_with_scaleup'].max():.3f}")
    print(f"  % scaled up (>1.0): {(results['gate_formula_with_scaleup'] > 1.01).mean()*100:.1f}%")
    print(f"  % at min (0.5): {(results['gate_formula_with_scaleup'] <= 0.51).mean()*100:.1f}%")
    
    # Crisis periods
    print("\n" + "="*70)
    print("CRISIS PERIOD ANALYSIS")
    print("="*70)
    
    crisis_periods = [
        ('2008 Financial Crisis', '2008-09-01', '2008-12-31'),
        ('2011 European Debt', '2011-07-01', '2011-10-31'),
        ('2014-2016 Oil Crash', '2014-06-01', '2016-02-29'),
        ('2020 COVID', '2020-02-01', '2020-04-30'),
        ('2022 Ukraine War', '2022-02-15', '2022-04-15')
    ]
    
    for name, start, end in crisis_periods:
        mask = (results.index >= start) & (results.index <= end)
        if mask.sum() > 0:
            crisis_data = results[mask]
            print(f"\n{name}:")
            print(f"  Mean OVX: {crisis_data['ovx_level'].mean():.1f}")
            print(f"  Mean gate (conservative): {crisis_data['gate_formula_no_scaleup'].mean():.3f}")
            print(f"  Mean gate (aggressive): {crisis_data['gate_formula_with_scaleup'].mean():.3f}")
    
    # Save output
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results.to_csv(output_path)
    print(f"\n{'='*70}")
    print(f"✓ Gates saved to: {output_csv}")
    print(f"  {len(results)} days of gates generated")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate formula-based OVX gates"
    )
    parser.add_argument(
        '--ovx_csv',
        default='data/ovx_daily.csv',
        help='Path to OVX data CSV'
    )
    parser.add_argument(
        '--output_csv',
        default='output/oil_signals/gates_ovx_formula.csv',
        help='Path for output gates CSV'
    )
    parser.add_argument(
        '--baseline_ovx',
        type=float,
        default=30.0,
        help='Baseline OVX level (default: 30.0)'
    )
    parser.add_argument(
        '--power',
        type=float,
        default=0.5,
        help='Power for transformation (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    main(
        ovx_csv=args.ovx_csv,
        output_csv=args.output_csv,
        baseline_ovx=args.baseline_ovx,
        power=args.power
    )