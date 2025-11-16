"""
Apply OVX Gates to Portfolio Weights

Applies Formula and HMM gates to baseline weights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple


def load_csv_flexible_date(csv_path, verbose=True):
    """Load CSV with flexible date column (handles 'time' or 'date')."""
    df = pd.read_csv(csv_path)
    
    date_col = None
    if 'time' in df.columns:
        date_col = 'time'
    elif 'date' in df.columns:
        date_col = 'date'
    else:
        raise ValueError(f"No date column in {csv_path}. Expected 'time' or 'date'. Found: {list(df.columns)}")
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'date'})
    df = df.set_index('date').sort_index()
    
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    if verbose:
        print(f"  Loaded: {len(df)} rows")
        print(f"  Range: {df.index.min()} to {df.index.max()}")
    
    return df


def align_dates_preserve_time(weights: pd.DataFrame, gates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align weights and gates by date only, preserving weights' time component."""
    weights_dates = weights.index.normalize()
    gates_dates = gates.index.normalize()
    
    common_dates = weights_dates.intersection(gates_dates)
    
    if len(common_dates) == 0:
        raise ValueError(
            f"No overlapping dates!\n"
            f"  Weights: {weights.index.min().date()} to {weights.index.max().date()}\n"
            f"  Gates: {gates.index.min().date()} to {gates.index.max().date()}"
        )
    
    weights_mask = weights_dates.isin(common_dates)
    gates_mask = gates_dates.isin(common_dates)
    
    weights_aligned = weights[weights_mask].copy()
    gates_aligned = gates[gates_mask].copy()
    
    weights_aligned['_date_only'] = weights_aligned.index.normalize()
    gates_aligned['_date_only'] = gates_aligned.index.normalize()
    
    weights_aligned = weights_aligned.sort_values('_date_only')
    gates_aligned = gates_aligned.sort_values('_date_only')
    
    if len(weights_aligned) != len(gates_aligned):
        raise ValueError(f"After alignment: weights has {len(weights_aligned)} rows but gates has {len(gates_aligned)} rows")
    
    weights_aligned = weights_aligned.drop('_date_only', axis=1)
    gates_aligned = gates_aligned.drop('_date_only', axis=1)
    
    print(f"\n✓ Aligned {len(weights_aligned)} dates")
    print(f"  Range: {weights_aligned.index.min().date()} to {weights_aligned.index.max().date()}")
    
    return weights_aligned, gates_aligned


def apply_gate_to_weights(
    weights: pd.DataFrame,
    gate_values: pd.Series,
    asset_columns: list
) -> pd.DataFrame:
    """Apply gate scaling to asset weights."""
    scaled_weights = weights.copy()
    
    for asset in asset_columns:
        if asset in scaled_weights.columns:
            scaled_weights[asset] = scaled_weights[asset] * gate_values.values
    
    return scaled_weights


def calculate_scaling_statistics(
    original_weights: pd.DataFrame,
    scaled_weights: pd.DataFrame,
    asset_columns: list,
    gate_name: str
):
    """Print statistics about the scaling effect."""
    original_exposure = original_weights[asset_columns].abs().sum(axis=1)
    scaled_exposure = scaled_weights[asset_columns].abs().sum(axis=1)
    scaling_ratio = scaled_exposure / (original_exposure + 1e-10)
    
    print(f"\n{gate_name} Scaling Statistics:")
    print(f"  Mean scaling: {scaling_ratio.mean():.3f}")
    print(f"  Median scaling: {scaling_ratio.median():.3f}")
    print(f"  Min scaling: {scaling_ratio.min():.3f}")
    print(f"  Max scaling: {scaling_ratio.max():.3f}")
    print(f"  % days scaled down (<1.0): {(scaling_ratio < 0.99).mean()*100:.1f}%")
    print(f"  % days scaled up (>1.0): {(scaling_ratio > 1.01).mean()*100:.1f}%")
    print(f"  % days neutral (~1.0): {((scaling_ratio >= 0.99) & (scaling_ratio <= 1.01)).mean()*100:.1f}%")


def main(
    weights_csv: str = "output/signals/weights_after_caps.csv",
    gates_formula_csv: str = "output/oil_signals/gates_ovx_formula.csv",
    gates_hmm_csv: str = "output/oil_signals/gates_ovx_hmm.csv",
    gates_hmm_proper_csv: str = "output/oil_signals/gates_ovx_hmm_stable.csv",
    output_dir: str = "output/signals"
):
    """
    Apply OVX gates to baseline weights.
    
    Generates output files for:
    - Formula (conservative/aggressive)
    - HMM Original (conservative/aggressive)
    - HMM Proper (conservative/aggressive)
    """
    print("="*70)
    print("APPLYING OVX GATES TO WEIGHTS")
    print("="*70)
    
    print(f"\nLoading baseline weights from: {weights_csv}")
    weights_df = load_csv_flexible_date(weights_csv, verbose=True)
    
    asset_columns = [col for col in weights_df.columns if col != 'date']
    
    print(f"\nAssets found: {asset_columns}")
    print(f"Total days: {len(weights_df)}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each gate type
    gate_configs = [
        ('formula', gates_formula_csv, 'gate_formula_no_scaleup', 'gate_formula_with_scaleup'),
        ('hmm', gates_hmm_csv, 'gate_hmm_no_scaleup', 'gate_hmm_with_scaleup'),
        ('hmm_proper', gates_hmm_proper_csv, 'gate_hmm_no_scaleup', 'gate_hmm_with_scaleup')
    ]
    
    for gate_type, gates_csv, conservative_col, aggressive_col in gate_configs:
        print(f"\n{'='*70}")
        print(f"PROCESSING {gate_type.upper()} GATES")
        print(f"{'='*70}")
        
        try:
            print(f"\nLoading gates: {gates_csv}")
            gates_df = load_csv_flexible_date(gates_csv, verbose=True)
            
            weights_aligned, gates_aligned = align_dates_preserve_time(weights_df, gates_df)
            
        except FileNotFoundError:
            print(f"  ✗ Gates file not found: {gates_csv}")
            print(f"  Skipping {gate_type} gates")
            continue
        except ValueError as e:
            print(f"  ✗ Error: {e}")
            print(f"  Skipping {gate_type} gates")
            continue
        
        if conservative_col not in gates_aligned.columns:
            print(f"  ✗ Column '{conservative_col}' not found in gates file")
            print(f"  Available columns: {list(gates_aligned.columns)}")
            print(f"  Skipping {gate_type} gates")
            continue
        
        # Apply conservative gate
        print(f"\n  Applying conservative gate ({conservative_col})...")
        conservative_gate = gates_aligned[conservative_col]
        
        weights_conservative = apply_gate_to_weights(
            weights_aligned,
            conservative_gate,
            asset_columns
        )
        
        calculate_scaling_statistics(
            weights_aligned,
            weights_conservative,
            asset_columns,
            f"{gate_type.upper()} Conservative"
        )
        
        output_file = output_path / f"weights_with_ovx_{gate_type}_no_scaleup.csv"
        weights_conservative.to_csv(output_file)
        print(f"  ✓ Saved: {output_file}")
        
        # Apply aggressive gate
        if aggressive_col not in gates_aligned.columns:
            print(f"  ✗ Column '{aggressive_col}' not found in gates file")
            continue
        
        print(f"\n  Applying aggressive gate ({aggressive_col})...")
        aggressive_gate = gates_aligned[aggressive_col]
        
        weights_aggressive = apply_gate_to_weights(
            weights_aligned,
            aggressive_gate,
            asset_columns
        )
        
        calculate_scaling_statistics(
            weights_aligned,
            weights_aggressive,
            asset_columns,
            f"{gate_type.upper()} Aggressive"
        )
        
        output_file = output_path / f"weights_with_ovx_{gate_type}_with_scaleup.csv"
        weights_aggressive.to_csv(output_file)
        print(f"  ✓ Saved: {output_file}")
    
    print("\n" + "="*70)
    print("✓ ALL GATES APPLIED SUCCESSFULLY")
    print("="*70)
    print(f"\nGenerated files in: {output_dir}/")
    print("  Formula:")
    print("    - weights_with_ovx_formula_no_scaleup.csv")
    print("    - weights_with_ovx_formula_with_scaleup.csv")
    print("  HMM (Original - Posterior Blending):")
    print("    - weights_with_ovx_hmm_no_scaleup.csv")
    print("    - weights_with_ovx_hmm_with_scaleup.csv")
    print("  HMM (Proper - Constrained + Regularized):")
    print("    - weights_with_ovx_hmm_proper_no_scaleup.csv")
    print("    - weights_with_ovx_hmm_proper_with_scaleup.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply OVX gates to portfolio weights"
    )
    parser.add_argument('--weights_csv', default='output/signals/weights_after_caps.csv')
    parser.add_argument('--gates_formula_csv', default='output/oil_signals/gates_ovx_formula.csv')
    parser.add_argument('--gates_hmm_csv', default='output/oil_signals/gates_ovx_hmm.csv')
    parser.add_argument('--gates_hmm_proper_csv', default='output/oil_signals/gates_ovx_hmm_stable.csv')
    parser.add_argument('--output_dir', default='output/signals')
    
    args = parser.parse_args()
    
    main(
        weights_csv=args.weights_csv,
        gates_formula_csv=args.gates_formula_csv,
        gates_hmm_csv=args.gates_hmm_csv,
        gates_hmm_proper_csv=args.gates_hmm_proper_csv,
        output_dir=args.output_dir
    )