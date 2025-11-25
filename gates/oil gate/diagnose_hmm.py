"""
Diagnose HMM Issues Without Loading Pickle

Just analyze the gates CSV to see what went wrong.
"""

import pandas as pd
import numpy as np

print("="*70)
print("HMM DIAGNOSTIC ANALYSIS")
print("="*70)

# Load HMM gates
print("\nLoading HMM gates...")
hmm_gates = pd.read_csv('output/oil_signals/gates_ovx_hmm.csv')

# Convert date column
date_col = 'time' if 'time' in hmm_gates.columns else 'date'
hmm_gates[date_col] = pd.to_datetime(hmm_gates[date_col])
hmm_gates = hmm_gates.set_index(date_col)

print(f"Loaded {len(hmm_gates)} days of gates")

# Load formula gates for comparison
print("\nLoading formula gates for comparison...")
formula_gates = pd.read_csv('output/oil_signals/gates_ovx_formula.csv')
date_col = 'time' if 'time' in formula_gates.columns else 'date'
formula_gates[date_col] = pd.to_datetime(formula_gates[date_col])
formula_gates = formula_gates.set_index(date_col)

print("\n" + "="*70)
print("GATE VARIABILITY COMPARISON")
print("="*70)

# Conservative gates
print("\nConservative Gates (No Scale-Up):")
print(f"  Formula:")
print(f"    Mean:  {formula_gates['gate_formula_no_scaleup'].mean():.4f}")
print(f"    Std:   {formula_gates['gate_formula_no_scaleup'].std():.4f}")
print(f"    Min:   {formula_gates['gate_formula_no_scaleup'].min():.4f}")
print(f"    Max:   {formula_gates['gate_formula_no_scaleup'].max():.4f}")
print(f"    Range: {formula_gates['gate_formula_no_scaleup'].max() - formula_gates['gate_formula_no_scaleup'].min():.4f}")

print(f"\n  HMM:")
print(f"    Mean:  {hmm_gates['gate_hmm_no_scaleup'].mean():.4f}")
print(f"    Std:   {hmm_gates['gate_hmm_no_scaleup'].std():.4f}")
print(f"    Min:   {hmm_gates['gate_hmm_no_scaleup'].min():.4f}")
print(f"    Max:   {hmm_gates['gate_hmm_no_scaleup'].max():.4f}")
print(f"    Range: {hmm_gates['gate_hmm_no_scaleup'].max() - hmm_gates['gate_hmm_no_scaleup'].min():.4f}")

# Aggressive gates
print("\nAggressive Gates (With Scale-Up):")
print(f"  Formula:")
print(f"    Mean:  {formula_gates['gate_formula_with_scaleup'].mean():.4f}")
print(f"    Std:   {formula_gates['gate_formula_with_scaleup'].std():.4f}")
print(f"    Min:   {formula_gates['gate_formula_with_scaleup'].min():.4f}")
print(f"    Max:   {formula_gates['gate_formula_with_scaleup'].max():.4f}")
print(f"    Range: {formula_gates['gate_formula_with_scaleup'].max() - formula_gates['gate_formula_with_scaleup'].min():.4f}")

print(f"\n  HMM:")
print(f"    Mean:  {hmm_gates['gate_hmm_with_scaleup'].mean():.4f}")
print(f"    Std:   {hmm_gates['gate_hmm_with_scaleup'].std():.4f}")
print(f"    Min:   {hmm_gates['gate_hmm_with_scaleup'].min():.4f}")
print(f"    Max:   {hmm_gates['gate_hmm_with_scaleup'].max():.4f}")
print(f"    Range: {hmm_gates['gate_hmm_with_scaleup'].max() - hmm_gates['gate_hmm_with_scaleup'].min():.4f}")

# Regime analysis
if 'state' in hmm_gates.columns:
    print("\n" + "="*70)
    print("REGIME DISTRIBUTION")
    print("="*70)
    
    regime_counts = hmm_gates['state'].value_counts().sort_index()
    total = len(hmm_gates)
    
    print("\nState distribution:")
    for state, count in regime_counts.items():
        print(f"  State {state}: {count:5d} days ({count/total*100:5.1f}%)")
    
    # Check if any state dominates
    dominant_pct = regime_counts.max() / total * 100
    if dominant_pct > 60:
        print(f"\n⚠️  WARNING: State {regime_counts.idxmax()} dominates ({dominant_pct:.1f}%)")
        print("   HMM collapsed to mostly one state")

# Regime probabilities
print("\n" + "="*70)
print("REGIME PROBABILITY ANALYSIS")
print("="*70)

prob_cols = [col for col in hmm_gates.columns if col.startswith('prob_')]

if prob_cols:
    print("\nMean regime probabilities:")
    for col in prob_cols:
        regime_name = col.replace('prob_', '')
        mean_prob = hmm_gates[col].mean()
        std_prob = hmm_gates[col].std()
        print(f"  {regime_name:10s}: {mean_prob:.3f} (std: {std_prob:.3f})")
    
    # Check if probabilities are too uniform
    if len(prob_cols) >= 2:
        prob_data = hmm_gates[prob_cols]
        max_probs = prob_data.max(axis=1)
        mean_max_prob = max_probs.mean()
        
        print(f"\nMean max probability: {mean_max_prob:.3f}")
        
        if mean_max_prob < 0.6:
            print("⚠️  WARNING: Low regime confidence (mean max < 0.6)")
            print("   Posteriors are blending regimes too much")

# Check correlation with OVX
print("\n" + "="*70)
print("CORRELATION WITH OVX")
print("="*70)

if 'ovx_level' in hmm_gates.columns:
    corr_formula_cons = formula_gates['gate_formula_no_scaleup'].corr(formula_gates['ovx_level'])
    corr_hmm_cons = hmm_gates['gate_hmm_no_scaleup'].corr(hmm_gates['ovx_level'])
    
    print("\nCorrelation between gate and OVX level:")
    print(f"  Formula: {corr_formula_cons:.3f}")
    print(f"  HMM:     {corr_hmm_cons:.3f}")
    
    if abs(corr_hmm_cons) < 0.3:
        print("\n⚠️  WARNING: HMM gates barely respond to OVX changes")

# Crisis period analysis
print("\n" + "="*70)
print("CRISIS PERIOD BEHAVIOR")
print("="*70)

crisis_periods = [
    ('2008 Financial Crisis', '2008-09-01', '2008-12-31'),
    ('2015-2016 Oil Crash', '2015-12-01', '2016-03-31'),
    ('2020 COVID', '2020-02-01', '2020-04-30'),
    ('2022 Ukraine War', '2022-02-15', '2022-04-15')
]

for name, start, end in crisis_periods:
    try:
        # Formula
        formula_mask = (formula_gates.index >= start) & (formula_gates.index <= end)
        formula_crisis = formula_gates.loc[formula_mask, 'gate_formula_no_scaleup']
        
        # HMM
        hmm_mask = (hmm_gates.index >= start) & (hmm_gates.index <= end)
        hmm_crisis = hmm_gates.loc[hmm_mask, 'gate_hmm_no_scaleup']
        
        if len(formula_crisis) > 0 and len(hmm_crisis) > 0:
            print(f"\n{name}:")
            print(f"  Formula mean gate: {formula_crisis.mean():.3f} (min: {formula_crisis.min():.3f})")
            print(f"  HMM mean gate:     {hmm_crisis.mean():.3f} (min: {hmm_crisis.min():.3f})")
            
            if hmm_crisis.mean() > 0.9:
                print(f"  ⚠️  HMM barely derisked during crisis")
    except:
        pass

# Summary
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

issues = []

# Check 1: Low variability
if hmm_gates['gate_hmm_no_scaleup'].std() < 0.10:
    issues.append("✗ HMM gates have very low variability (std < 0.10)")

# Check 2: Stuck near 1.0
if hmm_gates['gate_hmm_no_scaleup'].mean() > 0.90 and hmm_gates['gate_hmm_no_scaleup'].mean() < 1.05:
    issues.append("✗ HMM gates stuck near 1.0 (mean ~0.95-1.0)")

# Check 3: Small range
gate_range = hmm_gates['gate_hmm_no_scaleup'].max() - hmm_gates['gate_hmm_no_scaleup'].min()
if gate_range < 0.3:
    issues.append(f"✗ HMM gate range too small ({gate_range:.3f} < 0.3)")

# Check 4: Conservative and aggressive are similar
if abs(hmm_gates['gate_hmm_no_scaleup'].mean() - hmm_gates['gate_hmm_with_scaleup'].mean()) < 0.05:
    issues.append("✗ Conservative and aggressive gates are too similar")

if issues:
    print("\nProblems found:")
    for issue in issues:
        print(f"  {issue}")
    
    print("\nRecommendation:")
    print("  → Try Fix #1: Hard Regime Assignment (ovx_gate_hmm_fixed.py)")
    print("  → Or Fix #4: Threshold-Based Regimes (simpler, might work better)")
else:
    print("\n✓ HMM gates look reasonable!")
    print("  The issue might be elsewhere (e.g., weighting, backtest logic)")

print("\n" + "="*70)