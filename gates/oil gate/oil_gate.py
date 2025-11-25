import os
from datetime import datetime
import numpy as np
import pandas as pd

OIL_DATA_FILE = "data/oil_prices.csv"
OUTPUT_DIR = os.path.join("output", "oil_signals")

THRESHOLD_HIGH = 0.02
THRESHOLD_LOW = -0.02
SMOOTH_DAYS = 5


def load_oil_prices(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Oil data file not found: {path}")
    
    df = pd.read_csv(path)
    
    # Find date column (consistent with your other scripts)
    date_col = "time" if "time" in df.columns else "date"
    
    if date_col not in df.columns:
        raise ValueError(f"No date column found. Expected 'date' or 'time'. Found: {list(df.columns)}")
    
    # Parse dates and set index
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    
    # Check for required columns
    required = ["near_month", "next_month"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    
    # Keep only required columns and convert to float
    oil = df[required].astype(float)
    
    # Remove duplicates (keep last) - same as your price loading
    oil = oil[~oil.index.duplicated(keep="last")]
    
    print(f"Loaded oil data: {len(oil)} rows from {oil.index.min().date()} to {oil.index.max().date()}")
    print(f"Missing values: near_month={oil['near_month'].isna().sum()}, next_month={oil['next_month'].isna().sum()}")
    
    return oil


def calculate_oil_slope(oil_df, smooth_days=SMOOTH_DAYS):
    # Raw slope
    raw_slope = (oil_df['next_month'] - oil_df['near_month']) / oil_df['near_month']
    
    # Smooth to reduce noise (similar to your signal smoothing)
    if smooth_days > 1:
        slope = raw_slope.rolling(smooth_days, min_periods=1).mean()
    else:
        slope = raw_slope
    
    return slope


def compute_gate_value(slope, threshold_high=THRESHOLD_HIGH, threshold_low=THRESHOLD_LOW):
    # Vectorized gate calculation
    gate = pd.Series(index=slope.index, dtype=float)
    
    # Full exposure when slope is high (contango)
    gate[slope >= threshold_high] = 1.0
    
    # Zero exposure when slope is very negative (strong backwardation)
    gate[slope <= threshold_low] = 0.0
    
    # Linear interpolation in between
    middle_mask = (slope > threshold_low) & (slope < threshold_high)
    gate[middle_mask] = (slope[middle_mask] - threshold_low) / (threshold_high - threshold_low)
    
    return gate


def generate_reason(slope, threshold_high=THRESHOLD_HIGH, threshold_low=THRESHOLD_LOW):
    reasons = pd.Series(index=slope.index, dtype=str)
    
    # Strong contango
    reasons[slope >= threshold_high] = "oil_contango_strong"
    
    # Moderate contango
    reasons[(slope >= 0) & (slope < threshold_high)] = "oil_contango_moderate"
    
    # Neutral
    reasons[(slope >= threshold_low) & (slope < 0)] = "oil_neutral"
    
    # Moderate backwardation
    reasons[(slope >= threshold_low) & (slope < threshold_low / 2)] = "oil_backwardation_moderate"
    
    # Strong backwardation
    reasons[slope < threshold_low] = "oil_backwardation_strong"
    
    return reasons


def build_oil_gate(oil_df, smooth_days=SMOOTH_DAYS, 
                   threshold_high=THRESHOLD_HIGH, threshold_low=THRESHOLD_LOW):
    # Calculate slope
    slope = calculate_oil_slope(oil_df, smooth_days)
    
    # Compute gate value
    gate_value = compute_gate_value(slope, threshold_high, threshold_low)
    
    # Generate reasons
    reason = generate_reason(slope, threshold_high, threshold_low)
    
    # Combine into output table
    result = pd.DataFrame({
        'gate_value': gate_value,
        'oil_slope': slope,
        'reason': reason
    })
    
    result.index.name = 'date'
    
    return result


def save_gate(gate_df, out_dir=OUTPUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    
    # Reset index to include date column
    output = gate_df.reset_index()
    
    # Add metadata (consistent with your signal files)
    output['gate_type'] = 'oil'
    output['calculation_date'] = datetime.now().strftime("%Y-%m-%d")
    
    # Reorder columns
    output = output[['date', 'gate_value', 'oil_slope', 'reason', 'gate_type', 'calculation_date']]
    
    # Generate file paths
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_timestamped = os.path.join(out_dir, f"gates_oil_{ts}.csv")
    path_latest = os.path.join(out_dir, "gates_oil_latest.csv")

    # Save files
    output.to_csv(path_timestamped, index=False)
    output.to_csv(path_latest, index=False)
    
    print("Saved:", path_timestamped)
    print("Saved:", path_latest)
    
    return output


def print_summary(gate_df):
    """Print summary statistics of the oil gate."""
    print("\n" + "="*60)
    print("OIL GATE SUMMARY")
    print("="*60)
    
    print(f"\nDate range: {gate_df.index.min().date()} to {gate_df.index.max().date()}")
    print(f"Total days: {len(gate_df)}")
    
    print(f"\nGate Value Statistics:")
    print(f"  Mean:   {gate_df['gate_value'].mean():.4f}")
    print(f"  Median: {gate_df['gate_value'].median():.4f}")
    print(f"  Std:    {gate_df['gate_value'].std():.4f}")
    print(f"  Min:    {gate_df['gate_value'].min():.4f}")
    print(f"  Max:    {gate_df['gate_value'].max():.4f}")
    
    print(f"\nOil Slope Statistics:")
    print(f"  Mean:   {gate_df['oil_slope'].mean():.4f}")
    print(f"  Median: {gate_df['oil_slope'].median():.4f}")
    print(f"  Std:    {gate_df['oil_slope'].std():.4f}")
    
    print(f"\nReason Distribution:")
    reason_counts = gate_df['reason'].value_counts().sort_index()
    for reason, count in reason_counts.items():
        pct = 100 * count / len(gate_df)
        print(f"  {reason:30s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\nDays with gate < 0.5 (reduced exposure): {(gate_df['gate_value'] < 0.5).sum()}")
    print(f"Days with gate = 0.0 (zero exposure):    {(gate_df['gate_value'] == 0.0).sum()}")
    print(f"Days with gate = 1.0 (full exposure):    {(gate_df['gate_value'] == 1.0).sum()}")
    print("="*60 + "\n")


def main():
    print("Starting Oil Gate Calculation...")
    print(f"Parameters: smooth_days={SMOOTH_DAYS}, threshold_high={THRESHOLD_HIGH}, threshold_low={THRESHOLD_LOW}\n")
    
    # Load oil prices
    oil = load_oil_prices(OIL_DATA_FILE)
    
    # Build gate
    gate = build_oil_gate(oil, SMOOTH_DAYS, THRESHOLD_HIGH, THRESHOLD_LOW)
    
    # Print summary
    print_summary(gate)
    
    # Save results
    output = save_gate(gate, OUTPUT_DIR)
    
    print(f"\nTotal rows exported: {len(output):,}")
    print("\nOil gate calculation complete!")
    
    return gate


if __name__ == "__main__":
    result = main()
