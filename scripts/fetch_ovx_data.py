"""
Fetch OVX Data from FRED or Yahoo Finance

Downloads CBOE Crude Oil Volatility Index (OVX) historical data.
Saves cleaned data with calculated features.

No API key required for FRED!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def fetch_ovx_from_fred(start_date: str = '2007-05-10') -> pd.DataFrame:
    """
    Fetch OVX data from FRED (Federal Reserve Economic Data).
    
    FREE - No API key required!
    
    Parameters:
    -----------
    start_date : str
        Start date for data (OVX began 2007-05-10)
    
    Returns:
    --------
    df : pd.DataFrame
        OVX data with columns: date, ovx_close
    """
    print("Fetching OVX from FRED...")
    
    # FRED direct download URL for OVX
    fred_url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        "id=OVXCLS"
        f"&cosd={start_date}"
    )
    
    try:
        # Download CSV directly from FRED (don't parse dates initially)
        df = pd.read_csv(fred_url)
        
        # Check what columns we got
        if 'DATE' not in df.columns and 'date' not in df.columns:
            # Try first column as date
            date_col = df.columns[0]
            value_col = df.columns[1]
        else:
            date_col = 'DATE' if 'DATE' in df.columns else 'date'
            value_col = 'OVXCLS' if 'OVXCLS' in df.columns else df.columns[1]
        
        # Rename columns
        df = df.rename(columns={
            date_col: 'date',
            value_col: 'ovx_close'
        })
        
        # Parse dates
        df['date'] = pd.to_datetime(df['date'])
        
        # Remove missing values (marked as '.')
        df['ovx_close'] = pd.to_numeric(df['ovx_close'], errors='coerce')
        df = df.dropna(subset=['ovx_close'])
        
        print(f"  ✓ Successfully fetched {len(df)} days from FRED")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    
    except Exception as e:
        print(f"  ✗ FRED fetch failed: {e}")
        return None


def fetch_ovx_from_yahoo(start_date: str = '2007-05-10') -> pd.DataFrame:
    """
    Fetch OVX data from Yahoo Finance (fallback).
    
    Parameters:
    -----------
    start_date : str
        Start date for data
    
    Returns:
    --------
    df : pd.DataFrame
        OVX data with columns: date, ovx_close
    """
    print("Fetching OVX from Yahoo Finance (fallback)...")
    
    try:
        import yfinance as yf
        
        # OVX ticker on Yahoo
        ticker = "^OVX"
        
        # Download data
        ovx = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)
        
        if len(ovx) == 0:
            print(f"  ✗ No data returned from Yahoo Finance")
            return None
        
        # Handle both single-level and multi-level column indices
        if isinstance(ovx.columns, pd.MultiIndex):
            # Multi-level columns (newer yfinance)
            close_col = ('Close', ticker)
            if close_col not in ovx.columns:
                close_col = ovx.columns[ovx.columns.get_level_values(0) == 'Close'][0]
            close_data = ovx[close_col]
        else:
            # Single-level columns (older yfinance or single ticker)
            close_data = ovx['Close']
        
        # Prepare dataframe
        df = pd.DataFrame({
            'date': ovx.index,
            'ovx_close': close_data.values
        })
        
        df = df.reset_index(drop=True)
        df = df.dropna(subset=['ovx_close'])
        
        print(f"  ✓ Successfully fetched {len(df)} days from Yahoo Finance")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    
    except ImportError:
        print("  ✗ yfinance not installed. Install with: pip install yfinance")
        return None
    except Exception as e:
        print(f"  ✗ Yahoo Finance fetch failed: {e}")
        return None


def fetch_ovx_from_pandas_datareader(start_date: str = '2007-05-10') -> pd.DataFrame:
    """
    Fetch OVX data using pandas_datareader (another fallback).
    
    Parameters:
    -----------
    start_date : str
        Start date for data
    
    Returns:
    --------
    df : pd.DataFrame
        OVX data with columns: date, ovx_close
    """
    print("Fetching OVX from FRED via pandas_datareader (alternative method)...")
    
    try:
        from pandas_datareader import data as pdr
        
        # Fetch from FRED using pandas_datareader
        ovx = pdr.DataReader('OVXCLS', 'fred', start=start_date)
        
        if len(ovx) == 0:
            print(f"  ✗ No data returned")
            return None
        
        # Prepare dataframe
        df = pd.DataFrame({
            'date': ovx.index,
            'ovx_close': ovx['OVXCLS'].values
        })
        
        df = df.reset_index(drop=True)
        df = df.dropna(subset=['ovx_close'])
        
        print(f"  ✓ Successfully fetched {len(df)} days")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        return df
    
    except ImportError:
        print("  ✗ pandas_datareader not installed. Install with: pip install pandas_datareader")
        return None
    except Exception as e:
        print(f"  ✗ pandas_datareader fetch failed: {e}")
        return None


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional features from raw OVX data.
    
    Features:
    - Daily change
    - 5-day moving average
    - 20-day moving average
    - 20-day rolling std
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    # Daily change
    df['ovx_change'] = df['ovx_close'].diff()
    
    # 5-day MA (for smoothing)
    df['ovx_ma5'] = df['ovx_close'].rolling(window=5, min_periods=1).mean()
    
    # 20-day MA (for longer trend)
    df['ovx_ma20'] = df['ovx_close'].rolling(window=20, min_periods=1).mean()
    
    # 20-day rolling std (for volatility of volatility)
    df['ovx_std20'] = df['ovx_close'].rolling(window=20, min_periods=1).std()
    
    # Fill initial NaNs
    df = df.fillna(method='bfill')
    
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate OVX data quality.
    
    Checks:
    - Reasonable range (10-200 typical)
    - No excessive gaps
    - Sufficient history
    """
    print("\nValidating data quality...")
    
    # Check range
    min_ovx = df['ovx_close'].min()
    max_ovx = df['ovx_close'].max()
    
    if min_ovx < 5 or max_ovx > 500:
        print(f"  ⚠ Warning: OVX range unusual [{min_ovx:.1f}, {max_ovx:.1f}]")
        print(f"    Expected range: [10, 200]")
    else:
        print(f"  ✓ OVX range looks good: [{min_ovx:.1f}, {max_ovx:.1f}]")
    
    # Check for gaps
    df = df.sort_values('date')
    date_diffs = df['date'].diff()
    
    # Allow for weekends (max 3 days gap for weekdays)
    large_gaps = date_diffs[date_diffs > pd.Timedelta(days=5)]
    
    if len(large_gaps) > 10:
        print(f"  ⚠ Warning: {len(large_gaps)} large gaps found in data")
    else:
        print(f"  ✓ No major gaps in data")
    
    # Check history length
    years = (df['date'].max() - df['date'].min()).days / 365.25
    
    if years < 5:
        print(f"  ⚠ Warning: Only {years:.1f} years of data")
        print(f"    Recommended: 5+ years")
    else:
        print(f"  ✓ Sufficient history: {years:.1f} years")
    
    # Check for missing values
    missing = df['ovx_close'].isna().sum()
    
    if missing > 0:
        print(f"  ⚠ Warning: {missing} missing values")
        return False
    else:
        print(f"  ✓ No missing values")
    
    return True


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of OVX data."""
    print("\n" + "="*70)
    print("OVX DATA SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nDate Range:")
    print(f"  Start: {df['date'].min().date()}")
    print(f"  End:   {df['date'].max().date()}")
    print(f"  Days:  {len(df):,}")
    print(f"  Years: {(df['date'].max() - df['date'].min()).days / 365.25:.1f}")
    
    print(f"\nOVX Level Statistics:")
    print(f"  Mean:   {df['ovx_close'].mean():.2f}")
    print(f"  Median: {df['ovx_close'].median():.2f}")
    print(f"  Std:    {df['ovx_close'].std():.2f}")
    print(f"  Min:    {df['ovx_close'].min():.2f}")
    print(f"  Max:    {df['ovx_close'].max():.2f}")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90]:
        val = df['ovx_close'].quantile(p/100)
        print(f"  {p}th: {val:.2f}")
    
    # Regime breakdown
    print(f"\nRegime Breakdown:")
    calm = (df['ovx_close'] < 25).sum()
    normal = ((df['ovx_close'] >= 25) & (df['ovx_close'] < 40)).sum()
    elevated = ((df['ovx_close'] >= 40) & (df['ovx_close'] < 60)).sum()
    extreme = (df['ovx_close'] >= 60).sum()
    
    print(f"  Very Low (<25):   {calm:5d} days ({calm/len(df)*100:5.1f}%)")
    print(f"  Normal (25-40):   {normal:5d} days ({normal/len(df)*100:5.1f}%)")
    print(f"  Elevated (40-60): {elevated:5d} days ({elevated/len(df)*100:5.1f}%)")
    print(f"  Extreme (60+):    {extreme:5d} days ({extreme/len(df)*100:5.1f}%)")
    
    # Notable periods
    print(f"\nNotable High Periods (OVX > 60):")
    high_periods = df[df['ovx_close'] > 60].groupby(
        (df['ovx_close'] <= 60).cumsum()
    )
    
    for _, group in list(high_periods)[:5]:  # Show first 5
        if len(group) > 5:  # Only show sustained periods
            start = group['date'].min().date()
            end = group['date'].max().date()
            max_ovx = group['ovx_close'].max()
            print(f"  {start} to {end}: Peak OVX = {max_ovx:.1f}")


def main(
    output_csv: str = "data/ovx_daily.csv",
    start_date: str = "2007-05-10",
    force_yahoo: bool = False
):
    """
    Main function to fetch and process OVX data.
    
    Parameters:
    -----------
    output_csv : str
        Path to save output CSV
    start_date : str
        Start date for data
    force_yahoo : bool
        Force Yahoo Finance (skip FRED)
    """
    print("="*70)
    print("FETCHING OVX DATA")
    print("="*70)
    print(f"\nStart date: {start_date}")
    print(f"Output: {output_csv}")
    
    # Attempt to fetch data
    df = None
    
    if not force_yahoo:
        # Try FRED first (no API key needed)
        df = fetch_ovx_from_fred(start_date)
    
    if df is None:
        # Try pandas_datareader
        df = fetch_ovx_from_pandas_datareader(start_date)
    
    if df is None:
        # Fallback to Yahoo Finance
        df = fetch_ovx_from_yahoo(start_date)
    
    if df is None:
        print("\n" + "="*70)
        print("✗ FAILED TO FETCH OVX DATA")
        print("="*70)
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Install alternative libraries:")
        print("   pip install yfinance")
        print("   pip install pandas_datareader")
        print("3. Try Yahoo Finance: --force_yahoo flag")
        print("4. Manually download from: https://fred.stlouisfed.org/series/OVXCLS")
        print("   Save as data/ovx_daily.csv with columns: date,ovx_close")
        return
    
    # Calculate additional features
    print("\nCalculating features...")
    df = calculate_features(df)
    print("  ✓ Added: ovx_change, ovx_ma5, ovx_ma20, ovx_std20")
    
    # Validate data
    is_valid = validate_data(df)
    
    if not is_valid:
        print("\n⚠ Data validation found issues (but continuing anyway)")
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print(f"✓ OVX DATA SAVED SUCCESSFULLY")
    print("="*70)
    print(f"\nFile: {output_csv}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print("\nReady to use in OVX gate calculations!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch OVX data from FRED or Yahoo Finance"
    )
    parser.add_argument(
        '--output_csv',
        default='data/ovx_daily.csv',
        help='Path to save OVX data CSV (default: data/ovx_daily.csv)'
    )
    parser.add_argument(
        '--start_date',
        default='2007-05-10',
        help='Start date for data (default: 2007-05-10, OVX inception)'
    )
    parser.add_argument(
        '--force_yahoo',
        action='store_true',
        help='Force Yahoo Finance (skip FRED)'
    )
    
    args = parser.parse_args()
    
    main(
        output_csv=args.output_csv,
        start_date=args.start_date,
        force_yahoo=args.force_yahoo
    )