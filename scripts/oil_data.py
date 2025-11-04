import pandas as pd
import os
from datetime import datetime

def fetch_from_fred():
    print("Fetching WTI Crude Oil prices from FRED...")
    
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO"
    
    try:
        df = pd.read_csv(url)
        df.columns = ['time', 'near_month']
        df['time'] = pd.to_datetime(df['time'])
        df['near_month'] = pd.to_numeric(df['near_month'], errors='coerce')
        df = df.dropna()
        
        df = df[df['time'] >= '2003-01-01']
        
        df['next_month'] = df['near_month'].shift(-21)
        avg_spread = (df['next_month'] - df['near_month']).mean()
        df['next_month'].fillna(df['near_month'] + avg_spread, inplace=True)
        
        df = df.dropna().reset_index(drop=True)
        
        print(f"Fetched {len(df)} days of data")
        print(f"Date range: {df['time'].min().date()} to {df['time'].max().date()}")
        print(f"Near month price: ${df['near_month'].mean():.2f} ± ${df['near_month'].std():.2f}")
        
        df['slope'] = (df['next_month'] - df['near_month']) / df['near_month']
        print(f"Mean slope: {df['slope'].mean():.4f}")
        
        os.makedirs('data', exist_ok=True)
        output_path = 'data/oil_prices.csv'
        df[['time', 'near_month', 'next_month']].to_csv(output_path, index=False)
        
        print(f"\nSaved to: {output_path}")
        print("\nOil data ready! Run:")
        print("  python scripts/oil_gate.py")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    fetch_from_fred()