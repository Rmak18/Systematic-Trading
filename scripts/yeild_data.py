import pandas as pd
import os
from datetime import datetime

def fetch_yield_data():
    print("Fetching US Treasury Yield data from FRED...")
    
    url_10y = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
    url_2y = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2"
    
    try:
        df_10y = pd.read_csv(url_10y)
        df_10y.columns = ['time', 'yield_10y']
        
        df_2y = pd.read_csv(url_2y)
        df_2y.columns = ['time', 'yield_2y']
        
        df_10y['time'] = pd.to_datetime(df_10y['time'])
        df_2y['time'] = pd.to_datetime(df_2y['time'])
        
        yields = pd.merge(df_10y, df_2y, on='time', how='inner')
        
        yields['yield_10y'] = pd.to_numeric(yields['yield_10y'], errors='coerce')
        yields['yield_2y'] = pd.to_numeric(yields['yield_2y'], errors='coerce')
        
        yields = yields.dropna()
        yields = yields[yields['time'] >= '2003-01-01']
        yields = yields.sort_values('time').reset_index(drop=True)
        
        yields['spread'] = yields['yield_10y'] - yields['yield_2y']
        
        print(f"Fetched {len(yields)} days of data")
        print(f"Date range: {yields['time'].min().date()} to {yields['time'].max().date()}")
        print(f"\n10-Year Yield Statistics:")
        print(f"  Mean:   {yields['yield_10y'].mean():.2f}%")
        print(f"  Median: {yields['yield_10y'].median():.2f}%")
        print(f"  Min:    {yields['yield_10y'].min():.2f}%")
        print(f"  Max:    {yields['yield_10y'].max():.2f}%")
        
        print(f"\n2-Year Yield Statistics:")
        print(f"  Mean:   {yields['yield_2y'].mean():.2f}%")
        print(f"  Median: {yields['yield_2y'].median():.2f}%")
        print(f"  Min:    {yields['yield_2y'].min():.2f}%")
        print(f"  Max:    {yields['yield_2y'].max():.2f}%")
        
        print(f"\nYield Curve Spread (10Y - 2Y):")
        print(f"  Mean:   {yields['spread'].mean():.2f}%")
        print(f"  Median: {yields['spread'].median():.2f}%")
        print(f"  Min:    {yields['spread'].min():.2f}% (most inverted)")
        print(f"  Max:    {yields['spread'].max():.2f}% (steepest)")
        
        inverted_days = (yields['spread'] < 0).sum()
        print(f"\nInverted curve days (spread < 0): {inverted_days} ({100*inverted_days/len(yields):.1f}%)")
        
        os.makedirs('data', exist_ok=True)
        output_path = 'data/yield_curve.csv'
        yields[['time', 'yield_10y', 'yield_2y']].to_csv(output_path, index=False)
        
        print(f"\nSaved to: {output_path}")
        print("\nYield curve data ready! You can now build the yield gate.")
        
        return yields
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    fetch_yield_data()