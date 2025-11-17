"""
Fill missing data in fredgraph.csv using forward fill method
"""

import pandas as pd

# print("Loading fredgraph.csv...")
# df = pd.read_csv('fredgraph.csv')

# print(f"Original data shape: {df.shape}")
# print(f"Missing values per column:")
# print(df.isnull().sum())

# # Convert observation_date to datetime
# df['observation_date'] = pd.to_datetime(df['observation_date'])

# # Forward fill missing values (use last known value)
# print("\nApplying forward fill to missing values...")
# df_filled = df.copy()

# # Forward fill all yield columns (all columns except observation_date)
# yield_columns = [col for col in df.columns if col != 'observation_date']
# df_filled[yield_columns] = df_filled[yield_columns].fillna(method='ffill')

# print(f"\nAfter forward fill:")
# print(f"Missing values per column:")
# print(df_filled.isnull().sum())

# # Save the filled data back to fredgraph.csv
# df_filled.to_csv('fredgraph.csv', index=False)

# print("\n Successfully saved filled data to fredgraph.csv")
# print(f"Date range: {df_filled['observation_date'].min()} to {df_filled['observation_date'].max()}")
# print(f"Total rows: {len(df_filled)}")

# Load each file
cu = pd.read_csv('capacity_util.csv', parse_dates=['observation_date'])
cu.columns = ['Date', 'CU']

ffr = pd.read_csv('fed_funds.csv', parse_dates=['observation_date'])
ffr.columns = ['Date', 'FFR']

cpi = pd.read_csv('cpi.csv', parse_dates=['observation_date'])
cpi.columns = ['Date', 'CPI']

# Merge
macro_df = cu.merge(ffr, on='Date').merge(cpi, on='Date')

# Calculate inflation (YoY %)
macro_df['INFL'] = macro_df['CPI'].pct_change(12) * 100
macro_df = macro_df.drop(columns=['CPI'])

# Save
macro_df.to_csv('us_macro_data.csv', index=False)
print("✓ Created us_macro_data.csv")