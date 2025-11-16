# Quick script to create validation returns
import pandas as pd

# Load prices and weights
prices = pd.read_csv('data/prices_daily.csv', index_col=0, parse_dates=True)
weights = pd.read_csv('output/signals/weights_after_caps.csv', index_col=0, parse_dates=True)

# Calculate returns
asset_returns = prices.pct_change()
portfolio_returns = (weights * asset_returns).sum(axis=1)

# Save validation period
val_returns = portfolio_returns.loc['2016-01-01':'2018-12-31']
val_returns.to_csv('output/validation_returns.csv')

print(f"Saved {len(val_returns)} days of validation returns")