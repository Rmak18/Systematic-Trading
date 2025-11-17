# State Gate Model: Complete Implementation Guide

## Overview

The State Gate model is a machine learning approach to **regime-dependent yield curve modeling** from the paper "Machine Learning and the Yield Curve: Tree-Based Macroeconomic Regime Switching" (Bie, He, Diebold, Li, 2025).

**What it does:**
- Identifies distinct macroeconomic regimes based on observable variables (Federal Funds Rate, Inflation, Capacity Utilization)
- Characterizes how the yield curve behaves differently in each regime
- Provides regime-specific volatility estimates for position sizing in trading strategies

**Key Innovation:**
Unlike traditional hidden Markov regime-switching models, State Gate regimes are:
- **Observable**: Based on current macro data, not latent states
- **Interpretable**: Clear economic meaning (e.g., "high inflation, low rates")
- **Actionable**: Can be used in real-time for strategy adjustments

---

## How the Model Works

### Step 1: Extract Yield Curve Factors

The model uses the **Dynamic Nelson-Siegel (DNS)** framework to represent the entire yield curve with just 3 factors:

```
y(τ) = Level + Slope × f₁(τ) + Curvature × f₂(τ)
```

Where:
- **Level**: Long-term interest rate level
- **Slope**: Short-term vs long-term rate difference
- **Curvature**: Medium-term bulge in the curve

**Input:** Treasury yields at multiple maturities (3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y)
**Output:** Monthly time series of (Level, Slope, Curvature)

### Step 2: Build Decision Tree on Macro Variables

The algorithm builds a **regression tree** that partitions time periods based on macroeconomic variables:

1. **Candidate splits**: Test thresholds on FFR, INFL, CU (e.g., "FFR ≤ 60th percentile?")
2. **Evaluation criterion**: Marginal likelihood of the DNS model
   - Better splits = yield factors have more distinct dynamics in each partition
3. **Greedy selection**: Pick the split that maximizes likelihood improvement
4. **Stopping rule**: Stop when max_leaves reached or improvements too small

**Result:** A tree structure like:
```
                FFR ≤ 0.6?
               /          \
          Yes /            \ No
             /              \
      INFL ≤ 0.6?        Regime 0
        /      \         (High FFR)
    Yes/        \No
      /          \
 Regime 1     Regime 2
(Low FFR,    (Low FFR,
 Low Infl)    High Infl)
```

### Step 3: Estimate Regime-Specific Parameters

For each regime g, the model estimates:

- **Transition matrix Aₐ**: How factors evolve over time
- **Covariance matrix Hₐ**: Volatility of factor innovations
- **Mean μₐ**: Average factor levels

**Key finding from the paper:**
- Regime 1 (high FFR): Macro variables predict yield factors (macro-spanning fails)
- Regimes 2 & 3: Macro variables don't predict yields (macro-spanning holds)

### Step 4: Calculate Regime-Specific Scales

Based on regime volatilities, position sizing scales are computed:

```
Scale_regime = mean_volatility / regime_volatility
```

Clipped to [0.3, 1.5] for safety.

**Interpretation:**
- **High volatility regime** → **Low scale** → Reduce positions
- **Low volatility regime** → **High scale** → Increase positions

---

## Data Requirements

### 1. Treasury Yields Data (`us_treasury_yields.csv`)

**Format:**
```csv
Date,3M,6M,1Y,2Y,3Y,5Y,7Y,10Y,20Y,30Y
2003-01-01,1.22,1.25,1.42,1.80,2.22,3.05,3.62,4.07,5.05,5.18
2003-02-01,1.20,1.19,1.24,1.53,1.91,2.69,3.24,3.71,4.70,4.90
...
```

**Requirements:**
- Monthly frequency (month-end or month-start, must match macro data)
- At least 3 maturities required
- No missing values (use forward fill if needed)

**How to obtain:**
Download from FRED:
```
https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO,DGS6MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10,DGS20,DGS30
```

### 2. Macro Data (`us_macro_data.csv`)

**Format:**
```csv
Date,CU,FFR,INFL
2003-01-01,75.9737,1.24,2.757456387169399
2003-02-01,76.0964,1.26,3.146067415730336
...
```

**Variables:**
- **CU**: Capacity Utilization (% of total capacity in use)
- **FFR**: Federal Funds Rate (%)
- **INFL**: Year-over-year CPI inflation (%)

**Requirements:**
- Monthly frequency
- Dates must align with yields data
- No missing values

**How to obtain:**
Download from FRED:
- Capacity Utilization: `https://fred.stlouisfed.org/series/TCU`
- Federal Funds Rate: `https://fred.stlouisfed.org/series/FEDFUNDS`
- CPI: `https://fred.stlouisfed.org/series/CPIAUCSL` (then calculate YoY % change)

---

## Installation & Setup

### Prerequisites

```bash
# Python 3.8+
python3 --version

# Required packages
pip install pandas numpy scipy matplotlib
```

### File Structure

```
your_project/
├── state_gate.py              # Main model code
├── us_treasury_yields.csv     # Yields data (monthly)
├── us_macro_data.csv          # Macro data (monthly)
├── merge_data.py              # Data preprocessing (optional)
└── state gate research.pdf    # Original paper
```

---

## Running the Model

### Basic Usage

```bash
python3 state_gate.py \
    --yields_csv us_treasury_yields.csv \
    --macro_csv us_macro_data.csv \
    --max_leaves 3 \
    --min_rel_improve 0.005 \
    --out_regimes_csv regimes.csv \
    --plot_png regimes_plot.png
```

### Parameters Explained

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--yields_csv` | Path to yields data | Required | Your yields file |
| `--macro_csv` | Path to macro data | Optional | Your macro file |
| `--max_leaves` | Number of regimes | 3 | 3 (per paper) |
| `--min_rel_improve` | Min improvement to split | 0.02 | 0.005-0.01 for more splits |
| `--lam` | NS decay parameter | 0.0609 | 0.0609 (standard) |
| `--out_regimes_csv` | Output regime assignments | regimes.csv | Your choice |
| `--plot_png` | Output plot | regimes_plot.png | Your choice |

### Advanced Parameters

**To force more/fewer splits:**
- **More regimes**: Lower `min_rel_improve` (e.g., 0.001)
- **Fewer regimes**: Raise `min_rel_improve` (e.g., 0.05)

**To use different macro variables:**
Edit the macro CSV to include your variables, model will automatically use all numeric columns.

---

## Understanding the Output

### 1. Console Output

```
TSMOM Scale: 0.2710

============================================================
REGIME SUMMARY
============================================================
Tree structure: {
  "leaf": false,
  "feature": "2Y",
  "threshold": 0.424,
  "left": {...},
  "right": {...}
}

Regime distribution:
  Regime 0: 55 months (20.2%) | Vol: 5.9513 | Scale: 1.500
  Regime 1: 173 months (63.6%) | Vol: 11.0812 | Scale: 1.017
  Regime 2: 44 months (16.2%) | Vol: 16.7814 | Scale: 0.672
============================================================
```

**Interpretation:**
- **TSMOM Scale**: Global scale (ignore this, use regime-specific scales)
- **Tree structure**: Shows how regimes are defined
- **Regime distribution**:
  - Count and % of months in each regime
  - **Vol**: RMS volatility of NS factors in that regime
  - **Scale**: Position sizing multiplier (inverse of relative volatility)

### 2. Regime Assignments (`regimes.csv`)

```csv
Date,Regime,Scale
2003-01-01,1,1.017
2003-02-01,1,1.017
2003-03-01,1,1.017
2008-10-01,2,0.672
2008-11-01,2,0.672
2022-06-01,0,1.500
...
```

**How to use:**
```python
import pandas as pd

regimes = pd.read_csv('regimes.csv', parse_dates=['Date'])
regimes['Date'] = pd.to_datetime(regimes['Date'])

# Merge with your trading signals
strategy = strategy.merge(regimes, on='Date')

# Adjust position size
strategy['position'] = strategy['base_position'] * strategy['Scale']
```

### 3. Visualization (`regimes_plot.png`)

Shows:
- 10Y yield over time
- Background shading by regime
- Clear visual separation of market conditions

---

## Practical Application: TSMOM Strategy

### Integration Example

```python
import pandas as pd
import numpy as np

# 1. Load your TSMOM signals
signals = pd.read_csv('tsmom_signals.csv', parse_dates=['Date'])
# Assume: Date, Signal (-1, 0, +1), BaseSize

# 2. Load regime-based scales
regimes = pd.read_csv('regimes.csv', parse_dates=['Date'])

# 3. Merge
df = signals.merge(regimes, on='Date', how='left')

# 4. Apply regime-adjusted sizing
df['AdjustedSize'] = df['BaseSize'] * df['Scale']
df['FinalPosition'] = df['Signal'] * df['AdjustedSize']

# Example:
# Regime 0 (stable, high FFR): Scale 1.5 → Size up 50%
# Regime 1 (normal): Scale 1.0 → Normal sizing
# Regime 2 (volatile, high infl): Scale 0.67 → Size down 33%
```

### Backtesting Considerations

1. **Regime persistence**: Regimes can last several months - don't rebalance daily
2. **Transition costs**: Account for rebalancing when regimes change
3. **Look-ahead bias**: Ensure you only use macro data available at month-end
4. **Out-of-sample**: Test on recent data not used in regime estimation

---

## Troubleshooting

### Common Issues

**Error: "No matching dates"**
- Check that Date columns in yields and macro CSVs have same format
- Both should be month-end OR month-start (not mixed)
- Use: `df['Date'] = pd.to_datetime(df['Date']).dt.to_period('M').dt.to_timestamp()`

**Error: "Not enough data in regime"**
- Model requires min 24 months per regime
- Use longer data sample (10+ years recommended)
- Or increase `min_rel_improve` to get fewer, larger regimes

**Error: "Tree not splitting"**
- `min_rel_improve` too high - lower it (try 0.005)
- Data may not have clear regimes - check macro data varies enough

**Scales seem off**
- Scales are relative within your sample
- Should be in range [0.3, 1.5] by design
- If all ~1.0, regimes have similar volatility

---

## References

**Original Paper:**
Bie, S., He, J., Diebold, F. X., & Li, J. (2025). "Machine Learning and the Yield Curve: Tree-Based Macroeconomic Regime Switching." arXiv:2408.12863v2

**Key Citations:**
- Nelson & Siegel (1987): Parsimonious yield curve modeling
- Diebold & Li (2006): Dynamic Nelson-Siegel model
- Breiman et al. (1984): CART decision trees

**Data Sources:**
- FRED (Federal Reserve Economic Data): https://fred.stlouisfed.org/
- Liu & Wu (2021): Treasury yield curve data

---

## Quick Start Checklist

- [ ] Install Python 3.8+ and required packages
- [ ] Download treasury yields from FRED
- [ ] Download macro data (CU, FFR, CPI)
- [ ] Ensure both CSVs are monthly and dates align
- [ ] Forward-fill any missing values
- [ ] Run state_gate.py with basic parameters
- [ ] Check output: 3 regimes with distinct volatilities
- [ ] Merge regimes.csv with your trading signals
- [ ] Backtest with regime-adjusted position sizing

---

## Contact & Support

For questions about the methodology, refer to the original paper.

For implementation issues, check:
1. Data format matches examples above
2. Dates are properly aligned
3. No missing values in critical variables (FFR, INFL)

