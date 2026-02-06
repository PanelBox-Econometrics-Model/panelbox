# How to Load Your Own Panel Data

> Step-by-step guide for preparing and loading your panel datasets into PanelBox.

## Overview

PanelBox works with pandas DataFrames that have:
- An **entity column** (firm ID, country code, individual ID, etc.)
- A **time column** (year, quarter, date, etc.)
- One or more **variable columns** (dependent and independent variables)

This guide shows you how to prepare your data from various sources.

## Quick Start

**Minimal example:**

```python
import pandas as pd
import panelbox as pb

# Load your CSV
data = pd.read_csv('my_panel_data.csv')

# Ensure proper column names
# data should have: entity_id, time, y, x1, x2, ...

# Estimate a model
model = pb.FixedEffects(
    formula="y ~ x1 + x2",
    data=data,
    entity_col="entity_id",  # Your entity column name
    time_col="time"          # Your time column name
)

results = model.fit()
print(results.summary())
```

## Data Format Requirements

### 1. Long Format (Required)

PanelBox requires **long format** where each row is an entity-time observation:

✅ **Correct (Long Format):**
```
   firm  year   sales  assets
0     1  2020   100.0   500.0
1     1  2021   120.0   520.0
2     1  2022   135.0   540.0
3     2  2020    80.0   300.0
4     2  2021    85.0   310.0
5     2  2022    90.0   320.0
```

❌ **Incorrect (Wide Format):**
```
   firm  sales_2020  sales_2021  sales_2022
0     1       100.0       120.0       135.0
1     2        80.0        85.0        90.0
```

**Convert wide to long:**

```python
# If your data is in wide format
data_wide = pd.read_csv('data_wide.csv')

# Convert to long
data_long = pd.melt(
    data_wide,
    id_vars=['firm'],
    value_vars=['sales_2020', 'sales_2021', 'sales_2022'],
    var_name='year',
    value_name='sales'
)

# Clean year column
data_long['year'] = data_long['year'].str.extract('(\d+)').astype(int)

print(data_long.head())
```

### 2. Column Types

**Entity column:**
- Can be `int`, `str`, or `categorical`
- Examples: firm ID (1, 2, 3), country code ('USA', 'CAN'), CUSIP

**Time column:**
- Can be `int`, `datetime`, or `period`
- Examples: year (2020, 2021), date ('2020-01-01'), quarter ('2020Q1')

**Variable columns:**
- Must be numeric (`float` or `int`)
- Missing values: Use `NaN` (PanelBox handles them automatically)

**Check types:**

```python
print(data.dtypes)

# entity_id      int64
# time           int64
# y            float64
# x1           float64
# x2           float64
```

**Fix types if needed:**

```python
# Convert entity to integer
data['entity_id'] = data['entity_id'].astype(int)

# Convert time to integer (if year)
data['time'] = data['time'].astype(int)

# Convert date string to datetime
data['time'] = pd.to_datetime(data['time'])

# Convert datetime to year
data['year'] = data['time'].dt.year
```

### 3. Panel Structure

**Check your panel:**

```python
# Number of entities
n_entities = data['entity_id'].nunique()
print(f"Entities (N): {n_entities}")

# Number of time periods
n_periods = data['time'].nunique()
print(f"Time periods (T): {n_periods}")

# Total observations
print(f"Total obs: {len(data)}")

# Is it balanced?
obs_per_entity = data.groupby('entity_id').size()
is_balanced = (obs_per_entity == n_periods).all()
print(f"Balanced: {is_balanced}")

if not is_balanced:
    print(f"Obs per entity: min={obs_per_entity.min()}, max={obs_per_entity.max()}")
```

## Loading from Different Sources

### From CSV

```python
import pandas as pd

# Basic CSV load
data = pd.read_csv('panel_data.csv')

# With options
data = pd.read_csv(
    'panel_data.csv',
    sep=',',              # Delimiter
    encoding='utf-8',     # Encoding
    parse_dates=['date'], # Parse date columns
    dtype={'firm_id': int, 'year': int}  # Specify types
)

print(data.head())
```

### From Excel

```python
# Single sheet
data = pd.read_excel('panel_data.xlsx', sheet_name='Sheet1')

# Multiple sheets (if different years)
import pandas as pd

sheets = pd.read_excel('panel_data.xlsx', sheet_name=None)  # Load all sheets

# Combine sheets (assuming each sheet is a year)
frames = []
for year, df in sheets.items():
    df['year'] = int(year)
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
print(data.head())
```

### From Stata (.dta)

```python
# Load Stata file
data = pd.read_stata('panel_data.dta')

# Convert Stata index to columns if needed
data = data.reset_index()

print(data.head())
```

### From SQL Database

```python
import pandas as pd
import sqlite3  # or pymysql, psycopg2, etc.

# Connect to database
conn = sqlite3.connect('database.db')

# Query panel data
query = """
SELECT firm_id, year, sales, assets, employees
FROM panel_table
WHERE year BETWEEN 2015 AND 2023
ORDER BY firm_id, year
"""

data = pd.read_sql_query(query, conn)
conn.close()

print(data.head())
```

### From R (.rds, .RData)

```python
import pyreadr

# Load .rds file
result = pyreadr.read_r('panel_data.rds')
data = result[None]  # Default object

# Load .RData file
result = pyreadr.read_r('panel_data.RData')
data = result['df_name']  # Specify object name

print(data.head())
```

### From Web APIs

```python
import pandas as pd
import requests

# Example: World Bank API (GDP data)
url = "http://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD"
params = {
    'format': 'json',
    'date': '2010:2020',
    'per_page': 1000
}

response = requests.get(url, params=params)
json_data = response.json()

# Parse JSON to DataFrame (structure depends on API)
data = pd.DataFrame(json_data[1])

# Clean and reshape as needed
# ...
```

## Common Data Preparation Tasks

### 1. Handling Missing Values

**Inspect missingness:**

```python
# Count missing by variable
print(data.isnull().sum())

# Percentage missing
print(data.isnull().mean() * 100)

# Missing pattern by entity
missing_by_entity = data.groupby('entity_id').apply(lambda x: x.isnull().sum())
print(missing_by_entity)
```

**Options:**

```python
# Option 1: Drop rows with any missing
data_clean = data.dropna()

# Option 2: Drop rows with missing in specific columns
data_clean = data.dropna(subset=['y', 'x1', 'x2'])

# Option 3: Forward fill within entities
data_clean = data.sort_values(['entity_id', 'time'])
data_clean['x1'] = data_clean.groupby('entity_id')['x1'].ffill()

# Option 4: Keep missing (PanelBox handles NaN automatically)
# Just ensure no missing in dependent variable
data_clean = data.dropna(subset=['y'])
```

### 2. Creating Lagged Variables

```python
# Sort by entity and time first!
data = data.sort_values(['entity_id', 'time'])

# Create lag within each entity
data['y_lag1'] = data.groupby('entity_id')['y'].shift(1)
data['x1_lag1'] = data.groupby('entity_id')['x1'].shift(1)

# Multiple lags
data['y_lag2'] = data.groupby('entity_id')['y'].shift(2)

# Lead (forward shift)
data['y_lead1'] = data.groupby('entity_id')['y'].shift(-1)

print(data.head(10))
```

### 3. Creating Differences

```python
# First difference
data['delta_y'] = data.groupby('entity_id')['y'].diff()

# Log difference (growth rate)
import numpy as np
data['log_y'] = np.log(data['y'])
data['growth_y'] = data.groupby('entity_id')['log_y'].diff()

print(data[['entity_id', 'time', 'y', 'delta_y', 'growth_y']].head(10))
```

### 4. Creating Time-Invariant Variables

```python
# Entity mean
data['y_mean'] = data.groupby('entity_id')['y'].transform('mean')

# Entity-specific constant (e.g., size category)
entity_size = data.groupby('entity_id')['assets'].mean()
entity_size_cat = pd.cut(entity_size, bins=3, labels=['Small', 'Medium', 'Large'])

data = data.merge(
    entity_size_cat.rename('size_category').reset_index(),
    on='entity_id',
    how='left'
)

print(data[['entity_id', 'time', 'assets', 'size_category']].head())
```

### 5. Encoding Categorical Variables

```python
# For time-varying categoricals
data = pd.get_dummies(data, columns=['industry'], drop_first=True)

# For time-invariant categoricals (will be dropped in FE)
# Keep as is or create separate variable
```

### 6. Winsorizing Outliers

```python
from scipy.stats import mstats

# Winsorize at 1st and 99th percentiles
data['x1_wins'] = mstats.winsorize(data['x1'], limits=[0.01, 0.01])

# Or manually
p01 = data['x1'].quantile(0.01)
p99 = data['x1'].quantile(0.99)
data['x1_wins'] = data['x1'].clip(lower=p01, upper=p99)
```

### 7. Balancing Panel

**Keep only balanced subset:**

```python
# Count observations per entity
obs_counts = data.groupby('entity_id').size()

# Keep entities with all time periods
n_periods = data['time'].nunique()
balanced_entities = obs_counts[obs_counts == n_periods].index

data_balanced = data[data['entity_id'].isin(balanced_entities)]

print(f"Original: {data['entity_id'].nunique()} entities")
print(f"Balanced: {data_balanced['entity_id'].nunique()} entities")
```

## Complete Example Workflow

```python
import pandas as pd
import numpy as np
import panelbox as pb

# 1. Load data
data = pd.read_csv('firm_data.csv')

# 2. Check structure
print("Original data:")
print(data.head())
print(f"\nShape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# 3. Rename columns to standard names
data = data.rename(columns={
    'company_id': 'firm_id',
    'fiscal_year': 'year',
    'total_sales': 'sales',
    'total_assets': 'assets',
    'num_employees': 'employees'
})

# 4. Convert types
data['firm_id'] = data['firm_id'].astype(int)
data['year'] = data['year'].astype(int)

# 5. Sort by entity and time
data = data.sort_values(['firm_id', 'year'])

# 6. Handle missing values
print("\nMissing values:")
print(data.isnull().sum())

# Drop rows with missing dependent variable
data = data.dropna(subset=['sales'])

# Forward-fill missing regressors within firms
for col in ['assets', 'employees']:
    data[col] = data.groupby('firm_id')[col].ffill()

# Drop remaining missing
data = data.dropna()

# 7. Create additional variables
# Log variables
data['log_sales'] = np.log(data['sales'])
data['log_assets'] = np.log(data['assets'])

# Lagged sales
data['log_sales_lag1'] = data.groupby('firm_id')['log_sales'].shift(1)

# Sales growth
data['sales_growth'] = data.groupby('firm_id')['log_sales'].diff()

# 8. Winsorize outliers
from scipy.stats import mstats
data['sales_growth_wins'] = mstats.winsorize(
    data['sales_growth'].dropna(),
    limits=[0.01, 0.01]
)

# 9. Check final panel structure
print("\nFinal panel structure:")
print(f"Entities: {data['firm_id'].nunique()}")
print(f"Time periods: {data['year'].nunique()}")
print(f"Total obs: {len(data)}")

obs_per_firm = data.groupby('firm_id').size()
print(f"Balanced: {(obs_per_firm == obs_per_firm.iloc[0]).all()}")

# 10. Estimate model
model = pb.FixedEffects(
    formula="log_sales ~ log_assets + employees",
    data=data,
    entity_col="firm_id",
    time_col="year",
    entity_effects=True,
    time_effects=True
)

results = model.fit(cov_type='clustered')
print("\n" + "="*80)
print(results.summary())

# 11. Save cleaned data
data.to_csv('firm_data_cleaned.csv', index=False)
print("\nCleaned data saved to 'firm_data_cleaned.csv'")
```

## Troubleshooting

### Error: "KeyError: 'entity_col'"

**Cause:** Column name doesn't match

**Solution:** Check exact column name (case-sensitive)

```python
print(data.columns)  # Check exact names
# Use correct name in entity_col parameter
```

### Error: "ValueError: could not convert string to float"

**Cause:** Non-numeric data in variable columns

**Solution:** Check and convert

```python
# Find non-numeric values
print(data['x1'].dtype)
print(data[pd.to_numeric(data['x1'], errors='coerce').isnull()])

# Convert, forcing errors to NaN
data['x1'] = pd.to_numeric(data['x1'], errors='coerce')
```

### Warning: "Panel is unbalanced"

**Info only:** PanelBox handles unbalanced panels automatically

**If you want balanced:** Follow balancing steps above

### Error: "Not enough observations"

**Cause:** Too many missing values after cleaning

**Solution:**
- Relax missing value handling
- Use different time period
- Check data quality

## Best Practices

✅ **Do:**
- Always sort by entity and time before creating lags
- Check for duplicates (entity-time pairs should be unique)
- Inspect missing value patterns
- Save intermediate cleaned datasets
- Document all transformations

❌ **Don't:**
- Assume data is already sorted
- Ignore missing values without investigating
- Create lags without grouping by entity
- Delete outliers without justification
- Mix wide and long formats

## Next Steps

**After loading data:**

1. **[Tutorial 1: Getting Started](../tutorials/01_getting_started.md)**: Estimate your first model

2. **[How-To: Choose Model](choose_model.md)**: Decide which estimator to use

3. **[Tutorial 2: Static Models](../tutorials/02_static_models.md)**: Learn Fixed Effects, Random Effects

**Advanced data preparation:**
- Merging multiple datasets
- Panel-specific transformations
- Handling highly unbalanced panels
- Time-varying treatment indicators

---

**Need help?** Open an issue on [GitHub](https://github.com/PanelBox-Econometrics-Model/panelbox/issues) with a sample of your data structure.
