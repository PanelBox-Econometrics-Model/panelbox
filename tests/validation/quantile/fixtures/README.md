# Quantile Regression Test Fixtures

This directory contains benchmark datasets for validating PanelBox quantile regression implementations against reference implementations (R's quantreg/rqpd packages).

## Generated Datasets

All datasets are balanced panels with N=100 entities and T=10 time periods (1000 observations total).

### 1. test_data_simple.csv
**Purpose**: Basic heteroskedastic panel with known coefficients
**Columns**: entity, time, y, x1, x2, x3

**Data Generating Process**:
```
y_it = 1 + 2*x1_it - 1.5*x2_it + 0.5*x3_it + alpha_i + epsilon_it
```

Where:
- alpha_i ~ N(0, 1) : entity fixed effects
- epsilon_it ~ (1 + 0.5*x1_it) * N(0, 1) : heteroskedastic errors
- x1_it ~ N(5, 2)
- x2_it ~ N(3, 1)
- x3_it ~ Uniform(0, 10)

**Use for**: Testing basic panel quantile regression with heteroskedasticity

### 2. test_data_location_scale.csv
**Purpose**: Location-scale model for MSS (2019) validation
**Columns**: entity, time, y, x1, x2

**Data Generating Process**:
```
y_it = X'α + σ(X) * ε_it
```

Where:
- α = [1, 2, -1] : location parameters (constant, x1, x2)
- log(σ) = X'γ with γ = [0.5, 0.3, -0.2] : log-scale parameters
- X = [1, x1_it, x2_it]
- ε_it ~ N(0, 1)
- x1_it ~ N(5, 2)
- x2_it ~ Uniform(0, 10)

**Use for**: Testing location-scale quantile regression methods

### 3. test_data_heterogeneity.csv
**Purpose**: Panel with quantile-varying coefficients
**Columns**: entity, time, y, x1, x2

**Data Generating Process**:
```
Q_y(τ|X) = β₀(τ) + β₁(τ)*x1 + β₂(τ)*x2 + αᵢ
```

Where:
- β₁(τ) = 2 + 0.5*Φ⁻¹(τ) : increases with quantile τ
- β₂(τ) = -1.5 : constant across quantiles
- β₀(τ) = 1
- αᵢ ~ N(0, 1) : entity fixed effects
- x1_it ~ N(5, 2)
- x2_it ~ N(3, 1)

**Use for**: Testing methods that capture quantile-varying effects

## True Parameters

The file `true_parameters.json` contains the known DGP parameters for validation:

```json
{
  "simple": {
    "intercept": 1.0,
    "x1": 2.0,
    "x2": -1.5,
    "x3": 0.5,
    "sigma_alpha": 1.0
  },
  "location_scale": {
    "location": [1, 2, -1],
    "scale": [0.5, 0.3, -0.2]
  }
}
```

## Regenerating Datasets

To regenerate all datasets with the same seed:

```bash
cd tests/validation/quantile/fixtures
python create_test_data.py
```

Or from project root:

```bash
python tests/validation/quantile/fixtures/create_test_data.py
```

## Validation Checklist

- ✅ All datasets have 1000 observations (100 entities × 10 time periods)
- ✅ No NaN or infinite values
- ✅ Entity IDs range from 0 to 99
- ✅ Time IDs range from 0 to 9
- ✅ Datasets readable by both Python (pandas) and R (read.csv)
- ✅ True parameters saved in JSON format

## Usage in Tests

```python
import pandas as pd
from pathlib import Path

fixtures_dir = Path(__file__).parent / 'fixtures'

# Load simple panel
df = pd.read_csv(fixtures_dir / 'test_data_simple.csv')

# Load true parameters
import json
with open(fixtures_dir / 'true_parameters.json') as f:
    true_params = json.load(f)
```

In R:

```r
fixtures_dir <- 'tests/validation/quantile/fixtures'

# Load simple panel
df <- read.csv(file.path(fixtures_dir, 'test_data_simple.csv'))

# Load true parameters
library(jsonlite)
params <- fromJSON(file.path(fixtures_dir, 'true_parameters.json'))
```
