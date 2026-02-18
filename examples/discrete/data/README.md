# Discrete Choice Models - Data Dictionary

**Version**: 1.0
**Last Updated**: 2026-02-16

---

## Overview

This directory contains datasets used throughout the discrete choice tutorial series. All datasets are either synthetic (generated for pedagogical purposes) or derived from publicly available sources.

---

## Datasets

### 1. Labor Participation (`labor_participation.csv`)

**Description**: Panel data on labor force participation decisions of married women.

**Structure**: Balanced panel with N individuals observed over T periods

**Variables**:

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `id` | int | Individual identifier | 1 to N |
| `year` | int | Time period | 1 to T |
| `lfp` | binary | Labor force participation | 0 = not participating, 1 = participating |
| `age` | int | Age in years | 20-65 |
| `educ` | int | Years of education | 8-20 |
| `kids` | int | Number of children | 0-5 |
| `married` | binary | Marital status | 0 = not married, 1 = married |
| `exper` | int | Years of work experience | 0-40 |

**Sample Size**: Variable (specified in tutorial)

**Used In**: Notebooks 01, 02, 03, 04

**Data Generating Process**:
```
Pr(lfp = 1) = Λ(-3.0 + 0.05*age - 0.0005*age² + 0.15*educ + 0.03*exper - 0.5*kids + 0.3*married + αᵢ)
```
where Λ is the logistic CDF and αᵢ ~ N(0,1) is an individual-specific effect.

**Source**: Synthetic data generated using `utils/data_generators.py::generate_labor_data()`

---

### 2. Job Training (`job_training.csv`)

**Description**: Panel data on job training program participation and employment outcomes.

**Structure**: Balanced panel with N individuals over T periods

**Variables**:

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `id` | int | Individual identifier | 1 to N |
| `year` | int | Time period | 1 to T |
| `training` | binary | Participated in training this period | 0 = no, 1 = yes |
| `wage` | float | Hourly wage in dollars | 5-50 |
| `employed` | binary | Employment status | 0 = unemployed, 1 = employed |
| `prior_training` | int | Number of prior training programs | 0-3 |

**Sample Size**: Variable (specified in tutorial)

**Used In**: Notebooks 02, 08 (dynamic models)

**Notes**:
- `training` exhibits state dependence (prior participation increases future probability)
- Useful for demonstrating dynamic discrete choice models
- Initial conditions matter for consistent estimation

**Source**: Synthetic data with realistic labor market dynamics

---

### 3. Transportation Choice (`transportation_choice.csv`)

**Description**: Cross-sectional data on transportation mode choice for commuting.

**Structure**: Cross-section with one observation per individual, multiple alternatives

**Variables**:

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `id` | int | Individual identifier | 1 to N |
| `year` | int | Survey year | 2020 |
| `choice` | categorical | Chosen transportation mode | 0=bus, 1=car, 2=metro, 3=bike |
| `cost_bus` | float | Cost of bus in dollars | 1-10 |
| `time_bus` | float | Travel time by bus in minutes | 15-90 |
| `cost_car` | float | Cost of car (fuel + parking) | 5-30 |
| `time_car` | float | Travel time by car in minutes | 10-60 |
| `cost_metro` | float | Cost of metro | 2-8 |
| `time_metro` | float | Travel time by metro in minutes | 12-70 |
| `cost_bike` | float | Cost of bike (maintenance) | 0-2 |
| `time_bike` | float | Travel time by bike in minutes | 20-100 |
| `income` | float | Annual income in thousands | 20-150 |
| `age` | int | Age in years | 18-70 |

**Sample Size**: ~5,000 observations

**Used In**: Notebooks 05 (conditional logit), 06 (multinomial logit)

**Notes**:
- Choice-specific attributes: `cost_X` and `time_X` for each mode X
- Individual-specific attributes: `income` and `age`
- Allows demonstration of both conditional logit (choice-specific) and multinomial logit (individual-specific) specifications

**Source**: Synthetic data based on urban transportation patterns

---

### 4. Credit Rating (`credit_rating.csv`)

**Description**: Panel data on credit ratings (ordered categorical outcome).

**Structure**: Balanced panel with N individuals over T periods

**Variables**:

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `id` | int | Individual identifier | 1 to N |
| `year` | int | Time period | 1 to T |
| `rating` | ordered | Credit rating category | 0=poor, 1=fair, 2=good, 3=excellent |
| `income` | float | Annual income in thousands | 15-200 |
| `debt_ratio` | float | Debt-to-income ratio | 0.1-0.8 |
| `age` | int | Age in years | 22-70 |
| `payment_history` | int | Months of on-time payments | 0-120 |

**Sample Size**: Variable (specified in tutorial)

**Used In**: Notebook 07 (ordered models)

**Ordered Categories**:
- 0: Poor (high default risk)
- 1: Fair (moderate risk)
- 2: Good (low risk)
- 3: Excellent (very low risk)

**Data Generating Process**:
Uses ordered probit with latent variable:
```
y* = 0.0001*income - 3.0*debt_ratio + 0.01*age + 0.02*payment_history + αᵢ + εᵢₜ
rating = k if τₖ < y* ≤ τₖ₊₁
```
where τ are threshold parameters.

**Source**: Synthetic data generated using `utils/data_generators.py::generate_ordered_data()`

---

### 5. Career Choice (`career_choice.csv`)

**Description**: Cross-sectional data on career choices of recent college graduates.

**Structure**: Cross-section with one observation per graduate

**Variables**:

| Variable | Type | Description | Values/Range |
|----------|------|-------------|--------------|
| `id` | int | Individual identifier | 1 to N |
| `year` | int | Graduation year | 2020 |
| `career` | categorical | Chosen career path | 0=industry, 1=academia, 2=government, 3=nonprofit |
| `wage` | float | Expected starting wage (thousands) | 30-120 |
| `prestige` | int | Job prestige rating | 1-10 |
| `hours` | int | Expected weekly hours | 35-70 |
| `travel_required` | binary | Significant travel required | 0 = no, 1 = yes |
| `education` | int | Years of education | 16-22 |
| `gpa` | float | Undergraduate GPA | 2.0-4.0 |
| `stem_major` | binary | STEM major | 0 = no, 1 = yes |

**Sample Size**: ~2,000 observations

**Used In**: Notebooks 06 (multinomial logit), 09 (case study)

**Career Categories**:
- 0: Industry/private sector
- 1: Academia/research
- 2: Government
- 3: Nonprofit/NGO

**Notes**:
- Demonstrates multinomial choice with many individual-specific covariates
- No choice-specific attributes (use multinomial logit, not conditional logit)
- Useful for interpretation of relative risk ratios

**Source**: Synthetic data based on career choice literature

---

## Data Loading Convention

All tutorial notebooks use this standardized pattern:

```python
from pathlib import Path
import pandas as pd

# Determine data path relative to notebook location
DATA_DIR = Path("..") / "data"

# Load specific dataset
data = pd.read_csv(DATA_DIR / "labor_participation.csv")

# For panel data, set index
data = data.set_index(['id', 'year'])
```

---

## Data Generation Scripts

All synthetic datasets can be regenerated using functions in `utils/data_generators.py`:

```python
from discrete.utils.data_generators import (
    generate_labor_data,
    generate_multinomial_choice_data,
    generate_ordered_data
)

# Generate labor participation data
labor = generate_labor_data(n_individuals=1000, n_periods=5, seed=42)
labor.to_csv("labor_participation.csv", index=False)

# Generate multinomial choice data
multi_wide, multi_long = generate_multinomial_choice_data(n_obs=5000, seed=42)
multi_wide.to_csv("transportation_choice.csv", index=False)

# Generate ordered categorical data
ordered = generate_ordered_data(n_individuals=800, n_periods=4, seed=42)
ordered.to_csv("credit_rating.csv", index=False)
```

**Important**: Use the same `seed` parameter for reproducibility across tutorials.

---

## Data Quality and Validation

### Checks Performed

All datasets have been validated for:

1. **Missing values**: No missing data in key variables
2. **Outliers**: Realistic ranges for all continuous variables
3. **Balance**: Panel datasets are balanced (equal observations per individual)
4. **Consistency**: Logical relationships between variables maintained
5. **Sample size**: Sufficient variation for estimation

### Known Limitations

1. **Synthetic data**: Generated data may not capture all real-world complexities
2. **Simplified dynamics**: Dynamic processes use simplified specifications
3. **No measurement error**: Covariates measured without error (unlike real data)
4. **Perfect categories**: Categorical variables have no ambiguous cases

**Pedagogical Note**: These limitations are intentional to facilitate learning. Real-world applications will require additional data cleaning and validation steps.

---

## Extending the Datasets

### Adding New Variables

To add variables to existing datasets:

1. Modify the data generation function in `utils/data_generators.py`
2. Regenerate the dataset with a new filename (e.g., `labor_participation_v2.csv`)
3. Document new variables in this README
4. Update relevant notebooks

### Creating New Datasets

To create entirely new datasets:

1. Write a new generation function in `utils/data_generators.py`
2. Follow naming conventions: `generate_[purpose]_data()`
3. Include comprehensive docstrings
4. Add entry to this README with complete variable descriptions
5. Create example notebook demonstrating usage

---

## License and Attribution

All datasets in this directory are:

- **Synthetic data**: Generated for PanelBox tutorials
- **License**: MIT License (same as PanelBox)
- **Attribution**: Not required but appreciated

If you use these datasets in publications or teaching:

```bibtex
@misc{panelbox_data,
  title={Discrete Choice Tutorial Datasets},
  author={PanelBox Contributors},
  year={2026},
  howpublished={\url{https://github.com/panelbox/panelbox/examples/discrete/data}}
}
```

---

## Contact

For questions about the datasets or to report issues:

- **GitHub Issues**: https://github.com/panelbox/panelbox/issues
- **Discussions**: https://github.com/panelbox/panelbox/discussions

---

**Last Updated**: 2026-02-16 by PanelBox Contributors
