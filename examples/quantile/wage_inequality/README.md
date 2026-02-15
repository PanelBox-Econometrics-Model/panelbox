# Wage Inequality Analysis

This example demonstrates a complete wage inequality analysis using quantile regression.

## Overview

The analysis examines three key aspects of wage inequality:

1. **Heterogeneous Returns to Education** - How education affects wages across the distribution
2. **Gender Wage Gap Decomposition** - Machado-Mata decomposition into explained/unexplained components
3. **Union Wage Premium** - How unions affect workers at different wage levels

## Quick Start

```python
from wage_analysis import WageInequalityAnalysis

# Initialize analysis (uses simulated data by default)
analysis = WageInequalityAnalysis()

# Run complete analysis
analysis.descriptive_statistics()
analysis.analyze_education_returns()
analysis.analyze_gender_gap()
analysis.analyze_union_effects()

# Generate report
analysis.generate_report()
```

## Using Your Own Data

```python
# Load your own panel data
analysis = WageInequalityAnalysis(data_path='your_data.csv')
```

Required columns:
- `log_wage` - Log hourly wage
- `education` - Years of education
- `experience` - Years of labor market experience
- `female` - Gender indicator (1=female, 0=male)
- `union` - Union membership (1=member, 0=non-member)
- `person_id` - Individual identifier
- `year` - Time period

## Key Findings

### 1. Returns to Education

Returns to education are **heterogeneous** across the wage distribution:
- Bottom 10%: 4.5% per additional year
- Median: 5.5% per additional year
- Top 10%: 7.0% per additional year

**Implication:** Education is more valuable for high earners, potentially increasing inequality.

### 2. Gender Wage Gap

The gender wage gap **decreases** across the distribution:
- Bottom 10%: 25% gap
- Median: 20% gap
- Top 10%: 15% gap

**Decomposition** (at median):
- Total gap: 20%
- Explained by characteristics: 6%
- Unexplained (discrimination): 14% (70% of total)

**Implication:** Most of the gap is unexplained, suggesting discrimination affects women throughout the distribution.

### 3. Union Wage Premium

Union wage premium is **largest at the bottom**:
- Bottom 10%: 20% premium
- Median: 15% premium
- Top 10%: 10% premium

**Implication:** Unions compress the wage distribution by benefiting low-wage workers more.

## Outputs

The analysis generates:
- `simulated_wage_data.csv` - Simulated panel dataset
- `education_returns.png` - Plot of heterogeneous education returns
- `gender_gap_decomposition.png` - Gender gap decomposition plots
- `union_premium.png` - Union wage premium plot
- `wage_inequality_report.html` - Complete HTML report

## Methods

### Quantile Regression

Unlike OLS which estimates the conditional mean:

$$E[Y|X] = X\beta$$

Quantile regression estimates the conditional quantiles:

$$Q_Y(\tau|X) = X\beta(\tau)$$

This allows us to see how effects vary across the distribution.

### Machado-Mata Decomposition

The gender wage gap at quantile τ can be decomposed as:

$$\Delta(\tau) = \underbrace{(X_M - X_F)'\beta_M(\tau)}_{\text{Explained}} + \underbrace{X_F'(\beta_M(\tau) - \beta_F(\tau))}_{\text{Unexplained}}$$

Where:
- $X_M, X_F$ = Average characteristics of men/women
- $\beta_M(\tau), \beta_F(\tau)$ = Coefficients for men/women at quantile τ

### Quantile Treatment Effects (QTE)

The union wage premium at quantile τ is estimated as:

$$QTE(\tau) = Q_{Y|D=1}(\tau) - Q_{Y|D=0}(\tau)$$

Where D=1 indicates union membership.

## References

- Machado, J. A., & Mata, J. (2005). Counterfactual decomposition of changes in wage distributions using quantile regression. *Journal of Applied Econometrics*, 20(4), 445-465.
- Canay, I. A. (2011). A simple approach to quantile regression for panel data. *The Econometrics Journal*, 14(3), 368-386.
- Firpo, S., Fortin, N. M., & Lemieux, T. (2009). Unconditional quantile regressions. *Econometrica*, 77(3), 953-973.

## Citation

If you use this example in your research, please cite:

```bibtex
@software{panelbox_qr,
  title = {PanelBox: Panel Data Econometrics in Python},
  author = {PanelBox Development Team},
  year = {2025},
  url = {https://github.com/panelbox/panelbox}
}
```
