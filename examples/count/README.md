# Count Models Tutorials

**Version:** 1.0.0
**Last Updated:** 2026-02-16

## Overview

This directory contains comprehensive tutorials for count data econometrics using PanelBox. Count models are essential for analyzing outcomes that represent non-negative integers, such as the number of patents filed, doctor visits, or trade flows between countries.

## Why Count Models?

Standard linear regression (OLS) is inappropriate for count data because:
- Predicted values can be negative
- Assumes constant variance (count data is often overdispersed)
- Ignores the discrete nature of the outcome
- Performs poorly with excess zeros

Count models address these issues through:
- Non-linear specifications ensuring positive predictions
- Flexible variance structures
- Maximum likelihood estimation tailored to count distributions
- Special handling for zero-inflated data

## Tutorial Series

### Beginner Level

#### 01. Poisson Introduction (60-75 min)
**File:** `notebooks/01_poisson_introduction.ipynb`
**Prerequisites:** Basic regression knowledge

Learn the fundamentals of Poisson regression for count data:
- When to use Poisson models
- Interpretation as incidence rate ratios (IRRs)
- Comparison with OLS
- Basic diagnostics and assumption testing
- **Dataset:** Healthcare visits (N=2,000)

### Intermediate Level

#### 02. Negative Binomial (60 min)
**File:** `notebooks/02_negative_binomial.ipynb`
**Prerequisites:** Tutorial 01

Handle overdispersed count data:
- Detecting overdispersion
- Negative Binomial Type I and Type II
- Comparison with Poisson
- Overdispersion tests
- **Dataset:** Firm patents (N=1,500, T=5)

#### 06. Marginal Effects (65 min)
**File:** `notebooks/06_marginal_effects_count.ipynb`
**Prerequisites:** Tutorials 01-02

Compute and interpret marginal effects:
- Average Marginal Effects (AME)
- Marginal Effects at the Mean (MEM)
- Marginal Effects at Representative Values (MER)
- Standard errors and confidence intervals
- **Dataset:** Policy impact (N=1,200)

### Intermediate-Advanced Level

#### 03. Fixed and Random Effects (75 min)
**File:** `notebooks/03_fe_re_count.ipynb`
**Prerequisites:** Tutorials 01-02, panel data basics

Analyze count panel data:
- Fixed Effects Poisson (Hausman et al. 1984)
- Random Effects models
- Hausman specification test
- Within and between variation
- **Dataset:** City crime (N=150, T=10)

### Advanced Level

#### 04. PPML and Gravity Models (90 min)
**File:** `notebooks/04_ppml_gravity.ipynb`
**Prerequisites:** Tutorial 01

Apply Poisson Pseudo-Maximum Likelihood (PPML) to gravity equations:
- Santos Silva & Tenreyro (2006) methodology
- Handling zeros in log-linear models
- High-dimensional fixed effects
- Trade elasticities
- **Dataset:** Bilateral trade (N=10,000, T=15)

#### 05. Zero-Inflated Models (70 min)
**File:** `notebooks/05_zero_inflated.ipynb`
**Prerequisites:** Tutorials 01-02

Model excess zeros with two-part specifications:
- Zero-Inflated Poisson (ZIP)
- Zero-Inflated Negative Binomial (ZINB)
- Vuong test for model selection
- Interpreting dual processes
- **Dataset:** Healthcare with excess zeros (N=3,000)

#### 07. Innovation Case Study (90-120 min)
**File:** `notebooks/07_innovation_case_study.ipynb`
**Prerequisites:** All previous tutorials

Complete analysis of firm innovation:
- Full workflow from data exploration to publication-ready results
- Model selection across Poisson, NB, ZIP, ZINB
- Marginal effects and policy simulations
- Robustness checks
- **Dataset:** Firm innovation (N=500, T=8)

## Learning Pathway

### Recommended Sequences

**For Beginners:**
1. Tutorial 01 (Poisson Introduction)
2. Tutorial 02 (Negative Binomial)
3. Tutorial 06 (Marginal Effects)
4. Tutorial 07 (Case Study)

**For Panel Data Focus:**
1. Tutorial 01 (Poisson Introduction)
2. Tutorial 03 (Fixed/Random Effects)
3. Tutorial 04 (PPML Gravity)

**For Applied Researchers:**
1. Tutorial 01 (Poisson Introduction)
2. Tutorial 02 (Negative Binomial)
3. Tutorial 05 (Zero-Inflated)
4. Tutorial 06 (Marginal Effects)
5. Tutorial 07 (Case Study)

## Prerequisites

### Required Knowledge
- Basic statistics (regression, hypothesis testing)
- Python programming fundamentals
- Pandas and NumPy basics
- Understanding of panel data (for tutorials 03-04)

### Software Requirements
- Python 3.8+
- PanelBox 0.7.0+
- pandas, numpy, matplotlib, seaborn
- Jupyter Notebook or JupyterLab

Install dependencies:
```bash
pip install panelbox pandas numpy matplotlib seaborn jupyter
```

Or with conda:
```bash
conda install -c conda-forge panelbox pandas numpy matplotlib seaborn jupyter
```

## Quick Start

1. **Navigate to the directory:**
   ```bash
   cd examples/count
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open the first tutorial:**
   - Navigate to `notebooks/01_poisson_introduction.ipynb`
   - Follow the instructions in the notebook

4. **Check your setup:**
   ```python
   import panelbox
   import pandas as pd
   from pathlib import Path

   # Verify PanelBox version
   print(f"PanelBox version: {panelbox.__version__}")

   # Test data access
   data_path = Path('data/healthcare_visits.csv')
   df = pd.read_csv(data_path)
   print(f"Data loaded: {df.shape}")
   ```

## Directory Structure

```
count/
├── README.md                    # This file
├── __init__.py                  # Module initialization
├── GETTING_STARTED.md           # Detailed setup guide
├── notebooks/                   # Tutorial notebooks
│   ├── 01_poisson_introduction.ipynb
│   ├── 02_negative_binomial.ipynb
│   ├── 03_fe_re_count.ipynb
│   ├── 04_ppml_gravity.ipynb
│   ├── 05_zero_inflated.ipynb
│   ├── 06_marginal_effects_count.ipynb
│   └── 07_innovation_case_study.ipynb
├── data/                        # Tutorial datasets
│   ├── README.md
│   ├── *.csv                    # 7 datasets
│   └── codebooks/               # Variable documentation
├── outputs/                     # Generated outputs
│   ├── figures/                 # Plots
│   ├── tables/                  # Tables
│   └── results/                 # Model objects
├── solutions/                   # Exercise solutions
├── utils/                       # Helper functions
│   ├── data_generators.py
│   ├── visualization_helpers.py
│   └── diagnostics_helpers.py
└── tests/                       # Integrity tests
```

## Datasets

All datasets are simulated based on real-world stylized facts:

| Dataset | Obs | Purpose | Key Features |
|---------|-----|---------|--------------|
| healthcare_visits.csv | 2,000 | Poisson intro | Mild overdispersion, 15% zeros |
| firm_patents.csv | 1,500×5 | Negative Binomial | Severe overdispersion |
| city_crime.csv | 150×10 | Panel FE/RE | Balanced panel |
| bilateral_trade.csv | 10,000×15 | PPML gravity | 23% zeros, HDFE |
| healthcare_zinb.csv | 3,000 | Zero-inflated | 60% zeros |
| policy_impact.csv | 1,200 | Marginal effects | Treatment effects |
| firm_innovation_full.csv | 500×8 | Complete case | 35% zeros |

See `data/README.md` for complete documentation.

## Key References

### Foundational Papers

**Poisson Regression:**
- Hausman, J., Hall, B. H., & Griliches, Z. (1984). Econometric models for count data with an application to the patents-R&D relationship. *Econometrica*, 52(4), 909-938.

**PPML and Gravity:**
- Santos Silva, J. M. C., & Tenreyro, S. (2006). The log of gravity. *The Review of Economics and Statistics*, 88(4), 641-658.
- Santos Silva, J. M. C., & Tenreyro, S. (2011). Further simulation evidence on the performance of the Poisson pseudo-maximum likelihood estimator. *Economics Letters*, 112(2), 220-222.

**Negative Binomial:**
- Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of Count Data* (2nd ed.). Cambridge University Press.

**Zero-Inflated Models:**
- Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. *Technometrics*, 34(1), 1-14.
- Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. *Econometrica*, 57(2), 307-333.

**Panel Count Models:**
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 19.

### Applied Examples

- Head, K., & Mayer, T. (2014). Gravity equations: Workhorse, toolkit, and cookbook. In *Handbook of International Economics* (Vol. 4, pp. 131-195).
- Blundell, R., Griffith, R., & Windmeijer, F. (2002). Individual effects and dynamics in count data models. *Journal of Econometrics*, 108(1), 113-131.

## Common Issues and Troubleshooting

### Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'panelbox'`
**Solution:** Install PanelBox: `pip install panelbox`

**Problem:** Old version of PanelBox
**Solution:** Upgrade: `pip install --upgrade panelbox`

### Data Loading Issues

**Problem:** `FileNotFoundError` when loading data
**Solution:** Check your working directory and use relative paths:
```python
from pathlib import Path
data_path = Path(__file__).parent / 'data' / 'healthcare_visits.csv'
```

**Problem:** Encoding errors when reading CSV
**Solution:** Specify encoding: `pd.read_csv(path, encoding='utf-8')`

### Runtime Issues

**Problem:** Convergence warnings in optimization
**Solution:**
- Check for multicollinearity
- Scale large variables
- Try different starting values
- See notebook diagnostics sections

**Problem:** Singular matrix errors
**Solution:** Remove perfectly collinear variables (e.g., include N-1 dummies)

## Getting Help

### Within Tutorials
- Each notebook has extensive comments and explanations
- Exercise solutions are in `solutions/` directory
- Check the relevant codebook in `data/codebooks/`

### External Resources
- **PanelBox Documentation:** https://panelbox.readthedocs.io
- **Issue Tracker:** https://github.com/panelbox/panelbox/issues
- **Stack Overflow:** Tag questions with `panelbox` and `count-data`

### Contact
For questions about these tutorials:
- Open an issue on the PanelBox GitHub repository
- Refer to the specific tutorial number in your question

## Contributing

Found an error or have a suggestion?
1. Check existing issues on GitHub
2. Open a new issue with:
   - Tutorial number and section
   - Description of the issue
   - Suggested fix (if applicable)

## License

These tutorials are released under the MIT License, same as PanelBox.

## Citation

If you use these tutorials in academic work, please cite:

```bibtex
@misc{panelbox_count_tutorials,
  title={PanelBox Count Models Tutorials},
  author={PanelBox Development Team},
  year={2026},
  howpublished={\url{https://github.com/panelbox/panelbox}},
  note={Version 1.0.0}
}
```

---

**Ready to start?** Open `GETTING_STARTED.md` for detailed setup instructions, or jump directly to `notebooks/01_poisson_introduction.ipynb`.
