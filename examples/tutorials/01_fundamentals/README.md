# Module 1: Fundamentals of Panel Data Analysis

**Level**: Beginner to Intermediate
**Total Time**: 3.5-4.5 hours
**Prerequisites**: Basic Python, Pandas, and econometrics knowledge

## Learning Objectives

By completing this module, you will be able to:
- Load, validate, and transform panel data structures
- Specify econometric models using R-style formulas
- Estimate basic panel models and interpret results
- Create and visualize spatial weight matrices

## Notebooks

### 01 - Introduction to Panel Data Structures
**Time**: 45-60 min | **Level**: Beginner

Learn how to work with panel data in PanelBox:
- What is panel data? (cross-sectional + time dimensions)
- Loading datasets with `PanelData`
- Validating panel structure
- Transformations (demeaning, lags, first differences)

[Open Notebook](01_introduction_panel_data.ipynb)

### 02 - Model Specification with Formulas
**Time**: 50-60 min | **Level**: Beginner-Intermediate

Master the formula syntax for model specification:
- R-style formula syntax (patsy)
- Transformations (log, polynomial, interactions)
- Entity and time effects
- Advanced formulas (absorbing variables)

[Open Notebook](02_formulas_specification.ipynb)

**Prerequisites**: Notebook 01

### 03 - Estimation and Results Interpretation
**Time**: 60-75 min | **Level**: Intermediate

Estimate models and interpret results:
- Estimating Pooled OLS
- Understanding coefficients, standard errors, p-values
- Robust and clustered standard errors
- Exporting results (LaTeX, Markdown, JSON)

[Open Notebook](03_estimation_interpretation.ipynb)

**Prerequisites**: Notebooks 01, 02

### 04 - Spatial Fundamentals (Optional)
**Time**: 60-70 min | **Level**: Intermediate

Introduction to spatial econometrics:
- Creating spatial weight matrices (contiguity, distance, KNN)
- Row and spectral normalization
- Visualizing spatial connections
- Spatial lags and autocorrelation

[Open Notebook](04_spatial_fundamentals.ipynb)

**Prerequisites**: Notebook 01
**Note**: Independent from notebooks 02-03; required for spatial models module

## Recommended Order

### Linear Path (Recommended for Beginners)
```
01 → 02 → 03 → [Next Module: Classical Estimators]
```
Skip notebook 04 initially; return if you need spatial models.

### Spatial Path
```
01 → 04 → [Next Module: Spatial Models]
```
Then return to 02-03 before advanced spatial models.

### Complete Path
```
01 → 02 → 03 → 04 → [Choose Next Module]
```

## Datasets Used

- **Grunfeld Investment Data** (notebooks 01-03): Corporate investment data from 10 US firms (1935-1954)
- **Simulated Spatial Data** (notebook 04): Grid-based synthetic data for spatial demonstrations

All datasets are available in `../../datasets/`

## Competency Checkpoints

### After Notebook 01
- [ ] Can load panel datasets
- [ ] Can validate panel structure
- [ ] Can apply transformations (demeaning, lags)

### After Notebook 02
- [ ] Can write R-style formulas
- [ ] Can use transformations (log, I())
- [ ] Can create interactions and polynomials

### After Notebook 03
- [ ] Can estimate Pooled OLS models
- [ ] Can interpret coefficients, p-values, confidence intervals
- [ ] Can compare different standard error types
- [ ] Can export results to multiple formats

### After Notebook 04
- [ ] Can create spatial weight matrices
- [ ] Can normalize matrices (row, spectral)
- [ ] Can visualize spatial connections

## Setup

### Installing Dependencies
Ensure you have all required packages:
```bash
cd ../../scripts
python setup_environment.py
```

### Starting Jupyter
```bash
cd /path/to/examples/tutorials/01_fundamentals
jupyter notebook
```

Then open `01_introduction_panel_data.ipynb` to begin.

## Next Steps

After completing this module:
- **Module 2**: Classical Estimators (Fixed Effects, Random Effects)
- **Module 3**: Dynamic GMM (Arellano-Bond)
- **Module 4**: Spatial Panel Models

## Additional Resources

### Documentation
- **Main Documentation**: https://panelbox.readthedocs.io
- **API Reference**: https://panelbox.readthedocs.io/en/latest/api/
- **Getting Started Guide**: [../../GETTING_STARTED.md](../../GETTING_STARTED.md)
- **Examples Overview**: [../../README.md](../../README.md)

### Related Examples
After completing this module, explore:
- **Static Models**: [../../static_models/](../../static_models/) - More OLS, FE, RE examples
- **Standard Errors**: [../../standard_errors/](../../standard_errors/) - Deep dive into robust inference
- **GMM Examples**: [../../gmm/](../../gmm/) - Dynamic panel models (advanced)

### Reference Materials
- **Dataset Documentation**: [../../datasets/README.md](../../datasets/README.md)
- **Utility Scripts**: [../../scripts/](../../scripts/) - Helper functions used in notebooks

## Need Help?

If you get stuck:
1. Check the documentation: https://panelbox.readthedocs.io
2. Review the solutions notebooks in `../../solutions/01_fundamentals/` (coming soon)
3. Consult the [Getting Started Guide](../../GETTING_STARTED.md) for common issues
4. Open an issue: https://github.com/panelbox/panelbox/issues

## Contributing

Found a typo or have suggestions for improvements?
- Open an issue describing the problem
- Submit a pull request with corrections
- Provide feedback in discussions

---

**Module Status**: Notebooks in development
**Last Updated**: 2026-02-16
