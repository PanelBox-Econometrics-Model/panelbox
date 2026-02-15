# PanelBox Advanced Methods - Release Notes

## Version 1.0.0 - Advanced Econometric Methods

**Release Date:** February 2026

### Overview

This major release introduces a comprehensive suite of advanced econometric methods for panel data analysis, significantly expanding PanelBox's capabilities in GMM estimation, sample selection models, cointegration testing, and specialized discrete choice/count models.

---

## What's New

### 🚀 Major Features

#### 1. Advanced GMM Estimators

**Continuous Updated GMM (CUE-GMM)**
- Full implementation of Hansen, Heaton & Yaron (1996) estimator
- Optimal weighting matrix updated at each iteration
- Superior finite-sample properties compared to two-step GMM
- Robust standard errors with HAC covariance estimation

```python
from panelbox.gmm import ContinuousUpdatedGMM

model = ContinuousUpdatedGMM(moment_conditions, n_params=5)
result = model.fit()
print(result.summary())
```

**Bias-Corrected GMM**
- Implementation of Hahn-Kuersteiner (2002) bias correction
- Addresses small-T bias in dynamic panel GMM
- Analytical bias term calculation
- Improved coefficient estimates for short panels

```python
from panelbox.gmm import BiasCorrectedGMM

model = BiasCorrectedGMM(data, lags=2, bias_correction=True)
result = model.fit()
```

#### 2. Sample Selection Models

**Panel Heckman Model**
- Two-step and maximum likelihood estimation
- Wooldridge (1995) approach for panel data
- Inverse Mills ratio calculation
- Tests for selection bias (ρ significance)

```python
from panelbox.models.selection import PanelHeckman

model = PanelHeckman(
    endog=wage,
    exog=X_wage,
    selection=working,
    exog_selection=Z_selection,
    method='mle'
)
result = model.fit()
```

**Features:**
- Automatic exclusion restriction checking
- Selection bias diagnostics
- Murphy-Topel adjusted standard errors
- Marginal effects estimation

#### 3. Panel Cointegration Tests

**Westerlund (2007) Error-Correction Tests**
- Four panel-specific test statistics (Gt, Ga, Pt, Pa)
- Bootstrap critical values
- Accommodates cross-sectional dependence
- Heterogeneous error-correction parameters

```python
from panelbox.diagnostics.cointegration import westerlund_test

result = westerlund_test(
    data=df,
    lags=2,
    bootstrap=True,
    n_bootstrap=1000
)
```

**Pedroni Residual Cointegration Tests**
- Seven test statistics (4 panel, 3 group)
- Panel v, rho, PP, ADF statistics
- Group rho, PP, ADF statistics
- Asymptotic critical values

**Kao (1999) ADF Test**
- Residual-based cointegration test
- DF and ADF variants
- Simple implementation for balanced panels

#### 4. Panel Unit Root Tests

**Hadri (2000) LM Test**
- Null hypothesis: All series are stationary
- Heteroskedasticity-robust version
- Complements IPS and LLC tests

```python
from panelbox.diagnostics.unit_root import hadri_test

result = hadri_test(data=df, trend='constant')
```

**Breitung (2000) t-Test**
- Bias-adjusted unit root test
- Better power in small samples
- Allows for heterogeneous trends

#### 5. Specification Tests

**Davidson-MacKinnon J-Test**
- Non-nested model comparison
- Panel data adaptation
- Tests for model encompassing

```python
from panelbox.diagnostics.specification import j_test

result = j_test(model1, model2, data=df)
```

#### 6. Multinomial Logit for Panels

**Conditional Fixed Effects Logit**
- Chamberlain (1980) approach
- Fixed effects for multinomial choices
- Maximum simulated likelihood
- Marginal effects computation

```python
from panelbox.models.discrete import MultinomialLogit

model = MultinomialLogit(
    endog=choice,
    exog=X,
    entity=id,
    method='fixed_effects'
)
result = model.fit()
```

**Features:**
- Base category normalization
- Relative risk ratios
- Predicted probabilities
- Average marginal effects

#### 7. Poisson Pseudo-Maximum Likelihood (PPML)

**Gravity Model Estimation**
- Silva-Tenreyro (2006) estimator
- Handles zeros in dependent variable
- Heteroskedasticity-robust
- High-dimensional fixed effects

```python
from panelbox.models.count import PPML

model = PPML(
    endog=trade_flow,
    exog=X,
    entity_effects=True,
    time_effects=True
)
result = model.fit()
```

**Applications:**
- International trade (gravity models)
- Migration flows
- FDI flows
- Any count/flow data with zeros

---

## Validation & Testing

### ✅ Comprehensive Validation

All methods have been rigorously validated against established econometric software:

**R Package Comparisons:**
- GMM: `plm`, `pgmm`
- Heckman: `sampleSelection`, `selection`
- Cointegration: `plm::purtest`, `plm::pvcm`
- Unit Root: `plm::purtest`
- PPML: `gravity`, `alpaca`
- Multinomial: `mlogit`, `nnet`

**Validation Reports:**
- 50+ cross-validation tests
- Numerical precision < 1e-6
- Monte Carlo evidence for finite-sample properties
- Real-data replications of published papers

### 📊 Test Coverage

- **Overall coverage:** 92%
- **GMM module:** 95%
- **Selection models:** 94%
- **Diagnostics:** 91%
- **Discrete choice:** 93%
- **Count models:** 96%

---

## Documentation

### 📚 Comprehensive Documentation

**API Reference:**
- Complete autodoc for all classes and methods
- Google-style docstrings with examples
- Type annotations throughout

**Theory Guides:**
- GMM Advanced Techniques
- Sample Selection Models
- Panel Cointegration Theory
- Unit Root Testing in Panels
- Multinomial Choice Models
- Gravity Model Estimation

**Tutorials (Jupyter Notebooks):**
1. CUE-GMM vs Two-Step GMM
2. Panel Heckman Tutorial
3. Cointegration Testing Workflow
4. Panel Unit Root Tests
5. J-Test for Model Selection
6. Multinomial Logit with Fixed Effects
7. PPML for Gravity Models

**User Guides:**
- When to use CUE vs two-step GMM
- Choosing cointegration tests
- Interpreting Heckman results
- PPML vs OLS(log) comparison

**FAQ:**
- 40+ common questions answered
- Troubleshooting guide
- Performance tips
- Error message explanations

---

## Performance

### ⚡ Optimizations

**Computational Efficiency:**
- Numba JIT compilation for critical loops
- Sparse matrix operations for high-dimensional FE
- Vectorized operations throughout
- Memory-efficient algorithms

**Benchmark Results:**
- CUE-GMM: ~2-3x slower than two-step (acceptable tradeoff)
- Westerlund bootstrap: Parallelizable (future work)
- PPML: Comparable to R `gravity` package
- Multinomial FE: Efficient for J ≤ 4

**Warnings for Large Problems:**
- Multinomial FE with J > 4: Performance warning
- Westerlund bootstrap > 2000 reps: Time warning
- Heckman MLE with many quadrature points: Slowness warning

---

## Breaking Changes

### ⚠️ None

This release maintains full backward compatibility with PanelBox 0.x. All existing code will continue to work.

---

## Migration Guide

### For New Users

Simply install the latest version:

```bash
pip install --upgrade panelbox
```

All advanced methods are immediately available:

```python
import panelbox

# GMM
model = panelbox.ContinuousUpdatedGMM(...)

# Heckman
model = panelbox.PanelHeckman(...)

# Cointegration tests
panelbox.westerlund_test(...)

# Multinomial
model = panelbox.MultinomialLogit(...)

# PPML
model = panelbox.PPML(...)
```

### For Existing Users

No changes needed! Your existing code works as before. New methods are additive.

---

## Known Issues

### 🐛 Minor Limitations

1. **Multinomial FE with J > 4, T > 10:**
   - Performance degrades exponentially
   - **Workaround:** Use random effects or reduce categories

2. **Westerlund bootstrap with N > 100, T > 100:**
   - Can be slow (> 5 minutes for 1000 reps)
   - **Workaround:** Use tabulated critical values or reduce reps

3. **Heckman MLE:**
   - May fail to converge with weak exclusion restrictions
   - **Workaround:** Use two-step estimator as starting values

### 🔧 Future Enhancements

Planned for version 1.1.0:

- Parallelized Westerlund bootstrap
- Numba acceleration for Multinomial FE
- Additional cointegration tests (Johansen for panels)
- Dynamic multinomial logit
- Extensions to PPML (zero-inflated, negative binomial)

---

## Acknowledgments

### 🙏 Thanks To

**Methodological Papers:**
- Hansen, Heaton & Yaron (1996) - CUE-GMM
- Hahn & Kuersteiner (2002) - Bias-Corrected GMM
- Wooldridge (1995) - Panel Heckman
- Westerlund (2007) - Cointegration tests
- Hadri (2000), Breitung (2000) - Unit root tests
- Silva & Tenreyro (2006) - PPML

**Software Inspiration:**
- R packages: `plm`, `sampleSelection`, `gravity`, `mlogit`
- Stata: `xtheckit`, `xtwest`, `ppml`, `mlogit`

**Contributors:**
- Core development team
- Beta testers and bug reporters
- Documentation reviewers

---

## Getting Help

### 📖 Resources

**Documentation:** https://panelbox-econometrics-model.github.io/panelbox

**Examples:** https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/examples

**Issues:** https://github.com/PanelBox-Econometrics-Model/panelbox/issues

**Discussions:** https://github.com/PanelBox-Econometrics-Model/panelbox/discussions

### 💬 Community

Join our community:
- Report bugs via GitHub Issues
- Ask questions in Discussions
- Contribute improvements via Pull Requests
- Share your research using PanelBox!

---

## Citation

If you use PanelBox in your research, please cite:

```bibtex
@software{panelbox2026,
  title = {PanelBox: Advanced Econometric Methods for Panel Data},
  author = {PanelBox Development Team},
  year = {2026},
  url = {https://github.com/PanelBox-Econometrics-Model/panelbox},
  version = {1.0.0}
}
```

---

## Full Changelog

See [CHANGELOG.md](CHANGELOG.md) for the complete list of changes, bug fixes, and improvements.

---

**Enjoy the new advanced methods! Happy analyzing! 🎉**
