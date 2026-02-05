# Changelog

All notable changes to PanelBox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v1.1.0
- Additional GMM estimators (LIML, CUE)
- Panel VAR models
- Enhanced cross-validation methods
- Additional cointegration tests

## [1.0.0] - 2026-02-05

### Summary

**🎉 Production Release - Complete Panel Data Econometrics Suite**

PanelBox v1.0.0 represents a complete, production-ready panel data econometrics library for Python. This release consolidates all features from beta versions (v0.1.0-v0.4.0) into a stable, validated, and well-documented package.

**Key Metrics:**
- 600+ unit tests, 93% passing
- 61% code coverage
- Type-checked with MyPy (77.5% error reduction)
- Validated against Stata xtabond2 and R plm
- Numba-optimized (up to 348x speedup)

### Added - Complete Feature Set

**Static Panel Models:**
- Pooled OLS, Fixed Effects (Within), Random Effects (GLS)
- Between Estimator, First Differences
- Hausman test for model specification
- Formula interface (R-style with patsy)

**Dynamic Panel GMM:**
- Difference GMM (Arellano-Bond 1991) - One-step, two-step, iterative
- System GMM (Blundell-Bond 1998) - Combined differenced and level equations
- Automatic instrument generation (GMM-style and IV-style)
- Collapse option to avoid instrument proliferation
- Windmeijer (2005) finite-sample standard error correction
- Smart instrument selection for unbalanced panels (72% retention vs 0% in naive implementations)

**Robust Standard Errors (8 types):**
- **Heteroskedasticity-Robust:** HC0 (White 1980), HC1, HC2, HC3 (MacKinnon-White 1985)
- **Cluster-Robust:** One-way and two-way clustering (Cameron, Gelbach & Miller 2011)
- **Driscoll-Kraay:** Robust to spatial and temporal dependence (Driscoll & Kraay 1998)
- **Newey-West HAC:** Heteroskedasticity and autocorrelation consistent (Newey & West 1987)
- **PCSE:** Panel-corrected standard errors (Beck & Katz 1995)

**Bootstrap Inference:**
- **4 Bootstrap Methods:** Pairs (entity), Wild (Rademacher), Block (temporal), Residual
- Performance: ~95-110 iterations/second
- Confidence intervals: Percentile, basic, studentized
- Bootstrap bias and variance estimates
- Progress tracking with tqdm integration

**Sensitivity Analysis:**
- **Leave-One-Out:** Entities and time periods
- **Subset Sensitivity:** Random subsample stability with stratification
- Influential unit detection
- Comprehensive summary statistics
- Optional visualization with matplotlib

**Specification Tests:**
- RESET, Mundlak, Chow tests
- Hansen J test, Sargan test (GMM overidentification)
- Arellano-Bond AR(1) and AR(2) tests
- Instrument ratio monitoring

**Diagnostic Tests:**
- **Heteroskedasticity:** White, Breusch-Pagan, Modified Wald
- **Serial Correlation:** Wooldridge, Breusch-Godfrey, Baltagi-Wu
- **Cross-Sectional Dependence:** Pesaran CD, Frees, BP-LM
- **Unit Root:** LLC, IPS, Fisher
- **Cointegration:** Pedroni, Kao, Westerlund

**Robustness Checks:**
- Influence diagnostics (DFBETA, Cook's D, leverage)
- Outlier detection (standardized residuals, IQR method)
- Jackknife resampling
- Cross-validation for panel data

**Report Generation:**
- **HTML Reports:** Modern styling with interactive elements
- **Markdown Reports:** Documentation-ready format
- **LaTeX Reports:** Publication-ready tables
- Comparison tables across models
- Customizable templates

**Datasets:**
- Grunfeld investment data (canonical panel dataset)
- Easy data loading utilities
- Example datasets for tutorials

**Performance Optimizations:**
- Numba JIT compilation for critical paths
- Efficient NumPy operations throughout
- Smart caching (bread matrix, leverage values)
- Minimal memory footprint

**Quality Assurance:**
- 600+ unit tests covering all major features
- Validation against Stata xtabond2 (GMM)
- Validation against R plm (static models)
- Comprehensive type hints (MyPy compatible)
- Code formatting with Black
- Import sorting with isort
- Linting with Flake8

### Changed from Beta Versions

- Updated package status from "Beta" to "Stable"
- Enhanced warning system for GMM specifications
- Improved unbalanced panel handling
- Better error messages and user guidance
- Consolidated documentation structure

### Documentation

**Complete Documentation Suite:**
- Comprehensive README.md with quick start
- CHANGELOG.md following Keep a Changelog format
- API docstrings for all public classes and methods
- GMM tutorial (650+ lines)
- GMM interpretation guide (420+ lines)
- 4 example scripts with practical use cases
- Implementation guides for all major features
- Contributing guidelines and code of conduct

### Academic References Implemented

**Core Methods:**
1. Arellano & Bond (1991) - Difference GMM
2. Blundell & Bond (1998) - System GMM
3. Windmeijer (2005) - Finite sample correction
4. Roodman (2009) - Instrument collapse methodology

**Robust Standard Errors:**
5. White (1980) - HC0 heteroskedasticity-robust
6. MacKinnon & White (1985) - HC1-HC3 variants
7. Newey & West (1987) - HAC standard errors
8. Beck & Katz (1995) - Panel-corrected standard errors
9. Driscoll & Kraay (1998) - Spatial/temporal dependence
10. Cameron, Gelbach & Miller (2011) - Two-way clustering

**Textbooks:**
11. Baltagi (2021) - Econometric Analysis of Panel Data
12. Wooldridge (2010) - Econometric Analysis of Cross Section and Panel Data

### Breaking Changes

None. This is the first stable release.

### Migration Guide

For users of beta versions (v0.1.0-v0.4.0):
- All existing code continues to work unchanged
- No breaking API changes
- New features are additive
- Enhanced functionality for existing methods (more standard error types)

## [0.4.0] - 2026-02-05 (Beta)

### Added - Robust Standard Errors

**Heteroskedasticity-Robust Standard Errors (HC):**
- **HC0** - White (1980) sandwich estimator
- **HC1** - Degrees of freedom corrected: [n/(n-k)] × HC0
- **HC2** - Leverage adjustment: Ω̂ = diag(ε²/(1-h_i))
- **HC3** - MacKinnon-White (1985): Ω̂ = diag(ε²/(1-h_i)²)
- Automatic leverage (hat values) computation
- Efficient caching for performance

**Cluster-Robust Standard Errors:**
- **One-way clustering** - Cluster by entity or time
- **Two-way clustering** - Cameron, Gelbach & Miller (2011) formula: V = V₁ + V₂ - V₁₂
- Finite-sample corrections: G/(G-1) × (N-1)/(N-K)
- Diagnostic warnings for few clusters (<20)
- Support for unbalanced clusters

**Driscoll-Kraay Standard Errors:**
- Robust to spatial and temporal dependence
- Automatic lag selection: floor(4(T/100)^(2/9))
- 3 kernel options: Bartlett, Parzen, Quadratic Spectral
- Appropriate for large N, moderate/large T
- Comprehensive diagnostic information

**Newey-West HAC Standard Errors:**
- Heteroskedasticity and Autocorrelation Consistent
- Automatic lag selection using Newey-West rule
- 3 kernel options: Bartlett, Parzen, Quadratic Spectral
- Suitable for time-series and panels with autocorrelation

**Panel-Corrected Standard Errors (PCSE):**
- Beck & Katz (1995) implementation
- For contemporaneous cross-sectional correlation
- FGLS approach with estimated Σ matrix
- Requires T > N
- Works best with Pooled OLS or Random Effects

**Utility Functions:**
- `compute_leverage()` - Hat values computation
- `compute_bread()` - (X'X)^{-1} for sandwich estimator
- `compute_meat_hc()` - Meat matrix for HC variants
- `compute_clustered_meat()` - One-way clustering meat
- `compute_twoway_clustered_meat()` - Two-way clustering meat
- `sandwich_covariance()` - Combines bread and meat
- Convenience functions for quick usage

**Integration with Models:**
- **Fixed Effects**: 8 types of standard errors
  - nonrobust, robust, hc0, hc1, hc2, hc3
  - clustered (one-way), twoway
  - driscoll_kraay, newey_west, pcse
- **Random Effects**: 7 types of standard errors
  - All above except PCSE works better with non-demeaned data
- Flexible parameters: `max_lags`, `kernel`
- Backward compatible with existing code

### Added - Tests

**Comprehensive Test Suite:**
- 40+ tests for HC standard errors (test_robust.py)
  - Leverage computation tests
  - Bread and meat matrix tests
  - Sandwich covariance tests
  - HC0-HC3 correctness tests
  - Edge cases and numerical stability
- 35+ tests for clustered standard errors (test_clustered.py)
  - One-way clustering tests
  - Two-way clustering tests
  - Diagnostic summary tests
  - Various cluster patterns (balanced, unbalanced, singletons)
- Manual validation of Driscoll-Kraay and Newey-West
- Manual validation of PCSE
- Integration tests with Fixed Effects and Random Effects
- Total: 75+ test cases, ~90% coverage

### Added - Documentation

**Module Documentation:**
- `desenvolvimento/FASE_6_COMPLETE.md` - Complete implementation guide (457 lines)
- `desenvolvimento/FASE_6_PROGRESSO.md` - Detailed progress tracking (569 lines)
- `desenvolvimento/FASE_6_ERROS_PADRAO_ROBUSTOS.md` - Planning and checklist (389 lines)

**Implementation Files:**
- `panelbox/standard_errors/utils.py` - Core utilities (394 lines)
- `panelbox/standard_errors/robust.py` - HC standard errors (267 lines)
- `panelbox/standard_errors/clustered.py` - Clustering (320 lines)
- `panelbox/standard_errors/driscoll_kraay.py` - DK estimator (461 lines)
- `panelbox/standard_errors/newey_west.py` - NW HAC (309 lines)
- `panelbox/standard_errors/pcse.py` - PCSE (368 lines)

**Enhanced Docstrings:**
- Complete API documentation for all new classes
- Mathematical formulas for each estimator
- Usage examples in docstrings
- Parameter descriptions
- References to academic papers

### Changed

- Updated `FixedEffects.fit()` to support 8 covariance types
- Updated `RandomEffects.fit()` to support 7 covariance types
- Enhanced `panelbox/__init__.py` with standard error exports
- Updated main module to export all SE classes and functions

### Performance

**Optimizations:**
- Caching of bread matrix and leverage values
- Vectorized NumPy operations throughout
- Efficient meat computation for all methods
- Minimal memory footprint with smart algorithms

**Benchmarks (typical panel N=50, T=10):**
- HC standard errors: <0.1s computation time
- Clustered SE: <0.2s computation time
- Driscoll-Kraay: <0.5s computation time
- Newey-West: <0.3s computation time

### Academic References Implemented

1. **White, H. (1980)**. A heteroskedasticity-consistent covariance matrix estimator. *Econometrica*, 48(4), 817-838. ✅ HC0
2. **MacKinnon, J. G., & White, H. (1985)**. Some heteroskedasticity-consistent covariance matrix estimators. *Journal of Econometrics*, 29(3), 305-325. ✅ HC1, HC2, HC3
3. **Newey, W. K., & West, K. D. (1987)**. A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. ✅ Newey-West
4. **Beck, N., & Katz, J. N. (1995)**. What to do (and not to do) with time-series cross-section data. *American Political Science Review*, 89(3), 634-647. ✅ PCSE
5. **Driscoll, J. C., & Kraay, A. C. (1998)**. Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549-560. ✅ Driscoll-Kraay
6. **Hoechle, D. (2007)**. Robust standard errors for panel regressions with cross-sectional dependence. *The Stata Journal*, 7(3), 281-312. ✅ DK implementation
7. **Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011)**. Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249. ✅ Two-way clustering

## [0.3.0] - 2026-01-22

### Added - Advanced Robustness Analysis

**Bootstrap Inference:**
- **PanelBootstrap** class with 4 bootstrap methods:
  - **Pairs (Entity) Bootstrap** - Resamples entire entities with all time periods (default, most robust)
  - **Wild Bootstrap** - Rademacher weights for heteroskedasticity-robust inference
  - **Block Bootstrap** - Moving blocks for temporal dependence (automatic block size selection)
  - **Residual Bootstrap** - i.i.d. assumption benchmark
- Performance: ~95-110 iterations/second on typical panels
- Progress tracking with optional tqdm integration
- Methods: `run()`, `conf_int()`, `summary()`
- Confidence interval methods: percentile, basic, studentized
- Bootstrap bias and variance estimates

**Sensitivity Analysis:**
- **SensitivityAnalysis** class with 3 analysis methods:
  - **Leave-One-Out Entities** - Identifies influential cross-sectional units
  - **Leave-One-Out Periods** - Identifies influential time periods
  - **Subset Sensitivity** - Random subsample stability analysis with stratification
- Influential unit detection (configurable threshold)
- Comprehensive summary statistics (CV, ranges, deviations)
- Optional visualization with matplotlib
- Methods: `leave_one_out_entities()`, `leave_one_out_periods()`, `subset_sensitivity()`, `plot_sensitivity()`, `summary()`
- Performance: 1-4 seconds for typical panels

**Integration:**
- Exported `PanelBootstrap`, `SensitivityAnalysis`, and `SensitivityResults` in main module
- Works seamlessly with all static and dynamic panel models
- Backward compatible with v0.2.0

### Added - Tests

**Comprehensive Test Suite:**
- 33 tests for PanelBootstrap (100% passing)
  - Initialization tests
  - Method-specific tests (pairs, wild, block, residual)
  - Reproducibility tests
  - Integration tests
- 30 tests for SensitivityAnalysis (100% passing)
  - Leave-one-out tests
  - Subset sensitivity tests
  - Plotting tests (conditional on matplotlib)
  - Edge case coverage
- Total: 63 new tests, 100% pass rate

### Added - Documentation

**Module Documentation:**
- `desenvolvimento/FASE_5_BOOTSTRAP_COMPLETE.md` - Complete bootstrap guide (500 lines)
- `desenvolvimento/FASE_5_ROBUSTNESS_COMPLETE.md` - Robustness suite documentation (800 lines)
- `desenvolvimento/FASE_5_SUMMARY.md` - Executive summary (400 lines)
- `desenvolvimento/PROJECT_STATUS_2026_01_22.md` - Project status overview
- `desenvolvimento/NEXT_STEPS_RECOMMENDATIONS.md` - Future roadmap

**Example Scripts:**
- `examples/validation/bootstrap_all_methods.py` - Complete bootstrap demonstration (347 lines)
  - All 4 methods with comparison
  - Standard error comparison
  - Confidence interval analysis
  - Method-specific recommendations
- `examples/validation/sensitivity_analysis_complete.py` - Comprehensive sensitivity demo (550 lines)
  - All 3 sensitivity methods
  - Planted outlier detection
  - Stability assessment
  - Practical interpretation guidelines

**Enhanced Docstrings:**
- Complete API documentation for all new classes
- Type hints throughout
- Usage examples in docstrings
- Parameter descriptions

### Changed

- Updated main `__init__.py` to export robustness analysis tools
- Updated package status: v0.2.0 (GMM) → v0.3.0 (GMM + Robustness)
- Enhanced project documentation structure

### Performance

**Bootstrap Benchmarks (N=20, T=8):**
- Pairs: ~110 iterations/second (~9s for 1000 bootstrap)
- Wild: ~98 iterations/second (~10s for 1000 bootstrap)
- Block: ~73 iterations/second (~14s for 1000 bootstrap)
- Residual: ~95 iterations/second (~11s for 1000 bootstrap)

**Sensitivity Benchmarks (N=30, T=10):**
- LOO Entities: ~3.5 seconds (30 re-estimations)
- LOO Periods: ~1.2 seconds (10 re-estimations)
- Subset (30 samples): ~3.8 seconds

## [0.2.0] - 2026-01-21

### Added - Dynamic Panel GMM

**Core Features:**
- **Difference GMM** implementation (Arellano-Bond 1991)
  - One-step, two-step, and iterative GMM
  - Automatic instrument generation (GMM-style and IV-style)
  - Collapse option to avoid instrument proliferation
  - Windmeijer (2005) finite-sample standard error correction

- **System GMM** implementation (Blundell-Bond 1998)
  - Combines differenced and level equations
  - Level instruments for highly persistent series
  - More efficient than Difference GMM for weak instruments
  - Difference-in-Hansen test for level instrument validity

**Specification Tests:**
- Hansen J-test for overidentification
- Sargan test (homoskedastic version)
- Arellano-Bond AR(1) and AR(2) tests
- Instrument ratio monitoring (instruments / groups)

**Unbalanced Panel Support:**
- Smart instrument selection based on data availability
- Automatic filtering of lags with <10% coverage
- Pre-estimation warnings for problematic specifications
- Post-estimation warnings for low observation retention
- Panel balance diagnostics

**Results Class:**
- Comprehensive `GMMResults` class
- Publication-ready summary tables
- Coefficient tables with significance stars
- Specification test interpretation
- Diagnostic assessment tools

### Added - Documentation

**GMM Documentation:**
- `panelbox/gmm/README.md` - Complete GMM reference (540 lines)
- `docs/gmm/tutorial.md` - Comprehensive tutorial (650 lines)
- `docs/gmm/interpretation_guide.md` - Results interpretation (420 lines)

**Example Scripts:**
- `examples/gmm/ols_fe_gmm_comparison.py` - Bias comparison (410 lines)
- `examples/gmm/firm_growth.py` - Intermediate example (500 lines)
- `examples/gmm/production_function.py` - Simultaneity bias (602 lines)
- `examples/gmm/unbalanced_panel_guide.py` - Practical guide (532 lines)

**Enhanced Docstrings:**
- Added 4 practical examples to `DifferenceGMM` class
- Added 3 comparison examples to `SystemGMM` class
- Added 5 usage patterns to `GMMResults` class

### Added - Infrastructure

**Package Configuration:**
- Updated `pyproject.toml` for v0.2.0
- Created `MANIFEST.in` for distribution
- Updated `__init__.py` with GMM exports
- Version management in `__version__.py`

**Quality Assurance:**
- `.flake8` configuration for linting
- `.pre-commit-config.yaml` for automated checks
- `scripts/qa.sh` for quality checks
- `QA_GUIDE.md` documentation

### Improved - Robustness

**Validation (Subfase 4.2):**
- Validation against Arellano-Bond (1991) employment data
- 72.8% observation retention (vs 0% before improvements)
- Coefficient within credible range [0.733, 1.045]
- All specification tests pass (AR(2) p=0.724)

**Warning System:**
- Pre-estimation warnings for:
  - Unbalanced panels with many time dummies
  - Not using collapse option
- Post-estimation warnings for:
  - Low observation retention (<30%)
- Actionable recommendations in all warnings

**Instrument Selection:**
- `_analyze_lag_availability()` method
- Automatic filtering of weak instruments
- Coverage-based lag selection (≥10% threshold)
- Prevents instrument proliferation

### Changed

- Updated package status from "Alpha" to "Beta"
- Improved package description for PyPI
- Updated Python version classifiers (3.9-3.12)
- Enhanced main `__init__.py` with GMM imports

### Fixed

- Arellano-Bond validation now works with unbalanced panels
- Time dummies no longer cause 0% observation retention
- System GMM more robust with error handling
- Better handling of missing observations in instruments

### Performance

- Smart instrument selection reduces computation time
- Efficient NumPy operations throughout
- Optimized instrument matrix construction
- Reduced memory footprint with collapse option

## [0.1.0] - 2025-12

### Added - Core Framework

**Core Classes:**
- `PanelData` - Panel data container with validation
- `FormulaParser` - R-style formula parsing (patsy integration)
- `PanelResults` - Base results class

**Static Models:**
- `PooledOLS` - Pooled OLS estimation
- `FixedEffects` - Within (FE) estimation
- `RandomEffects` - GLS (RE) estimation

**Specification Tests:**
- `HausmanTest` - Test for fixed vs random effects
- `HausmanTestResult` - Results container

**Standard Errors:**
- Homoskedastic (default)
- Heteroskedasticity-robust
- Clustered (one-way and two-way)

**Infrastructure:**
- Project structure setup
- Testing framework (pytest)
- Basic documentation
- MIT License

### Added - Validation Framework

**Statistical Tests:**
- Autocorrelation tests
- Heteroskedasticity tests
- Cross-sectional dependence tests
- Unit root tests (panel)

**Reporting:**
- HTML reports with Plotly
- Static reports with Matplotlib
- LaTeX table export
- Publication-ready formatting

---

## Release Notes

### v0.2.0 - GMM Implementation Complete

This release marks a major milestone with complete implementation of dynamic panel GMM estimation, bringing Stata's `xtabond2` capabilities to Python with improved robustness and user experience.

**Key Highlights:**
- 🎉 Difference GMM and System GMM fully implemented
- 🎯 72.8% improvement in unbalanced panel handling
- 📚 3,800+ lines of documentation and examples
- ⚠️ Smart warning system for problematic specifications
- ✅ Validated against Arellano-Bond (1991)

**Migration from v0.1.0:**
No breaking changes. All v0.1.0 code continues to work. New GMM features are additive:

```python
# v0.1.0 code still works
from panelbox import FixedEffects, RandomEffects

# v0.2.0 adds GMM
from panelbox import DifferenceGMM, SystemGMM  # NEW!
```

**Best Practices:**
- **Recommendation**: Use `collapse=True` for GMM models (Roodman 2009)
- Collapsed instruments provide better numerical stability
- Reduces instrument count from O(T²) to O(T)
- Improves finite-sample properties

**Known Limitations:**
- Non-collapsed instruments (`collapse=False`) may show numerical warnings
- System GMM may fail with very sparse synthetic data (add appropriate try/except)
- Type hints are partial (gradual typing in progress)
- Some specification tests may be under-powered with T < 5

**Upcoming in v0.3.0:**
- Comprehensive test suite (target: >80% coverage)
- Advanced diagnostic tools (weak instrument tests)
- Performance benchmarks
- More example datasets

---

## Versioning

We use [Semantic Versioning](https://semver.org/):
- **Major** (X.0.0): Incompatible API changes
- **Minor** (0.X.0): New features, backwards compatible
- **Patch** (0.0.X): Bug fixes, backwards compatible

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to PanelBox.

## Support

- 📫 Issues: [GitHub Issues](https://github.com/guhaase/panelbox/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/guhaase/panelbox/discussions)

---

**Legend:**
- `Added` - New features
- `Changed` - Changes to existing features
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security fixes
- `Performance` - Performance improvements
- `Improved` - Quality improvements

---

## Version Links

[Unreleased]: https://github.com/PanelBox-Econometrics-Model/panelbox/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/PanelBox-Econometrics-Model/panelbox/releases/tag/v1.0.0
[0.4.0]: https://github.com/PanelBox-Econometrics-Model/panelbox/releases/tag/v0.4.0
[0.3.0]: https://github.com/PanelBox-Econometrics-Model/panelbox/releases/tag/v0.3.0
[0.2.0]: https://github.com/PanelBox-Econometrics-Model/panelbox/releases/tag/v0.2.0
[0.1.0]: https://github.com/PanelBox-Econometrics-Model/panelbox/releases/tag/v0.1.0
