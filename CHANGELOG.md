# Changelog

All notable changes to PanelBox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Robust standard errors (HC0-HC3, Driscoll-Kraay, Newey-West)
- Additional GMM estimators (LIML, CUE)
- Cross-validation for panel data
- Jackknife inference
- Outlier detection and influence diagnostics
- Panel VAR models
- Cointegration tests

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
