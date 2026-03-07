<div align="center">
  <img src="https://raw.githubusercontent.com/PanelBox-Econometrics-Model/panelbox/main/docs/assets/images/logo.svg" alt="PanelBox Logo" width="400">

  <h1>PanelBox</h1>

  <p><strong>Panel Data Econometrics in Python</strong></p>

[![CI](https://github.com/PanelBox-Econometrics-Model/panelbox/actions/workflows/tests.yml/badge.svg)](https://github.com/PanelBox-Econometrics-Model/panelbox/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox/branch/main/graph/badge.svg)](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI version](https://badge.fury.io/py/panelbox.svg)](https://badge.fury.io/py/panelbox)
[![Python versions](https://img.shields.io/pypi/pyversions/panelbox)](https://pypi.org/project/panelbox/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Development Status](https://img.shields.io/badge/development%20status-stable-brightgreen)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/panelbox?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/panelbox)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18828968.svg)](https://doi.org/10.5281/zenodo.18828968)
[![Documentation](https://readthedocs.org/projects/panelbox/badge/?version=latest)](https://panelbox.readthedocs.io/)

</div>

---

PanelBox is a comprehensive Python library for panel data econometrics, with **70+ models across 13 families**, **50+ diagnostic tests**, and **35+ interactive charts**. It brings the capabilities of Stata's `xtabond2`, `xtreg`, `xtfrontier`, and R's `plm`, `splm`, `frontier` to Python with a modern, unified API.

## Installation

```bash
pip install panelbox
```

## Quick Start

```python
import panelbox as pb

# Load bundled dataset (103 datasets available)
data = pb.datasets.load_grunfeld()

# Fixed Effects model
fe = pb.FixedEffects(
    formula="invest ~ value + capital",
    data=data,
    entity_col="firm",
    time_col="year"
)
results = fe.fit(cov_type='clustered')
print(results.summary())
```

## Model Families

### Static Panel Models

| Model | Description |
|-------|-------------|
| `PooledOLS` | Pooled OLS estimation |
| `FixedEffects` | Within estimator (entity/time/two-way) |
| `RandomEffects` | GLS estimation |
| `BetweenEstimator` | Between-groups estimator |
| `FirstDifferenceEstimator` | First-difference estimator |

### Dynamic Panel GMM

| Model | Description |
|-------|-------------|
| `DifferenceGMM` | Arellano-Bond (1991) |
| `SystemGMM` | Blundell-Bond (1998) |
| `ContinuousUpdatedGMM` | CUE-GMM (Hansen-Heaton-Yaron 1996) |
| `BiasCorrectedGMM` | Hahn-Kuersteiner (2002) bias correction |

Full diagnostic suite: Hansen J, Sargan, AR(1)/AR(2), Windmeijer correction, instrument ratio monitoring, overfit diagnostics.

### Panel VAR

| Model | Description |
|-------|-------------|
| `PanelVAR` | Panel Vector Autoregression (OLS/GMM) |
| `PanelVECM` | Panel Vector Error Correction Model |

Includes IRF, FEVD, Granger causality network visualization, lag selection (AIC/BIC/HQIC), and Johansen cointegration rank test.

### Spatial Models

| Model | Description |
|-------|-------------|
| `SpatialLag` | Spatial Autoregressive Model (SAR) |
| `SpatialError` | Spatial Error Model (SEM) |
| `SpatialDurbin` | Spatial Durbin Model (SDM) |
| `GeneralNestingSpatial` | General Nesting Spatial (GNS) |
| `DynamicSpatialPanel` | Dynamic spatial panel models |

### Stochastic Frontier Analysis

| Model | Description |
|-------|-------------|
| `StochasticFrontier` | SFA with half-normal, exponential, truncated-normal, gamma |
| `FourComponentSFA` | Persistent/transient inefficiency decomposition |

JLMS, BC, and Mode efficiency estimators. TFP decomposition and frontier visualization.

### Count Data Models

| Model | Description |
|-------|-------------|
| `PoissonFixedEffects` | Conditional MLE (Hausman-Hall-Griliches 1984) |
| `RandomEffectsPoisson` | RE Poisson (Gamma/Normal mixing) |
| `NegativeBinomial` | NB2 for overdispersion |
| `ZeroInflatedPoisson` | ZIP model |
| `ZeroInflatedNegativeBinomial` | ZINB model |
| `PPML` | Poisson Pseudo-ML (gravity models) |

### Discrete Choice Models

| Model | Description |
|-------|-------------|
| `FixedEffectsLogit` | Conditional logit (Chamberlain 1980) |
| `RandomEffectsProbit` | RE probit with GHQ integration |
| `OrderedLogit` / `OrderedProbit` | Ordered choice models |
| `MultinomialLogit` | Multinomial choice (FE/RE/Pooled) |

### Quantile Regression

| Model | Description |
|-------|-------------|
| `FixedEffectsQuantile` | Koenker (2004) FE quantile regression |
| `CanayTwoStep` | Canay (2011) two-step estimator |
| `LocationScale` | MSS (2019) location-scale models |
| `DynamicQuantile` | Dynamic panel quantile |
| `QuantileTreatmentEffects` | Quantile treatment effects |

### Selection & Censored Models

| Model | Description |
|-------|-------------|
| `PanelHeckman` | Two-step Heckman (Wooldridge 1995) and MLE |
| `PanelIV` | Panel IV/2SLS estimation |

## Diagnostic Tests (50+)

| Category | Tests |
|----------|-------|
| **Unit Root** | LLC, IPS, Fisher, Hadri, Breitung |
| **Cointegration** | Kao, Pedroni (7 stats), Westerlund (4 stats) |
| **Specification** | Hausman, Mundlak, RESET, Chow, Davidson-MacKinnon J/Cox |
| **Heteroskedasticity** | Breusch-Pagan, White, Modified Wald |
| **Serial Correlation** | Wooldridge AR, Breusch-Godfrey, Baltagi-Wu |
| **Cross-Sectional Dependence** | Pesaran CD, Frees, Breusch-Pagan LM |
| **Spatial** | LM Lag/Error (standard + robust), Moran's I, Local LISA |
| **GMM** | Hansen J, Sargan, AR(1)/AR(2), weak instruments |
| **Frontier** | LR, Wald, skewness, Vuong, inefficiency presence |

## Robust Standard Errors (8 types)

- **HC0-HC3**: Heteroskedasticity-consistent (White, leverage-adjusted)
- **Clustered**: One-way (entity/time) and two-way (Cameron-Gelbach-Miller 2011)
- **Driscoll-Kraay**: Spatial and temporal dependence
- **Newey-West**: HAC for serial correlation
- **PCSE**: Panel-corrected (Beck-Katz 1995)
- **Spatial HAC**: For spatial panel models

## Visualization (35+ interactive charts)

```python
from panelbox.visualization import (
    create_residual_diagnostics,
    create_validation_charts,
    create_comparison_charts,
    create_panel_charts,
    export_charts,
)

# Residual diagnostics (Q-Q, fitted vs residual, scale-location, etc.)
charts = create_residual_diagnostics(results)
export_charts(charts, "diagnostics.html")

# Entity/time effects, between-within decomposition, panel structure
charts = create_panel_charts(results)
```

Three professional themes: `professional`, `academic`, `presentation`. Export to HTML, JSON, PNG, SVG, PDF.

## Experiment Pattern

The `PanelExperiment` class provides a factory-based workflow for comparing models:

```python
import panelbox as pb

data = pb.datasets.load_grunfeld()

experiment = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# Fit and compare models
experiment.fit_all_models(names=['pooled', 'fe', 're'])
comparison = experiment.compare_models(['pooled', 'fe', 're'])
print(f"Best model: {comparison.best_model}")

# Validate specification
validation = experiment.validate_model('fe')
validation.save_html('validation.html', test_type='validation')

# Residual diagnostics
residuals = experiment.analyze_residuals('fe')
print(residuals.summary())

# Master report linking all sub-reports
experiment.save_master_report('report.html', theme='professional', reports=[...])
```

## Bundled Datasets (103 datasets)

```python
from panelbox.datasets import load_dataset, list_datasets, list_categories

# Browse categories
print(list_categories())
# ['censored', 'count', 'diagnostics', 'discrete', 'frontier', 'gmm',
#  'marginal_effects', 'production', 'quantile', 'spatial', 'standard_errors',
#  'validation', 'var']

data = load_dataset("healthcare_visits")
grunfeld = load_dataset("grunfeld")
```

All 80+ example notebooks use `load_dataset()` and work directly in Google Colab.

## Comparison with Other Packages

| Feature | PanelBox | linearmodels | pyfixest | splm (R) |
|---------|----------|--------------|----------|----------|
| Static panel (FE/RE) | 5 models | 5 models | 2 models | - |
| Dynamic GMM | 4 models | - | - | - |
| Spatial models | 5 models | - | - | 4 models |
| Count data | 9 models | - | Poisson | - |
| Discrete choice | 9 models | - | - | - |
| Quantile regression | 8 models | - | - | - |
| Stochastic frontier | 2 models | - | - | - |
| Panel VAR/VECM | 2 models | - | - | - |
| Diagnostic tests | 50+ | ~5 | ~5 | ~10 |
| Interactive charts | 35+ | - | - | - |
| Robust SE types | 8 | 4 | 3 | 2 |
| Bundled datasets | 103 | 10 | 5 | - |

## Requirements

- Python >= 3.9
- NumPy, Pandas, SciPy, statsmodels, scikit-learn
- Plotly, Matplotlib, Seaborn (visualization)
- Numba, Joblib (performance)

See `pyproject.toml` for full dependency list.

## Documentation

- [User Guide](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/docs/user-guide) - Comprehensive guides for all model families
- [API Reference](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/docs/api) - Full API documentation
- [Tutorials](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/examples/tutorials) - Interactive Jupyter notebooks
- [Examples](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/examples) - 80+ example notebooks across all model families
- [Theory](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/docs/theory) - Econometric theory guides
- [Benchmarks](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/docs/benchmarks) - Validation against R/Stata

## Citation

```bibtex
@software{panelbox2026,
  author = {Haase, Gustavo and Dourado, Paulo},
  title = {PanelBox: Panel Data Econometrics in Python},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/PanelBox-Econometrics-Model/panelbox}
}
```

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/LICENSE).

## Support

- [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues)
- [Documentation](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/docs)
- [Discussions](https://github.com/PanelBox-Econometrics-Model/panelbox/discussions)
- [Changelog](https://github.com/PanelBox-Econometrics-Model/panelbox/blob/main/CHANGELOG.md)

---

**Made with care for econometricians and researchers**
