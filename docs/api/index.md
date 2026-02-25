---
title: "API Reference"
description: "Complete API reference for PanelBox — panel data econometrics in Python"
---

# API Reference

PanelBox provides 70+ models, 50+ diagnostic tests, and interactive HTML reports for panel data econometrics. This reference documents every public class, function, and result container.

## Quick Navigation

| Module | Import Path | Key Classes |
|--------|-------------|-------------|
| [Core](core.md) | `panelbox.core` | `PanelData`, `PanelResults`, `SerializableMixin` |
| [Static Models](static-models.md) | `panelbox.models.static` | `PooledOLS`, `FixedEffects`, `RandomEffects`, `BetweenEstimator`, `FirstDifferenceEstimator` |
| [GMM](gmm.md) | `panelbox.gmm` | `DifferenceGMM`, `SystemGMM`, `ContinuousUpdatedGMM`, `BiasCorrectedGMM` |
| [Spatial](spatial.md) | `panelbox.models.spatial` | `SpatialLag`, `SpatialError`, `SpatialDurbin`, `DynamicSpatialPanel`, `GeneralNestingSpatial` |
| [Frontier](frontier.md) | `panelbox.frontier` | `StochasticFrontier`, `FourComponentSFA`, True FE/RE |
| [Quantile](quantile.md) | `panelbox.models.quantile` | `PooledQuantile`, `FixedEffectsQuantile`, `CanayTwoStep`, `LocationScale` |
| [VAR](var.md) | `panelbox.var` | `PanelVAR`, `PanelVECM`, `PanelVARData` |
| [Discrete](discrete.md) | `panelbox.models.discrete` | `PooledLogit`, `PooledProbit`, `FixedEffectsLogit`, `MultinomialLogit`, `OrderedLogit` |
| [Count](count.md) | `panelbox.models.count` | `PooledPoisson`, `NegativeBinomial`, `PPML`, `ZeroInflatedPoisson` |
| [Censored & Selection](censored.md) | `panelbox.models.censored`, `.selection` | `PooledTobit`, `HonoreTrimmedEstimator`, `PanelHeckman` |
| [IV](iv.md) | `panelbox.models.iv` | `PanelIV` |
| [Standard Errors](standard-errors.md) | `panelbox.standard_errors` | `RobustStandardErrors`, `ClusteredStandardErrors`, `DriscollKraayStandardErrors` |
| [Marginal Effects](marginal-effects.md) | `panelbox.marginal_effects` | `compute_ame`, `compute_mem`, `compute_mer` |
| [Validation](validation.md) | `panelbox.validation` | `HausmanTest`, `WooldridgeARTest`, `ModifiedWaldTest`, `PesaranCDTest` |
| [Diagnostics](diagnostics.md) | `panelbox.diagnostics` | `hadri_test`, `breitung_test`, `kao_test`, `pedroni_test` |
| [Visualization](visualization.md) | `panelbox.visualization` | `ChartFactory`, `ChartRegistry`, themes |
| [Report](report.md) | `panelbox.report` | `ReportManager`, `HTMLExporter`, `LaTeXExporter` |
| [Experiment](experiment.md) | `panelbox.experiment` | `PanelExperiment`, result containers |
| [Datasets](datasets.md) | `panelbox.datasets` | `load_grunfeld`, `load_abdata`, `list_datasets` |

## Module Categories

### Data & Infrastructure

<div class="grid cards" markdown>

-   **[Core](core.md)**

    ---

    `PanelData` container, `PanelResults` base class, serialization, formula parsing.

    `from panelbox.core import PanelData, PanelResults`

-   **[Datasets](datasets.md)**

    ---

    Built-in example datasets for learning and testing.

    `from panelbox.datasets import load_grunfeld, load_abdata`

</div>

### Linear Models

<div class="grid cards" markdown>

-   **[Static Models](static-models.md)**

    ---

    Pooled OLS, Fixed Effects, Random Effects, Between, First Difference.

    `from panelbox.models.static import FixedEffects`

-   **[IV](iv.md)**

    ---

    Instrumental Variables / 2SLS for endogenous regressors.

    `from panelbox.models.iv import PanelIV`

</div>

### Dynamic Models

<div class="grid cards" markdown>

-   **[GMM](gmm.md)**

    ---

    Arellano-Bond, Blundell-Bond, CUE-GMM, Bias-Corrected GMM.

    `from panelbox.gmm import DifferenceGMM, SystemGMM`

-   **[VAR](var.md)**

    ---

    Panel VAR, VECM, impulse responses, Granger causality, forecasting.

    `from panelbox.var import PanelVAR, PanelVECM`

</div>

### Spatial Models

<div class="grid cards" markdown>

-   **[Spatial](spatial.md)**

    ---

    SAR, SEM, SDM, Dynamic Spatial, GNS with direct/indirect effects.

    `from panelbox.models.spatial import SpatialLag, SpatialDurbin`

</div>

### Nonlinear Models

<div class="grid cards" markdown>

-   **[Discrete Choice](discrete.md)**

    ---

    Logit, Probit, FE Logit, Multinomial, Conditional, Ordered models.

    `from panelbox.models.discrete import PooledLogit, MultinomialLogit`

-   **[Count](count.md)**

    ---

    Poisson, Negative Binomial, PPML, Zero-Inflated models.

    `from panelbox.models.count import PooledPoisson, PPML`

-   **[Censored & Selection](censored.md)**

    ---

    Tobit, Honore trimmed estimator, Heckman selection correction.

    `from panelbox.models.censored import PooledTobit`

-   **[Frontier](frontier.md)**

    ---

    Stochastic Frontier Analysis, Four-Component SFA, True FE/RE.

    `from panelbox.frontier import StochasticFrontier`

-   **[Quantile](quantile.md)**

    ---

    Pooled, FE, Canay, Location-Scale, Dynamic quantile regression.

    `from panelbox.models.quantile import PooledQuantile`

</div>

### Inference & Diagnostics

<div class="grid cards" markdown>

-   **[Standard Errors](standard-errors.md)**

    ---

    Robust, clustered, Driscoll-Kraay, Newey-West, PCSE, Spatial HAC.

    `from panelbox.standard_errors import robust_covariance, driscoll_kraay`

-   **[Marginal Effects](marginal-effects.md)**

    ---

    Average, at-mean, and at-representative marginal effects.

    `from panelbox.marginal_effects import compute_ame`

-   **[Validation](validation.md)**

    ---

    Hausman, Mundlak, Wooldridge AR, Breusch-Pagan, Pesaran CD, unit roots.

    `from panelbox.validation import HausmanTest, WooldridgeARTest`

-   **[Diagnostics](diagnostics.md)**

    ---

    Unit root, cointegration, specification, spatial diagnostics.

    `from panelbox.diagnostics.unit_root import hadri_test`

</div>

### Reporting & Workflow

<div class="grid cards" markdown>

-   **[Visualization](visualization.md)**

    ---

    28+ chart types with Plotly, professional/academic/presentation themes.

    `from panelbox.visualization import ChartFactory`

-   **[Report](report.md)**

    ---

    HTML, LaTeX, and Markdown export with interactive reports.

    `from panelbox.report import ReportManager`

-   **[Experiment](experiment.md)**

    ---

    Factory-based model management with automated validation.

    `from panelbox import PanelExperiment`

</div>

## Common Patterns

### Model Estimation

All model classes follow a consistent `model.fit()` pattern:

```python
from panelbox import FixedEffects

# Create model
model = FixedEffects("y ~ x1 + x2", data, entity_col="firm", time_col="year")

# Fit with default standard errors
result = model.fit()

# Fit with robust standard errors
result = model.fit(cov_type="robust")

# Fit with clustered standard errors
result = model.fit(cov_type="clustered")

# View results
print(result.summary())
```

### Result Objects

All estimation results inherit from `PanelResults` and share common attributes:

```python
result.params        # Coefficient estimates (pd.Series)
result.std_errors    # Standard errors (pd.Series)
result.tstats        # t-statistics (pd.Series)
result.pvalues       # p-values (pd.Series)
result.conf_int()    # Confidence intervals (pd.DataFrame)
result.rsquared      # R-squared
result.nobs          # Number of observations
result.summary()     # Formatted summary table
```

### Top-Level Imports

The most common classes are available directly from the `panelbox` namespace:

```python
import panelbox as pb

# Core
pb.PanelData, pb.PanelResults

# Static models
pb.PooledOLS, pb.FixedEffects, pb.RandomEffects

# GMM
pb.DifferenceGMM, pb.SystemGMM

# Datasets
pb.load_grunfeld(), pb.load_abdata()

# Experiment
pb.PanelExperiment
```

### Complete Workflow Example

```python
import panelbox as pb

# Load data
data = pb.load_grunfeld()

# Estimate models
fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
re = pb.RandomEffects("invest ~ value + capital", data, "firm", "year")
fe_result = fe.fit(cov_type="robust")
re_result = re.fit()

# Hausman test: FE vs RE
from panelbox.validation import HausmanTest
hausman = HausmanTest()
h_result = hausman.run(fe_result, re_result)
print(h_result.summary())

# Generate report
experiment = pb.PanelExperiment(data, "invest ~ value + capital", "firm", "year")
experiment.fit_all_models(names=["fe", "re"])
experiment.save_master_report("panel_analysis.html")
```

## See Also

- [Getting Started Guide](../getting-started/index.md) — Installation and first steps
- [Tutorials](../tutorials/index.md) — Step-by-step guides for each model family
- [Theory](../theory/panel-fundamentals.md) — Econometric theory and derivations
- [FAQ](../faq/general.md) — Frequently asked questions
