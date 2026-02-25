---
title: "General FAQ"
description: "Frequently asked questions about PanelBox — installation, data formats, model selection, standard errors, and results interpretation."
---

# General FAQ

Common questions for beginners and intermediate users of PanelBox.

!!! tip "Looking for more?"
    - **Advanced methods** (GMM, VAR, Heckman, cointegration): [Advanced FAQ](advanced.md)
    - **Spatial econometrics**: [Spatial FAQ](spatial.md)
    - **Error messages and debugging**: [Troubleshooting](troubleshooting.md)

---

## Getting Started

??? question "How do I install PanelBox?"

    Install from PyPI:

    ```bash
    pip install panelbox
    ```

    For the full installation with all optional dependencies (spatial, visualization, reports):

    ```bash
    pip install panelbox[all]
    ```

    See the [Getting Started guide](../getting-started/index.md) for detailed instructions.

??? question "What Python versions are supported?"

    PanelBox supports **Python 3.9 and later**. We recommend Python 3.10+ for the best experience.

??? question "What data format does PanelBox expect?"

    PanelBox expects a **long-format pandas DataFrame** with one row per entity-time observation. You specify the entity column and time column when creating a model:

    ```python
    from panelbox import FixedEffects

    # data must have columns for entity, time, and variables
    model = FixedEffects("invest ~ value + capital", data, "firm", "year")
    result = model.fit()
    ```

    Key requirements:

    - **Long format**: each row is one entity at one time period
    - **Entity column**: identifies the cross-sectional unit (firm, country, individual)
    - **Time column**: identifies the time period (year, quarter, date)
    - **Variable columns**: must be numeric (`float` or `int`)

    If your data is in **wide format**, convert it first:

    ```python
    import pandas as pd

    data_long = pd.melt(
        data_wide,
        id_vars=["firm"],
        value_vars=["sales_2020", "sales_2021", "sales_2022"],
        var_name="year",
        value_name="sales"
    )
    data_long["year"] = data_long["year"].str.extract(r"(\d+)").astype(int)
    ```

    For a complete guide, see [How to Load Data](../getting-started/quickstart.md).

??? question "How do I load my CSV data?"

    ```python
    import pandas as pd
    from panelbox import FixedEffects

    # 1. Load CSV
    data = pd.read_csv("my_panel_data.csv")

    # 2. Check structure
    print(data.head())
    print(f"Entities: {data['entity_id'].nunique()}")
    print(f"Periods: {data['year'].nunique()}")

    # 3. Estimate a model
    model = FixedEffects(
        formula="y ~ x1 + x2",
        data=data,
        entity_col="entity_id",
        time_col="year"
    )
    results = model.fit()
    print(results.summary())
    ```

    PanelBox also supports loading from Excel, Stata (.dta), SQL databases, and R files. See [How to Load Data](../getting-started/quickstart.md) for all formats.

??? question "What datasets come built-in?"

    PanelBox ships with classic econometric datasets:

    ```python
    from panelbox.datasets import load_grunfeld, load_abdata, list_datasets

    # List all available datasets
    print(list_datasets())

    # Grunfeld: firm investment (10 firms, 20 years)
    grunfeld = load_grunfeld()

    # ABdata: employment dynamics (140 firms, 9 years)
    abdata = load_abdata()
    ```

??? question "Can PanelBox handle unbalanced panels?"

    Yes. Most PanelBox estimators handle unbalanced panels automatically. You can check your panel's balance:

    ```python
    from panelbox.core import PanelData

    panel = PanelData(data, "firm", "year")
    print(f"Balanced: {panel.is_balanced}")
    print(f"Entities: {panel.n_entities}, Periods: {panel.n_periods}")
    ```

---

## Model Selection

??? question "How do I choose between Fixed Effects and Random Effects?"

    Run the **Hausman test**:

    ```python
    from panelbox import FixedEffects, RandomEffects
    from panelbox.validation import HausmanTest

    fe = FixedEffects("invest ~ value + capital", data, "firm", "year")
    re = RandomEffects("invest ~ value + capital", data, "firm", "year")

    fe_results = fe.fit()
    re_results = re.fit()

    hausman = HausmanTest(fe_results, re_results)
    result = hausman.run()
    print(result.conclusion)
    ```

    **Interpretation:**

    - **p < 0.05** → Use **Fixed Effects** (entity effects are correlated with regressors)
    - **p >= 0.05** → **Random Effects** is more efficient

    If the Hausman test statistic is negative, use the **Mundlak test** instead (see [Common Pitfalls](#common-pitfalls)).

    For a detailed decision tree, see [How to Choose a Model](../getting-started/choosing-model.md).

??? question "When should I use GMM instead of Fixed Effects?"

    Use **GMM** (Difference or System GMM) when your model includes a **lagged dependent variable**:

    ```text
    y_it = α * y_{i,t-1} + β * X_it + η_i + ε_it
    ```

    Including $y_{i,t-1}$ as a regressor creates correlation with the error term, making Fixed Effects **biased** (Nickell bias). GMM uses instrumental variables to handle this endogeneity.

    **Rule of thumb:**

    - No lagged dependent variable → Fixed Effects or Random Effects
    - Lagged dependent variable present → GMM

    ```python
    from panelbox.gmm import SystemGMM

    model = SystemGMM(
        "n ~ L.n + w + k", data, "id", "year",
        gmm_instruments=["L.n"],
        iv_instruments=["w", "k"]
    )
    result = model.fit()
    ```

    See the [GMM tutorial](../tutorials/gmm.md) for details.

??? question "Which spatial model should I use?"

    Use the **LM test decision tree**:

    | LM Test Result | Recommended Model |
    |---|---|
    | Only LM-lag significant | SAR (Spatial Lag) |
    | Only LM-error significant | SEM (Spatial Error) |
    | Both significant → check robust versions | SDM or GNS |
    | Robust LM-lag significant | SAR |
    | Robust LM-error significant | SEM |

    ```python
    from panelbox.diagnostics import lm_lag_test, lm_error_test

    lm_lag = lm_lag_test(result, W)
    lm_err = lm_error_test(result, W)

    print(f"LM-lag p-value: {lm_lag.pvalue:.4f}")
    print(f"LM-error p-value: {lm_err.pvalue:.4f}")
    ```

    When in doubt, LeSage & Pace (2009) recommend starting with **SDM** (Spatial Durbin Model). See [Spatial FAQ](spatial.md) for details.

??? question "When do I need PPML?"

    Use **PPML** (Poisson Pseudo-Maximum Likelihood) for:

    - **Gravity models** with zero trade flows (OLS on logs drops zeros)
    - **Heteroskedastic** count-like data
    - When the dependent variable is **non-negative with many zeros**

    ```python
    from panelbox.models.count import PPML

    ppml = PPML(data, dep_var="trade",
                exog_vars=["log_distance", "log_gdp_i", "log_gdp_j"])
    result = ppml.fit()
    ```

    See [Advanced FAQ](advanced.md#ppml-count-models) for more details.

---

## Standard Errors

??? question "Which standard errors should I use?"

    **Decision tree:**

    | Situation | Standard Error Type | Code |
    |---|---|---|
    | Default / baseline | Non-robust | `cov_type='nonrobust'` |
    | Heteroskedasticity | Robust (HC1) | `cov_type='robust'` |
    | Panel data (most common) | Clustered by entity | `cov_type='clustered'` |
    | Cross-sectional dependence | Driscoll-Kraay | Use `DriscollKraayStandardErrors` |
    | Spatial correlation | Spatial HAC | Use `SpatialHAC` |
    | Both spatial and temporal | Panel-corrected (PCSE) | Use `PanelCorrectedStandardErrors` |

    **Safe default** for panel data: cluster by entity.

    ```python
    results = model.fit(cov_type="clustered")
    ```

??? question "How do I cluster standard errors?"

    ```python
    # Cluster by entity (most common)
    results = model.fit(cov_type="clustered")

    # Two-way clustering (entity and time)
    from panelbox.standard_errors import twoway_cluster
    vcov = twoway_cluster(results)
    ```

??? question "What's the difference between HC0, HC1, HC2, HC3?"

    These are different heteroskedasticity-consistent (HC) variance estimators:

    | Variant | Correction | Best for |
    |---|---|---|
    | **HC0** | None (White, 1980) | Large samples |
    | **HC1** | Degrees-of-freedom | General use (most common) |
    | **HC2** | Leverage-based | Moderate samples |
    | **HC3** | Jackknife-like | Small samples, most conservative |

    In practice, **HC1** is the default and works well for most applications. Use **HC3** for small samples where you want conservative inference.

---

## Results Interpretation

??? question "How do I get coefficients, p-values, and confidence intervals?"

    ```python
    results = model.fit()

    # Coefficients
    print(results.params)

    # Standard errors
    print(results.std_errors)

    # P-values
    print(results.pvalues)

    # Confidence intervals
    print(results.conf_int(alpha=0.05))

    # Full summary table
    print(results.summary())
    ```

??? question "How do I compute marginal effects?"

    For nonlinear models (logit, probit, count), coefficients are not directly interpretable. Compute marginal effects:

    ```python
    from panelbox.marginal_effects import compute_ame, compute_mem

    # Average Marginal Effects (AME)
    ame = compute_ame(results, data)
    print(ame)

    # Marginal Effects at the Mean (MEM)
    mem = compute_mem(results, data)
    print(mem)
    ```

    For ordered models:

    ```python
    from panelbox.marginal_effects import compute_ordered_ame
    ame = compute_ordered_ame(results, data)
    ```

    See the [Marginal Effects tutorial](../tutorials/marginal-effects.md) for detailed examples.

??? question "How do I compare two models?"

    Use the `PanelExperiment` workflow:

    ```python
    from panelbox import PanelExperiment

    exp = PanelExperiment(data, "invest ~ value + capital", "firm", "year")
    exp.fit_all_models(["pooled", "fe", "re"])
    comparison = exp.compare_models(["pooled", "fe", "re"])
    print(comparison.summary())
    ```

    For formal model testing between non-nested models, use the J-test:

    ```python
    from panelbox.diagnostics.specification import j_test
    result = j_test(result1, result2)
    print(result.summary())
    ```

??? question "How do I export results to LaTeX or HTML?"

    ```python
    from panelbox.report import ReportManager

    report = ReportManager(results)

    # LaTeX table
    report.export_latex("results.tex")

    # HTML report with interactive charts
    report.export_html("analysis.html")
    ```

    Or use the full Experiment workflow for a master report:

    ```python
    exp = PanelExperiment(data, formula, entity, time)
    exp.fit_all_models(["pooled", "fe", "re"])
    exp.save_master_report("master_report.html")
    ```

---

## Common Pitfalls

??? question "My Hausman test statistic is negative — what does this mean?"

    A negative Hausman test statistic can occur when the estimated variance difference is not positive semi-definite. This does **not** mean FE or RE is better — the test is simply unreliable in this case.

    **Solution:** Use the **Mundlak test** instead:

    ```python
    from panelbox.validation import MundlakTest

    mundlak = MundlakTest(data, "invest ~ value + capital", "firm", "year")
    result = mundlak.run()
    print(result.conclusion)
    ```

    The Mundlak test is a robust alternative that adds group means of regressors to the RE model and tests their joint significance.

??? question "My GMM has too many instruments — what should I do?"

    Instrument proliferation leads to overfitting and makes the Hansen J test unreliable. **Rule of thumb:** number of instruments should be less than N (number of entities).

    **Solutions:**

    1. Use `collapse=True` to reduce instrument count:
    ```python
    model = SystemGMM(
        "n ~ L.n + w + k", data, "id", "year",
        gmm_instruments=["L.n"],
        iv_instruments=["w", "k"],
        collapse=True
    )
    ```

    2. Limit lag depth:
    ```python
    model = SystemGMM(
        ...,
        max_lags=2  # Use at most 2 lags as instruments
    )
    ```

    3. Check the instrument-to-entity ratio after estimation:
    ```python
    result = model.fit()
    print(f"Instruments: {result.n_instruments}")
    print(f"Entities: {result.n_entities}")
    # Instruments should be < N
    ```

??? question "R-squared is low — is my model bad?"

    Not necessarily. In panel data econometrics, **R-squared is less meaningful** than in cross-sectional analysis:

    - **Within R-squared** (FE) only captures time variation, which is often small
    - A low R-squared does not mean the coefficients are wrong or insignificant
    - What matters more: coefficient significance, diagnostic tests, and economic interpretation

    Focus on:

    - Statistical significance of coefficients
    - Correct signs (consistent with theory)
    - Diagnostic test results (Hausman, serial correlation, etc.)
    - Robustness across specifications

??? question "My model shows 'ConvergenceWarning' — what should I do?"

    For MLE-based models (logit, probit, Heckman, SFA), convergence issues often arise from:

    1. **Poor starting values** — try providing manual starting values
    2. **Model too complex** — simplify the specification
    3. **Data issues** — check for perfect separation (logit/probit) or outliers

    ```python
    # Increase iterations
    result = model.fit(maxiter=500)

    # Try different optimizer
    result = model.fit(method="bfgs")
    ```

    See [Troubleshooting](troubleshooting.md#estimation-errors) for detailed solutions.

---

## What's Next?

<div class="grid cards" markdown>

-   **[Advanced FAQ](advanced.md)**

    Technical questions about GMM, VAR, Heckman, cointegration, and performance optimization.

-   **[Spatial FAQ](spatial.md)**

    Questions specific to spatial econometrics — weight matrices, model selection, effects interpretation.

-   **[Troubleshooting](troubleshooting.md)**

    Common error messages, debugging strategies, and step-by-step solutions.

-   **[How to Choose a Model](../getting-started/choosing-model.md)**

    Decision trees and checklists for selecting the right panel estimator.

</div>
