---
title: "Troubleshooting"
description: "Troubleshooting guide for PanelBox — common error messages, installation issues, data problems, estimation failures, and performance optimization."
---

# Troubleshooting Guide

Step-by-step solutions for common errors and problems when using PanelBox.

!!! tip "Looking for conceptual answers?"
    - **General questions**: [General FAQ](general.md)
    - **Advanced methods**: [Advanced FAQ](advanced.md)
    - **Spatial models**: [Spatial FAQ](spatial.md)

---

## Installation Issues

??? question "`ModuleNotFoundError: No module named 'panelbox'`"

    PanelBox is not installed or not in the active Python environment.

    **Solution:**

    ```bash
    pip install panelbox
    ```

    If using conda:

    ```bash
    conda activate your_env
    pip install panelbox
    ```

    Verify installation:

    ```python
    import panelbox
    print(panelbox.__version__)
    ```

??? question "Optional dependency errors (plotly, kaleido, scipy)"

    Some PanelBox features require optional dependencies. If you see errors like `ModuleNotFoundError: No module named 'plotly'`:

    **Solution — install all optional dependencies:**

    ```bash
    pip install panelbox[all]
    ```

    **Or install specific extras:**

    ```bash
    # Visualization only
    pip install plotly kaleido

    # Spatial only
    pip install libpysal geopandas

    # Full scientific stack
    pip install scipy statsmodels
    ```

??? question "Version conflicts with numpy/scipy/pandas"

    If you see errors like `ImportError: numpy.core.multiarray failed to import`:

    **Solution:**

    ```bash
    # Upgrade all dependencies together
    pip install --upgrade panelbox numpy scipy pandas

    # Or pin compatible versions
    pip install "numpy>=1.24,<2.0" "scipy>=1.10" "pandas>=2.0"
    ```

    Check your current versions:

    ```python
    import numpy, scipy, pandas
    print(f"numpy: {numpy.__version__}")
    print(f"scipy: {scipy.__version__}")
    print(f"pandas: {pandas.__version__}")
    ```

??? question "`ImportError: cannot import name 'PanelVAR'`"

    Most PanelBox classes must be imported from their submodules, not from the top-level package.

    ```python
    # Correct
    from panelbox.var import PanelVAR
    from panelbox.gmm import DifferenceGMM, SystemGMM
    from panelbox.models.spatial import SpatialLag

    # Incorrect
    from panelbox import PanelVAR  # Will fail
    ```

    See the [API Reference](../api/index.md) for correct import paths.

---

## Data Issues

??? question "`ValueError: data must have MultiIndex` or similar index errors"

    PanelBox models expect a DataFrame with entity and time columns specified as parameters.

    **Solution — provide entity and time columns:**

    ```python
    from panelbox import FixedEffects

    # Pass entity_col and time_col explicitly
    model = FixedEffects(
        formula="y ~ x1 + x2",
        data=data,
        entity_col="firm_id",
        time_col="year"
    )
    ```

    If your data uses a MultiIndex, reset it:

    ```python
    data = data.reset_index()
    ```

??? question "Unbalanced panel warnings"

    Most PanelBox estimators handle unbalanced panels automatically. The warning is informational — it tells you that entities have different numbers of time periods.

    **To check balance:**

    ```python
    obs_per_entity = data.groupby("entity_id").size()
    print(f"Min periods: {obs_per_entity.min()}")
    print(f"Max periods: {obs_per_entity.max()}")
    print(f"Balanced: {(obs_per_entity == obs_per_entity.iloc[0]).all()}")
    ```

    **To force a balanced panel** (if needed):

    ```python
    n_periods = data["year"].nunique()
    balanced_entities = obs_per_entity[obs_per_entity == n_periods].index
    data_balanced = data[data["entity_id"].isin(balanced_entities)]
    ```

??? question "Missing values — how are they handled?"

    Behavior varies by model:

    - **Static models** (FE, RE, Pooled OLS): drop observations with missing values in formula variables
    - **GMM**: drops observations with missing in dependent or independent variables; lagged instruments handle their own missingness
    - **MLE models** (logit, probit, Heckman): require complete cases

    **Best practice:** Check missingness before estimation:

    ```python
    print(data[["y", "x1", "x2"]].isnull().sum())

    # Drop rows with missing dependent variable
    data = data.dropna(subset=["y"])

    # Forward-fill covariates within entities (use with caution)
    data["x1"] = data.groupby("entity_id")["x1"].ffill()
    ```

??? question "My panel has gaps (non-consecutive time periods)"

    Gaps in time periods can affect models that use lags (GMM, VAR, dynamic models).

    **Check for gaps:**

    ```python
    def check_gaps(group):
        times = sorted(group["year"].unique())
        expected = list(range(min(times), max(times) + 1))
        return set(expected) - set(times)

    gaps = data.groupby("entity_id").apply(check_gaps)
    entities_with_gaps = gaps[gaps.apply(len) > 0]
    print(f"Entities with gaps: {len(entities_with_gaps)}")
    ```

    **Solutions:**

    - Remove entities with gaps: `data = data[~data["entity_id"].isin(entities_with_gaps.index)]`
    - For VAR/VECM: ensure continuous time periods or use `allow_unbalanced=True`
    - For GMM: gaps in instruments are handled automatically, but check results carefully

??? question "`ValueError: could not convert string to float`"

    Non-numeric data in variable columns.

    **Solution:**

    ```python
    # Find problematic values
    print(data["x1"].dtype)
    non_numeric = data[pd.to_numeric(data["x1"], errors="coerce").isnull()]
    print(non_numeric)

    # Convert, replacing non-numeric with NaN
    data["x1"] = pd.to_numeric(data["x1"], errors="coerce")
    ```

---

## Estimation Errors

??? question "`LinAlgError: Singular matrix`"

    **Causes:**

    1. **Perfect collinearity** — two or more variables are linearly dependent
    2. **Too many dummy variables** — time dummies exceed available degrees of freedom
    3. **Too many instruments in GMM** — instrument matrix is rank-deficient

    **Solutions:**

    ```python
    # 1. Check for collinearity
    corr = data[["x1", "x2", "x3"]].corr()
    print(corr)
    # Remove variables with |correlation| > 0.95

    # 2. For GMM: reduce instruments
    from panelbox.gmm import SystemGMM
    model = SystemGMM(
        ...,
        collapse=True,    # Reduce instrument count
        max_lags=2         # Limit lag depth
    )

    # 3. Check for constant variables
    for col in ["x1", "x2", "x3"]:
        if data[col].std() == 0:
            print(f"WARNING: {col} has zero variance")
    ```

??? question "`ConvergenceWarning` — model did not converge"

    For MLE-based models (logit, probit, Heckman, SFA, count models):

    **Solutions:**

    1. **Increase maximum iterations:**
    ```python
    result = model.fit(maxiter=500)
    ```

    2. **Try a different optimizer:**
    ```python
    result = model.fit(method="bfgs")    # More robust than Newton
    result = model.fit(method="l-bfgs-b")  # Good for large problems
    ```

    3. **Provide better starting values:**
    ```python
    # Use a simpler model's estimates as starting values
    simple_result = simple_model.fit()
    result = complex_model.fit(starting_values=simple_result.params)
    ```

    4. **Simplify the model** — remove interaction terms, reduce number of covariates

    5. **Scale your variables** — variables on very different scales can cause numerical issues:
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data[["x1", "x2"]] = scaler.fit_transform(data[["x1", "x2"]])
    ```

??? question "GMM results are unstable or unreasonable"

    **Common causes:**

    - **Too many instruments** → overfitting
    - **Weak instruments** → large standard errors, unstable coefficients
    - **N < L** (entities < instruments) → rank-deficient weight matrix

    **Solutions:**

    ```python
    from panelbox.gmm import SystemGMM, GMMOverfitDiagnostic

    # 1. Use collapse to reduce instruments
    model = SystemGMM(..., collapse=True, max_lags=2)
    result = model.fit()

    # 2. Check instrument count vs entities
    print(f"Instruments: {result.n_instruments}")
    print(f"Entities: {result.n_entities}")
    # Instruments should be < N

    # 3. Run overfit diagnostic
    diag = GMMOverfitDiagnostic(result)
    print(diag.summary())

    # 4. Compare one-step vs two-step
    result_1step = model.fit(two_step=False)
    result_2step = model.fit(two_step=True)
    # If very different → instrument issues
    ```

??? question "MLE did not converge (Heckman, SFA, logit)"

    **For Heckman specifically:**

    ```python
    from panelbox.models.selection import PanelHeckman

    # 1. Use two-step as starting values for MLE
    result_2step = heckman.fit(method="two_step")
    result_mle = heckman.fit(
        method="mle",
        starting_values=result_2step.params
    )

    # 2. Reduce quadrature points (Heckman MLE)
    result = heckman.fit(method="mle", quadrature_points=10)

    # 3. Just use two-step (reliable fallback)
    result = heckman.fit(method="two_step")
    ```

    **For SFA/Frontier:**

    ```python
    from panelbox.frontier import StochasticFrontier

    # Try different starting values or distributions
    sfa = StochasticFrontier(data, ..., distribution="half_normal")
    result = sfa.fit(maxiter=500)
    ```

??? question "Negative variance estimates"

    Negative variance components typically indicate **model misspecification**:

    - Random Effects model assumes $\text{Var}(\alpha_i) > 0$, but the data does not support this
    - May occur with very small between-entity variation

    **Solutions:**

    - Switch to Fixed Effects (does not estimate variance components)
    - Check that the entity and time columns are correctly specified
    - Verify that there is meaningful cross-entity variation

---

## Diagnostic Test Errors

??? question "Hansen J test returns NaN"

    **Cause:** The model is **under-identified** — the number of instruments is less than or equal to the number of parameters, giving $df = n_{instruments} - n_{params} \leq 0$.

    This commonly occurs with `collapse=True` combined with `time_dummies=True` (default), where the number of time dummies exceeds the collapsed instrument count.

    **Solutions:**

    ```python
    # 1. Check degrees of freedom
    print(f"Instruments: {result.n_instruments}")
    print(f"Parameters: {result.n_params}")
    print(f"Hansen df: {result.n_instruments - result.n_params}")
    # Must be > 0 for a valid Hansen test

    # 2. Use time_dummies=False with collapse=True
    model = SystemGMM(..., collapse=True, time_dummies=False)

    # 3. Add more instruments (increase max_lags)
    model = SystemGMM(..., collapse=True, max_lags=3)
    ```

??? question "Hausman test statistic is negative"

    A negative Hausman statistic occurs when the variance difference matrix $(V_{FE} - V_{RE})$ is not positive semi-definite. The test is unreliable in this case.

    **Solution — use the Mundlak test:**

    ```python
    from panelbox.validation import MundlakTest

    mundlak = MundlakTest(data, "y ~ x1 + x2", "entity_id", "year")
    result = mundlak.run()
    print(result.conclusion)
    # If p < 0.05: use Fixed Effects
    # If p >= 0.05: Random Effects is consistent
    ```

    The Mundlak test is always well-defined and provides a robust alternative.

??? question "Unit root tests give contradictory results"

    Different tests have different null hypotheses and power:

    | Tests Agree? | Hadri | IPS/LLC | Interpretation |
    |---|---|---|---|
    | Hadri rejects, IPS fails to reject | Rejects stationarity | Fails to reject unit root | **Unit root present** |
    | Hadri fails to reject, IPS rejects | Cannot reject stationarity | Rejects unit root | **Stationary** |
    | Both reject their nulls | — | — | Borderline (near unit root) |
    | Both fail to reject | — | — | Insufficient power |

    **Best practice:** Run a battery of tests and look for consensus:

    ```python
    from panelbox.diagnostics.unit_root import hadri_test, breitung_test
    from panelbox.validation import IPSTest, LLCTest

    # Stationarity null
    hadri = hadri_test(data, variable="y")

    # Unit root null
    ips = IPSTest(data, variable="y", entity_col="entity", time_col="time").run()
    llc = LLCTest(data, variable="y", entity_col="entity", time_col="time").run()

    print(f"Hadri (H0: stationary): p = {hadri.pvalue:.4f}")
    print(f"IPS (H0: unit root): p = {ips.pvalue:.4f}")
    print(f"LLC (H0: unit root): p = {llc.pvalue:.4f}")
    ```

    If results are ambiguous, try the **Breitung test** as a tiebreaker and check for **structural breaks** in the data.

??? question "Moran's I test gives unexpected results"

    - **Significant Moran's I on OLS residuals** → spatial autocorrelation exists, consider spatial models
    - **Significant Moran's I on spatial model residuals** → model has not fully captured spatial dependence; try SDM or GNS
    - **Non-significant Moran's I on data but theory suggests spatial effects** → check weight matrix specification; try different W constructions

---

## Report & Visualization Errors

??? question "HTML report is blank or not rendering"

    **Cause:** Plotly is not installed or not configured for your environment.

    **Solution:**

    ```bash
    pip install plotly
    ```

    For Jupyter notebooks:

    ```bash
    pip install plotly nbformat
    # If using JupyterLab
    jupyter labextension install jupyterlab-plotly
    ```

??? question "Export to PNG fails"

    **Cause:** The `kaleido` package (static image export engine) is not installed.

    **Solution:**

    ```bash
    pip install kaleido
    ```

    If `kaleido` installation fails on your platform:

    ```bash
    # Alternative: use orca (older, requires separate install)
    conda install -c plotly plotly-orca
    ```

??? question "Chart not rendering in Jupyter notebook"

    **Solutions:**

    1. **Use the HTML method:**
    ```python
    chart.to_html()  # Returns HTML string for inline display
    ```

    2. **Install Jupyter extension:**
    ```bash
    pip install plotly ipywidgets
    # For JupyterLab
    jupyter labextension install jupyterlab-plotly
    ```

    3. **Set renderer explicitly:**
    ```python
    import plotly.io as pio
    pio.renderers.default = "notebook"  # or "colab" for Google Colab
    ```

---

## Performance Issues

??? question "Estimation takes too long"

    **Identify the bottleneck and apply targeted solutions:**

    | Problem | Solution |
    |---|---|
    | Too many GMM instruments | `collapse=True`, `max_lags=2` |
    | CUE-GMM optimization | Use two-step for exploration, CUE for final results |
    | Bootstrap replications | Reduce `n_boot` to 499 (usually sufficient) |
    | Spatial ML (large N) | Use sparse weight matrices, Chebyshev approximation |
    | Heckman MLE | Reduce `quadrature_points` to 10, or use two-step |
    | FE multinomial logit | Use RE if J > 4 or T > 10 |
    | Cointegration bootstrap | Use asymptotic first; only bootstrap if borderline |

    **General advice:** Use fast methods for exploration, robust methods for final results.

??? question "Out of memory (MemoryError)"

    **Common causes and solutions:**

    - **Bootstrap with large N*T**: reduce `n_boot` to 499
    - **Spatial weight matrix**: use sparse format for large N
    - **Panel VAR IRF bootstrap**: reduce `n_boot` and `periods`, or process one impulse at a time

    ```python
    # Use sparse weight matrix
    from scipy import sparse
    W_sparse = sparse.csr_matrix(W_array)

    # Reduce bootstrap replications
    result = test(data, ..., n_bootstrap=499)

    # Process IRFs one at a time
    for impulse_var in variables:
        irf = result.irf(impulse=impulse_var, n_boot=500)
    ```

---

## Error Message Index

Quick reference for common error messages in alphabetical order:

| Error Message | Likely Cause | Solution |
|---|---|---|
| `ConvergenceWarning` | MLE optimization failed | Increase `maxiter`, try `method="bfgs"`, simplify model |
| `ImportError: cannot import name ...` | Wrong import path | Check [API Reference](../api/index.md) for correct imports |
| `KeyError: 'entity_col'` | Column name mismatch | Check `data.columns` for exact name (case-sensitive) |
| `LinAlgError: Singular matrix` | Perfect collinearity or rank deficiency | Remove correlated variables, reduce instruments |
| `MemoryError` | Dataset too large for available RAM | Reduce bootstrap, use sparse matrices, subset data |
| `ModuleNotFoundError` | Package not installed | `pip install panelbox` or `pip install panelbox[all]` |
| `RuntimeWarning: invalid value (NaN)` | Numerical instability | Scale variables, check for outliers, simplify model |
| `ValueError: could not convert` | Non-numeric data | Use `pd.to_numeric(col, errors="coerce")` |
| `ValueError: data must have MultiIndex` | Missing entity/time specification | Pass `entity_col` and `time_col` parameters |
| `Warning: Panel is unbalanced` | Informational — not an error | Most models handle automatically |
| `Warning: rho > 1 in Heckman` | Model misspecification | Check exclusion restriction, use two-step |
| `Warning: VAR is unstable` | Eigenvalues > 1 | Difference variables, use VECM, reduce lags |

---

## Debugging Checklist

When you encounter an issue, work through this systematic checklist:

### 1. Data

- [ ] Panel structure correct (entity and time columns identified)?
- [ ] No missing values in key variables (or handled explicitly)?
- [ ] Variables are numeric (no strings in regression columns)?
- [ ] No duplicate entity-time pairs?
- [ ] Sufficient observations (N and T)?

### 2. Model Specification

- [ ] Formula is correct (dependent ~ independent)?
- [ ] Entity and time column names match exactly?
- [ ] Appropriate model for the data (static vs dynamic, FE vs RE)?
- [ ] No perfect collinearity among regressors?

### 3. Estimation

- [ ] Model converged (check warnings)?
- [ ] Coefficients are reasonable in magnitude and sign?
- [ ] Standard errors are finite and non-zero?
- [ ] For GMM: instruments < N, Hansen J p-value > 0.10?

### 4. Diagnostics

- [ ] Residuals look random (no patterns)?
- [ ] Diagnostic tests pass (Hausman, serial correlation, etc.)?
- [ ] Results robust to alternative specifications?

---

## Getting Help

If this guide does not solve your problem:

1. **Prepare a minimal reproducible example:**
    ```python
    import pandas as pd
    from panelbox import FixedEffects

    # Minimal data that reproduces the error
    data = pd.DataFrame({...})
    model = FixedEffects(...)
    result = model.fit()  # Error occurs here
    ```

2. **Include version information:**
    ```python
    import panelbox, pandas, numpy, scipy
    print(f"panelbox: {panelbox.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"numpy: {numpy.__version__}")
    print(f"scipy: {scipy.__version__}")
    ```

3. **Open an issue** on [GitHub](https://github.com/panelbox/panelbox/issues) with:
    - Description of the problem
    - Reproducible example
    - Full error traceback
    - What you already tried

---

## See Also

- [General FAQ](general.md) — getting started, model selection, results interpretation
- [Advanced FAQ](advanced.md) — GMM, VAR, Heckman, cointegration
- [Spatial FAQ](spatial.md) — spatial econometrics questions
- [API Reference](../api/index.md) — correct import paths and signatures
