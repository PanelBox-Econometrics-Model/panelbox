---
title: "Advanced FAQ"
description: "Technical FAQ for experienced PanelBox users — GMM, Panel VAR, Heckman selection, cointegration, count models, and performance tips."
---

# Advanced FAQ

Technical questions for experienced users covering GMM, Panel VAR, Heckman selection models, cointegration, count models, discrete choice, and performance optimization.

!!! tip "New to PanelBox?"
    Start with the [General FAQ](general.md) for getting started, model selection, and results interpretation.

---

## GMM Advanced

??? question "How does CUE-GMM differ from two-step GMM?"

    **Continuous Updated Estimator (CUE-GMM)** jointly optimizes both the coefficient vector and the weighting matrix, making it invariant to the choice of initial weighting matrix.

    | Feature | Two-Step GMM | CUE-GMM |
    |---|---|---|
    | **Efficiency** | Good | Higher (asymptotically) |
    | **Computation** | Fast | Slower (iterative) |
    | **Weak instruments** | Sensitive | More robust |
    | **Sample requirement** | N > 50 | N > 200 |

    ```python
    from panelbox.gmm import DifferenceGMM, ContinuousUpdatedGMM

    # Two-step GMM (fast, for exploration)
    gmm2 = DifferenceGMM(data, dep_var="y", lags=[1], exog_vars=["x"])
    result2 = gmm2.fit(steps=2)

    # CUE-GMM (more efficient but slower)
    cue = ContinuousUpdatedGMM(data, dep_var="y", lags=[1], exog_vars=["x"])
    result_cue = cue.fit()
    ```

    **Use CUE when:** you need maximum efficiency, weak instruments are a concern, and computation time is acceptable. **Use two-step when:** quick results needed or N is small.

??? question "When should I use bias-corrected GMM?"

    Use `BiasCorrectedGMM` when standard GMM suffers from **O(1/N) bias** — typically with:

    - **Small N** (N < 100)
    - **Persistent dependent variable** (AR coefficient > 0.7)
    - **Short T** (T < 10)

    ```python
    from panelbox.gmm import BiasCorrectedGMM

    bc_gmm = BiasCorrectedGMM(data, dep_var="y", lags=[1])
    result = bc_gmm.fit(order=2)  # Second-order correction

    print(f"Bias magnitude: {result.bias_magnitude()}")
    # If > 10% of coefficient, correction is important
    ```

    Minimum sample sizes: first-order correction needs N >= 50, second-order needs N >= 100.

??? question "My GMM has convergence issues. What should I do?"

    **Troubleshooting steps:**

    1. **Use two-step starting values:**
    ```python
    cue = ContinuousUpdatedGMM(data, ...)
    result = cue.fit(
        starting_values="two_step",
        maxiter=500,
        tol=1e-6
    )
    ```

    2. **Reduce instrument count** — rule of thumb: instruments <= 2 * parameters:
    ```python
    model = DifferenceGMM(data, ..., collapse=True, max_lags=2)
    ```

    3. **Check for multicollinearity** and data quality (no NaN, outliers)

    4. **Fall back to two-step GMM** if CUE doesn't converge

??? question "How do I handle instrument proliferation?"

    Instrument proliferation occurs when the number of instruments exceeds the number of entities, causing overfitting and unreliable Hansen J tests.

    **Rule of thumb:** instruments < N (number of entities)

    **Solutions:**

    - Use `collapse=True` to reduce instruments
    - Limit lag depth with `max_lags=2`
    - Check the `GMMOverfitDiagnostic` after estimation:

    ```python
    from panelbox.gmm import GMMOverfitDiagnostic

    diag = GMMOverfitDiagnostic(result)
    print(diag.summary())
    # Warns if instruments/N ratio is too high
    ```

??? question "Hansen J test says everything is fine but results look wrong?"

    This is a classic symptom of **overfitting**. When you have too many instruments, the Hansen J test loses power and fails to reject even when instruments are invalid.

    **Red flags:**

    - Hansen J p-value very close to 1.0
    - Number of instruments >> N
    - Results change dramatically when reducing instruments

    **Solution:** re-estimate with `collapse=True` and fewer lags. A well-behaved Hansen J p-value is typically between 0.10 and 0.80.

---

## Panel VAR

??? question "When should I use VAR vs VECM?"

    | Condition | Model |
    |---|---|
    | Variables are **stationary** (I(0)) | Panel VAR |
    | Variables are **non-stationary** (I(1)) and **cointegrated** | Panel VECM |
    | Variables are I(1) and **not cointegrated** | VAR in first differences |

    **Recommended workflow:**

    ```python
    from panelbox.validation import IPSTest

    # 1. Test for unit roots
    for var in ["y1", "y2"]:
        test = IPSTest(data, variable=var, entity_col="entity", time_col="time")
        result = test.run()
        print(f"{var}: {'stationary' if result.pvalue < 0.05 else 'unit root'}")

    # 2. If all I(1), test cointegration
    from panelbox.diagnostics.cointegration import pedroni_test
    coint = pedroni_test(data, dependent="y1", covariates=["y2"])

    # 3. If cointegrated → VECM
    from panelbox.var import PanelVECM
    vecm = PanelVECM(data, ...)
    ```

??? question "How do I select the optimal lag order?"

    Use information criteria:

    ```python
    from panelbox.var import PanelVAR

    pvar = PanelVAR(data, variables=["y1", "y2"], entity_col="entity", time_col="time")
    lag_result = pvar.select_lag_order(max_lags=5, criterion="bic")
    print(f"Optimal lags: {lag_result.optimal_lag}")
    ```

    **Criteria:**

    - **BIC**: more conservative, penalizes extra parameters more — good default
    - **AIC**: less conservative, may overfit
    - **HQIC**: intermediate between AIC and BIC

    **Practical guidelines:** annual data → 1-2 lags; quarterly → 1-4; monthly → 1-12.

??? question "How do I interpret Granger causality in Panel VAR?"

    Granger causality is **predictive**, not structural. "$X$ Granger-causes $Y$" means past values of $X$ improve the prediction of $Y$, controlling for past $Y$.

    ```python
    result = pvar.fit(lags=2)
    gc = result.granger_causality("x1", "x2")
    print(f"p-value: {gc.pvalue:.4f}")
    # p < 0.05: x1 Granger-causes x2
    ```

    **Important:** Granger causality does NOT imply true causation. It is a statement about prediction, not about causal mechanisms.

    For panel data specifically, the **Dumitrescu-Hurlin test** has more power and allows for heterogeneity across entities.

??? question "IRF confidence bands — bootstrap vs asymptotic?"

    | Method | Pros | Cons |
    |---|---|---|
    | **Bootstrap** | More reliable in small samples; distribution-free | Slow (especially with many replications) |
    | **Asymptotic** | Fast | May be inaccurate in small samples |

    ```python
    # Bootstrap (recommended for final results)
    irf = result.irf(periods=10, ci_method="bootstrap", n_boot=1000)

    # Asymptotic (fast, for exploration)
    irf = result.irf(periods=10, ci_method="analytical")
    ```

    If bootstrap IRFs look asymmetric or strange, increase `n_boot` to 2000 or try `bootstrap_type="residual"`.

??? question "My Panel VAR is unstable (eigenvalues > 1). What should I do?"

    An unstable VAR means IRFs will diverge over time. Common causes:

    1. **Variables are non-stationary** — test with unit root tests, difference if needed
    2. **Too many lags** — try reducing the lag order
    3. **Cointegrated variables** — use VECM instead of VAR

    ```python
    # Check stability
    if not result.is_stable():
        eigenvalues = result.eigenvalues()
        print(f"Max eigenvalue modulus: {max(abs(eigenvalues)):.4f}")

        # Try fewer lags
        result_p1 = pvar.fit(lags=1)
        if result_p1.is_stable():
            print("Stable with 1 lag")
    ```

??? question "OLS vs GMM for Panel VAR — which should I use?"

    | Method | Use when | Pros | Cons |
    |---|---|---|---|
    | **OLS** | T >> N, exogeneity holds | Fast, simple | Nickell bias when T small |
    | **GMM** | T small (~10-20), N large | Consistent, handles endogeneity | Needs N large, sensitive to instruments |

    **Start with OLS** for exploration, then switch to GMM if diagnostics indicate problems (endogeneity, small T).

---

## Panel Heckman / Selection Models

??? question "When should I use the Heckman selection model?"

    Use `PanelHeckman` when your sample is **non-randomly selected**:

    - Wages observed only for workers (not unemployed)
    - Firm performance observed only for surviving firms
    - Exports observed only for exporting firms

    **Requirements:**

    1. You can model the selection decision
    2. An **exclusion restriction** exists (variable affects selection but not outcome)

    ```python
    from panelbox.models.selection import PanelHeckman

    heckman = PanelHeckman(
        data=data,
        outcome_formula="wage ~ educ + exper",
        selection_formula="work ~ age + kids + married",
        entity_id="person_id",
        time_id="year"
    )
    result = heckman.fit(method="two_step")
    ```

??? question "What are exclusion restrictions and why do they matter?"

    An **exclusion restriction** is a variable that:

    - Affects whether you **observe** the outcome (selection equation)
    - Does **not** directly affect the outcome itself

    **Example:** "number of young children" affects labor force participation (selection) but should not directly affect wages (outcome), after controlling for experience and education.

    Without a valid exclusion restriction, the Heckman model is identified only through functional form assumptions, making estimates fragile.

??? question "Two-step vs MLE for Heckman — which should I use?"

    | Feature | Two-Step | MLE |
    |---|---|---|
    | **Speed** | Fast | Slow |
    | **Efficiency** | Good | Best (asymptotically) |
    | **Robustness** | More robust to misspecification | Sensitive |
    | **Convergence** | Always works | May fail |
    | **Sample size** | Works with N < 100 | Better with N > 200 |

    **Recommendation:** Start with two-step. Use MLE for final results only if N > 200 and two-step shows significant selection.

    ```python
    # Start with two-step
    result_2step = heckman.fit(method="two_step")

    # If significant selection and large sample, try MLE
    result_mle = heckman.fit(
        method="mle",
        starting_values=result_2step.params  # Use two-step as initialization
    )
    ```

??? question "How do I test for selection bias?"

    ```python
    from panelbox.models.selection import test_selection_effect

    result = heckman.fit(method="two_step")

    # Test H0: no selection bias (rho = 0)
    sel_test = test_selection_effect(result)
    print(f"rho = {result.rho:.3f}, p-value = {sel_test.pvalue:.4f}")

    # If p < 0.05: selection bias exists, Heckman correction is needed
    # If p >= 0.05: no evidence of selection bias, OLS may be sufficient
    ```

??? question "What does rho > 1 mean in the Heckman model?"

    If $|\rho| > 1$, the model is **misspecified**. Possible causes:

    1. **Wrong exclusion restriction** — the instrument affects the outcome directly
    2. **Functional form misspecification** — add non-linear terms (e.g., `exper^2`)
    3. **Omitted variables** in the selection equation

    **Solution:** Use `method="two_step"` (more robust) and reconsider your exclusion restriction.

---

## Cointegration & Unit Root Tests

??? question "Which cointegration test should I use: Westerlund, Pedroni, or Kao?"

    | Feature | Westerlund | Pedroni | Kao |
    |---|---|---|---|
    | **Heterogeneity** | Allows | Allows | Assumes homogeneous |
    | **Power (small T)** | High | Medium | Medium |
    | **Interpretation** | ECM-based | Residual-based | Residual-based |
    | **Cross-section dependence** | Robust | Sensitive | Sensitive |
    | **Speed** | Slow | Fast | Fast |

    **Recommendation:** Use Westerlund for most cases. Use Pedroni if Westerlund is too slow. Use Kao only if you believe in homogeneous cointegrating vectors.

    ```python
    from panelbox.diagnostics.cointegration import westerlund_test, pedroni_test, kao_test

    # Westerlund (recommended)
    west = westerlund_test(data, dependent="y", covariates=["x1", "x2"])
    print(west.summary())

    # Pedroni (faster)
    ped = pedroni_test(data, dependent="y", covariates=["x1", "x2"])

    # Kao (homogeneous)
    kao = kao_test(data, dependent="y", covariates=["x1", "x2"])
    ```

??? question "All my unit root tests disagree. How do I interpret this?"

    Different tests have different null hypotheses and power properties:

    | Test | H0 | Good for |
    |---|---|---|
    | **Hadri** | Stationarity | Confirming unit roots |
    | **IPS** | Unit root | Detecting stationarity |
    | **LLC** | Unit root (homogeneous) | Homogeneous panels |
    | **Breitung** | Unit root | Robust to heterogeneous trends |
    | **Fisher** | Unit root | Unbalanced panels |

    **Interpretation guide:**

    - Hadri rejects + IPS fails to reject → **Unit root present**
    - Hadri fails to reject + IPS rejects → **Stationary**
    - Both reject → **Borderline** (near unit root); use Breitung as tiebreaker
    - Both fail to reject → **Low power**; increase sample or check for structural breaks

    **Best practice:** Run a battery of tests and look for consensus.

??? question "Which Pedroni statistics should I report?"

    Pedroni offers 7 test statistics, divided into:

    - **Panel statistics** (within-dimension): assume common AR dynamics — more powerful under homogeneity
    - **Group statistics** (between-dimension): allow heterogeneous AR dynamics — more robust

    **Recommendation:** Report the **group ADF** statistic (most commonly used) and the **panel ADF** statistic for robustness. If they agree, the result is reliable.

??? question "How do I set the trend specification for unit root tests?"

    | Specification | Code | Use when |
    |---|---|---|
    | No constant, no trend | `"nc"` | Rarely used |
    | Constant only | `"c"` | Default — most common |
    | Constant + trend | `"ct"` | Series has deterministic trend |

    **Rule:** If the series visually trends upward/downward, use `"ct"`. Otherwise use `"c"`.

---

## PPML / Count Models

??? question "When does PPML fail?"

    PPML can fail due to:

    1. **Separation**: a regressor perfectly predicts zero outcomes
    2. **Perfect prediction**: fitted values are exactly zero for some observations
    3. **Extreme outliers** in the dependent variable

    **Troubleshooting:**

    ```python
    from panelbox.models.count import PPML, PoissonQML

    ppml = PPML(data, dep_var="trade", exog_vars=[...])
    try:
        result = ppml.fit(maxiter=500)
    except Exception:
        # Fall back to Poisson QML (more robust)
        qml = PoissonQML(data, dep_var="trade", exog_vars=[...])
        result = qml.fit()
    ```

    Also try scaling variables (divide large values by 1e6) and removing extreme outliers.

??? question "How do I test for overdispersion (Poisson vs Negative Binomial)?"

    If the variance exceeds the mean, the data is overdispersed and Negative Binomial may be more appropriate:

    ```python
    # Quick check
    mean_y = data["y"].mean()
    var_y = data["y"].var()
    print(f"Variance/Mean ratio: {var_y / mean_y:.2f}")
    # If > 1.5: overdispersed, consider Negative Binomial
    ```

    For a formal test, estimate both models and compare:

    ```python
    from panelbox.models.count import PoissonFixedEffects, FixedEffectsNegativeBinomial

    pois = PoissonFixedEffects(data, ...).fit()
    nb = FixedEffectsNegativeBinomial(data, ...).fit()
    # If NB dispersion parameter is significant → use NB
    ```

??? question "Zero-inflated vs standard count models — when to use which?"

    | Model | Use when |
    |---|---|
    | **Standard Poisson/NB** | Zeros arise from the same process as positive counts |
    | **Zero-Inflated** | Zeros come from two processes: "structural zeros" (never events) + sampling zeros |

    **Example:** patent counts — some firms never innovate (structural zeros) while others innovate but happen to have zero patents in a given year (sampling zeros).

    ```python
    from panelbox.models.count import ZeroInflatedPoisson

    zip_model = ZeroInflatedPoisson(data, dep_var="patents",
                                     exog_vars=["rd_spending", "size"])
    result = zip_model.fit()
    ```

---

## Multinomial / Discrete Choice

??? question "IIA assumption — how do I test it?"

    The **Independence of Irrelevant Alternatives (IIA)** assumption in multinomial logit can be tested using:

    1. **Hausman-McFadden test**: estimate model with and without one alternative
    2. **Small-Hsiao test**: split sample and compare estimates

    If IIA is rejected, consider:

    - **Nested logit** (groups correlated alternatives)
    - **Mixed logit** (random coefficients)
    - **Conditional logit** with alternative-specific variables

??? question "Fixed Effects logit drops groups — why?"

    FE logit uses conditional maximum likelihood, which requires **within-group variation** in the dependent variable. Groups where the outcome never changes (always 0 or always 1) provide no information and are dropped.

    This is expected behavior, not an error. If many groups are dropped, consider:

    - Using Random Effects logit instead
    - Checking if your sample has sufficient variation
    - Using a longer time span

    ```python
    from panelbox.models.discrete import FixedEffectsLogit

    fe_logit = FixedEffectsLogit(data, dep_var="y", exog_vars=["x1", "x2"])
    result = fe_logit.fit()
    # Check how many groups were used vs dropped
    ```

---

## Performance Tips

??? question "My estimation is slow. How can I speed it up?"

    **General tips:**

    | Bottleneck | Solution |
    |---|---|
    | Too many GMM instruments | Use `collapse=True`, limit `max_lags` |
    | Two-step → CUE-GMM | Use two-step for exploration, CUE for final results only |
    | Bootstrap | Reduce `n_boot` (499 is often sufficient), or use asymptotic |
    | Large N spatial model | Use sparse weight matrices |
    | Heckman MLE | Reduce `quadrature_points` to 10 (from default 15) |
    | FE multinomial logit | Use RE if J > 4 or T > 10 |

    **Workflow recommendation:** Use fast methods (two-step, asymptotic, OLS) for exploration. Switch to robust methods (CUE, bootstrap, GMM) for final results.

??? question "Memory issues with large panels. What can I do?"

    - **Reduce bootstrap replications**: `n_boot=499` instead of 1999
    - **Use sparse weight matrices** for spatial models: `W.to_sparse()`
    - **Process by variable** instead of all at once (e.g., IRFs one impulse at a time)
    - **Subset your data** for initial exploration, then run on full data for final results

??? question "Heckman MLE is very slow. Are there alternatives?"

    1. **Reduce quadrature points** (biggest speedup):
    ```python
    result = heckman.fit(method="mle", quadrature_points=10)  # Default: 15
    ```

    2. **Use two-step as starting values:**
    ```python
    result_2step = heckman.fit(method="two_step")
    result_mle = heckman.fit(
        method="mle",
        starting_values=result_2step.params,
        quadrature_points=10
    )
    ```

    3. **Just use two-step** — efficiency loss is small for most applications:
    ```python
    result = heckman.fit(method="two_step")
    ```

    Approximate timings (N=1000, T=5): two-step ~5s, MLE (10 pts) ~30s, MLE (15 pts) ~60s.

---

## See Also

- [General FAQ](general.md) — getting started, model selection, results interpretation
- [Spatial FAQ](spatial.md) — spatial econometrics questions
- [Troubleshooting](troubleshooting.md) — error messages and debugging
- [GMM API Reference](../api/gmm.md)
- [VAR API Reference](../api/var.md)
- [Cointegration Theory](../diagnostics/cointegration/index.md)
