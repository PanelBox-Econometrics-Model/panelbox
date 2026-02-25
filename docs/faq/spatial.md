---
title: "Spatial FAQ"
description: "Frequently asked questions about spatial econometrics in PanelBox — model selection, weight matrices, parameter interpretation, and diagnostics."
---

# Spatial Econometrics FAQ

Questions specific to spatial panel models in PanelBox — when to use spatial models, how to choose between SAR/SEM/SDM, weight matrix construction, parameter interpretation, and performance.

!!! tip "Looking for other topics?"
    - **General questions**: [General FAQ](general.md)
    - **Advanced methods** (GMM, VAR, Heckman): [Advanced FAQ](advanced.md)
    - **Error messages and debugging**: [Troubleshooting](troubleshooting.md)

---

## When to Use Spatial Models

??? question "When should I use spatial models instead of standard panel models?"

    Consider spatial models when:

    - Your data has a **geographic or network structure** (regions, countries, firms in supply chains)
    - **Moran's I test** shows significant spatial autocorrelation (p < 0.05)
    - Economic theory suggests **spillovers** or interactions between units
    - You observe **clustering patterns** in OLS residuals
    - **Cross-sectional dependence tests** indicate spatial patterns

    Run diagnostics first:

    ```python
    from panelbox.diagnostics import MoranIPanelTest

    moran = MoranIPanelTest(data, variable="y", W=W)
    result = moran.run()

    if result.pvalue < 0.05:
        print("Significant spatial autocorrelation — consider spatial models")
    ```

??? question "How do I test for spatial autocorrelation?"

    The **Moran's I test** is the standard diagnostic:

    ```python
    from panelbox.diagnostics import MoranIPanelTest

    moran = MoranIPanelTest(data, variable="y", W=W)
    result = moran.run()
    print(f"Moran's I = {result.statistic:.4f}, p-value = {result.pvalue:.4f}")
    ```

    - **Significant Moran's I** (p < 0.05): spatial autocorrelation exists
    - **Positive I**: similar values cluster together (most common)
    - **Negative I**: dissimilar values cluster together (checkerboard pattern)

    For more detail, use **Local Moran's I (LISA)** to identify spatial clusters:

    ```python
    from panelbox.diagnostics import LocalMoranI

    lisa = LocalMoranI(data, W=W)
    lisa_result = lisa.run()
    # Identifies hot spots, cold spots, and spatial outliers
    ```

---

## Model Selection (SAR vs SEM vs SDM)

??? question "How do I choose between SAR, SEM, and SDM?"

    Use the **Lagrange Multiplier (LM) test decision tree**:

    | LM Test Results | Recommended Model | Interpretation |
    |---|---|---|
    | Only LM-lag significant | **SAR** (Spatial Lag) | Spatial dependence in dependent variable |
    | Only LM-error significant | **SEM** (Spatial Error) | Spatial dependence in error term |
    | Both significant → check robust versions | See below | Need more specific tests |
    | Robust LM-lag significant | **SAR** | Spatial lag dominates |
    | Robust LM-error significant | **SEM** | Spatial error dominates |
    | Both robust tests significant | **SDM** or **GNS** | Both types of dependence |

    ```python
    from panelbox.diagnostics import lm_lag_test, lm_error_test, run_lm_tests

    # Run all LM tests at once
    lm_results = run_lm_tests(result, W)
    print(lm_results)
    ```

    **When in doubt:** LeSage & Pace (2009) recommend starting with **SDM** (Spatial Durbin Model). SDM nests both SAR and SEM as special cases and avoids bias from omitting relevant spatial lags.

??? question "What are the differences between all the spatial models?"

    | Model | Equation | Parameters | Use case |
    |---|---|---|---|
    | **SAR** (Spatial Lag) | $y = \rho Wy + X\beta + \varepsilon$ | $\rho$ | Spillovers in outcomes |
    | **SEM** (Spatial Error) | $y = X\beta + u$, $u = \lambda Wu + \varepsilon$ | $\lambda$ | Spatially correlated omitted variables |
    | **SDM** (Spatial Durbin) | $y = \rho Wy + X\beta + WX\theta + \varepsilon$ | $\rho, \theta$ | Both outcome and covariate spillovers |
    | **GNS** (General Nesting) | $y = \rho Wy + X\beta + WX\theta + u$, $u = \lambda Wu + \varepsilon$ | $\rho, \theta, \lambda$ | All types of spatial dependence |

    ```python
    from panelbox.models.spatial import SpatialLag, SpatialError, SpatialDurbin

    sar = SpatialLag(formula, data, entity_col, time_col, W=W).fit()
    sem = SpatialError(formula, data, entity_col, time_col, W=W).fit()
    sdm = SpatialDurbin(formula, data, entity_col, time_col, W=W).fit()
    ```

??? question "My LM tests are both significant — what should I do?"

    When both LM-lag and LM-error are significant, use the **robust versions**:

    ```python
    from panelbox.diagnostics import run_lm_tests

    lm_results = run_lm_tests(result, W)
    # Check robust LM-lag and robust LM-error
    ```

    - If **robust LM-lag** is significant but robust LM-error is not → **SAR**
    - If **robust LM-error** is significant but robust LM-lag is not → **SEM**
    - If **both robust tests** are significant → **SDM** (Spatial Durbin Model) or **GNS**

    This follows the Anselin (2005) testing strategy.

---

## Spatial Weight Matrices

??? question "How do I create a spatial weight matrix?"

    PanelBox supports multiple methods:

    **1. Contiguity-based** (shared borders):
    ```python
    import libpysal
    W = libpysal.weights.Queen.from_dataframe(gdf)
    W_array = W.full()[0]
    ```

    **2. Distance-based** (within threshold):
    ```python
    import libpysal
    coords = data[["longitude", "latitude"]].values
    W = libpysal.weights.DistanceBand.from_array(coords, threshold=500)
    ```

    **3. k-Nearest Neighbors**:
    ```python
    W = libpysal.weights.KNN.from_dataframe(gdf, k=5)
    ```

    **4. Custom matrix** (e.g., trade flows, input-output):
    ```python
    import numpy as np
    W_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    # Row-standardize manually
    row_sums = W_matrix.sum(axis=1, keepdims=True)
    W_matrix = W_matrix / row_sums
    ```

??? question "Should I row-standardize my weight matrix?"

    **Yes, in almost all cases.** Row-standardization means each row sums to 1, so the spatial lag $Wy$ becomes a weighted average of neighbors.

    Benefits:

    - Facilitates interpretation (weighted average of neighbors)
    - Ensures $\rho$ is bounded between -1 and 1
    - Makes results comparable across different W specifications
    - Required by most spatial model implementations

    When **not** to row-standardize:

    - When using inverse distance weights with specific interpretation
    - Some network models with meaningful absolute edge weights
    - When preserving absolute distance is important for the research question

??? question "How does the cutoff distance affect results?"

    The choice of distance threshold for distance-based weight matrices can affect results. **Perform a sensitivity analysis:**

    ```python
    from panelbox.models.spatial import SpatialLag

    for threshold in [100, 200, 500, 1000]:
        W_t = create_distance_weights(coords, threshold=threshold)
        model = SpatialLag(formula, data, entity_col, time_col, W=W_t)
        result = model.fit()
        print(f"Threshold={threshold}km: rho={result.rho:.3f}, "
              f"beta={result.params['x1']:.3f}")
    ```

    If results are highly sensitive to the threshold, this suggests the spatial structure is not well-defined and you should explore alternative specifications.

??? question "How do I handle islands (units with no neighbors)?"

    Islands (isolated units with no neighbors) cause problems in spatial models because their row in W is all zeros.

    **Options:**

    1. **Remove islands** from the analysis
    2. **Connect to nearest neighbor** regardless of threshold
    3. **Increase the distance threshold** to include more connections

    ```python
    import numpy as np

    # Check for islands
    W_array = W.full()[0]
    islands = np.where(W_array.sum(axis=1) == 0)[0]
    if len(islands) > 0:
        print(f"Warning: {len(islands)} islands found")
        # Option 1: Remove them
        data_no_islands = data[~data.index.isin(islands)]
    ```

---

## Parameter Interpretation

??? question "What does rho (spatial lag parameter) mean?"

    The spatial lag parameter $\rho$ in SAR/SDM models measures the strength of **spatial spillovers in outcomes**:

    - **$\rho > 0$**: positive spatial dependence (similar values cluster together)
    - **$\rho < 0$**: negative spatial dependence (dissimilar neighbors)
    - **$\rho = 0$**: no spatial dependence
    - **$|\rho| < 1$**: required for model stability

    **Interpretation example:** $\rho = 0.3$ means a 10% increase in neighbors' average $y$ is associated with a 3% increase in own $y$. The **spatial multiplier** is $1/(1-\rho)$, so $\rho = 0.3$ gives a total multiplier of approximately 1.43.

??? question "What does lambda (spatial error parameter) mean?"

    The spatial error parameter $\lambda$ in SEM models captures **spatial correlation in unobserved factors**:

    - **$\lambda > 0$**: positive spatial correlation in errors (spatially correlated omitted variables)
    - **$\lambda < 0$**: negative spatial correlation in errors
    - **$|\lambda| < 1$**: required for stability

    Unlike $\rho$, $\lambda$ does not have a direct economic interpretation for spillovers. It indicates that omitted variables have a spatial pattern and that ignoring this would lead to inefficient estimates and biased standard errors.

??? question "How do I interpret direct vs indirect effects?"

    In spatial models (especially SAR and SDM), the coefficients $\beta$ do **not** directly give the marginal effects because of the spatial multiplier $(I - \rho W)^{-1}$.

    - **Direct effect**: impact of changing $X_i$ on $y_i$ (own-unit effect)
    - **Indirect effect**: impact of changing $X_i$ on $y_j$ for $j \neq i$ (spillover)
    - **Total effect**: direct + indirect

    ```python
    from panelbox.effects import compute_spatial_effects

    effects = compute_spatial_effects(result, W)
    print(f"Direct effect of x1: {effects.direct['x1']:.4f}")
    print(f"Indirect effect of x1: {effects.indirect['x1']:.4f}")
    print(f"Total effect of x1: {effects.total['x1']:.4f}")
    ```

    **Example:** If direct effect of education = 0.5 and indirect = 0.2:

    - 1% increase in own education → 0.5% increase in own outcome
    - 1% increase in own education → 0.2% increase in neighbors' outcomes
    - Total impact = 0.7%

    !!! warning "Do not interpret raw coefficients as marginal effects"
        In SAR/SDM models, always compute and report the effects decomposition rather than the raw $\beta$ coefficients.

??? question "What should I report in my paper?"

    For spatial panel results, report:

    1. **Diagnostics**: Moran's I test, LM tests (justifying model choice)
    2. **Model comparison**: AIC, BIC, Log-likelihood across OLS / SAR / SEM / SDM
    3. **Spatial parameters**: $\rho$ and/or $\lambda$ with standard errors
    4. **Effects decomposition**: direct, indirect, and total effects (for SAR/SDM)
    5. **Post-estimation**: residual Moran's I (should be non-significant)
    6. **Weight matrix**: description, summary statistics (avg. neighbors, sparsity)

---

## Dynamic Spatial Panels

??? question "When should I add a time lag to a spatial model?"

    Add a time lag ($y_{i,t-1}$) when there is **persistence over time** — the outcome depends on its own past values, not just spatial neighbors.

    ```python
    from panelbox.models.spatial import DynamicSpatialPanel

    dsp = DynamicSpatialPanel(
        formula, data, entity_col, time_col, W=W,
        lags=1  # Include y_{t-1}
    )
    result = dsp.fit()
    ```

    Examples requiring dynamic spatial models:

    - Regional GDP growth with both spatial spillovers and persistence
    - House prices influenced by neighbors and lagged own prices
    - Pollution levels with spatial diffusion and temporal inertia

??? question "How do I interpret the space-time lag (phi * W * y_{t-1})?"

    The space-time lag $\phi W y_{t-1}$ captures **diffusion effects** — how neighbors' past outcomes affect current own outcomes.

    - **$\phi > 0$**: neighbors' past high values pull your current value up
    - **$\phi < 0$**: neighbors' past high values push your current value down

    This differs from the contemporaneous spatial lag $\rho W y_t$, which captures instantaneous feedback.

---

## Diagnostics

??? question "How do I interpret Local Moran's I (LISA) results?"

    Local Moran's I identifies four types of spatial clusters:

    | Cluster Type | Meaning | Local I |
    |---|---|---|
    | **High-High** (hot spot) | High values surrounded by high neighbors | Positive |
    | **Low-Low** (cold spot) | Low values surrounded by low neighbors | Positive |
    | **High-Low** (spatial outlier) | High value surrounded by low neighbors | Negative |
    | **Low-High** (spatial outlier) | Low value surrounded by high neighbors | Negative |

    ```python
    from panelbox.diagnostics import LocalMoranI

    lisa = LocalMoranI(data, W=W)
    result = lisa.run()
    # Plot cluster map if geopandas + shapefile available
    ```

    Significant LISA statistics (after multiple-testing correction) identify locations that deviate from spatial randomness.

??? question "How do I test if the spatial model improved over OLS?"

    Use multiple criteria:

    **1. Likelihood Ratio Test:**
    ```python
    from scipy import stats

    lr_stat = 2 * (spatial_result.llf - ols_result.llf)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    print(f"LR test p-value: {p_value:.4f}")
    ```

    **2. Information Criteria:**
    ```python
    print(f"OLS AIC: {ols_result.aic:.2f}")
    print(f"SAR AIC: {sar_result.aic:.2f}")
    # Lower AIC is better
    ```

    **3. Residual Moran's I** (should be non-significant after spatial model):
    ```python
    moran_resid = MoranIPanelTest(spatial_result.resid, W=W)
    result = moran_resid.run()
    # p > 0.05: no remaining spatial autocorrelation
    ```

---

## Inference and Standard Errors

??? question "What is Spatial HAC and when should I use it?"

    Spatial HAC (Conley, 1999) standard errors are robust to both spatial and temporal correlation. Use them when:

    - You have both **spatial and temporal dependence**
    - Standard errors seem too small (overly optimistic)
    - Cross-sectional dependence persists in residuals

    ```python
    from panelbox.standard_errors import SpatialHAC

    shac = SpatialHAC(
        result,
        coords=data[["lon", "lat"]].values,
        spatial_cutoff=500,   # km for spatial correlation
        temporal_cutoff=2     # periods for temporal correlation
    )
    robust_se = shac.compute()
    ```

??? question "Driscoll-Kraay vs Spatial HAC — which should I use?"

    | Feature | Spatial HAC | Driscoll-Kraay |
    |---|---|---|
    | **Requires W** | No (uses distance cutoff) | No |
    | **Spatial structure** | Explicit (distance-based) | General cross-sectional dependence |
    | **Flexibility** | More specific | More agnostic |

    **Use Spatial HAC** when you know the spatial structure and can specify a distance cutoff.

    **Use Driscoll-Kraay** when the spatial structure is unknown or you want to be agnostic about the form of cross-sectional dependence.

    ```python
    from panelbox.standard_errors import DriscollKraayStandardErrors

    dk = DriscollKraayStandardErrors(result, lag_cutoff=2)
    robust_se = dk.compute()
    ```

---

## Performance

??? question "Spatial estimation is slow. How can I speed it up?"

    Spatial ML estimation requires computing the log-determinant of $(I - \rho W)$ for each optimization step, which is expensive for large N.

    **Performance guidelines:**

    | N (entities) | Estimation Time | Recommended Approach |
    |---|---|---|
    | < 500 | < 10 seconds | Standard ML |
    | 500 - 1,000 | 10 - 60 seconds | Standard ML, sparse W |
    | 1,000 - 5,000 | 1 - 5 minutes | Sparse matrices required |
    | 5,000 - 10,000 | 5 - 30 minutes | Chebyshev approximation |
    | > 10,000 | > 30 minutes | Consider GMM/2SLS |

    **Tips:**

    ```python
    # Use sparse weight matrix
    from scipy import sparse
    W_sparse = sparse.csr_matrix(W_array)

    # Pre-compute eigenvalues (speeds up repeated estimation)
    eigenvalues = np.linalg.eigvalsh(W_array)

    # Use Chebyshev approximation for log-determinant
    result = model.fit(method="chebyshev", order=50)
    ```

??? question "Can I use spatial models with very large panels (N > 10,000)?"

    For very large N, ML estimation becomes prohibitively slow. Alternatives:

    1. **GMM/2SLS estimation** — faster but less efficient
    2. **Chebyshev log-determinant approximation** — reduces computation cost
    3. **Sparse weight matrices** — critical for distance-based W
    4. **Subsample for exploration** — estimate on subset, then run on full data

    Monitor sparsity of your weight matrix:

    ```python
    sparsity = (W_array == 0).sum() / W_array.size
    print(f"Weight matrix sparsity: {sparsity:.1%}")
    # If > 90% sparse, use sparse matrix format for significant speedup
    ```

---

## See Also

- [General FAQ](general.md) — getting started, model selection, results interpretation
- [Advanced FAQ](advanced.md) — GMM, VAR, Heckman, cointegration
- [Troubleshooting](troubleshooting.md) — error messages and debugging
- [Spatial API Reference](../api/spatial.md)
- [Spatial Theory](../theory/spatial-theory.md)
- [Spatial Tutorial](../tutorials/spatial.md)
