CUE-GMM API Reference
=====================

.. currentmodule:: panelbox.gmm

Overview
--------

The Continuous Updated Estimator (CUE) for GMM provides better finite-sample properties than standard two-step GMM by continuously updating the weighting matrix during optimization.

.. autosummary::
   :toctree: generated/

   ContinuousUpdatedGMM

ContinuousUpdatedGMM
--------------------

.. autoclass:: ContinuousUpdatedGMM
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~ContinuousUpdatedGMM.fit
      ~ContinuousUpdatedGMM.j_statistic
      ~ContinuousUpdatedGMM.compare_with_two_step

   .. rubric:: Attributes

   .. autosummary::

      ~ContinuousUpdatedGMM.params_
      ~ContinuousUpdatedGMM.vcov_
      ~ContinuousUpdatedGMM.j_stat_
      ~ContinuousUpdatedGMM.j_pvalue_
      ~ContinuousUpdatedGMM.converged_

Examples
--------

Basic Usage
~~~~~~~~~~~

Simple IV regression with CUE-GMM::

    import pandas as pd
    import numpy as np
    from panelbox.gmm import ContinuousUpdatedGMM

    # Generate data
    np.random.seed(42)
    n = 500
    z1 = np.random.normal(0, 1, n)
    z2 = np.random.normal(0, 1, n)
    v = np.random.normal(0, 1, n)
    x = 0.5 + 0.8 * z1 + 0.6 * z2 + v
    epsilon = np.random.normal(0, 1, n) + 0.5 * v
    y = 1.0 + 2.0 * x + epsilon

    # Create DataFrame
    data = pd.DataFrame({
        'y': y, 'x': x, 'z1': z1, 'z2': z2,
        'entity': np.arange(n), 'time': 1
    })
    data = data.set_index(['entity', 'time'])

    # Estimate CUE-GMM
    model = ContinuousUpdatedGMM(
        data=data,
        dep_var='y',
        exog_vars=['x'],
        instruments=['z1', 'z2'],
        weighting='hac'
    )
    results = model.fit()

    # View results
    print(results.summary())

    # Check overidentification
    j_test = model.j_statistic()
    print(f"J-statistic: {j_test['statistic']:.4f}")
    print(f"p-value: {j_test['pvalue']:.4f}")

Weighting Options
~~~~~~~~~~~~~~~~~

HAC-robust weighting (Newey-West)::

    model = ContinuousUpdatedGMM(
        data=data,
        dep_var='y',
        exog_vars=['x'],
        instruments=['z1', 'z2'],
        weighting='hac',
        bandwidth='auto'  # Automatic bandwidth selection
    )

Cluster-robust weighting::

    model = ContinuousUpdatedGMM(
        data=data,
        dep_var='y',
        exog_vars=['x'],
        instruments=['z1', 'z2'],
        weighting='cluster'  # Cluster by entity (first index level)
    )

Homoskedastic weighting::

    model = ContinuousUpdatedGMM(
        data=data,
        dep_var='y',
        exog_vars=['x'],
        instruments=['z1', 'z2'],
        weighting='homoskedastic'
    )

Comparison with Two-Step GMM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare CUE-GMM efficiency with two-step GMM::

    from panelbox.gmm.estimator import GMMEstimator

    # Estimate two-step GMM
    estimator = GMMEstimator()
    ts_params, ts_vcov, _, _ = estimator.two_step(
        model.y, model.X, model.Z
    )

    # Create GMMResults for two-step
    # (see full example in validation tests)

    # Compare
    comparison = model.compare_with_two_step(ts_results)
    print(comparison)

Output::

                 CUE Coef   TS Coef      Diff   CUE SE    TS SE  Efficiency Ratio
    const         0.9876    0.9854    0.0022   0.0523   0.0567            1.16
    x             2.0123    2.0098    0.0025   0.0412   0.0445            1.17

Efficiency Ratio > 1 indicates CUE is more efficient (lower variance).

Custom Starting Values
~~~~~~~~~~~~~~~~~~~~~~

Provide custom starting values for optimization::

    # Use OLS estimates as starting values
    from sklearn.linear_model import LinearRegression

    X_arr = np.column_stack([np.ones(len(data)), data['x']])
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X_arr, data['y'])
    start_values = ols.coef_

    # Fit CUE-GMM with custom start
    results = model.fit(start_params=start_values)

Theory
------

CUE-GMM Objective
~~~~~~~~~~~~~~~~~

The CUE minimizes::

    Q(β) = g(β)' W(β)⁻¹ g(β)

where:
    - g(β) = (1/N) Σᵢ Zᵢ'εᵢ(β) are the moment conditions
    - W(β) = (1/N) Σᵢ gᵢ(β) gᵢ(β)' is the weighting matrix
    - W(β) is updated at each iteration (unlike two-step GMM)

Variance Estimator
~~~~~~~~~~~~~~~~~~

For CUE-GMM::

    Var(β̂) = (G' W⁻¹ G)⁻¹

where:
    - G = ∂g(β̂)/∂β' is the Jacobian of moments
    - W = weighting matrix at β̂

Hansen J-Test
~~~~~~~~~~~~~

Test of overidentifying restrictions::

    J = N × Q(β̂) ~ χ²(L - K)

where:
    - L = number of instruments
    - K = number of parameters
    - Reject if p-value < 0.05 → model misspecified

Notes
-----

**Computational Cost**

CUE-GMM is more expensive than two-step GMM because it recomputes the weighting
matrix W(β) at each iteration of the optimizer. For large datasets (N > 10,000),
consider using two-step GMM instead.

**Convergence**

CUE optimization can be sensitive to starting values. The default behavior uses
two-step GMM estimates as starting values, which generally works well. If
convergence fails, try:

- Different starting values
- Tighter tolerance (``tol=1e-8``)
- More iterations (``max_iter=200``)
- Regularization (``regularize=True``)

**Weighting Matrix Singularity**

If the weighting matrix W(β) is near-singular (high condition number), the
solver may fail. Enable regularization to add a small ridge::

    model = ContinuousUpdatedGMM(..., regularize=True)

This adds εI to W if needed, where ε is a small value.

**HAC Bandwidth**

The automatic bandwidth selection uses Newey-West formula::

    L = floor(4 × (T/100)^(2/9))

For custom bandwidth::

    model = ContinuousUpdatedGMM(..., bandwidth=5)

See Also
--------

BiasCorrectedGMM : Bias correction for dynamic panels
GMMDiagnostics : Diagnostic tests for GMM
DifferenceGMM : Arellano-Bond difference GMM
SystemGMM : Blundell-Bond system GMM

References
----------

.. [1] Hansen, L.P., Heaton, J., & Yaron, A. (1996). "Finite-Sample
       Properties of Some Alternative GMM Estimators." Journal of Business &
       Economic Statistics, 14(3), 262-280.

.. [2] Newey, W.K., & West, K.D. (1987). "A Simple, Positive Semi-Definite,
       Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
       Econometrica, 55(3), 703-708.
