"""
Continuous Updated GMM Estimator
==================================

Implements the Continuous Updated Estimator (CUE) for GMM estimation,
as developed by Hansen, Heaton, and Yaron (1996).

The CUE-GMM updates the weighting matrix W(β) continuously during optimization,
which provides better finite-sample properties than standard two-step GMM.

Classes
-------
ContinuousUpdatedGMM : CUE-GMM estimator

References
----------
.. [1] Hansen, L. P., Heaton, J., & Yaron, A. (1996). "Finite-Sample
       Properties of Some Alternative GMM Estimators." Journal of Business &
       Economic Statistics, 14(3), 262-280.

.. [2] Newey, W. K., & West, K. D. (1987). "A Simple, Positive Semi-Definite,
       Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
       Econometrica, 55(3), 703-708.

Examples
--------
>>> from panelbox.gmm import ContinuousUpdatedGMM
>>> model = ContinuousUpdatedGMM(
...     data=panel_data,
...     dep_var='y',
...     exog_vars=['x1', 'x2'],
...     instruments=['z1', 'z2', 'z3']
... )
>>> results = model.fit()
>>> print(results.summary())
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
from scipy.stats import chi2

from panelbox.gmm.estimator import GMMEstimator
from panelbox.gmm.results import GMMResults


class ContinuousUpdatedGMM:
    """
    Continuous Updated Estimator (CUE) for GMM.

    The CUE minimizes the GMM criterion function where the weighting matrix
    W(β) is continuously updated as a function of the parameters:

        β̂ᶜᵘᵉ = argmin Q(β) = g(β)' W(β)⁻¹ g(β)

    where g(β) are the moment conditions and W(β) = (1/N) Σᵢ gᵢ(β) gᵢ(β)'.

    This approach has better finite-sample properties than two-step GMM:
    - Lower finite-sample bias
    - Invariant to moment normalization
    - Often more efficient

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex (entity_id, time_id)
    dep_var : str
        Name of dependent variable
    exog_vars : List[str]
        Names of exogenous regressors
    instruments : List[str]
        Names of instrumental variables
    weighting : str, default='hac'
        Type of weighting matrix:
        - 'hac': HAC-robust (Newey-West)
        - 'cluster': Cluster-robust by entity
        - 'homoskedastic': Simple weighting
    bandwidth : Union[str, int], default='auto'
        Bandwidth for HAC estimation:
        - 'auto': Automatic selection (Newey-West)
        - int: Fixed bandwidth
    se_type : str, default='analytical'
        Type of standard errors:
        - 'analytical': Analytical HAC/cluster SEs
        - 'bootstrap': Bootstrap SEs
    n_bootstrap : int, default=999
        Number of bootstrap replications (if se_type='bootstrap')
    bootstrap_method : str, default='residual'
        Bootstrap method:
        - 'residual': Residual bootstrap
        - 'pairs': Pairs bootstrap
    max_iter : int, default=100
        Maximum iterations for optimization
    tol : float, default=1e-6
        Convergence tolerance
    regularize : bool, default=True
        Add small ridge to W if near-singular

    Attributes
    ----------
    params_ : np.ndarray
        Estimated parameters (after fit)
    vcov_ : np.ndarray
        Variance-covariance matrix
    bse_ : np.ndarray
        Bootstrap standard errors (if se_type='bootstrap')
    bootstrap_params_ : np.ndarray
        Bootstrap parameter samples (n_bootstrap x k)
    j_stat_ : float
        Hansen J-statistic for overidentification test
    converged_ : bool
        Whether optimization converged

    Methods
    -------
    fit : Estimate CUE-GMM
    j_statistic : Compute Hansen J-test
    compare_with_two_step : Compare efficiency with two-step GMM
    conf_int : Compute confidence intervals
    bse : Get bootstrap standard errors

    Notes
    -----
    The CUE optimization can be computationally intensive and may require
    good starting values. By default, two-step GMM estimates are used as
    starting values.

    The weighting matrix W(β) is updated at each iteration, which requires
    recomputing the HAC variance for each candidate β. This makes CUE
    significantly more expensive than two-step GMM.

    Examples
    --------
    Basic IV regression with CUE-GMM:

    >>> model = ContinuousUpdatedGMM(
    ...     data=data,
    ...     dep_var='wage',
    ...     exog_vars=['education', 'experience'],
    ...     instruments=['father_education', 'mother_education', 'siblings']
    ... )
    >>> results = model.fit()
    >>> print(f"J-statistic: {results.j_statistic:.4f}")
    >>> print(f"p-value: {results.j_pvalue:.4f}")

    Compare with two-step GMM:

    >>> from panelbox.gmm import TwoStepGMM
    >>> two_step = TwoStepGMM(data, dep_var, exog_vars, instruments)
    >>> ts_results = two_step.fit()
    >>> comparison = results.compare_with_two_step(ts_results)
    >>> print(comparison)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dep_var: str,
        exog_vars: List[str],
        instruments: List[str],
        weighting: str = "hac",
        bandwidth: Union[str, int] = "auto",
        se_type: str = "analytical",
        n_bootstrap: int = 999,
        bootstrap_method: str = "residual",
        max_iter: int = 100,
        tol: float = 1e-6,
        regularize: bool = True,
    ):
        """Initialize CUE-GMM estimator."""
        self.data = data
        self.dep_var = dep_var
        self.exog_vars = exog_vars
        self.instruments = instruments
        self.weighting = weighting
        self.bandwidth = bandwidth
        self.se_type = se_type
        self.n_bootstrap = n_bootstrap
        self.bootstrap_method = bootstrap_method
        self.max_iter = max_iter
        self.tol = tol
        self.regularize = regularize

        # Validate inputs
        self._validate_inputs()

        # Prepare data
        self._prepare_data()

        # Initialize attributes (set during fit)
        self.params_ = None
        self.vcov_ = None
        self.bse_ = None
        self.bootstrap_params_ = None
        self.j_stat_ = None
        self.j_pvalue_ = None
        self.converged_ = False
        self.niter_ = None
        self.criterion_value_ = None

    def _validate_inputs(self):
        """Validate input parameters."""
        # Check weighting type
        valid_weighting = ["hac", "cluster", "homoskedastic"]
        if self.weighting not in valid_weighting:
            raise ValueError(f"weighting must be one of {valid_weighting}, got {self.weighting}")

        # Check bandwidth
        if not (isinstance(self.bandwidth, (int, str))):
            raise TypeError("bandwidth must be int or 'auto'")
        if isinstance(self.bandwidth, str) and self.bandwidth != "auto":
            raise ValueError("bandwidth string must be 'auto'")
        if isinstance(self.bandwidth, int) and self.bandwidth < 0:
            raise ValueError("bandwidth must be non-negative")

        # Check se_type
        valid_se_types = ["analytical", "bootstrap"]
        if self.se_type not in valid_se_types:
            raise ValueError(f"se_type must be one of {valid_se_types}, got {self.se_type}")

        # Check bootstrap parameters
        if self.se_type == "bootstrap":
            if not isinstance(self.n_bootstrap, int) or self.n_bootstrap < 1:
                raise ValueError("n_bootstrap must be a positive integer")
            valid_boot_methods = ["residual", "pairs"]
            if self.bootstrap_method not in valid_boot_methods:
                raise ValueError(
                    f"bootstrap_method must be one of {valid_boot_methods}, "
                    f"got {self.bootstrap_method}"
                )

        # Check data
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("data must be pandas DataFrame")

        # Check variable names exist
        all_vars = [self.dep_var] + self.exog_vars + self.instruments
        missing = [v for v in all_vars if v not in self.data.columns]
        if missing:
            raise ValueError(f"Variables not found in data: {missing}")

    def _prepare_data(self):
        """Extract and validate data arrays."""
        # Extract dependent variable
        self.y = self.data[self.dep_var].values.reshape(-1, 1)

        # Extract exogenous variables (add intercept)
        X_df = self.data[self.exog_vars]
        self.X = np.column_stack([np.ones(len(X_df)), X_df.values])
        self.k = self.X.shape[1]  # Number of parameters (including intercept)

        # Extract instruments (add intercept)
        Z_df = self.data[self.instruments]
        self.Z = np.column_stack([np.ones(len(Z_df)), Z_df.values])
        self.n_instruments = self.Z.shape[1]

        # Sample size
        self.n = len(self.y)

        # Check dimensions
        if self.n_instruments < self.k:
            raise ValueError(
                f"Underidentified: {self.n_instruments} instruments < " f"{self.k} parameters"
            )

        # Overidentification degree
        self.overid_df = self.n_instruments - self.k

        # Number of moments (for performance warnings)
        self.m = self.n_instruments

    def _criterion(self, params: np.ndarray) -> float:
        """
        GMM criterion function Q(β) = g(β)' W(β)⁻¹ g(β).

        Parameters
        ----------
        params : np.ndarray
            Parameter vector (k x 1)

        Returns
        -------
        Q : float
            Criterion value (scalar)

        Notes
        -----
        The key difference from two-step GMM is that W is a function of β,
        requiring recomputation at each iteration.
        """
        # Compute residuals
        residuals = self.y - self.X @ params.reshape(-1, 1)

        # Compute moment conditions g(β) = (1/N) Z'ε
        moments = (1 / self.n) * (self.Z.T @ residuals)  # (n_instruments x 1)

        # Compute weighting matrix W(β)
        W = self._compute_weighting_matrix(params, residuals)

        # Compute criterion Q(β) = g' W⁻¹ g
        try:
            Q_matrix = moments.T @ linalg.solve(W, moments, assume_a="pos")
            Q = float(Q_matrix.item())  # Extract scalar from 1x1 matrix
        except linalg.LinAlgError:
            # Singular W, use pseudo-inverse with regularization
            if self.regularize:
                eps = 1e-8 * np.trace(W) / W.shape[0]
                W_reg = W + eps * np.eye(W.shape[0])
                Q_matrix = moments.T @ linalg.solve(W_reg, moments)
                Q = float(Q_matrix.item())
            else:
                Q_matrix = moments.T @ linalg.lstsq(W, moments)[0]
                Q = float(Q_matrix.item())

        return Q

    def _compute_weighting_matrix(self, params: np.ndarray, residuals: np.ndarray) -> np.ndarray:
        """
        Compute weighting matrix W(β).

        Parameters
        ----------
        params : np.ndarray
            Current parameter estimates
        residuals : np.ndarray
            Residuals ε = y - X'β

        Returns
        -------
        W : np.ndarray
            Weighting matrix (n_instruments x n_instruments)

        Notes
        -----
        W = (1/N) Σᵢ gᵢ(β) gᵢ(β)' where gᵢ(β) = Zᵢ εᵢ(β)

        For HAC-robust: Apply Newey-West kernel
        For cluster-robust: Cluster by entity
        For homoskedastic: Simple moment variance
        """
        # Compute moment contributions gᵢ(β) = Zᵢ εᵢ
        moment_contributions = self.Z * residuals  # (n x n_instruments)

        if self.weighting == "homoskedastic":
            # Simple moment variance: W = (1/N) Σ gᵢgᵢ'
            W = (1 / self.n) * (moment_contributions.T @ moment_contributions)

        elif self.weighting == "hac":
            # HAC-robust using Newey-West
            W = self._compute_hac_variance(moment_contributions)

        elif self.weighting == "cluster":
            # Cluster-robust (by entity if panel data)
            W = self._compute_cluster_variance(moment_contributions)

        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")

        return W

    def _compute_hac_variance(self, moment_contributions: np.ndarray) -> np.ndarray:
        """
        Compute HAC-robust variance using Newey-West.

        Parameters
        ----------
        moment_contributions : np.ndarray
            Moment contributions gᵢ = Zᵢ εᵢ (n x n_instruments)

        Returns
        -------
        W : np.ndarray
            HAC-robust variance matrix

        Notes
        -----
        Newey-West (1987) kernel:
        W = Γ₀ + Σₗ₌₁ᴸ w(l) (Γₗ + Γₗ')

        where Γₗ = (1/N) Σₜ gₜ gₜ₋ₗ' and w(l) = 1 - l/(L+1) (Bartlett kernel)
        """
        # Determine bandwidth
        if self.bandwidth == "auto":
            # Newey-West automatic bandwidth: L = floor(4(T/100)^(2/9))
            L = int(np.floor(4 * (self.n / 100) ** (2 / 9)))
        else:
            L = self.bandwidth

        # Γ₀: contemporaneous covariance
        Gamma_0 = (1 / self.n) * (moment_contributions.T @ moment_contributions)

        # Initialize W with Γ₀
        W = Gamma_0.copy()

        # Add lagged autocovariances with Bartlett weights
        for lag in range(1, L + 1):
            # Bartlett weight
            weight = 1 - lag / (L + 1)

            # Γₗ = (1/N) Σₜ gₜ gₜ₋ₗ'
            g_t = moment_contributions[lag:]
            g_t_lag = moment_contributions[:-lag]
            Gamma_l = (1 / self.n) * (g_t.T @ g_t_lag)

            # Add weighted sum: w(l)(Γₗ + Γₗ')
            W += weight * (Gamma_l + Gamma_l.T)

        return W

    def _compute_cluster_variance(self, moment_contributions: np.ndarray) -> np.ndarray:
        """
        Compute cluster-robust variance.

        Parameters
        ----------
        moment_contributions : np.ndarray
            Moment contributions gᵢ = Zᵢ εᵢ

        Returns
        -------
        W : np.ndarray
            Cluster-robust variance matrix

        Notes
        -----
        For panel data, clusters by entity (first level of MultiIndex).
        W = (1/N) Σ_c (Σᵢ∈c gᵢ)(Σᵢ∈c gᵢ)'
        """
        # Check if data has MultiIndex for clustering
        if not isinstance(self.data.index, pd.MultiIndex):
            warnings.warn("Data does not have MultiIndex, using homoskedastic variance")
            return (1 / self.n) * (moment_contributions.T @ moment_contributions)

        # Get cluster identifiers (entity level)
        clusters = self.data.index.get_level_values(0).values

        # Compute cluster sums
        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)

        # Initialize variance
        W = np.zeros((self.n_instruments, self.n_instruments))

        for cluster_id in unique_clusters:
            # Get observations in this cluster
            mask = clusters == cluster_id
            cluster_moments = moment_contributions[mask].sum(axis=0, keepdims=True).T

            # Add outer product
            W += cluster_moments @ cluster_moments.T

        # Scale by N
        W = W / self.n

        return W

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        verbose: bool = False,
    ) -> GMMResults:
        """
        Estimate CUE-GMM parameters.

        Parameters
        ----------
        start_params : np.ndarray, optional
            Starting values for optimization. If None, uses two-step GMM.
        method : str, default='L-BFGS-B'
            Optimization method for scipy.optimize.minimize
        verbose : bool, default=False
            Print optimization progress

        Returns
        -------
        results : GMMResults
            Estimation results object

        Notes
        -----
        The optimization minimizes Q(β) = g(β)' W(β)⁻¹ g(β).

        Good starting values are critical. By default, two-step GMM estimates
        are used. Multiple random starts can be tried by calling fit()
        multiple times with different start_params.

        Examples
        --------
        >>> results = model.fit()
        >>> print(results.summary())

        With custom starting values:

        >>> start = np.array([0.5, 0.1, -0.2])
        >>> results = model.fit(start_params=start)
        """
        # Performance warnings
        if self.m > 50:
            warnings.warn(
                "CUE-GMM with >50 moments may be very slow. "
                "Consider reducing moment conditions or using two-step GMM.",
                UserWarning,
            )

        if self.n > 10000:
            warnings.warn(
                "CUE-GMM with N>10,000 may take several minutes. "
                "Consider using two-step GMM for large samples.",
                UserWarning,
            )

        # Get starting values
        if start_params is None:
            # Use two-step GMM as starting values
            if verbose:
                print("Computing two-step GMM for starting values...")
            estimator = GMMEstimator(tol=self.tol, max_iter=self.max_iter)
            beta_ts, vcov_ts, _, _ = estimator.two_step(self.y, self.X, self.Z)
            start_params = beta_ts.flatten()
        else:
            start_params = np.asarray(start_params).flatten()

        # Validate starting values
        if len(start_params) != self.k:
            raise ValueError(f"start_params must have length {self.k}, got {len(start_params)}")

        # Optimize criterion function
        if verbose:
            print("Optimizing CUE-GMM criterion...")

        result = optimize.minimize(
            self._criterion,
            start_params,
            method=method,
            options={"disp": verbose, "maxiter": self.max_iter, "ftol": self.tol},
        )

        # Store results
        self.params_ = result.x
        self.converged_ = result.success
        self.niter_ = result.nit
        self.criterion_value_ = result.fun

        if not self.converged_:
            warnings.warn(f"CUE-GMM optimization did not converge: {result.message}")

        # Compute variance-covariance matrix
        if self.se_type == "analytical":
            self.vcov_ = self._compute_variance()
            self.bse_ = np.sqrt(np.diag(self.vcov_))
        elif self.se_type == "bootstrap":
            if verbose:
                print(f"Computing bootstrap SEs with {self.n_bootstrap} replications...")
            self.bootstrap_params_ = self._compute_bootstrap(verbose=verbose)
            self.bse_ = np.std(self.bootstrap_params_, axis=0, ddof=1)
            self.vcov_ = np.cov(self.bootstrap_params_, rowvar=False)

        # Compute J-statistic
        self.j_stat_ = self.n * self.criterion_value_
        self.j_pvalue_ = 1 - chi2.cdf(self.j_stat_, self.overid_df)

        # Build results object
        results = self._create_results()

        return results

    def _compute_variance(self) -> np.ndarray:
        """
        Compute variance-covariance matrix for CUE-GMM.

        Returns
        -------
        vcov : np.ndarray
            Variance-covariance matrix (k x k)

        Notes
        -----
        For CUE-GMM:
        V̂ = (Ḡ' Ŵ⁻¹ Ḡ)⁻¹

        where Ḡ = ∂g(β̂)/∂β' is the gradient of moment conditions,
        computed via numerical differentiation.
        """
        # Compute gradient of moments Ḡ = ∂g(β)/∂β'
        # g(β) = (1/N) Z'(y - X'β) = (1/N) Z'ε
        # ∂g/∂β = -(1/N) Z'X
        G = -(1 / self.n) * (self.Z.T @ self.X)  # (n_instruments x k)

        # Compute weighting matrix at β̂
        residuals = self.y - self.X @ self.params_.reshape(-1, 1)
        W = self._compute_weighting_matrix(self.params_, residuals)

        # Compute variance: V = (G' W⁻¹ G)⁻¹
        try:
            GtW = G.T @ linalg.solve(W, G, assume_a="pos")
            vcov = linalg.inv(GtW)
        except linalg.LinAlgError:
            warnings.warn("Singular variance matrix, using pseudo-inverse")
            if self.regularize:
                eps = 1e-8 * np.trace(W) / W.shape[0]
                W_reg = W + eps * np.eye(W.shape[0])
                GtW = G.T @ linalg.solve(W_reg, G)
            else:
                GtW = G.T @ linalg.lstsq(W, G)[0]
            vcov = linalg.pinv(GtW)

        return vcov

    def _compute_bootstrap(self, verbose: bool = False) -> np.ndarray:
        """
        Compute bootstrap standard errors.

        Parameters
        ----------
        verbose : bool, default=False
            Print progress information

        Returns
        -------
        bootstrap_params : np.ndarray
            Bootstrap parameter estimates (n_bootstrap x k)

        Notes
        -----
        Implements two bootstrap methods:
        - 'residual': Bootstrap residuals (recommended for IV/GMM)
        - 'pairs': Bootstrap (y, X, Z) pairs

        The residual bootstrap procedure:
        1. Estimate model, get residuals ε̂ᵢ
        2. For b = 1, ..., B:
           a. Draw residuals ε*ᵢ with replacement
           b. Generate y*ᵢ = X'β̂ + ε*ᵢ
           c. Re-estimate model with (y*, X, Z)
           d. Store β̂*_b
        3. Compute SE = std(β̂*_b)
        """
        bootstrap_params = np.zeros((self.n_bootstrap, self.k))

        # Compute residuals from original estimation
        residuals = self.y - self.X @ self.params_.reshape(-1, 1)

        # Set random seed for reproducibility
        rng = np.random.RandomState(42)

        for b in range(self.n_bootstrap):
            if verbose and (b + 1) % 100 == 0:
                print(f"  Bootstrap iteration {b + 1}/{self.n_bootstrap}")

            if self.bootstrap_method == "residual":
                # Residual bootstrap
                # Draw residuals with replacement
                boot_indices = rng.choice(self.n, size=self.n, replace=True)
                boot_residuals = residuals[boot_indices]

                # Generate bootstrap dependent variable
                y_boot = self.X @ self.params_.reshape(-1, 1) + boot_residuals

                # Use original X and Z
                X_boot = self.X
                Z_boot = self.Z

            elif self.bootstrap_method == "pairs":
                # Pairs bootstrap
                # Draw observations with replacement
                boot_indices = rng.choice(self.n, size=self.n, replace=True)
                y_boot = self.y[boot_indices]
                X_boot = self.X[boot_indices]
                Z_boot = self.Z[boot_indices]

            # Re-estimate model with bootstrap sample
            try:
                # Create temporary model with bootstrap data
                temp_params = self._estimate_bootstrap_sample(y_boot, X_boot, Z_boot)
                bootstrap_params[b] = temp_params
            except Exception as e:
                if verbose:
                    warnings.warn(f"Bootstrap iteration {b + 1} failed: {str(e)}")
                # Use original estimates if bootstrap fails
                bootstrap_params[b] = self.params_

        return bootstrap_params

    def _estimate_bootstrap_sample(self, y: np.ndarray, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Estimate CUE-GMM on a bootstrap sample.

        Parameters
        ----------
        y : np.ndarray
            Bootstrap dependent variable
        X : np.ndarray
            Bootstrap exogenous variables
        Z : np.ndarray
            Bootstrap instruments

        Returns
        -------
        params : np.ndarray
            Estimated parameters
        """
        # Temporarily store original data
        y_orig, X_orig, Z_orig = self.y, self.X, self.Z
        n_orig = self.n

        # Set bootstrap data
        self.y = y
        self.X = X
        self.Z = Z
        self.n = len(y)

        # Use original parameters as starting values for speed
        start = self.params_.copy()

        # Optimize with reduced tolerance for speed
        result = optimize.minimize(
            self._criterion,
            start,
            method="L-BFGS-B",
            options={"maxiter": 50, "ftol": 1e-4},
        )

        # Restore original data
        self.y, self.X, self.Z = y_orig, X_orig, Z_orig
        self.n = n_orig

        return result.x

    def _create_results(self) -> GMMResults:
        """Create GMMResults object with estimation results."""
        # Parameter names
        param_names = ["const"] + self.exog_vars

        # Convert params and standard errors to pd.Series
        params_series = pd.Series(self.params_, index=param_names)
        std_errors = np.sqrt(np.diag(self.vcov_))
        std_errors_series = pd.Series(std_errors, index=param_names)

        # Compute t-values and p-values
        tvalues_array = self.params_ / std_errors
        tvalues_series = pd.Series(tvalues_array, index=param_names)
        pvalues_array = 2 * (1 - stats.norm.cdf(np.abs(tvalues_array)))
        pvalues_series = pd.Series(pvalues_array, index=param_names)

        # Create Hansen J-test result
        from panelbox.gmm.results import TestResult

        hansen_j = TestResult(
            name="Hansen J-test",
            statistic=self.j_stat_,
            pvalue=self.j_pvalue_,
            df=self.overid_df,
            distribution="chi2",
            null_hypothesis="Overidentifying restrictions are valid",
        )

        # Create dummy tests for AR tests (not applicable to CUE-GMM)
        ar1_test = TestResult(
            name="AR(1) test",
            statistic=np.nan,
            pvalue=np.nan,
            distribution="normal",
            null_hypothesis="No first-order autocorrelation",
            conclusion="N/A",
        )
        ar2_test = TestResult(
            name="AR(2) test",
            statistic=np.nan,
            pvalue=np.nan,
            distribution="normal",
            null_hypothesis="No second-order autocorrelation",
            conclusion="N/A",
        )

        # Sargan test (same as J for CUE)
        sargan = TestResult(
            name="Sargan test",
            statistic=self.j_stat_,
            pvalue=self.j_pvalue_,
            df=self.overid_df,
            distribution="chi2",
            null_hypothesis="Overidentifying restrictions are valid",
        )

        # Get number of groups (entities) from data
        if isinstance(self.data.index, pd.MultiIndex):
            n_groups = self.data.index.get_level_values(0).nunique()
        else:
            n_groups = self.n

        # Create results object
        results = GMMResults(
            params=params_series,
            std_errors=std_errors_series,
            tvalues=tvalues_series,
            pvalues=pvalues_series,
            nobs=self.n,
            n_groups=n_groups,
            n_instruments=self.n_instruments,
            n_params=self.k,
            hansen_j=hansen_j,
            sargan=sargan,
            ar1_test=ar1_test,
            ar2_test=ar2_test,
            vcov=self.vcov_,
            converged=self.converged_,
            two_step=False,  # CUE is continuous updating
            windmeijer_corrected=False,
            model_type="CUE-GMM",
            transformation="levels",
        )

        return results

    def j_statistic(self) -> Dict[str, Union[float, bool]]:
        """
        Hansen J-test for overidentification.

        Returns
        -------
        test_result : dict
            Dictionary with:
            - 'statistic': J-statistic value
            - 'pvalue': p-value
            - 'df': degrees of freedom
            - 'reject': whether to reject H0 at 5% level

        Notes
        -----
        H0: Moment conditions are valid (model is correctly specified)
        H1: At least one moment condition is invalid

        J = N × Q(β̂) ~ χ²(L - K)

        where L = number of instruments, K = number of parameters.

        A low p-value suggests model misspecification or invalid instruments.

        Examples
        --------
        >>> results = model.fit()
        >>> j_test = results.j_statistic()
        >>> if j_test['reject']:
        ...     print("Warning: Model rejected by J-test")
        """
        if self.params_ is None:
            raise RuntimeError("Must call fit() before j_statistic()")

        reject = self.j_pvalue_ < 0.05

        return {
            "statistic": self.j_stat_,
            "pvalue": self.j_pvalue_,
            "df": self.overid_df,
            "reject": reject,
            "interpretation": self._interpret_j_test(reject),
        }

    def _interpret_j_test(self, reject: bool) -> str:
        """Generate interpretation of J-test."""
        if reject:
            return (
                f"Reject overidentification test (p={self.j_pvalue_:.4f}). "
                "Model may be misspecified or instruments invalid."
            )
        else:
            return (
                f"Do not reject overidentification test (p={self.j_pvalue_:.4f}). "
                "Moment conditions appear valid."
            )

    def compare_with_two_step(self, two_step_result: GMMResults) -> pd.DataFrame:
        """
        Compare CUE-GMM efficiency with two-step GMM.

        Parameters
        ----------
        two_step_result : GMMResults
            Results from two-step GMM estimation

        Returns
        -------
        comparison : pd.DataFrame
            Comparison of estimates and standard errors

        Notes
        -----
        CUE-GMM often has lower variance than two-step in finite samples.
        This method compares:
        - Parameter estimates
        - Standard errors
        - Relative efficiency (ratio of variances)

        Examples
        --------
        >>> cue_results = cue_model.fit()
        >>> ts_results = ts_model.fit()
        >>> comparison = cue_results.compare_with_two_step(ts_results)
        >>> print(comparison)
        """
        if self.params_ is None:
            raise RuntimeError("Must call fit() before compare_with_two_step()")

        # Extract estimates and SEs
        cue_params = self.params_
        cue_se = np.sqrt(np.diag(self.vcov_))

        ts_params = two_step_result.params.values
        ts_se = two_step_result.std_errors.values

        # Compute relative efficiency (ratio of variances)
        efficiency = (ts_se / cue_se) ** 2

        # Build comparison DataFrame
        param_names = ["const"] + self.exog_vars
        comparison = pd.DataFrame(
            {
                "CUE Coef": cue_params,
                "TS Coef": ts_params,
                "Diff": cue_params - ts_params,
                "CUE SE": cue_se,
                "TS SE": ts_se,
                "Efficiency Ratio": efficiency,
            },
            index=param_names,
        )

        return comparison

    def conf_int(self, alpha: float = 0.05, method: str = "normal") -> pd.DataFrame:
        """
        Compute confidence intervals for parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (e.g., 0.05 for 95% CI)
        method : str, default='normal'
            Method for computing confidence intervals:
            - 'normal': Normal approximation using SEs
            - 'percentile': Percentile method (requires bootstrap)
            - 'basic': Basic bootstrap CI (requires bootstrap)

        Returns
        -------
        ci : pd.DataFrame
            Confidence intervals with columns ['lower', 'upper']

        Notes
        -----
        Normal approximation:
            CI = β̂ ± z_(α/2) × SE(β̂)

        Percentile bootstrap:
            CI = [q_(α/2), q_(1-α/2)] of bootstrap distribution

        Basic bootstrap:
            CI = [2β̂ - q_(1-α/2), 2β̂ - q_(α/2)]

        Examples
        --------
        >>> results = model.fit()
        >>> ci = results.conf_int(alpha=0.05)
        >>> print(ci)
        """
        if self.params_ is None:
            raise RuntimeError("Must call fit() before conf_int()")

        param_names = ["const"] + self.exog_vars
        ci = pd.DataFrame(index=param_names, columns=["lower", "upper"])

        if method == "normal":
            # Normal approximation
            z_critical = stats.norm.ppf(1 - alpha / 2)
            se = self.bse_
            ci["lower"] = self.params_ - z_critical * se
            ci["upper"] = self.params_ + z_critical * se

        elif method == "percentile":
            # Percentile bootstrap
            if self.bootstrap_params_ is None:
                raise ValueError(
                    "Percentile method requires bootstrap. "
                    "Use se_type='bootstrap' when initializing model."
                )
            lower_pct = 100 * (alpha / 2)
            upper_pct = 100 * (1 - alpha / 2)
            ci["lower"] = np.percentile(self.bootstrap_params_, lower_pct, axis=0)
            ci["upper"] = np.percentile(self.bootstrap_params_, upper_pct, axis=0)

        elif method == "basic":
            # Basic bootstrap
            if self.bootstrap_params_ is None:
                raise ValueError(
                    "Basic method requires bootstrap. "
                    "Use se_type='bootstrap' when initializing model."
                )
            lower_pct = 100 * (alpha / 2)
            upper_pct = 100 * (1 - alpha / 2)
            boot_lower = np.percentile(self.bootstrap_params_, lower_pct, axis=0)
            boot_upper = np.percentile(self.bootstrap_params_, upper_pct, axis=0)
            ci["lower"] = 2 * self.params_ - boot_upper
            ci["upper"] = 2 * self.params_ - boot_lower

        else:
            raise ValueError(f"method must be 'normal', 'percentile', or 'basic', got {method}")

        return ci

    @property
    def bse(self) -> pd.Series:
        """
        Bootstrap standard errors.

        Returns
        -------
        bse : pd.Series
            Standard errors (bootstrap or analytical)

        Notes
        -----
        If se_type='bootstrap', returns bootstrap SEs.
        If se_type='analytical', returns analytical SEs.

        Examples
        --------
        >>> results = model.fit()
        >>> print(results.bse)
        """
        if self.bse_ is None:
            raise RuntimeError("Must call fit() before accessing bse")

        param_names = ["const"] + self.exog_vars
        return pd.Series(self.bse_, index=param_names)

    def __repr__(self) -> str:
        """String representation."""
        if self.params_ is None:
            status = "not fitted"
        else:
            status = f"fitted, J={self.j_stat_:.4f} (p={self.j_pvalue_:.4f})"

        return (
            f"ContinuousUpdatedGMM("
            f"dep_var='{self.dep_var}', "
            f"n_exog={len(self.exog_vars)}, "
            f"n_instruments={self.n_instruments}, "
            f"weighting='{self.weighting}', "
            f"status={status})"
        )
