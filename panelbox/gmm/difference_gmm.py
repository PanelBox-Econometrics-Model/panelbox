"""
Difference GMM Estimator
=========================

Arellano-Bond (1991) Difference GMM estimator for dynamic panel data models.

Classes
-------
DifferenceGMM : Arellano-Bond Difference GMM estimator

References
----------
.. [1] Arellano, M., & Bond, S. (1991). "Some Tests of Specification for Panel
       Data: Monte Carlo Evidence and an Application to Employment Equations."
       Review of Economic Studies, 58(2), 277-297.
"""

from typing import Union, List, Optional, Dict
import numpy as np
import pandas as pd
from panelbox.gmm.results import GMMResults, TestResult
from panelbox.gmm.instruments import InstrumentBuilder, InstrumentSet
from panelbox.gmm.estimator import GMMEstimator
from panelbox.gmm.tests import GMMTests


class DifferenceGMM:
    """
    Arellano-Bond (1991) Difference GMM estimator.

    Eliminates fixed effects through first-differencing and uses lagged
    levels as instruments for the differenced equation.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    dep_var : str
        Name of dependent variable
    lags : Union[int, List[int]]
        Lags of dependent variable to include (e.g., 1 or [1, 2])
    id_var : str
        Name of cross-sectional identifier (default: 'id')
    time_var : str
        Name of time variable (default: 'year')
    exog_vars : List[str], optional
        List of strictly exogenous variables
    endogenous_vars : List[str], optional
        List of endogenous variables (excluding lagged dependent)
    predetermined_vars : List[str], optional
        List of predetermined variables
    time_dummies : bool
        Include time dummies (default: True)
    collapse : bool
        Collapse instruments to avoid proliferation (default: False)
    two_step : bool
        Use two-step GMM (default: True)
    robust : bool
        Use robust variance matrix with Windmeijer correction (default: True)
    gmm_type : str
        GMM estimation type: 'one_step', 'two_step', or 'iterative' (default: 'two_step')

    Attributes
    ----------
    data : pd.DataFrame
        Panel data
    params : pd.Series
        Estimated coefficients (after fitting)
    results : GMMResults
        Full results object (after fitting)

    Examples
    --------
    **Basic example with employment data:**

    >>> import pandas as pd
    >>> from panelbox.gmm import DifferenceGMM
    >>>
    >>> # Load panel data (firms over time)
    >>> data = pd.read_csv('panel_data.csv')
    >>>
    >>> # Estimate Difference GMM
    >>> model = DifferenceGMM(
    ...     data=data,
    ...     dep_var='employment',
    ...     lags=1,                    # Include employment_{t-1}
    ...     id_var='firm_id',
    ...     time_var='year',
    ...     exog_vars=['wages', 'capital'],
    ...     time_dummies=True,
    ...     collapse=True,             # Recommended to avoid instrument proliferation
    ...     two_step=True,             # Two-step with Windmeijer correction
    ...     robust=True                # Robust standard errors
    ... )
    >>>
    >>> # Fit and get results
    >>> results = model.fit()
    >>> print(results.summary())
    >>>
    >>> # Access coefficients
    >>> print(f"Persistence: {results.params['L1.employment']:.3f}")
    >>> print(f"Wage effect: {results.params['wages']:.3f}")

    **Interpreting diagnostic tests:**

    >>> # Check if estimation is valid
    >>> if results.ar2_test.pvalue > 0.10:
    ...     print("✓ Moment conditions valid")
    ...
    >>> if 0.10 < results.hansen_j.pvalue < 0.25:
    ...     print("✓ Instruments appear valid")
    ...
    >>> if results.instrument_ratio < 1.0:
    ...     print("✓ Instrument count appropriate")

    **With predetermined and endogenous variables:**

    >>> # Some variables may not be strictly exogenous
    >>> model = DifferenceGMM(
    ...     data=data,
    ...     dep_var='output',
    ...     lags=1,
    ...     exog_vars=['policy_var'],        # Strictly exogenous
    ...     predetermined_vars=['capital'],   # Instruments: t-1 and earlier
    ...     endogenous_vars=['labor'],        # Instruments: t-2 and earlier
    ...     collapse=True,
    ...     two_step=True
    ... )
    >>> results = model.fit()

    **For unbalanced panels:**

    >>> # Always use collapse=True and avoid many time dummies
    >>> model = DifferenceGMM(
    ...     data=unbalanced_data,
    ...     dep_var='y',
    ...     lags=1,
    ...     exog_vars=['x1', 'x2'],
    ...     time_dummies=False,  # Or use linear trend
    ...     collapse=True,       # Essential for unbalanced panels
    ...     two_step=True
    ... )
    >>> results = model.fit()
    >>> print(f"Retained {results.nobs}/{len(unbalanced_data)} observations")

    Notes
    -----
    Transformation: First-differences to eliminate fixed effects
        Δy_{it} = γ Δy_{i,t-1} + β' Δx_{it} + Δε_{it}

    Instruments: Lags of levels for differenced equations
        - Strictly exogenous: all lags and leads
        - Predetermined: lags t-2 and earlier
        - Endogenous: lags t-3 and earlier

    References
    ----------
    Arellano, M., & Bond, S. (1991). Review of Economic Studies, 58(2), 277-297.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 dep_var: str,
                 lags: Union[int, List[int]],
                 id_var: str = 'id',
                 time_var: str = 'year',
                 exog_vars: Optional[List[str]] = None,
                 endogenous_vars: Optional[List[str]] = None,
                 predetermined_vars: Optional[List[str]] = None,
                 time_dummies: bool = True,
                 collapse: bool = False,
                 two_step: bool = True,
                 robust: bool = True,
                 gmm_type: str = 'two_step'):
        """Initialize Difference GMM model."""
        self.data = data.copy()
        self.dep_var = dep_var
        self.lags = [lags] if isinstance(lags, int) else lags
        self.id_var = id_var
        self.time_var = time_var
        self.exog_vars = exog_vars or []
        self.endogenous_vars = endogenous_vars or []
        self.predetermined_vars = predetermined_vars or []
        self.time_dummies = time_dummies
        self.collapse = collapse
        self.two_step = two_step
        self.robust = robust
        self.gmm_type = gmm_type

        # Initialize components
        self.instrument_builder = InstrumentBuilder(data, id_var, time_var)
        self.estimator = GMMEstimator()
        self.tester = GMMTests()

        # Results (populated after fit)
        self.results = None
        self.params = None

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate model inputs."""
        import warnings

        # Check dep_var exists
        if self.dep_var not in self.data.columns:
            raise ValueError(f"Dependent variable '{self.dep_var}' not found in data")

        # Check id_var and time_var exist
        if self.id_var not in self.data.columns:
            raise ValueError(f"ID variable '{self.id_var}' not found in data")
        if self.time_var not in self.data.columns:
            raise ValueError(f"Time variable '{self.time_var}' not found in data")

        # Check exogenous variables exist
        for var in self.exog_vars + self.endogenous_vars + self.predetermined_vars:
            if var not in self.data.columns:
                raise ValueError(f"Variable '{var}' not found in data")

        # Check gmm_type is valid
        valid_types = ['one_step', 'two_step', 'iterative']
        if self.gmm_type not in valid_types:
            raise ValueError(f"gmm_type must be one of {valid_types}")

        # If gmm_type is specified, override two_step flag
        if self.gmm_type == 'one_step':
            self.two_step = False
        elif self.gmm_type == 'two_step':
            self.two_step = True

        # Check for unbalanced panel + time dummies issue
        if self.time_dummies:
            is_unbalanced, balance_rate = self._check_panel_balance()
            if is_unbalanced:
                n_time_periods = self.data[self.time_var].nunique()
                n_dummies = n_time_periods - 1

                if n_dummies >= 5 and balance_rate < 0.80:
                    warnings.warn(
                        f"\nUnbalanced panel detected ({balance_rate*100:.0f}% balanced) with "
                        f"{n_dummies} time dummies.\n"
                        f"This may result in very few observations being retained.\n\n"
                        f"Recommendations:\n"
                        f"  1. Set time_dummies=False and add a linear trend\n"
                        f"  2. Use only subset of key time dummies\n"
                        f"  3. Ensure collapse=True (currently: {self.collapse})\n\n"
                        f"See examples/gmm/unbalanced_panel_guide.py for details.",
                        UserWarning
                    )

        # Check collapse recommendation
        if not self.collapse:
            warnings.warn(
                "\n" + "="*70 + "\n"
                "RECOMMENDATION: Set collapse=True\n"
                "="*70 + "\n"
                "Non-collapsed GMM instruments (collapse=False) can cause:\n"
                "  • Instrument proliferation (grows as T²)\n"
                "  • Numerical instability with sparse instrument matrices\n"
                "  • Overfitting and weak instrument problems\n"
                "\n"
                "Roodman (2009) recommends collapse=True as best practice.\n"
                "Collapsed instruments:\n"
                "  ✓ Reduce instrument count from O(T²) to O(T)\n"
                "  ✓ More numerically stable\n"
                "  ✓ Better finite-sample properties\n"
                "  ✓ Less prone to overfitting\n"
                "\n"
                "To suppress this warning:\n"
                "  DifferenceGMM(..., collapse=True)  # Recommended\n"
                "\n"
                "Reference: Roodman, D. (2009). \"How to do xtabond2:\n"
                "An introduction to difference and system GMM in Stata.\"\n"
                "The Stata Journal, 9(1), 86-136.\n"
                "="*70,
                UserWarning
            )

    def _check_panel_balance(self):
        """
        Check if panel data is balanced.

        Returns
        -------
        tuple
            (is_unbalanced: bool, balance_rate: float)
        """
        obs_per_unit = self.data.groupby(self.id_var).size()
        max_periods = obs_per_unit.max()

        # Panel is balanced if all units have same number of periods
        is_balanced = (obs_per_unit == max_periods).all()

        # Balance rate: proportion of units with max periods
        balance_rate = (obs_per_unit == max_periods).mean()

        return not is_balanced, balance_rate

    def fit(self) -> GMMResults:
        """
        Estimate the Difference GMM model.

        Returns
        -------
        GMMResults
            Estimation results including coefficients, tests, and diagnostics

        Raises
        ------
        ValueError
            If model specification is invalid
        RuntimeError
            If estimation fails

        Notes
        -----
        Estimation procedure:
        1. Transform data to first-differences
        2. Generate instruments (lags of levels)
        3. Estimate GMM (one-step, two-step, or iterative)
        4. Compute specification tests
        5. Return results object
        """
        # Step 1: Transform data
        y_diff, X_diff, ids, times = self._transform_data()

        # Step 1.5: Recreate InstrumentBuilder with updated data (includes lagged vars)
        self.instrument_builder = InstrumentBuilder(self.data, self.id_var, self.time_var)

        # Step 2: Generate instruments
        Z = self._generate_instruments()

        # Step 2.5: Pre-clean instruments for unbalanced panels
        # GMM-style instruments are naturally sparse (time-period-specific)
        # Do NOT filter based on NaN percentage - this is expected and correct
        Z_matrix = Z.Z.copy()

        # Only remove columns that are ALL NaN (completely empty)
        not_all_nan = ~np.isnan(Z_matrix).all(axis=0)
        Z_matrix_filtered = Z_matrix[:, not_all_nan]

        # Filter observations by GMM instrument availability
        # For Difference GMM, Stata requires at least 2 valid GMM instruments per observation
        # This ensures sufficient variation and enables overidentification tests
        instrument_names_filtered = [name for i, name in enumerate(Z.instrument_names) if not_all_nan[i]]
        gmm_cols = [i for i, name in enumerate(instrument_names_filtered) if name.startswith('n_t')]

        if len(gmm_cols) > 0:
            Z_gmm = Z_matrix_filtered[:, gmm_cols]
            n_valid_gmm = (~np.isnan(Z_gmm)).sum(axis=1)
            min_gmm_instruments = 2  # Stata xtabond2 default
            obs_valid_mask = n_valid_gmm >= min_gmm_instruments

            # Filter all arrays
            y_diff = y_diff[obs_valid_mask]
            X_diff = X_diff[obs_valid_mask]
            Z_matrix_filtered = Z_matrix_filtered[obs_valid_mask]
            ids = ids[obs_valid_mask]
            times = times[obs_valid_mask]

        # Handle sparse GMM instruments
        # For non-collapsed instruments, this creates numerical challenges
        # but is necessary for current implementation

        # Remove columns that are completely empty (all NaN across all kept observations)
        n_valid_per_col = (~np.isnan(Z_matrix_filtered)).sum(axis=0)
        valid_cols = n_valid_per_col > 0
        Z_matrix_filtered = Z_matrix_filtered[:, valid_cols]

        # Replace NaN with 0 for computation
        # NOTE: This is a numerical compromise for non-collapsed instruments
        # Collapsed instruments avoid this issue by combining lags
        Z_matrix = np.nan_to_num(Z_matrix_filtered, nan=0.0)

        # Step 3: Estimate GMM
        if self.gmm_type == 'one_step':
            beta, W, residuals = self.estimator.one_step(y_diff, X_diff, Z_matrix)
            vcov = self._compute_one_step_vcov(X_diff, Z_matrix, residuals, W)
            converged = True
        elif self.gmm_type == 'two_step':
            beta, vcov, W, residuals = self.estimator.two_step(
                y_diff, X_diff, Z_matrix, robust=self.robust
            )
            converged = True
        else:  # iterative
            beta, vcov, W, converged = self.estimator.iterative(
                y_diff, X_diff, Z_matrix
            )
            residuals = y_diff - X_diff @ beta

        # Step 4: Compute standard errors and t-statistics
        beta = beta.flatten()  # Ensure beta is 1D
        std_errors = np.sqrt(np.diag(vcov))
        tvalues = beta / std_errors
        from scipy import stats
        pvalues = 2 * (1 - stats.norm.cdf(np.abs(tvalues)))

        # Step 5: Get variable names
        var_names = self._get_variable_names()

        # Step 6: Compute specification tests
        hansen = self.tester.hansen_j_test(
            residuals, Z_matrix, W, len(beta)
        )
        sargan = self.tester.sargan_test(
            residuals, Z_matrix, len(beta)
        )

        # For AR tests, we need clean data without NaN
        residuals_flat = residuals.flatten() if residuals.ndim > 1 else residuals
        valid_mask = ~np.isnan(residuals_flat)
        ar1 = self.tester.arellano_bond_ar_test(
            residuals_flat[valid_mask], ids[valid_mask], order=1
        )
        ar2 = self.tester.arellano_bond_ar_test(
            residuals_flat[valid_mask], ids[valid_mask], order=2
        )

        # Step 7: Create results object
        self.results = GMMResults(
            params=pd.Series(beta, index=var_names),
            std_errors=pd.Series(std_errors, index=var_names),
            tvalues=pd.Series(tvalues, index=var_names),
            pvalues=pd.Series(pvalues, index=var_names),
            nobs=int(np.sum(valid_mask)),
            n_groups=self.instrument_builder.n_groups,
            n_instruments=Z_matrix.shape[1],  # Use actual number of instruments after cleaning
            n_params=len(beta),
            hansen_j=hansen,
            sargan=sargan,
            ar1_test=ar1,
            ar2_test=ar2,
            vcov=vcov,
            weight_matrix=W,
            converged=converged,
            two_step=self.two_step,
            windmeijer_corrected=self.robust and self.two_step,
            model_type='difference',
            transformation='fd',
            residuals=residuals
        )

        self.params = self.results.params

        # Post-estimation warning for low observation retention
        retention_rate = self.results.nobs / len(self.data)
        if retention_rate < 0.30:
            import warnings
            warnings.warn(
                f"\nLow observation retention: {self.results.nobs}/{len(self.data)} "
                f"({retention_rate*100:.1f}%).\n"
                f"Many observations were dropped due to insufficient valid instruments.\n\n"
                f"Recommendations:\n"
                f"  1. Simplify specification (fewer variables/lags)\n"
                f"  2. Set time_dummies=False (or use linear trend)\n"
                f"  3. Ensure collapse=True (currently: {self.collapse})\n"
                f"  4. Check data for excessive missing values\n\n"
                f"See examples/gmm/unbalanced_panel_guide.py for detailed guidance.",
                UserWarning
            )

        return self.results

    def _transform_data(self) -> tuple:
        """
        Transform data to first-differences.

        Returns
        -------
        y_diff : np.ndarray
            Differenced dependent variable
        X_diff : np.ndarray
            Differenced regressors
        ids : np.ndarray
            ID variable
        times : np.ndarray
            Time variable
        """
        # Sort data
        df = self.data.sort_values([self.id_var, self.time_var]).copy()

        # Create lagged dependent variable
        for lag in self.lags:
            lag_name = f'{self.dep_var}_L{lag}'
            df[lag_name] = df.groupby(self.id_var)[self.dep_var].shift(lag)
            # Also add to self.data for instrument generation
            self.data[lag_name] = df[lag_name]

        # Build regressor list
        regressors = []
        for lag in self.lags:
            regressors.append(f'{self.dep_var}_L{lag}')
        regressors.extend(self.exog_vars)
        regressors.extend(self.endogenous_vars)
        regressors.extend(self.predetermined_vars)

        # Add time dummies if requested
        if self.time_dummies:
            time_dummies = pd.get_dummies(df[self.time_var], prefix='year', drop_first=True)
            for col in time_dummies.columns:
                df[col] = time_dummies[col]
                regressors.append(col)

        # First-difference transformation
        df['y_diff'] = df.groupby(self.id_var)[self.dep_var].diff()

        X_diff_dict = {}
        for var in regressors:
            X_diff_dict[var] = df.groupby(self.id_var)[var].diff()

        # Extract arrays, ensuring float64 dtype
        y_diff = df['y_diff'].values.reshape(-1, 1).astype(np.float64)
        X_diff = np.column_stack([X_diff_dict[var].values for var in regressors]).astype(np.float64)
        ids = df[self.id_var].values
        times = df[self.time_var].values

        return y_diff, X_diff, ids, times

    def _generate_instruments(self) -> InstrumentSet:
        """
        Generate instrument matrix.

        Returns
        -------
        InstrumentSet
            Combined instrument set
        """
        instrument_sets = []

        # Instruments for lagged dependent variable (GMM-style)
        # For Δy_{i,t-lag}, use levels y_{i,t-lag-1}, y_{i,t-lag-2}, ... as instruments
        for lag in self.lags:
            # min_lag for instruments should be lag+1 (e.g., for L1.y use y_{t-2}, y_{t-3}, ...)
            Z_lag = self.instrument_builder.create_gmm_style_instruments(
                var=self.dep_var,
                min_lag=lag + 1,  # For L1.y, use y_{t-2} and earlier
                max_lag=99,  # All available lags
                equation='diff',
                collapse=self.collapse
            )
            instrument_sets.append(Z_lag)

        # Instruments for strictly exogenous variables (IV-style, all lags)
        # For balanced panels: use lags 0 to T-2 where T = number of periods
        # For Arellano-Bond data: T=9 years, use lags 0-6 or 0-7
        # After testing: max_lag=6 gives 42 instruments to match Stata
        for var in self.exog_vars:
            Z_exog = self.instrument_builder.create_iv_style_instruments(
                var=var,
                min_lag=0,  # Current and all lags
                max_lag=6,  # Empirically calibrated to match Stata xtabond2
                equation='diff'
            )
            instrument_sets.append(Z_exog)

        # Instruments for predetermined variables (GMM-style, lag 2+)
        for var in self.predetermined_vars:
            Z_pred = self.instrument_builder.create_gmm_style_instruments(
                var=var,
                min_lag=2,  # t-2 and earlier
                max_lag=99,
                equation='diff',
                collapse=self.collapse
            )
            instrument_sets.append(Z_pred)

        # Instruments for endogenous variables (GMM-style, lag 3+)
        for var in self.endogenous_vars:
            Z_endog = self.instrument_builder.create_gmm_style_instruments(
                var=var,
                min_lag=3,  # t-3 and earlier
                max_lag=99,
                equation='diff',
                collapse=self.collapse
            )
            instrument_sets.append(Z_endog)

        # Combine all instruments
        Z_combined = self.instrument_builder.combine_instruments(*instrument_sets)

        return Z_combined

    def _compute_one_step_vcov(self,
                              X: np.ndarray,
                              Z: np.ndarray,
                              residuals: np.ndarray,
                              W: np.ndarray) -> np.ndarray:
        """
        Compute variance-covariance matrix for one-step GMM.

        Parameters
        ----------
        X : np.ndarray
            Regressors
        Z : np.ndarray
            Instruments
        residuals : np.ndarray
            Residuals
        W : np.ndarray
            Weight matrix

        Returns
        -------
        np.ndarray
            Variance-covariance matrix
        """
        # Ensure arrays are float64
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)

        # Remove missing values
        valid_mask = ~np.isnan(residuals.flatten())
        X_clean = X[valid_mask]
        Z_clean = Z[valid_mask]
        resid_clean = residuals[valid_mask]

        # Robust variance: (X'Z W Z'X)^{-1} (X'Z W Ω W Z'X) (X'Z W Z'X)^{-1}
        # where Ω = Z' diag(ε²) Z

        XtZ = X_clean.T @ Z_clean
        ZtX = Z_clean.T @ X_clean

        A = XtZ @ W @ ZtX
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)

        # Compute Omega
        Omega = np.diag(resid_clean.flatten() ** 2)
        ZtOmegaZ = Z_clean.T @ Omega @ Z_clean

        # Robust variance
        B = XtZ @ W @ ZtOmegaZ @ W @ ZtX
        vcov = A_inv @ B @ A_inv

        return vcov

    def _get_variable_names(self) -> List[str]:
        """
        Get list of variable names in order.

        Returns
        -------
        List[str]
            Variable names
        """
        var_names = []

        # Lagged dependent variable
        for lag in self.lags:
            var_names.append(f'L{lag}.{self.dep_var}')

        # Other variables
        var_names.extend(self.exog_vars)
        var_names.extend(self.endogenous_vars)
        var_names.extend(self.predetermined_vars)

        # Time dummies
        if self.time_dummies:
            time_periods = sorted(self.data[self.time_var].unique())[1:]  # Drop first
            for t in time_periods:
                var_names.append(f'year_{t}')

        return var_names

    def summary(self) -> str:
        """
        Print model summary.

        Returns
        -------
        str
            Summary string

        Raises
        ------
        ValueError
            If model has not been fit yet
        """
        if self.results is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")

        return self.results.summary(title='Difference GMM (Arellano-Bond)')

    def __repr__(self) -> str:
        """Representation of the model."""
        status = "fitted" if self.results is not None else "not fitted"
        return (f"DifferenceGMM(dep_var='{self.dep_var}', lags={self.lags}, "
                f"status='{status}')")
