"""
System GMM Estimator
====================

Blundell-Bond (1998) System GMM estimator for dynamic panel data models.

Classes
-------
SystemGMM : Blundell-Bond System GMM estimator

References
----------
.. [1] Blundell, R., & Bond, S. (1998). "Initial Conditions and Moment
       Restrictions in Dynamic Panel Data Models." Journal of Econometrics,
       87(1), 115-143.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from panelbox.gmm.difference_gmm import DifferenceGMM
from panelbox.gmm.instruments import InstrumentBuilder, InstrumentSet
from panelbox.gmm.results import GMMResults


class SystemGMM(DifferenceGMM):
    """
    Blundell-Bond (1998) System GMM estimator.

    Combines difference and level equations in a stacked system:
    - Difference equations (instruments: lags of levels)
    - Level equations (instruments: lags of differences)

    Advantages over Difference GMM:
    - More efficient when series are persistent
    - Better precision for coefficient estimates
    - Additional moment conditions

    Requires assumption:
    E[Δy_{i,t-1} · η_i] = 0  (initial conditions)

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    dep_var : str
        Name of dependent variable
    lags : Union[int, List[int]]
        Lags of dependent variable to include
    id_var : str
        Name of cross-sectional identifier (default: 'id')
    time_var : str
        Name of time variable (default: 'year')
    exog_vars : List[str], optional
        List of strictly exogenous variables
    endogenous_vars : List[str], optional
        List of endogenous variables
    predetermined_vars : List[str], optional
        List of predetermined variables
    time_dummies : bool
        Include time dummies (default: True)
    collapse : bool
        Collapse instruments (default: False)
    two_step : bool
        Use two-step GMM (default: True)
    robust : bool
        Use robust variance with Windmeijer correction (default: True)
    gmm_type : str
        GMM type: 'one_step', 'two_step', 'iterative' (default: 'two_step')
    level_instruments : Dict, optional
        Configuration for level equation instruments
        Example: {'max_lags': 1} uses L.D.y as instrument

    Attributes
    ----------
    level_instruments : Dict
        Configuration for level equation instruments

    Examples
    --------
    **When to use System GMM:**

    System GMM is preferred over Difference GMM when:
    - Variables are highly persistent (AR coefficient near 1)
    - Lagged levels are weak instruments for differences
    - You want more efficient estimates (smaller standard errors)

    **Basic System GMM with production function:**

    >>> import pandas as pd
    >>> from panelbox.gmm import SystemGMM
    >>>
    >>> # Load production data
    >>> data = pd.read_csv('production.csv')
    >>>
    >>> # Estimate System GMM
    >>> model = SystemGMM(
    ...     data=data,
    ...     dep_var='output',
    ...     lags=1,                        # Include output_{t-1}
    ...     id_var='firm_id',
    ...     time_var='year',
    ...     exog_vars=['capital', 'labor'],
    ...     collapse=True,                 # Always recommended
    ...     two_step=True,
    ...     robust=True,
    ...     level_instruments={'max_lags': 1}  # Use Δy_{t-1} for level equation
    ... )
    >>>
    >>> results = model.fit()
    >>> print(results.summary())
    >>>
    >>> # Check if more efficient than Difference GMM
    >>> print(f"Standard error: {results.std_errors['L1.output']:.4f}")

    **Comparing Difference vs System GMM:**

    >>> from panelbox.gmm import DifferenceGMM, SystemGMM
    >>>
    >>> # Estimate both
    >>> diff_gmm = DifferenceGMM(
    ...     data=data,
    ...     dep_var='y',
    ...     lags=1,
    ...     exog_vars=['x1', 'x2'],
    ...     collapse=True,
    ...     two_step=True
    ... )
    >>> diff_results = diff_gmm.fit()
    >>>
    >>> sys_gmm = SystemGMM(
    ...     data=data,
    ...     dep_var='y',
    ...     lags=1,
    ...     exog_vars=['x1', 'x2'],
    ...     collapse=True,
    ...     two_step=True,
    ...     level_instruments={'max_lags': 1}
    ... )
    >>> sys_results = sys_gmm.fit()
    >>>
    >>> # Compare efficiency
    >>> coef_name = 'L1.y'
    >>> diff_se = diff_results.std_errors[coef_name]
    >>> sys_se = sys_results.std_errors[coef_name]
    >>> efficiency_gain = (diff_se - sys_se) / diff_se * 100
    >>> print(f"System GMM SE is {efficiency_gain:.1f}% smaller")
    >>>
    >>> # Check if both are valid
    >>> if sys_results.ar2_test.pvalue > 0.10 and sys_results.hansen_j.pvalue > 0.10:
    ...     print("System GMM preferred (more efficient and valid)")

    **With custom level instruments:**

    >>> # Control instrument depth for level equation
    >>> model = SystemGMM(
    ...     data=data,
    ...     dep_var='n',
    ...     lags=1,
    ...     exog_vars=['w', 'k'],
    ...     collapse=True,
    ...     level_instruments={'max_lags': 1}
    ... )
    >>> results = model.fit()

    Notes
    -----
    System combines:

    Difference equation:
        Δy_{it} = γ Δy_{i,t-1} + β' Δx_{it} + Δε_{it}
        Instruments: lags of levels (y_{i,t-2}, y_{i,t-3}, ...)

    Level equation:
        y_{it} = γ y_{i,t-1} + β' x_{it} + η_i + ε_{it}
        Instruments: lags of differences (Δy_{i,t-1}, Δy_{i,t-2}, ...)

    Critical assumption:
        E[Δy_{i,1} · η_i] = 0
        Violated if initial conditions are correlated with fixed effects

    References
    ----------
    Blundell, R., & Bond, S. (1998). Journal of Econometrics, 87(1), 115-143.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dep_var: str,
        lags: Union[int, List[int]],
        id_var: str = "id",
        time_var: str = "year",
        exog_vars: Optional[List[str]] = None,
        endogenous_vars: Optional[List[str]] = None,
        predetermined_vars: Optional[List[str]] = None,
        time_dummies: bool = True,
        collapse: bool = False,
        two_step: bool = True,
        robust: bool = True,
        gmm_type: str = "two_step",
        level_instruments: Optional[Dict] = None,
    ):
        """Initialize System GMM model."""
        # Initialize parent Difference GMM
        super().__init__(
            data=data,
            dep_var=dep_var,
            lags=lags,
            id_var=id_var,
            time_var=time_var,
            exog_vars=exog_vars,
            endogenous_vars=endogenous_vars,
            predetermined_vars=predetermined_vars,
            time_dummies=time_dummies,
            collapse=collapse,
            two_step=two_step,
            robust=robust,
            gmm_type=gmm_type,
        )

        # Level instruments configuration
        self.level_instruments = level_instruments or {"max_lags": 1}

    def fit(self) -> GMMResults:
        """
        Estimate the System GMM model.

        Returns
        -------
        GMMResults
            Estimation results

        Notes
        -----
        Estimation procedure:
        1. Create difference equations (as in Difference GMM)
        2. Create level equations
        3. Stack equations and instruments
        4. Estimate using stacked system
        5. Compute specification tests including Diff-in-Hansen
        """
        # Step 1 & 2: Transform data (both differences and levels)
        y_diff, X_diff, y_level, X_level, ids, times = self._transform_data_system()

        # Step 3: Generate instruments (difference + level)
        # Note: _generate_instruments_system will recreate InstrumentBuilder internally
        Z_diff, Z_level = self._generate_instruments_system()

        # Step 4: Stack equations
        y_stacked = np.vstack([y_diff, y_level])
        X_stacked = np.vstack([X_diff, X_level])
        Z_stacked_raw = self._stack_instruments(Z_diff, Z_level)

        # Clean instrument matrix before estimation
        # Remove observations and columns with NaNs
        valid_mask = self._get_valid_mask_system(y_stacked, X_stacked, Z_stacked_raw)
        y_stacked_clean = y_stacked[valid_mask]
        X_stacked_clean = X_stacked[valid_mask]
        Z_stacked_clean = Z_stacked_raw[valid_mask]

        # Remove instrument columns with remaining NaNs
        valid_instrument_cols = ~np.isnan(Z_stacked_clean).any(axis=0)
        if not valid_instrument_cols.any():
            raise ValueError("No valid instrument columns in System GMM. Check data quality.")
        Z_stacked_clean = Z_stacked_clean[:, valid_instrument_cols]

        # For tests later, keep track of the full stacked residuals
        residuals_full = np.full_like(y_stacked, np.nan)

        # Repeat ids and times for stacked system
        ids_stacked = np.concatenate([ids, ids])
        times_stacked = np.concatenate([times, times])

        # Step 5: Estimate GMM on stacked system (using cleaned data)
        if self.gmm_type == "one_step":
            beta, W, residuals_clean = self.estimator.one_step(
                y_stacked_clean, X_stacked_clean, Z_stacked_clean
            )
            vcov = self._compute_one_step_vcov(X_stacked_clean, Z_stacked_clean, residuals_clean, W)
            converged = True
        elif self.gmm_type == "two_step":
            beta, vcov, W, residuals_clean = self.estimator.two_step(
                y_stacked_clean, X_stacked_clean, Z_stacked_clean, robust=self.robust
            )
            converged = True
        else:  # iterative
            beta, vcov, W, converged = self.estimator.iterative(
                y_stacked_clean, X_stacked_clean, Z_stacked_clean
            )
            residuals_clean = y_stacked_clean - X_stacked_clean @ beta

        # Fill residuals in full array
        if residuals_full.ndim > 1:
            residuals_full[valid_mask] = residuals_clean.reshape(-1, 1)
        else:
            residuals_full[valid_mask] = residuals_clean.flatten()

        # Ensure beta is 1D for pandas Series
        beta = beta.flatten()

        # Step 6: Compute standard errors and statistics
        std_errors = np.sqrt(np.diag(vcov))
        tvalues = beta / std_errors
        from scipy import stats as scipy_stats

        pvalues = 2 * (1 - scipy_stats.norm.cdf(np.abs(tvalues)))

        # Step 7: Get variable names
        var_names = self._get_variable_names()

        # Step 8: Compute specification tests
        n_params = len(beta)

        # Hansen J-test on full system (use cleaned data)
        hansen = self.tester.hansen_j_test(residuals_clean, Z_stacked_clean, W, n_params)

        # Sargan test
        sargan = self.tester.sargan_test(residuals_clean, Z_stacked_clean, n_params)

        # AR tests (on difference residuals only)
        n_diff = len(y_diff)
        residuals_diff_only = residuals_full[:n_diff]
        ids_diff_only = ids_stacked[:n_diff]  # Use stacked ids, first half

        valid_mask_diff = ~np.isnan(residuals_diff_only.flatten())
        resid_diff_clean = residuals_diff_only.flatten()[valid_mask_diff]
        ids_diff_clean = ids_diff_only[valid_mask_diff]

        ar1 = self.tester.arellano_bond_ar_test(resid_diff_clean, ids_diff_clean, order=1)
        ar2 = self.tester.arellano_bond_ar_test(resid_diff_clean, ids_diff_clean, order=2)

        # Difference-in-Hansen test for level instruments
        # Note: Disabled when instrument columns are filtered due to dimension mismatches
        # This is a known limitation when dealing with sparse instrument coverage
        try:
            diff_hansen = self._compute_diff_hansen(residuals_full, Z_diff, Z_level, W, n_params)
        except (ValueError, np.linalg.LinAlgError):
            # If dimensions don't match (due to column filtering), skip test
            diff_hansen = None

        # Step 9: Create results object
        valid_mask_results = ~np.isnan(residuals_full.flatten())
        self.results = GMMResults(
            params=pd.Series(beta, index=var_names),
            std_errors=pd.Series(std_errors, index=var_names),
            tvalues=pd.Series(tvalues, index=var_names),
            pvalues=pd.Series(pvalues, index=var_names),
            nobs=int(np.sum(valid_mask)),
            n_groups=self.instrument_builder.n_groups,
            n_instruments=Z_stacked_clean.shape[1],
            n_params=n_params,
            hansen_j=hansen,
            sargan=sargan,
            ar1_test=ar1,
            ar2_test=ar2,
            diff_hansen=diff_hansen,
            vcov=vcov,
            weight_matrix=W,
            converged=converged,
            two_step=self.two_step,
            windmeijer_corrected=self.robust and self.two_step,
            model_type="system",
            transformation="fd",
            residuals=residuals_full,
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
                f"  4. Check data for excessive missing values\n"
                f"  5. Consider using DifferenceGMM (more robust for weak instruments)\n\n"
                f"See examples/gmm/unbalanced_panel_guide.py for detailed guidance.",
                UserWarning,
            )

        return self.results

    def _transform_data_system(self) -> tuple:
        """
        Transform data for System GMM (both differences and levels).

        Returns
        -------
        y_diff : np.ndarray
            Differenced dependent variable
        X_diff : np.ndarray
            Differenced regressors
        y_level : np.ndarray
            Level dependent variable
        X_level : np.ndarray
            Level regressors
        ids : np.ndarray
            ID variable
        times : np.ndarray
            Time variable
        """
        # Get difference transformation from parent
        y_diff, X_diff, ids, times = super()._transform_data()

        # Also need levels
        df = self.data.sort_values([self.id_var, self.time_var])

        # Create lagged dependent variable for levels
        for lag in self.lags:
            lag_name = f"{self.dep_var}_L{lag}"
            df[lag_name] = df.groupby(self.id_var)[self.dep_var].shift(lag)

        # Build regressor list (same as difference)
        regressors = []
        for lag in self.lags:
            regressors.append(f"{self.dep_var}_L{lag}")
        regressors.extend(self.exog_vars)
        regressors.extend(self.endogenous_vars)
        regressors.extend(self.predetermined_vars)

        # Add time dummies if requested
        if self.time_dummies:
            time_dummies = pd.get_dummies(df[self.time_var], prefix="year", drop_first=True)
            for col in time_dummies.columns:
                df[col] = time_dummies[col]
                if col not in regressors:
                    regressors.append(col)

        # Extract level data
        y_level = df[self.dep_var].values.reshape(-1, 1)
        X_level = np.column_stack([df[var].values for var in regressors])

        return y_diff, X_diff, y_level, X_level, ids, times

    def _generate_instruments_system(self) -> tuple:
        """
        Generate instruments for System GMM.

        Returns
        -------
        Z_diff : np.ndarray
            Instruments for difference equations
        Z_level : np.ndarray
            Instruments for level equations
        """
        # Difference equation instruments (same as Difference GMM)
        Z_diff = self._generate_instruments()

        # FIRST: Create ALL differenced variables and add to data
        df = self.data.sort_values([self.id_var, self.time_var]).copy()

        # Create differences of lagged dependent variable
        for lag in self.lags:
            lag_name = f"{self.dep_var}_L{lag}"
            if lag_name in df.columns:
                df[f"{lag_name}_diff"] = df.groupby(self.id_var)[lag_name].diff()
                self.data[f"{lag_name}_diff"] = df[f"{lag_name}_diff"]

        # Create differences of predetermined/endogenous variables
        for var in self.predetermined_vars + self.endogenous_vars:
            if var in df.columns:
                df[f"{var}_diff"] = df.groupby(self.id_var)[var].diff()
                self.data[f"{var}_diff"] = df[f"{var}_diff"]

        # SECOND: Recreate InstrumentBuilder with updated data
        self.instrument_builder = InstrumentBuilder(self.data, self.id_var, self.time_var)

        # THIRD: Generate level instruments using the differenced variables
        instrument_sets_level = []

        # For lagged dependent variable in levels, use differences as instruments
        for lag in self.lags:
            lag_name = f"{self.dep_var}_L{lag}"

            # Use lagged differences as instruments for levels
            max_lags_level = self.level_instruments.get("max_lags", 1)
            Z_level_lag = self.instrument_builder.create_gmm_style_instruments(
                var=f"{lag_name}_diff",
                min_lag=0,  # Can use contemporaneous difference
                max_lag=max_lags_level,
                equation="level",
                collapse=self.collapse,
            )
            instrument_sets_level.append(Z_level_lag)

        # For exogenous variables in levels, use themselves
        for var in self.exog_vars:
            Z_level_exog = self.instrument_builder.create_iv_style_instruments(
                var=var, min_lag=0, max_lag=0, equation="level"
            )
            instrument_sets_level.append(Z_level_exog)

        # For predetermined/endogenous in levels, use lagged differences
        for var in self.predetermined_vars + self.endogenous_vars:
            # Variable differences already created above
            max_lags_level = self.level_instruments.get("max_lags", 1)
            Z_level_var = self.instrument_builder.create_gmm_style_instruments(
                var=f"{var}_diff",
                min_lag=1,
                max_lag=max_lags_level,
                equation="level",
                collapse=self.collapse,
            )
            instrument_sets_level.append(Z_level_var)

        # Combine level instruments
        if instrument_sets_level:
            Z_level = self.instrument_builder.combine_instruments(*instrument_sets_level)
        else:
            # No level-specific instruments, use empty matrix
            Z_level = InstrumentSet(
                Z=np.empty((len(self.data), 0)),
                variable_names=[],
                instrument_names=[],
                equation="level",
                style="mixed",
                collapsed=False,
            )

        return Z_diff, Z_level

    def _stack_instruments(self, Z_diff: InstrumentSet, Z_level: InstrumentSet) -> np.ndarray:
        """
        Stack instruments for System GMM.

        Creates block-diagonal matrix:
        [ Z_diff     0     ]
        [   0     Z_level  ]

        Parameters
        ----------
        Z_diff : InstrumentSet
            Difference equation instruments
        Z_level : InstrumentSet
            Level equation instruments

        Returns
        -------
        np.ndarray
            Stacked instrument matrix
        """
        n_obs = Z_diff.n_obs

        # Filter out invalid instrument columns (all NaN or insufficient coverage)
        # For difference instruments
        Z_diff_clean = self._filter_invalid_columns(Z_diff.Z, min_coverage=0.10)

        # For level instruments
        Z_level_clean = self._filter_invalid_columns(Z_level.Z, min_coverage=0.10)

        # Create block diagonal matrix
        n_instruments_total = Z_diff_clean.shape[1] + Z_level_clean.shape[1]

        Z_stacked = np.zeros((2 * n_obs, n_instruments_total))

        # Fill difference block
        Z_stacked[:n_obs, : Z_diff_clean.shape[1]] = Z_diff_clean

        # Fill level block
        Z_stacked[n_obs:, Z_diff_clean.shape[1] :] = Z_level_clean

        return Z_stacked

    def _filter_invalid_columns(self, Z: np.ndarray, min_coverage: float = 0.10) -> np.ndarray:
        """
        Filter out instrument columns with insufficient coverage.

        Parameters
        ----------
        Z : np.ndarray
            Instrument matrix
        min_coverage : float
            Minimum fraction of non-NaN values required (default: 0.10 = 10%)

        Returns
        -------
        np.ndarray
            Filtered instrument matrix with only valid columns
        """
        if Z.shape[1] == 0:
            return Z

        # Count non-NaN values per column
        n_valid_per_col = (~np.isnan(Z)).sum(axis=0)
        n_obs = Z.shape[0]

        # Calculate coverage per column
        coverage = n_valid_per_col / n_obs

        # Keep columns with sufficient coverage
        valid_cols = coverage >= min_coverage

        # If no columns are valid, return at least one column (all zeros)
        # This prevents dimension errors, though estimation may fail later
        if not valid_cols.any():
            import warnings

            warnings.warn("No valid instrument columns found. System GMM may fail.")
            return np.zeros((n_obs, 1))

        return Z[:, valid_cols]

    def _get_valid_mask_system(
        self, y: np.ndarray, X: np.ndarray, Z: np.ndarray, min_instruments: Optional[int] = None
    ) -> np.ndarray:
        """
        Get mask of observations with sufficient valid data for System GMM.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Regressors
        Z : np.ndarray
            Instruments
        min_instruments : int, optional
            Minimum number of valid instruments required

        Returns
        -------
        np.ndarray
            Boolean mask of valid observations
        """
        y_valid = ~np.isnan(y).any(axis=1) if y.ndim > 1 else ~np.isnan(y)
        X_valid = ~np.isnan(X).any(axis=1)

        # For instruments, count how many are valid per observation
        Z_notnan = ~np.isnan(Z)
        n_valid_instruments = Z_notnan.sum(axis=1)

        # Determine minimum required instruments
        if min_instruments is None:
            k = X.shape[1] if X.ndim > 1 else 1
            min_instruments = k + 1

        Z_valid = n_valid_instruments >= min_instruments

        return y_valid & X_valid & Z_valid

    def _compute_diff_hansen(
        self,
        residuals: np.ndarray,
        Z_diff: InstrumentSet,
        Z_level: InstrumentSet,
        W_full: np.ndarray,
        n_params: int,
    ):
        """
        Compute Difference-in-Hansen test for level instruments.

        Tests the validity of level equation instruments by comparing
        Hansen J statistics with and without level instruments.

        Parameters
        ----------
        residuals : np.ndarray
            Residuals from full system
        Z_diff : InstrumentSet
            Difference instruments
        Z_level : InstrumentSet
            Level instruments
        W_full : np.ndarray
            Weight matrix from full system
        n_params : int
            Number of parameters

        Returns
        -------
        TestResult
            Difference-in-Hansen test result
        """
        # Full system instruments
        Z_full = self._stack_instruments(Z_diff, Z_level)

        # Subset system (difference only)
        n_obs = Z_diff.n_obs
        Z_subset = np.zeros((2 * n_obs, Z_diff.n_instruments))
        Z_subset[:n_obs, :] = Z_diff.Z
        # Level equations get same instruments as difference (for subset comparison)
        Z_subset[n_obs:, :] = Z_diff.Z

        # Compute weight matrix for subset
        # (simplified - in practice should re-estimate)
        W_subset = W_full[: Z_diff.n_instruments, : Z_diff.n_instruments]

        # Compute Difference-in-Hansen test
        diff_hansen = self.tester.difference_in_hansen(
            residuals=residuals,
            Z_full=Z_full,
            Z_subset=Z_subset,
            W_full=W_full,
            W_subset=W_subset,
            n_params=n_params,
            subset_name="level instruments",
        )

        return diff_hansen

    def summary(self) -> str:
        """
        Print model summary.

        Returns
        -------
        str
            Summary string
        """
        if self.results is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")

        return self.results.summary(title="System GMM (Blundell-Bond)")

    def __repr__(self) -> str:
        """Representation of the model."""
        status = "fitted" if self.results is not None else "not fitted"
        return f"SystemGMM(dep_var='{self.dep_var}', lags={self.lags}, " f"status='{status}')"
