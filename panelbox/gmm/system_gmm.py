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
    **Model and System:**

    System GMM stacks two sets of equations:

    1. **Difference equations** (Arellano-Bond):

        Δy_{it} = γ Δy_{i,t-1} + β' Δx_{it} + Δε_{it}

       Instruments: Lags of levels (y_{i,t-2}, y_{i,t-3}, ...)

    2. **Level equations** (additional moment conditions):

        y_{it} = γ y_{i,t-1} + β' x_{it} + η_i + ε_{it}

       Instruments: Lags of differences (Δy_{i,t-1}, Δy_{i,t-2}, ...)

    **Critical Additional Assumption:**

        E[Δy_{i,1} · η_i] = 0  (stationarity of initial conditions)

    This requires:
    - The process generating y_i started long before the first observation
    - Initial deviations from long-run mean are uncorrelated with fixed effects
    - Violated if panel starts at firm entry, policy change, etc.

    **When to Use System GMM:**

    Prefer System over Difference GMM when:

    - **Persistent series**: AR coefficient > 0.8 (levels weak instruments)
    - **Small T**: Few time periods (efficiency matters)
    - **Stationary process**: Initial conditions assumption plausible
    - **Need precision**: Want smaller standard errors

    Use Difference GMM when:

    - **Initial conditions suspect**: Panel starts at event time
    - **Non-stationary**: Unit root processes
    - **Conservative approach**: Fewer assumptions

    **Diagnostic Tests:**

    Same as Difference GMM, plus:

    - **Difference-in-Hansen test**: Tests validity of level instruments
      - p > 0.10: Fail to reject (level instruments valid)
      - p < 0.10: Reject (use Difference GMM instead)

    **Efficiency Gains:**

    System GMM typically reduces standard errors by 20-50% compared to
    Difference GMM when:

    - Series are persistent (ρ > 0.8)
    - Additional moment conditions are valid
    - Sample size is moderate (N > 50)

    **Instrument Control:**

    Use `level_instruments={'max_lags': k}` to control depth:

    - max_lags=1: Use only Δy_{t-1} (most conservative, recommended)
    - max_lags=2: Use Δy_{t-1}, Δy_{t-2}
    - Deeper lags rarely improve efficiency

    References
    ----------
    .. [1] Blundell, R., & Bond, S. (1998). "Initial Conditions and Moment
           Restrictions in Dynamic Panel Data Models." Journal of Econometrics,
           87(1), 115-143.
    .. [2] Roodman, D. (2009). "How to do xtabond2: An Introduction to
           Difference and System GMM in Stata." The Stata Journal, 9(1), 86-136.
    .. [3] Bond, S. R., Hoeffler, A., & Temple, J. (2001). "GMM Estimation of
           Empirical Growth Models." Economics Papers 2001-W21, Economics Group,
           Nuffield College, University of Oxford.

    See Also
    --------
    DifferenceGMM : Difference GMM (Arellano-Bond) estimator
    FixedEffects : Fixed Effects estimator (for static panels)
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
        gmm_max_lag: Optional[int] = None,
        iv_max_lag: int = 0,
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
            gmm_max_lag=gmm_max_lag,
            iv_max_lag=iv_max_lag,
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
        y_diff, X_diff, y_level, X_level, ids, times, valid_diff = self._transform_data_system()

        # Step 3: Generate instruments (difference + level)
        # Returns separate GMM and IV instruments for proper stacking
        Z_diff_gmm, Z_diff_iv, Z_level_gmm = self._generate_instruments_system()

        # Step 4: Stack equations
        n_diff = len(y_diff)
        y_stacked = np.vstack([y_diff, y_level])

        # Add constant column to X (0 for diff, 1 for level)
        const_col = np.vstack(
            [
                np.zeros((n_diff, 1)),
                np.ones((len(y_level), 1)),
            ]
        )
        X_stacked = np.vstack([X_diff, X_level])
        X_stacked = np.hstack([X_stacked, const_col])

        # Trim instrument matrices by valid_diff mask.
        # Z matrices have n_full rows (from self.data), but y/X are already
        # trimmed to n_valid rows by _transform_data_system.
        Z_dgmm = Z_diff_gmm.Z[valid_diff]
        Z_div = Z_diff_iv.Z[valid_diff]
        Z_lgmm = Z_level_gmm.Z[valid_diff]

        # Filter invalid columns and apply nan_to_num
        Z_dgmm_clean = self._filter_invalid_columns(Z_dgmm, min_coverage=0.10)
        Z_lgmm_clean = self._filter_invalid_columns(Z_lgmm, min_coverage=0.10)
        Z_div_clean = self._filter_invalid_columns(Z_div, min_coverage=0.10)

        # Build IV instruments as shared columns for level equation.
        # For exogenous vars, level equation uses levels (not differences).
        Z_liv_cols = []
        for var in self.exog_vars:
            Z_lev_iv = self.instrument_builder.create_iv_style_instruments(
                var=var, min_lag=0, max_lag=self.iv_max_lag, equation="level"
            )
            Z_liv_cols.append(
                self._filter_invalid_columns(Z_lev_iv.Z[valid_diff], min_coverage=0.10)
            )

        # Build stacked instrument matrix following pydynpd/xtabond2 convention:
        # [Z_gmm_diff    0         | Δx1  Δx2 |   0  ]  <- diff rows
        # [   0       Z_gmm_level  |  x1   x2 | _cons]  <- level rows
        n_gmm_diff = Z_dgmm_clean.shape[1]
        n_gmm_level = Z_lgmm_clean.shape[1]
        n_iv = Z_div_clean.shape[1]
        n_const = 1  # _cons column as instrument
        n_instruments_total = n_gmm_diff + n_gmm_level + n_iv + n_const

        Z_stacked_raw = np.zeros((2 * n_diff, n_instruments_total))

        # GMM diff block (block-diagonal: only in diff rows)
        col = 0
        Z_stacked_raw[:n_diff, col : col + n_gmm_diff] = Z_dgmm_clean
        col += n_gmm_diff

        # GMM level block (block-diagonal: only in level rows)
        Z_stacked_raw[n_diff:, col : col + n_gmm_level] = Z_lgmm_clean
        col += n_gmm_level

        # IV shared columns (Δx in diff rows, x in level rows)
        Z_stacked_raw[:n_diff, col : col + n_iv] = Z_div_clean
        if Z_liv_cols:
            Z_liv_stacked = np.hstack(Z_liv_cols) if len(Z_liv_cols) > 1 else Z_liv_cols[0]
            # Ensure same number of columns as diff IV
            Z_stacked_raw[n_diff:, col : col + n_iv] = Z_liv_stacked[:, :n_iv]
        col += n_iv

        # _cons as instrument (0 in diff rows, 1 in level rows)
        Z_stacked_raw[n_diff:, col] = 1.0

        # Build stacked ids for per-individual computation
        ids_stacked = np.concatenate([ids, ids])

        # Clean instrument matrix before estimation
        valid_mask = self._get_valid_mask_system(y_stacked, X_stacked, Z_stacked_raw)

        if not valid_mask.any():
            n_params = X_stacked.shape[1]
            raise ValueError(
                f"System GMM is under-identified: {n_instruments_total} instruments "
                f"< {n_params + 1} required (parameters + 1). "
                f"Reduce the number of parameters (e.g., set time_dummies=False) "
                f"or add more instruments."
            )

        y_stacked_clean = y_stacked[valid_mask]
        X_stacked_clean = X_stacked[valid_mask]
        Z_stacked_clean = Z_stacked_raw[valid_mask]
        ids_stacked_clean = ids_stacked[valid_mask]

        # Remove instrument columns with remaining NaNs
        valid_instrument_cols = ~np.isnan(Z_stacked_clean).any(axis=0)
        if not valid_instrument_cols.any():
            raise ValueError("No valid instrument columns in System GMM. Check data quality.")
        Z_stacked_clean = Z_stacked_clean[:, valid_instrument_cols]

        # Build H blocks for the stacked system (per individual)
        # For system GMM: H = [H_diff (tridiagonal), 0; 0, I_level (identity)]
        unique_ids = np.unique(ids_stacked_clean)
        H_blocks = {}
        for uid in unique_ids:
            mask = ids_stacked_clean == uid
            T_total = int(np.sum(mask))
            # Each individual has T_diff rows in diff block + T_level rows in level block
            # After trimming (Bug 7 fix), T_diff == T_level, so T_total = 2 * T_diff
            T_half = T_total // 2
            if T_half > 0:
                H_diff = self.estimator.build_H_matrix(T_half, "fd")
                H_level = np.eye(T_total - T_half)
                H_i = np.zeros((T_total, T_total))
                H_i[:T_half, :T_half] = H_diff
                H_i[T_half:, T_half:] = H_level
            else:
                H_i = np.eye(T_total)
            H_blocks[uid] = H_i

        # Step 5: Estimate GMM on stacked system (pass ids for per-individual weights)
        if self.gmm_type == "one_step":
            beta, W, residuals_clean = self.estimator.one_step(
                y_stacked_clean,
                X_stacked_clean,
                Z_stacked_clean,
                ids=ids_stacked_clean,
                H_blocks=H_blocks,
            )
            vcov = self.estimator.compute_one_step_robust_vcov(
                Z_stacked_clean,
                residuals_clean,
                ids_stacked_clean,
            )
            converged = True
        elif self.gmm_type == "two_step":
            beta, vcov, W, residuals_clean = self.estimator.two_step(
                y_stacked_clean,
                X_stacked_clean,
                Z_stacked_clean,
                ids=ids_stacked_clean,
                H_blocks=H_blocks,
                robust=self.robust,
            )
            converged = True
        else:  # iterative
            beta, vcov, W, converged = self.estimator.iterative(
                y_stacked_clean,
                X_stacked_clean,
                Z_stacked_clean,
                ids=ids_stacked_clean,
                H_blocks=H_blocks,
            )
            residuals_clean = y_stacked_clean - X_stacked_clean @ beta

        # Ensure beta is 1D for pandas Series
        beta = beta.flatten()

        # Step 6: Compute standard errors and statistics
        std_errors = np.sqrt(np.diag(vcov))
        tvalues = beta / std_errors
        from scipy import stats as scipy_stats

        pvalues = 2 * (1 - scipy_stats.norm.cdf(np.abs(tvalues)))

        # Step 7: Get variable names (add _cons for the constant)
        var_names = self._get_variable_names()
        var_names.append("_cons")

        # Step 8: Compute specification tests
        n_params = len(beta)

        # Hansen J-test using per-individual moments (correct formula)
        hansen = self.tester.hansen_j_test(
            residuals_clean,
            Z_stacked_clean,
            W,
            n_params,
            zs=self.estimator.zs,
            W2_inv=self.estimator.W2_inv,
            N=self.estimator.N,
            n_instruments=Z_stacked_clean.shape[1],
        )

        # Sargan test
        sargan = self.tester.sargan_test(residuals_clean, Z_stacked_clean, n_params)

        # AR tests (on difference residuals only - first half of stacked system)
        # Find which clean rows correspond to the diff block
        diff_mask_in_clean = valid_mask[: 2 * n_diff][:n_diff]  # valid mask for diff part
        residuals_full_diff = np.full(n_diff, np.nan)
        # Map clean residuals back to diff block
        clean_idx = 0
        for i in range(len(valid_mask)):
            if valid_mask[i]:
                if i < n_diff:
                    residuals_full_diff[i] = residuals_clean.flatten()[clean_idx]
                clean_idx += 1

        valid_diff_resid = ~np.isnan(residuals_full_diff)
        ar1 = self.tester.arellano_bond_ar_test(
            residuals_full_diff[valid_diff_resid], ids[valid_diff_resid], order=1
        )
        ar2 = self.tester.arellano_bond_ar_test(
            residuals_full_diff[valid_diff_resid], ids[valid_diff_resid], order=2
        )

        # Difference-in-Hansen test for level instruments
        try:
            diff_hansen = self._compute_diff_hansen(
                residuals_clean, Z_diff_gmm, Z_level_gmm, W, n_params, valid_diff
            )
        except (ValueError, np.linalg.LinAlgError, IndexError):
            diff_hansen = None

        # Bug 3 fix: nobs counts only diff equation observations (not stacked total)
        nobs = n_diff

        # Step 9: Create results object
        self.results = GMMResults(
            params=pd.Series(beta, index=var_names),
            std_errors=pd.Series(std_errors, index=var_names),
            tvalues=pd.Series(tvalues, index=var_names),
            pvalues=pd.Series(pvalues, index=var_names),
            nobs=nobs,
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
            residuals=residuals_clean,
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
        valid_diff : np.ndarray
            Boolean mask of valid rows (used to trim instrument matrices)
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

        # Extract level data (ALL rows initially)
        y_level_all = df[self.dep_var].values.reshape(-1, 1)
        X_level_all = np.column_stack([df[var].values for var in regressors])

        # Bug 7 fix: Trim level data to match diff equation valid rows
        # The diff equation drops first period (from .diff()) and possibly more
        # (from lagged variable NaN). Level equation must use the same (i,t) pairs.
        valid_diff = ~np.isnan(y_diff.flatten())
        if X_diff.ndim > 1:
            valid_diff &= ~np.isnan(X_diff).any(axis=1)

        y_diff = y_diff[valid_diff]
        X_diff = X_diff[valid_diff]
        y_level = y_level_all[valid_diff]
        X_level = X_level_all[valid_diff]
        ids = ids[valid_diff]
        times = times[valid_diff]

        return y_diff, X_diff, y_level, X_level, ids, times, valid_diff

    def _generate_instruments_system(self) -> tuple:
        """
        Generate instruments for System GMM.

        Following pydynpd/xtabond2 convention:
        - GMM instruments: block-diagonal (separate for diff and level equations)
        - IV instruments: shared columns (Δx in diff rows, x in level rows)
        - _cons: instrument column (0 in diff rows, 1 in level rows)

        Returns
        -------
        Z_diff_gmm : InstrumentSet
            GMM-style instruments for difference equations
        Z_diff_iv : InstrumentSet
            IV-style instruments for difference equations
        Z_level_gmm : InstrumentSet
            GMM-style instruments for level equations
        """
        # Generate diff instruments in separate groups (GMM vs IV)
        diff_gmm_sets = []
        diff_iv_sets = []

        # GMM-style instruments for lagged dependent (diff equation)
        for lag in self.lags:
            max_lag = self.gmm_max_lag if self.gmm_max_lag is not None else 99
            Z_lag = self.instrument_builder.create_gmm_style_instruments(
                var=self.dep_var,
                min_lag=lag + 1,
                max_lag=max_lag,
                equation="diff",
                collapse=self.collapse,
            )
            diff_gmm_sets.append(Z_lag)

        # GMM-style instruments for predetermined variables (diff equation)
        for var in self.predetermined_vars:
            Z_pred = self.instrument_builder.create_gmm_style_instruments(
                var=var, min_lag=2, max_lag=99, equation="diff", collapse=self.collapse
            )
            diff_gmm_sets.append(Z_pred)

        # GMM-style instruments for endogenous variables (diff equation)
        for var in self.endogenous_vars:
            Z_endog = self.instrument_builder.create_gmm_style_instruments(
                var=var, min_lag=3, max_lag=99, equation="diff", collapse=self.collapse
            )
            diff_gmm_sets.append(Z_endog)

        # IV-style instruments for exogenous variables (diff equation)
        for var in self.exog_vars:
            Z_exog = self.instrument_builder.create_iv_style_instruments(
                var=var, min_lag=0, max_lag=self.iv_max_lag, equation="diff"
            )
            diff_iv_sets.append(Z_exog)

        # Combine diff GMM instruments
        n_obs = len(self.data)
        if diff_gmm_sets:
            Z_diff_gmm = self.instrument_builder.combine_instruments(*diff_gmm_sets)
        else:
            Z_diff_gmm = InstrumentSet(
                Z=np.empty((n_obs, 0)),
                variable_names=[],
                instrument_names=[],
                equation="diff",
                style="gmm",
                collapsed=False,
            )

        if diff_iv_sets:
            Z_diff_iv = self.instrument_builder.combine_instruments(*diff_iv_sets)
        else:
            Z_diff_iv = InstrumentSet(
                Z=np.empty((n_obs, 0)),
                variable_names=[],
                instrument_names=[],
                equation="diff",
                style="iv",
                collapsed=False,
            )

        # Create differenced variables for level equation instruments
        df = self.data.sort_values([self.id_var, self.time_var]).copy()
        for lag in self.lags:
            lag_name = f"{self.dep_var}_L{lag}"
            if lag_name in df.columns:
                df[f"{lag_name}_diff"] = df.groupby(self.id_var)[lag_name].diff()
                self.data[f"{lag_name}_diff"] = df[f"{lag_name}_diff"]
        for var in self.predetermined_vars + self.endogenous_vars:
            if var in df.columns:
                df[f"{var}_diff"] = df.groupby(self.id_var)[var].diff()
                self.data[f"{var}_diff"] = df[f"{var}_diff"]

        # Recreate InstrumentBuilder with updated data
        self.instrument_builder = InstrumentBuilder(self.data, self.id_var, self.time_var)

        # Generate level GMM instruments only (no IV — those are shared columns)
        level_gmm_sets = []

        # For lagged dependent: use Δy_{t-1} as instrument (Blundell-Bond)
        # min_lag=0 of y_L1 with equation="level" computes:
        #   y_L1(t) - y_L1(t-1) = y_{t-1} - y_{t-2} = Δy_{t-1}
        # max_lag=0 ensures exactly 1 instrument per lagged dependent
        for lag in self.lags:
            lag_name = f"{self.dep_var}_L{lag}"
            Z_level_lag = self.instrument_builder.create_gmm_style_instruments(
                var=lag_name,
                min_lag=0,
                max_lag=0,
                equation="level",
                collapse=self.collapse,
            )
            level_gmm_sets.append(Z_level_lag)

        # For predetermined/endogenous: use lagged differences
        for var in self.predetermined_vars + self.endogenous_vars:
            max_lags_level = self.level_instruments.get("max_lags", 1)
            Z_level_var = self.instrument_builder.create_gmm_style_instruments(
                var=var,
                min_lag=1,
                max_lag=max_lags_level,
                equation="level",
                collapse=self.collapse,
            )
            level_gmm_sets.append(Z_level_var)

        # Combine level GMM instruments
        if level_gmm_sets:
            Z_level_gmm = self.instrument_builder.combine_instruments(*level_gmm_sets)
        else:
            Z_level_gmm = InstrumentSet(
                Z=np.empty((n_obs, 0)),
                variable_names=[],
                instrument_names=[],
                equation="level",
                style="gmm",
                collapsed=False,
            )

        return Z_diff_gmm, Z_diff_iv, Z_level_gmm

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

        Z_filtered = Z[:, valid_cols]

        # Replace NaN with 0 for computation.
        # GMM-style instruments are naturally sparse (NaN for unavailable lags).
        # This matches how Difference GMM handles sparse instruments.
        Z_filtered = np.nan_to_num(Z_filtered, nan=0.0)

        return Z_filtered

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

        return np.asarray(y_valid & X_valid & Z_valid)

    def _compute_diff_hansen(
        self,
        residuals: np.ndarray,
        Z_diff: InstrumentSet,
        Z_level: InstrumentSet,
        W_full: np.ndarray,
        n_params: int,
        valid_diff: Optional[np.ndarray] = None,
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
        valid_diff : np.ndarray, optional
            Boolean mask to trim instrument rows to match data

        Returns
        -------
        TestResult
            Difference-in-Hansen test result
        """
        # Trim instruments to match residuals if valid_diff provided
        if valid_diff is not None:
            Z_diff_Z = Z_diff.Z[valid_diff]
            Z_level_Z = Z_level.Z[valid_diff]
        else:
            Z_diff_Z = Z_diff.Z
            Z_level_Z = Z_level.Z

        n_obs = Z_diff_Z.shape[0]

        # Filter invalid columns
        Z_diff_clean = self._filter_invalid_columns(Z_diff_Z, min_coverage=0.10)
        Z_level_clean = self._filter_invalid_columns(Z_level_Z, min_coverage=0.10)

        # Full system instruments (block diagonal)
        n_instr_total = Z_diff_clean.shape[1] + Z_level_clean.shape[1]
        Z_full = np.zeros((2 * n_obs, n_instr_total))
        Z_full[:n_obs, : Z_diff_clean.shape[1]] = Z_diff_clean
        Z_full[n_obs:, Z_diff_clean.shape[1] :] = Z_level_clean

        # Subset system (difference only)
        Z_subset = np.zeros((2 * n_obs, Z_diff_clean.shape[1]))
        Z_subset[:n_obs, :] = Z_diff_clean
        Z_subset[n_obs:, :] = Z_diff_clean

        # Compute weight matrix for subset
        # (simplified - in practice should re-estimate)
        n_diff_instr = Z_diff_clean.shape[1]
        W_subset = W_full[:n_diff_instr, :n_diff_instr]

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
