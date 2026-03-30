"""
LSDVC (Least Squares Dummy Variable Corrected) estimator for dynamic panel data.

Corrects the Nickell (1981) bias in Fixed Effects estimation with lagged
dependent variables using the analytical bias formulas of Kiviet (1995).
Bootstrap parametric inference following Bruno (2005).

References
----------
.. [1] Kiviet, J. F. (1995). "On Bias, Inconsistency, and Efficiency of
       Various Estimators in Dynamic Panel Data Models."
       Journal of Econometrics, 68(1), 53-78.
.. [2] Bruno, G. S. F. (2005). "Approximating the Bias of the LSDV
       Estimator for Dynamic Unbalanced Panel Data Models."
       Economics Letters, 87(3), 361-366.
.. [3] Bun, M. J. G., & Kiviet, J. F. (2003). "On the Diminishing Returns
       of Higher-Order Terms in Asymptotic Expansions of Bias."
       Economics Letters, 79(2), 145-152.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.core.base_model import PanelModel
from panelbox.core.results import PanelResults

logger = logging.getLogger(__name__)


class LSDVCResults(PanelResults):
    """
    Results container for LSDVC estimation.

    Inherits from PanelResults and adds LSDVC-specific attributes
    for bias correction diagnostics and bootstrap inference.

    Additional Attributes
    ---------------------
    bias_correction : pd.Series
        Bias correction vector applied to each parameter.
    lsdv_params : pd.Series
        Parameters from LSDV (before bias correction).
    initial_estimator : str
        Method used for initial consistent estimate ('ah', 'ab', 'bb').
    initial_rho : float
        Initial consistent estimate of rho.
    n_iterations : int
        Number of iterations until convergence.
    converged : bool
        Whether the iterated bias correction converged.
    bias_order : int
        Order of the Kiviet bias approximation used.
    sigma2 : float
        Error variance estimate from LSDV.
    bootstrap_params : np.ndarray or None
        Bootstrap parameter distribution (B x K) if bootstrap was run.
    bootstrap_ci : tuple or None
        (ci_lower, ci_upper) bootstrap percentile confidence intervals.
    """

    def __init__(
        self,
        params: pd.Series,
        std_errors: pd.Series,
        cov_params: pd.DataFrame,
        resid: np.ndarray,
        fittedvalues: np.ndarray,
        model_info: dict[str, Any],
        data_info: dict[str, Any],
        *,
        bias_correction: pd.Series,
        lsdv_params: pd.Series,
        initial_estimator: str,
        initial_rho: float,
        n_iterations: int,
        converged: bool,
        bias_order: int,
        sigma2: float,
        bootstrap_params: np.ndarray | None = None,
        bootstrap_ci: tuple | None = None,
    ):
        super().__init__(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=resid,
            fittedvalues=fittedvalues,
            model_info=model_info,
            data_info=data_info,
        )

        self.bias_correction = bias_correction
        self.lsdv_params = lsdv_params
        self.initial_estimator = initial_estimator
        self.initial_rho = initial_rho
        self.n_iterations = n_iterations
        self.converged = converged
        self.bias_order = bias_order
        self.sigma2 = sigma2
        self.bootstrap_params = bootstrap_params
        self.bootstrap_ci = bootstrap_ci

    def summary(self, title: str | None = None) -> str:
        """
        Generate formatted summary of LSDVC estimation results.

        Includes coefficient table with bootstrap SEs, bias correction
        diagnostics, and comparison of LSDV vs LSDVC estimates.
        """
        lines = []

        # Header
        lines.append("=" * 78)
        if title is None:
            title = "LSDVC Estimation Results"
        lines.append(title.center(78))
        lines.append("=" * 78)

        # Model information
        lines.append(f"Formula: {self.formula}")
        lines.append(f"Model:   {self.model_type}")
        lines.append(f"Initial Estimator:         {'':>4}{self.initial_estimator.upper():>6}")
        lines.append(f"Bias Correction Order:     {'':>4}{self.bias_order:>6}")
        lines.append(
            f"Convergence:               {'':>4}{'Yes' if self.converged else 'No':>6}"
            f" ({self.n_iterations} iterations)"
        )
        lines.append("-" * 78)

        # Sample information
        lines.append(f"No. Observations:          {self.nobs:>10,}")
        lines.append(f"No. Entities:              {self.n_entities:>10,}")
        if self.n_periods is not None:
            lines.append(f"No. Time Periods:          {self.n_periods:>10,}")
        lines.append(f"Sigma^2:                   {self.sigma2:>10.4f}")

        se_type = "bootstrap" if self.bootstrap_params is not None else "none"
        lines.append(f"Standard Errors:           {se_type:>10}")
        if self.bootstrap_params is not None:
            lines.append(f"Bootstrap Replications:    {self.bootstrap_params.shape[0]:>10,}")

        lines.append("=" * 78)

        # Coefficient table
        lines.append(
            f"{'Variable':<15} {'Coef.':<12} {'Std.Err.':<12} "
            f"{'z':<8} {'P>|z|':<8} {'[0.025':<10} {'0.975]':<10}"
        )
        lines.append("-" * 78)

        if self.bootstrap_ci is not None:
            ci_lower_vals, ci_upper_vals = self.bootstrap_ci
        else:
            ci = self.conf_int(alpha=0.05)
            ci_lower_vals = ci["lower"]
            ci_upper_vals = ci["upper"]

        for var in self.params.index:
            coef = self.params[var]
            se = self.std_errors[var]
            if np.isnan(se) or se == 0:
                t_str = f"{'':>7}"
                p_str = f"{'':>7}"
                stars = ""
            else:
                t = coef / se
                p = 2 * (1 - stats.norm.cdf(abs(t)))
                t_str = f"{t:>7.3f}"
                p_str = f"{p:>7.4f}"
                if p < 0.001:
                    stars = "***"
                elif p < 0.01:
                    stars = "**"
                elif p < 0.05:
                    stars = "*"
                elif p < 0.10:
                    stars = "."
                else:
                    stars = ""

            ci_lo = (
                ci_lower_vals[var]
                if isinstance(ci_lower_vals, pd.Series)
                else ci_lower_vals[list(self.params.index).index(var)]
            )
            ci_hi = (
                ci_upper_vals[var]
                if isinstance(ci_upper_vals, pd.Series)
                else ci_upper_vals[list(self.params.index).index(var)]
            )

            lines.append(
                f"{var:<15} {coef:>11.4f} {se:>11.4f} {t_str} "
                f"{p_str} {ci_lo:>9.4f} {ci_hi:>9.4f} {stars}"
            )

        lines.append("=" * 78)
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")

        # Bias correction summary
        lines.append("-" * 78)
        lines.append("Bias Correction Summary".center(78))
        lines.append("-" * 78)
        lines.append(
            f"{'Variable':<15} {'LSDV':<12} {'LSDVC':<12} {'Bias Corr.':<12} {'% Change':<10}"
        )
        lines.append("-" * 78)
        for var in self.params.index:
            lsdv_val = self.lsdv_params[var]
            lsdvc_val = self.params[var]
            bias_val = self.bias_correction[var]
            if abs(lsdv_val) > 1e-10:
                pct = 100.0 * bias_val / lsdv_val
                pct_str = f"{pct:>9.2f}%"
            else:
                pct_str = f"{'N/A':>10}"
            lines.append(
                f"{var:<15} {lsdv_val:>11.4f} {lsdvc_val:>11.4f} {bias_val:>11.4f} {pct_str}"
            )
        lines.append("=" * 78)
        lines.append("")

        return "\n".join(lines)

    def bias_summary(self) -> str:
        """Show magnitude of bias correction per parameter."""
        lines = []
        lines.append("=" * 60)
        lines.append("LSDVC Bias Correction Detail".center(60))
        lines.append("=" * 60)
        lines.append(f"{'Variable':<15} {'LSDV':<12} {'LSDVC':<12} {'Correction':<12}")
        lines.append("-" * 60)
        for var in self.params.index:
            lines.append(
                f"{var:<15} {self.lsdv_params[var]:>11.4f} "
                f"{self.params[var]:>11.4f} {self.bias_correction[var]:>11.4f}"
            )
        lines.append("-" * 60)
        lines.append(f"Initial estimator: {self.initial_estimator.upper()}")
        lines.append(f"Initial rho:       {self.initial_rho:.6f}")
        lines.append(f"Bias order:        {self.bias_order}")
        lines.append(f"Iterations:        {self.n_iterations}")
        lines.append(f"Converged:         {self.converged}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def plot_bootstrap_distribution(self, variable=None):
        """
        Plot histogram of bootstrap parameter distribution.

        Parameters
        ----------
        variable : str, optional
            Variable name to plot. If None, plots the first parameter
            (lagged dependent variable).

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("matplotlib is required for plot_bootstrap_distribution") from err

        if self.bootstrap_params is None:
            raise ValueError("No bootstrap results available. Run fit() with n_bootstrap > 0.")

        param_names = list(self.params.index)
        if variable is None:
            idx = 0
            variable = param_names[0]
        else:
            if variable not in param_names:
                raise ValueError(f"Variable '{variable}' not found. Available: {param_names}")
            idx = param_names.index(variable)

        boot_vals = self.bootstrap_params[:, idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(boot_vals, bins=30, density=True, alpha=0.7, edgecolor="black")
        ax.axvline(
            self.params.iloc[idx],
            color="red",
            linestyle="--",
            label=f"LSDVC = {self.params.iloc[idx]:.4f}",
        )
        ax.axvline(
            self.lsdv_params.iloc[idx],
            color="blue",
            linestyle=":",
            label=f"LSDV = {self.lsdv_params.iloc[idx]:.4f}",
        )
        if self.bootstrap_ci is not None:
            ci_lo = self.bootstrap_ci[0]
            ci_hi = self.bootstrap_ci[1]
            lo = ci_lo[idx] if not isinstance(ci_lo, pd.Series) else ci_lo.iloc[idx]
            hi = ci_hi[idx] if not isinstance(ci_hi, pd.Series) else ci_hi.iloc[idx]
            ax.axvline(lo, color="green", linestyle="-.", alpha=0.7, label="95% CI")
            ax.axvline(hi, color="green", linestyle="-.", alpha=0.7)

        ax.set_xlabel(variable)
        ax.set_ylabel("Density")
        ax.set_title(f"Bootstrap Distribution: {variable}")
        ax.legend()
        fig.tight_layout()
        return fig


class LSDVC(PanelModel):
    """
    Least Squares Dummy Variable Corrected estimator.

    Estimates a dynamic panel model with fixed effects and corrects for
    Nickell (1981) bias using the analytical bias formulas of Kiviet (1995),
    extended by Bun & Kiviet (2003) to higher orders.

    The model is:
        y_{it} = rho * y_{i,t-1} + beta' X_{it} + alpha_i + e_{it}

    Parameters
    ----------
    formula : str
        Model formula in R-style syntax (e.g., "y ~ x1 + x2").
        The lagged dependent variable is added automatically.
    data : pd.DataFrame
        Panel data in long format.
    entity_col : str
        Column identifying entities (individuals/firms).
    time_col : str
        Column identifying time periods.
    lags : int, default=1
        Number of AR lags (currently only 1 supported).
    initial_estimator : str, default='ab'
        Initial consistent estimator for bias correction:
        - 'ah': Anderson-Hsiao IV
        - 'ab': Arellano-Bond Difference GMM (one-step)
        - 'bb': Blundell-Bond System GMM (one-step)
    bias_order : int, default=2
        Order of Kiviet bias approximation (1, 2, or 3).
    max_iter : int, default=50
        Maximum iterations for iterated bias correction.
    tol : float, default=1e-6
        Convergence tolerance for iterated bias correction.

    Examples
    --------
    >>> from panelbox.models.dynamic.lsdvc import LSDVC
    >>> from panelbox import load_grunfeld
    >>> data = load_grunfeld()
    >>> model = LSDVC(
    ...     "invest ~ value + capital", data, "firm", "year", initial_estimator="ab", bias_order=2
    ... )
    >>> results = model.fit()
    >>> print(results.summary())
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        lags: int = 1,
        initial_estimator: str = "ab",
        bias_order: int = 2,
        max_iter: int = 50,
        tol: float = 1e-6,
    ):
        super().__init__(formula, data, entity_col, time_col)
        if lags != 1:
            raise NotImplementedError("Only lags=1 is currently supported.")
        if initial_estimator not in ("ah", "ab", "bb"):
            raise ValueError(
                f"initial_estimator must be 'ah', 'ab', or 'bb', got '{initial_estimator}'"
            )
        if bias_order not in (1, 2, 3):
            raise ValueError(f"bias_order must be 1, 2, or 3, got {bias_order}")

        self.lags = lags
        self.initial_estimator = initial_estimator
        self.bias_order = bias_order
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        n_bootstrap: int = 500,
        ci_level: float = 0.95,
        seed: int | None = None,
        **kwargs,
    ) -> LSDVCResults:
        """
        Fit the LSDVC model with bootstrap inference.

        Pipeline:
        1. Build lag matrices
        2. LSDV estimation
        3. Initial consistent estimate
        4. Kiviet bias correction (iterative)
        5. Bootstrap parametric inference
        6. Build LSDVCResults

        Parameters
        ----------
        n_bootstrap : int, default=500
            Number of bootstrap replications. Set to 0 to skip bootstrap.
        ci_level : float, default=0.95
            Confidence level for bootstrap intervals.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        LSDVCResults
            Fitted model results with bias-corrected coefficients and
            bootstrap standard errors.
        """
        # Parse formula and get variable names
        dep_var = self.formula_parser.dependent
        exog_vars = self.formula_parser.regressors

        df = self.data.data.sort_values([self.data.entity_col, self.data.time_col]).copy()

        entity_col = self.data.entity_col
        time_col = self.data.time_col

        # Build lagged dependent variable
        df["_y_lag1"] = df.groupby(entity_col)[dep_var].shift(1)

        # Drop rows with missing lag
        df_valid = df.dropna(subset=["_y_lag1"]).copy()

        # Extract arrays
        y = df_valid[dep_var].values.astype(float)
        y_lag = df_valid["_y_lag1"].values.astype(float)
        X = df_valid[exog_vars].values.astype(float) if exog_vars else None

        entities = df_valid[entity_col].values
        unique_entities = np.unique(entities)
        N = len(unique_entities)

        # Map entities to integer indices for within transformation
        entity_map = {e: i for i, e in enumerate(unique_entities)}
        entity_idx = np.array([entity_map[e] for e in entities])

        # Count time periods per entity (for balanced: all same)
        T_per_entity = np.bincount(entity_idx)
        T = int(np.median(T_per_entity))  # representative T

        # Step 1: LSDV estimation
        lsdv_result = self._estimate_lsdv(y, y_lag, X, entity_idx, N)
        rho_lsdv = lsdv_result["rho"]
        beta_lsdv = lsdv_result["beta"]
        sigma2 = lsdv_result["sigma2"]

        logger.info(f"LSDV rho = {rho_lsdv:.6f}")

        # Step 2: Initial consistent estimate of rho
        rho_init = self._initial_consistent_estimate(dep_var, exog_vars, entity_col, time_col, df)
        logger.info(f"Initial consistent estimate ({self.initial_estimator}): rho = {rho_init:.6f}")

        # Step 3-5: Iterated bias correction
        result = self._iterate_bias_correction(
            rho_lsdv=rho_lsdv,
            beta_lsdv=beta_lsdv,
            sigma2=sigma2,
            T=T,
            N=N,
            entity_idx=entity_idx,
            rho_init=rho_init,
        )

        rho_lsdvc = result["rho_corrected"]
        n_iter = result["n_iter"]
        converged = result["converged"]

        logger.info(f"LSDVC rho = {rho_lsdvc:.6f} (iterations={n_iter}, converged={converged})")

        # Assemble corrected parameter vector
        # The bias correction only applies to rho; beta is from LSDV
        if beta_lsdv is not None:
            params_corrected = np.concatenate([[rho_lsdvc], beta_lsdv])
            params_lsdv_vec = np.concatenate([[rho_lsdv], beta_lsdv])
        else:
            params_corrected = np.array([rho_lsdvc])
            params_lsdv_vec = np.array([rho_lsdv])

        # Compute entity fixed effects (alpha_i) for bootstrap DGP
        W = np.column_stack([y_lag, X]) if X is not None else y_lag.reshape(-1, 1)
        alpha_i = np.zeros(N)
        for i in range(N):
            mask = entity_idx == i
            alpha_i[i] = np.mean(y[mask] - W[mask] @ params_corrected)

        # Extract observed initial y values (y_{i,0}) for bootstrap DGP
        # These are the first observations *before* the lag was applied
        y0_by_entity = np.zeros(N)
        for i, ent in enumerate(unique_entities):
            ent_data = df[df[entity_col] == ent].sort_values(time_col)
            y0_by_entity[i] = ent_data[dep_var].iloc[0]

        # Compute residuals with corrected params
        fitted_values = W @ params_corrected
        resid = y - fitted_values
        # Remove within-entity mean of residuals (FE residuals)
        for i in range(N):
            mask = entity_idx == i
            resid[mask] -= resid[mask].mean()

        # Parameter names
        param_names = [f"L1.{dep_var}"]
        if exog_vars:
            param_names.extend(exog_vars)

        n_obs = len(y)
        k = len(params_corrected)

        # Bias correction vector
        bias_correction = params_lsdv_vec - params_corrected

        # Step 6: Bootstrap inference
        bootstrap_params = None
        bootstrap_ci = None
        std_errors = np.full(k, np.nan)
        cov_params = np.full((k, k), np.nan)
        cov_type = "none"

        if n_bootstrap > 0:
            boot_result = self._bootstrap_inference(
                rho=rho_lsdvc,
                beta=beta_lsdv,
                alpha=alpha_i,
                sigma2=sigma2,
                X_orig=X,
                entity_idx=entity_idx,
                T_per_entity=T_per_entity,
                N=N,
                dep_var=dep_var,
                exog_vars=exog_vars,
                entity_col=entity_col,
                time_col=time_col,
                unique_entities=unique_entities,
                y0_by_entity=y0_by_entity,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                seed=seed,
            )
            std_errors = boot_result["std_errors"]
            bootstrap_params = boot_result["bootstrap_params"]
            bootstrap_ci = (boot_result["ci_lower"], boot_result["ci_upper"])
            cov_params = np.cov(bootstrap_params.T)
            cov_type = "bootstrap"

        # Build results
        model_info = {
            "model_type": "LSDVC",
            "formula": self.formula,
            "initial_estimator": self.initial_estimator,
            "bias_order": self.bias_order,
            "max_iter": self.max_iter,
            "n_iter": n_iter,
            "converged": converged,
            "rho_lsdv": rho_lsdv,
            "rho_init": rho_init,
            "sigma2": sigma2,
            "cov_type": cov_type,
            "cov_kwds": {"n_bootstrap": n_bootstrap, "ci_level": ci_level},
        }

        data_info = {
            "nobs": n_obs,
            "n_entities": N,
            "n_periods": T,
            "df_model": k,
            "df_resid": n_obs - k - N,
            "entity_index": unique_entities,
            "time_index": np.sort(df_valid[time_col].unique()),
        }

        results = LSDVCResults(
            params=pd.Series(params_corrected, index=param_names),
            std_errors=pd.Series(std_errors, index=param_names),
            cov_params=pd.DataFrame(cov_params, index=param_names, columns=param_names),
            resid=pd.Series(resid),
            fittedvalues=pd.Series(fitted_values),
            model_info=model_info,
            data_info=data_info,
            bias_correction=pd.Series(bias_correction, index=param_names),
            lsdv_params=pd.Series(params_lsdv_vec, index=param_names),
            initial_estimator=self.initial_estimator,
            initial_rho=rho_init,
            n_iterations=n_iter,
            converged=converged,
            bias_order=self.bias_order,
            sigma2=sigma2,
            bootstrap_params=bootstrap_params,
            bootstrap_ci=bootstrap_ci,
        )

        self._fitted = True
        self._results = results
        return results

    def _estimate_coefficients(self) -> np.ndarray:
        """Return estimated coefficients (required by PanelModel ABC)."""
        if self._results is not None:
            return self._results.params.values
        raise ValueError("Model not fitted. Call fit() first.")

    # ------------------------------------------------------------------
    # Internal: Bootstrap parametric inference
    # ------------------------------------------------------------------

    def _bootstrap_inference(
        self,
        rho: float,
        beta: np.ndarray | None,
        alpha: np.ndarray,
        sigma2: float,
        X_orig: np.ndarray | None,
        entity_idx: np.ndarray,
        T_per_entity: np.ndarray,
        N: int,
        dep_var: str,
        exog_vars: list[str],
        entity_col: str,
        time_col: str,
        unique_entities: np.ndarray,
        y0_by_entity: np.ndarray,
        n_bootstrap: int = 500,
        ci_level: float = 0.95,
        seed: int | None = None,
    ) -> dict:
        """
        Parametric bootstrap for LSDVC inference.

        Uses estimated parameters as DGP to generate B bootstrap panels,
        re-estimates LSDVC on each, and computes SEs and confidence
        intervals from the bootstrap distribution.

        Parameters
        ----------
        rho : float
            LSDVC estimate of rho.
        beta : ndarray or None
            LSDVC estimates of exogenous coefficients.
        alpha : ndarray, shape (N,)
            Entity fixed effects.
        sigma2 : float
            Error variance.
        X_orig : ndarray or None, shape (n, k)
            Original exogenous regressors.
        entity_idx : ndarray
            Integer entity indices for each observation.
        T_per_entity : ndarray
            Number of time periods per entity.
        N : int
            Number of entities.
        T : int
            Representative number of time periods.
        dep_var : str
            Dependent variable name.
        exog_vars : list[str]
            Exogenous variable names.
        entity_col, time_col : str
            Column names for entity and time.
        unique_entities : ndarray
            Unique entity identifiers.
        df_orig : pd.DataFrame
            Original data (for X structure and entity/time columns).
        n_bootstrap : int
            Number of bootstrap replications.
        ci_level : float
            Confidence level for intervals.
        seed : int or None
            Random seed.

        Returns
        -------
        dict with keys:
            - 'std_errors': ndarray (K,)
            - 'ci_lower': ndarray (K,)
            - 'ci_upper': ndarray (K,)
            - 'bootstrap_params': ndarray (B, K)
        """
        rng = np.random.default_rng(seed)

        if beta is not None:
            k = 1 + len(beta)
        else:
            k = 1

        bootstrap_params = np.full((n_bootstrap, k), np.nan)
        n_valid = 0

        # Organize X by entity for simulation
        X_by_entity = {}
        if X_orig is not None:
            for i in range(N):
                mask = entity_idx == i
                X_by_entity[i] = X_orig[mask]

        for b in range(n_bootstrap):
            try:
                # Simulate a panel
                sim_df = self._simulate_panel(
                    rho=rho,
                    beta=beta,
                    alpha=alpha,
                    sigma2=sigma2,
                    X_by_entity=X_by_entity,
                    T_per_entity=T_per_entity,
                    N=N,
                    unique_entities=unique_entities,
                    dep_var=dep_var,
                    exog_vars=exog_vars,
                    entity_col=entity_col,
                    time_col=time_col,
                    y0_by_entity=y0_by_entity,
                    rng=rng,
                )

                # Re-estimate LSDVC on simulated data
                boot_model = LSDVC(
                    formula=self.formula,
                    data=sim_df,
                    entity_col=entity_col,
                    time_col=time_col,
                    lags=self.lags,
                    initial_estimator=self.initial_estimator,
                    bias_order=self.bias_order,
                    max_iter=self.max_iter,
                    tol=self.tol,
                )
                boot_result = boot_model.fit(n_bootstrap=0)
                bootstrap_params[n_valid] = boot_result.params.values
                n_valid += 1
            except Exception as e:
                logger.debug(f"Bootstrap replication {b} failed: {e}")
                continue

        if n_valid < 10:
            raise RuntimeError(
                f"Only {n_valid} of {n_bootstrap} bootstrap replications "
                f"succeeded. Cannot compute reliable inference."
            )

        # Trim to valid replications
        bootstrap_params = bootstrap_params[:n_valid]

        # Compute standard errors and confidence intervals
        std_errors = np.std(bootstrap_params, axis=0, ddof=1)
        alpha_ci = 1 - ci_level
        ci_lower = np.percentile(bootstrap_params, 100 * alpha_ci / 2, axis=0)
        ci_upper = np.percentile(bootstrap_params, 100 * (1 - alpha_ci / 2), axis=0)

        logger.info(f"Bootstrap: {n_valid}/{n_bootstrap} valid replications")

        return {
            "std_errors": std_errors,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "bootstrap_params": bootstrap_params,
        }

    def _simulate_panel(
        self,
        rho: float,
        beta: np.ndarray | None,
        alpha: np.ndarray,
        sigma2: float,
        X_by_entity: dict,
        T_per_entity: np.ndarray,
        N: int,
        unique_entities: np.ndarray,
        dep_var: str,
        exog_vars: list[str],
        entity_col: str,
        time_col: str,
        y0_by_entity: np.ndarray,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Simulate a complete panel for one bootstrap replication.

        Uses the estimated DGP parameters to generate synthetic data:
            y_{it} = rho * y_{i,t-1} + X_{it} * beta + alpha_i + epsilon_{it}

        Uses observed initial values y_{i,0} from the original data to
        preserve realistic scale and dynamics (Bruno, 2005).

        Parameters
        ----------
        rho : float
            AR(1) coefficient.
        beta : ndarray or None
            Coefficients on exogenous regressors.
        alpha : ndarray, shape (N,)
            Entity fixed effects.
        sigma2 : float
            Error variance.
        X_by_entity : dict
            X matrices keyed by entity index.
        T_per_entity : ndarray
            Time periods per entity.
        N : int
            Number of entities.
        unique_entities : ndarray
            Entity labels.
        dep_var, exog_vars, entity_col, time_col : str/list
            Column names.
        y0_by_entity : ndarray, shape (N,)
            Observed initial y value for each entity (y_{i,0}).
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        pd.DataFrame
            Simulated panel data.
        """
        sigma = np.sqrt(sigma2)
        rows = []

        for i in range(N):
            Ti = int(T_per_entity[i])
            # Ti is the number of post-lag observations.
            # We generate Ti+1 periods total (index 0..Ti).

            # Use observed initial value
            y_prev = y0_by_entity[i]

            # Get X for this entity (if any)
            Xi = X_by_entity.get(i) if X_by_entity else None

            y_series = np.empty(Ti + 1)
            y_series[0] = y_prev

            for t in range(1, Ti + 1):
                eps = rng.normal(0, sigma)
                x_beta = 0.0
                if Xi is not None and beta is not None:
                    # Use the original X values (t-1 because Xi has Ti rows)
                    x_idx = min(t - 1, len(Xi) - 1)
                    x_beta = float(Xi[x_idx] @ beta)
                y_new = rho * y_prev + x_beta + alpha[i] + eps
                y_series[t] = y_new
                y_prev = y_new

            # Build rows: t=0..Ti
            entity_label = unique_entities[i]
            for t in range(Ti + 1):
                row = {entity_col: entity_label, time_col: t}
                row[dep_var] = y_series[t]
                if Xi is not None and exog_vars:
                    x_idx = min(t, len(Xi) - 1) if t > 0 else 0
                    if t > 0:
                        x_idx = min(t - 1, len(Xi) - 1)
                    else:
                        x_idx = 0
                    for j, var_name in enumerate(exog_vars):
                        row[var_name] = Xi[x_idx, j]
                rows.append(row)

        sim_df = pd.DataFrame(rows)
        sim_df = sim_df.sort_values([entity_col, time_col]).reset_index(drop=True)
        return sim_df

    # ------------------------------------------------------------------
    # Internal: LSDV (Fixed Effects with lagged dependent variable)
    # ------------------------------------------------------------------

    def _estimate_lsdv(
        self,
        y: np.ndarray,
        y_lag: np.ndarray,
        X: np.ndarray | None,
        entity_idx: np.ndarray,
        N: int,
    ) -> dict:
        """
        Estimate FE (within) with y_{t-1} as regressor.

        Parameters
        ----------
        y : array, shape (n,)
            Dependent variable.
        y_lag : array, shape (n,)
            Lagged dependent variable.
        X : array, shape (n, k) or None
            Exogenous regressors.
        entity_idx : array, shape (n,)
            Integer entity indices.
        N : int
            Number of entities.

        Returns
        -------
        dict with keys: rho, beta, params, resid, sigma2
        """
        n = len(y)

        # Within transformation: demean by entity
        y_dm = self._within_transform(y, entity_idx, N)
        y_lag_dm = self._within_transform(y_lag, entity_idx, N)

        if X is not None:
            X_dm = np.column_stack(
                [self._within_transform(X[:, j], entity_idx, N) for j in range(X.shape[1])]
            )
            W_dm = np.column_stack([y_lag_dm, X_dm])
        else:
            W_dm = y_lag_dm.reshape(-1, 1)

        # OLS on demeaned data
        params = np.linalg.lstsq(W_dm, y_dm, rcond=None)[0]

        resid = y_dm - W_dm @ params
        k = len(params)
        sigma2 = float(np.sum(resid**2) / (n - N - k))

        rho = float(params[0])
        beta = params[1:] if len(params) > 1 else None

        return {
            "rho": rho,
            "beta": beta,
            "params": params,
            "resid": resid,
            "sigma2": sigma2,
        }

    @staticmethod
    def _within_transform(x: np.ndarray, entity_idx: np.ndarray, N: int) -> np.ndarray:
        """Demean x by entity (within transformation)."""
        x_dm = x.copy().astype(float)
        for i in range(N):
            mask = entity_idx == i
            x_dm[mask] -= x_dm[mask].mean()
        return x_dm

    # ------------------------------------------------------------------
    # Internal: Initial consistent estimators
    # ------------------------------------------------------------------

    def _initial_consistent_estimate(
        self,
        dep_var: str,
        exog_vars: list[str],
        entity_col: str,
        time_col: str,
        df: pd.DataFrame,
    ) -> float:
        """
        Dispatch to AH, AB, or BB for initial consistent rho estimate.

        Returns
        -------
        float
            Consistent estimate of rho (AR(1) coefficient).
        """
        if self.initial_estimator == "ah":
            return self._ah_estimate(dep_var, exog_vars, entity_col, time_col, df)
        elif self.initial_estimator == "bb":
            return self._bb_estimate(dep_var, exog_vars, entity_col, time_col, df)
        else:  # 'ab'
            return self._ab_estimate(dep_var, exog_vars, entity_col, time_col, df)

    def _ah_estimate(self, dep_var, exog_vars, entity_col, time_col, df) -> float:
        """Anderson-Hsiao IV estimate of rho."""
        from panelbox.models.dynamic.anderson_hsiao import AndersonHsiao

        ah = AndersonHsiao(
            data=df,
            dep_var=dep_var,
            entity_col=entity_col,
            time_col=time_col,
            lags=1,
            exog_vars=exog_vars if exog_vars else None,
            instrument="levels",
        )
        result = ah.fit()
        return float(result["rho"])

    def _ab_estimate(self, dep_var, exog_vars, entity_col, time_col, df) -> float:
        """Arellano-Bond Difference GMM (one-step) estimate of rho."""
        from panelbox.gmm.difference_gmm import DifferenceGMM

        model = DifferenceGMM(
            data=df,
            dep_var=dep_var,
            lags=1,
            id_var=entity_col,
            time_var=time_col,
            exog_vars=exog_vars if exog_vars else None,
            time_dummies=False,
            collapse=True,
            two_step=False,
            robust=False,
        )
        results = model.fit()
        # First parameter is L1.dep_var
        return float(results.params.iloc[0])

    def _bb_estimate(self, dep_var, exog_vars, entity_col, time_col, df) -> float:
        """Blundell-Bond System GMM (one-step) estimate of rho."""
        from panelbox.gmm.system_gmm import SystemGMM

        model = SystemGMM(
            data=df,
            dep_var=dep_var,
            lags=1,
            id_var=entity_col,
            time_var=time_col,
            exog_vars=exog_vars if exog_vars else None,
            time_dummies=False,
            collapse=True,
            two_step=False,
            robust=False,
        )
        results = model.fit()
        return float(results.params.iloc[0])

    # ------------------------------------------------------------------
    # Internal: Kiviet (1995) bias formulas
    # ------------------------------------------------------------------

    def _compute_kiviet_bias(
        self,
        rho: float,
        sigma2: float,
        T: int,
        N: int,
        order: int = 2,
    ) -> float:
        """
        Compute analytical bias of Kiviet (1995) for the LSDV estimator.

        The bias of the within estimator of rho in an AR(1) panel model is:

            bias(rho_hat) = B1/NT + B2/NT^2 + B3/NT^3 + ...

        Parameters
        ----------
        rho : float
            Current estimate of rho (AR coefficient).
        sigma2 : float
            Error variance estimate.
        T : int
            Number of time periods (after losing one to lag).
        N : int
            Number of entities.
        order : int
            Bias approximation order (1, 2, or 3).

        Returns
        -------
        float
            Estimated bias of the LSDV estimator.

        Notes
        -----
        Follows Kiviet (1995, eq. 3.4-3.6) and Bun & Kiviet (2003).

        Order 1 (Nickell bias, leading term):
            B1 = -(1 + rho) / (T - 1)

        Order 2 (Kiviet 1995 refinement):
            B2 = B1 - (1/(T-1)^2) * correction_term

        Order 3 (Bun & Kiviet 2003 further refinement):
            B3 = B2 + higher_order_terms
        """
        if T <= 1:
            return 0.0

        # -----------------------------------------------------------------
        # Order 1: Leading Nickell (1981) bias term
        # Nickell (1981) showed the LSDV bias for rho in AR(1)-FE is
        # approximately -(1+rho)/(T-1) for large N, fixed T.
        # -----------------------------------------------------------------
        bias_1 = -(1.0 + rho) / (T - 1)

        if order == 1:
            return bias_1

        # -----------------------------------------------------------------
        # Order 2: Kiviet (1995) refinement
        # Adds terms involving rho^T and geometric sums that capture
        # the finite-T structure of the within transformation.
        # -----------------------------------------------------------------
        # Geometric sum S = sum_{j=0}^{T-2} rho^j = (1 - rho^{T-1})/(1 - rho)
        # when |rho| < 1
        if abs(rho) < 1.0 - 1e-12:
            S = (1.0 - rho ** (T - 1)) / (1.0 - rho)
        else:
            S = float(T - 1)

        # Second-order correction from Kiviet (1995)
        # This accounts for the interaction between the within transformation
        # and the AR structure
        correction_2 = (
            -(1.0 + rho) / (T - 1) * (1.0 / (T - 1) * ((2.0 * rho * S) / (T - 1) - (1.0 + rho)))
        )

        bias_2 = bias_1 + correction_2

        if order == 2:
            return bias_2

        # -----------------------------------------------------------------
        # Order 3: Bun & Kiviet (2003) higher-order refinement
        # -----------------------------------------------------------------
        # Sum of squares: S2 = sum_{j=0}^{T-2} j * rho^j
        if abs(rho) < 1.0 - 1e-12:
            # S2 = rho * d/d(rho) [S] = rho * [(T-1)*rho^{T-2}*(rho-1) + S] / (1-rho)
            # More stable: direct computation
            S2 = sum(j * rho**j for j in range(T - 1))
        else:
            S2 = sum(float(j) for j in range(T - 1))

        correction_3 = (
            (1.0 + rho)
            / (T - 1) ** 3
            * (3.0 * rho * S2 / (T - 1) - 2.0 * rho * S**2 / (T - 1) ** 2 + rho * (1.0 + rho))
        )

        bias_3 = bias_2 + correction_3

        return bias_3

    # ------------------------------------------------------------------
    # Internal: Iterated bias correction
    # ------------------------------------------------------------------

    def _iterate_bias_correction(
        self,
        rho_lsdv: float,
        beta_lsdv: np.ndarray | None,
        sigma2: float,
        T: int,
        N: int,
        entity_idx: np.ndarray,
        rho_init: float,
    ) -> dict:
        """
        Iterate bias correction until convergence.

        Uses the initial consistent estimate to compute bias, corrects the
        LSDV estimate, then re-computes bias using the corrected estimate.
        Repeats until convergence or max_iter.

        Parameters
        ----------
        rho_lsdv : float
            Uncorrected LSDV estimate of rho.
        beta_lsdv : array or None
            LSDV estimates of exogenous coefficients.
        sigma2 : float
            Error variance from LSDV.
        T : int
            Number of time periods.
        N : int
            Number of entities.
        entity_idx : array
            Entity indices.
        rho_init : float
            Initial consistent estimate of rho.

        Returns
        -------
        dict
            - rho_corrected: bias-corrected rho
            - n_iter: number of iterations
            - converged: whether iteration converged
            - bias: final bias estimate
        """
        rho_current = rho_init
        bias = 0.0
        rho_new = rho_current

        for iteration in range(1, self.max_iter + 1):
            # Compute bias at current rho
            bias = self._compute_kiviet_bias(
                rho=rho_current,
                sigma2=sigma2,
                T=T,
                N=N,
                order=self.bias_order,
            )

            # Correct LSDV estimate
            rho_new = rho_lsdv - bias

            # Check convergence
            if abs(rho_new - rho_current) < self.tol:
                logger.debug(f"Bias correction converged in {iteration} iterations")
                return {
                    "rho_corrected": rho_new,
                    "n_iter": iteration,
                    "converged": True,
                    "bias": bias,
                }

            rho_current = rho_new

        logger.warning(f"Bias correction did not converge in {self.max_iter} iterations")
        return {
            "rho_corrected": rho_new,
            "n_iter": self.max_iter,
            "converged": False,
            "bias": bias,
        }
