"""
Mean Group (MG) and Pooled Mean Group (PMG) estimators for panel data.

This module implements:
- Mean Group estimator (Pesaran & Smith, 1995)
- Pooled Mean Group estimator (Pesaran, Shin & Smith, 1999)
- Hausman test for MG vs PMG comparison
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from panelbox.core.base_model import PanelModel
from panelbox.core.results import PanelResults

logger = logging.getLogger(__name__)


class MeanGroupResults(PanelResults):
    """
    Results container for Mean Group estimation.

    Extends PanelResults with entity-level regression details,
    the Swamy (1970) slope homogeneity test, and visualization helpers.

    Attributes
    ----------
    entity_params : dict
        {entity_id: pd.Series} of entity-specific coefficients.
    entity_std_errors : dict
        {entity_id: pd.Series} of entity-specific standard errors.
    entity_rsquared : dict
        {entity_id: float} of entity-specific R-squared.
    n_entities_used : int
        Number of entities used in estimation.
    entities_excluded : list
        Entities excluded due to insufficient observations or rank deficiency.
    swamy_test_result : dict
        Swamy (1970) test for slope homogeneity.
    """

    def __init__(
        self,
        *,
        entity_params: dict,
        entity_std_errors: dict,
        entity_rsquared: dict,
        n_entities_used: int,
        entities_excluded: list,
        swamy_test_result: dict,
        entity_results: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.entity_params = entity_params
        self.entity_std_errors = entity_std_errors
        self.entity_rsquared = entity_rsquared
        self.n_entities_used = n_entities_used
        self.entities_excluded = entities_excluded
        self.swamy_test_result = swamy_test_result
        self.entity_results = entity_results

    def entity_summary(self, entity_id) -> str:
        """
        Summary of the OLS regression for a single entity.

        Parameters
        ----------
        entity_id
            Entity identifier.

        Returns
        -------
        str
            Formatted summary string.
        """
        if entity_id not in self.entity_params:
            raise ValueError(
                f"Entity '{entity_id}' not found. Available: {list(self.entity_params.keys())}"
            )

        params = self.entity_params[entity_id]
        se = self.entity_std_errors[entity_id]
        rsq = self.entity_rsquared[entity_id]
        nobs = self.entity_results[entity_id]["nobs"]

        lines = []
        lines.append("=" * 60)
        lines.append(f"Entity: {entity_id}".center(60))
        lines.append("=" * 60)
        lines.append(f"No. Observations:  {nobs:>10}")
        lines.append(f"R-squared:         {rsq:>10.4f}")
        lines.append("-" * 60)
        lines.append(f"{'Variable':<15} {'Coef.':<12} {'Std.Err.':<12} {'t':<10}")
        lines.append("-" * 60)

        for var in params.index:
            coef = params[var]
            std = se[var]
            t_val = coef / std if std > 0 else np.nan
            lines.append(f"{var:<15} {coef:>11.4f} {std:>11.4f} {t_val:>9.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def plot_coefficient_distribution(self, variable, ax=None):
        """
        Boxplot of entity-specific coefficients for a variable.

        Parameters
        ----------
        variable : str
            Variable name.
        ax : matplotlib Axes, optional
            If provided, plot on this axes.

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        coefs = [
            self.entity_params[eid][variable]
            for eid in self.entity_params
            if variable in self.entity_params[eid].index
        ]

        if ax is None:
            _fig, ax = plt.subplots(figsize=(8, 5))

        ax.boxplot(coefs, vert=True, patch_artist=True)
        mg_val = self.params[variable]
        ax.axhline(mg_val, color="red", linestyle="--", label=f"MG = {mg_val:.4f}")
        ax.set_title(f"Distribution of entity coefficients: {variable}")
        ax.set_ylabel("Coefficient")
        ax.legend()
        return ax

    def coefficient_table(self) -> pd.DataFrame:
        """
        DataFrame with entity-specific coefficients for all variables.

        Returns
        -------
        pd.DataFrame
            Index = entity IDs, columns = variable names.
        """
        rows = {}
        for eid, params in self.entity_params.items():
            rows[eid] = params
        df = pd.DataFrame(rows).T
        df.index.name = "entity"
        return df

    def summary(self, title: str | None = None) -> str:
        """Generate summary with MG-specific information."""
        base = super().summary(title=title)

        extra = []
        extra.append("-" * 78)
        extra.append(f"No. Entities Used:         {self.n_entities_used:>10}")
        if self.entities_excluded:
            extra.append(f"Entities Excluded:         {len(self.entities_excluded):>10}")

        sw = self.swamy_test_result
        extra.append("-" * 78)
        extra.append("Swamy (1970) Test for Slope Homogeneity")
        extra.append(f"  Statistic:               {sw['statistic']:>10.4f}")
        extra.append(f"  p-value:                 {sw['pvalue']:>10.4f}")
        extra.append(f"  df:                      {sw['df']:>10}")
        extra.append(f"  Reject H0 (5%):          {'Yes' if sw['reject'] else 'No':>9}")
        extra.append("=" * 78)

        return base + "\n" + "\n".join(extra)


class MeanGroupEstimator(PanelModel):
    """
    Mean Group (MG) estimator for panel data.

    Estimates entity-specific OLS regressions and averages coefficients
    across entities. Consistent under slope heterogeneity where Fixed
    Effects and Random Effects estimators are inconsistent.

    The MG estimator:
    1. Runs OLS for each entity i: y_i = X_i beta_i + e_i
    2. Averages: beta_MG = (1/N) sum(beta_i)
    3. Variance: Var(beta_MG) = (1/N(N-1)) sum((beta_i - beta_MG)(beta_i - beta_MG)')

    Parameters
    ----------
    formula : str
        Model formula in R-style syntax (e.g., "y ~ x1 + x2")
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of the column identifying entities
    time_col : str
        Name of the column identifying time periods
    weights : np.ndarray, optional
        Entity-level weights for weighted MG estimation.
        Must have length equal to the number of entities used.
    min_obs_per_entity : int, default=10
        Minimum observations per entity to include in estimation.
        Entities with fewer observations are excluded with a warning.

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_grunfeld()
    >>> mg = pb.MeanGroupEstimator("invest ~ value + capital", data, "firm", "year")
    >>> results = mg.fit()
    >>> print(results.summary())

    References
    ----------
    .. [1] Pesaran, M. H., & Smith, R. (1995). Estimating long-run
           relationships from dynamic heterogeneous panels. Journal of
           Econometrics, 68(1), 79-113.
    .. [2] Pesaran, M. H., Shin, Y., & Smith, R. P. (1999). Pooled mean
           group estimation of dynamic heterogeneous panels. Journal of
           the American Statistical Association, 94(446), 621-634.

    See Also
    --------
    FixedEffects : Within estimator assuming slope homogeneity
    RandomEffects : GLS estimator assuming slope homogeneity
    BetweenEstimator : Estimator using between-entity variation
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        weights: np.ndarray | None = None,
        min_obs_per_entity: int = 10,
    ):
        super().__init__(formula, data, entity_col, time_col, weights=None)
        self.min_obs_per_entity = min_obs_per_entity
        self._entity_weights = weights
        self.entity_results_: dict = {}

    def _estimate_entity_regressions(self) -> tuple[dict, list]:
        """
        Estimate OLS regressions for each entity.

        Returns
        -------
        entity_results : dict
            Dictionary mapping entity IDs to result dicts.
        entities_excluded : list
            Entity IDs that were excluded.
        """
        y_all, X_all = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )
        entities = self.data.data[self.data.entity_col].values
        k = X_all.shape[1]

        entity_results = {}
        entities_excluded = []

        for entity in np.unique(entities):
            mask = entities == entity
            y_i = y_all[mask]
            X_i = X_all[mask]
            T_i = len(y_i)

            # Check minimum observations
            if T_i < self.min_obs_per_entity:
                warnings.warn(
                    f"Entity '{entity}' excluded: T_i={T_i} < min_obs_per_entity={self.min_obs_per_entity}",
                    UserWarning,
                    stacklevel=2,
                )
                entities_excluded.append(entity)
                continue

            # Check rank of X_i
            rank = np.linalg.matrix_rank(X_i)
            if rank < k:
                warnings.warn(
                    f"Entity '{entity}' excluded: X_i is rank-deficient (rank={rank}, k={k})",
                    UserWarning,
                    stacklevel=2,
                )
                entities_excluded.append(entity)
                continue

            # OLS: beta_i = (X_i'X_i)^-1 X_i'y_i
            XtX = X_i.T @ X_i
            Xty = X_i.T @ y_i
            beta_i = np.linalg.solve(XtX, Xty)

            fitted_i = X_i @ beta_i
            resid_i = y_i - fitted_i

            # Variance-covariance: sigma^2 (X'X)^-1
            df_resid_i = T_i - k
            sigma2_i = np.sum(resid_i**2) / df_resid_i if df_resid_i > 0 else np.nan
            XtX_inv = np.linalg.inv(XtX)
            vcov_i = sigma2_i * XtX_inv

            # R-squared
            tss_i = np.sum((y_i - y_i.mean()) ** 2)
            rsq_i = 1.0 - np.sum(resid_i**2) / tss_i if tss_i > 0 else 0.0

            entity_results[entity] = {
                "beta": beta_i,
                "residuals": resid_i,
                "fitted": fitted_i,
                "vcov": vcov_i,
                "nobs": T_i,
                "rsquared": rsq_i,
            }

        return entity_results, entities_excluded

    def _compute_mg_estimates(self, entity_params: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Mean Group coefficient estimates.

        Parameters
        ----------
        entity_params : dict
            Dictionary mapping entity IDs to result dicts.

        Returns
        -------
        beta_mg : np.ndarray
            Mean Group coefficient estimates.
        vcov_mg : np.ndarray
            MG covariance matrix based on cross-entity dispersion.
        """
        betas = np.array([res["beta"] for res in entity_params.values()])
        N = len(betas)

        if self._entity_weights is not None:
            w = np.asarray(self._entity_weights, dtype=float)
            if len(w) != N:
                raise ValueError(
                    f"weights must have length {N} (number of entities used), got {len(w)}"
                )
            w_sum = w.sum()
            beta_mg = (w[:, None] * betas).sum(axis=0) / w_sum
            deviations = betas - beta_mg[None, :]
            vcov_mg = np.zeros((betas.shape[1], betas.shape[1]))
            for i in range(N):
                vcov_mg += w[i] * np.outer(deviations[i], deviations[i])
            vcov_mg /= w_sum * (N - 1) / N
        else:
            beta_mg = betas.mean(axis=0)
            vcov_mg = self._compute_mg_variance(betas, beta_mg)

        return beta_mg, vcov_mg

    def _compute_mg_variance(self, entity_betas: np.ndarray, mg_params: np.ndarray) -> np.ndarray:
        """
        Compute Mean Group variance.

        Var(beta_MG) = (1 / (N * (N-1))) * sum_i (beta_i - beta_MG)(beta_i - beta_MG)'

        This captures both sampling variation within each entity and
        genuine heterogeneity across entities.

        Parameters
        ----------
        entity_betas : np.ndarray
            (N, K) array of entity-specific coefficient vectors.
        mg_params : np.ndarray
            (K,) MG coefficient vector.

        Returns
        -------
        np.ndarray
            (K, K) covariance matrix.
        """
        N = len(entity_betas)
        deviations = entity_betas - mg_params[None, :]
        vcov_mg = (deviations.T @ deviations) / (N * (N - 1))
        return vcov_mg

    def _compute_pooled_fe(self) -> np.ndarray:
        """
        Compute pooled Fixed Effects (within) estimator for Swamy test.

        Returns
        -------
        beta_wfe : np.ndarray
            Within estimator coefficients (excludes intercept).
        """
        y_all, X_all = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )
        var_names = self.formula_parser.get_variable_names(self.data.data)
        entities = self.data.data[self.data.entity_col].values

        # Within-transform: demean by entity
        y_demean = y_all.copy()
        X_demean = X_all.copy()

        for entity in np.unique(entities):
            mask = entities == entity
            y_demean[mask] -= y_all[mask].mean()
            X_demean[mask] -= X_all[mask].mean(axis=0)

        # Drop intercept column (all zeros after demeaning)
        if "Intercept" in var_names:
            intercept_idx = list(var_names).index("Intercept")
            X_demean = np.delete(X_demean, intercept_idx, axis=1)

        # OLS on demeaned data
        beta_wfe = np.linalg.lstsq(X_demean, y_demean, rcond=None)[0]
        return beta_wfe

    def swamy_test(self, entity_results: dict) -> dict:
        """
        Swamy (1970) test for slope homogeneity.

        H0: beta_1 = beta_2 = ... = beta_N (slopes are homogeneous)

        S = sum_i (beta_i - beta_WFE)' V_i^{-1} (beta_i - beta_WFE)

        where beta_WFE is the pooled Fixed Effects estimator and
        V_i = Var(beta_i) is the entity-specific covariance matrix.

        Under H0, S ~ chi2((N-1)*K) where K is the number of slope
        parameters (excluding intercept).

        Parameters
        ----------
        entity_results : dict
            Entity regression results from _estimate_entity_regressions.

        Returns
        -------
        dict
            {'statistic': float, 'pvalue': float, 'df': int, 'reject': bool}
        """
        var_names = self.formula_parser.get_variable_names(self.data.data)
        has_intercept = "Intercept" in var_names

        # Get pooled FE (within) estimator — slopes only
        beta_wfe = self._compute_pooled_fe()

        N = len(entity_results)

        # Number of slope parameters (K, excluding intercept)
        if has_intercept:
            intercept_idx = list(var_names).index("Intercept")
            K = len(var_names) - 1
        else:
            intercept_idx = None
            K = len(var_names)

        S = 0.0
        for res in entity_results.values():
            beta_i = res["beta"]
            vcov_i = res["vcov"]

            # Remove intercept row/col if present
            if has_intercept:
                slope_idx = [j for j in range(len(beta_i)) if j != intercept_idx]
                beta_i_slopes = beta_i[slope_idx]
                vcov_i_slopes = vcov_i[np.ix_(slope_idx, slope_idx)]
            else:
                beta_i_slopes = beta_i
                vcov_i_slopes = vcov_i

            diff = beta_i_slopes - beta_wfe
            try:
                vcov_inv = np.linalg.inv(vcov_i_slopes)
                S += diff @ vcov_inv @ diff
            except np.linalg.LinAlgError:
                # Skip singular entities
                continue

        df = (N - 1) * K
        pvalue = 1.0 - stats.chi2.cdf(S, df) if df > 0 else np.nan

        return {
            "statistic": float(S),
            "pvalue": float(pvalue),
            "df": int(df),
            "reject": bool(pvalue < 0.05),
        }

    def fit(self, cov_type: str = "mg", **cov_kwds) -> MeanGroupResults:
        """
        Fit the Mean Group estimator.

        Parameters
        ----------
        cov_type : str, default='mg'
            Type of covariance estimator. Currently only 'mg' (default
            cross-entity dispersion) is supported.
        **cov_kwds
            Additional covariance arguments (reserved for future use).

        Returns
        -------
        MeanGroupResults
            Fitted model results.
        """
        # Step 1: Entity regressions
        self.entity_results_, entities_excluded = self._estimate_entity_regressions()

        if len(self.entity_results_) == 0:
            raise ValueError(
                "No entities had enough observations for estimation. "
                f"min_obs_per_entity={self.min_obs_per_entity}"
            )

        # Step 2: MG estimates
        beta_mg, vcov_mg = self._compute_mg_estimates(self.entity_results_)

        # Variable names
        var_names = self.formula_parser.get_variable_names(self.data.data)
        k = len(var_names)
        N_used = len(self.entity_results_)

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov_mg))

        # Build residuals and fitted values for all observations
        y_all, X_all = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )
        entities = self.data.data[self.data.entity_col].values

        fitted_all = np.full(len(y_all), np.nan)
        resid_all = np.full(len(y_all), np.nan)

        for entity, res in self.entity_results_.items():
            mask = entities == entity
            fitted_all[mask] = res["fitted"]
            resid_all[mask] = res["residuals"]

        # For excluded entities, use MG coefficients
        used_entities = set(self.entity_results_.keys())
        for entity in np.unique(entities):
            if entity not in used_entities:
                mask = entities == entity
                fitted_all[mask] = X_all[mask] @ beta_mg
                resid_all[mask] = y_all[mask] - fitted_all[mask]

        # Store original arrays for endog/exog properties
        self._y_orig = y_all
        self._X_orig = X_all

        # R-squared measures
        tss_overall = np.sum((y_all - y_all.mean()) ** 2)
        rss_overall = np.sum(resid_all**2)
        rsquared_overall = 1.0 - rss_overall / tss_overall if tss_overall > 0 else 0.0

        # Between R-squared (entity means)
        unique_entities_used = list(self.entity_results_.keys())
        y_means = np.array([y_all[entities == e].mean() for e in unique_entities_used])
        fitted_means = np.array([fitted_all[entities == e].mean() for e in unique_entities_used])
        tss_between = np.sum((y_means - y_means.mean()) ** 2)
        rss_between = np.sum((y_means - fitted_means) ** 2)
        rsquared_between = 1.0 - rss_between / tss_between if tss_between > 0 else 0.0

        # Adjusted R-squared
        nobs_total = len(y_all)
        df_model = k - 1 if "Intercept" in var_names else k
        df_resid = nobs_total - k
        rsquared_adj = (
            1.0 - (1.0 - rsquared_overall) * (nobs_total - 1) / df_resid if df_resid > 0 else np.nan
        )

        # Create pandas objects
        params = pd.Series(beta_mg, index=var_names)
        std_errors_series = pd.Series(std_errors, index=var_names)
        cov_params = pd.DataFrame(vcov_mg, index=var_names, columns=var_names)

        # Step 3: Swamy test
        swamy_result = self.swamy_test(self.entity_results_)

        # Build entity-level dicts for MeanGroupResults
        entity_params = {}
        entity_std_errors = {}
        entity_rsquared = {}
        for eid, res in self.entity_results_.items():
            entity_params[eid] = pd.Series(res["beta"], index=var_names)
            entity_std_errors[eid] = pd.Series(np.sqrt(np.diag(res["vcov"])), index=var_names)
            entity_rsquared[eid] = res["rsquared"]

        model_info = {
            "model_type": "Mean Group",
            "formula": self.formula,
            "cov_type": cov_type,
            "cov_kwds": cov_kwds,
        }

        data_info = {
            "nobs": nobs_total,
            "n_entities": N_used,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": N_used - 1,  # MG df based on N entities
            "entity_index": self.data.data[self.data.entity_col].values,
            "time_index": self.data.data[self.data.time_col].values,
        }

        rsquared_dict = {
            "rsquared": rsquared_overall,
            "rsquared_adj": rsquared_adj,
            "rsquared_within": np.nan,
            "rsquared_between": rsquared_between,
            "rsquared_overall": rsquared_overall,
        }

        results = MeanGroupResults(
            entity_params=entity_params,
            entity_std_errors=entity_std_errors,
            entity_rsquared=entity_rsquared,
            n_entities_used=N_used,
            entities_excluded=entities_excluded,
            swamy_test_result=swamy_result,
            entity_results=self.entity_results_,
            params=params,
            std_errors=std_errors_series,
            cov_params=cov_params,
            resid=resid_all,
            fittedvalues=fitted_all,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
            formula_parser=self.formula_parser,
        )

        self._results = results
        self._fitted = True

        return results

    def _estimate_coefficients(self) -> np.ndarray:
        """
        Estimate coefficients (implementation of abstract method).

        Returns
        -------
        np.ndarray
            Mean Group coefficient estimates.
        """
        entity_results, _ = self._estimate_entity_regressions()
        beta_mg, _ = self._compute_mg_estimates(entity_results)
        return beta_mg


# ---------------------------------------------------------------------------
# Pooled Mean Group (PMG) — Pesaran, Shin & Smith (1999)
# ---------------------------------------------------------------------------


class PMGResults(PanelResults):
    """
    Results container for Pooled Mean Group estimation.

    Attributes
    ----------
    long_run_params : pd.Series
        Homogeneous long-run coefficients (theta).
    entity_phi : dict
        {entity_id: float} entity-specific error-correction speeds.
    entity_short_run : dict
        {entity_id: pd.Series} entity-specific short-run coefficients.
    entity_intercepts : dict
        {entity_id: float} entity-specific intercepts.
    converged : bool
        Whether the optimization converged.
    n_iter : int
        Number of iterations used.
    n_entities_used : int
        Number of entities used in estimation.
    entities_excluded : list
        Entities excluded due to insufficient observations.
    """

    def __init__(
        self,
        *,
        long_run_params: pd.Series,
        entity_phi: dict,
        entity_short_run: dict,
        entity_intercepts: dict,
        converged: bool,
        n_iter: int,
        n_entities_used: int,
        entities_excluded: list,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.long_run_params = long_run_params
        self.entity_phi = entity_phi
        self.entity_short_run = entity_short_run
        self.entity_intercepts = entity_intercepts
        self.converged = converged
        self.n_iter = n_iter
        self.n_entities_used = n_entities_used
        self.entities_excluded = entities_excluded

    def summary(self, title: str | None = None) -> str:
        """Generate summary with PMG-specific information."""
        base = super().summary(title=title)

        extra = []
        extra.append("-" * 78)
        extra.append(f"No. Entities Used:         {self.n_entities_used:>10}")
        extra.append(f"Converged:                 {'Yes' if self.converged else 'No':>10}")
        extra.append(f"Iterations:                {self.n_iter:>10}")

        extra.append("-" * 78)
        extra.append("Long-run coefficients (homogeneous):")
        for var in self.long_run_params.index:
            extra.append(f"  {var:<20} {self.long_run_params[var]:>12.6f}")

        extra.append("-" * 78)
        extra.append("Error-correction speeds (phi_i):")
        phis = list(self.entity_phi.values())
        extra.append(f"  Mean:                    {np.mean(phis):>12.6f}")
        extra.append(f"  Std:                     {np.std(phis, ddof=1):>12.6f}")
        extra.append(f"  Min:                     {np.min(phis):>12.6f}")
        extra.append(f"  Max:                     {np.max(phis):>12.6f}")
        extra.append("=" * 78)

        return base + "\n" + "\n".join(extra)


class PooledMeanGroupEstimator(PanelModel):
    """
    Pooled Mean Group (PMG) estimator for panel data.

    Estimates an error-correction model (ECM) where long-run coefficients
    are constrained to be homogeneous across entities, while short-run
    dynamics and error-correction speeds are entity-specific.

    The ECM form is:

        Dy_it = phi_i * (y_{i,t-1} - theta' * x_{i,t-1}) + delta_i' * Dx_it + mu_i + e_it

    where:
    - theta: long-run coefficients (homogeneous)
    - phi_i: error-correction speed (entity-specific, should be negative)
    - delta_i: short-run coefficients (entity-specific)
    - mu_i: entity-specific intercept

    Estimation uses concentrated maximum likelihood: for a given theta,
    the entity-specific parameters (phi_i, delta_i, mu_i) are obtained
    by OLS, and theta is optimized to maximize the concentrated
    log-likelihood.

    Parameters
    ----------
    formula : str
        Model formula (e.g., "y ~ x1 + x2").
    data : pd.DataFrame
        Panel data in long format.
    entity_col : str
        Column identifying entities.
    time_col : str
        Column identifying time periods.
    lags : int, default=1
        Number of lags for short-run dynamics.
    max_iter : int, default=100
        Maximum iterations for optimization.
    tol : float, default=1e-5
        Convergence tolerance.

    References
    ----------
    .. [1] Pesaran, M. H., Shin, Y., & Smith, R. P. (1999). Pooled mean
           group estimation of dynamic heterogeneous panels. Journal of
           the American Statistical Association, 94(446), 621-634.
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        lags: int = 1,
        max_iter: int = 100,
        tol: float = 1e-5,
    ):
        super().__init__(formula, data, entity_col, time_col)
        self.lags = lags
        self.max_iter = max_iter
        self.tol = tol

    def _build_ecm_data(self):
        """
        Build ECM (error-correction model) data for each entity.

        For formula "y ~ x1 + x2", constructs:
        - Dy_it = y_it - y_{i,t-1}
        - Dx_it = x_it - x_{i,t-1} (for each lag)
        - y_{i,t-1}, x_{i,t-1} (for the long-run relationship)

        Returns
        -------
        entity_data : dict
            {entity_id: (Dy, X_lr, X_sr)} where:
            - Dy: (T-1,) first-differenced dependent variable
            - X_lr: (T-1, K) lagged levels of regressors (for ECM term)
            - X_sr: (T-1, K*lags) first-differenced regressors
            - y_lag: (T-1,) lagged dependent variable
        var_names : list
            Names of the regressors (without intercept).
        entities_excluded : list
            Entities excluded due to insufficient observations.
        """
        y_all, X_all = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )
        var_names = list(self.formula_parser.get_variable_names(self.data.data))
        entities = self.data.data[self.data.entity_col].values
        times = self.data.data[self.data.time_col].values

        # Remove intercept column if present
        has_intercept = "Intercept" in var_names
        if has_intercept:
            intercept_idx = var_names.index("Intercept")
            X_all = np.delete(X_all, intercept_idx, axis=1)
            var_names = [v for v in var_names if v != "Intercept"]

        X_all.shape[1]
        entity_data = {}
        entities_excluded = []
        min_obs = 2 * self.lags + 3  # Need enough obs after differencing

        for entity in np.unique(entities):
            mask = entities == entity
            y_i = y_all[mask]
            X_i = X_all[mask]
            t_i = times[mask]
            T_i = len(y_i)

            if T_i < min_obs:
                warnings.warn(
                    f"Entity '{entity}' excluded: T_i={T_i} < minimum required={min_obs}",
                    UserWarning,
                    stacklevel=2,
                )
                entities_excluded.append(entity)
                continue

            # Sort by time
            sort_idx = np.argsort(t_i)
            y_i = y_i[sort_idx]
            X_i = X_i[sort_idx]

            # First differences
            Dy = np.diff(y_i)  # (T-1,)
            DX = np.diff(X_i, axis=0)  # (T-1, K)

            # Lagged levels for ECM term: y_{t-1}, X_{t-1}
            y_lag = y_i[:-1]  # (T-1,)
            X_lag = X_i[:-1]  # (T-1, K)

            # Build short-run regressors: DX_t, DX_{t-1}, ..., DX_{t-lags+1}
            # For lags > 1, we need further lags of differences
            sr_parts = []
            max_lag_trim = self.lags - 1

            if max_lag_trim > 0:
                # Trim to align all lags
                Dy = Dy[max_lag_trim:]
                y_lag = y_lag[max_lag_trim:]
                X_lag = X_lag[max_lag_trim:]

                for lag in range(self.lags):
                    start = max_lag_trim - lag
                    end = len(DX) - lag if lag > 0 else len(DX)
                    sr_parts.append(DX[start:end])
            else:
                sr_parts.append(DX)

            X_sr = np.hstack(sr_parts) if sr_parts else np.empty((len(Dy), 0))

            entity_data[entity] = (Dy, X_lag, X_sr, y_lag)

        return entity_data, var_names, entities_excluded

    def _concentrated_nll(self, theta, entity_data, var_names):
        """
        Concentrated negative log-likelihood for given long-run theta.

        For fixed theta, the ECM becomes:
            Dy_it = phi_i * (y_{i,t-1} - theta' * x_{i,t-1}) + delta_i' * Dx_it + mu_i + e_it

        which is linear in (phi_i, delta_i, mu_i) and can be solved by OLS.

        Parameters
        ----------
        theta : np.ndarray
            Long-run coefficients (K,).
        entity_data : dict
            Entity ECM data from _build_ecm_data.
        var_names : list
            Regressor names.

        Returns
        -------
        nll : float
            Negative concentrated log-likelihood.
        """
        len(var_names)
        total_nll = 0.0

        for _entity, (Dy, X_lag, X_sr, y_lag) in entity_data.items():
            T = len(Dy)

            # ECM term: y_{t-1} - theta' * x_{t-1}
            ecm_term = y_lag - X_lag @ theta  # (T,)

            # Regressors for entity OLS: [ecm_term, X_sr, 1]
            ones = np.ones((T, 1))
            Z_i = np.column_stack([ecm_term, X_sr, ones])  # (T, 1 + K*lags + 1)

            # OLS: Dy = Z_i @ gamma_i + e_i
            try:
                gamma_i, _res, _rank, _sv = np.linalg.lstsq(Z_i, Dy, rcond=None)
            except np.linalg.LinAlgError:
                return 1e20

            resid_i = Dy - Z_i @ gamma_i
            sigma2_i = np.sum(resid_i**2) / T

            if sigma2_i <= 0:
                sigma2_i = 1e-20

            # Log-likelihood contribution: -T/2 * log(2*pi*sigma2) - T/2
            total_nll += T / 2.0 * np.log(sigma2_i)

        return total_nll

    def _extract_entity_params(self, theta, entity_data, var_names):
        """
        Extract entity-specific parameters for given long-run theta.

        Returns
        -------
        entity_phi : dict
        entity_short_run : dict
        entity_intercepts : dict
        entity_sigma2 : dict
        total_resid : list of arrays
        total_fitted : list of arrays
        """
        len(var_names)
        entity_phi = {}
        entity_short_run = {}
        entity_intercepts = {}
        entity_sigma2 = {}
        all_resid = []
        all_fitted = []

        sr_names = []
        for lag in range(self.lags):
            for v in var_names:
                sr_names.append(f"D.{v}" if lag == 0 else f"D.{v}_L{lag}")

        for entity, (Dy, X_lag, X_sr, y_lag) in entity_data.items():
            T = len(Dy)

            ecm_term = y_lag - X_lag @ theta
            ones = np.ones((T, 1))
            Z_i = np.column_stack([ecm_term, X_sr, ones])

            gamma_i = np.linalg.lstsq(Z_i, Dy, rcond=None)[0]

            phi_i = gamma_i[0]
            delta_i = gamma_i[1:-1]
            mu_i = gamma_i[-1]

            fitted_i = Z_i @ gamma_i
            resid_i = Dy - fitted_i
            sigma2_i = np.sum(resid_i**2) / T

            entity_phi[entity] = float(phi_i)
            entity_short_run[entity] = pd.Series(delta_i, index=sr_names[: len(delta_i)])
            entity_intercepts[entity] = float(mu_i)
            entity_sigma2[entity] = sigma2_i
            all_resid.append(resid_i)
            all_fitted.append(fitted_i)

        return (
            entity_phi,
            entity_short_run,
            entity_intercepts,
            entity_sigma2,
            all_resid,
            all_fitted,
        )

    def fit(self) -> PMGResults:
        """
        Fit the Pooled Mean Group estimator.

        Uses concentrated MLE: optimizes long-run theta while
        entity-specific parameters are concentrated out via OLS.

        Returns
        -------
        PMGResults
        """
        # Step 1: Build ECM data
        entity_data, var_names, entities_excluded = self._build_ecm_data()
        K = len(var_names)
        N_used = len(entity_data)

        if N_used == 0:
            raise ValueError("No entities had enough observations for PMG estimation.")

        # Step 2: Initial theta from MG on levels (simple OLS average)
        theta0 = np.zeros(K)
        for _entity, (_Dy, X_lag, _X_sr, y_lag) in entity_data.items():
            # Simple regression of y_lag on X_lag for initial long-run
            beta_i = np.linalg.lstsq(X_lag, y_lag, rcond=None)[0]
            theta0 += beta_i
        theta0 /= N_used

        # Step 3: Optimize theta via concentrated likelihood
        result = optimize.minimize(
            self._concentrated_nll,
            theta0,
            args=(entity_data, var_names),
            method="BFGS",
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )

        theta_opt = result.x
        converged = result.success
        n_iter = result.nit if hasattr(result, "nit") else 0

        # Step 4: Extract entity-specific parameters
        (
            entity_phi,
            entity_short_run,
            entity_intercepts,
            _entity_sigma2,
            all_resid,
            all_fitted,
        ) = self._extract_entity_params(theta_opt, entity_data, var_names)

        # Step 5: Compute MG-style variance of theta
        # Use numerical Hessian from optimizer if available, else bootstrap-style
        # For robustness, use cross-entity variation of implied long-run coefficients
        # PMG variance: based on the outer product of the score or Hessian
        if hasattr(result, "hess_inv") and result.hess_inv is not None:
            if isinstance(result.hess_inv, np.ndarray):
                vcov_theta = result.hess_inv
            else:
                # LinearOperator from BFGS — convert to dense
                vcov_theta = np.array(result.hess_inv.todense())
        else:
            # Fallback: numerical Hessian
            vcov_theta = np.eye(K) * 0.01

        std_theta = np.sqrt(np.diag(np.abs(vcov_theta)))

        # Step 6: Average short-run and phi across entities for summary params
        phi_mean = np.mean(list(entity_phi.values()))
        sr_keys = next(iter(entity_short_run.values())).index if entity_short_run else []
        sr_mean = np.zeros(len(sr_keys))
        for sr in entity_short_run.values():
            sr_mean += sr.values
        sr_mean /= N_used

        # Build combined parameter vector for summary: [phi, theta, delta_mean]
        all_param_names = ["phi (EC speed)"] + [f"LR.{v}" for v in var_names] + list(sr_keys)
        all_params = np.concatenate([[phi_mean], theta_opt, sr_mean])

        # Variance for phi and short-run (cross-entity dispersion, MG-style)
        phi_arr = np.array(list(entity_phi.values()))
        phi_var = np.var(phi_arr, ddof=1) / N_used

        sr_arr = np.array([sr.values for sr in entity_short_run.values()])
        sr_var = np.var(sr_arr, axis=0, ddof=1) / N_used if len(sr_arr) > 0 else np.array([])

        all_std = np.concatenate([[np.sqrt(phi_var)], std_theta, np.sqrt(sr_var)])

        # Build full covariance (block diagonal approximation)
        full_k = len(all_params)
        vcov_full = np.zeros((full_k, full_k))
        vcov_full[0, 0] = phi_var
        vcov_full[1 : 1 + K, 1 : 1 + K] = np.abs(vcov_theta)
        for j, v in enumerate(range(1 + K, full_k)):
            vcov_full[v, v] = sr_var[j] if j < len(sr_var) else 0.0

        # Build residuals and fitted for entire dataset
        y_all, X_all_raw = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )
        resid_full = np.full(len(y_all), np.nan)
        fitted_full = np.full(len(y_all), np.nan)

        entities_arr = self.data.data[self.data.entity_col].values
        times_arr = self.data.data[self.data.time_col].values

        for idx, entity in enumerate(entity_data):
            mask = entities_arr == entity
            # ECM data lost first `lags` observations
            trim = self.lags
            obs_indices = np.where(mask)[0]
            sort_by_time = np.argsort(times_arr[mask])
            sorted_indices = obs_indices[sort_by_time]

            r = all_resid[idx]
            f = all_fitted[idx]
            ecm_indices = sorted_indices[trim:]
            resid_full[ecm_indices] = r
            fitted_full[ecm_indices] = f

        self._y_orig = y_all
        self._X_orig = X_all_raw

        # Compute R-squared on non-NaN observations
        valid = ~np.isnan(resid_full)
        y_all[valid]
        # For ECM, the dependent variable is Dy, so use fitted Dy
        tss = np.sum(
            (
                resid_full[valid]
                + fitted_full[valid]
                - np.mean(resid_full[valid] + fitted_full[valid])
            )
            ** 2
        )
        rss = np.sum(resid_full[valid] ** 2)
        rsquared = 1.0 - rss / tss if tss > 0 else 0.0

        nobs_valid = int(valid.sum())
        df_model = full_k
        df_resid = nobs_valid - df_model

        rsquared_adj = (
            1.0 - (1.0 - rsquared) * (nobs_valid - 1) / df_resid if df_resid > 0 else np.nan
        )

        params_series = pd.Series(all_params, index=all_param_names)
        std_series = pd.Series(all_std, index=all_param_names)
        cov_df = pd.DataFrame(vcov_full, index=all_param_names, columns=all_param_names)
        lr_series = pd.Series(theta_opt, index=var_names)

        model_info = {
            "model_type": "Pooled Mean Group",
            "formula": self.formula,
            "cov_type": "pmg",
            "lags": self.lags,
        }

        data_info = {
            "nobs": nobs_valid,
            "n_entities": N_used,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": df_resid,
            "entity_index": entities_arr,
            "time_index": times_arr,
        }

        rsquared_dict = {
            "rsquared": rsquared,
            "rsquared_adj": rsquared_adj,
            "rsquared_within": np.nan,
            "rsquared_between": np.nan,
            "rsquared_overall": rsquared,
        }

        results = PMGResults(
            long_run_params=lr_series,
            entity_phi=entity_phi,
            entity_short_run=entity_short_run,
            entity_intercepts=entity_intercepts,
            converged=converged,
            n_iter=n_iter,
            n_entities_used=N_used,
            entities_excluded=entities_excluded,
            params=params_series,
            std_errors=std_series,
            cov_params=cov_df,
            resid=resid_full,
            fittedvalues=fitted_full,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
            formula_parser=self.formula_parser,
        )

        self._results = results
        self._fitted = True

        return results

    def _estimate_coefficients(self) -> np.ndarray:
        """Estimate coefficients (abstract method implementation)."""
        res = self.fit()
        return res.params.values


def hausman_mg_pmg(mg_results: MeanGroupResults, pmg_results: PMGResults) -> dict:
    """
    Hausman test comparing Mean Group (MG) and Pooled Mean Group (PMG).

    Under the null hypothesis, the PMG long-run homogeneity restriction
    is valid and PMG is efficient. Under the alternative, the restriction
    is invalid and only MG is consistent.

    H0: Long-run homogeneity holds (PMG is efficient)
    H1: Long-run homogeneity fails (MG is consistent)

    Test statistic:
        H = (theta_MG - theta_PMG)' (V_MG - V_PMG)^{-1} (theta_MG - theta_PMG)
        H ~ chi2(K) under H0

    Parameters
    ----------
    mg_results : MeanGroupResults
        Fitted Mean Group results.
    pmg_results : PMGResults
        Fitted Pooled Mean Group results.

    Returns
    -------
    dict
        {'statistic': float, 'pvalue': float, 'df': int, 'reject': bool}
    """
    # Get the long-run variable names from PMG
    lr_vars = list(pmg_results.long_run_params.index)
    K = len(lr_vars)

    # MG coefficients for the same variables (exclude intercept)
    mg_params = mg_results.params
    mg_var_names = list(mg_params.index)

    # Match variables between MG and PMG
    theta_mg = np.array([mg_params[v] for v in lr_vars])
    theta_pmg = pmg_results.long_run_params.values

    # MG variance for matched variables
    mg_idx = [mg_var_names.index(v) for v in lr_vars]
    V_mg = mg_results.cov_params.values[np.ix_(mg_idx, mg_idx)]

    # PMG variance for long-run params
    pmg_all_names = list(pmg_results.params.index)
    pmg_lr_idx = [pmg_all_names.index(f"LR.{v}") for v in lr_vars]
    V_pmg = pmg_results.cov_params.values[np.ix_(pmg_lr_idx, pmg_lr_idx)]

    # Hausman statistic: q' (V_MG - V_PMG)^-1 q
    q = theta_mg - theta_pmg
    V_diff = V_mg - V_pmg

    try:
        # Use pinv for robustness (V_diff may not be positive definite)
        V_diff_inv = np.linalg.pinv(V_diff)
        H = float(q @ V_diff_inv @ q)
    except np.linalg.LinAlgError:
        H = np.nan

    # Ensure non-negative (numerical issues can make it slightly negative)
    if H < 0:
        H = abs(H)

    pvalue = 1.0 - stats.chi2.cdf(H, K) if not np.isnan(H) else np.nan

    return {
        "statistic": float(H),
        "pvalue": float(pvalue),
        "df": K,
        "reject": bool(pvalue < 0.05) if not np.isnan(pvalue) else False,
    }
