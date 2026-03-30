"""
Seemingly Unrelated Regressions (SUR) estimator for panel data.

This module provides the SUR estimator which treats each entity as an equation
in a system of seemingly unrelated regressions. The estimator accounts for
cross-equation error correlations via Feasible Generalized Least Squares (FGLS).

References
----------
.. [1] Zellner, A. (1962). An Efficient Method of Estimating Seemingly
       Unrelated Regressions and Tests for Aggregation Bias. Journal of the
       American Statistical Association, 57(298), 348-368.
.. [2] Zellner, A. & Huang, D. S. (1962). Further Properties of Efficient
       Estimators for Seemingly Unrelated Regression Equations. International
       Economic Review, 3(3), 300-313.
.. [3] McElroy, M. B. (1977). Goodness of Fit for Seemingly Unrelated
       Regressions. Journal of Econometrics, 6(3), 381-387.
.. [4] Breusch, T. S. & Pagan, A. R. (1980). The Lagrange Multiplier Test
       and its Applications to Model Specification in Econometrics. Review of
       Economic Studies, 47(1), 239-253.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.core.base_model import PanelModel
from panelbox.core.formula_parser import FormulaParser
from panelbox.core.results import PanelResults

logger = logging.getLogger(__name__)


class SURResults(PanelResults):
    """
    Results container for SUR estimation.

    Extends PanelResults with SUR-specific attributes.

    Attributes
    ----------
    entity_params : dict
        Per-entity coefficient estimates {entity_id: pd.Series}.
    entity_std_errors : dict
        Per-entity standard errors {entity_id: pd.Series}.
    entity_rsquared : dict
        Per-entity R-squared {entity_id: float}.
    sigma_matrix : np.ndarray
        Estimated cross-equation covariance matrix (N x N).
    correlation_matrix : np.ndarray
        Cross-equation correlation matrix (N x N).
    system_rsquared : float
        McElroy system R-squared.
    n_iterations : int
        Number of FGLS iterations performed (0 if not iterated).
    converged : bool
        Whether iterated SUR converged.
    ols_params : dict
        Pre-GLS OLS coefficient estimates {entity_id: pd.Series}.
    efficiency_gain : dict
        Ratio of SUR SE to OLS SE {entity_id: pd.Series}.
    bp_independence_test : dict
        Breusch-Pagan independence test results.
    """

    def __init__(
        self,
        entity_params,
        entity_std_errors,
        entity_rsquared,
        sigma_matrix,
        correlation_matrix,
        system_rsquared,
        n_iterations,
        converged,
        ols_params,
        efficiency_gain,
        bp_independence_test,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.entity_params = entity_params
        self.entity_std_errors = entity_std_errors
        self.entity_rsquared = entity_rsquared
        self.sigma_matrix = sigma_matrix
        self.correlation_matrix = correlation_matrix
        self.system_rsquared = system_rsquared
        self.n_iterations = n_iterations
        self.converged = converged
        self.ols_params = ols_params
        self.efficiency_gain = efficiency_gain
        self.bp_independence_test = bp_independence_test

    def system_summary(self) -> str:
        """
        Generate system-level summary with Sigma matrix and tests.

        Returns
        -------
        str
            Formatted system summary.
        """
        lines = []
        entities = list(self.entity_params.keys())
        N = len(entities)

        lines.append("=" * 78)
        lines.append("SUR System Summary".center(78))
        lines.append("=" * 78)

        # Model info
        lines.append(f"Formula: {self.formula}")
        lines.append(f"No. Equations:             {N:>10}")
        lines.append(f"No. Observations:          {self.nobs:>10,}")
        if self.n_iterations > 0:
            lines.append(f"Iterations:                {self.n_iterations:>10}")
            lines.append(f"Converged:                 {'Yes' if self.converged else 'No':>10}")
        lines.append(f"System R² (McElroy):       {self.system_rsquared:>10.4f}")
        lines.append("-" * 78)

        # Cross-equation covariance matrix
        lines.append("Cross-equation Covariance Matrix (Sigma):")
        entity_labels = [str(e) for e in entities]
        header = "         " + "  ".join(f"{e:>10}" for e in entity_labels)
        lines.append(header)
        for i, ei in enumerate(entity_labels):
            row = f"{ei:<9}" + "  ".join(f"{self.sigma_matrix[i, j]:>10.4f}" for j in range(N))
            lines.append(row)
        lines.append("")

        # Cross-equation correlation matrix
        lines.append("Cross-equation Correlation Matrix:")
        lines.append(header)
        for i, ei in enumerate(entity_labels):
            row = f"{ei:<9}" + "  ".join(
                f"{self.correlation_matrix[i, j]:>10.4f}" for j in range(N)
            )
            lines.append(row)
        lines.append("")

        # Breusch-Pagan test
        bp = self.bp_independence_test
        lines.append("Breusch-Pagan Independence Test:")
        lines.append("  H0: Cross-equation errors are independent (Sigma is diagonal)")
        lines.append(f"  LM statistic:            {bp['statistic']:>10.4f}")
        lines.append(f"  p-value:                 {bp['pvalue']:>10.4f}")
        lines.append(f"  df:                      {bp['df']:>10}")
        lines.append("")

        # Per-equation R-squared
        lines.append("Per-equation R²:")
        for entity in entities:
            lines.append(f"  {entity!s:<20} {self.entity_rsquared[entity]:>10.4f}")
        lines.append("")

        # Efficiency gain
        lines.append("Efficiency Gain (SE_SUR / SE_OLS):")
        lines.append("  Values < 1 indicate SUR is more efficient than OLS.")
        for entity in entities:
            gains = self.efficiency_gain[entity]
            lines.append(f"  Entity: {entity}")
            for var in gains.index:
                lines.append(f"    {var:<15} {gains[var]:>10.4f}")
        lines.append("")

        lines.append("=" * 78)
        return "\n".join(lines)

    def equation_summary(self, entity_id) -> str:
        """
        Generate summary for a single equation.

        Parameters
        ----------
        entity_id : hashable
            Entity identifier.

        Returns
        -------
        str
            Formatted equation summary.
        """
        if entity_id not in self.entity_params:
            raise ValueError(
                f"Entity '{entity_id}' not found. "
                f"Available entities: {list(self.entity_params.keys())}"
            )

        params = self.entity_params[entity_id]
        se = self.entity_std_errors[entity_id]
        r2 = self.entity_rsquared[entity_id]
        ols_p = self.ols_params[entity_id]
        gain = self.efficiency_gain[entity_id]

        lines = []
        lines.append("=" * 78)
        lines.append(f"SUR Equation Summary: Entity = {entity_id}".center(78))
        lines.append("=" * 78)
        lines.append(f"R-squared:                 {r2:>10.4f}")
        lines.append("-" * 78)

        lines.append(
            f"{'Variable':<15} {'SUR Coef.':<12} {'SUR SE':<12} {'OLS Coef.':<12} {'Gain':<8}"
        )
        lines.append("-" * 78)

        for var in params.index:
            lines.append(
                f"{var:<15} {params[var]:>11.4f} {se[var]:>11.4f} "
                f"{ols_p[var]:>11.4f} {gain[var]:>7.4f}"
            )

        lines.append("=" * 78)
        return "\n".join(lines)

    def plot_correlation_matrix(self, ax=None):
        """
        Plot heatmap of cross-equation correlation matrix.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the heatmap.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("matplotlib is required for plotting.") from err

        entities = list(self.entity_params.keys())
        labels = [str(e) for e in entities]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(self.correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title("Cross-equation Correlation Matrix")

        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{self.correlation_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        plt.colorbar(im, ax=ax)
        return ax


class SUR(PanelModel):
    """
    Seemingly Unrelated Regressions (SUR) estimator for panel data.

    Each entity is treated as a separate equation. The SUR estimator first
    runs OLS equation-by-equation to estimate the cross-equation error
    covariance matrix (Sigma), then applies FGLS using the Kronecker
    structure of the covariance without materializing the full matrix.

    Parameters
    ----------
    formula : str or dict
        Model formula in R-style syntax (e.g., "y ~ x1 + x2").
        If str, applied to all entities.
        If dict, maps entity identifiers to per-entity formulas, e.g.
        ``{'firm1': 'invest ~ value + capital', 'firm2': 'invest ~ value'}``.
    data : pd.DataFrame
        Panel data in long format.
    entity_col : str
        Column identifying entities (each entity = one equation).
    time_col : str
        Column identifying time periods.
    homogeneous : bool, default=False
        If True, constrain all entities to share the same coefficients.
        Only valid when formula is a str (same formula for all entities).
    iterate : bool, default=False
        If True, iterate FGLS until convergence (ISUR, equivalent to MLE
        under normality).
    max_iter : int, default=100
        Maximum iterations for iterated SUR.
    tol : float, default=1e-6
        Convergence tolerance for iterated SUR (relative change in beta).

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_grunfeld()
    >>> sur = pb.SUR("invest ~ value + capital", data, "firm", "year")
    >>> res = sur.fit()
    >>> print(res.summary())

    Heterogeneous formulas per entity:

    >>> sur = pb.SUR(
    ...     formula={"firm1": "invest ~ value + capital", "firm2": "invest ~ value"},
    ...     data=data,
    ...     entity_col="firm",
    ...     time_col="year",
    ... )
    """

    def __init__(
        self,
        formula: str | dict,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        homogeneous: bool = False,
        iterate: bool = False,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        # Handle dict formulas: validate and store, then pass a representative
        # formula string to the base class
        self._formula_dict = None
        self._entity_parsers = None

        if isinstance(formula, dict):
            if not formula:
                raise ValueError("formula dict must not be empty.")
            # Validate entities exist in data
            data_entities = set(data[entity_col].unique())
            for entity_key in formula:
                if entity_key not in data_entities:
                    raise ValueError(
                        f"Entity '{entity_key}' in formula dict not found in data. "
                        f"Available entities: {sorted(data_entities)}"
                    )
            self._formula_dict = formula
            # Use first formula as representative for base class
            representative_formula = next(iter(formula.values()))
            super().__init__(representative_formula, data, entity_col, time_col)
            # Build per-entity parsers
            self._entity_parsers = {}
            for entity_key, entity_formula in formula.items():
                self._entity_parsers[entity_key] = FormulaParser(entity_formula).parse()
            # Store the dict as the formula attribute for display
            self.formula = formula
        else:
            super().__init__(formula, data, entity_col, time_col)

        self.homogeneous = homogeneous
        self.iterate = iterate
        self.max_iter = max_iter
        self.tol = tol

    def _prepare_entity_data(self):
        """
        Prepare per-entity design matrices and dependent variables.

        When formula is a dict, each entity gets its own design matrix
        built from its specific formula. When formula is a str, all entities
        share the same design matrix structure.

        Returns
        -------
        entity_data : dict
            Keys are entity identifiers, values are dicts with
            'y' (T_i,), 'X' (T_i, k_i), 'time' arrays, 'var_names' list.
        entities : list
            Ordered list of entity identifiers.
        """
        entities_col = self.data.data[self.data.entity_col].values
        time_col = self.data.data[self.data.time_col].values
        unique_entities = sorted(np.unique(entities_col).tolist())

        if self._formula_dict is not None:
            # Heterogeneous formulas: only include entities in the dict
            # (entities not in dict are excluded from estimation)
            dict_entities = [e for e in unique_entities if e in self._formula_dict]
            entity_data = {}
            for entity in dict_entities:
                mask = entities_col == entity
                entity_df = self.data.data[mask]
                parser = self._entity_parsers[entity]
                y_i, X_i = parser.build_design_matrices(entity_df, return_type="array")
                var_names_i = parser.get_variable_names(entity_df)
                entity_data[entity] = {
                    "y": y_i.ravel(),
                    "X": X_i,
                    "time": time_col[mask],
                    "var_names": var_names_i,
                }
            return entity_data, dict_entities
        else:
            # Homogeneous formula: same design matrix for all entities
            y_all, X_all = self.formula_parser.build_design_matrices(
                self.data.data, return_type="array"
            )
            var_names = self.formula_parser.get_variable_names(self.data.data)
            entity_data = {}
            for entity in unique_entities:
                mask = entities_col == entity
                entity_data[entity] = {
                    "y": y_all[mask].ravel(),
                    "X": X_all[mask],
                    "time": time_col[mask],
                    "var_names": var_names,
                }
            return entity_data, unique_entities

    def _estimate_ols_by_entity(self, entity_data):
        """
        Run OLS equation-by-equation.

        Parameters
        ----------
        entity_data : dict
            Per-entity data from _prepare_entity_data.

        Returns
        -------
        ols_results : dict
            Per-entity results with keys 'beta', 'resid', 'fitted'.
        """
        ols_results = {}
        for entity, edata in entity_data.items():
            y_i = edata["y"]
            X_i = edata["X"]
            # OLS: beta_i = (X_i'X_i)^{-1} X_i'y_i
            beta_i = np.linalg.lstsq(X_i, y_i, rcond=None)[0]
            fitted_i = X_i @ beta_i
            resid_i = y_i - fitted_i
            ols_results[entity] = {
                "beta": beta_i,
                "resid": resid_i,
                "fitted": fitted_i,
            }
        return ols_results

    def _estimate_sigma(self, entity_data, ols_results, entities):
        """
        Estimate cross-equation covariance matrix Sigma.

        For balanced panels: sigma_ij = (1/T) * e_i' * e_j
        For unbalanced panels: uses common time periods between entities.

        Parameters
        ----------
        entity_data : dict
            Per-entity data.
        ols_results : dict
            Per-entity OLS results with residuals.
        entities : list
            Ordered entity identifiers.

        Returns
        -------
        sigma : np.ndarray
            N x N covariance matrix.
        """
        N = len(entities)
        sigma = np.zeros((N, N))

        for i in range(N):
            ei = entities[i]
            time_i = entity_data[ei]["time"]
            resid_i = ols_results[ei]["resid"]

            for j in range(i, N):
                ej = entities[j]
                time_j = entity_data[ej]["time"]
                resid_j = ols_results[ej]["resid"]

                # Find common time periods
                common_times, idx_i, idx_j = np.intersect1d(time_i, time_j, return_indices=True)
                T_ij = len(common_times)

                if T_ij > 0:
                    sigma[i, j] = resid_i[idx_i] @ resid_j[idx_j] / T_ij
                else:
                    sigma[i, j] = 0.0

                sigma[j, i] = sigma[i, j]

        return sigma

    def _fgls_estimation(self, sigma, entity_data, entities):
        """
        FGLS estimation using Kronecker structure efficiently.

        Computes beta_SUR = (X'(Sigma^{-1} kron I)X)^{-1} X'(Sigma^{-1} kron I)y
        without materializing the full (NT x NT) matrix.

        Parameters
        ----------
        sigma : np.ndarray
            N x N cross-equation covariance matrix.
        entity_data : dict
            Per-entity data.
        entities : list
            Ordered entity identifiers.

        Returns
        -------
        beta_sur : dict
            Per-entity coefficient vectors.
        vcov : np.ndarray
            Full system covariance matrix of coefficients.
        beta_vec : np.ndarray
            Stacked coefficient vector.
        """
        N = len(entities)
        sigma_inv = np.linalg.pinv(sigma)

        # Determine block sizes (k_i per entity)
        k_list = [entity_data[e]["X"].shape[1] for e in entities]
        total_k = sum(k_list)
        offsets = np.cumsum([0, *k_list])

        # Build system matrices
        A = np.zeros((total_k, total_k))
        b = np.zeros(total_k)

        for i in range(N):
            ei = entities[i]
            X_i = entity_data[ei]["X"]
            entity_data[ei]["y"]
            time_i = entity_data[ei]["time"]

            for j in range(N):
                ej = entities[j]
                X_j = entity_data[ej]["X"]
                y_j = entity_data[ej]["y"]
                time_j = entity_data[ej]["time"]

                sij = sigma_inv[i, j]
                if abs(sij) < 1e-15:
                    continue

                # For unbalanced panels, align on common time periods
                common_times, idx_i, idx_j = np.intersect1d(time_i, time_j, return_indices=True)
                if len(common_times) == 0:
                    continue

                Xi_c = X_i[idx_i]
                Xj_c = X_j[idx_j]
                yj_c = y_j[idx_j]

                # Accumulate block (i, j) of A
                A[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] += sij * (Xi_c.T @ Xj_c)

                # Accumulate block i of b (only once per j)
                b[offsets[i] : offsets[i + 1]] += sij * (Xi_c.T @ yj_c)

        # Solve system
        vcov = np.linalg.pinv(A)
        beta_vec = vcov @ b

        # Unpack per-entity betas
        beta_sur = {}
        for i, entity in enumerate(entities):
            beta_sur[entity] = beta_vec[offsets[i] : offsets[i + 1]]

        return beta_sur, vcov, beta_vec

    @staticmethod
    def _compute_correlation_matrix(sigma):
        """
        Compute correlation matrix from covariance matrix.

        Parameters
        ----------
        sigma : np.ndarray
            N x N covariance matrix.

        Returns
        -------
        corr : np.ndarray
            N x N correlation matrix.
        """
        d = np.sqrt(np.diag(sigma))
        d[d == 0] = 1.0  # avoid division by zero
        corr = sigma / np.outer(d, d)
        np.fill_diagonal(corr, 1.0)
        return corr

    @staticmethod
    def _breusch_pagan_test(sigma, T):
        """
        Breusch-Pagan Lagrange Multiplier test for cross-equation independence.

        H0: Sigma is diagonal (equations are independent).

        Parameters
        ----------
        sigma : np.ndarray
            N x N cross-equation covariance matrix.
        T : int
            Number of common time periods.

        Returns
        -------
        dict
            Test results with 'statistic', 'pvalue', 'df'.
        """
        N = sigma.shape[0]
        d = np.sqrt(np.diag(sigma))
        d[d == 0] = 1.0

        # LM = T * sum_{i>j} r_ij^2
        lm_stat = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                r_ij = sigma[i, j] / (d[i] * d[j])
                lm_stat += r_ij**2

        lm_stat *= T
        df = N * (N - 1) // 2
        pvalue = 1.0 - stats.chi2.cdf(lm_stat, df) if df > 0 else np.nan

        return {"statistic": float(lm_stat), "pvalue": float(pvalue), "df": int(df)}

    @staticmethod
    def _compute_system_rsquared(sigma, entity_data, beta_sur, entities):
        """
        Compute McElroy (1977) system R-squared.

        R²_sys = 1 - (e'(Sigma^{-1} kron I)e) / (y~'(Sigma^{-1} kron I)y~)

        where y~ = y - mean(y) per equation.

        Parameters
        ----------
        sigma : np.ndarray
            N x N cross-equation covariance matrix.
        entity_data : dict
            Per-entity data.
        beta_sur : dict
            Per-entity SUR coefficient vectors.
        entities : list
            Ordered entity identifiers.

        Returns
        -------
        float
            System R-squared value.
        """
        N = len(entities)
        sigma_inv = np.linalg.pinv(sigma)

        # Numerator: e'(Sigma^{-1} kron I)e
        # Denominator: y~'(Sigma^{-1} kron I)y~
        numerator = 0.0
        denominator = 0.0

        for i in range(N):
            ei = entities[i]
            X_i = entity_data[ei]["X"]
            y_i = entity_data[ei]["y"]
            e_i = y_i - X_i @ beta_sur[ei]
            yt_i = y_i - np.mean(y_i)
            time_i = entity_data[ei]["time"]

            for j in range(N):
                ej = entities[j]
                X_j = entity_data[ej]["X"]
                y_j = entity_data[ej]["y"]
                e_j = y_j - X_j @ beta_sur[ej]
                yt_j = y_j - np.mean(y_j)
                time_j = entity_data[ej]["time"]

                sij_inv = sigma_inv[i, j]
                if abs(sij_inv) < 1e-15:
                    continue

                # Align on common time periods
                common_times, idx_i, idx_j = np.intersect1d(time_i, time_j, return_indices=True)
                if len(common_times) == 0:
                    continue

                numerator += sij_inv * (e_i[idx_i] @ e_j[idx_j])
                denominator += sij_inv * (yt_i[idx_i] @ yt_j[idx_j])

        if denominator == 0:
            return 0.0

        r2_sys = 1.0 - numerator / denominator
        return float(r2_sys)

    def _fgls_homogeneous(self, sigma, entity_data, entities):
        """
        FGLS estimation with homogeneous (pooled) slopes.

        All entities share the same coefficient vector beta. The estimator is:
        beta = (X'(Sigma^{-1} kron I)X)^{-1} X'(Sigma^{-1} kron I)y

        where X is the same design matrix replicated for each entity.

        Parameters
        ----------
        sigma : np.ndarray
            N x N cross-equation covariance matrix.
        entity_data : dict
            Per-entity data.
        entities : list
            Ordered entity identifiers.

        Returns
        -------
        beta : np.ndarray
            Common coefficient vector (K,).
        vcov : np.ndarray
            Covariance matrix of coefficients (K x K).
        """
        N = len(entities)
        sigma_inv = np.linalg.pinv(sigma)

        K = entity_data[entities[0]]["X"].shape[1]
        A = np.zeros((K, K))
        b = np.zeros(K)

        for i in range(N):
            ei = entities[i]
            X_i = entity_data[ei]["X"]
            entity_data[ei]["y"]
            time_i = entity_data[ei]["time"]

            for j in range(N):
                ej = entities[j]
                X_j = entity_data[ej]["X"]
                y_j = entity_data[ej]["y"]
                time_j = entity_data[ej]["time"]

                sij = sigma_inv[i, j]
                if abs(sij) < 1e-15:
                    continue

                common_times, idx_i, idx_j = np.intersect1d(time_i, time_j, return_indices=True)
                if len(common_times) == 0:
                    continue

                Xi_c = X_i[idx_i]
                Xj_c = X_j[idx_j]
                yj_c = y_j[idx_j]

                A += sij * (Xi_c.T @ Xj_c)
                b += sij * (Xi_c.T @ yj_c)

        vcov = np.linalg.pinv(A)
        beta = vcov @ b
        return beta, vcov

    def fit(self, cov_type: str = "sur", **cov_kwds) -> SURResults:  # noqa: C901
        """
        Fit the SUR model.

        Steps:
        1. OLS equation-by-equation to get residuals.
        2. Estimate Sigma from residuals.
        3. FGLS using Kronecker structure.
        4. Optionally iterate until convergence (ISUR).

        Parameters
        ----------
        cov_type : str, default='sur'
            Covariance type. Currently only 'sur' is supported.
        **cov_kwds
            Additional covariance keyword arguments (reserved for future use).

        Returns
        -------
        SURResults
            Fitted model results.
        """
        entity_data, entities = self._prepare_entity_data()
        N = len(entities)

        # Get per-entity variable names
        entity_var_names = {}
        for entity in entities:
            entity_var_names[entity] = entity_data[entity]["var_names"]

        # Step 1: OLS by entity
        ols_results = self._estimate_ols_by_entity(entity_data)

        # Store OLS params for comparison
        ols_params = {}
        for entity in entities:
            vn = entity_var_names[entity]
            ols_params[entity] = pd.Series(ols_results[entity]["beta"], index=vn)

        # Compute OLS standard errors for efficiency gain comparison
        ols_std_errors = {}
        for entity in entities:
            vn = entity_var_names[entity]
            X_i = entity_data[entity]["X"]
            resid_i = ols_results[entity]["resid"]
            T_i = len(resid_i)
            k_i = X_i.shape[1]
            s2_i = np.sum(resid_i**2) / (T_i - k_i) if T_i > k_i else np.sum(resid_i**2)
            XtX_inv = np.linalg.pinv(X_i.T @ X_i)
            ols_se_i = np.sqrt(np.diag(s2_i * XtX_inv))
            ols_std_errors[entity] = pd.Series(ols_se_i, index=vn)

        # Step 2: Estimate Sigma
        sigma = self._estimate_sigma(entity_data, ols_results, entities)

        # Step 3: FGLS — branch on homogeneous vs heterogeneous
        if self.homogeneous:
            beta_pooled, vcov = self._fgls_homogeneous(sigma, entity_data, entities)
            # All entities share the same beta
            beta_sur = {e: beta_pooled.copy() for e in entities}
            beta_vec = beta_pooled
        else:
            beta_sur, vcov, beta_vec = self._fgls_estimation(sigma, entity_data, entities)

        # Step 4: Iterate if requested
        n_iterations = 1
        converged = not self.iterate

        if self.iterate:
            prev_beta = beta_vec.copy()

            for iteration in range(2, self.max_iter + 1):
                # Update residuals with FGLS betas
                fgls_results = {}
                for entity in entities:
                    X_i = entity_data[entity]["X"]
                    y_i = entity_data[entity]["y"]
                    fitted_i = X_i @ beta_sur[entity]
                    resid_i = y_i - fitted_i
                    fgls_results[entity] = {
                        "beta": beta_sur[entity],
                        "resid": resid_i,
                        "fitted": fitted_i,
                    }

                # Re-estimate Sigma
                sigma = self._estimate_sigma(entity_data, fgls_results, entities)

                # Re-estimate FGLS
                if self.homogeneous:
                    beta_pooled, vcov = self._fgls_homogeneous(sigma, entity_data, entities)
                    beta_sur = {e: beta_pooled.copy() for e in entities}
                    beta_vec = beta_pooled
                else:
                    beta_sur, vcov, beta_vec = self._fgls_estimation(sigma, entity_data, entities)

                # Check convergence on beta
                beta_change = np.max(np.abs(beta_vec - prev_beta)) / (1 + np.max(np.abs(prev_beta)))
                if beta_change < self.tol:
                    converged = True
                    n_iterations = iteration
                    break

                prev_beta = beta_vec.copy()
                n_iterations = iteration

            if not converged:
                logger.warning("Iterated SUR did not converge after %d iterations", self.max_iter)

        # Build entity_params and entity_std_errors
        entity_params = {}
        entity_std_errors = {}

        if self.homogeneous:
            # All entities share the same beta and vcov
            vn = entity_var_names[entities[0]]
            se_vec = np.sqrt(np.diag(vcov))
            for entity in entities:
                entity_params[entity] = pd.Series(beta_vec.copy(), index=vn)
                entity_std_errors[entity] = pd.Series(se_vec, index=vn)
        else:
            k_list = [entity_data[e]["X"].shape[1] for e in entities]
            offsets = np.cumsum([0, *k_list])
            for i, entity in enumerate(entities):
                vn = entity_var_names[entity]
                entity_params[entity] = pd.Series(beta_sur[entity], index=vn)
                se_block = np.sqrt(
                    np.diag(vcov[offsets[i] : offsets[i + 1], offsets[i] : offsets[i + 1]])
                )
                entity_std_errors[entity] = pd.Series(se_block, index=vn)

        # Compute per-entity R-squared
        entity_rsquared = {}
        for entity in entities:
            X_i = entity_data[entity]["X"]
            y_i = entity_data[entity]["y"]
            fitted_i = X_i @ beta_sur[entity]
            resid_i = y_i - fitted_i
            tss_i = np.sum((y_i - np.mean(y_i)) ** 2)
            rss_i = np.sum(resid_i**2)
            entity_rsquared[entity] = 1.0 - rss_i / tss_i if tss_i > 0 else 0.0

        # Efficiency gain: SE_SUR / SE_OLS (only when var_names match)
        efficiency_gain = {}
        for entity in entities:
            if list(entity_std_errors[entity].index) == list(ols_std_errors[entity].index):
                efficiency_gain[entity] = entity_std_errors[entity] / ols_std_errors[entity]
            else:
                # Heterogeneous formulas: compute gain on common variables
                common_vars = entity_std_errors[entity].index.intersection(
                    ols_std_errors[entity].index
                )
                if len(common_vars) > 0:
                    efficiency_gain[entity] = (
                        entity_std_errors[entity][common_vars] / ols_std_errors[entity][common_vars]
                    )
                else:
                    efficiency_gain[entity] = pd.Series(dtype=float)

        # Compute residuals and fitted values for all observations
        entities_col = self.data.data[self.data.entity_col].values
        n_obs = len(self.data.data)
        fitted_all = np.zeros(n_obs)
        resid_all = np.zeros(n_obs)

        for entity in entities:
            mask = entities_col == entity
            X_i = entity_data[entity]["X"]
            y_i = entity_data[entity]["y"]
            fitted_all[mask] = X_i @ beta_sur[entity]
            resid_all[mask] = y_i - X_i @ beta_sur[entity]

        # System-level params: stack all entity betas
        all_param_names = []
        if self.homogeneous:
            vn = entity_var_names[entities[0]]
            all_param_names = list(vn)
        else:
            for entity in entities:
                for v in entity_var_names[entity]:
                    all_param_names.append(f"{entity}_{v}")

        params = pd.Series(beta_vec, index=all_param_names)
        std_errors_vec = np.sqrt(np.diag(vcov))
        std_errors = pd.Series(std_errors_vec, index=all_param_names)
        cov_params_df = pd.DataFrame(vcov, index=all_param_names, columns=all_param_names)

        # Correlation matrix
        correlation_matrix = self._compute_correlation_matrix(sigma)

        # Breusch-Pagan test — use minimum common T across all pairs
        T_list = [len(entity_data[e]["y"]) for e in entities]
        T_common = int(np.min(T_list))
        bp_test = self._breusch_pagan_test(sigma, T_common)

        # McElroy system R-squared
        system_rsquared = self._compute_system_rsquared(sigma, entity_data, beta_sur, entities)

        # R-squared (standard) — use stacked y from entity_data
        y_stacked = np.concatenate([entity_data[e]["y"] for e in entities])
        tss = np.sum((y_stacked - y_stacked.mean()) ** 2)
        rss = np.sum(resid_all**2)
        rsquared = 1.0 - rss / tss if tss > 0 else 0.0

        total_k = len(all_param_names)
        df_model = total_k - (1 if self.homogeneous else N)  # subtract intercepts
        df_resid = n_obs - total_k

        rsquared_adj = 1.0 - (1.0 - rsquared) * (n_obs - 1) / df_resid if df_resid > 0 else 0.0

        formula_display = self.formula
        model_type = "SUR (Seemingly Unrelated Regressions)"
        if self.homogeneous:
            model_type += " — Homogeneous"

        model_info = {
            "model_type": model_type,
            "formula": formula_display,
            "cov_type": cov_type,
            "cov_kwds": cov_kwds,
            "entity_effects": False,
            "time_effects": False,
        }

        data_info = {
            "nobs": n_obs,
            "n_entities": N,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": df_resid,
            "entity_index": np.array(entities),
            "time_index": None,
        }

        rsquared_dict = {
            "rsquared": rsquared,
            "rsquared_adj": rsquared_adj,
            "rsquared_within": np.nan,
            "rsquared_between": np.nan,
            "rsquared_overall": rsquared,
        }

        results = SURResults(
            entity_params=entity_params,
            entity_std_errors=entity_std_errors,
            entity_rsquared=entity_rsquared,
            sigma_matrix=sigma,
            correlation_matrix=correlation_matrix,
            system_rsquared=system_rsquared,
            n_iterations=n_iterations,
            converged=converged,
            ols_params=ols_params,
            efficiency_gain=efficiency_gain,
            bp_independence_test=bp_test,
            params=params,
            std_errors=std_errors,
            cov_params=cov_params_df,
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

    def lr_test_homogeneity(self):
        """
        Likelihood Ratio test for slope homogeneity.

        Tests H0: beta_i = beta (slopes equal across entities)
        against H1: slopes differ across entities.

        LR = T * (log|Sigma_restricted| - log|Sigma_unrestricted|)

        Under H0, LR ~ chi2((N-1)*K) where K is the number of regressors
        (including intercept) and N is the number of entities.

        Returns
        -------
        dict
            Test results with 'statistic', 'pvalue', 'df', 'reject_h0'.

        Raises
        ------
        ValueError
            If formula is a dict (heterogeneous formulas cannot be pooled).
        """
        if self._formula_dict is not None:
            raise ValueError(
                "LR homogeneity test requires the same formula for all entities. "
                "Cannot test with heterogeneous (dict) formulas."
            )

        entity_data, entities = self._prepare_entity_data()
        N = len(entities)
        K = entity_data[entities[0]]["X"].shape[1]
        T_list = [len(entity_data[e]["y"]) for e in entities]
        T = int(np.min(T_list))

        # Unrestricted model: separate betas per entity
        ols_unrestricted = self._estimate_ols_by_entity(entity_data)
        sigma_unrestricted = self._estimate_sigma(entity_data, ols_unrestricted, entities)

        # Restricted model: same beta for all entities (homogeneous)
        # First estimate sigma from OLS, then do homogeneous FGLS,
        # then re-estimate sigma from restricted residuals
        beta_pooled, _ = self._fgls_homogeneous(sigma_unrestricted, entity_data, entities)
        restricted_results = {}
        for entity in entities:
            X_i = entity_data[entity]["X"]
            y_i = entity_data[entity]["y"]
            fitted_i = X_i @ beta_pooled
            resid_i = y_i - fitted_i
            restricted_results[entity] = {
                "beta": beta_pooled,
                "resid": resid_i,
                "fitted": fitted_i,
            }
        sigma_restricted = self._estimate_sigma(entity_data, restricted_results, entities)

        # LR statistic
        log_det_restricted = np.linalg.slogdet(sigma_restricted)[1]
        log_det_unrestricted = np.linalg.slogdet(sigma_unrestricted)[1]
        lr_stat = T * (log_det_restricted - log_det_unrestricted)
        lr_stat = max(lr_stat, 0.0)  # numerical safety

        df = (N - 1) * K
        pvalue = 1.0 - stats.chi2.cdf(lr_stat, df) if df > 0 else np.nan

        return {
            "statistic": float(lr_stat),
            "pvalue": float(pvalue),
            "df": int(df),
            "reject_h0": float(pvalue) < 0.05 if not np.isnan(pvalue) else None,
        }

    def _estimate_coefficients(self) -> np.ndarray:
        """Estimate coefficients (abstract method implementation)."""
        entity_data, entities = self._prepare_entity_data()
        ols_results = self._estimate_ols_by_entity(entity_data)
        sigma = self._estimate_sigma(entity_data, ols_results, entities)
        if self.homogeneous:
            beta, _ = self._fgls_homogeneous(sigma, entity_data, entities)
            return beta
        _, _, beta_vec = self._fgls_estimation(sigma, entity_data, entities)
        return beta_vec
