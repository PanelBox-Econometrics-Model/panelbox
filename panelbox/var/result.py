"""
Results containers for Panel VAR models.

This module provides result classes for Panel VAR estimation and lag selection.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.var.inference import WaldTestResult, f_test_exclusion, wald_test
from panelbox.visualization.var_plots import plot_stability as _plot_stability


class PanelVARResult:
    """
    Results container for Panel VAR estimation.

    This class stores estimation results and provides methods for inference,
    testing, and reporting.

    Parameters
    ----------
    params_by_eq : List[np.ndarray]
        List of coefficient arrays, one per equation
    std_errors_by_eq : List[np.ndarray]
        List of standard error arrays, one per equation
    cov_by_eq : List[np.ndarray]
        List of covariance matrices, one per equation
    resid_by_eq : List[np.ndarray]
        List of residual arrays, one per equation
    fitted_by_eq : List[np.ndarray]
        List of fitted value arrays, one per equation
    endog_names : List[str]
        Names of endogenous variables
    exog_names : List[str]
        Names of exogenous variables (including lags and deterministic terms)
    model_info : Dict[str, Any]
        Model information
    data_info : Dict[str, Any]
        Data information

    Attributes
    ----------
    K : int
        Number of endogenous variables
    p : int
        Number of lags
    N : int
        Number of entities
    n_obs : int
        Number of observations
    params_by_eq : List[np.ndarray]
        Coefficients by equation
    A_matrices : List[np.ndarray]
        Coefficient matrices A_1, A_2, ..., A_p (each K×K)
    """

    def __init__(
        self,
        params_by_eq: List[np.ndarray],
        std_errors_by_eq: List[np.ndarray],
        cov_by_eq: List[np.ndarray],
        resid_by_eq: List[np.ndarray],
        fitted_by_eq: List[np.ndarray],
        endog_names: List[str],
        exog_names: List[str],
        model_info: Dict[str, Any],
        data_info: Dict[str, Any],
    ):
        self.params_by_eq = params_by_eq
        self.std_errors_by_eq = std_errors_by_eq
        self.cov_by_eq = cov_by_eq
        self.resid_by_eq = resid_by_eq
        self.fitted_by_eq = fitted_by_eq
        self.endog_names = endog_names
        self.exog_names = exog_names
        self.model_info = model_info
        self.data_info = data_info

        # Extract key info
        self.K = len(endog_names)
        self.p = model_info["lags"]
        self.N = data_info["n_entities"]
        self.n_obs = data_info["n_obs"]
        self.method = model_info["method"]
        self.cov_type = model_info["cov_type"]

        # Compute residual covariance matrix
        self._compute_residual_covariance()

        # Compute information criteria
        self._compute_information_criteria()

        # Extract A matrices
        self._extract_coefficient_matrices()

        # Compute companion matrix (lazy)
        self._companion_matrix = None
        self._eigenvalues = None

    def _compute_residual_covariance(self) -> None:
        """Compute residual covariance matrix Σ̂."""
        n = self.n_obs
        residuals = np.column_stack(self.resid_by_eq)  # (n, K)
        self.Sigma = (residuals.T @ residuals) / n

    def _compute_information_criteria(self) -> None:
        """Compute AIC, BIC, HQIC for the system."""
        n = self.n_obs
        K = self.K
        p = self.p

        # Number of parameters per equation
        n_params_per_eq = K * p
        if self.model_info.get("trend") in ["constant", "both"]:
            n_params_per_eq += 1
        if self.model_info.get("trend") in ["trend", "both"]:
            n_params_per_eq += 1
        if self.model_info.get("n_exog", 0) > 0:
            n_params_per_eq += self.model_info["n_exog"]

        # Total parameters in system
        total_params = K * n_params_per_eq

        # Log-determinant of residual covariance
        log_det_sigma = np.log(np.linalg.det(self.Sigma))

        # Information criteria
        self.aic = log_det_sigma + (2 * total_params) / n
        self.bic = log_det_sigma + (total_params * np.log(n)) / n
        self.hqic = log_det_sigma + (2 * total_params * np.log(np.log(n))) / n

        # Log-likelihood (for reporting)
        self.loglik = -0.5 * n * (K * np.log(2 * np.pi) + log_det_sigma + K)

    def _extract_coefficient_matrices(self) -> None:
        """Extract coefficient matrices A_1, A_2, ..., A_p."""
        K = self.K
        p = self.p

        self.A_matrices = []

        for lag in range(p):
            # Extract coefficients for this lag from all equations
            A_l = np.zeros((K, K))
            for k in range(K):
                # In each equation, lags are organized as:
                # L1.y1, L1.y2, ..., L1.yK, L2.y1, L2.y2, ..., L2.yK, ...
                start_idx = lag * K
                end_idx = start_idx + K
                A_l[k, :] = self.params_by_eq[k][start_idx:end_idx]

            self.A_matrices.append(A_l)

    def companion_matrix(self) -> np.ndarray:
        """
        Compute companion matrix representation.

        Returns
        -------
        F : np.ndarray
            Companion matrix (Kp × Kp)

        Notes
        -----
        The companion form represents the VAR(p) as a VAR(1):

        Y_t = F * Y_{t-1} + u_t

        where Y_t = [y_t, y_{t-1}, ..., y_{t-p+1}]' and

        F = [[A_1, A_2, ..., A_p],
             [I,   0,   ..., 0  ],
             [0,   I,   ..., 0  ],
             ...
             [0,   0,   ..., I,  0]]

        The eigenvalues of F determine stability.
        """
        if self._companion_matrix is not None:
            return self._companion_matrix

        K = self.K
        p = self.p
        Kp = K * p

        F = np.zeros((Kp, Kp))

        # First K rows: [A_1, A_2, ..., A_p]
        for lag in range(p):
            F[:K, lag * K : (lag + 1) * K] = self.A_matrices[lag]

        # Remaining rows: identity blocks on subdiagonal
        for i in range(1, p):
            F[i * K : (i + 1) * K, (i - 1) * K : i * K] = np.eye(K)

        self._companion_matrix = F
        return F

    @property
    def eigenvalues(self) -> np.ndarray:
        """
        Eigenvalues of the companion matrix.

        Returns
        -------
        eigenvalues : np.ndarray (complex)
            Eigenvalues of companion matrix

        Warnings
        --------
        Issues a warning if any eigenvalue has modulus >= 1 (unstable system).
        """
        if self._eigenvalues is None:
            F = self.companion_matrix()
            self._eigenvalues = np.linalg.eigvals(F)

            # Check stability and issue warning if unstable
            max_modulus = float(np.max(np.abs(self._eigenvalues)))
            if max_modulus >= 1.0:
                import warnings

                unstable_eigs = self._eigenvalues[np.abs(self._eigenvalues) >= 1.0]
                n_unstable = len(unstable_eigs)

                warnings.warn(
                    f"Panel VAR system is UNSTABLE: {n_unstable} eigenvalue(s) have modulus >= 1. "
                    f"Maximum eigenvalue modulus: {max_modulus:.6f}. "
                    f"Impulse responses and forecasts may diverge.",
                    UserWarning,
                    stacklevel=2,
                )

        return self._eigenvalues

    @property
    def max_eigenvalue_modulus(self) -> float:
        """Maximum modulus of eigenvalues."""
        return float(np.max(np.abs(self.eigenvalues)))

    def is_stable(self) -> bool:
        """
        Check if the VAR system is stable.

        Returns
        -------
        stable : bool
            True if all eigenvalues have modulus < 1
        """
        return self.max_eigenvalue_modulus < 1.0

    @property
    def stability_margin(self) -> float:
        """
        Distance from instability.

        Returns
        -------
        margin : float
            1 - max(|eigenvalues|)
        """
        return 1.0 - self.max_eigenvalue_modulus

    def coef_matrix(self, lag: int) -> pd.DataFrame:
        """
        Get coefficient matrix A_l as DataFrame.

        Parameters
        ----------
        lag : int
            Lag number (1 to p)

        Returns
        -------
        pd.DataFrame
            Coefficient matrix (K × K) with row/column names
        """
        if lag < 1 or lag > self.p:
            raise ValueError(f"lag must be between 1 and {self.p}, got {lag}")

        A_l = self.A_matrices[lag - 1]
        return pd.DataFrame(A_l, index=self.endog_names, columns=self.endog_names)

    def equation_summary(self, k: int) -> pd.DataFrame:
        """
        Get summary table for equation k.

        Parameters
        ----------
        k : int
            Equation index (0 to K-1)

        Returns
        -------
        pd.DataFrame
            Summary table with coef, std err, t, p-value, CI
        """
        if k < 0 or k >= self.K:
            raise ValueError(f"Equation index k must be between 0 and {self.K - 1}, got {k}")

        params = self.params_by_eq[k]
        std_errors = self.std_errors_by_eq[k]

        # t-statistics and p-values
        tvalues = params / std_errors
        pvalues = 2 * (1 - stats.t.cdf(np.abs(tvalues), df=self.n_obs - len(params)))

        # Confidence intervals (95%)
        ci_lower = params - 1.96 * std_errors
        ci_upper = params + 1.96 * std_errors

        # Build DataFrame
        summary_df = pd.DataFrame(
            {
                "coef": params,
                "std err": std_errors,
                "t": tvalues,
                "P>|t|": pvalues,
                "[0.025": ci_lower,
                "0.975]": ci_upper,
            },
            index=self.exog_names,
        )

        return summary_df

    def summary_system(self) -> str:
        """
        Generate compact system-level summary (no coefficient tables).

        Returns
        -------
        str
            Formatted system summary

        Examples
        --------
        >>> result = model.fit()
        >>> print(result.summary_system())
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Panel VAR System Summary")
        lines.append("=" * 60)
        lines.append("")

        # Model specification
        lines.append("Specification:")
        lines.append(f"  Variables (K): {self.K} [{', '.join(self.endog_names)}]")
        lines.append(f"  Lags (p): {self.p}")
        lines.append(f"  Entities (N): {self.N}")
        lines.append(f"  Observations: {self.n_obs}")
        lines.append(f"  Method: {self.method.upper()}")
        lines.append(f"  Covariance: {self.cov_type}")
        lines.append("")

        # Information criteria
        lines.append("Information Criteria:")
        lines.append(f"  AIC:  {self.aic:>10.6f}")
        lines.append(f"  BIC:  {self.bic:>10.6f}")
        lines.append(f"  HQIC: {self.hqic:>10.6f}")
        lines.append(f"  Log-Likelihood: {self.loglik:>10.2f}")
        lines.append("")

        # Stability
        stable_str = "Yes" if self.is_stable() else "No (UNSTABLE!)"
        lines.append("Stability:")
        lines.append(f"  Stable: {stable_str}")
        lines.append(f"  Max eigenvalue modulus: {self.max_eigenvalue_modulus:.6f}")
        lines.append(f"  Stability margin: {self.stability_margin:.6f}")
        lines.append("")

        # Residual covariance matrix
        lines.append("Residual Covariance Matrix (Σ̂):")
        for i, name_i in enumerate(self.endog_names):
            row_vals = []
            for j, name_j in enumerate(self.endog_names):
                row_vals.append(f"{self.Sigma[i, j]:>10.6f}")
            lines.append(f"  {name_i:>6s}: " + "  ".join(row_vals))
        lines.append("")

        # Coefficient matrix summary (just magnitudes, not full table)
        lines.append("Coefficient Matrices:")
        for lag in range(self.p):
            A_l = self.A_matrices[lag]
            max_coef = np.max(np.abs(A_l))
            frobenius_norm = np.linalg.norm(A_l, "fro")
            lines.append(
                f"  A_{lag+1}: max|coef|={max_coef:.4f}, ||A_{lag+1}||_F={frobenius_norm:.4f}"
            )
        lines.append("")

        lines.append("=" * 60)
        lines.append("Note: Use .summary() for detailed coefficient tables")
        lines.append("=" * 60)

        return "\n".join(lines)

    def summary(self, equation: Optional[int] = None) -> str:
        """
        Generate summary report.

        Parameters
        ----------
        equation : int, optional
            If specified, show only this equation (0 to K-1)

        Returns
        -------
        str
            Formatted summary
        """
        lines = []
        lines.append("=" * 75)
        lines.append("Panel VAR Results")
        lines.append("=" * 75)
        lines.append(f"Method: {self.method.upper()}")
        lines.append(f"Number of entities (N): {self.N}")
        lines.append(f"Number of observations: {self.n_obs}")
        lines.append(f"Lags (p): {self.p}")
        lines.append(f"Number of equations (K): {self.K}")
        lines.append(f"Covariance type: {self.cov_type}")
        lines.append("")

        # System information criteria
        lines.append("System Information Criteria:")
        lines.append(f"  AIC:  {self.aic:.6f}")
        lines.append(f"  BIC:  {self.bic:.6f}")
        lines.append(f"  HQIC: {self.hqic:.6f}")
        lines.append(f"  Log-Likelihood: {self.loglik:.2f}")
        lines.append("")

        # Stability
        stable_str = "Yes" if self.is_stable() else "No"
        lines.append("Stability:")
        lines.append(f"  Stable: {stable_str}")
        lines.append(f"  Max eigenvalue modulus: {self.max_eigenvalue_modulus:.6f}")
        if not self.is_stable():
            lines.append("  WARNING: System is unstable (explosive)!")
        lines.append("")

        # Equation summaries
        if equation is not None:
            # Show only one equation
            equations_to_show = [equation]
        else:
            # Show all equations
            equations_to_show = list(range(self.K))

        for k in equations_to_show:
            lines.append("-" * 75)
            lines.append(f"Equation {k + 1}: {self.endog_names[k]}")
            lines.append("-" * 75)

            summary_df = self.equation_summary(k)

            # Format table
            lines.append(summary_df.to_string())
            lines.append("")

            # R-squared (within transformation)
            resid = self.resid_by_eq[k]
            fitted = self.fitted_by_eq[k]
            y = fitted + resid
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            if ss_tot > 0:
                rsq = 1 - ss_res / ss_tot
                lines.append(f"R-squared: {rsq:.4f}")
            lines.append("")

        lines.append("=" * 75)

        return "\n".join(lines)

    def wald_test(
        self, R: np.ndarray, r: Optional[np.ndarray] = None, equation: int = 0
    ) -> WaldTestResult:
        """
        Perform Wald test for linear restrictions on equation k.

        Parameters
        ----------
        R : np.ndarray
            Restriction matrix (q x k)
        r : np.ndarray, optional
            Right-hand side (default: zeros)
        equation : int, default=0
            Equation index

        Returns
        -------
        WaldTestResult
            Test result
        """
        if equation < 0 or equation >= self.K:
            raise ValueError(f"equation must be between 0 and {self.K - 1}, got {equation}")

        params = self.params_by_eq[equation]
        cov_params = self.cov_by_eq[equation]

        return wald_test(params, cov_params, R, r)

    def test_granger_causality(self, causing_var: str, caused_var: str) -> WaldTestResult:
        """
        Test Granger causality: does causing_var Granger-cause caused_var?

        Parameters
        ----------
        causing_var : str
            Name of the causing variable
        caused_var : str
            Name of the caused variable

        Returns
        -------
        WaldTestResult
            Wald test result for joint significance

        Notes
        -----
        Tests H0: All lags of causing_var have zero coefficient in the
        equation for caused_var.
        """
        # Find equation index for caused_var
        try:
            eq_idx = self.endog_names.index(caused_var)
        except ValueError:
            raise ValueError(f"Variable '{caused_var}' not found in endogenous variables")

        # Find indices of causing_var lags in the regressors
        causing_indices = []
        for lag in range(1, self.p + 1):
            lag_name = f"L{lag}.{causing_var}"
            try:
                idx = self.exog_names.index(lag_name)
                causing_indices.append(idx)
            except ValueError:
                pass

        if len(causing_indices) == 0:
            raise ValueError(f"No lags of '{causing_var}' found in regressors")

        # Perform F-test for exclusion
        params = self.params_by_eq[eq_idx]
        cov_params = self.cov_by_eq[eq_idx]

        result = f_test_exclusion(params, cov_params, causing_indices)
        result.hypothesis = f"{causing_var} does not Granger-cause {caused_var}"

        return result

    def granger_causality(self, cause: str, effect: str):
        """
        Test Granger causality with enhanced result formatting.

        This is an enhanced version of test_granger_causality that returns
        a more detailed GrangerCausalityResult object.

        Parameters
        ----------
        cause : str
            Name of the causing variable
        effect : str
            Name of the effect variable

        Returns
        -------
        GrangerCausalityResult
            Enhanced test result with formatted summary

        Examples
        --------
        >>> result = model.fit()
        >>> gc = result.granger_causality('gdp', 'inflation')
        >>> print(gc.summary())
        """
        from panelbox.var.causality import granger_causality_wald

        # Get equation index
        eq_idx = self.endog_names.index(effect)

        # Perform Granger causality test
        gc_result = granger_causality_wald(
            params=self.params_by_eq[eq_idx],
            cov_params=self.cov_by_eq[eq_idx],
            exog_names=self.exog_names,
            causing_var=cause,
            caused_var=effect,
            lags=self.p,
            n_obs=self.n_obs,
        )

        return gc_result

    def granger_causality_matrix(self, significance_level: float = 0.05) -> pd.DataFrame:
        """
        Compute Granger causality matrix for all variable pairs.

        Parameters
        ----------
        significance_level : float, default=0.05
            Significance level for marking (not currently used in output,
            but reserved for future formatting)

        Returns
        -------
        pd.DataFrame
            Matrix of p-values (K × K) where element (i,j) is the p-value
            for testing "variable i Granger-causes variable j"
            Diagonal elements are NaN.

        Examples
        --------
        >>> result = model.fit()
        >>> gc_matrix = result.granger_causality_matrix()
        >>> print(gc_matrix)
        """
        from panelbox.var.causality import granger_causality_matrix

        return granger_causality_matrix(self, significance_level)

    def dumitrescu_hurlin(self, cause: str, effect: str, use_raw_data: bool = True):
        """
        Perform Dumitrescu-Hurlin (2012) Granger causality test for heterogeneous panels.

        This test allows for heterogeneous coefficients across entities and is more
        appropriate for panel data than the standard Wald test.

        Parameters
        ----------
        cause : str
            Name of the causing variable
        effect : str
            Name of the effect variable
        use_raw_data : bool, default=True
            If True, re-estimate individual regressions using raw data.
            If False, use residuals from the pooled VAR (less accurate).

        Returns
        -------
        DumitrescuHurlinResult
            Test result with W_bar, Z_tilde, and Z_bar statistics

        Examples
        --------
        >>> result = model.fit()
        >>> dh = result.dumitrescu_hurlin('gdp', 'inflation')
        >>> print(dh.summary())

        References
        ----------
        Dumitrescu, E. I., & Hurlin, C. (2012). Testing for Granger non-causality
        in heterogeneous panels. Economic modelling, 29(4), 1450-1460.
        """
        from panelbox.var.causality import dumitrescu_hurlin_test

        # Get raw data from data_info
        if use_raw_data and "data" in self.data_info:
            data = self.data_info["data"]
            entity_col = self.data_info.get("entity_col", "entity")
            time_col = self.data_info.get("time_col", "time")

            return dumitrescu_hurlin_test(
                data=data,
                cause=cause,
                effect=effect,
                lags=self.p,
                entity_col=entity_col,
                time_col=time_col,
            )
        else:
            raise NotImplementedError(
                "Dumitrescu-Hurlin test from fitted VAR residuals not yet implemented. "
                "Please ensure raw data is available in data_info."
            )

    def instantaneous_causality(self, var1: str, var2: str):
        """
        Test for instantaneous (contemporaneous) causality between two variables.

        Parameters
        ----------
        var1 : str
            First variable name
        var2 : str
            Second variable name

        Returns
        -------
        InstantaneousCausalityResult
            Test result with correlation and LR statistic

        Examples
        --------
        >>> result = model.fit()
        >>> ic = result.instantaneous_causality('gdp', 'inflation')
        >>> print(ic.summary())
        """
        from panelbox.var.causality import instantaneous_causality

        # Get indices
        idx1 = self.endog_names.index(var1)
        idx2 = self.endog_names.index(var2)

        # Get residuals
        resid1 = self.resid_by_eq[idx1]
        resid2 = self.resid_by_eq[idx2]

        return instantaneous_causality(resid1, resid2, var1, var2)

    def instantaneous_causality_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute instantaneous causality matrix for all variable pairs.

        Returns
        -------
        corr_matrix : pd.DataFrame
            Correlation matrix of residuals (K × K)
        pvalue_matrix : pd.DataFrame
            P-value matrix for LR tests (K × K)

        Examples
        --------
        >>> result = model.fit()
        >>> corr, pvals = result.instantaneous_causality_matrix()
        >>> print(corr)
        >>> print(pvals)
        """
        from panelbox.var.causality import instantaneous_causality_matrix

        return instantaneous_causality_matrix(self)

    def to_latex(self, equation: Optional[int] = None) -> str:
        """
        Export results to LaTeX table format.

        Parameters
        ----------
        equation : int, optional
            If specified, export only this equation. Otherwise, export all equations.

        Returns
        -------
        str
            LaTeX table code

        Examples
        --------
        >>> latex_code = results.to_latex()
        >>> # Save to file
        >>> with open('var_results.tex', 'w') as f:
        ...     f.write(latex_code)
        """
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append(f"\\caption{{Panel VAR Results (K={self.K}, p={self.p}, N={self.N})}}")
        lines.append("\\begin{tabular}{l" + "c" * (self.K if equation is None else 1) + "}")
        lines.append("\\hline\\hline")

        # Header
        if equation is None:
            header = "Variable & " + " & ".join(self.endog_names) + " \\\\"
        else:
            header = f"Variable & {self.endog_names[equation]} \\\\"
        lines.append(header)
        lines.append("\\hline")

        # Determine which equations to show
        equations_to_show = list(range(self.K)) if equation is None else [equation]

        # Get all regressor names
        all_regressors = self.exog_names

        # For each regressor, show coefficients across equations
        for i, reg_name in enumerate(all_regressors):
            row_cells = [reg_name.replace("_", "\\_")]
            for eq in equations_to_show:
                coef = self.params_by_eq[eq][i]
                se = self.std_errors_by_eq[eq][i]
                t_val = coef / se
                # Add significance stars
                p_val = 2 * (
                    1 - stats.t.cdf(np.abs(t_val), df=self.n_obs - len(self.params_by_eq[eq]))
                )
                stars = ""
                if p_val < 0.01:
                    stars = "^{***}"
                elif p_val < 0.05:
                    stars = "^{**}"
                elif p_val < 0.1:
                    stars = "^{*}"

                row_cells.append(f"{coef:.4f}{stars}")

            lines.append(" & ".join(row_cells) + " \\\\")

            # Add standard errors in parentheses
            se_cells = [""]
            for eq in equations_to_show:
                se = self.std_errors_by_eq[eq][i]
                se_cells.append(f"({se:.4f})")
            lines.append(" & ".join(se_cells) + " \\\\")

        lines.append("\\hline")

        # Summary statistics
        lines.append(
            "\\multicolumn{"
            + str(len(equations_to_show) + 1)
            + "}{l}{\\textit{Model Statistics}} \\\\"
        )
        lines.append(f"N & \\multicolumn{{{len(equations_to_show)}}}{{c}}{{{self.N}}} \\\\")
        lines.append(
            f"Observations & \\multicolumn{{{len(equations_to_show)}}}{{c}}{{{self.n_obs}}} \\\\"
        )
        lines.append(f"Lags & \\multicolumn{{{len(equations_to_show)}}}{{c}}{{{self.p}}} \\\\")
        lines.append(f"AIC & \\multicolumn{{{len(equations_to_show)}}}{{c}}{{{self.aic:.4f}}} \\\\")
        lines.append(f"BIC & \\multicolumn{{{len(equations_to_show)}}}{{c}}{{{self.bic:.4f}}} \\\\")
        stable_str = "Yes" if self.is_stable() else "No"
        lines.append(
            f"Stable & \\multicolumn{{{len(equations_to_show)}}}{{c}}{{{stable_str}}} \\\\"
        )

        lines.append("\\hline\\hline")
        lines.append(
            "\\multicolumn{"
            + str(len(equations_to_show) + 1)
            + "}{l}{\\footnotesize $^{*}p<0.1$; $^{**}p<0.05$; $^{***}p<0.01$} \\\\"
        )
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def to_html(self, equation: Optional[int] = None) -> str:
        """
        Export results to HTML table format.

        Parameters
        ----------
        equation : int, optional
            If specified, export only this equation. Otherwise, export all equations.

        Returns
        -------
        str
            HTML table code

        Examples
        --------
        >>> html_code = results.to_html()
        >>> # Save to file
        >>> with open('var_results.html', 'w') as f:
        ...     f.write(html_code)
        """
        lines = []
        lines.append('<div class="panel-var-results">')
        lines.append(f"<h3>Panel VAR Results (K={self.K}, p={self.p}, N={self.N})</h3>")
        lines.append('<table border="1" class="dataframe">')
        lines.append("  <thead>")
        lines.append('    <tr style="text-align: right;">')
        lines.append("      <th>Variable</th>")

        # Determine which equations to show
        equations_to_show = list(range(self.K)) if equation is None else [equation]

        for eq in equations_to_show:
            lines.append(f"      <th>{self.endog_names[eq]}</th>")
        lines.append("    </tr>")
        lines.append("  </thead>")
        lines.append("  <tbody>")

        # Get all regressor names
        all_regressors = self.exog_names

        # For each regressor, show coefficients across equations
        for i, reg_name in enumerate(all_regressors):
            lines.append("    <tr>")
            lines.append(f"      <td><b>{reg_name}</b></td>")

            for eq in equations_to_show:
                coef = self.params_by_eq[eq][i]
                se = self.std_errors_by_eq[eq][i]
                t_val = coef / se
                # Add significance stars
                p_val = 2 * (
                    1 - stats.t.cdf(np.abs(t_val), df=self.n_obs - len(self.params_by_eq[eq]))
                )
                stars = ""
                if p_val < 0.01:
                    stars = "***"
                elif p_val < 0.05:
                    stars = "**"
                elif p_val < 0.1:
                    stars = "*"

                lines.append(f"      <td>{coef:.4f}{stars}<br/><small>({se:.4f})</small></td>")

            lines.append("    </tr>")

        lines.append("  </tbody>")
        lines.append("</table>")

        # Summary statistics
        lines.append('<div class="model-stats">')
        lines.append(f"<p><b>Model Statistics:</b></p>")
        lines.append(f"<p>N: {self.N} | Observations: {self.n_obs} | Lags: {self.p}</p>")
        lines.append(f"<p>AIC: {self.aic:.4f} | BIC: {self.bic:.4f} | HQIC: {self.hqic:.4f}</p>")
        stable_str = "Yes" if self.is_stable() else "No"
        lines.append(f"<p>Stable: {stable_str}</p>")
        lines.append(f"<p><small>*p&lt;0.1; **p&lt;0.05; ***p&lt;0.01</small></p>")
        lines.append("</div>")
        lines.append("</div>")

        return "\n".join(lines)

    def plot_stability(
        self, backend: str = "matplotlib", figsize: tuple = (8, 8), show: bool = True
    ) -> Optional[object]:
        """
        Plot eigenvalues of companion matrix with unit circle.

        This visualizes the stability of the VAR system. A VAR is stable if all
        eigenvalues lie within the unit circle (modulus < 1).

        Parameters
        ----------
        backend : str, default="matplotlib"
            Plotting backend: "matplotlib" or "plotly"
        figsize : tuple, default=(8, 8)
            Figure size for matplotlib (width, height in inches)
        show : bool, default=True
            Whether to display the plot immediately

        Returns
        -------
        fig : matplotlib.figure.Figure or plotly.graph_objects.Figure or None
            Figure object if show=False, otherwise None

        Notes
        -----
        The plot shows:
        - Unit circle in blue (dashed line)
        - Eigenvalues as points:
          - Green circles: stable (modulus < 1)
          - Red X markers: unstable (modulus >= 1)
        - Annotations with modulus for each eigenvalue

        Examples
        --------
        >>> # Plot stability
        >>> results.plot_stability()
        >>>
        >>> # Get figure for customization
        >>> fig = results.plot_stability(show=False)
        >>> # With matplotlib:
        >>> fig.savefig('stability.png')
        >>>
        >>> # Use plotly for interactive plot
        >>> results.plot_stability(backend='plotly')
        """
        title = f"VAR({self.p}) Stability - Eigenvalues of Companion Matrix"
        return _plot_stability(
            self.eigenvalues, title=title, backend=backend, figsize=figsize, show=show
        )

    def irf(
        self,
        periods: int = 20,
        method: str = "cholesky",
        shock_size: Union[str, float] = "one_std",
        cumulative: bool = False,
        order: Optional[List[str]] = None,
        ci_method: Optional[str] = None,
        n_bootstrap: int = 500,
        ci_level: float = 0.95,
        bootstrap_ci_method: str = "percentile",
        n_jobs: int = -1,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> "IRFResult":
        """
        Compute Impulse Response Functions (IRFs).

        Parameters
        ----------
        periods : int, default=20
            Number of periods (horizons) to compute
        method : str, default='cholesky'
            IRF method:
            - 'cholesky': Orthogonalized IRF using Cholesky decomposition
            - 'generalized': Generalized IRF (Pesaran-Shin), order-invariant
        shock_size : str or float, default='one_std'
            Size of shock:
            - 'one_std': one standard deviation shock (default)
            - float: shock of specified size
        cumulative : bool, default=False
            If True, compute cumulative IRFs
        order : list of str, optional
            Variable ordering for Cholesky decomposition.
            If None, uses original order.
            Only relevant for method='cholesky'.
        ci_method : str, optional
            Method for confidence intervals:
            - None: No confidence intervals (default)
            - 'bootstrap': Bootstrap confidence intervals
        n_bootstrap : int, default=500
            Number of bootstrap iterations (only if ci_method='bootstrap')
        ci_level : float, default=0.95
            Confidence level (only if ci_method='bootstrap')
        bootstrap_ci_method : str, default='percentile'
            Bootstrap CI method:
            - 'percentile': Standard percentile method
            - 'bias_corrected': Bias-corrected percentile (Hall)
        n_jobs : int, default=-1
            Number of parallel jobs for bootstrap (-1 = all cores)
        seed : int, optional
            Random seed for bootstrap reproducibility
        verbose : bool, default=True
            Show progress bar for bootstrap

        Returns
        -------
        IRFResult
            IRF results container

        Examples
        --------
        >>> result = model.fit()
        >>> # Orthogonalized IRF (Cholesky)
        >>> irf_chol = result.irf(periods=20, method='cholesky')
        >>> print(irf_chol.summary())
        >>>
        >>> # Generalized IRF
        >>> irf_gen = result.irf(periods=20, method='generalized')
        >>>
        >>> # Cumulative IRF
        >>> irf_cum = result.irf(periods=20, cumulative=True)
        >>>
        >>> # Custom ordering
        >>> irf_ord = result.irf(order=['inflation', 'gdp', 'interest_rate'])

        Notes
        -----
        Cholesky IRFs depend on variable ordering. Variables earlier in the
        order are treated as more "exogenous" (respond only to own shocks
        contemporaneously).

        Generalized IRFs are invariant to ordering but shocks are not
        orthogonal (they reflect actual correlations).

        References
        ----------
        .. [1] Lütkepohl, H. (2005). New Introduction to Multiple Time
               Series Analysis. Springer-Verlag.
        .. [2] Pesaran, H. H., & Shin, Y. (1998). Generalized impulse response
               analysis in linear multivariate models. Economics letters, 58(1), 17-29.
        """
        from panelbox.var.irf import (
            IRFResult,
            bootstrap_irf,
            compute_cumulative_irf,
            compute_irf_cholesky,
            compute_irf_generalized,
            compute_phi_non_orthogonalized,
        )

        # Check stability
        if not self.is_stable():
            warnings.warn(
                "VAR system is UNSTABLE (max eigenvalue modulus >= 1). "
                "IRFs may diverge and not converge to zero.",
                UserWarning,
            )

        # Handle ordering
        if order is not None:
            # Validate ordering
            if set(order) != set(self.endog_names):
                raise ValueError(
                    f"order must contain exactly the endogenous variables: {self.endog_names}"
                )

            # Reorder variables
            order_indices = [self.endog_names.index(var) for var in order]

            # Reorder A matrices
            A_matrices_reordered = []
            for A in self.A_matrices:
                # Reorder both rows and columns
                A_reordered = A[np.ix_(order_indices, order_indices)]
                A_matrices_reordered.append(A_reordered)

            # Reorder Sigma
            Sigma_reordered = self.Sigma[np.ix_(order_indices, order_indices)]

            var_names = order
            A_matrices = A_matrices_reordered
            Sigma = Sigma_reordered

            if method == "cholesky":
                warnings.warn(
                    f"Variable ordering for Cholesky decomposition: {order}. "
                    "Earlier variables are treated as more exogenous.",
                    UserWarning,
                )
        else:
            var_names = self.endog_names
            A_matrices = self.A_matrices
            Sigma = self.Sigma
            order = None

        # Compute IRFs
        if method == "cholesky":
            irf_matrix = compute_irf_cholesky(A_matrices, Sigma, periods, shock_size)
        elif method == "generalized":
            # First compute non-orthogonalized Phi
            Phi = compute_phi_non_orthogonalized(A_matrices, periods)
            # Then compute GIRF
            irf_matrix = compute_irf_generalized(Phi, Sigma, periods)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'cholesky' or 'generalized'.")

        # Compute cumulative if requested
        if cumulative:
            irf_matrix = compute_cumulative_irf(irf_matrix)

        # Create result object
        irf_result = IRFResult(
            irf_matrix=irf_matrix,
            var_names=var_names,
            periods=periods,
            method=method,
            shock_size=shock_size,
            cumulative=cumulative,
            ordering=order,
        )

        # Compute bootstrap confidence intervals if requested
        if ci_method == "bootstrap":
            # Get residuals
            residuals = np.column_stack(self.resid_by_eq)  # (n_obs, K)

            # Apply ordering to residuals if needed
            if order is not None:
                residuals = residuals[:, order_indices]

            # Compute bootstrap CI
            ci_lower, ci_upper, bootstrap_dist = bootstrap_irf(
                A_matrices=A_matrices,
                Sigma=Sigma,
                residuals=residuals,
                periods=periods,
                method=method,
                n_bootstrap=n_bootstrap,
                ci_level=ci_level,
                cumulative=cumulative,
                ci_method=bootstrap_ci_method,
                n_jobs=n_jobs,
                seed=seed,
                verbose=verbose,
            )

            # Attach to result object
            irf_result.ci_lower = ci_lower
            irf_result.ci_upper = ci_upper
            irf_result.ci_level = ci_level
            irf_result.bootstrap_dist = bootstrap_dist

        return irf_result

    def fevd(
        self,
        periods: int = 20,
        method: str = "cholesky",
        order: Optional[List[str]] = None,
    ) -> "FEVDResult":
        """
        Compute Forecast Error Variance Decomposition (FEVD).

        Parameters
        ----------
        periods : int, default=20
            Number of periods (horizons) to compute
        method : str, default='cholesky'
            FEVD method:
            - 'cholesky': FEVD based on Cholesky decomposition
            - 'generalized': Generalized FEVD (Pesaran-Shin), order-invariant
        order : list of str, optional
            Variable ordering for Cholesky decomposition.
            If None, uses original order.
            Only relevant for method='cholesky'.

        Returns
        -------
        FEVDResult
            FEVD results container

        Examples
        --------
        >>> result = model.fit()
        >>> # Cholesky FEVD
        >>> fevd_chol = result.fevd(periods=20, method='cholesky')
        >>> print(fevd_chol.summary())
        >>>
        >>> # Generalized FEVD
        >>> fevd_gen = result.fevd(periods=20, method='generalized')
        >>>
        >>> # Custom ordering
        >>> fevd_ord = result.fevd(order=['inflation', 'gdp', 'interest_rate'])

        Notes
        -----
        FEVD decomposes the forecast error variance of each variable into
        contributions from shocks to all variables in the system.

        Cholesky FEVD depends on variable ordering. Variables earlier in the
        order are treated as more "exogenous".

        Generalized FEVD is invariant to ordering but values may not sum
        exactly to 100% before normalization.

        References
        ----------
        .. [1] Lütkepohl, H. (2005). New Introduction to Multiple Time
               Series Analysis. Springer-Verlag. Chapter 2.
        .. [2] Pesaran, H. H., & Shin, Y. (1998). Generalized impulse response
               analysis in linear multivariate models. Economics letters, 58(1), 17-29.
        """
        from panelbox.var.fevd import FEVDResult, compute_fevd_cholesky, compute_fevd_generalized
        from panelbox.var.irf import compute_irf_cholesky, compute_phi_non_orthogonalized

        # Check stability
        if not self.is_stable():
            warnings.warn(
                "VAR system is UNSTABLE (max eigenvalue modulus >= 1). "
                "FEVD may not be well-defined.",
                UserWarning,
            )

        # Handle ordering
        if order is not None:
            # Validate ordering
            if set(order) != set(self.endog_names):
                raise ValueError(
                    f"order must contain exactly the endogenous variables: {self.endog_names}"
                )

            # Reorder variables
            order_indices = [self.endog_names.index(var) for var in order]

            # Reorder A matrices
            A_matrices_reordered = []
            for A in self.A_matrices:
                A_reordered = A[np.ix_(order_indices, order_indices)]
                A_matrices_reordered.append(A_reordered)

            # Reorder Sigma
            Sigma_reordered = self.Sigma[np.ix_(order_indices, order_indices)]

            var_names = order
            A_matrices = A_matrices_reordered
            Sigma = Sigma_reordered

            if method == "cholesky":
                warnings.warn(
                    f"Variable ordering for Cholesky decomposition: {order}. "
                    "Earlier variables are treated as more exogenous.",
                    UserWarning,
                )
        else:
            var_names = self.endog_names
            A_matrices = self.A_matrices
            Sigma = self.Sigma
            order = None

        # Compute FEVD
        if method == "cholesky":
            # Compute Cholesky IRFs first
            Phi = compute_irf_cholesky(A_matrices, Sigma, periods)
            P = np.linalg.cholesky(Sigma)
            fevd_matrix = compute_fevd_cholesky(Phi, P, Sigma, periods)
        elif method == "generalized":
            # Compute non-orthogonalized Phi
            Phi = compute_phi_non_orthogonalized(A_matrices, periods)
            fevd_matrix = compute_fevd_generalized(Phi, Sigma, periods)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'cholesky' or 'generalized'.")

        # Create result object
        fevd_result = FEVDResult(
            decomposition=fevd_matrix,
            var_names=var_names,
            periods=periods,
            method=method,
            ordering=order,
        )

        return fevd_result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PanelVARResult(K={self.K}, p={self.p}, N={self.N}, "
            f"n_obs={self.n_obs}, method='{self.method}', stable={self.is_stable()})"
        )


class PanelVARGMMResult(PanelVARResult):
    """
    Results container for Panel VAR GMM estimation.

    Extends PanelVARResult with GMM-specific diagnostics and properties.

    Additional Attributes
    ---------------------
    instruments : np.ndarray
        Instrument matrix Z used in GMM estimation
    n_instruments : int
        Total number of instruments used
    n_instruments_by_eq : List[int], optional
        Number of instruments per equation (for multi-equation systems)
    instrument_type : str
        Type of instruments: 'all' (all available lags) or 'collapsed' (Roodman collapsed)
    gmm_step : str
        GMM step: 'one-step' or 'two-step'
    entity_ids : np.ndarray, optional
        Entity identifiers for AR tests
    windmeijer_corrected : bool
        Whether Windmeijer finite-sample correction was applied
    """

    def __init__(
        self,
        params_by_eq: List[np.ndarray],
        std_errors_by_eq: List[np.ndarray],
        cov_by_eq: List[np.ndarray],
        resid_by_eq: List[np.ndarray],
        fitted_by_eq: List[np.ndarray],
        endog_names: List[str],
        exog_names: List[str],
        model_info: Dict[str, Any],
        data_info: Dict[str, Any],
        instruments: np.ndarray,
        n_instruments: int,
        instrument_type: str = "all",
        gmm_step: str = "two-step",
        entity_ids: Optional[np.ndarray] = None,
        n_instruments_by_eq: Optional[List[int]] = None,
        windmeijer_corrected: bool = False,
    ):
        # Initialize parent class
        super().__init__(
            params_by_eq=params_by_eq,
            std_errors_by_eq=std_errors_by_eq,
            cov_by_eq=cov_by_eq,
            resid_by_eq=resid_by_eq,
            fitted_by_eq=fitted_by_eq,
            endog_names=endog_names,
            exog_names=exog_names,
            model_info=model_info,
            data_info=data_info,
        )

        # GMM-specific attributes
        self.instruments = instruments
        self.n_instruments = n_instruments
        self.n_instruments_by_eq = n_instruments_by_eq
        self.instrument_type = instrument_type
        self.gmm_step = gmm_step
        self.entity_ids = entity_ids
        self.windmeijer_corrected = windmeijer_corrected

        # Cache for diagnostics
        self._gmm_diagnostics = None

    @property
    def gmm_diagnostics(self):
        """
        Get GMM diagnostics object.

        Returns
        -------
        GMMDiagnostics
            Diagnostic test suite
        """
        if self._gmm_diagnostics is None:
            from panelbox.var.diagnostics import GMMDiagnostics

            # Stack residuals into matrix (n_obs × K)
            residuals = np.column_stack(self.resid_by_eq)

            # Total number of parameters
            n_params_total = sum(len(p) for p in self.params_by_eq)

            self._gmm_diagnostics = GMMDiagnostics(
                residuals=residuals,
                instruments=self.instruments,
                n_params=n_params_total,
                n_entities=self.N,
                entity_ids=self.entity_ids,
            )

        return self._gmm_diagnostics

    def hansen_j_test(self) -> Dict:
        """
        Perform Hansen J test for over-identifying restrictions.

        Returns
        -------
        dict
            Hansen J test results with keys:
            - 'statistic': J statistic
            - 'p_value': p-value
            - 'df': degrees of freedom
            - 'interpretation': interpretation
            - 'warnings': list of warnings
        """
        return self.gmm_diagnostics.hansen_j_test()

    def sargan_test(self) -> Dict:
        """
        Perform Sargan test (non-robust alternative to Hansen J).

        Returns
        -------
        dict
            Sargan test results
        """
        return self.gmm_diagnostics.sargan_test()

    def ar_test(self, order: int = 1) -> Dict:
        """
        Perform Arellano-Bond AR test for serial correlation.

        Parameters
        ----------
        order : int
            Order of autocorrelation (1 or 2)

        Returns
        -------
        dict
            AR test results
        """
        return self.gmm_diagnostics.ar_test(order=order)

    def instrument_diagnostics(self) -> str:
        """
        Generate comprehensive instrument diagnostics report.

        Returns
        -------
        str
            Formatted diagnostic report including:
            - Instrument counts and ratios
            - Hansen J test
            - AR tests (if entity_ids available)
            - Warnings and suggestions

        Examples
        --------
        >>> result = model.fit_gmm()
        >>> print(result.instrument_diagnostics())
        """
        return self.gmm_diagnostics.format_diagnostics_report(
            include_ar_tests=(self.entity_ids is not None)
        )

    def compare_one_step_two_step(self, result_other: "PanelVARGMMResult") -> str:
        """
        Compare one-step and two-step GMM results.

        Large divergence between one-step and two-step estimates can indicate:
        - Instrument proliferation
        - Weak instruments
        - Model misspecification

        Parameters
        ----------
        result_other : PanelVARGMMResult
            The other GMM result to compare with (must be estimated with different gmm_step)

        Returns
        -------
        str
            Formatted comparison report

        Examples
        --------
        >>> result_1step = model.fit_gmm(gmm_step='one-step')
        >>> result_2step = model.fit_gmm(gmm_step='two-step')
        >>> print(result_1step.compare_one_step_two_step(result_2step))
        """
        import numpy as np

        lines = []
        lines.append("=" * 70)
        lines.append("One-Step vs Two-Step GMM Comparison")
        lines.append("=" * 70)
        lines.append(f"This result:  {self.gmm_step}")
        lines.append(f"Other result: {result_other.gmm_step}")
        lines.append("")

        # Flatten all coefficients
        params_self = np.concatenate([p.flatten() for p in self.params_by_eq])
        params_other = np.concatenate([p.flatten() for p in result_other.params_by_eq])

        if len(params_self) != len(params_other):
            lines.append("ERROR: Results have different numbers of parameters!")
            lines.append("=" * 70)
            return "\n".join(lines)

        # Compute differences
        abs_diff = np.abs(params_self - params_other)
        pct_diff = np.zeros_like(abs_diff)
        for i in range(len(params_self)):
            if abs(params_other[i]) > 1e-10:
                pct_diff[i] = abs_diff[i] / abs(params_other[i]) * 100
            else:
                pct_diff[i] = abs_diff[i] * 100  # Treat as absolute when baseline near zero

        # Summary statistics
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        max_pct_diff = np.max(pct_diff)
        mean_pct_diff = np.mean(pct_diff)

        lines.append("Coefficient Differences:")
        lines.append(f"  Max absolute difference:     {max_abs_diff:.6f}")
        lines.append(f"  Mean absolute difference:    {mean_abs_diff:.6f}")
        lines.append(f"  Max percentage difference:   {max_pct_diff:.2f}%")
        lines.append(f"  Mean percentage difference:  {mean_pct_diff:.2f}%")
        lines.append("")

        # Number of coefficients with large differences
        n_large_diff = np.sum(pct_diff > 10.0)
        pct_large_diff = n_large_diff / len(pct_diff) * 100

        lines.append(
            f"Coefficients with >10% change:  {n_large_diff} / {len(pct_diff)} ({pct_large_diff:.1f}%)"
        )
        lines.append("")

        # Diagnosis
        if max_pct_diff < 5.0:
            diagnosis = "✓ EXCELLENT: One-step and two-step estimates are very close"
            interpretation = "Instruments appear strong and well-specified"
        elif max_pct_diff < 10.0:
            diagnosis = "✓ GOOD: One-step and two-step estimates are reasonably close"
            interpretation = "Instruments appear adequate"
        elif max_pct_diff < 25.0:
            diagnosis = "◐ MODERATE: Some divergence between one-step and two-step"
            interpretation = "Consider checking for instrument proliferation or weak instruments"
        else:
            diagnosis = "⚠ WARNING: Large divergence between one-step and two-step"
            interpretation = (
                "This suggests problems with instruments:\n"
                "  - Too many instruments (proliferation)\n"
                "  - Weak instruments\n"
                "  - Consider using collapsed instruments or reducing max_instruments"
            )

        lines.append(f"DIAGNOSIS: {diagnosis}")
        lines.append(f"  {interpretation}")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def summary(self, equation: Optional[int] = None) -> str:
        """
        Generate summary report with GMM diagnostics.

        Parameters
        ----------
        equation : int, optional
            If specified, show only this equation (0 to K-1)

        Returns
        -------
        str
            Formatted summary with GMM-specific information
        """
        # Get base summary from parent class
        base_summary = super().summary(equation=equation)

        # Add GMM-specific section
        gmm_lines = []
        gmm_lines.append("")
        gmm_lines.append("=" * 75)
        gmm_lines.append("GMM Estimation Details")
        gmm_lines.append("=" * 75)
        gmm_lines.append(f"GMM step: {self.gmm_step}")
        gmm_lines.append(f"Instrument type: {self.instrument_type}")
        gmm_lines.append(f"Number of instruments: {self.n_instruments}")
        if self.n_instruments_by_eq is not None:
            gmm_lines.append(f"  By equation: {self.n_instruments_by_eq}")
        gmm_lines.append(f"Windmeijer correction: {'Yes' if self.windmeijer_corrected else 'No'}")
        gmm_lines.append("")

        # Add Hansen J test
        hansen_result = self.hansen_j_test()
        gmm_lines.append("Hansen J Test for Over-identifying Restrictions:")
        gmm_lines.append(f"  Statistic: {hansen_result['statistic']:.4f}")
        gmm_lines.append(f"  P-value: {hansen_result['p_value']:.4f}")
        gmm_lines.append(f"  DF: {hansen_result['df']}")
        gmm_lines.append(f"  {hansen_result['interpretation']}")

        if hansen_result.get("warnings"):
            gmm_lines.append("  Warnings:")
            for warning in hansen_result["warnings"]:
                gmm_lines.append(f"    - {warning}")

        # Add AR tests if available
        if self.entity_ids is not None:
            gmm_lines.append("")
            gmm_lines.append("Serial Correlation Tests:")

            # AR(1)
            ar1_result = self.ar_test(order=1)
            gmm_lines.append(
                f"  AR(1): z = {ar1_result['statistic']:.3f}, "
                f"p-value = {ar1_result['p_value']:.4f}"
            )
            gmm_lines.append(f"    {ar1_result['interpretation']}")

            # AR(2)
            ar2_result = self.ar_test(order=2)
            gmm_lines.append(
                f"  AR(2): z = {ar2_result['statistic']:.3f}, "
                f"p-value = {ar2_result['p_value']:.4f}"
            )
            gmm_lines.append(f"    {ar2_result['interpretation']}")

        gmm_lines.append("=" * 75)

        # Combine base summary with GMM section
        return base_summary + "\n" + "\n".join(gmm_lines)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PanelVARGMMResult(K={self.K}, p={self.p}, N={self.N}, "
            f"n_obs={self.n_obs}, method='{self.method}', "
            f"gmm_step='{self.gmm_step}', n_instruments={self.n_instruments}, "
            f"stable={self.is_stable()})"
        )


class LagOrderResult:
    """
    Results container for lag order selection.

    This class stores results from testing multiple lag orders.

    Parameters
    ----------
    criteria_df : pd.DataFrame
        DataFrame with columns: lags, AIC, BIC, HQIC, MBIC (optional)
    selected : Dict[str, int]
        Dictionary mapping criterion name to selected lag order
    """

    def __init__(self, criteria_df: pd.DataFrame, selected: Dict[str, int]):
        self.criteria_df = criteria_df
        self.selected = selected

    def summary(self) -> str:
        """
        Generate summary table.

        Returns
        -------
        str
            Formatted summary
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Lag Order Selection")
        lines.append("=" * 60)
        lines.append("")

        # Format table with asterisks for selected lags
        # Build table manually to add asterisks
        df_display = self.criteria_df.copy()

        # Build header
        cols = df_display.columns.tolist()
        header_line = "  ".join([f"{col:>12}" for col in cols])
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Build rows with asterisks for selected lags
        for _, row in df_display.iterrows():
            row_vals = []
            for col in cols:
                value = row[col]
                if col == "lags":
                    row_vals.append(f"{int(value):>12}")
                elif col in self.selected and row["lags"] == self.selected[col]:
                    # Add asterisk to selected value
                    row_vals.append(f"{value:>11.6f}*")
                else:
                    row_vals.append(f"{value:>12.6f}")
            lines.append("  ".join(row_vals))
        lines.append("")
        lines.append("Selected lags:")
        for criterion, lag in self.selected.items():
            lines.append(f"  {criterion}: {lag}")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def plot(self, backend: str = "plotly"):
        """
        Plot information criteria vs lag order.

        Parameters
        ----------
        backend : str, default='plotly'
            Plotting backend: 'plotly' or 'matplotlib'

        Returns
        -------
        fig
            Figure object (plotly.graph_objects.Figure or matplotlib.figure.Figure)

        Examples
        --------
        >>> lag_results = model.select_lag_order(max_lags=8)
        >>> fig = lag_results.plot(backend='plotly')
        >>> fig.show()
        """
        if backend == "plotly":
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
            except ImportError:
                raise ImportError(
                    "Plotly is required for this plot. Install with: pip install plotly"
                )

            # Create subplots - one for each criterion
            criteria = [c for c in ["AIC", "BIC", "HQIC", "MBIC"] if c in self.criteria_df.columns]
            n_criteria = len(criteria)

            fig = make_subplots(
                rows=1, cols=n_criteria, subplot_titles=criteria, horizontal_spacing=0.12
            )

            for i, criterion in enumerate(criteria, 1):
                # Plot line
                fig.add_trace(
                    go.Scatter(
                        x=self.criteria_df["lags"],
                        y=self.criteria_df[criterion],
                        mode="lines+markers",
                        name=criterion,
                        line=dict(width=2),
                        marker=dict(size=8),
                        showlegend=False,
                    ),
                    row=1,
                    col=i,
                )

                # Mark selected lag
                if criterion in self.selected:
                    selected_lag = self.selected[criterion]
                    mask = self.criteria_df["lags"] == selected_lag
                    selected_value = self.criteria_df.loc[mask, criterion].values[0]

                    fig.add_trace(
                        go.Scatter(
                            x=[selected_lag],
                            y=[selected_value],
                            mode="markers",
                            marker=dict(
                                size=12,
                                color="red",
                                symbol="star",
                                line=dict(width=2, color="darkred"),
                            ),
                            name=f"Selected (p={selected_lag})",
                            showlegend=False,
                        ),
                        row=1,
                        col=i,
                    )

                # Update axes
                fig.update_xaxes(title_text="Lag Order", row=1, col=i)
                fig.update_yaxes(title_text=criterion, row=1, col=i)

            fig.update_layout(
                title_text="Lag Order Selection - Information Criteria",
                height=400,
                width=300 * n_criteria,
                showlegend=False,
            )

            return fig

        elif backend == "matplotlib":
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                raise ImportError(
                    "Matplotlib is required for this plot. Install with: pip install matplotlib"
                )

            criteria = [c for c in ["AIC", "BIC", "HQIC", "MBIC"] if c in self.criteria_df.columns]
            n_criteria = len(criteria)

            fig, axes = plt.subplots(1, n_criteria, figsize=(5 * n_criteria, 4))
            if n_criteria == 1:
                axes = [axes]

            for ax, criterion in zip(axes, criteria):
                # Plot line
                ax.plot(
                    self.criteria_df["lags"],
                    self.criteria_df[criterion],
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    label=criterion,
                )

                # Mark selected lag
                if criterion in self.selected:
                    selected_lag = self.selected[criterion]
                    mask = self.criteria_df["lags"] == selected_lag
                    selected_value = self.criteria_df.loc[mask, criterion].values[0]

                    ax.plot(
                        selected_lag,
                        selected_value,
                        marker="*",
                        markersize=15,
                        color="red",
                        markeredgewidth=2,
                        markeredgecolor="darkred",
                        label=f"Selected (p={selected_lag})",
                    )

                ax.set_xlabel("Lag Order")
                ax.set_ylabel(criterion)
                ax.set_title(criterion)
                ax.grid(True, alpha=0.3)
                ax.legend()

            fig.suptitle("Lag Order Selection - Information Criteria", fontsize=14, y=1.02)
            plt.tight_layout()

            return fig

        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'plotly' or 'matplotlib'.")

    def __repr__(self) -> str:
        """String representation."""
        return f"LagOrderResult(selected_by_BIC={self.selected.get('BIC', 'N/A')})"
