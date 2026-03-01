"""
GMM Results Classes.

===================

Data structures for GMM estimation results and specification tests.

Classes
-------
TestResult : Results from a single specification test
GMMResults : Complete results from GMM estimation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.core.serialization import SerializableMixin

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """
    Results from a specification test.

    Attributes
    ----------
    name : str
        Name of the test (e.g., 'Hansen J-test', 'AR(2) test')
    statistic : float
        Test statistic value
    pvalue : float
        P-value for the test
    df : Optional[int]
        Degrees of freedom (for chi-square tests)
    distribution : str
        Distribution of test statistic ('chi2' or 'normal')
    null_hypothesis : str
        Description of the null hypothesis
    conclusion : str
        Human-readable conclusion ('PASS', 'REJECT', 'WARNING')
    details : Dict
        Additional test-specific details

    Examples
    --------
    >>> test = TestResult(
    ...     name="Hansen J-test",
    ...     statistic=12.5,
    ...     pvalue=0.185,
    ...     df=10,
    ...     distribution="chi2",
    ...     null_hypothesis="Instruments are valid",
    ...     conclusion="PASS",
    ... )
    >>> print(test)
    Hansen J-test: statistic=12.500, p-value=0.185 [PASS]
    """

    name: str
    statistic: float
    pvalue: float
    df: int | None = None
    distribution: str = "chi2"
    null_hypothesis: str = ""
    conclusion: str = ""
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        """Auto-determine conclusion if not provided."""
        if not self.conclusion:
            self.conclusion = self._determine_conclusion()

    def _determine_conclusion(self) -> str:
        """
        Determine test conclusion based on p-value.

        Returns
        -------
        str
            'PASS', 'REJECT', or 'WARNING'
        """
        if self.name.startswith("Hansen") or self.name.startswith("Sargan"):
            # For overid tests: reject if p-value too low OR too high
            if self.pvalue < 0.10:
                return "REJECT"
            elif self.pvalue > 0.25:
                return "WARNING"
            else:
                return "PASS"
        elif "AR(2)" in self.name:
            # For AR(2): should NOT reject (p-value > 0.10)
            if self.pvalue < 0.10:
                return "REJECT"
            else:
                return "PASS"
        elif "AR(1)" in self.name:
            # For AR(1): expected to reject (informational only)
            if self.pvalue < 0.10:
                return "EXPECTED"
            else:
                return "PASS"
        else:
            # Generic: reject if p-value < 0.05
            if self.pvalue < 0.05:
                return "REJECT"
            else:
                return "PASS"

    def __str__(self) -> str:
        """String representation."""
        if self.df is not None:
            return (
                f"{self.name}: statistic={self.statistic:.3f}, "
                f"p-value={self.pvalue:.4f}, df={self.df} [{self.conclusion}]"
            )
        else:
            return (
                f"{self.name}: statistic={self.statistic:.3f}, "
                f"p-value={self.pvalue:.4f} [{self.conclusion}]"
            )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "df": self.df,
            "conclusion": self.conclusion,
        }


@dataclass
class GMMResults(SerializableMixin):
    """
    Results from GMM estimation.

    Attributes
    ----------
    params : pd.Series
        Estimated coefficients
    std_errors : pd.Series
        Standard errors
    tvalues : pd.Series
        T-statistics
    pvalues : pd.Series
        P-values for coefficients
    nobs : int
        Number of observations used
    n_groups : int
        Number of cross-sectional units
    n_instruments : int
        Number of instruments
    n_params : int
        Number of parameters estimated
    hansen_j : TestResult
        Hansen J-test of overidentifying restrictions
    sargan : TestResult
        Sargan test (non-robust version)
    ar1_test : TestResult
        Arellano-Bond AR(1) test
    ar2_test : TestResult
        Arellano-Bond AR(2) test
    diff_hansen : Optional[TestResult]
        Difference-in-Hansen test (System GMM only)
    vcov : np.ndarray
        Variance-covariance matrix of parameters
    weight_matrix : np.ndarray
        GMM weight matrix
    converged : bool
        Whether estimation converged
    two_step : bool
        Whether two-step GMM was used
    windmeijer_corrected : bool
        Whether Windmeijer correction was applied
    model_type : str
        Type of model ('difference' or 'system')
    transformation : str
        Transformation used ('fd' or 'fod')
    residuals : Optional[np.ndarray]
        Model residuals
    fitted_values : Optional[np.ndarray]
        Fitted values

    Examples
    --------
    **Accessing estimation results:**

    >>> from panelbox.gmm import DifferenceGMM
    >>>
    >>> model = DifferenceGMM(data=data, dep_var="y", lags=1, exog_vars=["x1", "x2"])
    >>> results = model.fit()
    >>>
    >>> # Print formatted summary
    >>> print(results.summary())
    >>>
    >>> # Access coefficients
    >>> print(results.params)
    >>> print(f"Persistence: {results.params['L1.y']:.3f}")
    >>>
    >>> # Access standard errors and p-values
    >>> print(results.std_errors)
    >>> print(results.pvalues)
    >>>
    >>> # Confidence intervals
    >>> ci = results.conf_int(alpha=0.05)
    >>> print(ci)

    **Checking diagnostic tests:**

    >>> # Hansen J-test (instruments validity)
    >>> if 0.10 < results.hansen_j.pvalue < 0.25:
    ...     print("✓ Instruments appear valid")
    ... elif results.hansen_j.pvalue < 0.10:
    ...     print("✗ Instruments rejected")
    ... else:
    ...     print("⚠ Warning: p-value very high (possible weak instruments)")
    >>>
    >>> # AR(2) test (moment conditions)
    >>> if results.ar2_test.pvalue > 0.10:
    ...     print("✓ Moment conditions valid")
    ... else:
    ...     print("✗ AR(2) rejected - moment conditions invalid!")
    >>>
    >>> # Instrument ratio (rule of thumb: < 1.0)
    >>> if results.instrument_ratio < 1.0:
    ...     print(f"✓ Instrument ratio acceptable: {results.instrument_ratio:.3f}")
    ... else:
    ...     print(f"⚠ Too many instruments: {results.instrument_ratio:.3f}")

    **Exporting results:**

    >>> # LaTeX table for papers
    >>> latex = results.to_latex()
    >>> with open("gmm_results.tex", "w") as f:
    ...     f.write(latex)
    >>>
    >>> # DataFrame for further analysis
    >>> df = results.to_dataframe()
    >>> print(df)

    **Comparing models:**

    >>> # Estimate both Difference and System GMM
    >>> diff_results = DifferenceGMM(...).fit()
    >>> sys_results = SystemGMM(...).fit()
    >>>
    >>> # Compare coefficient estimates
    >>> print(f"Difference GMM: {diff_results.params['L1.y']:.3f}")
    >>> print(f"System GMM:     {sys_results.params['L1.y']:.3f}")
    >>>
    >>> # Compare efficiency (standard errors)
    >>> diff_se = diff_results.std_errors["L1.y"]
    >>> sys_se = sys_results.std_errors["L1.y"]
    >>> print(f"Efficiency gain: {(diff_se - sys_se) / diff_se * 100:.1f}%")
    >>>
    >>> # Check which model is valid
    >>> diff_valid = (
    ...     diff_results.ar2_test.pvalue > 0.10 and 0.10 < diff_results.hansen_j.pvalue < 0.50
    ... )
    >>> sys_valid = sys_results.ar2_test.pvalue > 0.10 and 0.10 < sys_results.hansen_j.pvalue < 0.50
    >>>
    >>> if diff_valid and sys_valid:
    ...     if sys_se < diff_se * 0.9:
    ...         print("Prefer System GMM (more efficient)")
    ...     else:
    ...         print("Both valid, similar efficiency")

    **Accessing model diagnostics:**

    >>> print(f"Observations: {results.nobs}")
    >>> print(f"Groups: {results.n_groups}")
    >>> print(f"Instruments: {results.n_instruments}")
    >>> print(f"Parameters: {results.n_params}")
    >>> print(f"Model type: {results.model_type}")
    >>> print(f"Two-step: {results.two_step}")
    >>> print(f"Windmeijer corrected: {results.windmeijer_corrected}")

    See Also
    --------
    DifferenceGMM : Arellano-Bond Difference GMM estimator
    SystemGMM : Blundell-Bond System GMM estimator
    """

    # Coefficients
    params: pd.Series
    std_errors: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series

    # Model info
    nobs: int
    n_groups: int
    n_instruments: int
    n_params: int

    # Tests
    hansen_j: TestResult
    sargan: TestResult
    ar1_test: TestResult
    ar2_test: TestResult
    diff_hansen: TestResult | None = None

    # Matrices
    vcov: np.ndarray | None = None
    weight_matrix: np.ndarray | None = None

    # Flags
    converged: bool = True
    two_step: bool = True
    windmeijer_corrected: bool = False

    # Model metadata
    model_type: str = "difference"
    transformation: str = "fd"

    # Additional data
    residuals: np.ndarray | None = None
    fitted_values: np.ndarray | None = None

    # Metadata for predict/forecast
    dep_var: str = ""
    exog_vars: list[str] = field(default_factory=list)
    id_var: str = ""
    time_var: str = ""
    n_lags: int = 1
    endogenous_vars: list[str] = field(default_factory=list)
    predetermined_vars: list[str] = field(default_factory=list)

    @property
    def instrument_ratio(self) -> float:
        """
        Ratio of instruments to groups.

        Rule of thumb: Should be <= 1.0 to avoid instrument proliferation.

        Returns
        -------
        float
            n_instruments / n_groups
        """
        return self.n_instruments / self.n_groups

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Confidence intervals for parameters.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CI)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['lower', 'upper']
        """
        z_crit = stats.norm.ppf(1 - alpha / 2)
        lower = self.params - z_crit * self.std_errors
        upper = self.params + z_crit * self.std_errors

        return pd.DataFrame({"lower": lower, "upper": upper}, index=self.params.index)

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:  # noqa: C901
        """
        Predict dependent variable in LEVELS using estimated coefficients.

        The GMM model is estimated in first-differences but prediction
        is done in levels:
            y_hat_t = rho_1 * y_{t-1} + ... + rho_p * y_{t-p} + beta' * X_t

        Parameters
        ----------
        new_data : pd.DataFrame
            DataFrame containing:
            - Columns for lags of dependent variable (named 'L1.{dep_var}',
              'L2.{dep_var}', etc.) OR the dependent variable itself
              (lags will be computed automatically)
            - Columns for all exogenous variables
            - Columns id_var and time_var

        Returns
        -------
        np.ndarray
            Predicted values in levels. Observations where lags are not
            available (first periods per entity) will be NaN.
        """
        if not self.dep_var:
            raise ValueError(
                "Cannot predict: model metadata (dep_var, exog_vars) not stored. "
                "Re-estimate the model with the latest version of panelbox."
            )

        X_cols = []

        # 1. Lag columns of dependent variable
        for lag in range(1, self.n_lags + 1):
            lag_col_name = f"L{lag}.{self.dep_var}"
            if lag_col_name in new_data.columns:
                X_cols.append(new_data[lag_col_name].values)
            elif self.dep_var in new_data.columns:
                lagged = new_data.groupby(self.id_var)[self.dep_var].shift(lag)
                X_cols.append(lagged.values)
            else:
                raise ValueError(
                    f"Cannot find lag {lag} of '{self.dep_var}'. "
                    f"Provide column '{lag_col_name}' or '{self.dep_var}' in new_data."
                )

        # 2. Exogenous variables
        for var in self.exog_vars:
            if var not in new_data.columns:
                raise ValueError(f"Exogenous variable '{var}' not found in new_data.")
            X_cols.append(new_data[var].values)

        # 3. Endogenous variables (other than lagged dep)
        for var in self.endogenous_vars:
            if var not in new_data.columns:
                raise ValueError(f"Endogenous variable '{var}' not found in new_data.")
            X_cols.append(new_data[var].values)

        # 4. Predetermined variables
        for var in self.predetermined_vars:
            if var not in new_data.columns:
                raise ValueError(f"Predetermined variable '{var}' not found in new_data.")
            X_cols.append(new_data[var].values)

        # 5. Time dummies and constant (match remaining params)
        n_structural = (
            self.n_lags
            + len(self.exog_vars)
            + len(self.endogenous_vars)
            + len(self.predetermined_vars)
        )
        remaining_params = list(self.params.index[n_structural:])
        for param_name in remaining_params:
            if param_name == "_cons":
                X_cols.append(np.ones(len(new_data)))
            elif param_name in new_data.columns:
                X_cols.append(new_data[param_name].values)
            elif param_name.startswith("year_"):
                # Time dummy — try to reconstruct from time_var
                try:
                    year_val = int(param_name.replace("year_", ""))
                    X_cols.append((new_data[self.time_var] == year_val).astype(float).values)
                except (ValueError, KeyError):
                    X_cols.append(np.zeros(len(new_data)))
            else:
                X_cols.append(np.zeros(len(new_data)))

        X = np.column_stack(X_cols)

        if X.shape[1] != len(self.params):
            raise ValueError(
                f"Feature matrix has {X.shape[1]} columns but model has "
                f"{len(self.params)} parameters. Check variable names."
            )

        return X @ self.params.values

    def forecast(
        self,
        last_obs: dict,
        future_exog: pd.DataFrame,
        steps: int = 1,
    ) -> pd.DataFrame:
        """
        Iterative multi-step forecast for dynamic panel models.

        Uses estimated AR coefficients to generate h-step-ahead forecasts,
        feeding predicted values back as lags for subsequent periods.

        Parameters
        ----------
        last_obs : dict
            Dictionary mapping entity_id -> list of last observed values.
            For AR(1): {entity: [y_T]}
            For AR(2): {entity: [y_{T-1}, y_T]} (oldest first)
        future_exog : pd.DataFrame
            DataFrame with future exogenous variable values.
            Must have columns: id_var, time_var, and all exog_vars.
            Must have ``steps`` rows per entity.
        steps : int, default=1
            Number of periods to forecast ahead.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: id_var, time_var, 'forecast'.
            One row per entity per forecast step.
        """
        if not self.dep_var:
            raise ValueError("Cannot forecast: model metadata not stored.")

        # Extract AR coefficients
        ar_coefs = []
        for lag in range(1, self.n_lags + 1):
            lag_name = f"L{lag}.{self.dep_var}"
            if lag_name in self.params.index:
                ar_coefs.append(self.params[lag_name])
            else:
                raise ValueError(f"AR coefficient '{lag_name}' not found in params.")
        ar_coefs = np.array(ar_coefs)

        # Exog + endogenous + predetermined coefficients
        other_var_names = self.exog_vars + self.endogenous_vars + self.predetermined_vars
        beta = np.array([self.params[v] for v in other_var_names])

        all_forecasts = []

        for entity_id, y_history in last_obs.items():
            y_buffer = list(y_history)

            entity_mask = future_exog[self.id_var] == entity_id
            entity_future = future_exog[entity_mask].sort_values(self.time_var)

            if len(entity_future) < steps:
                raise ValueError(
                    f"Entity {entity_id}: expected {steps} future periods, "
                    f"got {len(entity_future)}."
                )

            for h in range(steps):
                # AR component
                ar_component = 0.0
                for lag_idx in range(self.n_lags):
                    if len(y_buffer) > lag_idx:
                        ar_component += ar_coefs[lag_idx] * y_buffer[-(lag_idx + 1)]

                # Exog component
                x_h = entity_future.iloc[h][other_var_names].values.astype(float)
                exog_component = beta @ x_h

                y_hat = ar_component + exog_component

                y_buffer.append(y_hat)

                all_forecasts.append(
                    {
                        self.id_var: entity_id,
                        self.time_var: entity_future.iloc[h][self.time_var],
                        "forecast": y_hat,
                    }
                )

        return pd.DataFrame(all_forecasts)

    def summary(self, title: str | None = None) -> str:
        """
        Generate Stata-style regression table.

        Parameters
        ----------
        title : str, optional
            Custom title for the table

        Returns
        -------
        str
            Formatted regression table
        """
        if title is None:
            title = f"{self.model_type.capitalize()} GMM"
            if self.transformation == "fod":
                title += " (FOD)"

        # Build table
        lines = []
        lines.append("=" * 78)
        lines.append(f"{title:^78}")
        lines.append("=" * 78)

        # Model info
        lines.append(f"Number of observations:     {self.nobs:>10,}")
        lines.append(f"Number of groups:           {self.n_groups:>10,}")
        lines.append(f"Number of instruments:      {self.n_instruments:>10}")
        lines.append(f"Instrument ratio:           {self.instrument_ratio:>10.3f}")

        gmm_type = "Two-step" if self.two_step else "One-step"
        if self.windmeijer_corrected:
            gmm_type += " (Windmeijer)"
        lines.append(f"GMM type:                   {gmm_type:>10}")
        lines.append("-" * 78)

        # Coefficients table
        lines.append(
            f"{'Variable':<20} {'Coef.':>12} {'Std.Err.':>12} "
            f"{'z':>8} {'P>|z|':>8} {'[95% Conf. Int.]':>20}"
        )
        lines.append("-" * 78)

        ci = self.conf_int()
        for var in self.params.index:
            coef = self.params[var]
            se = self.std_errors[var]
            t = self.tvalues[var]
            p = self.pvalues[var]
            ci_lower = ci.loc[var, "lower"]
            ci_upper = ci.loc[var, "upper"]

            # Significance stars
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"

            lines.append(
                f"{var:<20} {coef:>12.6f} {se:>12.6f} "
                f"{t:>8.2f} {p:>8.4f} "
                f"[{ci_lower:>9.6f}, {ci_upper:>9.6f}] {stars}"
            )

        lines.append("=" * 78)

        # Specification tests
        lines.append("Specification Tests:")
        lines.append("-" * 78)
        lines.append(str(self.hansen_j))
        lines.append(str(self.sargan))
        lines.append(str(self.ar1_test))
        lines.append(str(self.ar2_test))
        if self.diff_hansen:
            lines.append(str(self.diff_hansen))
        lines.append("=" * 78)

        # Legend
        lines.append("Significance: * p<0.05, ** p<0.01, *** p<0.001")

        if self.model_type == "difference":
            lines.append("Transformation: First-differences")
        elif self.transformation == "fod":
            lines.append("Transformation: Forward Orthogonal Deviations")

        if self.windmeijer_corrected:
            lines.append("Standard errors: Windmeijer (2005) corrected")
        else:
            lines.append("Standard errors: Robust")

        lines.append("=" * 78)

        return "\n".join(lines)

    def to_latex(self, caption: str = "GMM Estimation Results", label: str = "tab:gmm") -> str:
        """
        Generate LaTeX table.

        Parameters
        ----------
        caption : str
            Table caption
        label : str
            LaTeX label for cross-referencing

        Returns
        -------
        str
            LaTeX table code
        """
        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"    \centering")
        lines.append(f"    \\caption{{{caption}}}")
        lines.append(f"    \\label{{{label}}}")
        lines.append(r"    \begin{tabular}{lcccc}")
        lines.append(r"        \toprule")
        lines.append(r"        Variable & Coefficient & Std. Error & z-stat & P-value \\")
        lines.append(r"        \midrule")

        for var in self.params.index:
            coef = self.params[var]
            se = self.std_errors[var]
            t = self.tvalues[var]
            p = self.pvalues[var]

            # Significance stars
            stars = ""
            if p < 0.001:
                stars = r"^{***}"
            elif p < 0.01:
                stars = r"^{**}"
            elif p < 0.05:
                stars = r"^{*}"

            # Escape underscores in variable names
            var_escaped = var.replace("_", r"\_")

            lines.append(
                f"        {var_escaped} & {coef:.4f}{stars} & {se:.4f} & {t:.2f} & {p:.4f} \\\\"
            )

        lines.append(r"        \midrule")
        lines.append(f"        Observations & \\multicolumn{{4}}{{c}}{{{self.nobs:,}}} \\\\")
        lines.append(f"        Groups & \\multicolumn{{4}}{{c}}{{{self.n_groups:,}}} \\\\")
        lines.append(f"        Instruments & \\multicolumn{{4}}{{c}}{{{self.n_instruments}}} \\\\")
        lines.append(
            f"        Hansen J (p-val) & \\multicolumn{{4}}{{c}}{{{self.hansen_j.pvalue:.4f}}} \\\\"
        )
        lines.append(
            f"        AR(2) (p-val) & \\multicolumn{{4}}{{c}}{{{self.ar2_test.pvalue:.4f}}} \\\\"
        )
        lines.append(r"        \bottomrule")
        lines.append(r"    \end{tabular}")
        lines.append(r"    \begin{tablenotes}")
        lines.append(r"        \small")
        lines.append(r"        \item Significance: * $p<0.05$, ** $p<0.01$, *** $p<0.001$")

        gmm_type = "Two-step" if self.two_step else "One-step"
        if self.windmeijer_corrected:
            gmm_type += " (Windmeijer corrected)"
        lines.append(f"        \\item GMM type: {gmm_type}")

        lines.append(r"    \end{tablenotes}")
        lines.append(r"\end{table}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert results to dictionary.

        Returns
        -------
        dict
            Dictionary with all results
        """
        return {
            "params": self.params.to_dict(),
            "std_errors": self.std_errors.to_dict(),
            "pvalues": self.pvalues.to_dict(),
            "nobs": self.nobs,
            "n_groups": self.n_groups,
            "n_instruments": self.n_instruments,
            "hansen_j": self.hansen_j.to_dict(),
            "sargan": self.sargan.to_dict(),
            "ar1_test": self.ar1_test.to_dict(),
            "ar2_test": self.ar2_test.to_dict(),
            "instrument_ratio": self.instrument_ratio,
            "converged": self.converged,
        }

    def _build_report_data(self) -> dict:
        """
        Build data dictionary compatible with GMMTransformer.

        Returns
        -------
        dict
            Dictionary with keys expected by GMMTransformer.
        """
        coefficients = []
        ci = self.conf_int()
        for var in self.params.index:
            coefficients.append(
                {
                    "name": var,
                    "coef": float(self.params[var]),
                    "se": float(self.std_errors[var]),
                    "tstat": float(self.tvalues[var]),
                    "pvalue": float(self.pvalues[var]),
                    "ci_lower": float(ci.loc[var, "lower"]),
                    "ci_upper": float(ci.loc[var, "upper"]),
                }
            )

        return {
            "model_type": f"{self.model_type.capitalize()} GMM",
            "nobs": self.nobs,
            "n_groups": self.n_groups,
            "n_instruments": self.n_instruments,
            "two_step": self.two_step,
            "coefficients": coefficients,
            "hansen_test": {
                "statistic": float(self.hansen_j.statistic),
                "pvalue": float(self.hansen_j.pvalue),
                "df": self.hansen_j.df,
            },
            "ar_tests": {
                "ar1": {
                    "statistic": float(self.ar1_test.statistic),
                    "pvalue": float(self.ar1_test.pvalue),
                },
                "ar2": {
                    "statistic": float(self.ar2_test.statistic),
                    "pvalue": float(self.ar2_test.pvalue),
                },
            },
        }

    def to_html_report(
        self,
        title: str | None = None,
        subtitle: str | None = None,
    ) -> str:
        """
        Generate a complete, self-contained HTML report with PanelBox branding.

        Parameters
        ----------
        title : str, optional
            Report title. Defaults to "GMM Results".
        subtitle : str, optional
            Report subtitle.

        Returns
        -------
        str
            Complete self-contained HTML report.

        Examples
        --------
        >>> html = results.to_html_report(title="System GMM Results")
        >>> with open("gmm_report.html", "w") as f:
        ...     f.write(html)
        """
        from panelbox.report import GMMTransformer, ReportManager

        transformer = GMMTransformer(self._build_report_data())
        data = transformer.transform()

        mgr = ReportManager()
        return mgr.generate_gmm_report(data, title=title, subtitle=subtitle)

    def __repr__(self) -> str:
        """Repr showing key info."""
        return (
            f"GMMResults(model='{self.model_type}', "
            f"nobs={self.nobs}, n_groups={self.n_groups}, "
            f"n_instruments={self.n_instruments})"
        )
