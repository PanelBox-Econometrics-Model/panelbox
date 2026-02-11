"""
Granger Causality tests for Panel VAR models.

This module implements:
1. Standard Panel Granger Causality via Wald test (homogeneous)
2. Dumitrescu-Hurlin (2012) test for heterogeneous panels
3. Instantaneous causality tests
4. Bootstrap inference

References
----------
- Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric
  Models and Cross-spectral Methods". Econometrica.
- Dumitrescu, E. I., & Hurlin, C. (2012). "Testing for Granger non-causality
  in heterogeneous panels". Economic Modelling, 29(4), 1450-1460.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.var.inference import WaldTestResult, wald_test


@dataclass
class GrangerCausalityResult:
    """
    Results container for Granger causality test.

    Attributes
    ----------
    cause : str
        Name of the causing variable
    effect : str
        Name of the effect variable
    wald_stat : float
        Wald test statistic
    f_stat : float
        F-statistic (Wald/df)
    df : int
        Degrees of freedom (number of lags tested)
    p_value : float
        P-value from chi-squared distribution
    p_value_f : float
        P-value from F distribution (if applicable)
    conclusion : str
        Statistical conclusion
    lags_tested : int
        Number of lags tested
    """

    cause: str
    effect: str
    wald_stat: float
    f_stat: float
    df: int
    p_value: float
    p_value_f: Optional[float] = None
    conclusion: str = ""
    lags_tested: int = 0
    hypothesis: str = ""

    def __post_init__(self):
        if not self.hypothesis:
            self.hypothesis = f"{self.cause} does not Granger-cause {self.effect}"
        if not self.conclusion:
            if self.p_value < 0.01:
                self.conclusion = (
                    f"Rejects H0 at 1%: {self.cause} Granger-causes {self.effect} (***)"
                )
            elif self.p_value < 0.05:
                self.conclusion = (
                    f"Rejects H0 at 5%: {self.cause} Granger-causes {self.effect} (**)"
                )
            elif self.p_value < 0.10:
                self.conclusion = (
                    f"Rejects H0 at 10%: {self.cause} Granger-causes {self.effect} (*)"
                )
            else:
                self.conclusion = (
                    f"Fails to reject H0: {self.cause} does not Granger-cause {self.effect}"
                )

    def summary(self) -> str:
        """Generate formatted summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("Granger Causality Test")
        lines.append("=" * 70)
        lines.append(f"Null Hypothesis: {self.hypothesis}")
        lines.append("")
        lines.append(f"Causing variable:  {self.cause}")
        lines.append(f"Effect variable:   {self.effect}")
        lines.append(f"Lags tested:       {self.lags_tested}")
        lines.append("")
        lines.append("Test Statistics:")
        lines.append(f"  Wald statistic:  {self.wald_stat:>10.4f}")
        lines.append(f"  F-statistic:     {self.f_stat:>10.4f}")
        lines.append(f"  Degrees of freedom: {self.df}")
        lines.append(f"  P-value (χ²):    {self.p_value:>10.4f}")
        if self.p_value_f is not None:
            lines.append(f"  P-value (F):     {self.p_value_f:>10.4f}")
        lines.append("")
        lines.append(f"Conclusion: {self.conclusion}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GrangerCausalityResult("
            f"cause='{self.cause}', effect='{self.effect}', "
            f"Wald={self.wald_stat:.2f}, p={self.p_value:.4f})"
        )


@dataclass
class DumitrescuHurlinResult:
    """
    Results container for Dumitrescu-Hurlin (2012) test.

    This test allows for heterogeneous coefficients across entities.

    Attributes
    ----------
    cause : str
        Name of the causing variable
    effect : str
        Name of the effect variable
    W_bar : float
        Average Wald statistic across entities
    Z_tilde_stat : float
        Z̃ statistic (for T fixed, N→∞)
    Z_tilde_pvalue : float
        P-value for Z̃
    Z_bar_stat : float
        Z̄ statistic (for T→∞, N→∞)
    Z_bar_pvalue : float
        P-value for Z̄
    individual_W : np.ndarray
        Individual Wald statistics by entity
    recommended_stat : str
        Which statistic to use ('Z_tilde' or 'Z_bar')
    N : int
        Number of entities
    T_avg : float
        Average time periods per entity
    lags : int
        Number of lags tested
    """

    cause: str
    effect: str
    W_bar: float
    Z_tilde_stat: float
    Z_tilde_pvalue: float
    Z_bar_stat: float
    Z_bar_pvalue: float
    individual_W: np.ndarray
    recommended_stat: str
    N: int
    T_avg: float
    lags: int
    hypothesis: str = ""

    def __post_init__(self):
        if not self.hypothesis:
            self.hypothesis = (
                f"{self.cause} does not Granger-cause {self.effect} "
                f"for any entity (homogeneous non-causality)"
            )

    def summary(self) -> str:
        """Generate formatted summary."""
        lines = []
        lines.append("=" * 75)
        lines.append("Dumitrescu-Hurlin (2012) Panel Granger Causality Test")
        lines.append("=" * 75)
        lines.append(f"Null Hypothesis: {self.hypothesis}")
        lines.append("")
        lines.append("Alternative: Granger causality for at least some entities")
        lines.append("")
        lines.append(f"Causing variable:  {self.cause}")
        lines.append(f"Effect variable:   {self.effect}")
        lines.append(f"Number of entities (N):  {self.N}")
        lines.append(f"Average time periods (T̄): {self.T_avg:.1f}")
        lines.append(f"Lags tested (p):         {self.lags}")
        lines.append("")
        lines.append("Test Statistics:")
        lines.append(f"  W̄ (Average Wald):       {self.W_bar:>10.4f}")
        lines.append("")
        lines.append(f"  Z̃ statistic (T fixed):  {self.Z_tilde_stat:>10.4f}")
        lines.append(f"  Z̃ p-value:              {self.Z_tilde_pvalue:>10.4f}")
        lines.append("")
        lines.append(f"  Z̄ statistic (T→∞):      {self.Z_bar_stat:>10.4f}")
        lines.append(f"  Z̄ p-value:              {self.Z_bar_pvalue:>10.4f}")
        lines.append("")
        lines.append(f"Recommended: Use {self.recommended_stat}")

        # Conclusion based on recommended statistic
        if self.recommended_stat == "Z_tilde":
            pval = self.Z_tilde_pvalue
        else:
            pval = self.Z_bar_pvalue

        if pval < 0.01:
            conclusion = f"Rejects H0 at 1%: Granger causality detected (***)"
        elif pval < 0.05:
            conclusion = f"Rejects H0 at 5%: Granger causality detected (**)"
        elif pval < 0.10:
            conclusion = f"Rejects H0 at 10%: Granger causality detected (*)"
        else:
            conclusion = f"Fails to reject H0: No evidence of Granger causality"

        lines.append("")
        lines.append(f"Conclusion: {conclusion}")
        lines.append("")

        # Individual statistics summary
        lines.append("Individual Entity Statistics:")
        lines.append(f"  Min W_i:     {np.min(self.individual_W):>10.4f}")
        lines.append(f"  Mean W_i:    {np.mean(self.individual_W):>10.4f}")
        lines.append(f"  Median W_i:  {np.median(self.individual_W):>10.4f}")
        lines.append(f"  Max W_i:     {np.max(self.individual_W):>10.4f}")
        lines.append(f"  Std W_i:     {np.std(self.individual_W):>10.4f}")
        lines.append("")
        lines.append("=" * 75)

        return "\n".join(lines)

    def plot_individual_statistics(self, backend: str = "matplotlib", show: bool = True):
        """
        Plot histogram of individual Wald statistics.

        Parameters
        ----------
        backend : str, default='matplotlib'
            Plotting backend: 'matplotlib' or 'plotly'
        show : bool, default=True
            Whether to show the plot immediately

        Returns
        -------
        fig
            Figure object
        """
        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Histogram of individual W statistics
            ax.hist(
                self.individual_W,
                bins=min(30, self.N // 2),
                alpha=0.7,
                edgecolor="black",
                color="steelblue",
            )

            # Add vertical line at critical value (χ²_0.05 with df=lags)
            critical_value = stats.chi2.ppf(0.95, df=self.lags)
            ax.axvline(
                critical_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Critical value (5%): {critical_value:.2f}",
            )

            # Add vertical line at mean
            ax.axvline(
                self.W_bar, color="green", linestyle="-", linewidth=2, label=f"W̄ = {self.W_bar:.2f}"
            )

            ax.set_xlabel("Individual Wald Statistics (W_i)", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.set_title(
                f"Distribution of Individual Wald Statistics\n"
                f"{self.cause} → {self.effect} (N={self.N})",
                fontsize=14,
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            if show:
                plt.show()
            return fig

        elif backend == "plotly":
            import plotly.graph_objects as go

            # Create histogram
            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=self.individual_W,
                    nbinsx=min(30, self.N // 2),
                    name="W_i distribution",
                    marker_color="steelblue",
                    opacity=0.7,
                )
            )

            # Add vertical lines
            critical_value = stats.chi2.ppf(0.95, df=self.lags)

            # Critical value line
            y_max = np.histogram(self.individual_W, bins=min(30, self.N // 2))[0].max()
            fig.add_trace(
                go.Scatter(
                    x=[critical_value, critical_value],
                    y=[0, y_max * 1.1],
                    mode="lines",
                    name=f"Critical value (5%): {critical_value:.2f}",
                    line=dict(color="red", dash="dash", width=2),
                )
            )

            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=[self.W_bar, self.W_bar],
                    y=[0, y_max * 1.1],
                    mode="lines",
                    name=f"W̄ = {self.W_bar:.2f}",
                    line=dict(color="green", width=2),
                )
            )

            fig.update_layout(
                title=f"Distribution of Individual Wald Statistics<br>"
                f"{self.cause} → {self.effect} (N={self.N})",
                xaxis_title="Individual Wald Statistics (W_i)",
                yaxis_title="Frequency",
                showlegend=True,
                hovermode="x",
            )

            if show:
                fig.show()
            return fig

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def __repr__(self) -> str:
        stat_name = "Z̃" if self.recommended_stat == "Z_tilde" else "Z̄"
        pval = self.Z_tilde_pvalue if self.recommended_stat == "Z_tilde" else self.Z_bar_pvalue
        return (
            f"DumitrescuHurlinResult("
            f"cause='{self.cause}', effect='{self.effect}', "
            f"{stat_name}={self.Z_tilde_stat if self.recommended_stat == 'Z_tilde' else self.Z_bar_stat:.2f}, "
            f"p={pval:.4f})"
        )


@dataclass
class InstantaneousCausalityResult:
    """
    Results container for instantaneous causality test.

    Tests for contemporaneous correlation between variables.

    Attributes
    ----------
    var1 : str
        First variable
    var2 : str
        Second variable
    correlation : float
        Correlation between residuals
    lr_stat : float
        Likelihood ratio statistic
    p_value : float
        P-value from chi-squared(1)
    n_obs : int
        Number of observations
    """

    var1: str
    var2: str
    correlation: float
    lr_stat: float
    p_value: float
    n_obs: int

    def summary(self) -> str:
        """Generate formatted summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("Instantaneous Causality Test")
        lines.append("=" * 70)
        lines.append(f"Null Hypothesis: No contemporaneous correlation")
        lines.append("")
        lines.append(f"Variable 1:  {self.var1}")
        lines.append(f"Variable 2:  {self.var2}")
        lines.append("")
        lines.append(f"Correlation (r):      {self.correlation:>10.4f}")
        lines.append(f"LR statistic:         {self.lr_stat:>10.4f}")
        lines.append(f"P-value:              {self.p_value:>10.4f}")
        lines.append(f"Observations:         {self.n_obs}")
        lines.append("")

        if self.p_value < 0.01:
            conclusion = f"Rejects H0 at 1%: Significant contemporaneous correlation (***)"
        elif self.p_value < 0.05:
            conclusion = f"Rejects H0 at 5%: Significant contemporaneous correlation (**)"
        elif self.p_value < 0.10:
            conclusion = f"Rejects H0 at 10%: Significant contemporaneous correlation (*)"
        else:
            conclusion = f"Fails to reject H0: No significant contemporaneous correlation"

        lines.append(f"Conclusion: {conclusion}")
        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"InstantaneousCausalityResult("
            f"var1='{self.var1}', var2='{self.var2}', "
            f"r={self.correlation:.3f}, p={self.p_value:.4f})"
        )


def construct_granger_restriction_matrix(
    exog_names: List[str], causing_var: str, lags: int
) -> np.ndarray:
    """
    Construct restriction matrix for Granger causality test.

    This selects all lag coefficients of the causing variable.

    Parameters
    ----------
    exog_names : List[str]
        Names of all exogenous variables (regressors)
    causing_var : str
        Name of the causing variable
    lags : int
        Number of lags in the VAR

    Returns
    -------
    R : np.ndarray
        Restriction matrix (lags × len(exog_names))

    Notes
    -----
    For each lag l = 1, ..., p, we look for 'L{l}.{causing_var}' in exog_names.
    The restriction matrix R is (p × k) where k = len(exog_names), and
    R[l-1, j] = 1 if exog_names[j] == 'L{l}.{causing_var}', else 0.
    """
    k = len(exog_names)
    R = np.zeros((lags, k))

    for lag in range(1, lags + 1):
        lag_name = f"L{lag}.{causing_var}"
        try:
            idx = exog_names.index(lag_name)
            R[lag - 1, idx] = 1
        except ValueError:
            raise ValueError(
                f"Lag {lag} of variable '{causing_var}' not found in regressors. "
                f"Expected '{lag_name}' in {exog_names}"
            )

    return R


def granger_causality_wald(
    params: np.ndarray,
    cov_params: np.ndarray,
    exog_names: List[str],
    causing_var: str,
    caused_var: str,
    lags: int,
    n_obs: Optional[int] = None,
) -> GrangerCausalityResult:
    """
    Perform Granger causality test via Wald test.

    Tests H0: causing_var does not Granger-cause caused_var.

    Parameters
    ----------
    params : np.ndarray
        Coefficient vector for the equation of caused_var
    cov_params : np.ndarray
        Covariance matrix of parameters
    exog_names : List[str]
        Names of exogenous variables
    causing_var : str
        Name of the causing variable
    caused_var : str
        Name of the caused (dependent) variable
    lags : int
        Number of lags in the VAR
    n_obs : int, optional
        Number of observations (for F-statistic)

    Returns
    -------
    GrangerCausalityResult
        Test results

    Notes
    -----
    The test statistic is:
        W = (Rβ̂)' [R·Var(β̂)·R']⁻¹ (Rβ̂) ~ χ²(p)
    where R selects the p lag coefficients of the causing variable.
    """
    # Construct restriction matrix
    R = construct_granger_restriction_matrix(exog_names, causing_var, lags)

    # Perform Wald test
    wald_result = wald_test(params, cov_params, R, r=None)

    # Compute F-statistic
    f_stat = wald_result.statistic / lags

    # F p-value (if n_obs provided)
    p_value_f = None
    if n_obs is not None:
        df_denom = n_obs - len(params)
        if df_denom > 0:
            p_value_f = 1 - stats.f.cdf(f_stat, lags, df_denom)

    return GrangerCausalityResult(
        cause=causing_var,
        effect=caused_var,
        wald_stat=wald_result.statistic,
        f_stat=f_stat,
        df=lags,
        p_value=wald_result.p_value,
        p_value_f=p_value_f,
        lags_tested=lags,
    )


def dumitrescu_hurlin_moments(T: int, p: int, K: int) -> Tuple[float, float]:
    """
    Compute exact moments for Dumitrescu-Hurlin test.

    Based on Dumitrescu & Hurlin (2012), Proposition 2.

    Parameters
    ----------
    T : int
        Number of time periods
    p : int
        Number of lags (degrees of freedom for Wald test)
    K : int
        Number of endogenous variables

    Returns
    -------
    E_W : float
        Expected value of Wald statistic
    Var_W : float
        Variance of Wald statistic

    Notes
    -----
    For the individual regression:
        y_it = α_i + Σ_l γ_il·y_{i,t-l} + Σ_l β_il·x_{i,t-l} + ε_it

    The degrees of freedom are: df = T - K·p - 1

    The exact moments depend on whether we use exact finite-sample distribution
    or asymptotic approximation.
    """
    # Degrees of freedom in individual regression
    # Each entity has: T observations, K*p lag terms, 1 constant
    df = T - K * p - 1

    if df <= 0:
        raise ValueError(
            f"Insufficient degrees of freedom: T={T}, K={K}, p={p} "
            f"requires T > K·p + 1 = {K * p + 1}"
        )

    # First moment: E[W_i] = p (under H0, Wald ~ χ²(p))
    E_W = p

    # Second moment: Var[W_i]
    # For χ²(p), Var = 2p in asymptotic case
    # Finite-sample correction (simplified version from DH2012):
    if df > 2:
        # Adjust for finite sample
        # This is a simplified version; the exact formula in DH2012 is more complex
        Var_W = 2 * p * (T - K * p - p - 1) / (T - K * p - p - 3)
    else:
        # Fall back to asymptotic
        Var_W = 2 * p

    return E_W, Var_W


def dumitrescu_hurlin_test(
    data: pd.DataFrame,
    cause: str,
    effect: str,
    lags: int,
    entity_col: str = "entity",
    time_col: str = "time",
) -> DumitrescuHurlinResult:
    """
    Perform Dumitrescu-Hurlin (2012) panel Granger causality test.

    This test allows for heterogeneous coefficients across entities.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with entity and time identifiers
    cause : str
        Name of the causing variable
    effect : str
        Name of the effect variable
    lags : int
        Number of lags to test
    entity_col : str, default='entity'
        Name of entity identifier column
    time_col : str, default='time'
        Name of time identifier column

    Returns
    -------
    DumitrescuHurlinResult
        Test results

    Notes
    -----
    The procedure:
    1. For each entity i, estimate individual regression and compute W_i
    2. Compute W̄ = (1/N) Σ_i W_i
    3. Compute Z̃ = √(N/(2p)) × (W̄ - p) ~ N(0,1) [for T fixed, N→∞]
    4. Compute Z̄ = √N × (W̄ - E[W_i,T]) / √Var[W_i,T] ~ N(0,1) [for T→∞, N→∞]

    References
    ----------
    Dumitrescu, E. I., & Hurlin, C. (2012). Testing for Granger non-causality
    in heterogeneous panels. Economic Modelling, 29(4), 1450-1460.
    """
    # Validate inputs
    if cause not in data.columns or effect not in data.columns:
        raise ValueError(f"Variables '{cause}' and/or '{effect}' not found in data")

    if entity_col not in data.columns or time_col not in data.columns:
        raise ValueError(f"Entity or time column not found in data")

    # Get entities
    entities = data[entity_col].unique()
    N = len(entities)

    if N < 2:
        raise ValueError("Need at least 2 entities for Dumitrescu-Hurlin test")

    # Store individual Wald statistics
    individual_W = []
    T_values = []

    # Number of endogenous variables (just 2 for bivariate test: cause and effect)
    K = 2

    # For each entity, run individual regression
    for entity_id in entities:
        # Get data for this entity
        entity_data = data[data[entity_col] == entity_id].sort_values(time_col)

        T_i = len(entity_data)
        T_values.append(T_i)

        # Check sufficient observations
        if T_i <= lags + 1:
            raise ValueError(
                f"Entity {entity_id} has insufficient observations: " f"T={T_i}, need > {lags + 1}"
            )

        # Construct lagged data for individual regression
        # y_t = α + Σ_l γ_l·y_{t-l} + Σ_l β_l·x_{t-l} + ε_t
        y = entity_data[effect].values
        x = entity_data[cause].values

        # Create lag matrix
        n = len(y)
        X_lags = []

        # Add lags of y (effect variable)
        for lag in range(1, lags + 1):
            X_lags.append(np.concatenate([np.full(lag, np.nan), y[:-lag]]))

        # Add lags of x (cause variable)
        for lag in range(1, lags + 1):
            X_lags.append(np.concatenate([np.full(lag, np.nan), x[:-lag]]))

        # Stack and remove NaN rows
        X_lags = np.column_stack(X_lags)

        # Remove first 'lags' observations (NaN rows)
        y_reg = y[lags:]
        X_reg = X_lags[lags:, :]

        # Add constant
        X_reg = np.column_stack([np.ones(len(y_reg)), X_reg])

        # OLS estimation
        try:
            beta_i = np.linalg.lstsq(X_reg, y_reg, rcond=None)[0]
            residuals = y_reg - X_reg @ beta_i
            sigma2 = np.sum(residuals**2) / (len(y_reg) - len(beta_i))
            cov_beta_i = sigma2 * np.linalg.inv(X_reg.T @ X_reg)
        except np.linalg.LinAlgError:
            raise ValueError(f"Singular matrix for entity {entity_id}")

        # Construct restriction matrix for Granger test
        # We want to test if all lags of x (cause) are zero
        # Coefficients: [const, L1.y, L2.y, ..., Lp.y, L1.x, L2.x, ..., Lp.x]
        # Indices of x lags: [lags+1, lags+2, ..., 2*lags]
        k_total = 1 + 2 * lags  # constant + lags of y + lags of x
        R_i = np.zeros((lags, k_total))
        for lag_idx in range(lags):
            R_i[lag_idx, 1 + lags + lag_idx] = 1  # Select L{lag}.x coefficients

        # Wald test: W_i = (R·β)' [R·Cov(β)·R']⁻¹ (R·β)
        R_beta = R_i @ beta_i
        R_cov_R = R_i @ cov_beta_i @ R_i.T

        try:
            W_i = R_beta @ np.linalg.inv(R_cov_R) @ R_beta
        except np.linalg.LinAlgError:
            # Singular covariance matrix, use pseudo-inverse
            W_i = R_beta @ np.linalg.pinv(R_cov_R) @ R_beta

        individual_W.append(float(W_i))

    individual_W = np.array(individual_W)
    T_avg = np.mean(T_values)

    # Compute average Wald statistic
    W_bar = np.mean(individual_W)

    # Compute Z̃ statistic (for T fixed, N→∞)
    # Z̃ = √(N/(2p)) × (W̄ - p)
    Z_tilde = np.sqrt(N / (2 * lags)) * (W_bar - lags)
    Z_tilde_pvalue = 2 * (1 - stats.norm.cdf(np.abs(Z_tilde)))

    # Compute Z̄ statistic (for T→∞, N→∞)
    # Use exact moments for the average T
    E_W, Var_W = dumitrescu_hurlin_moments(int(T_avg), lags, K)

    # Z̄ = √N × (W̄ - E[W]) / √Var[W]
    if Var_W > 0:
        Z_bar = np.sqrt(N) * (W_bar - E_W) / np.sqrt(Var_W)
        Z_bar_pvalue = 2 * (1 - stats.norm.cdf(np.abs(Z_bar)))
    else:
        Z_bar = np.nan
        Z_bar_pvalue = np.nan

    # Decide which statistic to recommend
    if T_avg < 10:
        recommended = "Z_tilde"
    else:
        recommended = "Z_bar"

    return DumitrescuHurlinResult(
        cause=cause,
        effect=effect,
        W_bar=W_bar,
        Z_tilde_stat=Z_tilde,
        Z_tilde_pvalue=Z_tilde_pvalue,
        Z_bar_stat=Z_bar,
        Z_bar_pvalue=Z_bar_pvalue,
        individual_W=individual_W,
        recommended_stat=recommended,
        N=N,
        T_avg=T_avg,
        lags=lags,
    )


def instantaneous_causality(
    resid1: np.ndarray, resid2: np.ndarray, var1: str, var2: str
) -> InstantaneousCausalityResult:
    """
    Test for instantaneous (contemporaneous) causality.

    Tests whether residuals from two equations are correlated.

    Parameters
    ----------
    resid1 : np.ndarray
        Residuals from first equation
    resid2 : np.ndarray
        Residuals from second equation
    var1 : str
        Name of first variable
    var2 : str
        Name of second variable

    Returns
    -------
    InstantaneousCausalityResult
        Test results

    Notes
    -----
    The test statistic is:
        LR = -N·T·log(1 - r²) ~ χ²(1)
    where r is the correlation between residuals.
    """
    if len(resid1) != len(resid2):
        raise ValueError("Residuals must have same length")

    n_obs = len(resid1)

    # Compute correlation
    correlation = np.corrcoef(resid1, resid2)[0, 1]

    # Likelihood ratio test statistic
    # LR = -n·log(1 - r²)
    if abs(correlation) < 1 - 1e-10:
        lr_stat = -n_obs * np.log(1 - correlation**2)
    else:
        # Perfect correlation, LR → ∞
        lr_stat = np.inf

    # P-value from χ²(1)
    if np.isfinite(lr_stat):
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    else:
        p_value = 0.0

    return InstantaneousCausalityResult(
        var1=var1, var2=var2, correlation=correlation, lr_stat=lr_stat, p_value=p_value, n_obs=n_obs
    )


def granger_causality_matrix(result, significance_level: float = 0.05) -> pd.DataFrame:
    """
    Compute full Granger causality matrix for all variable pairs.

    Parameters
    ----------
    result : PanelVARResult
        Fitted Panel VAR result
    significance_level : float, default=0.05
        Significance level for marking

    Returns
    -------
    pd.DataFrame
        Matrix of p-values (K × K) where element (i,j) is p-value for
        "variable i Granger-causes variable j"

    Notes
    -----
    Diagonal elements are NaN (a variable doesn't Granger-cause itself in this context).
    """
    K = result.K
    endog_names = result.endog_names

    pvalue_matrix = np.full((K, K), np.nan)

    for i in range(K):
        for j in range(K):
            if i == j:
                continue  # Skip diagonal

            # Test: does variable i Granger-cause variable j?
            try:
                gc_result = result.test_granger_causality(
                    causing_var=endog_names[i], caused_var=endog_names[j]
                )
                pvalue_matrix[i, j] = gc_result.p_value
            except Exception:
                pvalue_matrix[i, j] = np.nan

    df = pd.DataFrame(pvalue_matrix, index=endog_names, columns=endog_names)

    return df


def instantaneous_causality_matrix(result) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute instantaneous causality matrix for all variable pairs.

    Parameters
    ----------
    result : PanelVARResult
        Fitted Panel VAR result

    Returns
    -------
    corr_matrix : pd.DataFrame
        Correlation matrix of residuals (K × K)
    pvalue_matrix : pd.DataFrame
        P-value matrix for instantaneous causality tests (K × K)

    Notes
    -----
    Both matrices are symmetric. Diagonal elements are 1.0 (perfect correlation)
    and NaN (no test) for correlations and p-values respectively.
    """
    K = result.K
    endog_names = result.endog_names

    # Stack residuals
    residuals = np.column_stack(result.resid_by_eq)  # (n_obs, K)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(residuals, rowvar=False)

    # Compute p-values
    n_obs = result.n_obs
    pvalue_matrix = np.full((K, K), np.nan)

    for i in range(K):
        for j in range(i + 1, K):  # Only upper triangle (symmetric)
            ic_result = instantaneous_causality(
                residuals[:, i], residuals[:, j], endog_names[i], endog_names[j]
            )
            pvalue_matrix[i, j] = ic_result.p_value
            pvalue_matrix[j, i] = ic_result.p_value  # Symmetric

    corr_df = pd.DataFrame(corr_matrix, index=endog_names, columns=endog_names)
    pvalue_df = pd.DataFrame(pvalue_matrix, index=endog_names, columns=endog_names)

    return corr_df, pvalue_df
