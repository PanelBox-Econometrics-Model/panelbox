"""
Four-component stochastic frontier model (Kumbhakar et al., 2014).

This module implements the revolutionary four-component model that decomposes
production into:
    y_it = α + x_it'β + μ_i - η_i + v_it - u_it

where:
    μ_i: Random heterogeneity (technology differences)
    η_i: Persistent inefficiency (structural, long-run)
    v_it: Random noise (shocks)
    u_it: Transient inefficiency (managerial, short-run)

This decomposition is CRITICAL for policy-making as it separates:
    - Structural inefficiency (requires investment) from
    - Managerial inefficiency (requires training/incentives)

References:
    Kumbhakar, S. C., Lien, G., & Hardaker, J. B. (2014).
        Technical efficiency in competing panel data models: a study of
        Norwegian grain farming. Journal of Productivity Analysis, 41(2), 321-337.

    Colombi, R., Kumbhakar, S. C., Martini, G., & Vittadini, G. (2014).
        Closed-skew normality in stochastic frontiers with individual effects
        and long/short-run efficiency. Journal of Productivity Analysis, 42, 123-136.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import log_ndtr


def step1_within_estimator(
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Step 1: Within (FE) estimator to separate α_i and ε_it.

    Model: y_it = α_i + x_it'β + ε_it

    The within transformation removes the entity-specific effects α_i
    by demeaning each variable within entities.

    Parameters:
        y: Dependent variable (n,)
        X: Exogenous variables including constant (n, k)
        entity_id: Entity identifier (n,) as integer codes
        time_id: Time identifier (n,) as integer codes

    Returns:
        Dictionary containing:
            - beta: Fixed effects estimates of β (k,)
            - alpha_i: Estimated fixed effects (N,)
            - epsilon_it: Estimated residuals ε_it (n,)
    """
    unique_entities = np.unique(entity_id)
    N = len(unique_entities)
    n = len(y)

    # Demean within entities (within transformation)
    y_demeaned = np.zeros(n)
    X_demeaned = np.zeros_like(X)

    for i in unique_entities:
        mask = entity_id == i
        y_demeaned[mask] = y[mask] - y[mask].mean()
        X_demeaned[mask] = X[mask] - X[mask].mean(axis=0)

    # OLS on demeaned data (drops constant automatically)
    # Remove constant column if present
    X_std = X_demeaned.std(axis=0)
    non_constant_cols = X_std > 1e-10
    X_demeaned_no_const = X_demeaned[:, non_constant_cols]

    if X_demeaned_no_const.shape[1] == 0:
        raise ValueError("No non-constant variables remain after demeaning")

    # Estimate β (only for non-constant variables)
    beta_no_const = np.linalg.lstsq(X_demeaned_no_const, y_demeaned, rcond=None)[0]

    # Reconstruct full β vector (with zeros for constant)
    beta_fe = np.zeros(X.shape[1])
    beta_fe[non_constant_cols] = beta_no_const

    # Compute α_i as mean of (y_it - x_it'β) for each entity
    alpha_i = np.zeros(N)
    for i in unique_entities:
        mask = entity_id == i
        alpha_i[i] = (y[mask] - X[mask] @ beta_fe).mean()

    # Residuals: ε_it = y_it - α_i - x_it'β
    epsilon_it = y - alpha_i[entity_id] - X @ beta_fe

    return {
        "beta": beta_fe,
        "alpha_i": alpha_i,
        "epsilon_it": epsilon_it,
    }


def step2_separate_transient(
    epsilon_it: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Step 2: Separate transient inefficiency (u_it) from noise (v_it).

    For each time period t, we have:
        ε_it = v_it - u_it

    We pool across all periods and estimate a cross-sectional SFA model
    with half-normal inefficiency distribution.

    Parameters:
        epsilon_it: Residuals from Step 1 (n,)
        entity_id: Entity identifier (n,)
        time_id: Time identifier (n,)
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
            - u_it: Transient inefficiency estimates (n,)
            - v_it: Noise estimates (n,)
            - sigma_v: Noise standard deviation
            - sigma_u: Transient inefficiency standard deviation
    """

    def loglik_halfnormal_cross(theta, epsilon):
        """Half-normal log-likelihood for cross-section.

        Model: ε = v - u, where v ~ N(0, σ²_v), u ~ |N(0, σ²_u)|
        """
        ln_sigma_v_sq, ln_sigma_u_sq = theta

        sigma_v_sq = np.exp(ln_sigma_v_sq)
        sigma_u_sq = np.exp(ln_sigma_u_sq)
        sigma_sq = sigma_v_sq + sigma_u_sq
        sigma = np.sqrt(sigma_sq)
        lambda_param = np.sqrt(sigma_u_sq / sigma_v_sq)

        # Log-likelihood (Aigner et al., 1977)
        # L = (2/σ) φ(ε/σ) Φ(-ελ/σ)
        ll = np.sum(
            np.log(2)
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * (epsilon**2) / sigma_sq
            + log_ndtr(-epsilon * lambda_param / sigma)
        )

        return -ll  # Negative for minimization

    # Starting values based on variance decomposition
    epsilon_var = np.var(epsilon_it)
    start = [np.log(epsilon_var / 2), np.log(epsilon_var / 2)]

    # Optimize
    result = minimize(
        loglik_halfnormal_cross,
        start,
        args=(epsilon_it,),
        method="L-BFGS-B",
    )

    if not result.success and verbose:
        print(f"Warning: Step 2 optimization did not converge: {result.message}")

    # Extract parameters
    sigma_v_sq = np.exp(result.x[0])
    sigma_u_sq = np.exp(result.x[1])
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # JLMS (Jondrow et al., 1982) estimates of u_it
    sigma_star_sq = (sigma_v_sq * sigma_u_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)
    mu_star = -epsilon_it * sigma_u_sq / sigma_sq

    # E[u | ε] = σ* [φ(μ*/σ*) / Φ(μ*/σ*) - μ*/σ*]
    #          = σ* [mills_ratio + μ*/σ*]
    arg = mu_star / sigma_star
    phi_arg = stats.norm.pdf(arg)
    Phi_arg = stats.norm.cdf(arg)
    mills = phi_arg / (Phi_arg + 1e-10)

    u_it_estimates = sigma_star * (mills + arg)

    # Residual noise: v_it = ε_it + u_it
    v_it_estimates = epsilon_it + u_it_estimates

    return {
        "u_it": u_it_estimates,
        "v_it": v_it_estimates,
        "sigma_v": sigma_v,
        "sigma_u": sigma_u,
    }


def step3_separate_persistent(
    alpha_i: np.ndarray,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Step 3: Separate persistent inefficiency (η_i) from heterogeneity (μ_i).

    Model: α_i = μ_i - η_i

    We estimate a cross-sectional SFA model on the fixed effects with
    half-normal inefficiency distribution.

    Parameters:
        alpha_i: Fixed effects from Step 1 (N,)
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
            - eta_i: Persistent inefficiency estimates (N,)
            - mu_i: Heterogeneity estimates (N,)
            - sigma_mu: Heterogeneity standard deviation
            - sigma_eta: Persistent inefficiency standard deviation
    """

    def loglik_halfnormal_cross(theta, alpha):
        """Half-normal log-likelihood for cross-section.

        Model: α = μ - η, where μ ~ N(0, σ²_μ), η ~ |N(0, σ²_η)|
        """
        ln_sigma_mu_sq, ln_sigma_eta_sq = theta

        sigma_mu_sq = np.exp(ln_sigma_mu_sq)
        sigma_eta_sq = np.exp(ln_sigma_eta_sq)
        sigma_sq = sigma_mu_sq + sigma_eta_sq
        sigma = np.sqrt(sigma_sq)
        lambda_param = np.sqrt(sigma_eta_sq / sigma_mu_sq)

        # Log-likelihood
        ll = np.sum(
            np.log(2)
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * (alpha**2) / sigma_sq
            + log_ndtr(-alpha * lambda_param / sigma)
        )

        return -ll

    # Starting values
    alpha_var = np.var(alpha_i)
    start = [np.log(alpha_var / 2), np.log(alpha_var / 2)]

    # Optimize
    result = minimize(
        loglik_halfnormal_cross,
        start,
        args=(alpha_i,),
        method="L-BFGS-B",
    )

    if not result.success and verbose:
        print(f"Warning: Step 3 optimization did not converge: {result.message}")

    # Extract parameters
    sigma_mu_sq = np.exp(result.x[0])
    sigma_eta_sq = np.exp(result.x[1])
    sigma_sq = sigma_mu_sq + sigma_eta_sq
    sigma_mu = np.sqrt(sigma_mu_sq)
    sigma_eta = np.sqrt(sigma_eta_sq)

    # JLMS estimates of η_i
    sigma_star_sq = (sigma_mu_sq * sigma_eta_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)
    mu_star = -alpha_i * sigma_eta_sq / sigma_sq

    arg = mu_star / sigma_star
    phi_arg = stats.norm.pdf(arg)
    Phi_arg = stats.norm.cdf(arg)
    mills = phi_arg / (Phi_arg + 1e-10)

    eta_i = sigma_star * (mills + arg)

    # Heterogeneity: μ_i = α_i + η_i
    mu_i = alpha_i + eta_i

    return {
        "eta_i": eta_i,
        "mu_i": mu_i,
        "sigma_mu": sigma_mu,
        "sigma_eta": sigma_eta,
    }


@dataclass
class FourComponentResult:
    """Results from four-component SFA model.

    This class stores all estimated components and provides methods
    to compute various efficiency measures.

    Attributes:
        model: Reference to FourComponentSFA model
        beta: Frontier parameter estimates
        alpha_i: Fixed effects (α_i = μ_i - η_i)
        mu_i: Random heterogeneity estimates
        eta_i: Persistent inefficiency estimates
        u_it: Transient inefficiency estimates
        v_it: Noise estimates
        sigma_v: Noise standard deviation
        sigma_u: Transient inefficiency standard deviation
        sigma_mu: Heterogeneity standard deviation
        sigma_eta: Persistent inefficiency standard deviation
    """

    model: "FourComponentSFA"
    beta: np.ndarray
    alpha_i: np.ndarray
    mu_i: np.ndarray
    eta_i: np.ndarray
    u_it: np.ndarray
    v_it: np.ndarray
    sigma_v: float
    sigma_u: float
    sigma_mu: float
    sigma_eta: float

    def persistent_efficiency(self) -> pd.DataFrame:
        """Compute persistent technical efficiency: TE_p,i = exp(-η_i).

        Persistent efficiency is time-invariant and reflects structural
        factors like location, equipment quality, etc.

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - persistent_efficiency: TE_p,i ∈ (0, 1]
        """
        te_persistent = np.exp(-self.eta_i)

        # Map entity codes to original entity IDs if available
        entity_mapping = self.model.data[self.model.entity].unique()

        df = pd.DataFrame(
            {
                "entity": entity_mapping,
                "persistent_efficiency": te_persistent,
            }
        )

        return df

    def transient_efficiency(self) -> pd.DataFrame:
        """Compute transient technical efficiency: TE_t,it = exp(-u_it).

        Transient efficiency varies over time and reflects managerial
        factors like training, motivation, etc.

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - time: Time identifier
                - transient_efficiency: TE_t,it ∈ (0, 1]
        """
        te_transient = np.exp(-self.u_it)

        # Get original entity and time IDs
        entity_values = self.model.data[self.model.entity].values
        time_values = self.model.data[self.model.time].values

        df = pd.DataFrame(
            {
                "entity": entity_values,
                "time": time_values,
                "transient_efficiency": te_transient,
            }
        )

        return df

    def overall_efficiency(self) -> pd.DataFrame:
        """Compute overall efficiency: TE_o,it = TE_p,i × TE_t,it.

        Overall efficiency is the product of persistent and transient
        efficiency components.

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - time: Time identifier
                - overall_efficiency: TE_o,it ∈ (0, 1]
                - persistent_efficiency: TE_p,i
                - transient_efficiency: TE_t,it
        """
        te_persistent = np.exp(-self.eta_i)
        te_transient = np.exp(-self.u_it)

        # Match persistent to observations
        te_persistent_matched = te_persistent[self.model.entity_id]
        te_overall = te_persistent_matched * te_transient

        # Get original entity and time IDs
        entity_values = self.model.data[self.model.entity].values
        time_values = self.model.data[self.model.time].values

        df = pd.DataFrame(
            {
                "entity": entity_values,
                "time": time_values,
                "overall_efficiency": te_overall,
                "persistent_efficiency": te_persistent_matched,
                "transient_efficiency": te_transient,
            }
        )

        return df

    def decomposition(self) -> pd.DataFrame:
        """Return full decomposition of all 4 components.

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - time: Time identifier
                - mu_i: Random heterogeneity
                - eta_i: Persistent inefficiency
                - u_it: Transient inefficiency
                - v_it: Random noise
        """
        # Get original entity and time IDs
        entity_values = self.model.data[self.model.entity].values
        time_values = self.model.data[self.model.time].values

        df = pd.DataFrame(
            {
                "entity": entity_values,
                "time": time_values,
                "mu_i": self.mu_i[self.model.entity_id],
                "eta_i": self.eta_i[self.model.entity_id],
                "u_it": self.u_it,
                "v_it": self.v_it,
            }
        )

        return df

    def print_summary(self):
        """Print comprehensive summary statistics."""
        print("\n" + "=" * 70)
        print("FOUR-COMPONENT STOCHASTIC FRONTIER MODEL")
        print("=" * 70)

        print(f"\nModel: {self.model.frontier_type}")
        print(f"Method: Multi-step estimation (Kumbhakar et al., 2014)")

        print(f"\nSample:")
        print(f"  Observations:  {self.model.n_obs:>10,}")
        print(f"  Entities:      {self.model.n_entities:>10,}")
        print(f"  Time periods:  {self.model.n_periods:>10,}")
        print(f"  Balanced:      {self.model.is_balanced}")

        print(f"\nFrontier Coefficients:")
        for i, name in enumerate(self.model.exog_names):
            print(f"  {name:<15} {self.beta[i]:>12.6f}")

        print(f"\nVariance Components:")
        print(f"  σ²_v  (noise):                {self.sigma_v**2:>12.6f}")
        print(f"  σ²_u  (transient ineff.):     {self.sigma_u**2:>12.6f}")
        print(f"  σ²_μ  (heterogeneity):        {self.sigma_mu**2:>12.6f}")
        print(f"  σ²_η  (persistent ineff.):    {self.sigma_eta**2:>12.6f}")

        total_var = self.sigma_v**2 + self.sigma_u**2 + self.sigma_mu**2 + self.sigma_eta**2
        print(f"\n  Total variance:               {total_var:>12.6f}")

        print(f"\nVariance Shares:")
        print(f"  Noise:              {100 * self.sigma_v**2 / total_var:>10.2f}%")
        print(f"  Transient ineff.:   {100 * self.sigma_u**2 / total_var:>10.2f}%")
        print(f"  Heterogeneity:      {100 * self.sigma_mu**2 / total_var:>10.2f}%")
        print(f"  Persistent ineff.:  {100 * self.sigma_eta**2 / total_var:>10.2f}%")

        # Efficiency statistics
        te_persistent = np.exp(-self.eta_i)
        te_transient = np.exp(-self.u_it)
        te_overall = te_persistent[self.model.entity_id] * te_transient

        print(f"\nEfficiency Summary:")
        print(f"  {'Type':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*60}")
        print(
            f"  {'Persistent TE':<20} {te_persistent.mean():>10.4f} "
            f"{te_persistent.std():>10.4f} {te_persistent.min():>10.4f} "
            f"{te_persistent.max():>10.4f}"
        )
        print(
            f"  {'Transient TE':<20} {te_transient.mean():>10.4f} "
            f"{te_transient.std():>10.4f} {te_transient.min():>10.4f} "
            f"{te_transient.max():>10.4f}"
        )
        print(
            f"  {'Overall TE':<20} {te_overall.mean():>10.4f} "
            f"{te_overall.std():>10.4f} {te_overall.min():>10.4f} "
            f"{te_overall.max():>10.4f}"
        )

        # Interpretation guide
        print(f"\nInterpretation:")
        print(f"  Persistent inefficiency (η_i): Structural, long-run")
        print(f"    → Requires investment, location changes, major upgrades")
        print(f"  Transient inefficiency (u_it): Managerial, short-run")
        print(f"    → Requires training, motivation, better organization")

        print("=" * 70)


class FourComponentSFA:
    """Four-component stochastic frontier model (Kumbhakar et al., 2014).

    Decomposes output into 4 components:
        y_it = α + x_it'β + μ_i - η_i + v_it - u_it

    where:
        μ_i: Random heterogeneity (technology differences)
        η_i: Persistent inefficiency (structural, long-run)
        v_it: Random noise (shocks)
        u_it: Transient inefficiency (managerial, short-run)

    Estimation uses the multi-step approach (Kumbhakar et al., 2014):
        Step 1: Within (FE) estimator → α_i, ε_it
        Step 2: SFA on ε_it → u_it, v_it
        Step 3: SFA on α_i → η_i, μ_i

    This decomposition is CRITICAL for policy-making as it identifies:
        - What requires structural investment (η_i)
        - What requires managerial improvement (u_it)

    Parameters:
        data: Panel DataFrame with entity and time identifiers
        depvar: Dependent variable name (usually in logs)
        exog: List of exogenous variable names
        entity: Entity identifier column name
        time: Time identifier column name
        frontier_type: 'production' or 'cost'

    Example:
        >>> # Hospital efficiency analysis
        >>> model = FourComponentSFA(
        ...     data=hospital_panel,
        ...     depvar='log_patients_treated',
        ...     exog=['log_doctors', 'log_nurses', 'log_beds'],
        ...     entity='hospital_id',
        ...     time='year',
        ...     frontier_type='production',
        ... )
        >>> result = model.fit(verbose=True)
        >>>
        >>> # Get efficiency measures
        >>> te_persistent = result.persistent_efficiency()
        >>> te_transient = result.transient_efficiency()
        >>> te_overall = result.overall_efficiency()
        >>>
        >>> # Interpretation:
        >>> # - Low persistent TE → Hospital needs structural investment
        >>> # - Low transient TE → Hospital needs better management
        >>> # - High persistent, low transient → Training can help!
        >>> # - Low persistent, high transient → Investment needed!

    References:
        Kumbhakar, S. C., Lien, G., & Hardaker, J. B. (2014).
            Technical efficiency in competing panel data models: a study of
            Norwegian grain farming. Journal of Productivity Analysis, 41(2), 321-337.

        Colombi, R., Kumbhakar, S. C., Martini, G., & Vittadini, G. (2014).
            Closed-skew normality in stochastic frontiers with individual effects
            and long/short-run efficiency. Journal of Productivity Analysis, 42, 123-136.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        depvar: str,
        exog: List[str],
        entity: str,
        time: str,
        frontier_type: str = "production",
    ):
        """Initialize FourComponentSFA model.

        Parameters:
            data: Panel data with entity and time identifiers
            depvar: Dependent variable (e.g., 'log_output')
            exog: Exogenous variables (e.g., ['log_labor', 'log_capital'])
            entity: Entity identifier (e.g., 'firm_id')
            time: Time identifier (e.g., 'year')
            frontier_type: 'production' or 'cost'
        """
        if entity is None or time is None:
            raise ValueError("Four-component model requires both entity and time identifiers")

        self.data = data.copy()
        self.depvar = depvar
        self.exog = exog
        self.entity = entity
        self.time = time
        self.frontier_type = frontier_type

        # Prepare data
        self._prepare_data()

    def _prepare_data(self):
        """Prepare arrays for estimation."""
        # Sort by entity and time
        self.data = self.data.sort_values([self.entity, self.time]).reset_index(drop=True)

        # Extract arrays
        self.y = self.data[self.depvar].values

        # Exogenous variables
        X = self.data[self.exog].values

        # Add constant if not present
        X_std = X.std(axis=0)
        has_constant = np.any(X_std < 1e-10)

        if not has_constant:
            self.X = np.column_stack([np.ones(len(X)), X])
            self.exog_names = ["const"] + self.exog
        else:
            self.X = X
            self.exog_names = self.exog

        # Entity and time IDs (as integer codes)
        self.entity_id = pd.Categorical(self.data[self.entity]).codes
        self.time_id = pd.Categorical(self.data[self.time]).codes

        # Sample statistics
        self.n_obs = len(self.y)
        self.n_entities = len(np.unique(self.entity_id))
        self.n_periods = len(np.unique(self.time_id))

        # Check if balanced
        entity_counts = pd.Series(self.entity_id).value_counts()
        self.is_balanced = (entity_counts == entity_counts.iloc[0]).all()

    def fit(self, verbose: bool = False) -> FourComponentResult:
        """Fit the four-component model using multi-step estimation.

        Parameters:
            verbose: Whether to print detailed progress

        Returns:
            FourComponentResult with all estimates and efficiency measures
        """
        if verbose:
            print("=" * 70)
            print("FOUR-COMPONENT SFA MODEL - MULTI-STEP ESTIMATION")
            print("=" * 70)
            print(
                f"\nData: {self.n_obs} observations, {self.n_entities} entities, "
                f"{self.n_periods} periods"
            )

        # Step 1: Within estimator
        if verbose:
            print("\n" + "-" * 70)
            print("Step 1: Within (Fixed Effects) Estimator")
            print("-" * 70)

        step1_result = step1_within_estimator(self.y, self.X, self.entity_id, self.time_id)

        if verbose:
            print(f"  Frontier coefficients (β):")
            for i, name in enumerate(self.exog_names):
                print(f"    {name:<15} {step1_result['beta'][i]:>12.6f}")
            print(
                f"  Fixed effects (α_i): [{step1_result['alpha_i'].min():.4f}, "
                f"{step1_result['alpha_i'].max():.4f}]"
            )
            print(
                f"  Residuals (ε_it):    [{step1_result['epsilon_it'].min():.4f}, "
                f"{step1_result['epsilon_it'].max():.4f}]"
            )

        # Step 2: Separate transient inefficiency
        if verbose:
            print("\n" + "-" * 70)
            print("Step 2: Separate Transient Inefficiency (u_it) from Noise (v_it)")
            print("-" * 70)

        step2_result = step2_separate_transient(
            step1_result["epsilon_it"],
            self.entity_id,
            self.time_id,
            verbose=verbose,
        )

        if verbose:
            print(f"  σ_v (noise):                {step2_result['sigma_v']:.6f}")
            print(f"  σ_u (transient inefficiency): {step2_result['sigma_u']:.6f}")
            print(
                f"  λ = σ_u/σ_v:                {step2_result['sigma_u']/step2_result['sigma_v']:.4f}"
            )

        # Step 3: Separate persistent inefficiency
        if verbose:
            print("\n" + "-" * 70)
            print("Step 3: Separate Persistent Inefficiency (η_i) from Heterogeneity (μ_i)")
            print("-" * 70)

        step3_result = step3_separate_persistent(
            step1_result["alpha_i"],
            verbose=verbose,
        )

        if verbose:
            print(f"  σ_μ (heterogeneity):          {step3_result['sigma_mu']:.6f}")
            print(f"  σ_η (persistent inefficiency): {step3_result['sigma_eta']:.6f}")
            print(
                f"  λ = σ_η/σ_μ:                  {step3_result['sigma_eta']/step3_result['sigma_mu']:.4f}"
            )

        # Create result object
        result = FourComponentResult(
            model=self,
            beta=step1_result["beta"],
            alpha_i=step1_result["alpha_i"],
            mu_i=step3_result["mu_i"],
            eta_i=step3_result["eta_i"],
            u_it=step2_result["u_it"],
            v_it=step2_result["v_it"],
            sigma_v=step2_result["sigma_v"],
            sigma_u=step2_result["sigma_u"],
            sigma_mu=step3_result["sigma_mu"],
            sigma_eta=step3_result["sigma_eta"],
        )

        if verbose:
            print("\n" + "=" * 70)
            print("ESTIMATION COMPLETE")
            print("=" * 70)
            result.print_summary()

        return result
