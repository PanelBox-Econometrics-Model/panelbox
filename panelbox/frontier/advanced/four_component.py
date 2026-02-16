"""Four-component SFA model (Kumbhakar et al., 2014).

This module implements the four-component stochastic frontier model that
decomposes output into:
    - Random heterogeneity (μ_i)
    - Persistent inefficiency (η_i)
    - Random noise (v_it)
    - Transient inefficiency (u_it)

References:
    Kumbhakar, S. C., Lien, G., & Hardaker, J. B. (2014).
        Technical efficiency in competing panel data models: a study of
        Norwegian grain farming. Journal of Productivity Analysis, 41(2), 321-337.

    Colombi, R., Kumbhakar, S. C., Martini, G., & Vittadini, G. (2014).
        Closed-skew normality in stochastic frontiers with individual effects
        and long/short-run efficiency. Journal of Productivity Analysis, 42, 123-136.
"""

from dataclasses import dataclass
from typing import Dict, List

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

    The within estimator applies the fixed effects transformation by
    demeaning the data within each entity. This separates the entity-specific
    effects (α_i) from the time-varying error (ε_it).

    Parameters:
        y: Dependent variable (n,)
        X: Exogenous variables including constant (n, k)
        entity_id: Entity identifier codes (n,)
        time_id: Time period identifier codes (n,)

    Returns:
        Dictionary containing:
            - beta: Fixed effects estimates of β (k,)
            - alpha_i: Estimated fixed effects (N,)
            - epsilon_it: Estimated residuals (n,)

    Notes:
        The within transformation removes the constant term, so β does not
        include an intercept. The fixed effects α_i capture all time-invariant
        heterogeneity.
    """
    unique_entities = np.unique(entity_id)
    N = len(unique_entities)
    n = len(y)

    # Demean within entities
    y_demeaned = np.zeros(n)
    X_demeaned = np.zeros_like(X)

    for i in unique_entities:
        mask = entity_id == i
        y_demeaned[mask] = y[mask] - y[mask].mean()
        X_demeaned[mask] = X[mask] - X[mask].mean(axis=0)

    # OLS on demeaned data
    beta_fe = np.linalg.lstsq(X_demeaned, y_demeaned, rcond=None)[0]

    # Compute α_i as mean of (y_it - x_it'β) for each i
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

    For each time period t:
        ε_it = v_it - u_it
        Estimate SFA cross-section with half-normal inefficiency

    This step applies a cross-sectional SFA model to the residuals from Step 1.
    The pooled estimation assumes constant variance across time periods.

    Parameters:
        epsilon_it: Residuals from Step 1 (n,)
        entity_id: Entity identifier codes (n,)
        time_id: Time period identifier codes (n,)
        verbose: Print convergence warnings

    Returns:
        Dictionary containing:
            - u_it: Transient inefficiency estimates (n,)
            - v_it: Noise estimates (n,)
            - sigma_v: Noise standard deviation
            - sigma_u: Transient inefficiency standard deviation

    Notes:
        Uses JLMS (Jondrow et al., 1982) estimator for inefficiency.
        Assumes half-normal distribution for u_it.
    """
    n = len(epsilon_it)

    def loglik_halfnormal_cross(theta, epsilon):
        """Half-normal log-likelihood for cross-section."""
        ln_sigma_v_sq, ln_sigma_u_sq = theta

        sigma_v_sq = np.exp(ln_sigma_v_sq)
        sigma_u_sq = np.exp(ln_sigma_u_sq)
        sigma_sq = sigma_v_sq + sigma_u_sq
        sigma = np.sqrt(sigma_sq)
        lambda_param = np.sqrt(sigma_u_sq / sigma_v_sq)

        # Log-likelihood
        ll = np.sum(
            np.log(2)
            - np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * (epsilon**2) / sigma_sq
            + log_ndtr(-epsilon * lambda_param / sigma)
        )

        return -ll  # Negative for minimization

    # Starting values
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
        print(f"Warning: Step 2 optimization did not converge")

    # Extract parameters
    sigma_v_sq = np.exp(result.x[0])
    sigma_u_sq = np.exp(result.x[1])
    sigma_sq = sigma_v_sq + sigma_u_sq
    sigma_v = np.sqrt(sigma_v_sq)
    sigma_u = np.sqrt(sigma_u_sq)

    # JLMS estimates of u_it
    sigma_star_sq = (sigma_v_sq * sigma_u_sq) / sigma_sq
    sigma_star = np.sqrt(sigma_star_sq)
    mu_star = -epsilon_it * sigma_u_sq / sigma_sq

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
           Estimate SFA cross-section with half-normal inefficiency

    This step applies a cross-sectional SFA model to the fixed effects from Step 1.
    This separates structural inefficiency (persistent) from random heterogeneity.

    Parameters:
        alpha_i: Fixed effects from Step 1 (N,)
        verbose: Print convergence warnings

    Returns:
        Dictionary containing:
            - eta_i: Persistent inefficiency estimates (N,)
            - mu_i: Heterogeneity estimates (N,)
            - sigma_mu: Heterogeneity standard deviation
            - sigma_eta: Persistent inefficiency standard deviation

    Notes:
        Uses JLMS (Jondrow et al., 1982) estimator for inefficiency.
        Assumes half-normal distribution for η_i.
    """
    N = len(alpha_i)

    def loglik_halfnormal_cross(theta, alpha):
        """Half-normal log-likelihood."""
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
        print(f"Warning: Step 3 optimization did not converge")

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


class FourComponentSFA:
    """Four-component SFA model (Kumbhakar et al., 2014).

    Decomposes output into 4 components:
        y_it = α + x_it'β + μ_i - η_i + v_it - u_it

    where:
        μ_i: Random heterogeneity (technology differences)
        η_i: Persistent inefficiency (structural, long-run)
        v_it: Random noise (shocks)
        u_it: Transient inefficiency (managerial, short-run)

    Estimation: Multi-step approach (Kumbhakar et al., 2014)
        Step 1: Within (FE) estimator → α_i, ε_it
        Step 2: SFA on ε_it → u_it, v_it
        Step 3: SFA on α_i → η_i, μ_i

    Parameters:
        data: Panel DataFrame
        depvar: Dependent variable name
        exog: List of exogenous variable names
        entity: Entity identifier column name
        time: Time identifier column name
        frontier_type: 'production' or 'cost'

    Attributes:
        data: Input panel data
        depvar: Dependent variable name
        exog: Exogenous variable names
        entity: Entity column name
        time: Time column name
        frontier_type: Type of frontier
        y: Dependent variable array
        X: Exogenous variables matrix (with constant)
        entity_id: Entity ID codes
        time_id: Time ID codes
        n_obs: Number of observations
        n_entities: Number of entities
        n_periods: Number of time periods

    Example:
        >>> model = FourComponentSFA(
        ...     data=panel_df,
        ...     depvar='log_output',
        ...     exog=['log_labor', 'log_capital'],
        ...     entity='firm_id',
        ...     time='year',
        ...     frontier_type='production',
        ... )
        >>> result = model.fit()
        >>> te_persistent = result.persistent_efficiency()
        >>> te_transient = result.transient_efficiency()
        >>> te_overall = result.overall_efficiency()

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
        self.data = self.data.sort_values([self.entity, self.time])

        # Extract arrays
        self.y = self.data[self.depvar].values
        X = self.data[self.exog].values

        # Add constant
        self.X = np.column_stack([np.ones(len(X)), X])
        self.exog_names = ["const"] + self.exog

        # Entity and time IDs
        self.entity_id = pd.Categorical(self.data[self.entity]).codes
        self.time_id = pd.Categorical(self.data[self.time]).codes

        self.n_obs = len(self.y)
        self.n_entities = len(np.unique(self.entity_id))
        self.n_periods = len(np.unique(self.time_id))

    def fit(self, verbose: bool = False):
        """Fit the four-component model.

        Parameters:
            verbose: Print estimation progress

        Returns:
            FourComponentResult with all estimates

        Notes:
            This method runs all three estimation steps sequentially:
            1. Within estimator for fixed effects
            2. Cross-sectional SFA for transient inefficiency
            3. Cross-sectional SFA for persistent inefficiency
        """
        if verbose:
            print("=" * 60)
            print("Four-Component SFA Model")
            print("=" * 60)

        # Step 1: Within estimator
        if verbose:
            print("\nStep 1: Within (FE) estimator...")

        step1_result = step1_within_estimator(self.y, self.X, self.entity_id, self.time_id)

        if verbose:
            print(f"  β̂ = {step1_result['beta']}")
            print(
                f"  Range of α̂_i: [{step1_result['alpha_i'].min():.4f}, "
                f"{step1_result['alpha_i'].max():.4f}]"
            )

        # Step 2: Separate transient inefficiency
        if verbose:
            print("\nStep 2: Separating transient inefficiency...")

        step2_result = step2_separate_transient(
            step1_result["epsilon_it"],
            self.entity_id,
            self.time_id,
            verbose=verbose,
        )

        if verbose:
            print(f"  σ̂_v = {step2_result['sigma_v']:.4f} (noise)")
            print(f"  σ̂_u = {step2_result['sigma_u']:.4f} (transient inefficiency)")

        # Step 3: Separate persistent inefficiency
        if verbose:
            print("\nStep 3: Separating persistent inefficiency...")

        step3_result = step3_separate_persistent(
            step1_result["alpha_i"],
            verbose=verbose,
        )

        if verbose:
            print(f"  σ̂_μ = {step3_result['sigma_mu']:.4f} (heterogeneity)")
            print(f"  σ̂_η = {step3_result['sigma_eta']:.4f} (persistent inefficiency)")

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
            print("\n" + "=" * 60)
            print("Estimation complete!")
            print("=" * 60)
            result.print_summary()

        return result


@dataclass
class FourComponentResult:
    """Results from four-component SFA model.

    Attributes:
        model: Reference to FourComponentSFA model
        beta: Coefficient estimates (k,)
        alpha_i: Fixed effects (N,)
        mu_i: Heterogeneity estimates (N,)
        eta_i: Persistent inefficiency estimates (N,)
        u_it: Transient inefficiency estimates (n,)
        v_it: Noise estimates (n,)
        sigma_v: Noise standard deviation
        sigma_u: Transient inefficiency standard deviation
        sigma_mu: Heterogeneity standard deviation
        sigma_eta: Persistent inefficiency standard deviation
    """

    model: "FourComponentSFA"
    beta: np.ndarray
    alpha_i: np.ndarray
    mu_i: np.ndarray  # Heterogeneity
    eta_i: np.ndarray  # Persistent inefficiency
    u_it: np.ndarray  # Transient inefficiency
    v_it: np.ndarray  # Noise
    sigma_v: float
    sigma_u: float
    sigma_mu: float
    sigma_eta: float

    def persistent_efficiency(self) -> pd.DataFrame:
        """Compute persistent technical efficiency: TE_p,i = exp(-η_i).

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - persistent_efficiency: Persistent efficiency scores

        Notes:
            Persistent efficiency ranges from 0 to 1, where 1 indicates
            perfect efficiency in the structural/long-run component.
        """
        te_persistent = np.exp(-self.eta_i)

        df = pd.DataFrame(
            {
                "entity": range(len(te_persistent)),
                "persistent_efficiency": te_persistent,
            }
        )

        return df

    def transient_efficiency(self) -> pd.DataFrame:
        """Compute transient technical efficiency: TE_t,it = exp(-u_it).

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - time: Time identifier
                - transient_efficiency: Transient efficiency scores

        Notes:
            Transient efficiency ranges from 0 to 1, where 1 indicates
            perfect efficiency in the managerial/short-run component.
        """
        te_transient = np.exp(-self.u_it)

        df = pd.DataFrame(
            {
                "entity": self.model.entity_id,
                "time": self.model.time_id,
                "transient_efficiency": te_transient,
            }
        )

        return df

    def overall_efficiency(self) -> pd.DataFrame:
        """Compute overall efficiency: TE_o,it = TE_p,i × TE_t,it.

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - time: Time identifier
                - overall_efficiency: Overall efficiency scores
                - persistent_efficiency: Persistent component
                - transient_efficiency: Transient component

        Notes:
            Overall efficiency is the product of persistent and transient
            efficiency. This gives the total technical efficiency accounting
            for both structural and managerial factors.
        """
        te_persistent = np.exp(-self.eta_i)
        te_transient = np.exp(-self.u_it)

        # Match persistent to observations
        te_persistent_matched = te_persistent[self.model.entity_id]
        te_overall = te_persistent_matched * te_transient

        df = pd.DataFrame(
            {
                "entity": self.model.entity_id,
                "time": self.model.time_id,
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
                - mu_i: Heterogeneity
                - eta_i: Persistent inefficiency
                - u_it: Transient inefficiency
                - v_it: Noise

        Notes:
            This provides the complete decomposition:
                y_it = x_it'β + μ_i - η_i + v_it - u_it
        """
        df = pd.DataFrame(
            {
                "entity": self.model.entity_id,
                "time": self.model.time_id,
                "mu_i": self.mu_i[self.model.entity_id],
                "eta_i": self.eta_i[self.model.entity_id],
                "u_it": self.u_it,
                "v_it": self.v_it,
            }
        )

        return df

    def print_summary(self):
        """Print summary statistics.

        Displays:
            - Sample information
            - Variance components and their shares
            - Efficiency summary statistics
        """
        print("\n" + "=" * 60)
        print("FOUR-COMPONENT SFA RESULTS")
        print("=" * 60)

        print(f"\nSample:")
        print(f"  Observations: {self.model.n_obs}")
        print(f"  Entities: {self.model.n_entities}")
        print(f"  Time periods: {self.model.n_periods}")

        print(f"\nVariance Components:")
        print(f"  σ²_v  (noise):                {self.sigma_v**2:.6f}")
        print(f"  σ²_u  (transient ineff.):     {self.sigma_u**2:.6f}")
        print(f"  σ²_μ  (heterogeneity):        {self.sigma_mu**2:.6f}")
        print(f"  σ²_η  (persistent ineff.):    {self.sigma_eta**2:.6f}")

        total_var = self.sigma_v**2 + self.sigma_u**2 + self.sigma_mu**2 + self.sigma_eta**2
        print(f"\n  Total variance: {total_var:.6f}")

        print(f"\nVariance Shares:")
        print(f"  Noise:              {100 * self.sigma_v**2 / total_var:.2f}%")
        print(f"  Transient ineff.:   {100 * self.sigma_u**2 / total_var:.2f}%")
        print(f"  Heterogeneity:      {100 * self.sigma_mu**2 / total_var:.2f}%")
        print(f"  Persistent ineff.:  {100 * self.sigma_eta**2 / total_var:.2f}%")

        te_persistent = np.exp(-self.eta_i)
        te_transient = np.exp(-self.u_it)
        te_overall = te_persistent[self.model.entity_id] * te_transient

        print(f"\nEfficiency Summary:")
        print(
            f"  Persistent TE:  mean={te_persistent.mean():.4f}, "
            f"std={te_persistent.std():.4f}, "
            f"min={te_persistent.min():.4f}, max={te_persistent.max():.4f}"
        )
        print(
            f"  Transient TE:   mean={te_transient.mean():.4f}, "
            f"std={te_transient.std():.4f}, "
            f"min={te_transient.min():.4f}, max={te_transient.max():.4f}"
        )
        print(
            f"  Overall TE:     mean={te_overall.mean():.4f}, "
            f"std={te_overall.std():.4f}, "
            f"min={te_overall.min():.4f}, max={te_overall.max():.4f}"
        )

        print("=" * 60)

    def bootstrap(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        random_state: int = None,
        verbose: bool = False,
    ) -> "BootstrapResult":
        """Compute bootstrap confidence intervals for efficiency estimates.

        Parameters:
            n_bootstrap: Number of bootstrap replications
            confidence_level: Confidence level for intervals (default 0.95)
            random_state: Random seed for reproducibility
            verbose: Print progress

        Returns:
            BootstrapResult with confidence intervals

        Notes:
            Uses parametric bootstrap by resampling entities with replacement.
            This preserves the panel structure within each entity.
        """
        if random_state is not None:
            np.random.seed(random_state)

        alpha = 1 - confidence_level
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100

        # Storage for bootstrap estimates
        persistent_eff_boot = np.zeros((n_bootstrap, self.model.n_entities))
        transient_eff_boot = np.zeros((n_bootstrap, self.model.n_obs))
        overall_eff_boot = np.zeros((n_bootstrap, self.model.n_obs))

        variance_components_boot = {
            "sigma_v": np.zeros(n_bootstrap),
            "sigma_u": np.zeros(n_bootstrap),
            "sigma_mu": np.zeros(n_bootstrap),
            "sigma_eta": np.zeros(n_bootstrap),
        }

        if verbose:
            print(f"\nRunning {n_bootstrap} bootstrap replications...")

        unique_entities = np.unique(self.model.entity_id)

        for b in range(n_bootstrap):
            if verbose and (b + 1) % 10 == 0:
                print(f"  Replication {b + 1}/{n_bootstrap}")

            # Resample entities with replacement
            boot_entities = np.random.choice(
                unique_entities, size=len(unique_entities), replace=True
            )

            # Create bootstrap sample
            boot_indices = []
            for entity in boot_entities:
                entity_indices = np.where(self.model.entity_id == entity)[0]
                boot_indices.extend(entity_indices)

            boot_indices = np.array(boot_indices)

            # Extract bootstrap data
            y_boot = self.model.y[boot_indices]
            X_boot = self.model.X[boot_indices]
            entity_id_boot = self.model.entity_id[boot_indices]
            time_id_boot = self.model.time_id[boot_indices]

            # Reindex entity IDs
            entity_map = {old: new for new, old in enumerate(np.unique(entity_id_boot))}
            entity_id_boot = np.array([entity_map[e] for e in entity_id_boot])

            try:
                # Estimate model on bootstrap sample
                step1_boot = step1_within_estimator(y_boot, X_boot, entity_id_boot, time_id_boot)

                step2_boot = step2_separate_transient(
                    step1_boot["epsilon_it"],
                    entity_id_boot,
                    time_id_boot,
                    verbose=False,
                )

                step3_boot = step3_separate_persistent(
                    step1_boot["alpha_i"],
                    verbose=False,
                )

                # Store efficiency estimates
                # Note: Bootstrap sample may have different number of unique entities
                n_entities_boot = len(step3_boot["eta_i"])
                persistent_eff_boot[b, :n_entities_boot] = np.exp(-step3_boot["eta_i"])
                if n_entities_boot < self.model.n_entities:
                    persistent_eff_boot[b, n_entities_boot:] = np.nan

                transient_eff_boot[b] = np.exp(-step2_boot["u_it"])
                overall_eff_boot[b] = (
                    np.exp(-step3_boot["eta_i"])[entity_id_boot] * transient_eff_boot[b]
                )

                # Store variance components
                variance_components_boot["sigma_v"][b] = step2_boot["sigma_v"]
                variance_components_boot["sigma_u"][b] = step2_boot["sigma_u"]
                variance_components_boot["sigma_mu"][b] = step3_boot["sigma_mu"]
                variance_components_boot["sigma_eta"][b] = step3_boot["sigma_eta"]

            except Exception as e:
                if verbose:
                    print(f"  Warning: Bootstrap replication {b+1} failed: {e}")
                # Use NaN for failed replications
                persistent_eff_boot[b] = np.nan
                transient_eff_boot[b] = np.nan
                overall_eff_boot[b] = np.nan
                for key in variance_components_boot:
                    variance_components_boot[key][b] = np.nan

        if verbose:
            print("  Bootstrap complete!")

        # Compute confidence intervals
        persistent_ci = np.nanpercentile(persistent_eff_boot, [lower_pct, upper_pct], axis=0)
        transient_ci = np.nanpercentile(transient_eff_boot, [lower_pct, upper_pct], axis=0)
        overall_ci = np.nanpercentile(overall_eff_boot, [lower_pct, upper_pct], axis=0)

        variance_ci = {
            key: np.nanpercentile(vals, [lower_pct, upper_pct])
            for key, vals in variance_components_boot.items()
        }

        return BootstrapResult(
            result=self,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            persistent_ci=persistent_ci,
            transient_ci=transient_ci,
            overall_ci=overall_ci,
            variance_ci=variance_ci,
            persistent_eff_boot=persistent_eff_boot,
            transient_eff_boot=transient_eff_boot,
            overall_eff_boot=overall_eff_boot,
            variance_components_boot=variance_components_boot,
        )


@dataclass
class BootstrapResult:
    """Results from bootstrap inference.

    Attributes:
        result: Original FourComponentResult
        n_bootstrap: Number of bootstrap replications
        confidence_level: Confidence level used
        persistent_ci: CI for persistent efficiency (2, N)
        transient_ci: CI for transient efficiency (2, n)
        overall_ci: CI for overall efficiency (2, n)
        variance_ci: CI for variance components
        persistent_eff_boot: All bootstrap samples for persistent efficiency
        transient_eff_boot: All bootstrap samples for transient efficiency
        overall_eff_boot: All bootstrap samples for overall efficiency
        variance_components_boot: All bootstrap samples for variance components
    """

    result: FourComponentResult
    n_bootstrap: int
    confidence_level: float
    persistent_ci: np.ndarray
    transient_ci: np.ndarray
    overall_ci: np.ndarray
    variance_ci: Dict[str, np.ndarray]
    persistent_eff_boot: np.ndarray
    transient_eff_boot: np.ndarray
    overall_eff_boot: np.ndarray
    variance_components_boot: Dict[str, np.ndarray]

    def persistent_efficiency_ci(self) -> pd.DataFrame:
        """Return persistent efficiency with confidence intervals.

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - persistent_efficiency: Point estimate
                - ci_lower: Lower bound of CI
                - ci_upper: Upper bound of CI
        """
        te_persistent = np.exp(-self.result.eta_i)

        df = pd.DataFrame(
            {
                "entity": range(len(te_persistent)),
                "persistent_efficiency": te_persistent,
                "ci_lower": self.persistent_ci[0],
                "ci_upper": self.persistent_ci[1],
            }
        )

        return df

    def transient_efficiency_ci(self) -> pd.DataFrame:
        """Return transient efficiency with confidence intervals.

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - time: Time identifier
                - transient_efficiency: Point estimate
                - ci_lower: Lower bound of CI
                - ci_upper: Upper bound of CI
        """
        te_transient = np.exp(-self.result.u_it)

        df = pd.DataFrame(
            {
                "entity": self.result.model.entity_id,
                "time": self.result.model.time_id,
                "transient_efficiency": te_transient,
                "ci_lower": self.transient_ci[0],
                "ci_upper": self.transient_ci[1],
            }
        )

        return df

    def overall_efficiency_ci(self) -> pd.DataFrame:
        """Return overall efficiency with confidence intervals.

        Returns:
            DataFrame with columns:
                - entity: Entity identifier
                - time: Time identifier
                - overall_efficiency: Point estimate
                - ci_lower: Lower bound of CI
                - ci_upper: Upper bound of CI
        """
        te_persistent = np.exp(-self.result.eta_i)
        te_transient = np.exp(-self.result.u_it)
        te_persistent_matched = te_persistent[self.result.model.entity_id]
        te_overall = te_persistent_matched * te_transient

        df = pd.DataFrame(
            {
                "entity": self.result.model.entity_id,
                "time": self.result.model.time_id,
                "overall_efficiency": te_overall,
                "ci_lower": self.overall_ci[0],
                "ci_upper": self.overall_ci[1],
            }
        )

        return df

    def print_summary(self):
        """Print summary of bootstrap results."""
        print("\n" + "=" * 60)
        print("BOOTSTRAP CONFIDENCE INTERVALS")
        print("=" * 60)
        print(f"\nBootstrap replications: {self.n_bootstrap}")
        print(f"Confidence level: {self.confidence_level * 100:.1f}%")

        print(f"\nVariance Components:")
        for key in ["sigma_v", "sigma_u", "sigma_mu", "sigma_eta"]:
            point = getattr(self.result, key)
            ci_lower, ci_upper = self.variance_ci[key]
            print(f"  {key}: {point:.4f}  [{ci_lower:.4f}, {ci_upper:.4f}]")

        te_persistent = np.exp(-self.result.eta_i)
        te_transient = np.exp(-self.result.u_it)
        te_overall = te_persistent[self.result.model.entity_id] * te_transient

        print(f"\nEfficiency Estimates (mean ± CI width):")
        pers_ci_width = (self.persistent_ci[1] - self.persistent_ci[0]).mean()
        trans_ci_width = (self.transient_ci[1] - self.transient_ci[0]).mean()
        overall_ci_width = (self.overall_ci[1] - self.overall_ci[0]).mean()

        print(f"  Persistent:  {te_persistent.mean():.4f} ± {pers_ci_width:.4f}")
        print(f"  Transient:   {te_transient.mean():.4f} ± {trans_ci_width:.4f}")
        print(f"  Overall:     {te_overall.mean():.4f} ± {overall_ci_width:.4f}")

        print("=" * 60)
