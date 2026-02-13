"""
Bootstrap inference for Granger causality tests in Panel VAR.

This module implements:
1. Restricted bootstrap (under H0)
2. Wild bootstrap (robust to heteroskedasticity)
3. Pairs bootstrap
4. Parallelized computation

References
----------
- Kilian, L. (1998). "Small-Sample Confidence Intervals for Impulse Response Functions".
- Davidson, R., & MacKinnon, J. G. (2004). "Econometric Theory and Methods".
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from tqdm import tqdm

from panelbox.var.causality import dumitrescu_hurlin_test, granger_causality_wald


@dataclass
class BootstrapGrangerResult:
    """
    Results container for bootstrapped Granger causality test.

    Attributes
    ----------
    cause : str
        Name of the causing variable
    effect : str
        Name of the effect variable
    observed_stat : float
        Observed test statistic
    p_value_asymptotic : float
        Asymptotic p-value (from chi-squared)
    p_value_bootstrap : float
        Bootstrap p-value
    bootstrap_dist : np.ndarray
        Bootstrap distribution of test statistics
    ci_lower : float
        Lower bound of 95% bootstrap CI
    ci_upper : float
        Upper bound of 95% bootstrap CI
    n_bootstrap : int
        Number of bootstrap iterations
    bootstrap_type : str
        Type of bootstrap ('residual', 'wild', 'pairs')
    """

    cause: str
    effect: str
    observed_stat: float
    p_value_asymptotic: float
    p_value_bootstrap: float
    bootstrap_dist: np.ndarray
    ci_lower: float
    ci_upper: float
    n_bootstrap: int
    bootstrap_type: str

    def summary(self) -> str:
        """Generate formatted summary."""
        lines = []
        lines.append("=" * 75)
        lines.append("Bootstrap Granger Causality Test")
        lines.append("=" * 75)
        lines.append(f"Cause: {self.cause}")
        lines.append(f"Effect: {self.effect}")
        lines.append(f"Bootstrap type: {self.bootstrap_type}")
        lines.append(f"Bootstrap iterations: {self.n_bootstrap}")
        lines.append("")
        lines.append("Test Statistics:")
        lines.append(f"  Observed Wald:          {self.observed_stat:>10.4f}")
        lines.append(f"  P-value (asymptotic):   {self.p_value_asymptotic:>10.4f}")
        lines.append(f"  P-value (bootstrap):    {self.p_value_bootstrap:>10.4f}")
        lines.append("")
        lines.append("Bootstrap Distribution:")
        lines.append(f"  Mean:     {np.mean(self.bootstrap_dist):>10.4f}")
        lines.append(f"  Std:      {np.std(self.bootstrap_dist):>10.4f}")
        lines.append(f"  Min:      {np.min(self.bootstrap_dist):>10.4f}")
        lines.append(f"  Max:      {np.max(self.bootstrap_dist):>10.4f}")
        lines.append("")
        lines.append(f"95% Bootstrap CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]")
        lines.append("")

        # Conclusion
        if self.p_value_bootstrap < 0.01:
            conclusion = "Rejects H0 at 1% (bootstrap) (***)"
        elif self.p_value_bootstrap < 0.05:
            conclusion = "Rejects H0 at 5% (bootstrap) (**)"
        elif self.p_value_bootstrap < 0.10:
            conclusion = "Rejects H0 at 10% (bootstrap) (*)"
        else:
            conclusion = "Fails to reject H0 (bootstrap)"

        lines.append(f"Conclusion: {conclusion}")
        lines.append("=" * 75)

        return "\n".join(lines)

    def plot_bootstrap_distribution(self, backend: str = "matplotlib", show: bool = True):
        """
        Plot bootstrap distribution with observed statistic.

        Parameters
        ----------
        backend : str, default='matplotlib'
            Plotting backend: 'matplotlib' or 'plotly'
        show : bool, default=True
            Whether to show the plot

        Returns
        -------
        fig
            Figure object
        """
        if backend == "matplotlib":
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            # Histogram of bootstrap statistics
            ax.hist(
                self.bootstrap_dist,
                bins=50,
                alpha=0.7,
                edgecolor="black",
                color="steelblue",
                density=True,
            )

            # Observed statistic
            ax.axvline(
                self.observed_stat,
                color="red",
                linestyle="-",
                linewidth=2,
                label=f"Observed: {self.observed_stat:.2f}",
            )

            # Critical value at 95% (one-sided)
            critical_95 = np.percentile(self.bootstrap_dist, 95)
            ax.axvline(
                critical_95,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"95% critical: {critical_95:.2f}",
            )

            # Shade rejection region
            x_max = max(self.observed_stat, np.max(self.bootstrap_dist)) * 1.1
            ax.axvspan(critical_95, x_max, alpha=0.2, color="red", label="Rejection region (5%)")

            ax.set_xlabel("Wald Statistic", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_title(
                f"Bootstrap Distribution: {self.cause} → {self.effect}\n"
                f"(n_bootstrap={self.n_bootstrap}, type={self.bootstrap_type})",
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

            fig = go.Figure()

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=self.bootstrap_dist,
                    nbinsx=50,
                    name="Bootstrap distribution",
                    marker_color="steelblue",
                    opacity=0.7,
                    histnorm="probability density",
                )
            )

            # Get y-axis range for vertical lines
            hist_vals, _ = np.histogram(self.bootstrap_dist, bins=50, density=True)
            y_max = hist_vals.max() * 1.1

            # Observed statistic
            fig.add_trace(
                go.Scatter(
                    x=[self.observed_stat, self.observed_stat],
                    y=[0, y_max],
                    mode="lines",
                    name=f"Observed: {self.observed_stat:.2f}",
                    line=dict(color="red", width=2),
                )
            )

            # Critical value
            critical_95 = np.percentile(self.bootstrap_dist, 95)
            fig.add_trace(
                go.Scatter(
                    x=[critical_95, critical_95],
                    y=[0, y_max],
                    mode="lines",
                    name=f"95% critical: {critical_95:.2f}",
                    line=dict(color="orange", dash="dash", width=2),
                )
            )

            fig.update_layout(
                title=f"Bootstrap Distribution: {self.cause} → {self.effect}<br>"
                f"(n_bootstrap={self.n_bootstrap}, type={self.bootstrap_type})",
                xaxis_title="Wald Statistic",
                yaxis_title="Density",
                showlegend=True,
            )

            if show:
                fig.show()
            return fig

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def __repr__(self) -> str:
        return (
            f"BootstrapGrangerResult("
            f"cause='{self.cause}', effect='{self.effect}', "
            f"obs={self.observed_stat:.2f}, "
            f"p_boot={self.p_value_bootstrap:.4f})"
        )


def _single_bootstrap_iteration_restricted(
    y: np.ndarray,
    X_restricted: np.ndarray,
    X_unrestricted: np.ndarray,
    R: np.ndarray,
    bootstrap_type: str,
    seed: int,
) -> float:
    """
    Single bootstrap iteration for restricted bootstrap.

    This is designed to be called in parallel.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (original data)
    X_restricted : np.ndarray
        Design matrix under H0 (restricted model)
    X_unrestricted : np.ndarray
        Design matrix for full model
    R : np.ndarray
        Restriction matrix
    bootstrap_type : str
        Type: 'residual' or 'wild'
    seed : int
        Random seed for this iteration

    Returns
    -------
    wald_stat : float
        Wald statistic for this bootstrap sample
    """
    np.random.seed(seed)

    n = len(y)

    # Estimate restricted model
    beta_restricted = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
    fitted_restricted = X_restricted @ beta_restricted
    residuals_restricted = y - fitted_restricted

    # Generate bootstrap sample under H0
    if bootstrap_type == "residual":
        # Resample residuals with replacement
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        residuals_bootstrap = residuals_restricted[bootstrap_indices]
    elif bootstrap_type == "wild":
        # Wild bootstrap with Rademacher weights
        wild_weights = np.random.choice([-1, 1], size=n)
        residuals_bootstrap = residuals_restricted * wild_weights
    else:
        raise ValueError(f"Unknown bootstrap_type: {bootstrap_type}")

    # Reconstruct y
    y_bootstrap = fitted_restricted + residuals_bootstrap

    # Estimate unrestricted model on bootstrap sample
    try:
        beta_unrestricted = np.linalg.lstsq(X_unrestricted, y_bootstrap, rcond=None)[0]
        residuals_unrestricted = y_bootstrap - X_unrestricted @ beta_unrestricted
        sigma2 = np.sum(residuals_unrestricted**2) / (n - len(beta_unrestricted))
        cov_beta = sigma2 * np.linalg.inv(X_unrestricted.T @ X_unrestricted)

        # Wald test
        R_beta = R @ beta_unrestricted
        R_cov_R = R @ cov_beta @ R.T
        wald_stat = R_beta @ np.linalg.inv(R_cov_R) @ R_beta
    except np.linalg.LinAlgError:
        # Singular matrix, return NaN
        wald_stat = np.nan

    return float(wald_stat)


def _single_bootstrap_granger_iteration(
    data: pd.DataFrame,
    causing_var: str,
    caused_var: str,
    lags: int,
    entity_col: str,
    time_col: str,
    bootstrap_type: str,
    seed: int,
) -> float:
    """
    Single bootstrap iteration for Granger causality test.

    This implements the RESTRICTED bootstrap procedure:
    1. Estimate restricted model (H0: no causality)
    2. Get residuals from restricted model
    3. Generate bootstrap sample under H0
    4. Re-estimate unrestricted model on bootstrap data
    5. Compute Wald statistic

    Parameters
    ----------
    data : pd.DataFrame
        Original panel data
    causing_var : str
        Causing variable
    caused_var : str
        Effect variable
    lags : int
        Number of lags
    entity_col : str
        Entity column name
    time_col : str
        Time column name
    bootstrap_type : str
        'residual', 'wild', or 'pairs'
    seed : int
        Random seed

    Returns
    -------
    float
        Wald statistic for this bootstrap iteration
    """
    np.random.seed(seed)

    from panelbox.var.data import PanelVARData
    from panelbox.var.model import PanelVAR

    try:
        if bootstrap_type == "pairs":
            # Pairs bootstrap: resample entire entities
            entities = data[entity_col].unique()
            boot_entities = np.random.choice(entities, size=len(entities), replace=True)

            # Build bootstrap data
            boot_data_list = []
            for new_id, orig_id in enumerate(boot_entities):
                entity_data = data[data[entity_col] == orig_id].copy()
                entity_data[entity_col] = new_id
                boot_data_list.append(entity_data)

            boot_data = pd.concat(boot_data_list, ignore_index=True)

        else:  # residual or wild bootstrap
            # Step 1: Estimate RESTRICTED model (only caused_var, no causing_var)
            restricted_data = PanelVARData(
                data,
                endog_vars=[caused_var],
                entity_col=entity_col,
                time_col=time_col,
                lags=lags,
            )
            restricted_model = PanelVAR(restricted_data)
            restricted_result = restricted_model.fit()

            # Get fitted and residuals
            fitted_restricted = restricted_result.fitted_by_eq[0]
            resid_restricted = restricted_result.resid_by_eq[0]

            # Step 2: Generate bootstrap residuals
            n_resid = len(resid_restricted)

            if bootstrap_type == "residual":
                # Simple residual resampling
                resid_boot = np.random.choice(resid_restricted, size=n_resid, replace=True)
            elif bootstrap_type == "wild":
                # Wild bootstrap with Rademacher weights
                wild_weights = np.random.choice([-1, 1], size=n_resid)
                resid_boot = resid_restricted * wild_weights
            else:
                raise ValueError(f"Unknown bootstrap_type: {bootstrap_type}")

            # Step 3: Reconstruct bootstrap dependent variable
            y_boot = fitted_restricted + resid_boot

            # Step 4: Map back to original data structure
            boot_data = data.copy()
            boot_data = boot_data.sort_values([entity_col, time_col]).reset_index(drop=True)

            # Identify which rows were used in VAR estimation
            # (first 'lags' observations per entity are dropped)
            entities = boot_data[entity_col].unique()
            y_boot_full = boot_data[caused_var].values.copy()

            idx = 0
            for entity in entities:
                entity_mask = boot_data[entity_col] == entity
                entity_indices = np.where(entity_mask)[0]

                if len(entity_indices) > lags:
                    # Start after lags observations
                    start_idx = entity_indices[lags]
                    end_idx = entity_indices[-1] + 1
                    n_obs_entity = end_idx - start_idx

                    if idx + n_obs_entity <= len(y_boot):
                        y_boot_full[start_idx:end_idx] = y_boot[idx : idx + n_obs_entity]
                        idx += n_obs_entity

            boot_data[caused_var] = y_boot_full

        # Step 5: Re-estimate UNRESTRICTED model on bootstrap data
        unrestricted_data = PanelVARData(
            boot_data,
            endog_vars=[causing_var, caused_var],
            entity_col=entity_col,
            time_col=time_col,
            lags=lags,
        )
        unrestricted_model = PanelVAR(unrestricted_data)
        unrestricted_result = unrestricted_model.fit()

        # Step 6: Compute Wald statistic
        gc_result = unrestricted_result.test_granger_causality(causing_var, caused_var)
        return gc_result.statistic

    except Exception:
        # If anything fails, return NaN
        return np.nan


def bootstrap_granger_test(
    result,
    causing_var: str,
    caused_var: str,
    n_bootstrap: int = 999,
    bootstrap_type: str = "wild",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    show_progress: bool = True,
) -> BootstrapGrangerResult:
    """
    Bootstrap Granger causality test with restricted bootstrap.

    Parameters
    ----------
    result : PanelVARResult
        Fitted Panel VAR result
    causing_var : str
        Name of the causing variable
    caused_var : str
        Name of the effect variable
    n_bootstrap : int, default=999
        Number of bootstrap iterations
    bootstrap_type : str, default='wild'
        Type of bootstrap:
        - 'residual': Resample residuals (assumes i.i.d.)
        - 'wild': Wild bootstrap with Rademacher weights (robust to heteroskedasticity)
        - 'pairs': Resample entire entities (preserves all dependence)
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all cores)
    random_state : int, optional
        Random seed for reproducibility
    show_progress : bool, default=True
        Whether to show progress bar

    Returns
    -------
    BootstrapGrangerResult
        Bootstrap test results

    Notes
    -----
    The restricted bootstrap procedure:
    1. Estimate restricted model (H0: no Granger causality)
    2. Obtain restricted residuals
    3. For each bootstrap iteration:
       a. Resample residuals (or apply wild weights)
       b. Reconstruct y from restricted fitted values + bootstrap residuals
       c. Re-estimate unrestricted model
       d. Compute Wald statistic
    4. Bootstrap p-value = proportion of bootstrap stats ≥ observed stat
    """
    # Get observed Granger test result
    observed_result = result.test_granger_causality(causing_var, caused_var)
    observed_stat = observed_result.statistic
    p_value_asymptotic = observed_result.pvalue

    # Check if raw data is available
    if "data" not in result.data_info:
        raise ValueError(
            "Bootstrap requires access to raw panel data. "
            "Ensure the data is stored in data_info during model fitting."
        )

    # Get raw data
    data = result.data_info["data"]
    entity_col = result.data_info.get("entity_col", "entity")
    time_col = result.data_info.get("time_col", "time")
    lags = result.p

    # Set random state
    if random_state is not None:
        np.random.seed(random_state)

    # Generate seeds for parallel execution
    seeds = (
        np.random.randint(0, 2**31, size=n_bootstrap)
        if random_state is None
        else np.arange(random_state, random_state + n_bootstrap)
    )

    # Helper function for single bootstrap iteration
    def _single_iteration(seed):
        return _single_bootstrap_granger_iteration(
            data=data,
            causing_var=causing_var,
            caused_var=caused_var,
            lags=lags,
            entity_col=entity_col,
            time_col=time_col,
            bootstrap_type=bootstrap_type,
            seed=seed,
        )

    # Run bootstrap in parallel
    if show_progress:
        print(f"Running {n_bootstrap} bootstrap iterations ({bootstrap_type} bootstrap)...")

    bootstrap_stats = Parallel(n_jobs=n_jobs)(
        delayed(_single_iteration)(seed) for seed in (tqdm(seeds) if show_progress else seeds)
    )

    # Filter out NaN values (failed iterations)
    bootstrap_dist = np.array([s for s in bootstrap_stats if not np.isnan(s)])

    n_failed = n_bootstrap - len(bootstrap_dist)
    if n_failed > 0:
        print(f"Warning: {n_failed}/{n_bootstrap} bootstrap iterations failed")

    # Compute bootstrap p-value
    p_value_bootstrap = np.mean(bootstrap_dist >= observed_stat)

    # Compute bootstrap confidence interval
    ci_lower = np.percentile(bootstrap_dist, 2.5)
    ci_upper = np.percentile(bootstrap_dist, 97.5)

    return BootstrapGrangerResult(
        cause=causing_var,
        effect=caused_var,
        observed_stat=observed_stat,
        p_value_asymptotic=p_value_asymptotic,
        p_value_bootstrap=p_value_bootstrap,
        bootstrap_dist=bootstrap_dist,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_bootstrap=len(bootstrap_dist),
        bootstrap_type=bootstrap_type,
    )


def bootstrap_dumitrescu_hurlin(
    data: pd.DataFrame,
    cause: str,
    effect: str,
    lags: int,
    entity_col: str = "entity",
    time_col: str = "time",
    n_bootstrap: int = 999,
    bootstrap_type: str = "wild",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    show_progress: bool = True,
) -> Dict:
    """
    Bootstrap Dumitrescu-Hurlin test.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    cause : str
        Causing variable
    effect : str
        Effect variable
    lags : int
        Number of lags
    entity_col : str, default='entity'
        Entity column name
    time_col : str, default='time'
        Time column name
    n_bootstrap : int, default=999
        Number of bootstrap iterations
    bootstrap_type : str, default='wild'
        Bootstrap type ('residual' or 'wild')
    n_jobs : int, default=-1
        Number of parallel jobs
    random_state : int, optional
        Random seed
    show_progress : bool, default=True
        Show progress bar

    Returns
    -------
    dict
        Bootstrap results with keys:
        - 'observed_Z_tilde': observed Z̃ statistic
        - 'observed_Z_bar': observed Z̄ statistic
        - 'p_value_bootstrap_Z_tilde': bootstrap p-value for Z̃
        - 'p_value_bootstrap_Z_bar': bootstrap p-value for Z̄
        - 'bootstrap_dist_Z_tilde': bootstrap distribution of Z̃
        - 'bootstrap_dist_Z_bar': bootstrap distribution of Z̄

    Notes
    -----
    Bootstrap for panel data is more complex because we need to preserve
    the panel structure. This implementation uses entity-level bootstrap.
    """
    raise NotImplementedError(
        "Dumitrescu-Hurlin bootstrap requires careful handling of panel structure. "
        "This will be completed in the next phase."
    )


# Placeholder for future implementation
def _panel_block_bootstrap(
    data: pd.DataFrame, entity_col: str, time_col: str, block_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Perform block bootstrap for panel data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    entity_col : str
        Entity identifier column
    time_col : str
        Time identifier column
    block_size : int, optional
        Block size for moving block bootstrap

    Returns
    -------
    pd.DataFrame
        Bootstrapped panel data

    Notes
    -----
    For panel data, we can bootstrap:
    1. Entities (resample entire entity time series)
    2. Time blocks (resample blocks of time periods)
    3. Both (more complex)

    The choice depends on the data structure and assumptions.
    """
    raise NotImplementedError("Panel block bootstrap to be implemented")
