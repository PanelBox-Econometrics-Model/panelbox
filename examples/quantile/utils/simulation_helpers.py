"""
Data simulation helpers for Quantile Regression tutorials.

All functions generate synthetic panel data with specific properties
designed for quantile regression analysis. Each function sets np.random.seed
for reproducibility.
"""

import numpy as np
import pandas as pd


def generate_card_education(
    n_individuals: int = 500, n_years: int = 6, seed: int = 42
) -> pd.DataFrame:
    """
    Generate education-wage panel with heterogeneous returns.

    Education has larger returns at higher quantiles (inequality amplification).
    Female coefficient becomes more negative at high quantiles (glass ceiling).

    Parameters
    ----------
    n_individuals : int
        Number of individuals
    n_years : int
        Number of time periods
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns: id, year, lwage, educ, exper, black, south,
        married, female, union, hours, age.
    """
    np.random.seed(seed)

    n_obs = n_individuals * n_years
    ids = np.repeat(np.arange(1, n_individuals + 1), n_years)
    years = np.tile(np.arange(1, n_years + 1), n_individuals)

    # Time-invariant characteristics
    educ = np.repeat(np.clip(np.random.normal(12, 2.5, n_individuals), 6, 20), n_years)
    female = np.repeat(np.random.binomial(1, 0.50, n_individuals), n_years)
    black = np.repeat(np.random.binomial(1, 0.20, n_individuals), n_years)
    south = np.repeat(np.random.binomial(1, 0.35, n_individuals), n_years)

    # Time-varying characteristics
    base_age = np.repeat(np.random.normal(30, 6, n_individuals), n_years)
    age = np.clip(base_age + np.tile(np.arange(n_years), n_individuals), 18, 65)
    exper = np.clip(age - educ - 6 + np.random.normal(0, 1, n_obs), 0, 40)
    married = np.random.binomial(1, 0.55, n_obs)
    union = np.random.binomial(1, 0.25, n_obs)
    hours = np.clip(np.random.normal(40, 8, n_obs), 10, 70)

    # Individual fixed effects (ability)
    alpha_i = np.repeat(np.random.normal(0, 0.20, n_individuals), n_years)

    # Heteroskedastic errors: sigma depends on education and gender
    sigma_i = np.exp(0.1 + 0.05 * educ - 0.03 * female)

    # Generate quantile-heterogeneous errors
    # Use asymmetric Laplace to create heterogeneous returns
    u_uniform = np.random.uniform(0, 1, n_obs)
    # Asymmetric Laplace quantile function at tau=0.5
    tau_star = 0.5
    v = np.where(
        u_uniform < tau_star,
        (1.0 / tau_star) * np.log(u_uniform / tau_star),
        -(1.0 / (1 - tau_star)) * np.log((1 - u_uniform) / (1 - tau_star)),
    )
    epsilon = sigma_i * v * 0.3

    # Wage equation with heterogeneous coefficients
    # Base coefficients (at median)
    beta_educ = 0.065
    beta_exper = 0.030
    beta_exper2 = -0.0004
    beta_female = -0.18
    beta_black = -0.10
    beta_south = -0.08
    beta_married = 0.05
    beta_union = 0.12

    # Quantile-varying part embedded in errors
    # Education effect increases with quantile (through heteroskedasticity)
    # Female gap increases with quantile (glass ceiling)
    lwage = (
        0.8
        + beta_educ * educ
        + beta_exper * exper
        + beta_exper2 * exper**2
        + beta_female * female
        + beta_black * black
        + beta_south * south
        + beta_married * married
        + beta_union * union
        + alpha_i
        + epsilon
        # Add education-quantile interaction through the error
        + 0.02 * educ * (v * 0.1)
        # Add female-quantile interaction (glass ceiling)
        - 0.06 * female * np.abs(v * 0.1)
    )

    df = pd.DataFrame(
        {
            "id": ids,
            "year": years,
            "lwage": np.round(lwage, 4),
            "educ": np.round(educ, 1),
            "exper": np.round(exper, 1),
            "black": black,
            "south": south,
            "married": married,
            "female": female,
            "union": union,
            "hours": np.round(hours, 1),
            "age": np.round(age, 1),
        }
    )

    return df


def generate_firm_production(n_firms: int = 500, n_years: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Generate firm production panel with heterogeneous productivity FE.

    Some firms have quantile-varying FE (location shift violated).
    Profit has quantile-dependent persistence for dynamic QR.

    Parameters
    ----------
    n_firms : int
        Number of firms
    n_years : int
        Number of time periods
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns: firm_id, year, log_output, log_capital,
        log_labor, log_materials, profit, size, sector, exporter.
    """
    np.random.seed(seed)

    n_obs = n_firms * n_years
    firm_ids = np.repeat(np.arange(1, n_firms + 1), n_years)
    years = np.tile(np.arange(2010, 2010 + n_years), n_firms)

    # Firm-level characteristics
    sectors = np.repeat(
        np.random.choice(
            ["Manufacturing", "Services", "Technology", "Energy", "Finance"],
            n_firms,
        ),
        n_years,
    )
    exporter = np.repeat(np.random.binomial(1, 0.30, n_firms), n_years)

    # Factor inputs with persistence
    log_capital_base = np.repeat(np.random.normal(7.0, 1.5, n_firms), n_years)
    log_capital = log_capital_base + np.random.normal(0, 0.2, n_obs)

    log_labor_base = np.repeat(np.random.normal(5.0, 1.2, n_firms), n_years)
    log_labor = log_labor_base + np.random.normal(0, 0.15, n_obs)

    log_materials_base = np.repeat(np.random.normal(7.5, 1.3, n_firms), n_years)
    log_materials = log_materials_base + np.random.normal(0, 0.2, n_obs)

    # Firm-specific productivity (fixed effects)
    alpha_i = np.repeat(np.random.normal(0, 0.5, n_firms), n_years)

    # Some firms have quantile-varying FE (for location shift test)
    np.repeat(np.random.normal(0, 0.3, n_firms), n_years)

    # Production function: Cobb-Douglas
    epsilon = np.random.normal(0, 0.3, n_obs)
    log_output = alpha_i + 0.30 * log_capital + 0.35 * log_labor + 0.25 * log_materials + epsilon

    # Size
    size = 0.5 * log_capital + 0.3 * log_labor + np.random.normal(0, 0.5, n_obs)

    # Profit with quantile-dependent persistence
    # The persistence parameter varies with the quantile of the innovation:
    #   rho(u) = rho_base + rho_het * (u - 0.5)
    # where u ~ U(0,1) is the innovation quantile.
    # This means: at low quantiles (bad shocks), persistence is LOW
    #             at high quantiles (good shocks), persistence is HIGH
    # "Winners keep winning, losers are volatile"

    # Capital and labor as separate controls
    capital = (log_capital - log_capital.mean()) / log_capital.std()
    labor = (log_labor - log_labor.mean()) / log_labor.std()
    size_std = (size - size.mean()) / size.std()

    rho_base = 0.45  # median persistence
    rho_het = 0.50  # heterogeneity: rho(0.1) ~ 0.25, rho(0.9) ~ 0.65

    from scipy.stats import norm as _norm

    profit = np.zeros(n_obs)
    for i in range(n_firms):
        start = i * n_years
        profit[start] = np.random.normal(0, 0.10)

        for t in range(1, n_years):
            # Draw innovation quantile
            u = np.random.uniform(0, 1)
            # Quantile-dependent persistence
            rho_t = np.clip(rho_base + rho_het * (u - 0.5), 0.05, 0.85)
            # Innovation (location-scale)
            eps = _norm.ppf(u) * 0.08

            profit[start + t] = (
                rho_t * profit[start + t - 1]
                + 0.02 * size_std[start + t]
                + 0.015 * capital[start + t]
                + 0.01 * labor[start + t]
                + eps
            )

    df = pd.DataFrame(
        {
            "firm_id": firm_ids,
            "year": years,
            "log_output": np.round(log_output, 4),
            "log_capital": np.round(log_capital, 4),
            "log_labor": np.round(log_labor, 4),
            "log_materials": np.round(log_materials, 4),
            "profit": np.round(profit, 4),
            "size": np.round(size, 4),
            "capital": np.round(capital, 4),
            "labor": np.round(labor, 4),
            "sector": sectors,
            "exporter": exporter,
        }
    )

    return df


def generate_financial_returns(
    n_firms: int = 200, n_months: int = 60, seed: int = 42
) -> pd.DataFrame:
    """
    Generate financial returns panel with fat tails and size-dependent volatility.

    Location shift assumption violated (asymmetric firm-specific risk).

    Parameters
    ----------
    n_firms : int
        Number of firms
    n_months : int
        Number of months
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns: firm_id, month, returns, size,
        book_to_market, momentum, volatility, sector.
    """
    np.random.seed(seed)

    n_obs = n_firms * n_months
    firm_ids = np.repeat(np.arange(1, n_firms + 1), n_months)
    months = np.tile(np.arange(1, n_months + 1), n_firms)

    # Firm-level characteristics
    sectors = np.repeat(
        np.random.choice(
            ["Technology", "Finance", "Healthcare", "Energy", "Consumer"],
            n_firms,
        ),
        n_months,
    )

    # Size (log market cap) — slowly varying
    size_base = np.repeat(np.random.normal(8.0, 2.0, n_firms), n_months)
    size = size_base + np.random.normal(0, 0.1, n_obs)

    # Book-to-market ratio
    btm_base = np.repeat(np.clip(np.random.normal(0.7, 0.4, n_firms), 0.1, 2.0), n_months)
    book_to_market = btm_base + np.random.normal(0, 0.05, n_obs)

    # Momentum (past 12-month return)
    momentum = np.random.normal(0.10, 0.30, n_obs)

    # Firm-specific alpha (asymmetric — varies with quantile)
    alpha_i = np.repeat(np.random.normal(0, 0.5, n_firms), n_months)

    # Size-dependent volatility
    sigma_i = np.exp(0.5 - 0.15 * size_base / size_base.std())

    # Fat-tailed errors (t distribution)
    epsilon = np.random.standard_t(5, n_obs)

    # Returns
    returns = (
        alpha_i * 0.3
        + 0.5 * (size - size.mean()) / size.std()
        + 0.3 * book_to_market
        + 0.2 * momentum
        + sigma_i * epsilon
    )

    # Historical volatility (realized)
    volatility = np.abs(returns) * 0.5 + np.random.exponential(0.15, n_obs)

    df = pd.DataFrame(
        {
            "firm_id": firm_ids,
            "month": months,
            "returns": np.round(returns, 4),
            "size": np.round(size, 4),
            "book_to_market": np.round(book_to_market, 4),
            "momentum": np.round(momentum, 4),
            "volatility": np.round(volatility, 4),
            "sector": sectors,
        }
    )

    return df


def generate_labor_program(n_individuals: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate labor training program data with progressive QTE.

    Treatment effects decrease with quantile (helps low earners more).

    Parameters
    ----------
    n_individuals : int
        Number of individuals
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns: id, period, treatment, earnings,
        education, age, experience, female.
    """
    np.random.seed(seed)

    n_obs = n_individuals * 2  # 2 periods
    ids = np.repeat(np.arange(1, n_individuals + 1), 2)
    periods = np.tile([0, 1], n_individuals)

    # Time-invariant characteristics
    education = np.repeat(np.clip(np.random.normal(12, 3, n_individuals), 6, 20), 2)
    age = np.repeat(np.clip(np.random.normal(30, 8, n_individuals), 18, 55), 2)
    experience = np.repeat(np.clip(np.random.normal(8, 5, n_individuals), 0, 30), 2)
    female = np.repeat(np.random.binomial(1, 0.45, n_individuals), 2)

    # Treatment assignment (randomized)
    treatment = np.repeat(np.random.binomial(1, 0.50, n_individuals), 2)

    # Individual-level baseline earnings
    mu_i = np.repeat(np.clip(np.random.normal(2500, 500, n_individuals), 800, 5000), 2)

    # Individual-specific variance
    sigma_i = np.repeat(np.random.uniform(200, 600, n_individuals), 2)

    # Base earnings
    earnings_base = mu_i + 100 * education + 30 * age + np.random.normal(0, 1, n_obs) * sigma_i

    # Progressive treatment effect:
    # The effect varies with the individual's position in the earnings distribution
    # Low earners benefit more
    # We implement this by making the treatment effect depend on the individual's
    # position: delta(u) where u is the individual's rank
    u_rank = np.repeat(np.random.uniform(0, 1, n_individuals), 2)
    # delta(tau) = 800 - 600*tau -> delta(0.1)=740, delta(0.5)=500, delta(0.9)=260
    treatment_effect = 800 - 600 * u_rank

    # Apply treatment in post-period
    post = (periods == 1).astype(float)
    earnings = earnings_base + treatment * post * treatment_effect

    # Ensure positive
    earnings = np.clip(earnings, 100, None)

    df = pd.DataFrame(
        {
            "id": ids,
            "period": periods,
            "treatment": treatment,
            "earnings": np.round(earnings, 2),
            "education": np.round(education, 1),
            "age": np.round(age, 1),
            "experience": np.round(experience, 1),
            "female": female,
        }
    )

    return df


def generate_crossing_example(
    n_units: int = 300, n_periods: int = 8, seed: int = 42
) -> pd.DataFrame:
    """
    Generate data designed to produce quantile crossing.

    X-dependent heteroskedasticity with misspecified linear model.

    Parameters
    ----------
    n_units : int
        Number of units
    n_periods : int
        Number of periods
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns: id, t, y, x1, x2.
    """
    np.random.seed(seed)

    n_obs = n_units * n_periods
    ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    t = np.tile(np.arange(1, n_periods + 1), n_units)

    # Correlated covariates
    mean = [0, 0]
    cov = [[1.0, 0.6], [0.6, 1.0]]
    x_raw = np.random.multivariate_normal(mean, cov, n_obs)
    x1 = x_raw[:, 0]
    x2 = x_raw[:, 1]

    # DGP: heteroskedasticity depends on x1
    epsilon = np.random.normal(0, 1, n_obs)
    y = 2 + 1.5 * x1 - 0.8 * x2 + (0.5 + 0.3 * x1) * epsilon

    df = pd.DataFrame(
        {
            "id": ids,
            "t": t,
            "y": np.round(y, 4),
            "x1": np.round(x1, 4),
            "x2": np.round(x2, 4),
        }
    )

    return df


def generate_location_shift(
    n_units: int = 400, n_periods: int = 10, seed: int = 42
) -> pd.DataFrame:
    """
    Generate data with two groups: one satisfying location shift, one not.

    Parameters
    ----------
    n_units : int
        Number of units
    n_periods : int
        Number of periods
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns: id, t, y, x1, x2, group.
    """
    np.random.seed(seed)

    n_per_group = n_units // 2
    n_obs = n_units * n_periods

    ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    t = np.tile(np.arange(1, n_periods + 1), n_units)

    # Group assignment
    group_labels = np.repeat(
        np.array(["shift"] * n_per_group + ["no_shift"] * n_per_group),
        n_periods,
    )

    # Covariates
    x1 = np.random.normal(0, 1, n_obs)
    x2 = np.random.normal(0, 1, n_obs)

    # Fixed effects
    alpha_i = np.repeat(np.random.normal(0, 1.5, n_units), n_periods)
    # Quantile-varying part for no_shift group
    delta_i = np.repeat(np.random.normal(0, 0.8, n_units), n_periods)

    # Errors — uniform for quantile structure
    u = np.random.uniform(0, 1, n_obs)
    # Convert to standard normal quantiles
    from scipy import stats

    epsilon = stats.norm.ppf(u)

    # DGP
    y = np.zeros(n_obs)

    # Group 'shift': alpha_i is constant across quantiles (Canay valid)
    shift_mask = group_labels == "shift"
    y[shift_mask] = (
        10 + alpha_i[shift_mask] + 1.0 * x1[shift_mask] + 0.5 * x2[shift_mask] + epsilon[shift_mask]
    )

    # Group 'no_shift': alpha_i varies with quantile (Canay invalid)
    no_shift_mask = group_labels == "no_shift"
    y[no_shift_mask] = (
        10
        + alpha_i[no_shift_mask]
        + delta_i[no_shift_mask] * (u[no_shift_mask] - 0.5)
        + 1.0 * x1[no_shift_mask]
        + 0.5 * x2[no_shift_mask]
        + epsilon[no_shift_mask]
    )

    df = pd.DataFrame(
        {
            "id": ids,
            "t": t,
            "y": np.round(y, 4),
            "x1": np.round(x1, 4),
            "x2": np.round(x2, 4),
            "group": group_labels,
        }
    )

    return df


def generate_heteroskedastic(
    n_units: int = 500, n_periods: int = 8, seed: int = 42
) -> pd.DataFrame:
    """
    Generate panel with explicit location-scale structure.

    He-Zhu test should reject. Location-scale model recovers DGP.

    Parameters
    ----------
    n_units : int
        Number of units
    n_periods : int
        Number of periods
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns: id, t, y, x1, x2, x3.
    """
    np.random.seed(seed)

    n_obs = n_units * n_periods
    ids = np.repeat(np.arange(1, n_units + 1), n_periods)
    t = np.tile(np.arange(1, n_periods + 1), n_units)

    # Covariates
    x1 = np.random.normal(0, 1, n_obs)
    x2 = np.random.normal(0, 1, n_obs)
    x3 = np.random.normal(0, 1, n_obs)

    # Location-scale DGP
    mu = 1.0 + 0.5 * x1 + 0.3 * x2
    sigma = np.exp(0.2 + 0.3 * x1 - 0.2 * x2)

    # Errors
    epsilon = np.random.normal(0, 1, n_obs)

    y = mu + sigma * epsilon

    df = pd.DataFrame(
        {
            "id": ids,
            "t": t,
            "y": np.round(y, 4),
            "x1": np.round(x1, 4),
            "x2": np.round(x2, 4),
            "x3": np.round(x3, 4),
        }
    )

    return df


def generate_treatment_effects(n_individuals: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generate DiD panel with regressive treatment effects (contrast with labor_program).

    Treatment effects increase with quantile (helps high earners more).

    Parameters
    ----------
    n_individuals : int
        Number of individuals
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns: id, period, treatment, earnings,
        education, age, female.
    """
    np.random.seed(seed)

    n_obs = n_individuals * 2
    ids = np.repeat(np.arange(1, n_individuals + 1), 2)
    periods = np.tile([0, 1], n_individuals)

    # Time-invariant characteristics
    education = np.repeat(np.clip(np.random.normal(12, 3, n_individuals), 6, 20), 2)
    age = np.repeat(np.clip(np.random.normal(35, 10, n_individuals), 18, 60), 2)
    female = np.repeat(np.random.binomial(1, 0.45, n_individuals), 2)

    # Treatment assignment
    treatment = np.repeat(np.random.binomial(1, 0.50, n_individuals), 2)

    # Individual baseline
    mu_i = np.repeat(np.clip(np.random.normal(3000, 500, n_individuals), 1000, 6000), 2)
    sigma_i = np.repeat(np.random.uniform(300, 800, n_individuals), 2)

    # Base earnings
    earnings_base = mu_i + 120 * education + 25 * age + np.random.normal(0, 1, n_obs) * sigma_i

    # Regressive treatment effect: helps high earners more
    u_rank = np.repeat(np.random.uniform(0, 1, n_individuals), 2)
    # delta(tau) = 100 + 800*tau -> delta(0.1)=180, delta(0.5)=500, delta(0.9)=820
    treatment_effect = 100 + 800 * u_rank

    post = (periods == 1).astype(float)
    earnings = earnings_base + treatment * post * treatment_effect
    earnings = np.clip(earnings, 100, None)

    df = pd.DataFrame(
        {
            "id": ids,
            "period": periods,
            "treatment": treatment,
            "earnings": np.round(earnings, 2),
            "education": np.round(education, 1),
            "age": np.round(age, 1),
            "female": female,
        }
    )

    return df
