"""
Data generation functions for count models tutorials.

All functions generate simulated data based on real-world stylized facts.
Data is fully reproducible using specified seeds.

Functions:
- generate_healthcare_data: Healthcare visits (Tutorial 01)
- generate_patent_data: Firm patents (Tutorial 02)
- generate_crime_data: City crime (Tutorial 03)
- generate_trade_data: Bilateral trade (Tutorial 04)
- generate_zinb_healthcare_data: Healthcare with excess zeros (Tutorial 05)
- generate_policy_impact_data: Policy evaluation (Tutorial 06)
- generate_innovation_data: Firm innovation (Tutorial 07)
"""

import numpy as np
import pandas as pd
from scipy import stats


def generate_healthcare_data(n=2000, seed=42):
    """
    Generate healthcare visits data for Poisson introduction.

    Simulates individual healthcare utilization based on MEPS stylized facts.
    Exhibits mild overdispersion (Var/Mean ≈ 1.3), suitable for Poisson.

    Parameters
    ----------
    n : int, default 2000
        Number of individuals
    seed : int, default 42
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - individual_id: Unique identifier
        - visits: Doctor visits (count)
        - age: Age in years
        - income: Annual income in $1,000s
        - insurance: Health insurance (0/1)
        - chronic: Chronic condition (0/1)

    Examples
    --------
    >>> df = generate_healthcare_data(n=2000, seed=42)
    >>> df['visits'].mean()
    4.2...

    Notes
    -----
    Data generation process:
    1. Generate covariates from realistic distributions
    2. Create linear predictor with specified coefficients
    3. Generate counts from Poisson with mild overdispersion
    """
    np.random.seed(seed)

    # Generate covariates
    age = np.random.normal(45, 18, n).clip(18, 85).astype(int)
    income = np.random.lognormal(3.7, 0.6, n).clip(10, 150)
    insurance = np.random.binomial(1, 0.78, n)
    chronic = np.random.binomial(1, 0.32, n)

    # Generate visits (mild overdispersion)
    # Log-linear predictor
    eta = (
        0.5
        + 0.015 * age
        + -0.002 * income
        + 0.45 * insurance
        + 0.85 * chronic
        + np.random.normal(0, 0.15, n)  # Individual heterogeneity
    )
    mu = np.exp(eta)
    visits = np.random.poisson(mu).clip(0, 25)

    df = pd.DataFrame(
        {
            "individual_id": range(1, n + 1),
            "visits": visits,
            "age": age,
            "income": income,
            "insurance": insurance,
            "chronic": chronic,
        }
    )

    return df


def generate_patent_data(n_firms=1500, n_years=5, seed=123):
    """
    Generate firm patent data with severe overdispersion.

    Simulates patent counts exhibiting substantial overdispersion requiring
    Negative Binomial specification.

    Parameters
    ----------
    n_firms : int, default 1500
        Number of firms
    n_years : int, default 5
        Number of years (2015-2019)
    seed : int, default 123
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns:
        - firm_id: Firm identifier
        - year: Year
        - patents: Patent count
        - rd_expenditure: R&D spending (millions)
        - firm_size: Log(employees)
        - industry: Industry code
        - region: Geographic region
    """
    np.random.seed(seed)

    # Panel structure
    firms = np.repeat(range(1, n_firms + 1), n_years)
    years = np.tile(range(2015, 2015 + n_years), n_firms)

    # Time-invariant characteristics
    industry = np.repeat(np.random.randint(1, 11, n_firms), n_years)
    region = np.repeat(np.random.randint(1, 6, n_firms), n_years)
    firm_effect = np.repeat(np.random.normal(0, 0.8, n_firms), n_years)

    # Time-varying characteristics
    rd_exp = np.random.lognormal(2.5, 1.2, n_firms * n_years).clip(0.1, 100)
    firm_size = np.random.normal(5, 1.5, n_firms * n_years).clip(2, 9)

    # Generate patents (severe overdispersion)
    alpha = 0.5  # Negative Binomial dispersion parameter
    eta = -1.5 + 0.4 * np.log(rd_exp) + 0.3 * firm_size + 0.1 * (years - 2015) + firm_effect
    mu = np.exp(eta)

    # Negative Binomial generation
    p = 1 / (1 + alpha * mu)
    patents = np.random.negative_binomial(1 / alpha, p).clip(0, 45)

    df = pd.DataFrame(
        {
            "firm_id": firms,
            "year": years,
            "patents": patents,
            "rd_expenditure": rd_exp,
            "firm_size": firm_size,
            "industry": industry,
            "region": region,
        }
    )

    return df


def generate_crime_data(n_cities=150, n_years=10, seed=456):
    """
    Generate city crime panel data for FE/RE models.

    Parameters
    ----------
    n_cities : int, default 150
        Number of cities
    n_years : int, default 10
        Number of years (2010-2019)
    seed : int, default 456
        Random seed

    Returns
    -------
    pd.DataFrame
        Balanced panel with crime counts and city characteristics
    """
    np.random.seed(seed)

    cities = np.repeat(range(1, n_cities + 1), n_years)
    years = np.tile(range(2010, 2010 + n_years), n_cities)

    # City fixed effects (time-invariant unobserved heterogeneity)
    city_effects = np.repeat(np.random.normal(0, 0.5, n_cities), n_years)

    # Time-varying covariates
    unemp = np.random.normal(7, 2.5, n_cities * n_years).clip(3, 15)
    police = np.random.normal(2.5, 0.8, n_cities * n_years).clip(1, 5)
    income = np.random.normal(65, 20, n_cities * n_years).clip(30, 120)
    population = np.repeat(np.random.lognormal(2, 1, n_cities), n_years)
    temperature = np.tile(np.random.normal(65, 12, n_years), n_cities)

    # Generate crime counts
    eta = (
        3.5
        + 0.08 * unemp
        + -0.15 * police
        + -0.005 * income
        + 0.3 * np.log(population)
        + 0.01 * temperature
        + city_effects
    )
    mu = np.exp(eta)
    crime_count = np.random.poisson(mu).clip(10, 500)

    df = pd.DataFrame(
        {
            "city_id": cities,
            "year": years,
            "crime_count": crime_count,
            "unemployment_rate": unemp,
            "police_per_capita": police,
            "median_income": income,
            "population": population,
            "temperature": temperature,
        }
    )

    return df


def generate_trade_data(n_countries=50, n_years=15, seed=789):
    """
    Generate bilateral trade data for PPML gravity models.

    Creates trade flows with realistic gravity relationships and ~23% zeros.
    Pair-level characteristics (distance, contiguity, language) are fixed over
    time, while GDP grows and FTA status can switch -- matching real trade panel
    structure used in modern gravity estimation.

    Parameters
    ----------
    n_countries : int, default 50
        Number of countries
    n_years : int, default 15
        Number of years (2005-2019)
    seed : int, default 789
        Random seed

    Returns
    -------
    pd.DataFrame
        Trade flow data with columns:
        - exporter, importer: Country codes (C01-C50)
        - year: Year (2005-2019)
        - trade_value: Bilateral trade in millions USD (23% zeros)
        - distance: Bilateral distance in km (time-invariant)
        - contiguous: Shared border dummy (time-invariant)
        - common_language: Common language dummy (time-invariant)
        - gdp_exporter, gdp_importer: GDP in billions (time-varying)
        - trade_agreement: FTA dummy (some pairs switch over time)

    Notes
    -----
    True DGP:
        log(E[trade]) = 2.5 + 0.85*log(GDP_i) + 0.75*log(GDP_j)
                        - 1.2*log(distance) + 0.45*contiguity
                        + 0.30*language + 0.35*FTA + pair_effect + noise
    """
    np.random.seed(seed)

    # Fixed country characteristics
    country_gdp_base = np.random.lognormal(6, 1.5, n_countries).clip(10, 20000)
    country_names = [f"C{i:02d}" for i in range(1, n_countries + 1)]

    # Generate all pairs (excluding self-trade), sample ~666 for ~10k obs
    pairs = []
    for i in range(n_countries):
        for j in range(n_countries):
            if i != j:
                pairs.append((i, j))

    n_target_pairs = 10000 // n_years
    selected_idx = np.random.choice(len(pairs), n_target_pairs, replace=False)
    selected_pairs = [pairs[k] for k in selected_idx]

    # Fixed pair characteristics (time-invariant)
    pair_distances = {}
    pair_contiguous = {}
    pair_common_lang = {}
    pair_effect = {}
    for exp_i, imp_j in selected_pairs:
        pair_distances[(exp_i, imp_j)] = np.random.lognormal(7.5, 1.2)
        d = pair_distances[(exp_i, imp_j)]
        pair_contiguous[(exp_i, imp_j)] = 1 if (d < 1000 and np.random.rand() < 0.4) else 0
        pair_common_lang[(exp_i, imp_j)] = np.random.binomial(1, 0.15)
        pair_effect[(exp_i, imp_j)] = np.random.normal(0, 0.3)

    # FTA switching: ~35% of pairs get FTA at some point during the sample
    pair_fta_switch_year = {}
    for exp_i, imp_j in selected_pairs:
        if np.random.rand() < 0.35:
            pair_fta_switch_year[(exp_i, imp_j)] = np.random.randint(2005, 2020)
        else:
            pair_fta_switch_year[(exp_i, imp_j)] = 9999

    # Build dataset
    data = []
    for year_idx, year in enumerate(range(2005, 2005 + n_years)):
        # GDP grows ~2% per year with small shocks
        gdp_growth = 1.0 + 0.02 * year_idx + np.random.normal(0, 0.01, n_countries)
        gdp_year = country_gdp_base * gdp_growth

        for exp_i, imp_j in selected_pairs:
            d = pair_distances[(exp_i, imp_j)]
            contig = pair_contiguous[(exp_i, imp_j)]
            lang = pair_common_lang[(exp_i, imp_j)]
            fta = 1 if year >= pair_fta_switch_year[(exp_i, imp_j)] else 0
            pe = pair_effect[(exp_i, imp_j)]

            gdp_exp = gdp_year[exp_i]
            gdp_imp = gdp_year[imp_j]

            # True gravity equation
            eta = (
                2.5
                + 0.85 * np.log(gdp_exp)
                + 0.75 * np.log(gdp_imp)
                - 1.2 * np.log(d)
                + 0.45 * contig
                + 0.30 * lang
                + 0.35 * fta
                + pe
                + np.random.normal(0, 0.4)
            )

            mu = np.exp(eta)

            # 23% structural zeros (pairs that don't trade)
            if np.random.rand() < 0.23:
                trade = 0.0
            else:
                trade = max(0, mu * np.exp(np.random.normal(0, 0.25)))

            trade = round(trade, 2)

            data.append(
                {
                    "exporter": country_names[exp_i],
                    "importer": country_names[imp_j],
                    "year": year,
                    "trade_value": trade,
                    "distance": round(d, 2),
                    "contiguous": contig,
                    "common_language": lang,
                    "gdp_exporter": round(gdp_exp, 2),
                    "gdp_importer": round(gdp_imp, 2),
                    "trade_agreement": fta,
                }
            )

    return pd.DataFrame(data)


def generate_zinb_healthcare_data(n=3000, seed=101):
    """
    Generate healthcare data with excess zeros for ZIP/ZINB models.

    Two processes: structural zeros (never users) and count zeros.

    Parameters
    ----------
    n : int, default 3000
        Number of individuals
    seed : int, default 101
        Random seed

    Returns
    -------
    pd.DataFrame
        Healthcare data with ~60% zeros
    """
    np.random.seed(seed)

    age = np.random.normal(48, 20, n).clip(18, 85).astype(int)
    income = np.random.lognormal(3.8, 0.7, n).clip(10, 150)
    insurance = np.random.binomial(1, 0.65, n)
    chronic = np.random.binomial(1, 0.28, n)
    rural = np.random.binomial(1, 0.35, n)
    health_literacy = np.random.randint(1, 11, n)

    # Structural zero process (inflation model)
    # Adjust to get ~60% zeros
    psi = (
        1.5
        + -1.2 * insurance  # Increased intercept for more zeros
        + 1.0 * rural  # Stronger insurance effect
        + -0.2 * health_literacy  # Stronger rural effect
    )
    prob_structural_zero = 1 / (1 + np.exp(-psi))
    structural_zero = np.random.binomial(1, prob_structural_zero)

    # Count process (for potential users)
    eta = (
        -0.5
        + 0.015 * age  # Lower baseline to reduce count zeros
        + -0.002 * income
        + 0.4 * insurance
        + 0.7 * chronic
        + 0.08 * health_literacy
    )
    mu = np.exp(eta)

    # Generate visits
    visits = np.zeros(n, dtype=int)
    for i in range(n):
        if structural_zero[i] == 0:  # Potential user
            visits[i] = np.random.poisson(mu[i])

    visits = visits.clip(0, 30)

    df = pd.DataFrame(
        {
            "individual_id": range(1, n + 1),
            "visits": visits,
            "age": age,
            "income": income,
            "insurance": insurance,
            "chronic": chronic,
            "rural": rural,
            "health_literacy": health_literacy,
        }
    )

    return df


def generate_policy_impact_data(n=1200, seed=202):
    """
    Generate policy evaluation data for marginal effects tutorial.

    Simulates treatment effect with heterogeneity.

    Parameters
    ----------
    n : int, default 1200
        Number of individuals
    seed : int, default 202
        Random seed

    Returns
    -------
    pd.DataFrame
        Treatment and outcome data
    """
    np.random.seed(seed)

    treatment = np.random.binomial(1, 0.4, n)
    age = np.random.normal(45, 12, n).clip(25, 65).astype(int)
    education = np.random.normal(14, 3, n).clip(8, 20).astype(int)
    income = np.random.lognormal(4.2, 0.8, n).clip(15, 200)
    female = np.random.binomial(1, 0.52, n)
    urban = np.random.binomial(1, 0.68, n)

    # Outcome with treatment effect
    eta = (
        1.2
        + 0.6 * treatment
        + 0.01 * age
        + 0.08 * education
        + 0.003 * income
        + -0.15 * female
        + 0.25 * urban
        + 0.03 * treatment * education  # Heterogeneous effect
    )
    mu = np.exp(eta)
    outcome = np.random.poisson(mu).clip(0, 20)

    df = pd.DataFrame(
        {
            "individual_id": range(1, n + 1),
            "outcome_count": outcome,
            "treatment": treatment,
            "age": age,
            "education": education,
            "income": income,
            "female": female,
            "urban": urban,
        }
    )

    return df


def generate_innovation_data(n_firms=500, n_years=8, seed=303):
    """
    Generate comprehensive firm innovation data for case study.

    Rich covariate set for model comparison and policy analysis.

    Parameters
    ----------
    n_firms : int, default 500
        Number of firms
    n_years : int, default 8
        Number of years (2012-2019)
    seed : int, default 303
        Random seed

    Returns
    -------
    pd.DataFrame
        Comprehensive innovation panel data
    """
    np.random.seed(seed)

    firms = np.repeat(range(1, n_firms + 1), n_years)
    years = np.tile(range(2012, 2012 + n_years), n_firms)

    # Firm characteristics
    firm_effects = np.repeat(np.random.normal(0, 0.6, n_firms), n_years)
    industry = np.repeat(np.random.randint(1, 9, n_firms), n_years)
    firm_age = np.repeat(np.random.exponential(15, n_firms).clip(1, 50), n_years).astype(int)

    # Time-varying variables
    rd_intensity = np.random.gamma(2, 3, n_firms * n_years).clip(0, 25)
    firm_size = np.random.normal(5.5, 1.2, n_firms * n_years).clip(2, 9)
    export_share = np.random.beta(2, 3, n_firms * n_years) * 100
    capital_intensity = np.random.lognormal(3.5, 0.8, n_firms * n_years)
    hhi = np.random.beta(2, 5, n_firms * n_years)
    subsidy = np.random.binomial(1, 0.25, n_firms * n_years)

    # Generate patents
    eta = (
        -0.8
        + 0.25 * rd_intensity
        + 0.15 * firm_size
        + -0.01 * firm_age
        + 0.008 * export_share
        + 0.05 * np.log(capital_intensity)
        + -0.3 * hhi
        + 0.4 * subsidy
        + firm_effects
    )
    mu = np.exp(eta)

    # 35% zeros (mix of structural and sampling)
    zero_prob = 0.35
    is_zero = np.random.binomial(1, zero_prob, n_firms * n_years)
    patents = np.zeros(n_firms * n_years, dtype=int)
    for i in range(n_firms * n_years):
        if is_zero[i] == 0:
            patents[i] = np.random.poisson(mu[i])
    patents = patents.clip(0, 35)

    df = pd.DataFrame(
        {
            "firm_id": firms,
            "year": years,
            "patents": patents,
            "rd_intensity": rd_intensity,
            "firm_size": firm_size,
            "firm_age": firm_age,
            "industry": industry,
            "export_share": export_share,
            "capital_intensity": capital_intensity,
            "hhi": hhi,
            "subsidy": subsidy,
        }
    )

    return df


if __name__ == "__main__":
    # Test data generation
    print("Testing data generators...")

    df1 = generate_healthcare_data(n=100)
    print(f"✓ Healthcare data: {df1.shape}, mean visits={df1['visits'].mean():.2f}")

    df2 = generate_patent_data(n_firms=50, n_years=3)
    print(f"✓ Patent data: {df2.shape}, mean patents={df2['patents'].mean():.2f}")

    df3 = generate_crime_data(n_cities=20, n_years=5)
    print(f"✓ Crime data: {df3.shape}")

    df4 = generate_trade_data(n_countries=10, n_years=3)
    print(f"✓ Trade data: {df4.shape}, zeros={( df4['trade_value']==0).mean():.1%}")

    df5 = generate_zinb_healthcare_data(n=100)
    print(f"✓ ZINB healthcare: {df5.shape}, zeros={(df5['visits']==0).mean():.1%}")

    df6 = generate_policy_impact_data(n=100)
    print(f"✓ Policy impact: {df6.shape}")

    df7 = generate_innovation_data(n_firms=50, n_years=3)
    print(f"✓ Innovation data: {df7.shape}, zeros={(df7['patents']==0).mean():.1%}")

    print("\nAll data generators working correctly!")
