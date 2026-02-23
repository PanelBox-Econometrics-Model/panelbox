"""
Data generation functions for Diagnostics tutorial series.

All functions generate simulated panel data for testing unit root,
cointegration, specification, and spatial diagnostics. Data is fully
reproducible using specified seeds.

Functions:
- generate_penn_world_table: Country macro panel (Tutorial 01)
- generate_prices_panel: Regional price indices (Tutorial 01)
- generate_oecd_macro: OECD macro panel (Tutorial 02)
- generate_ppp_data: PPP panel (Tutorial 02)
- generate_interest_rates: Interest rate panel (Tutorial 02)
- generate_nlswork: NLS-like wage panel (Tutorial 03)
- generate_firm_productivity: Firm production panel (Tutorial 03)
- generate_trade_panel: Bilateral trade gravity panel (Tutorial 03)
- generate_us_counties: US county economic panel (Tutorial 04)
- generate_eu_regions: EU regional panel (Tutorial 04)
"""


import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Country codes and names used across generators
# ---------------------------------------------------------------------------

OECD_30 = [
    "AUS",
    "AUT",
    "BEL",
    "CAN",
    "CHE",
    "CHL",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GBR",
    "GRC",
    "HUN",
    "IRL",
    "ISL",
    "ISR",
    "ITA",
    "JPN",
    "KOR",
    "LUX",
    "MEX",
    "NLD",
    "NOR",
    "NZL",
    "POL",
    "PRT",
    "SWE",
]

OECD_20 = [
    "Australia",
    "Austria",
    "Belgium",
    "Canada",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Ireland",
    "Italy",
    "Japan",
    "Netherlands",
    "Norway",
    "Portugal",
    "Spain",
    "Sweden",
    "Switzerland",
    "United Kingdom",
    "United States",
]

EU_COUNTRIES = [
    "DE",
    "FR",
    "IT",
    "ES",
    "NL",
    "BE",
    "AT",
    "PT",
    "GR",
    "SE",
    "DK",
    "FI",
    "IE",
    "PL",
    "CZ",
]

US_STATES = [
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "FL",
    "GA",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NY",
    "NC",
    "OH",
]


# ---------------------------------------------------------------------------
# 1. Penn World Table — country macro panel
# ---------------------------------------------------------------------------


def generate_penn_world_table(
    n_countries: int = 30,
    n_years: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate macro panel with I(1) GDP, capital, employment.

    Simulates macro panel data inspired by the Penn World Table with
    realistic I(1) behaviour for GDP and capital, and I(0) behaviour for
    labour share.

    Parameters
    ----------
    n_countries : int, default 30
        Number of countries.
    n_years : int, default 50
        Number of time periods (years).
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: countrycode, year, rgdpna, rkna, emp, labsh, pop, hc.
    """
    np.random.seed(seed)

    countries = OECD_30[:n_countries]
    years = np.arange(1970, 1970 + n_years)
    records = []

    for country in countries:
        # Country-specific parameters
        mu_gdp = np.random.normal(0.02, 0.005)
        delta_gdp = np.random.normal(0.0001, 0.00005)
        sigma_gdp = np.random.uniform(0.01, 0.04)

        # Initial values (log scale)
        log_gdp = np.random.normal(13.5, 0.8)
        log_cap = log_gdp + np.random.normal(0.8, 0.2)
        log_emp = np.random.normal(2.5, 0.8)
        log_pop = log_emp + np.random.normal(1.2, 0.3)

        labsh_mean = np.random.normal(0.55, 0.05)
        labsh = labsh_mean
        hc_base = np.random.normal(2.5, 0.3)
        hc = hc_base

        for t_idx, yr in enumerate(years):
            # GDP: random walk with drift + trend
            shock = np.random.normal(0, sigma_gdp)
            log_gdp += mu_gdp + delta_gdp * t_idx + shock

            # Capital: co-moves with GDP
            log_cap += mu_gdp * 1.05 + np.random.normal(0, sigma_gdp * 0.8)

            # Employment: slow trend
            log_emp += np.random.normal(0.005, 0.01)

            # Population: slow trend
            log_pop += np.random.normal(0.008, 0.005)

            # Labour share: I(0) mean-reverting
            labsh = labsh_mean + 0.7 * (labsh - labsh_mean) + np.random.normal(0, 0.02)
            labsh = np.clip(labsh, 0.25, 0.80)

            # Human capital: slow upward trend
            hc += np.random.normal(0.015, 0.005)
            hc = np.clip(hc, 1.0, 5.0)

            records.append(
                {
                    "countrycode": country,
                    "year": yr,
                    "rgdpna": np.exp(log_gdp) / 1e3,  # millions
                    "rkna": np.exp(log_cap) / 1e3,
                    "emp": np.exp(log_emp),
                    "labsh": round(labsh, 4),
                    "pop": np.exp(log_pop),
                    "hc": round(hc, 3),
                }
            )

    df = pd.DataFrame(records)
    # Round numeric columns
    for col in ["rgdpna", "rkna", "emp", "pop"]:
        df[col] = df[col].round(2)
    return df


# ---------------------------------------------------------------------------
# 2. Prices panel — regional price indices
# ---------------------------------------------------------------------------


def generate_prices_panel(
    n_regions: int = 40,
    n_years: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate regional price panel with I(1) price index.

    Price index follows a random walk with positive drift (inflation).
    First-differenced log price (inflation) is I(0).

    Parameters
    ----------
    n_regions : int, default 40
    n_years : int, default 30
    seed : int, default 42

    Returns
    -------
    pd.DataFrame
        Columns: region, year, price_index, log_price, inflation.
    """
    np.random.seed(seed)

    regions = [f"Region_{i + 1:02d}" for i in range(n_regions)]
    years = np.arange(1990, 1990 + n_years)
    records = []

    for region in regions:
        pi = np.random.normal(0.03, 0.01)  # mean inflation
        log_p = np.log(100.0)  # base = 100
        prev_log_p = log_p

        for t_idx, yr in enumerate(years):
            if t_idx > 0:
                shock = np.random.normal(0, 0.01)
                log_p += pi + shock
            inflation = log_p - prev_log_p if t_idx > 0 else 0.0
            records.append(
                {
                    "region": region,
                    "year": yr,
                    "price_index": round(np.exp(log_p), 2),
                    "log_price": round(log_p, 6),
                    "inflation": round(inflation, 6),
                }
            )
            prev_log_p = log_p

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. OECD macro — consumption-income cointegration
# ---------------------------------------------------------------------------


def generate_oecd_macro(
    n_countries: int = 20,
    n_years: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate OECD macro panel with cointegrated consumption-income.

    Consumption and income are cointegrated with country-specific MPC
    (marginal propensity to consume) around 0.85.

    Parameters
    ----------
    n_countries : int, default 20
    n_years : int, default 40
    seed : int, default 42

    Returns
    -------
    pd.DataFrame
        Columns: country, year, consumption, income, investment, log_C, log_Y.
    """
    np.random.seed(seed)

    countries = OECD_20[:n_countries]
    years = np.arange(1980, 1980 + n_years)
    records = []

    for country in countries:
        mu_y = np.random.normal(0.02, 0.005)
        sigma_y = np.random.uniform(0.015, 0.035)
        beta = np.random.normal(0.85, 0.05)  # MPC
        rho_u = np.random.uniform(0.3, 0.6)

        log_y = np.random.normal(10.0, 0.5)
        u_c = 0.0  # cointegration error

        # Investment-income cointegration (weaker)
        beta_inv = np.random.normal(0.22, 0.04)
        u_inv = 0.0

        for yr in years:
            # Income: random walk with drift
            log_y += mu_y + np.random.normal(0, sigma_y)

            # Consumption: cointegrated with income
            u_c = rho_u * u_c + np.random.normal(0, 0.02)
            log_c = beta * log_y + u_c

            # Investment: cointegrated (weaker)
            u_inv = 0.7 * u_inv + np.random.normal(0, 0.05)
            log_inv = beta_inv * log_y + u_inv + np.random.normal(8.5, 0.0)

            records.append(
                {
                    "country": country,
                    "year": yr,
                    "consumption": round(np.exp(log_c), 2),
                    "income": round(np.exp(log_y), 2),
                    "investment": round(np.exp(log_inv), 2),
                    "log_C": round(log_c, 6),
                    "log_Y": round(log_y, 6),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. PPP data — purchasing power parity
# ---------------------------------------------------------------------------


def generate_ppp_data(
    n_countries: int = 25,
    n_years: int = 35,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate PPP panel with cointegrated exchange rates and price ratios.

    Exchange rate and relative price level are cointegrated, with PPP
    deviations having a half-life of 3-5 years.

    Parameters
    ----------
    n_countries : int, default 25
    n_years : int, default 35
    seed : int, default 42

    Returns
    -------
    pd.DataFrame
        Columns: country, year, exchange_rate, price_domestic,
        price_foreign, log_S, log_P_ratio.
    """
    np.random.seed(seed)

    country_codes = [
        "GBR",
        "DEU",
        "FRA",
        "JPN",
        "CAN",
        "AUS",
        "CHE",
        "SWE",
        "NOR",
        "DNK",
        "NZL",
        "KOR",
        "MEX",
        "BRA",
        "IND",
        "ZAF",
        "SGP",
        "HKG",
        "TWN",
        "THA",
        "MYS",
        "IDN",
        "PHL",
        "CHL",
        "COL",
    ][:n_countries]

    years = np.arange(1985, 1985 + n_years)
    records = []

    # US price: I(1) with drift
    log_p_us = np.log(100.0)
    us_prices = []
    for yr in years:
        log_p_us += np.random.normal(0.025, 0.008)
        us_prices.append(log_p_us)

    for country in country_codes:
        rho_z = np.random.uniform(0.85, 0.95)  # persistence of PPP deviation
        sigma_z = np.random.uniform(0.03, 0.06)

        log_p_dom = np.log(100.0) + np.random.normal(0, 0.3)
        z = np.random.normal(0, 0.1)  # initial PPP deviation

        for t_idx, yr in enumerate(years):
            # Domestic price: I(1) with drift
            log_p_dom += np.random.normal(0.03, 0.012)

            # PPP deviation: AR(1) — stationary
            z = rho_z * z + np.random.normal(0, sigma_z)

            # Exchange rate: PPP + deviation
            log_p_ratio = log_p_dom - us_prices[t_idx]
            log_s = log_p_ratio + z

            records.append(
                {
                    "country": country,
                    "year": yr,
                    "exchange_rate": round(np.exp(log_s), 4),
                    "price_domestic": round(np.exp(log_p_dom), 2),
                    "price_foreign": round(np.exp(us_prices[t_idx]), 2),
                    "log_S": round(log_s, 6),
                    "log_P_ratio": round(log_p_ratio, 6),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Interest rates — international panel
# ---------------------------------------------------------------------------


def generate_interest_rates(
    n_countries: int = 15,
    n_years: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate international interest rate panel for IRP testing.

    Domestic rates follow a common global factor (US rate) plus
    country-specific spread. Spread may be I(0).

    Parameters
    ----------
    n_countries : int, default 15
    n_years : int, default 30
    seed : int, default 42

    Returns
    -------
    pd.DataFrame
        Columns: country, year, domestic_rate, us_rate, spread, forward_premium.
    """
    np.random.seed(seed)

    country_codes = [
        "GBR",
        "DEU",
        "FRA",
        "JPN",
        "CAN",
        "AUS",
        "CHE",
        "SWE",
        "NOR",
        "DNK",
        "NZL",
        "KOR",
        "MEX",
        "BRA",
        "IND",
    ][:n_countries]

    years = np.arange(1990, 1990 + n_years)

    # US rate: I(1) random walk bounded
    us_rate = 0.05
    us_rates = []
    for yr in years:
        us_rate += np.random.normal(-0.0005, 0.008)
        us_rate = np.clip(us_rate, 0.001, 0.15)
        us_rates.append(us_rate)

    records = []
    for country in country_codes:
        spread_mean = np.random.normal(0.01, 0.015)
        spread_mean = max(spread_mean, -0.02)
        rho_s = np.random.uniform(0.6, 0.85)
        spread = spread_mean

        for t_idx, yr in enumerate(years):
            spread = spread_mean + rho_s * (spread - spread_mean) + np.random.normal(0, 0.005)
            domestic = us_rates[t_idx] + spread
            domestic = max(domestic, 0.001)

            # Forward premium: related to spread + noise
            fp = spread + np.random.normal(0, 0.003)

            records.append(
                {
                    "country": country,
                    "year": yr,
                    "domestic_rate": round(domestic, 6),
                    "us_rate": round(us_rates[t_idx], 6),
                    "spread": round(spread, 6),
                    "forward_premium": round(fp, 6),
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. NLS Work — synthetic wage panel
# ---------------------------------------------------------------------------


def generate_nlswork(
    n_individuals: int = 4000,
    n_periods: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic NLS-like wage panel with correlated ability.

    Ability correlates with education so that the Hausman test should
    reject the null of RE consistency.

    Parameters
    ----------
    n_individuals : int, default 4000
    n_periods : int, default 15
    seed : int, default 42

    Returns
    -------
    pd.DataFrame
        Columns: idcode, year, ln_wage, experience, tenure, education,
        union, married, hours, industry.

    Notes
    -----
    DGP:
        ln_wage_it = 0.04*exp + (-0.0007)*exp² + 0.02*tenure
                     + 0.15*union + 0.05*married + alpha_i + eps_it
        alpha_i = 0.06 * education_i + eta_i,  eta_i ~ N(0, 0.15)
        eps_it ~ N(0, 0.10)
    The education-ability correlation (gamma=0.06) ensures the Hausman
    test will reject H0 (RE consistency).
    """
    np.random.seed(seed)

    # Biennial years inspired by NLS
    base_years = np.arange(1968, 1968 + n_periods * 2, 2)[:n_periods]

    records = []
    for i in range(1, n_individuals + 1):
        # Time-invariant characteristics
        education = int(np.clip(np.random.normal(12, 2), 8, 20))
        industry = np.random.randint(1, 13)

        # Unobserved ability correlated with education
        eta_i = np.random.normal(0, 0.15)
        alpha_i = 0.06 * education + eta_i

        # Initial conditions
        exp0 = max(0, np.random.normal(2, 2))
        tenure0 = 0.0
        married = int(np.random.random() < 0.35)

        for t_idx, yr in enumerate(base_years):
            experience = exp0 + t_idx * 2.0 + np.random.normal(0, 0.3)
            experience = max(0, experience)

            # Tenure: resets occasionally (job change)
            if np.random.random() < 0.12:
                tenure0 = 0.0
            tenure = tenure0 + np.random.exponential(0.5)
            tenure0 = tenure
            tenure = min(tenure, experience)

            # Time-varying
            union = int(np.random.random() < 0.25)
            if t_idx > 0 and np.random.random() < 0.08:
                married = 1 - married

            hours = np.clip(np.random.normal(38, 8), 10, 60)

            # Wage equation
            eps = np.random.normal(0, 0.10)
            ln_wage = (
                0.04 * experience
                - 0.0007 * experience**2
                + 0.02 * tenure
                + 0.15 * union
                + 0.05 * married
                + alpha_i
                + eps
            )

            records.append(
                {
                    "idcode": i,
                    "year": int(yr),
                    "ln_wage": round(ln_wage, 4),
                    "experience": round(experience, 2),
                    "tenure": round(tenure, 2),
                    "education": education,
                    "union": union,
                    "married": married,
                    "hours": round(hours, 1),
                    "industry": industry,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Firm productivity — Cobb-Douglas production
# ---------------------------------------------------------------------------


def generate_firm_productivity(
    n_firms: int = 200,
    n_years: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate firm production panel for specification testing.

    Cobb-Douglas production function where materials and R&D have
    significant contributions. J-test should prefer model with materials
    over basic K,L specification.

    Parameters
    ----------
    n_firms : int, default 200
    n_years : int, default 20
    seed : int, default 42

    Returns
    -------
    pd.DataFrame
        Columns: firm_id, year, log_output, log_capital, log_labor,
        log_materials, rd_intensity, sector, exporter.
    """
    np.random.seed(seed)

    sectors = ["Manufacturing", "Services", "Technology", "Energy", "Consumer"]
    years = np.arange(2000, 2000 + n_years)
    records = []

    for firm in range(1, n_firms + 1):
        sector = sectors[np.random.randint(0, len(sectors))]
        exporter = int(np.random.random() < 0.30)

        alpha_i = np.random.normal(0, 0.3)  # firm fixed effect

        log_k = np.random.normal(7.0, 1.0)
        log_l = np.random.normal(5.0, 0.8)
        log_m = np.random.normal(7.5, 0.8)

        for yr in years:
            # Slow-moving inputs
            log_k += np.random.normal(0.03, 0.05)
            log_l += np.random.normal(0.01, 0.03)
            log_m += np.random.normal(0.02, 0.04)

            rd = max(0, np.random.normal(0.02, 0.03))

            eps = np.random.normal(0, 0.15)
            log_output = (
                alpha_i
                + 0.30 * log_k
                + 0.35 * log_l
                + 0.25 * log_m
                + 0.05 * (rd * 100)  # scale rd for effect size
                + eps
            )

            records.append(
                {
                    "firm_id": firm,
                    "year": int(yr),
                    "log_output": round(log_output, 4),
                    "log_capital": round(log_k, 4),
                    "log_labor": round(log_l, 4),
                    "log_materials": round(log_m, 4),
                    "rd_intensity": round(rd, 4),
                    "sector": sector,
                    "exporter": exporter,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 8. Trade panel — gravity model
# ---------------------------------------------------------------------------


def generate_trade_panel(
    n_pairs: int = 300,
    n_years: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate bilateral trade gravity panel.

    Gravity model: exports depend on exporter/importer GDP, distance,
    tariffs, and bilateral dummies.

    Parameters
    ----------
    n_pairs : int, default 300
    n_years : int, default 15
    seed : int, default 42

    Returns
    -------
    pd.DataFrame
        Columns: pair_id, year, log_exports, log_gdp_i, log_gdp_j,
        log_distance, tariff, border, language.
    """
    np.random.seed(seed)

    years = np.arange(2005, 2005 + n_years)
    records = []

    for pair in range(1, n_pairs + 1):
        pair_id = f"PAIR_{pair:04d}"

        # Time-invariant bilateral characteristics
        log_dist = np.random.normal(8.0, 1.0)
        border = int(np.random.random() < 0.15)
        language = int(np.random.random() < 0.20)

        alpha_ij = np.random.normal(0, 0.5)  # pair fixed effect

        log_gdp_i0 = np.random.normal(12, 1.5)
        log_gdp_j0 = np.random.normal(12, 1.5)

        for t_idx, yr in enumerate(years):
            log_gdp_i = log_gdp_i0 + 0.025 * t_idx + np.random.normal(0, 0.02)
            log_gdp_j = log_gdp_j0 + 0.025 * t_idx + np.random.normal(0, 0.02)

            tariff = max(0, np.random.normal(0.05, 0.04) - 0.001 * t_idx)

            eps = np.random.normal(0, 0.3)
            log_exports = (
                alpha_ij
                + 0.8 * log_gdp_i
                + 0.7 * log_gdp_j
                - 1.2 * log_dist
                - 0.3 * (tariff * 10)
                + 0.5 * border
                + 0.3 * language
                + eps
            )

            records.append(
                {
                    "pair_id": pair_id,
                    "year": int(yr),
                    "log_exports": round(log_exports, 4),
                    "log_gdp_i": round(log_gdp_i, 4),
                    "log_gdp_j": round(log_gdp_j, 4),
                    "log_distance": round(log_dist, 4),
                    "tariff": round(tariff, 4),
                    "border": border,
                    "language": language,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 9. US counties — spatial panel
# ---------------------------------------------------------------------------


def generate_us_counties(
    n_counties: int = 200,
    n_years: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate US county economic panel with spatial dependence.

    Counties are placed on a grid with spatial autoregressive (SAR)
    structure in the dependent variable.

    Parameters
    ----------
    n_counties : int, default 200
    n_years : int, default 10
    seed : int, default 42

    Returns
    -------
    tuple of (DataFrame, W_contiguity, W_distance, coordinates_df)
        DataFrame columns: county_id, state, year, unemployment, log_income,
        log_population, manufacturing_share, education_pct.
        W_contiguity: (n_counties, n_counties) row-normalized queen contiguity.
        W_distance: (n_counties, n_counties) row-normalized inverse distance.
        coordinates_df: DataFrame with county_id, latitude, longitude.
    """
    np.random.seed(seed)

    N = n_counties
    years = np.arange(2010, 2010 + n_years)

    # Place counties on a grid with perturbation
    side = int(np.ceil(np.sqrt(N)))
    grid_lat = []
    grid_lon = []
    for idx in range(N):
        row = idx // side
        col = idx % side
        # US bounding box roughly: lat 25-48, lon -125 to -70
        lat = 25 + (row / side) * 23 + np.random.normal(0, 0.5)
        lon = -125 + (col / side) * 55 + np.random.normal(0, 0.5)
        grid_lat.append(lat)
        grid_lon.append(lon)

    coords = np.column_stack([grid_lat, grid_lon])

    # Queen contiguity from grid
    W_cont = np.zeros((N, N))
    for i in range(N):
        row_i, col_i = i // side, i % side
        for j in range(i + 1, N):
            row_j, col_j = j // side, j % side
            if abs(row_i - row_j) <= 1 and abs(col_i - col_j) <= 1:
                W_cont[i, j] = 1.0
                W_cont[j, i] = 1.0
    np.fill_diagonal(W_cont, 0)

    # Row-normalize
    row_sums = W_cont.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W_cont_norm = W_cont / row_sums[:, np.newaxis]

    # Inverse distance matrix with threshold
    from scipy.spatial.distance import cdist

    dist_matrix = cdist(coords, coords, metric="euclidean")
    threshold = np.percentile(dist_matrix[dist_matrix > 0], 25)

    W_dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and dist_matrix[i, j] <= threshold:
                W_dist[i, j] = 1.0 / dist_matrix[i, j]
    np.fill_diagonal(W_dist, 0)

    row_sums_d = W_dist.sum(axis=1)
    row_sums_d[row_sums_d == 0] = 1
    W_dist_norm = W_dist / row_sums_d[:, np.newaxis]

    # Assign states
    states = []
    for idx in range(N):
        state_idx = min(idx // (N // len(US_STATES) + 1), len(US_STATES) - 1)
        states.append(US_STATES[state_idx])

    # Generate panel with SAR structure
    rho = 0.35  # spatial lag parameter
    I_N = np.eye(N)
    A = I_N - rho * W_cont_norm

    # County fixed effects
    alpha = np.random.normal(0, 0.3, N)

    records = []
    for yr in years:
        # Exogenous variables
        log_pop = np.random.normal(10, 1.5, N) + alpha * 0.5
        mfg_share = np.clip(np.random.normal(0.12, 0.06, N), 0.01, 0.40)
        edu_pct = np.clip(np.random.normal(0.25, 0.10, N), 0.05, 0.60)

        # Xbeta for unemployment
        X_beta = (
            alpha
            - 0.02 * log_pop
            + 0.15 * mfg_share
            - 0.10 * edu_pct
            + np.random.normal(0, 0.01, N)
        )

        # Solve SAR: y = rho*Wy + Xbeta + eps => (I - rho*W)y = Xbeta + eps
        eps = np.random.normal(0, 0.01, N)
        try:
            y = np.linalg.solve(A, X_beta + eps)
        except np.linalg.LinAlgError:
            y = X_beta + eps

        unemp = np.clip(0.06 + y, 0.01, 0.20)

        # Log income (correlated with education, negatively with unemployment)
        log_income = 10.5 - 2.0 * unemp + 0.3 * edu_pct + np.random.normal(0, 0.1, N)

        for i in range(N):
            records.append(
                {
                    "county_id": i + 1,
                    "state": states[i],
                    "year": int(yr),
                    "unemployment": round(float(unemp[i]), 4),
                    "log_income": round(float(log_income[i]), 4),
                    "log_population": round(float(log_pop[i]), 4),
                    "manufacturing_share": round(float(mfg_share[i]), 4),
                    "education_pct": round(float(edu_pct[i]), 4),
                }
            )

    df = pd.DataFrame(records)

    coords_df = pd.DataFrame(
        {
            "county_id": np.arange(1, N + 1),
            "latitude": np.round(grid_lat, 4),
            "longitude": np.round(grid_lon, 4),
        }
    )

    return df, W_cont_norm, W_dist_norm, coords_df


# ---------------------------------------------------------------------------
# 10. EU regions — spatial error panel
# ---------------------------------------------------------------------------


def generate_eu_regions(
    n_regions: int = 100,
    n_years: int = 15,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Generate EU regional panel with spatial error structure.

    Regions are placed on a grid approximating EU geography, with spatial
    error correlation in the GDP equation.

    Parameters
    ----------
    n_regions : int, default 100
    n_years : int, default 15
    seed : int, default 42

    Returns
    -------
    tuple of (DataFrame, W_contiguity, coordinates_df)
        DataFrame columns: region_id, country, year, gdp_per_capita,
        log_gdp_pc, fdi, rd_expenditure, infrastructure.
        W_contiguity: (n_regions, n_regions) row-normalized.
        coordinates_df: DataFrame with region_id, latitude, longitude.
    """
    np.random.seed(seed)

    N = n_regions
    years = np.arange(2005, 2005 + n_years)

    # Regions on a grid
    side = int(np.ceil(np.sqrt(N)))
    grid_lat = []
    grid_lon = []
    for idx in range(N):
        row = idx // side
        col = idx % side
        # EU: lat ~36-60, lon ~-10 to 30
        lat = 36 + (row / side) * 24 + np.random.normal(0, 0.3)
        lon = -10 + (col / side) * 40 + np.random.normal(0, 0.3)
        grid_lat.append(lat)
        grid_lon.append(lon)

    # Queen contiguity
    W = np.zeros((N, N))
    for i in range(N):
        ri, ci = i // side, i % side
        for j in range(i + 1, N):
            rj, cj = j // side, j % side
            if abs(ri - rj) <= 1 and abs(ci - cj) <= 1:
                W[i, j] = 1.0
                W[j, i] = 1.0
    np.fill_diagonal(W, 0)

    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1
    W_norm = W / row_sums[:, np.newaxis]

    # Assign countries
    n_per_country = max(1, N // len(EU_COUNTRIES))
    countries_assigned = []
    for idx in range(N):
        cidx = min(idx // n_per_country, len(EU_COUNTRIES) - 1)
        countries_assigned.append(EU_COUNTRIES[cidx])

    region_ids = [f"EU_{i + 1:03d}" for i in range(N)]

    # Spatial error structure: eps = lambda * W * eps + u
    lam = 0.4
    I_N = np.eye(N)
    B = I_N - lam * W_norm

    alpha = np.random.normal(0, 0.15, N)

    records = []
    for yr in years:
        fdi = np.clip(np.random.normal(0.03, 0.02, N), 0, 0.15)
        rd = np.clip(np.random.normal(0.015, 0.01, N), 0, 0.08)
        infra = np.clip(np.random.normal(0.6, 0.2, N), 0.1, 1.0)

        # X*beta
        Xb = alpha + 0.5 * fdi * 100 + 0.8 * rd * 100 + 0.3 * infra

        # Spatial error
        u = np.random.normal(0, 0.1, N)
        try:
            eps = np.linalg.solve(B, u)
        except np.linalg.LinAlgError:
            eps = u

        log_gdp_pc = 10.1 + Xb + eps
        gdp_pc = np.exp(log_gdp_pc) / 1000  # in thousands of euros

        for i in range(N):
            records.append(
                {
                    "region_id": region_ids[i],
                    "country": countries_assigned[i],
                    "year": int(yr),
                    "gdp_per_capita": round(float(gdp_pc[i]) * 1000, 2),
                    "log_gdp_pc": round(float(log_gdp_pc[i]), 4),
                    "fdi": round(float(fdi[i]), 4),
                    "rd_expenditure": round(float(rd[i]), 4),
                    "infrastructure": round(float(infra[i]), 4),
                }
            )

    df = pd.DataFrame(records)

    coords_df = pd.DataFrame(
        {
            "region_id": region_ids,
            "latitude": np.round(grid_lat, 4),
            "longitude": np.round(grid_lon, 4),
        }
    )

    return df, W_norm, coords_df
