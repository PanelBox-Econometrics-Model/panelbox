"""
Data generators for Stochastic Frontier Analysis tutorial series.

All functions generate synthetic panel/cross-section data with realistic
production frontier dynamics. Each function sets np.random.seed for reproducibility.

Functions:
- generate_hospital_data: Cross-section hospital production (200 hospitals)
- generate_farm_data: Cross-section agricultural production (300 farms)
- generate_bank_panel: Panel banking with BC92 time-varying (50 banks × 15 years)
- generate_airline_panel: Panel airline with Kumbhakar 1990 (25 airlines × 20 years)
- generate_manufacturing_panel: Panel manufacturing four-component (100 firms × 10 years)
- generate_electricity_panel: Panel electricity four-component (60 generators × 12 years)
- generate_hospital_panel: Panel hospital with BC95 determinants (80 hospitals × 10 years)
- generate_school_panel: Panel school with Wang 2002 (100 schools × 8 years)
- generate_dairy_farm_data: Cross-section dairy Translog (500 farms)
- generate_telecom_panel: Panel telecom with CSS (40 firms × 15 years)
- generate_brazilian_firms: Panel Brazilian manufacturing case study (500 firms × 10 years)
"""

import numpy as np
import pandas as pd


def generate_hospital_data(n_hospitals: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate cross-section hospital production data with half-normal inefficiency.

    Cobb-Douglas frontier:
        log_output = β₀ + β₁·log_labor + β₂·log_capital + β₃·log_supplies + β₄·teaching + v - u

    Parameters
    ----------
    n_hospitals : int, default 200
        Number of hospitals.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hospital_id, log_output, log_labor, log_capital,
        log_supplies, teaching, urban, ownership, beds.
    """
    rng = np.random.default_rng(seed)

    # Generate hospital characteristics
    teaching = rng.binomial(1, 0.20, size=n_hospitals)
    urban = rng.binomial(1, 0.60, size=n_hospitals)

    ownership_probs = [0.40, 0.30, 0.30]
    ownership_codes = rng.choice(
        ["public", "private", "nonprofit"], size=n_hospitals, p=ownership_probs
    )

    # Beds (original scale) — correlated with urban and teaching
    beds_base = rng.lognormal(mean=5.2, sigma=0.6, size=n_hospitals)
    beds = np.clip(beds_base * (1 + 0.3 * urban + 0.5 * teaching), 50, 1000).astype(int)

    # Generate inputs (log scale)
    log_labor = rng.normal(5.5, 0.9, size=n_hospitals) + 0.2 * urban + 0.3 * teaching
    log_capital = rng.normal(4.5, 1.0, size=n_hospitals) + 0.15 * urban
    log_supplies = rng.normal(6.0, 0.8, size=n_hospitals) + 0.1 * teaching

    # Production frontier parameters
    beta = [2.0, 0.45, 0.30, 0.20, 0.15]

    # Frontier output
    frontier = (
        beta[0]
        + beta[1] * log_labor
        + beta[2] * log_capital
        + beta[3] * log_supplies
        + beta[4] * teaching
    )

    # Noise
    v = rng.normal(0, np.sqrt(0.04), size=n_hospitals)

    # Half-normal inefficiency
    u = np.abs(rng.normal(0, np.sqrt(0.09), size=n_hospitals))

    # Teaching hospitals get ~5% efficiency bonus
    u = u * (1 - 0.05 * teaching)

    # Private hospitals slightly more efficient
    u = u * np.where(ownership_codes == "private", 0.85, 1.0)

    # Observed output
    log_output = frontier + v - u

    return pd.DataFrame(
        {
            "hospital_id": np.arange(1, n_hospitals + 1),
            "log_output": np.round(log_output, 4),
            "log_labor": np.round(log_labor, 4),
            "log_capital": np.round(log_capital, 4),
            "log_supplies": np.round(log_supplies, 4),
            "teaching": teaching,
            "urban": urban,
            "ownership": ownership_codes,
            "beds": beds,
        }
    )


def generate_farm_data(n_farms: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generate cross-section agricultural production data with exponential inefficiency.

    Cobb-Douglas frontier with irrigation and region effects.

    Parameters
    ----------
    n_farms : int, default 300
        Number of farms.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: farm_id, log_output, log_land, log_labor,
        log_fertilizer, irrigation, region, farm_size.
    """
    rng = np.random.default_rng(seed)

    # Farm characteristics
    irrigation = rng.binomial(1, 0.45, size=n_farms)
    regions = rng.choice(
        ["North", "South", "East", "West"], size=n_farms, p=[0.25, 0.30, 0.20, 0.25]
    )
    farm_sizes = rng.choice(["small", "medium", "large"], size=n_farms, p=[0.40, 0.35, 0.25])

    # Region technology levels
    region_effect = np.where(
        regions == "South",
        0.3,
        np.where(regions == "East", 0.15, np.where(regions == "West", 0.05, 0.0)),
    )

    # Generate inputs
    size_mult = np.where(farm_sizes == "large", 1.5, np.where(farm_sizes == "medium", 1.0, 0.6))
    log_land = rng.normal(3.5, 1.2, size=n_farms) + np.log(size_mult)
    log_labor = rng.normal(4.0, 0.8, size=n_farms) + 0.3 * np.log(size_mult)
    log_fertilizer = rng.normal(3.0, 1.0, size=n_farms) + 0.2 * irrigation

    # Production frontier
    beta = [1.5, 0.40, 0.35, 0.20]
    frontier = (
        beta[0]
        + beta[1] * log_land
        + beta[2] * log_labor
        + beta[3] * log_fertilizer
        + 0.10 * irrigation
        + region_effect
    )

    # Noise
    v = rng.normal(0, np.sqrt(0.03), size=n_farms)

    # Exponential inefficiency
    u = rng.exponential(scale=0.25, size=n_farms)

    # Irrigation improves efficiency
    u = u * (1 - 0.15 * irrigation)

    # Observed output
    log_output = frontier + v - u

    return pd.DataFrame(
        {
            "farm_id": np.arange(1, n_farms + 1),
            "log_output": np.round(log_output, 4),
            "log_land": np.round(log_land, 4),
            "log_labor": np.round(log_labor, 4),
            "log_fertilizer": np.round(log_fertilizer, 4),
            "irrigation": irrigation,
            "region": regions,
            "farm_size": farm_sizes,
        }
    )


def generate_bank_panel(n_banks: int = 50, n_years: int = 15, seed: int = 42) -> pd.DataFrame:
    """
    Generate balanced panel of banking data with BC92 time-varying inefficiency.

    u_it = u_i · exp(-η·(t - T)), η ~ 0.03 (slight efficiency improvement over time).

    Parameters
    ----------
    n_banks : int, default 50
        Number of banks.
    n_years : int, default 15
        Number of years.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: bank_id, year, log_output, log_labor,
        log_capital, log_deposits, ownership, size_category, region.
    """
    rng = np.random.default_rng(seed)
    N = n_banks
    T = n_years
    start_year = 2005
    years = np.arange(start_year, start_year + T)

    # Bank characteristics (time-invariant)
    ownership = rng.choice(["public", "private_domestic", "foreign"], size=N, p=[0.30, 0.40, 0.30])
    size_category = rng.choice(["small", "medium", "large"], size=N, p=[0.35, 0.40, 0.25])
    region = rng.choice(
        ["Region_1", "Region_2", "Region_3", "Region_4", "Region_5"],
        size=N,
        p=[0.20, 0.25, 0.20, 0.20, 0.15],
    )

    # Bank fixed effects
    alpha_i = rng.normal(0, 0.3, size=N)

    # Size effects on inputs
    size_scale = np.where(
        size_category == "large", 1.5, np.where(size_category == "medium", 1.0, 0.6)
    )

    # Base inefficiency per bank
    u_i = np.abs(rng.normal(0, np.sqrt(0.06), size=N))
    # Foreign banks more efficient
    u_i *= np.where(ownership == "foreign", 0.7, 1.0)
    # Large banks have scale advantages
    u_i *= np.where(size_category == "large", 0.8, 1.0)

    eta = 0.03  # BC92 decay parameter

    rows = []
    for t_idx, year in enumerate(years):
        for i in range(N):
            # Time-varying inefficiency (BC92)
            u_it = u_i[i] * np.exp(-eta * (t_idx - (T - 1)))

            # Inputs with mild time trends
            log_labor = rng.normal(7.0, 1.0) * size_scale[i] + 0.02 * t_idx + alpha_i[i] * 0.3
            log_capital = rng.normal(8.5, 1.2) * size_scale[i] + 0.03 * t_idx + alpha_i[i] * 0.4
            log_deposits = rng.normal(9.5, 1.3) * size_scale[i] + 0.025 * t_idx + alpha_i[i] * 0.5

            # Frontier
            beta = [1.5, 0.35, 0.30, 0.30]
            frontier = (
                beta[0]
                + beta[1] * log_labor
                + beta[2] * log_capital
                + beta[3] * log_deposits
                + alpha_i[i]
            )

            # Noise
            v_it = rng.normal(0, np.sqrt(0.03))

            log_output = frontier + v_it - u_it

            rows.append(
                {
                    "bank_id": i + 1,
                    "year": year,
                    "log_output": round(log_output, 4),
                    "log_labor": round(log_labor, 4),
                    "log_capital": round(log_capital, 4),
                    "log_deposits": round(log_deposits, 4),
                    "ownership": ownership[i],
                    "size_category": size_category[i],
                    "region": region[i],
                }
            )

    return pd.DataFrame(rows)


def generate_airline_panel(n_airlines: int = 25, n_years: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Generate airline panel data with Kumbhakar (1990) time pattern.

    Low-cost carriers more efficient. Efficiency improvement after 2010.

    Parameters
    ----------
    n_airlines : int, default 25
        Number of airlines.
    n_years : int, default 20
        Number of years.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: airline_id, year, log_output, log_labor,
        log_fuel, log_fleet, carrier_type.
    """
    rng = np.random.default_rng(seed)
    N = n_airlines
    T = n_years
    start_year = 2000
    years = np.arange(start_year, start_year + T)

    carrier_type = rng.choice(["legacy", "low_cost", "regional"], size=N, p=[0.40, 0.35, 0.25])

    # Base inefficiency
    u_i = np.abs(rng.normal(0, np.sqrt(0.08), size=N))
    u_i *= np.where(carrier_type == "low_cost", 0.6, 1.0)
    u_i *= np.where(carrier_type == "regional", 1.2, 1.0)

    # Kumbhakar (1990) time pattern parameters
    b_param = 0.15
    c_param = 0.02

    # Size effect by carrier type
    size_scale = np.where(
        carrier_type == "legacy", 1.3, np.where(carrier_type == "low_cost", 1.0, 0.7)
    )

    alpha_i = rng.normal(0, 0.2, size=N)

    rows = []
    for t_idx, year in enumerate(years):
        # Kumbhakar time function: G(t) = [1 + exp(b·t + c·t²)]⁻¹
        t_centered = (t_idx - T / 2) / T
        G_t = 1.0 / (1.0 + np.exp(b_param * t_centered + c_param * t_centered**2))

        # Post-2010 efficiency improvement
        post_consolidation = 1.0 if year >= 2010 else 0.0

        for i in range(N):
            u_it = u_i[i] * G_t * (1 - 0.1 * post_consolidation)

            log_labor = rng.normal(8.0, 1.0) * size_scale[i] + 0.01 * t_idx
            log_fuel = rng.normal(7.5, 1.2) * size_scale[i] + 0.015 * t_idx
            log_fleet = rng.normal(3.5, 0.8) * size_scale[i] + 0.02 * t_idx

            beta = [2.0, 0.40, 0.30, 0.25]
            frontier = (
                beta[0]
                + beta[1] * log_labor
                + beta[2] * log_fuel
                + beta[3] * log_fleet
                + alpha_i[i]
            )

            v_it = rng.normal(0, np.sqrt(0.03))
            log_output = frontier + v_it - u_it

            rows.append(
                {
                    "airline_id": i + 1,
                    "year": year,
                    "log_output": round(log_output, 4),
                    "log_labor": round(log_labor, 4),
                    "log_fuel": round(log_fuel, 4),
                    "log_fleet": round(log_fleet, 4),
                    "carrier_type": carrier_type[i],
                }
            )

    return pd.DataFrame(rows)


def generate_manufacturing_panel(
    n_firms: int = 100, n_years: int = 10, seed: int = 42
) -> pd.DataFrame:
    """
    Generate manufacturing panel with four-component error structure.

    y_it = α + β'x + μ_i - η_i + v_it - u_it

    Parameters
    ----------
    n_firms : int, default 100
        Number of firms.
    n_years : int, default 10
        Number of years.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: firm_id, year, log_output, log_labor,
        log_capital, log_materials, sector, exporter.
    """
    rng = np.random.default_rng(seed)
    N = n_firms
    T = n_years
    start_year = 2010
    years = np.arange(start_year, start_year + T)

    sectors = rng.choice(
        ["food", "textiles", "chemicals", "metals", "electronics"],
        size=N,
        p=[0.25, 0.20, 0.20, 0.20, 0.15],
    )
    exporter = rng.binomial(1, 0.35, size=N)

    # Four-component structure
    # μ_i: heterogeneity (can be positive or negative)
    mu_i = rng.normal(0, np.sqrt(0.15), size=N)
    # Sector effects on technology
    sector_effect = np.where(
        sectors == "electronics",
        0.3,
        np.where(
            sectors == "chemicals",
            0.2,
            np.where(sectors == "metals", 0.1, np.where(sectors == "food", 0.05, 0.0)),
        ),
    )
    mu_i += sector_effect

    # η_i: persistent inefficiency
    eta_i = np.abs(rng.normal(0, np.sqrt(0.08), size=N))
    # Exporters have lower persistent inefficiency
    eta_i *= np.where(exporter == 1, 0.7, 1.0)

    alpha_i = rng.normal(0, 0.15, size=N)

    rows = []
    for t_idx, year in enumerate(years):
        for i in range(N):
            # Transient components
            v_it = rng.normal(0, np.sqrt(0.03))
            u_it = np.abs(rng.normal(0, np.sqrt(0.05)))

            # Inputs with mild trends and firm heterogeneity
            log_labor = rng.normal(5.5, 1.2) + alpha_i[i] * 0.5 + 0.01 * t_idx
            log_capital = rng.normal(7.0, 1.5) + alpha_i[i] * 0.7 + 0.02 * t_idx
            log_materials = rng.normal(7.5, 1.3) + alpha_i[i] * 0.6 + 0.015 * t_idx

            # Production frontier
            beta = [1.0, 0.35, 0.30, 0.30]
            frontier = (
                beta[0] + beta[1] * log_labor + beta[2] * log_capital + beta[3] * log_materials
            )

            # Observed output with four components
            log_output = frontier + mu_i[i] - eta_i[i] + v_it - u_it

            rows.append(
                {
                    "firm_id": i + 1,
                    "year": year,
                    "log_output": round(log_output, 4),
                    "log_labor": round(log_labor, 4),
                    "log_capital": round(log_capital, 4),
                    "log_materials": round(log_materials, 4),
                    "sector": sectors[i],
                    "exporter": exporter[i],
                }
            )

    return pd.DataFrame(rows)


def generate_electricity_panel(
    n_generators: int = 60, n_years: int = 12, seed: int = 42
) -> pd.DataFrame:
    """
    Generate electricity generation panel with four-component model.

    Fuel-type heterogeneity: gas and hydro more efficient;
    nuclear has high persistent but low transient inefficiency.

    Parameters
    ----------
    n_generators : int, default 60
        Number of generators.
    n_years : int, default 12
        Number of years.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: generator_id, year, log_output, log_labor,
        log_capital, log_fuel, fuel_type.
    """
    rng = np.random.default_rng(seed)
    N = n_generators
    T = n_years
    start_year = 2008
    years = np.arange(start_year, start_year + T)

    fuel_type = rng.choice(["coal", "gas", "hydro", "nuclear"], size=N, p=[0.30, 0.30, 0.25, 0.15])

    # Heterogeneity by fuel type
    mu_i = rng.normal(0, np.sqrt(0.12), size=N)
    fuel_mu = np.where(
        fuel_type == "hydro",
        0.4,
        np.where(fuel_type == "gas", 0.25, np.where(fuel_type == "nuclear", 0.15, 0.0)),
    )
    mu_i += fuel_mu

    # Persistent inefficiency
    eta_i = np.abs(rng.normal(0, np.sqrt(0.07), size=N))
    # Nuclear: high persistent, gas/hydro: low persistent
    eta_i *= np.where(
        fuel_type == "nuclear",
        1.5,
        np.where(fuel_type == "gas", 0.7, np.where(fuel_type == "hydro", 0.6, 1.0)),
    )

    alpha_i = rng.normal(0, 0.2, size=N)

    # Transient inefficiency scale by fuel type
    u_scale = np.where(
        fuel_type == "nuclear",
        0.02,
        np.where(fuel_type == "hydro", 0.03, np.where(fuel_type == "gas", 0.04, 0.06)),
    )

    rows = []
    for _t_idx, year in enumerate(years):
        for i in range(N):
            v_it = rng.normal(0, np.sqrt(0.03))
            u_it = np.abs(rng.normal(0, np.sqrt(u_scale[i])))

            log_labor = rng.normal(4.5, 0.8) + alpha_i[i] * 0.3
            log_capital = rng.normal(6.0, 1.5) + alpha_i[i] * 0.5
            log_fuel = rng.normal(10.0, 2.0) + alpha_i[i] * 0.4

            # Hydro has low fuel input
            if fuel_type[i] == "hydro":
                log_fuel = rng.normal(6.0, 1.0)

            beta = [2.0, 0.20, 0.35, 0.40]
            frontier = beta[0] + beta[1] * log_labor + beta[2] * log_capital + beta[3] * log_fuel

            log_output = frontier + mu_i[i] - eta_i[i] + v_it - u_it

            rows.append(
                {
                    "generator_id": i + 1,
                    "year": year,
                    "log_output": round(log_output, 4),
                    "log_labor": round(log_labor, 4),
                    "log_capital": round(log_capital, 4),
                    "log_fuel": round(log_fuel, 4),
                    "fuel_type": fuel_type[i],
                }
            )

    return pd.DataFrame(rows)


def generate_hospital_panel(
    n_hospitals: int = 80, n_years: int = 10, seed: int = 42
) -> pd.DataFrame:
    """
    Generate hospital panel with BC95 inefficiency determinants.

    u_it ~ |N(μ_it, σ²_u)| where μ_it = δ₀ + δ₁·teaching + δ₂·accreditation + δ₃·occupancy_rate.
    δ = [0.5, -0.3, -0.25, -0.4] (accreditation and occupancy reduce inefficiency).

    Parameters
    ----------
    n_hospitals : int, default 80
        Number of hospitals.
    n_years : int, default 10
        Number of years.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hospital_id, year, log_output, log_labor,
        log_capital, log_supplies, teaching, accreditation, occupancy_rate, avg_stay.
    """
    rng = np.random.default_rng(seed)
    N = n_hospitals
    T = n_years
    start_year = 2010
    years = np.arange(start_year, start_year + T)

    # Hospital characteristics
    teaching = rng.binomial(1, 0.25, size=N)
    # Accreditation can change over time (initial + later acquisition)
    accred_initial = rng.binomial(1, 0.30, size=N)

    alpha_i = rng.normal(0, 0.2, size=N)

    rows = []
    for t_idx, year in enumerate(years):
        for i in range(N):
            # Accreditation: some hospitals acquire it over time
            if accred_initial[i] == 1:
                accreditation = 1
            else:
                # 5% chance per year of acquiring accreditation
                accreditation = 1 if rng.random() < 0.05 * t_idx else 0

            # Time-varying determinants
            occupancy_rate = np.clip(rng.normal(0.75, 0.12) + 0.005 * t_idx, 0.3, 0.98)
            avg_stay = np.clip(rng.normal(5.5, 2.0) - 0.1 * t_idx, 1.5, 15.0)

            # BC95 inefficiency mean
            delta = [0.5, -0.3, -0.25, -0.4]
            mu_u = (
                delta[0]
                + delta[1] * teaching[i]
                + delta[2] * accreditation
                + delta[3] * occupancy_rate
            )
            # Truncated normal inefficiency
            sigma_u = 0.2
            u_it = np.abs(rng.normal(max(mu_u, 0.01), sigma_u))

            # Inputs
            log_labor = rng.normal(5.5, 0.8) + alpha_i[i] * 0.4 + 0.01 * t_idx
            log_capital = rng.normal(4.5, 0.9) + alpha_i[i] * 0.3
            log_supplies = rng.normal(6.0, 0.7) + alpha_i[i] * 0.35 + 0.015 * t_idx

            # Production frontier
            beta = [2.0, 0.45, 0.30, 0.20]
            frontier = (
                beta[0]
                + beta[1] * log_labor
                + beta[2] * log_capital
                + beta[3] * log_supplies
                + 0.10 * teaching[i]
            )

            v_it = rng.normal(0, np.sqrt(0.03))
            log_output = frontier + v_it - u_it

            rows.append(
                {
                    "hospital_id": i + 1,
                    "year": year,
                    "log_output": round(log_output, 4),
                    "log_labor": round(log_labor, 4),
                    "log_capital": round(log_capital, 4),
                    "log_supplies": round(log_supplies, 4),
                    "teaching": teaching[i],
                    "accreditation": accreditation,
                    "occupancy_rate": round(occupancy_rate, 4),
                    "avg_stay": round(avg_stay, 2),
                }
            )

    return pd.DataFrame(rows)


def generate_school_panel(n_schools: int = 100, n_years: int = 8, seed: int = 42) -> pd.DataFrame:
    """
    Generate school panel with Wang (2002) heteroscedastic inefficiency.

    Both location and scale effects on inefficiency:
        μ_it = δ₀ + δ₁·teacher_experience + δ₂·class_size + δ₃·ses_index
        log(σ²_u,it) = γ₀ + γ₁·school_type_private + γ₂·log_budget

    Parameters
    ----------
    n_schools : int, default 100
        Number of schools.
    n_years : int, default 8
        Number of years.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: school_id, year, log_output, log_teachers,
        log_budget, log_facilities, teacher_experience, class_size, ses_index, school_type.
    """
    rng = np.random.default_rng(seed)
    N = n_schools
    T = n_years
    start_year = 2012
    years = np.arange(start_year, start_year + T)

    school_type = rng.choice(["public", "private", "charter"], size=N, p=[0.60, 0.25, 0.15])

    alpha_i = rng.normal(0, 0.15, size=N)

    # Base SES by school type
    ses_base = np.where(
        school_type == "private", 0.7, np.where(school_type == "charter", 0.55, 0.4)
    )

    rows = []
    for t_idx, year in enumerate(years):
        for i in range(N):
            # Time-varying characteristics
            teacher_experience = np.clip(rng.normal(12, 5) + 0.3 * t_idx, 1, 35)
            class_size = np.clip(rng.normal(25, 8), 10, 45)
            ses_index = np.clip(rng.normal(ses_base[i], 0.15), 0.05, 0.95)

            # Wang (2002) location effect
            delta = [0.3, -0.015, 0.012, -0.5]
            mu_u = (
                delta[0]
                + delta[1] * teacher_experience
                + delta[2] * class_size
                + delta[3] * ses_index
            )

            # Wang (2002) scale effect
            log_budget = rng.normal(8.5, 0.4) + 0.02 * t_idx
            is_private = 1 if school_type[i] == "private" else 0
            gamma = [0.0, -0.3, -0.1]
            log_sigma_u_sq = gamma[0] + gamma[1] * is_private + gamma[2] * log_budget
            sigma_u = np.sqrt(np.exp(log_sigma_u_sq))

            u_it = np.abs(rng.normal(max(mu_u, 0.01), sigma_u))

            # Inputs
            log_teachers = rng.normal(3.5, 0.6) + alpha_i[i] * 0.3
            log_facilities = rng.normal(4.0, 0.5) + alpha_i[i] * 0.2

            # Production frontier (education production function)
            beta = [3.0, 0.30, 0.25, 0.15]
            frontier = (
                beta[0] + beta[1] * log_teachers + beta[2] * log_budget + beta[3] * log_facilities
            )

            v_it = rng.normal(0, np.sqrt(0.02))
            log_output = frontier + v_it - u_it

            rows.append(
                {
                    "school_id": i + 1,
                    "year": year,
                    "log_output": round(log_output, 4),
                    "log_teachers": round(log_teachers, 4),
                    "log_budget": round(log_budget, 4),
                    "log_facilities": round(log_facilities, 4),
                    "teacher_experience": round(teacher_experience, 1),
                    "class_size": round(class_size, 1),
                    "ses_index": round(ses_index, 4),
                    "school_type": school_type[i],
                }
            )

    return pd.DataFrame(rows)


def generate_dairy_farm_data(n_farms: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate cross-section dairy farm data with Translog production function.

    Large N for robust testing and model comparison.
    Organic farms have different technology but not necessarily lower efficiency.

    Parameters
    ----------
    n_farms : int, default 500
        Number of dairy farms.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: farm_id, log_milk, log_cows, log_feed,
        log_land, log_labor, organic, breed, cooperative.
    """
    rng = np.random.default_rng(seed)
    N = n_farms

    organic = rng.binomial(1, 0.20, size=N)
    breed = rng.choice(["holstein", "jersey", "mixed"], size=N, p=[0.50, 0.25, 0.25])
    cooperative = rng.binomial(1, 0.45, size=N)

    # Generate inputs
    log_cows = rng.normal(3.5, 1.0, size=N)
    log_feed = rng.normal(8.0, 1.0, size=N)
    log_land = rng.normal(3.0, 1.2, size=N)
    log_labor = rng.normal(7.0, 0.8, size=N)

    # Organic farms: different input mix
    log_feed[organic == 1] -= 0.3  # less feed
    log_land[organic == 1] += 0.4  # more land

    # Translog production function (centered)
    lc = log_cows - log_cows.mean()
    lf = log_feed - log_feed.mean()
    ll = log_land - log_land.mean()
    llb = log_labor - log_labor.mean()

    # Main effects
    beta_0 = 4.0
    b_cows, b_feed, b_land, b_labor = 0.35, 0.25, 0.15, 0.20

    # Squared terms
    b_cows2, b_feed2, b_land2, b_labor2 = -0.03, -0.02, -0.01, -0.02

    # Interaction terms
    b_cows_feed = 0.05
    b_cows_land = 0.03
    b_feed_labor = 0.02

    frontier = (
        beta_0
        + b_cows * lc
        + b_feed * lf
        + b_land * ll
        + b_labor * llb
        + 0.5 * b_cows2 * lc**2
        + 0.5 * b_feed2 * lf**2
        + 0.5 * b_land2 * ll**2
        + 0.5 * b_labor2 * llb**2
        + b_cows_feed * lc * lf
        + b_cows_land * lc * ll
        + b_feed_labor * lf * llb
    )

    # Breed effects on technology
    breed_effect = np.where(breed == "holstein", 0.15, np.where(breed == "jersey", 0.05, 0.0))
    frontier += breed_effect

    # Organic technology shift (different but not necessarily worse)
    frontier += organic * 0.1

    # Noise
    v = rng.normal(0, np.sqrt(0.03), size=N)

    # Truncated normal inefficiency
    u = np.abs(rng.normal(0.2, np.sqrt(0.08), size=N))
    # Cooperative membership slightly reduces inefficiency
    u *= 1 - 0.1 * cooperative

    log_milk = frontier + v - u

    return pd.DataFrame(
        {
            "farm_id": np.arange(1, N + 1),
            "log_milk": np.round(log_milk, 4),
            "log_cows": np.round(log_cows, 4),
            "log_feed": np.round(log_feed, 4),
            "log_land": np.round(log_land, 4),
            "log_labor": np.round(log_labor, 4),
            "organic": organic,
            "breed": breed,
            "cooperative": cooperative,
        }
    )


def generate_telecom_panel(n_firms: int = 40, n_years: int = 15, seed: int = 42) -> pd.DataFrame:
    """
    Generate telecom panel with CSS distribution-free model.

    Time-varying efficiency with technology transitions (2G→3G→4G).

    Parameters
    ----------
    n_firms : int, default 40
        Number of telecom firms.
    n_years : int, default 15
        Number of years.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: firm_id, year, log_output, log_labor,
        log_capital, log_spectrum, technology, market_share.
    """
    rng = np.random.default_rng(seed)
    N = n_firms
    T = n_years
    start_year = 2005
    years = np.arange(start_year, start_year + T)

    alpha_i = rng.normal(0, 0.3, size=N)

    # Base inefficiency
    u_base = np.abs(rng.normal(0, np.sqrt(0.06), size=N))

    # Market shares (time-varying)
    initial_share = rng.dirichlet(np.ones(N) * 2)

    rows = []
    for t_idx, year in enumerate(years):
        # Technology transitions
        if year < 2010:
            tech_probs = [0.7, 0.3, 0.0]
        elif year < 2015:
            tech_probs = [0.2, 0.6, 0.2]
        else:
            tech_probs = [0.0, 0.3, 0.7]

        for i in range(N):
            technology = rng.choice(["2G", "3G", "4G"], p=tech_probs)

            # Technology effect on efficiency
            tech_effect = {"2G": 1.0, "3G": 0.8, "4G": 0.6}
            u_it = u_base[i] * tech_effect[technology]

            # CSS: time-varying inefficiency without parametric form
            u_it *= 1 - 0.02 * t_idx  # general improvement trend
            u_it = max(u_it, 0.001)

            # Market share evolution
            market_share = np.clip(initial_share[i] + rng.normal(0, 0.02), 0.01, 0.80)

            # Inputs
            log_labor = rng.normal(7.0, 1.0) + alpha_i[i] * 0.3 - 0.02 * t_idx
            log_capital = rng.normal(8.5, 1.3) + alpha_i[i] * 0.5 + 0.04 * t_idx
            log_spectrum = rng.normal(5.0, 0.8) + alpha_i[i] * 0.2 + 0.03 * t_idx

            beta = [1.5, 0.25, 0.40, 0.30]
            frontier = (
                beta[0] + beta[1] * log_labor + beta[2] * log_capital + beta[3] * log_spectrum
            )

            # Technology boost to frontier
            tech_boost = {"2G": 0.0, "3G": 0.3, "4G": 0.6}
            frontier += tech_boost[technology]

            v_it = rng.normal(0, np.sqrt(0.03))
            log_output = frontier + v_it - u_it

            rows.append(
                {
                    "firm_id": i + 1,
                    "year": year,
                    "log_output": round(log_output, 4),
                    "log_labor": round(log_labor, 4),
                    "log_capital": round(log_capital, 4),
                    "log_spectrum": round(log_spectrum, 4),
                    "technology": technology,
                    "market_share": round(market_share, 4),
                }
            )

    return pd.DataFrame(rows)


def generate_brazilian_firms(n_firms: int = 500, n_years: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Generate Brazilian manufacturing panel for complete case study.

    Complex DGP combining:
    - Translog production function (for RTS testing)
    - BC95 determinants: exporter, foreign_owned, r_and_d, firm_age
    - Four-component structure for TFP decomposition
    - Regional technology differences (SE more productive)
    - Sector-specific production functions
    - Structural break in 2015-2016 (Brazilian recession)

    Parameters
    ----------
    n_firms : int, default 500
        Number of firms.
    n_years : int, default 10
        Number of years.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: firm_id, year, log_output, log_labor,
        log_capital, log_materials, sector, region, exporter, foreign_owned,
        firm_age, r_and_d.
    """
    rng = np.random.default_rng(seed)
    N = n_firms
    T = n_years
    start_year = 2010
    years = np.arange(start_year, start_year + T)

    # Firm characteristics
    sectors = rng.choice(
        [
            "food",
            "textiles",
            "chemicals",
            "metals",
            "machinery",
            "electronics",
            "automotive",
            "pharma",
        ],
        size=N,
        p=[0.20, 0.12, 0.13, 0.13, 0.12, 0.12, 0.10, 0.08],
    )
    regions = rng.choice(["N", "NE", "SE", "S", "CO"], size=N, p=[0.08, 0.18, 0.40, 0.22, 0.12])
    exporter = rng.binomial(1, 0.30, size=N)
    foreign_owned = rng.binomial(1, 0.15, size=N)
    firm_age = np.clip(rng.normal(25, 15, size=N), 1, 80).astype(float)
    r_and_d = np.clip(rng.exponential(1.5, size=N), 0, 15)
    # Foreign firms and exporters invest more in R&D
    r_and_d *= 1 + 0.5 * exporter + 0.8 * foreign_owned

    # Regional technology parameters
    region_tech = np.where(
        regions == "SE",
        0.3,
        np.where(
            regions == "S",
            0.2,
            np.where(regions == "CO", 0.1, np.where(regions == "NE", -0.1, -0.2)),
        ),
    )

    # Sector-specific intercepts
    sector_intercept = {
        "food": 0.0,
        "textiles": -0.1,
        "chemicals": 0.2,
        "metals": 0.1,
        "machinery": 0.15,
        "electronics": 0.3,
        "automotive": 0.25,
        "pharma": 0.35,
    }
    sector_eff = np.array([sector_intercept[s] for s in sectors])

    # Four-component structure
    mu_i = rng.normal(0, np.sqrt(0.12), size=N) + region_tech + sector_eff

    eta_i = np.abs(rng.normal(0, np.sqrt(0.06), size=N))
    # BC95 determinants reduce persistent inefficiency
    eta_i *= np.exp(
        -0.15 * exporter
        - 0.20 * foreign_owned
        - 0.03 * r_and_d
        - 0.005 * np.clip(firm_age - 10, 0, None)
    )

    alpha_i = rng.normal(0, 0.15, size=N)

    rows = []
    for t_idx, year in enumerate(years):
        # Recession effect 2015-2016
        recession = 1.0
        if year == 2015:
            recession = 1.3  # higher inefficiency
        elif year == 2016:
            recession = 1.5

        for i in range(N):
            # Transient inefficiency
            v_it = rng.normal(0, np.sqrt(0.03))
            u_it = np.abs(rng.normal(0, np.sqrt(0.04))) * recession

            # Inputs with firm heterogeneity and time trends
            log_labor = rng.normal(4.5, 1.5) + alpha_i[i] * 0.5 + 0.01 * t_idx
            log_capital = rng.normal(7.0, 2.0) + alpha_i[i] * 0.7 + 0.02 * t_idx
            log_materials = rng.normal(7.5, 1.8) + alpha_i[i] * 0.6 + 0.015 * t_idx

            # Translog terms (centered)
            lk = log_capital - 7.0
            ll = log_labor - 4.5
            lm = log_materials - 7.5

            # Production frontier (Translog)
            frontier = (
                2.0
                + 0.35 * ll
                + 0.30 * lk
                + 0.30 * lm
                + 0.5 * (-0.02) * ll**2
                + 0.5 * (-0.015) * lk**2
                + 0.5 * (-0.01) * lm**2
                + 0.03 * ll * lk
                + 0.02 * ll * lm
                + 0.015 * lk * lm
            )

            # Time trend (technical progress)
            frontier += 0.01 * t_idx

            # Recession productivity shock
            if year in (2015, 2016):
                frontier -= 0.05

            # Firm age (updated each year)
            current_age = firm_age[i] + t_idx

            log_output = frontier + mu_i[i] - eta_i[i] + v_it - u_it

            rows.append(
                {
                    "firm_id": i + 1,
                    "year": year,
                    "log_output": round(log_output, 4),
                    "log_labor": round(log_labor, 4),
                    "log_capital": round(log_capital, 4),
                    "log_materials": round(log_materials, 4),
                    "sector": sectors[i],
                    "region": regions[i],
                    "exporter": exporter[i],
                    "foreign_owned": foreign_owned[i],
                    "firm_age": round(current_age, 1),
                    "r_and_d": round(r_and_d[i], 2),
                }
            )

    return pd.DataFrame(rows)
