"""
Data generators for Production & Deployment tutorials.

All functions produce reproducible panel datasets via explicit seed parameters.
See 00_ESTRUTURA.md for detailed DGP specifications.
"""

import numpy as np
import pandas as pd


def generate_firm_panel(n_firms=100, n_years=20, seed=42) -> pd.DataFrame:
    """
    Generate firm-level panel with FE-correlated regressors.

    DGP: investment = alpha_i + 0.3*value + 0.2*capital + 0.15*sales + epsilon
    where alpha_i is correlated with capital (firm fixed effects).

    Parameters
    ----------
    n_firms : int
        Number of firms.
    n_years : int
        Number of years per firm.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel with columns: firm_id, year, investment, value, capital, sales, sector.
    """
    rng = np.random.RandomState(seed)

    sectors = ["Manufacturing", "Technology", "Finance", "Healthcare", "Energy"]

    firm_ids = np.repeat(np.arange(1, n_firms + 1), n_years)
    years = np.tile(np.arange(2000, 2000 + n_years), n_firms)

    # Firm fixed effects (correlated with capital)
    alpha_i = rng.normal(0, 0.8, n_firms)
    alpha_i_expanded = np.repeat(alpha_i, n_years)

    # Sector assignment (fixed per firm)
    firm_sector = rng.choice(sectors, n_firms)
    sector_expanded = np.repeat(firm_sector, n_years)

    # Regressors with firm-level heterogeneity
    capital_base = rng.normal(4.5, 0.8, n_firms)
    capital = np.repeat(capital_base, n_years) + rng.normal(0, 0.4, n_firms * n_years)
    # Correlate alpha with capital
    alpha_i_corr = alpha_i + 0.3 * (capital_base - 4.5)
    alpha_i_expanded = np.repeat(alpha_i_corr, n_years)

    value = rng.normal(5.0, 1.5, n_firms * n_years)
    sales = rng.normal(6.0, 1.8, n_firms * n_years)

    # DGP
    epsilon = rng.normal(0, 0.5, n_firms * n_years)
    investment = alpha_i_expanded + 0.3 * value + 0.2 * capital + 0.15 * sales + epsilon

    df = pd.DataFrame(
        {
            "firm_id": firm_ids,
            "year": years,
            "investment": np.round(investment, 4),
            "value": np.round(value, 4),
            "capital": np.round(capital, 4),
            "sales": np.round(sales, 4),
            "sector": sector_expanded,
        }
    )
    return df


def generate_bank_lgd(n_contracts=200, n_months=15, seed=42) -> pd.DataFrame:
    """
    Generate banking LGD dynamic panel (AR(1) with exogenous variables).

    DGP: lgd_logit_t = 0.6*lgd_logit_{t-1} + 0.1*saldo_real + 0.05*pib_growth
                       - 0.03*collateral_ratio + fe_i + e_it

    Parameters
    ----------
    n_contracts : int
        Number of loan contracts.
    n_months : int
        Number of monthly periods.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel with columns: contract_id, month, lgd_logit, saldo_real,
        pib_growth, selic, collateral_ratio.
    """
    rng = np.random.RandomState(seed)

    # Contract fixed effects
    fe = rng.normal(0, 0.3, n_contracts)

    # Macro variables (common across contracts, vary by month)
    pib_growth_common = rng.normal(1.0, 2.0, n_months)
    selic_common = np.cumsum(rng.normal(0, 0.5, n_months)) + 10.0

    rows = []
    for i in range(n_contracts):
        lgd_prev = rng.normal(-1.5, 0.5)  # initial value
        saldo_base = rng.normal(10, 1.5)
        collateral_base = rng.normal(0.6, 0.15)

        for t in range(n_months):
            saldo_real = saldo_base + rng.normal(0, 0.3)
            pib_growth = pib_growth_common[t] + rng.normal(0, 0.3)
            selic = selic_common[t] + rng.normal(0, 0.3)
            collateral_ratio = max(0, collateral_base + rng.normal(0, 0.05))

            e = rng.normal(0, 0.3)
            lgd_logit = (
                0.6 * lgd_prev
                + 0.1 * saldo_real
                + 0.05 * pib_growth
                - 0.03 * collateral_ratio
                + fe[i]
                + e
            )
            rows.append(
                {
                    "contract_id": i + 1,
                    "month": t + 1,
                    "lgd_logit": round(lgd_logit, 4),
                    "saldo_real": round(saldo_real, 4),
                    "pib_growth": round(pib_growth, 4),
                    "selic": round(selic, 4),
                    "collateral_ratio": round(collateral_ratio, 4),
                }
            )
            lgd_prev = lgd_logit

    return pd.DataFrame(rows)


def generate_macro_quarterly(n_countries=30, n_quarters=40, seed=42) -> pd.DataFrame:
    """
    Generate macro quarterly panel with VAR(1) dynamics and country fixed effects.

    Parameters
    ----------
    n_countries : int
        Number of countries.
    n_quarters : int
        Number of quarterly periods.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel with columns: country, quarter, gdp_growth, inflation, interest_rate.
    """
    rng = np.random.RandomState(seed)

    country_names = [f"Country_{i + 1:02d}" for i in range(n_countries)]

    # Country fixed effects
    fe_gdp = rng.normal(2.0, 1.0, n_countries)
    fe_inf = rng.normal(3.0, 1.0, n_countries)
    fe_rate = rng.normal(5.0, 1.5, n_countries)

    rows = []
    for c in range(n_countries):
        gdp_prev = fe_gdp[c]
        inf_prev = fe_inf[c]
        rate_prev = fe_rate[c]

        for q in range(n_quarters):
            e_gdp = rng.normal(0, 0.5)
            e_inf = rng.normal(0, 0.4)
            e_rate = rng.normal(0, 0.6)

            # VAR(1) dynamics
            gdp = 0.3 * gdp_prev + 0.1 * inf_prev - 0.05 * rate_prev + 0.6 * fe_gdp[c] + e_gdp
            inf = 0.1 * gdp_prev + 0.4 * inf_prev + 0.05 * rate_prev + 0.5 * fe_inf[c] + e_inf
            rate = -0.05 * gdp_prev + 0.15 * inf_prev + 0.5 * rate_prev + 0.35 * fe_rate[c] + e_rate

            rows.append(
                {
                    "country": country_names[c],
                    "quarter": q + 1,
                    "gdp_growth": round(gdp, 4),
                    "inflation": round(inf, 4),
                    "interest_rate": round(rate, 4),
                }
            )
            gdp_prev, inf_prev, rate_prev = gdp, inf, rate

    return pd.DataFrame(rows)


def generate_new_firms(n_firms=20, n_years=5, original_firms=10, seed=42) -> pd.DataFrame:
    """
    Generate out-of-sample firm data (mix of known and new firms).

    Parameters
    ----------
    n_firms : int
        Total number of firms (original + new).
    n_years : int
        Number of years per firm.
    original_firms : int
        Number of firms from the original panel (IDs 1..original_firms).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Same columns as firm_panel.csv.
    """
    rng = np.random.RandomState(seed)

    sectors = ["Manufacturing", "Technology", "Finance", "Healthcare", "Energy"]

    # Mix of original firm IDs and new firm IDs
    known_ids = np.arange(1, original_firms + 1)
    new_ids = np.arange(101, 101 + (n_firms - original_firms))
    all_ids = np.concatenate([known_ids, new_ids])

    firm_ids = np.repeat(all_ids, n_years)
    years = np.tile(np.arange(2020, 2020 + n_years), n_firms)

    alpha_i = rng.normal(0, 0.8, n_firms)
    capital_base = rng.normal(4.5, 0.8, n_firms)
    alpha_i_corr = alpha_i + 0.3 * (capital_base - 4.5)
    alpha_i_expanded = np.repeat(alpha_i_corr, n_years)

    capital = np.repeat(capital_base, n_years) + rng.normal(0, 0.4, n_firms * n_years)
    value = rng.normal(5.0, 1.5, n_firms * n_years)
    sales = rng.normal(6.0, 1.8, n_firms * n_years)

    firm_sector = rng.choice(sectors, n_firms)
    sector_expanded = np.repeat(firm_sector, n_years)

    epsilon = rng.normal(0, 0.5, n_firms * n_years)
    investment = alpha_i_expanded + 0.3 * value + 0.2 * capital + 0.15 * sales + epsilon

    return pd.DataFrame(
        {
            "firm_id": firm_ids,
            "year": years,
            "investment": np.round(investment, 4),
            "value": np.round(value, 4),
            "capital": np.round(capital, 4),
            "sales": np.round(sales, 4),
            "sector": sector_expanded,
        }
    )


def generate_new_bank_data(n_contracts=50, n_months=3, seed=42) -> pd.DataFrame:
    """
    Generate new bank observations for production prediction.

    Same columns and DGP as bank_lgd.csv but smaller (out-of-sample).

    Parameters
    ----------
    n_contracts : int
        Number of loan contracts.
    n_months : int
        Number of monthly periods.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Same columns as bank_lgd.csv.
    """
    return generate_bank_lgd(n_contracts=n_contracts, n_months=n_months, seed=seed + 100)


def generate_future_macro(n_countries=30, n_quarters=4, seed=42) -> pd.DataFrame:
    """
    Generate future exogenous variables for forecasting (no dependent variable).

    Parameters
    ----------
    n_countries : int
        Number of countries.
    n_quarters : int
        Number of future quarters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: country, quarter, inflation, interest_rate (no gdp_growth).
    """
    rng = np.random.RandomState(seed + 200)

    country_names = [f"Country_{i + 1:02d}" for i in range(n_countries)]

    rows = []
    for c in range(n_countries):
        base_inf = rng.normal(3.0, 1.0)
        base_rate = rng.normal(5.0, 1.5)
        for q in range(n_quarters):
            rows.append(
                {
                    "country": country_names[c],
                    "quarter": 41 + q,  # future quarters after 40
                    "inflation": round(base_inf + rng.normal(0, 0.3), 4),
                    "interest_rate": round(base_rate + rng.normal(0, 0.4), 4),
                }
            )

    return pd.DataFrame(rows)
