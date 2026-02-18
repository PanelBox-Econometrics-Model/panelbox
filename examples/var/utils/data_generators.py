"""
Data generators for VAR tutorial series.

All functions generate synthetic panel data with realistic economic dynamics.
Each function sets np.random.seed for reproducibility.

Functions:
- generate_macro_panel: VAR(2) macro data (30 countries, 40 quarters)
- generate_energy_panel: Energy price transmission (25 countries, 60 quarters)
- generate_finance_panel: Asset returns (50 countries, 100 periods)
- generate_monetary_policy_panel: Monetary policy with crisis (25 countries, 80 quarters)
- generate_trade_panel: Trade for Granger causality (40 countries, 50 years)
- generate_ppp_panel: PPP for VECM cointegration (20 countries, 60 quarters)
- generate_interest_parity_panel: Interest parity for VECM (20 countries, 60 quarters)
- generate_dynamic_panel: Dynamic panel for GMM (100 countries, 15 years)
"""

from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Country name lists
# ---------------------------------------------------------------------------

_COUNTRIES_30 = [
    "USA",
    "GBR",
    "DEU",
    "FRA",
    "JPN",
    "CAN",
    "AUS",
    "ITA",
    "ESP",
    "NLD",
    "CHE",
    "SWE",
    "NOR",
    "DNK",
    "BEL",
    "AUT",
    "FIN",
    "IRL",
    "PRT",
    "GRC",
    "BRA",
    "MEX",
    "IND",
    "CHN",
    "KOR",
    "ZAF",
    "TUR",
    "POL",
    "CZE",
    "HUN",
]

_COUNTRIES_EXTENSION_20 = [
    "RUS",
    "IDN",
    "THA",
    "MYS",
    "PHL",
    "CHL",
    "COL",
    "PER",
    "ARG",
    "NGA",
    "EGY",
    "ISR",
    "SAU",
    "ARE",
    "QAT",
    "KWT",
    "SGP",
    "HKG",
    "TWN",
    "NZL",
]

_COUNTRIES_50 = _COUNTRIES_30 + _COUNTRIES_EXTENSION_20

_COUNTRIES_OECD_25 = [
    "USA",
    "GBR",
    "DEU",
    "FRA",
    "JPN",
    "CAN",
    "AUS",
    "ITA",
    "ESP",
    "NLD",
    "CHE",
    "SWE",
    "NOR",
    "DNK",
    "BEL",
    "AUT",
    "FIN",
    "IRL",
    "PRT",
    "GRC",
    "KOR",
    "MEX",
    "POL",
    "CZE",
    "HUN",
]

_COUNTRIES_40 = _COUNTRIES_30 + _COUNTRIES_EXTENSION_20[:10]

_COUNTRIES_20 = _COUNTRIES_30[:20]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _build_sigma(std_devs: np.ndarray, correlations: dict) -> np.ndarray:
    """Build a covariance matrix from standard deviations and pairwise correlations.

    Parameters
    ----------
    std_devs : array of standard deviations (length k)
    correlations : dict mapping (i, j) -> rho  (upper triangle only is fine)

    Returns
    -------
    Sigma : k x k positive-definite covariance matrix
    """
    k = len(std_devs)
    corr = np.eye(k)
    for (i, j), rho in correlations.items():
        corr[i, j] = rho
        corr[j, i] = rho
    D = np.diag(std_devs)
    Sigma = D @ corr @ D
    return Sigma


def _check_var_stability(A_matrices: list, label: str = "") -> None:
    """Verify that the VAR companion matrix has all eigenvalues inside the unit circle.

    Parameters
    ----------
    A_matrices : list of k x k coefficient matrices [A_1, A_2, ...]
    label : optional label for error message
    """
    p = len(A_matrices)
    k = A_matrices[0].shape[0]
    if p == 1:
        companion = A_matrices[0]
    else:
        # Build companion matrix for VAR(p)
        companion = np.zeros((k * p, k * p))
        for i, A in enumerate(A_matrices):
            companion[:k, i * k : (i + 1) * k] = A
        if p > 1:
            companion[k:, : k * (p - 1)] = np.eye(k * (p - 1))

    eigvals = np.linalg.eigvals(companion)
    max_mod = np.max(np.abs(eigvals))
    if max_mod >= 1.0:
        raise ValueError(
            f"VAR process '{label}' is not stable: " f"max eigenvalue modulus = {max_mod:.4f} >= 1"
        )


def _quarter_labels(start_year: int, n_quarters: int) -> list:
    """Generate a list of quarter labels like '2010-Q1', '2010-Q2', ..."""
    labels = []
    y, q = start_year, 1
    for _ in range(n_quarters):
        labels.append(f"{y}-Q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return labels


def _simulate_var(
    A_matrices: list,
    Sigma: np.ndarray,
    mu_i: np.ndarray,
    n_obs: int,
    burn_in: int = 50,
) -> np.ndarray:
    """Simulate a single-entity VAR(p) process.

    Parameters
    ----------
    A_matrices : list of k x k coefficient matrices [A_1, ..., A_p]
    Sigma : k x k innovation covariance matrix
    mu_i : k-vector of entity fixed effects (unconditional mean)
    n_obs : number of post-burn-in observations to return
    burn_in : number of initial observations to discard

    Returns
    -------
    Y : (n_obs, k) array
    """
    p = len(A_matrices)
    k = A_matrices[0].shape[0]
    total = n_obs + burn_in
    L = np.linalg.cholesky(Sigma)

    Y = np.zeros((total + p, k))
    # Initialise first p observations at the fixed effect level
    for t in range(p):
        Y[t] = mu_i

    for t in range(p, total + p):
        eps = L @ np.random.randn(k)
        val = mu_i.copy()
        for lag, A in enumerate(A_matrices):
            val = val + A @ (Y[t - 1 - lag] - mu_i)
        Y[t] = val + eps

    return Y[p + burn_in :]


# ---------------------------------------------------------------------------
# 1. Macro Panel  --  VAR(2)
# ---------------------------------------------------------------------------


def generate_macro_panel(
    n_countries: int = 30,
    n_quarters: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate macro panel data from a VAR(2) process.

    Variables: gdp_growth, inflation, interest_rate, unemployment, exchange_rate.

    Parameters
    ----------
    n_countries : number of countries (default 30)
    n_quarters : number of quarters (default 40)
    seed : random seed

    Returns
    -------
    DataFrame with columns: country, quarter, gdp_growth, inflation,
        interest_rate, unemployment, exchange_rate
    """
    np.random.seed(seed)

    countries = _COUNTRIES_30[:n_countries]
    quarters = _quarter_labels(2010, n_quarters)

    # A_1: 5x5
    A_1 = np.diag([0.4, 0.5, 0.6, 0.3, 0.45])
    A_1[0, 2] = -0.15  # GDP responds to interest_rate lag
    A_1[1, 0] = 0.10  # Inflation responds to GDP lag
    A_1[2, 1] = 0.20  # Interest_rate responds to inflation lag (Taylor rule)
    A_1[3, 0] = -0.25  # Unemployment responds to GDP lag (Okun's law)
    A_1[4, 2] = 0.10  # Exchange_rate responds to interest_rate lag (UIP)

    # A_2: 5x5 -- weaker second lag
    A_2 = np.diag([0.10, 0.05, 0.08, 0.05, 0.03])
    A_2[0, 1] = 0.05  # Small GDP-inflation feedback at lag 2
    A_2[2, 0] = 0.05  # Small interest-GDP feedback at lag 2
    A_2[3, 2] = 0.05  # Small unemployment-interest feedback at lag 2

    _check_var_stability([A_1, A_2], label="macro_panel")

    # Error covariance
    std_devs = np.sqrt([1.5, 1.8, 1.2, 2.0, 2.5])
    correlations = {
        (0, 1): 0.15,  # GDP-inflation
        (0, 3): -0.30,  # GDP-unemployment
        (0, 2): 0.05,
        (1, 2): 0.05,
        (2, 3): 0.05,
        (3, 4): 0.05,
    }
    Sigma = _build_sigma(std_devs, correlations)

    # Country fixed effects
    mu_global = np.array([2.0, 3.0, 4.0, 7.0, 100.0])
    sigma_mu = np.array([0.8, 1.0, 1.5, 2.0, 10.0])

    records = []
    for c_idx, cname in enumerate(countries):
        mu_i = mu_global + sigma_mu * np.random.randn(5)
        Y = _simulate_var([A_1, A_2], Sigma, mu_i, n_quarters, burn_in=50)

        # Post-processing
        Y[:, 2] = np.maximum(0.0, Y[:, 2])  # interest_rate >= 0
        Y[:, 3] = np.maximum(0.0, Y[:, 3])  # unemployment >= 0

        for t in range(n_quarters):
            records.append(
                {
                    "country": cname,
                    "quarter": quarters[t],
                    "gdp_growth": Y[t, 0],
                    "inflation": Y[t, 1],
                    "interest_rate": Y[t, 2],
                    "unemployment": Y[t, 3],
                    "exchange_rate": Y[t, 4],
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Energy Panel  --  VAR(1) price transmission chain
# ---------------------------------------------------------------------------


def generate_energy_panel(
    n_countries: int = 25,
    n_quarters: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate energy price panel from a VAR(1) price transmission chain.

    Variables (in logs): oil_price, gas_price, electricity_price.

    Parameters
    ----------
    n_countries : number of countries (default 25)
    n_quarters : number of quarters (default 60)
    seed : random seed

    Returns
    -------
    DataFrame with columns: country, quarter, oil_price, gas_price, electricity_price
    """
    np.random.seed(seed)

    countries = _COUNTRIES_30[:n_countries]
    quarters = _quarter_labels(2005, n_quarters)

    # A_1: 3x3 with transmission chain oil -> gas -> electricity
    A_1 = np.diag([0.7, 0.5, 0.4])
    A_1[1, 0] = 0.30  # oil -> gas
    A_1[2, 1] = 0.25  # gas -> electricity
    A_1[2, 0] = 0.10  # oil -> electricity

    _check_var_stability([A_1], label="energy_panel")

    std_devs = np.sqrt([0.04, 0.06, 0.02])
    correlations = {(0, 1): 0.30}
    Sigma = _build_sigma(std_devs, correlations)

    mu_global = np.array([4.0, 1.5, 4.5])
    sigma_mu = np.array([0.15, 0.20, 0.10])

    records = []
    for cname in countries:
        mu_i = mu_global + sigma_mu * np.random.randn(3)
        Y = _simulate_var([A_1], Sigma, mu_i, n_quarters, burn_in=50)

        for t in range(n_quarters):
            records.append(
                {
                    "country": cname,
                    "quarter": quarters[t],
                    "oil_price": Y[t, 0],
                    "gas_price": Y[t, 1],
                    "electricity_price": Y[t, 2],
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Finance Panel  --  VAR(1) asset returns
# ---------------------------------------------------------------------------


def generate_finance_panel(
    n_countries: int = 50,
    n_periods: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate financial asset return panel from a VAR(1) process.

    Variables: stock_return, bond_return, fx_return, commodity_return.

    Parameters
    ----------
    n_countries : number of countries (default 50)
    n_periods : number of time periods (default 100)
    seed : random seed

    Returns
    -------
    DataFrame with columns: country, time, stock_return, bond_return,
        fx_return, commodity_return
    """
    np.random.seed(seed)

    countries = _COUNTRIES_50[:n_countries]
    times = list(range(1, n_periods + 1))

    # A_1: 4x4
    A_1 = np.diag([0.10, 0.05, -0.02, 0.08])
    A_1[1, 0] = -0.05  # stock -> bond (flight to quality)
    A_1[2, 0] = 0.03  # stock -> fx

    _check_var_stability([A_1], label="finance_panel")

    std_devs = np.sqrt([9.0, 2.25, 4.0, 6.25])
    correlations = {
        (0, 1): -0.20,  # stock-bond negative
        (0, 2): 0.15,  # stock-fx positive
    }
    Sigma = _build_sigma(std_devs, correlations)

    mu_global = np.array([0.5, 0.3, 0.0, 0.2])
    sigma_mu = np.array([0.3, 0.1, 0.15, 0.2])

    records = []
    for cname in countries:
        mu_i = mu_global + sigma_mu * np.random.randn(4)
        Y = _simulate_var([A_1], Sigma, mu_i, n_periods, burn_in=50)

        for t in range(n_periods):
            records.append(
                {
                    "country": cname,
                    "time": times[t],
                    "stock_return": Y[t, 0],
                    "bond_return": Y[t, 1],
                    "fx_return": Y[t, 2],
                    "commodity_return": Y[t, 3],
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Monetary Policy Panel  --  structural break at 2008
# ---------------------------------------------------------------------------


def generate_monetary_policy_panel(
    n_countries: int = 25,
    n_quarters: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate monetary policy panel with structural break around 2008.

    Variables: gdp_growth, inflation, interest_rate, unemployment.

    Parameters
    ----------
    n_countries : number of countries (default 25)
    n_quarters : number of quarters (default 80, covering 2000-Q1 to 2019-Q4)
    seed : random seed

    Returns
    -------
    DataFrame with columns: country, quarter, gdp_growth, inflation,
        interest_rate, unemployment
    """
    np.random.seed(seed)

    countries = _COUNTRIES_OECD_25[:n_countries]
    quarters = _quarter_labels(2000, n_quarters)

    # Identify crisis quarter index: 2008-Q3 is quarter index (2008-2000)*4 + 2 = 34
    crisis_idx = 34

    # Pre-crisis A_1
    A_1_pre = np.diag([0.4, 0.5, 0.6, 0.3])
    A_1_pre[0, 2] = -0.15  # GDP responds to interest_rate
    A_1_pre[1, 0] = 0.10  # Inflation responds to GDP
    A_1_pre[2, 1] = 0.20  # Taylor rule
    A_1_pre[3, 0] = -0.25  # Okun's law

    # Post-crisis A_1: lower persistence, weaker transmission
    A_1_post = np.diag([0.35, 0.40, 0.45, 0.25])
    A_1_post[0, 2] = -0.08  # Weaker monetary transmission at ZLB
    A_1_post[1, 0] = 0.05  # Flatter Phillips curve
    A_1_post[2, 1] = 0.10  # Weaker Taylor rule
    A_1_post[3, 0] = -0.20  # Still Okun's law

    _check_var_stability([A_1_pre], label="monetary_pre_crisis")
    _check_var_stability([A_1_post], label="monetary_post_crisis")

    # Error covariance
    k = 4
    std_devs = np.sqrt([1.5, 1.5, 1.0, 2.0])
    correlations = {
        (0, 1): 0.10,
        (0, 3): -0.25,
        (1, 2): 0.10,
    }
    Sigma = _build_sigma(std_devs, correlations)

    mu_global = np.array([2.0, 2.5, 3.0, 7.5])
    sigma_mu = np.array([1.0, 0.8, 1.2, 2.5])

    burn_in = 50

    records = []
    for c_idx, cname in enumerate(countries):
        mu_i = mu_global + sigma_mu * np.random.randn(k)
        L = np.linalg.cholesky(Sigma)

        total = n_quarters + burn_in
        Y = np.zeros((total + 1, k))
        Y[0] = mu_i

        for t in range(1, total + 1):
            # Determine which regime we are in (relative to post-burn-in time)
            post_burn_t = t - burn_in  # index in the actual data timeline
            if post_burn_t < crisis_idx:
                A = A_1_pre
            else:
                A = A_1_post

            eps = L @ np.random.randn(k)
            Y[t] = mu_i + A @ (Y[t - 1] - mu_i) + eps

            # Crisis shock: negative GDP shock at 2008-Q3
            if post_burn_t == crisis_idx:
                Y[t, 0] -= 3.0 * np.sqrt(Sigma[0, 0])  # ~3 std dev negative shock

        # Extract post-burn-in data
        data = Y[burn_in + 1 : burn_in + 1 + n_quarters]

        # Post-processing
        data[:, 2] = np.maximum(0.0, data[:, 2])  # interest_rate >= 0
        data[:, 3] = np.maximum(0.0, data[:, 3])  # unemployment >= 0

        # Zero lower bound: for first 10 countries, clamp more aggressively post-crisis
        if c_idx < 10:
            for t in range(n_quarters):
                if t >= crisis_idx:
                    data[t, 2] = np.maximum(0.0, data[t, 2] - 0.5)

        for t in range(n_quarters):
            records.append(
                {
                    "country": cname,
                    "quarter": quarters[t],
                    "gdp_growth": data[t, 0],
                    "inflation": data[t, 1],
                    "interest_rate": data[t, 2],
                    "unemployment": data[t, 3],
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Trade Panel  --  Granger causality
# ---------------------------------------------------------------------------


def generate_trade_panel(
    n_countries: int = 40,
    n_years: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate trade panel for Granger causality analysis.

    Variables: exports, imports, gdp, fdi_inflows.

    Parameters
    ----------
    n_countries : number of countries (default 40)
    n_years : number of years (default 50)
    seed : random seed

    Returns
    -------
    DataFrame with columns: country, year, exports, imports, gdp, fdi_inflows
    """
    np.random.seed(seed)

    countries = _COUNTRIES_40[:n_countries]
    years = list(range(1, n_years + 1))

    # A_1: 4x4
    A_1 = np.diag([0.60, 0.55, 0.50, 0.40])
    A_1[2, 0] = 0.15  # exports -> gdp
    A_1[1, 2] = 0.20  # gdp -> imports
    A_1[0, 3] = 0.10  # fdi -> exports

    _check_var_stability([A_1], label="trade_panel")

    std_devs = np.sqrt([4.0, 3.5, 5.0, 6.0])
    correlations = {(0, 1): 0.40}  # export-import correlation
    Sigma = _build_sigma(std_devs, correlations)

    # Country fixed effects with trade-size heterogeneity
    # Larger countries have higher levels
    mu_global = np.array([10.0, 9.0, 50.0, 3.0])
    sigma_mu = np.array([4.0, 3.5, 20.0, 2.0])

    records = []
    for cname in countries:
        mu_i = mu_global + sigma_mu * np.random.randn(4)
        Y = _simulate_var([A_1], Sigma, mu_i, n_years, burn_in=50)

        for t in range(n_years):
            records.append(
                {
                    "country": cname,
                    "year": years[t],
                    "exports": Y[t, 0],
                    "imports": Y[t, 1],
                    "gdp": Y[t, 2],
                    "fdi_inflows": Y[t, 3],
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. PPP Panel  --  cointegrated I(1) variables for VECM
# ---------------------------------------------------------------------------


def generate_ppp_panel(
    n_countries: int = 20,
    n_quarters: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate PPP panel with cointegrated I(1) variables for VECM estimation.

    Variables: exchange_rate, price_domestic, price_foreign (all in logs).
    Cointegrating relation: exchange_rate ~ price_domestic - price_foreign + const.

    Parameters
    ----------
    n_countries : number of countries (default 20)
    n_quarters : number of quarters (default 60)
    seed : random seed

    Returns
    -------
    DataFrame with columns: country, quarter, exchange_rate, price_domestic,
        price_foreign
    """
    np.random.seed(seed)

    countries = _COUNTRIES_20[:n_countries]
    quarters = _quarter_labels(2005, n_quarters)

    records = []
    for cname in countries:
        # Country-specific fixed effects on levels
        fe_domestic = 4.5 + 0.15 * np.random.randn()
        fe_foreign = 4.3 + 0.10 * np.random.randn()
        fe_const = 1.0 + 0.05 * np.random.randn()

        # Generate price indices as random walks with drift
        eps_d = np.random.randn(n_quarters)
        eps_f = np.random.randn(n_quarters)

        price_domestic = np.zeros(n_quarters)
        price_foreign = np.zeros(n_quarters)

        price_domestic[0] = fe_domestic
        price_foreign[0] = fe_foreign

        for t in range(1, n_quarters):
            price_domestic[t] = price_domestic[t - 1] + 0.005 + 0.02 * eps_d[t]
            price_foreign[t] = price_foreign[t - 1] + 0.003 + 0.015 * eps_f[t]

        # Equilibrium exchange rate from PPP
        equilibrium_er = (price_domestic - price_foreign) + fe_const

        # Stationary deviation: AR(1) with rho=0.7
        rho_z = 0.70
        z = np.zeros(n_quarters)
        eps_z = 0.01 * np.random.randn(n_quarters)
        for t in range(1, n_quarters):
            z[t] = rho_z * z[t - 1] + eps_z[t]

        exchange_rate = equilibrium_er + z

        for t in range(n_quarters):
            records.append(
                {
                    "country": cname,
                    "quarter": quarters[t],
                    "exchange_rate": exchange_rate[t],
                    "price_domestic": price_domestic[t],
                    "price_foreign": price_foreign[t],
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Interest Parity Panel  --  cointegrated for VECM
# ---------------------------------------------------------------------------


def generate_interest_parity_panel(
    n_countries: int = 20,
    n_quarters: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate interest parity panel with cointegrated variables for VECM.

    Cointegrating relation: forward_rate - spot_rate ~ (i_domestic - i_foreign)/400.

    Variables: interest_domestic, interest_foreign, forward_rate, spot_rate.

    Parameters
    ----------
    n_countries : number of countries (default 20)
    n_quarters : number of quarters (default 60)
    seed : random seed

    Returns
    -------
    DataFrame with columns: country, quarter, interest_domestic,
        interest_foreign, forward_rate, spot_rate
    """
    np.random.seed(seed)

    countries = _COUNTRIES_20[:n_countries]
    quarters = _quarter_labels(2005, n_quarters)

    records = []
    for cname in countries:
        # Country-specific levels
        i_dom_level = 3.0 + 1.0 * np.random.randn()
        i_for_level = 2.5 + 0.8 * np.random.randn()
        spot_level = np.log(1.0 + 0.3 * np.abs(np.random.randn()))

        # Interest rates as random walks
        eps_id = np.random.randn(n_quarters)
        eps_if = np.random.randn(n_quarters)
        eps_s = np.random.randn(n_quarters)

        interest_domestic = np.zeros(n_quarters)
        interest_foreign = np.zeros(n_quarters)
        spot_rate = np.zeros(n_quarters)

        interest_domestic[0] = i_dom_level
        interest_foreign[0] = i_for_level
        spot_rate[0] = spot_level

        for t in range(1, n_quarters):
            interest_domestic[t] = interest_domestic[t - 1] + 0.05 * eps_id[t]
            interest_foreign[t] = interest_foreign[t - 1] + 0.04 * eps_if[t]
            spot_rate[t] = spot_rate[t - 1] + 0.005 + 0.02 * eps_s[t]

        # Forward rate from covered interest parity + AR(1) deviation
        rho_dev = 0.65
        deviation = np.zeros(n_quarters)
        eps_dev = 0.005 * np.random.randn(n_quarters)
        for t in range(1, n_quarters):
            deviation[t] = rho_dev * deviation[t - 1] + eps_dev[t]

        forward_rate = spot_rate + (interest_domestic - interest_foreign) / 400.0 + deviation

        for t in range(n_quarters):
            records.append(
                {
                    "country": cname,
                    "quarter": quarters[t],
                    "interest_domestic": interest_domestic[t],
                    "interest_foreign": interest_foreign[t],
                    "forward_rate": forward_rate[t],
                    "spot_rate": spot_rate[t],
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 8. Dynamic Panel  --  large N, small T for GMM
# ---------------------------------------------------------------------------


def generate_dynamic_panel(
    n_countries: int = 100,
    n_years: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate dynamic panel data suitable for GMM estimation (large N, small T).

    Variables: y1, y2, y3.

    Parameters
    ----------
    n_countries : number of countries (default 100)
    n_years : number of years (default 15)
    seed : random seed

    Returns
    -------
    DataFrame with columns: country, year, y1, y2, y3
    """
    np.random.seed(seed)

    countries = [f"Country_{i:03d}" for i in range(1, n_countries + 1)]
    years = list(range(1, n_years + 1))

    k = 3

    # A_1: 3x3
    A_1 = np.diag([0.50, 0.40, 0.35])
    A_1[1, 0] = 0.15  # y1 -> y2
    A_1[2, 1] = 0.10  # y2 -> y3

    _check_var_stability([A_1], label="dynamic_panel")

    # Error covariance with moderate cross-correlations
    std_devs = np.sqrt([1.0, 0.8, 1.2])
    correlations = {
        (0, 1): 0.20,
        (0, 2): 0.10,
        (1, 2): 0.15,
    }
    Sigma = _build_sigma(std_devs, correlations)
    L = np.linalg.cholesky(Sigma)

    burn_in = 50

    records = []
    for cname in countries:
        # Initial values (used to generate correlated fixed effects)
        y0 = np.random.randn(k) * 2.0

        # Fixed effects correlated with initial values: alpha_i = 0.3 * y_i0 + eta_i
        eta_i = np.random.randn(k) * 0.5
        alpha_i = 0.3 * y0 + eta_i

        total = n_years + burn_in
        Y = np.zeros((total + 1, k))
        Y[0] = y0

        for t in range(1, total + 1):
            eps = L @ np.random.randn(k)
            Y[t] = alpha_i + A_1 @ (Y[t - 1] - alpha_i) + eps

        # Extract post-burn-in data
        data = Y[burn_in + 1 : burn_in + 1 + n_years]

        for t in range(n_years):
            records.append(
                {
                    "country": cname,
                    "year": years[t],
                    "y1": data[t, 0],
                    "y2": data[t, 1],
                    "y3": data[t, 2],
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main block for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing data generators...")

    df1 = generate_macro_panel(n_countries=5, n_quarters=10)
    print(f"  macro_panel: {df1.shape}, cols={list(df1.columns)}")
    assert df1.shape == (50, 7), f"Expected (50, 7), got {df1.shape}"

    df2 = generate_energy_panel(n_countries=5, n_quarters=10)
    print(f"  energy_panel: {df2.shape}, cols={list(df2.columns)}")
    assert df2.shape == (50, 5), f"Expected (50, 5), got {df2.shape}"

    df3 = generate_finance_panel(n_countries=5, n_periods=10)
    print(f"  finance_panel: {df3.shape}, cols={list(df3.columns)}")
    assert df3.shape == (50, 6), f"Expected (50, 6), got {df3.shape}"

    df4 = generate_monetary_policy_panel(n_countries=5, n_quarters=10)
    print(f"  monetary_policy_panel: {df4.shape}, cols={list(df4.columns)}")
    assert df4.shape == (50, 6), f"Expected (50, 6), got {df4.shape}"

    df5 = generate_trade_panel(n_countries=5, n_years=10)
    print(f"  trade_panel: {df5.shape}, cols={list(df5.columns)}")
    assert df5.shape == (50, 6), f"Expected (50, 6), got {df5.shape}"

    df6 = generate_ppp_panel(n_countries=5, n_quarters=10)
    print(f"  ppp_panel: {df6.shape}, cols={list(df6.columns)}")
    assert df6.shape == (50, 5), f"Expected (50, 5), got {df6.shape}"

    df7 = generate_interest_parity_panel(n_countries=5, n_quarters=10)
    print(f"  interest_parity_panel: {df7.shape}, cols={list(df7.columns)}")
    assert df7.shape == (50, 6), f"Expected (50, 6), got {df7.shape}"

    df8 = generate_dynamic_panel(n_countries=10, n_years=5)
    print(f"  dynamic_panel: {df8.shape}, cols={list(df8.columns)}")
    assert df8.shape == (50, 5), f"Expected (50, 5), got {df8.shape}"

    print("\nAll data generators working correctly!")
