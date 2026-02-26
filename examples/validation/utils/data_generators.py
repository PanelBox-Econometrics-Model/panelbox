"""
data_generators.py — Synthetic dataset generators for the validation tutorial series.

Each generator returns a pandas DataFrame ready to be saved as CSV.  All functions
accept a ``random_state`` argument (default 42) so results are fully reproducible.

Usage
-----
Run this module as a script to write all datasets to the ``data/`` directory::

    python utils/data_generators.py

Or import individual generators::

    from utils.data_generators import generate_firmdata, load_dataset
"""

from __future__ import annotations

import os
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def _rng(random_state: int) -> np.random.Generator:
    return np.random.default_rng(random_state)


# ---------------------------------------------------------------------------
# 1. firmdata.csv — groupwise heteroskedasticity
# ---------------------------------------------------------------------------


def generate_firmdata(
    n_firms: int = 100,
    n_years: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate balanced firm-level panel with groupwise heteroskedasticity.

    Output columns
    --------------
    firm_id, year, sales, capital, labor, size_category

    Notes
    -----
    Residual variance is scaled by size_category:
    small  → σ = 1.0
    medium → σ = 2.0
    large  → σ = 4.0
    """
    rng = _rng(random_state)

    # Assign firms to size categories
    categories = np.repeat(
        ["small", "medium", "large"], [n_firms // 3, n_firms // 3, n_firms - 2 * (n_firms // 3)]
    )
    rng.shuffle(categories)
    size_map = {"small": 1.0, "medium": 2.0, "large": 4.0}

    rows = []
    for i, firm in enumerate(range(1, n_firms + 1)):
        cat = categories[i]
        sigma = size_map[cat]
        fe = rng.normal(0, 1)  # firm fixed effect
        capital_base = rng.uniform(5, 50)
        labor_base = rng.uniform(10, 200)

        for t, year in enumerate(range(2010, 2010 + n_years)):
            capital = capital_base + rng.normal(0, 2)
            labor = labor_base + rng.normal(0, 5)
            eps = rng.normal(0, sigma)
            sales = 10 + 0.3 * capital + 0.05 * labor + fe + eps
            rows.append(
                {
                    "firm_id": firm,
                    "year": year,
                    "sales": round(max(sales, 0), 4),
                    "capital": round(max(capital, 0.1), 4),
                    "labor": round(max(labor, 0.1), 4),
                    "size_category": cat,
                }
            )

    df = pd.DataFrame(rows)
    assert df.shape == (n_firms * n_years, 6), f"Unexpected shape {df.shape}"
    return df


# ---------------------------------------------------------------------------
# 2. macro_panel.csv — cross-sectional dependence + AR(1) errors
# ---------------------------------------------------------------------------


def generate_macro_panel(
    n_countries: int = 30,
    n_years: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate macro-level country panel with cross-sectional dependence and
    AR(1) serial correlation (ρ ≈ 0.6).  Common shock in 2008-2009.

    Output columns
    --------------
    country, year, gdp_growth, investment, trade_openness, region
    """
    rng = _rng(random_state)
    rho = 0.6
    start_year = 2000

    regions = ["Europe", "Asia", "Americas", "Africa", "Oceania"]
    country_regions = [regions[i % len(regions)] for i in range(n_countries)]

    # Common factor (cross-sectional dependence)
    common_factor = rng.normal(0, 1, n_years)
    common_factor[8] -= 3  # 2008 crisis
    common_factor[9] -= 2  # 2009 aftermath

    rows = []
    for c in range(n_countries):
        country = f"C{c + 1:03d}"
        region = country_regions[c]
        lambda_c = rng.uniform(0.5, 1.5)  # factor loading
        inv_base = rng.uniform(15, 35)
        trade_base = rng.uniform(20, 80)

        eps_prev = 0.0
        for t in range(n_years):
            year = start_year + t
            eps_idio = rng.normal(0, 0.5)
            eps = rho * eps_prev + eps_idio
            eps_prev = eps
            gdp_growth = 2.5 + lambda_c * common_factor[t] + eps + rng.normal(0, 0.3)
            investment = inv_base + rng.normal(0, 2)
            trade_openness = trade_base + rng.normal(0, 5)
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "gdp_growth": round(gdp_growth, 4),
                    "investment": round(max(investment, 0), 4),
                    "trade_openness": round(max(trade_openness, 0), 4),
                    "region": region,
                }
            )

    df = pd.DataFrame(rows)
    assert df.shape == (n_countries * n_years, 6), f"Unexpected shape {df.shape}"
    return df


# ---------------------------------------------------------------------------
# 3. small_panel.csv — i.i.d. errors, small-sample bootstrap demo
# ---------------------------------------------------------------------------


def generate_small_panel(
    n_entities: int = 20,
    n_periods: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Small balanced panel with i.i.d. N(0,1) errors.

    Output columns
    --------------
    entity, time, y, x1, x2
    """
    rng = _rng(random_state)
    rows = []
    for e in range(1, n_entities + 1):
        fe = rng.normal(0, 1)
        for t in range(1, n_periods + 1):
            x1 = rng.normal(0, 1)
            x2 = rng.normal(0, 1)
            eps = rng.normal(0, 1)
            y = 1.0 + 0.5 * x1 - 0.3 * x2 + fe + eps
            rows.append(
                {"entity": e, "time": t, "y": round(y, 4), "x1": round(x1, 4), "x2": round(x2, 4)}
            )

    df = pd.DataFrame(rows)
    assert df.shape == (n_entities * n_periods, 5), f"Unexpected shape {df.shape}"
    return df


# ---------------------------------------------------------------------------
# 4. sales_panel.csv — seasonal forecasting
# ---------------------------------------------------------------------------


def generate_sales_panel(
    n_firms: int = 50,
    n_quarters: int = 24,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Quarterly firm-level panel with seasonal component (sin/cos terms).

    Output columns
    --------------
    firm, quarter, sales, advertising, price, seasonality
    """
    rng = _rng(random_state)
    rows = []
    for f in range(1, n_firms + 1):
        fe = rng.normal(0, 2)
        adv_base = rng.uniform(1, 10)
        price_base = rng.uniform(50, 200)
        for q in range(1, n_quarters + 1):
            # Seasonal component
            season = 2.0 * np.sin(2 * np.pi * q / 4) + 1.0 * np.cos(2 * np.pi * q / 4)
            advertising = max(adv_base + rng.normal(0, 0.5), 0)
            price = max(price_base + rng.normal(0, 5), 1)
            eps = rng.normal(0, 1)
            sales = 20 + 0.8 * advertising - 0.1 * price + 2.0 * season + fe + eps
            rows.append(
                {
                    "firm": f,
                    "quarter": q,
                    "sales": round(max(sales, 0), 4),
                    "advertising": round(advertising, 4),
                    "price": round(price, 4),
                    "seasonality": round(season, 4),
                }
            )

    df = pd.DataFrame(rows)
    assert df.shape == (n_firms * n_quarters, 6), f"Unexpected shape {df.shape}"
    return df


# ---------------------------------------------------------------------------
# 5. macro_ts_panel.csv — structural break in 2008
# ---------------------------------------------------------------------------


def generate_macro_ts_panel(
    n_countries: int = 15,
    n_years: int = 40,
    break_year: int = 2008,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Long macro panel (1980–2019) with a structural break at ``break_year``.

    Output columns
    --------------
    country, year, inflation, policy_rate, unemployment
    """
    rng = _rng(random_state)
    start_year = 1980
    rows = []

    for c in range(1, n_countries + 1):
        country = f"CTR{c:02d}"
        # Pre-break and post-break intercepts differ
        infl_pre = rng.uniform(2, 8)
        infl_post = infl_pre - rng.uniform(0, 3)  # disinflation after break
        policy_base = rng.uniform(1, 6)
        unemp_base = rng.uniform(4, 12)
        eps_prev = 0.0

        for i, year in enumerate(range(start_year, start_year + n_years)):
            post = year >= break_year
            eps = 0.5 * eps_prev + rng.normal(0, 0.5)
            eps_prev = eps
            inflation = (infl_post if post else infl_pre) + eps + rng.normal(0, 0.5)
            policy_rate = max(policy_base + rng.normal(0, 0.5), 0)
            unemployment = max(unemp_base + rng.normal(0, 1), 0)
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "inflation": round(inflation, 4),
                    "policy_rate": round(policy_rate, 4),
                    "unemployment": round(unemployment, 4),
                }
            )

    df = pd.DataFrame(rows)
    assert df.shape == (n_countries * n_years, 5), f"Unexpected shape {df.shape}"
    return df


# ---------------------------------------------------------------------------
# 6. panel_with_outliers.csv — injected outliers
# ---------------------------------------------------------------------------


def generate_panel_with_outliers(
    n_firms: int = 80,
    n_years: int = 8,
    outlier_fraction: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Firm panel with ~5% artificially injected outliers.

    Output columns
    --------------
    firm, year, profit, revenue, costs, outlier_flag

    outlier_flag
    ------------
    0 = normal
    1 = artificial outlier (injected)
    2 = genuine extreme (natural tail)
    """
    rng = _rng(random_state)
    total = n_firms * n_years
    n_outliers = int(total * outlier_fraction)
    outlier_indices = set(rng.choice(total, size=n_outliers, replace=False).tolist())

    rows = []
    idx = 0
    for f in range(1, n_firms + 1):
        fe = rng.normal(0, 2)
        rev_base = rng.uniform(100, 500)
        for y in range(2015, 2015 + n_years):
            revenue = max(rev_base + rng.normal(0, 20), 0)
            costs = max(0.6 * revenue + rng.normal(0, 10), 0)
            eps = rng.normal(0, 5)
            profit = revenue - costs + fe + eps

            flag = 0
            if idx in outlier_indices:
                profit += rng.choice([-1, 1]) * rng.uniform(50, 200)
                flag = 1
            elif abs(eps) > 3 * 5:  # beyond 3σ naturally
                flag = 2

            rows.append(
                {
                    "firm": f,
                    "year": y,
                    "profit": round(profit, 4),
                    "revenue": round(revenue, 4),
                    "costs": round(costs, 4),
                    "outlier_flag": flag,
                }
            )
            idx += 1

    df = pd.DataFrame(rows)
    assert df.shape == (n_firms * n_years, 6), f"Unexpected shape {df.shape}"
    return df


# ---------------------------------------------------------------------------
# 7. real_firms.csv — natural heterogeneity, no artificial outliers
# ---------------------------------------------------------------------------


def generate_real_firms(
    n_firms: int = 120,
    n_years: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Firm panel mimicking real-world financial data with natural heterogeneity.

    Output columns
    --------------
    firm, year, roe, leverage, firm_age, industry
    """
    rng = _rng(random_state)
    industries = ["Manufacturing", "Finance", "Technology", "Retail", "Energy"]
    firm_industries = [industries[i % len(industries)] for i in range(n_firms)]
    rng.shuffle(firm_industries)

    rows = []
    for f in range(1, n_firms + 1):
        industry = firm_industries[f - 1]
        age_base = rng.integers(2, 50)
        leverage_base = rng.uniform(0.1, 0.8)
        roe_base = rng.normal(0.1, 0.05)

        for y in range(2018, 2018 + n_years):
            age = age_base + (y - 2018)
            leverage = min(max(leverage_base + rng.normal(0, 0.05), 0), 1)
            roe = roe_base - 0.2 * leverage + rng.normal(0, 0.03)
            rows.append(
                {
                    "firm": f,
                    "year": y,
                    "roe": round(roe, 4),
                    "leverage": round(leverage, 4),
                    "firm_age": int(age),
                    "industry": industry,
                }
            )

    df = pd.DataFrame(rows)
    assert df.shape == (n_firms * n_years, 6), f"Unexpected shape {df.shape}"
    return df


# ---------------------------------------------------------------------------
# 8. panel_comprehensive.csv — rich variable set for robustness checks
# ---------------------------------------------------------------------------


def generate_panel_comprehensive(
    n_entities: int = 100,
    n_periods: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Comprehensive balanced panel with many variables for robustness checks.

    Output columns
    --------------
    entity, time, y, x1, x2, x3, x4, x5, z1, z2, industry, region
    """
    rng = _rng(random_state)
    industries = ["A", "B", "C", "D"]
    regions = ["North", "South", "East", "West"]
    entity_industries = [industries[i % 4] for i in range(n_entities)]
    entity_regions = [regions[i % 4] for i in range(n_entities)]
    rng.shuffle(entity_industries)
    rng.shuffle(entity_regions)

    rows = []
    for e in range(1, n_entities + 1):
        fe = rng.normal(0, 1)
        industry = entity_industries[e - 1]
        region = entity_regions[e - 1]
        for t in range(1, n_periods + 1):
            x1 = rng.normal(0, 1)
            x2 = rng.normal(0, 1)
            x3 = rng.normal(1, 2)
            x4 = rng.uniform(0, 10)
            x5 = rng.normal(0, 0.5)
            z1 = x1 + rng.normal(0, 0.3)  # instrument correlated with x1
            z2 = rng.normal(0, 1)  # excluded instrument
            eps = rng.normal(0, 1)
            y = 2.0 + 0.4 * x1 - 0.2 * x2 + 0.1 * x3 + 0.05 * x4 + 0.3 * x5 + fe + eps
            rows.append(
                {
                    "entity": e,
                    "time": t,
                    "y": round(y, 4),
                    "x1": round(x1, 4),
                    "x2": round(x2, 4),
                    "x3": round(x3, 4),
                    "x4": round(x4, 4),
                    "x5": round(x5, 4),
                    "z1": round(z1, 4),
                    "z2": round(z2, 4),
                    "industry": industry,
                    "region": region,
                }
            )

    df = pd.DataFrame(rows)
    assert df.shape == (n_entities * n_periods, 12), f"Unexpected shape {df.shape}"
    return df


# ---------------------------------------------------------------------------
# 9. panel_unbalanced.csv — random attrition (truly missing rows)
# ---------------------------------------------------------------------------


def generate_panel_unbalanced(
    n_entities: int = 150,
    max_periods: int = 10,
    attrition_rate: float = 0.30,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Unbalanced panel created by random attrition (rows are dropped, not NaN-filled).

    Output columns
    --------------
    entity, time, outcome, treatment, covariate1, covariate2

    Expected row count ≈ n_entities × max_periods × (1 − attrition_rate)
    """
    rng = _rng(random_state)
    rows = []
    for e in range(1, n_entities + 1):
        fe = rng.normal(0, 1)
        treat_base = rng.choice([0, 1])
        for t in range(1, max_periods + 1):
            if rng.random() < attrition_rate:
                continue  # genuinely missing observation
            cov1 = rng.normal(0, 1)
            cov2 = rng.uniform(0, 5)
            treatment = 1 if (treat_base == 1 and t > max_periods // 2) else 0
            eps = rng.normal(0, 1)
            outcome = 1.5 + 0.6 * treatment + 0.3 * cov1 + 0.1 * cov2 + fe + eps
            rows.append(
                {
                    "entity": e,
                    "time": t,
                    "outcome": round(outcome, 4),
                    "treatment": int(treatment),
                    "covariate1": round(cov1, 4),
                    "covariate2": round(cov2, 4),
                }
            )

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a dataset by name from the ``data/`` directory.

    Parameters
    ----------
    name : str
        One of: firmdata, macro_panel, small_panel, sales_panel,
        macro_ts_panel, panel_with_outliers, real_firms,
        panel_comprehensive, panel_unbalanced.

    Returns
    -------
    pd.DataFrame
    """
    csv_path = _DATA_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {csv_path}. "
            "Run 'python utils/data_generators.py' to generate all datasets."
        )
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Script entry-point — generate and save all CSVs
# ---------------------------------------------------------------------------

_GENERATORS = {
    "firmdata": lambda: generate_firmdata(100, 10),
    "macro_panel": lambda: generate_macro_panel(30, 20),
    "small_panel": lambda: generate_small_panel(20, 10),
    "sales_panel": lambda: generate_sales_panel(50, 24),
    "macro_ts_panel": lambda: generate_macro_ts_panel(15, 40),
    "panel_with_outliers": lambda: generate_panel_with_outliers(80, 8),
    "real_firms": lambda: generate_real_firms(120, 5),
    "panel_comprehensive": lambda: generate_panel_comprehensive(100, 12),
    "panel_unbalanced": lambda: generate_panel_unbalanced(150, 10),
}


def generate_all(data_dir: str | None = None) -> None:
    """Generate all datasets and write them to *data_dir* (default: data/)."""
    out = pathlib.Path(data_dir) if data_dir else _DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, gen_fn in _GENERATORS.items():
        df = gen_fn()
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  Wrote {path.name}  ({len(df):,} rows × {df.shape[1]} cols)")


if __name__ == "__main__":
    print("Generating validation datasets...")
    generate_all()
    print("Done.")
