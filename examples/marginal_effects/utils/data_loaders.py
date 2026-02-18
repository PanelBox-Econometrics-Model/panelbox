"""
data_loaders.py
---------------
Functions to load datasets used in the Marginal Effects Tutorial Series.

Each loader first tries to read the corresponding CSV from the `data/`
directory. If the file is not found, a synthetic dataset with the same
structure is generated so that notebooks can run without external downloads.

Supported dataset names
-----------------------
- 'mroz'           : Mroz (1987) labor force participation
- 'mroz_hours'     : Extended Mroz with censored hours worked (for Tobit)
- 'patents'        : Firm-level patent counts and R&D expenditure
- 'doctor_visits'  : German health data with doctor-visit counts
- 'job_satisfaction': Ordered satisfaction scale 1-5
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(name: str, seed: int = 42) -> pd.DataFrame:
    """Load a dataset by name.

    Tries to read ``<DATA_DIR>/<name>.csv`` first. Falls back to a synthetic
    generator if the CSV file is absent.

    Parameters
    ----------
    name : str
        One of ``'mroz'``, ``'mroz_hours'``, ``'patents'``,
        ``'doctor_visits'``, ``'job_satisfaction'``.
    seed : int, optional
        Random seed used when generating synthetic data (default 42).

    Returns
    -------
    pd.DataFrame
    """
    csv_path = DATA_DIR / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    generators = {
        "mroz": _generate_mroz,
        "mroz_hours": _generate_mroz_hours,
        "patents": _generate_patents,
        "doctor_visits": _generate_doctor_visits,
        "job_satisfaction": _generate_job_satisfaction,
    }

    if name not in generators:
        raise ValueError(f"Unknown dataset '{name}'. " f"Valid options: {sorted(generators)}")

    rng = np.random.default_rng(seed)
    df = generators[name](rng)
    return df


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def _generate_mroz(rng: np.random.Generator, n: int = 753) -> pd.DataFrame:
    """Mroz (1987): women's labor force participation.

    Binary outcome  : ``inlf``  (1 = in labor force)
    Continuous vars : ``age``, ``educ``, ``exper``, ``nwifeinc``,
                      ``kidslt6``, ``kidsge6``
    """
    age = rng.integers(30, 60, size=n).astype(float)
    educ = rng.integers(6, 17, size=n).astype(float)
    exper = np.clip(age - educ - 6 + rng.normal(0, 2, n), 0, None)
    nwifeinc = np.exp(rng.normal(3.0, 0.8, n))
    kidslt6 = rng.integers(0, 4, size=n).astype(float)
    kidsge6 = rng.integers(0, 4, size=n).astype(float)

    xb = -1.5 + 0.06 * educ + 0.03 * exper - 0.02 * age - 0.8 * kidslt6 - 0.03 * nwifeinc
    prob = 1 / (1 + np.exp(-xb))
    inlf = rng.binomial(1, prob).astype(float)

    hours = np.where(inlf == 1, np.clip(rng.normal(1300, 600, n), 0, None), 0.0)
    wage = np.where(
        inlf == 1,
        np.exp(0.5 + 0.08 * educ + 0.02 * exper + rng.normal(0, 0.4, n)),
        np.nan,
    )

    return pd.DataFrame(
        {
            "inlf": inlf,
            "hours": hours,
            "age": age,
            "educ": educ,
            "exper": exper,
            "expersq": exper**2,
            "nwifeinc": nwifeinc,
            "kidslt6": kidslt6,
            "kidsge6": kidsge6,
            "wage": wage,
        }
    )


def _generate_mroz_hours(rng: np.random.Generator, n: int = 753) -> pd.DataFrame:
    """Extended Mroz: hours worked (censored at 0) — for Tobit models."""
    base = _generate_mroz(rng, n)
    # hours is already censored at 0 in the base dataset; add a few extras
    base["lwage"] = np.where(
        base["wage"] > 0,
        np.log(base["wage"].clip(lower=0.01)),
        np.nan,
    )
    return base


def _generate_patents(rng: np.random.Generator, n: int = 346) -> pd.DataFrame:
    """Firm-level patent counts and R&D.

    Count outcome : ``patents``
    Regressors    : ``log_rnd``, ``log_sales``, ``log_capital``,
                    ``industry``, ``year``
    """
    log_rnd = rng.normal(3.0, 1.2, n)
    log_sales = log_rnd + rng.normal(1.5, 0.5, n)
    log_capital = log_sales - rng.normal(0.5, 0.3, n)
    industry = rng.integers(1, 6, size=n)
    year = rng.integers(1975, 1980, size=n)

    log_mu = -1.0 + 0.5 * log_rnd + 0.2 * log_sales + rng.normal(0, 0.3, n)
    mu = np.exp(log_mu)
    patents = rng.poisson(mu).astype(float)

    return pd.DataFrame(
        {
            "patents": patents,
            "log_rnd": log_rnd,
            "log_sales": log_sales,
            "log_capital": log_capital,
            "industry": industry,
            "year": year,
        }
    )


def _generate_doctor_visits(rng: np.random.Generator, n: int = 5190) -> pd.DataFrame:
    """German health data: number of doctor visits.

    Count outcome : ``docvis``
    Regressors    : ``age``, ``female``, ``educ``, ``hhninc``,
                    ``public``, ``addon``
    """
    age = rng.integers(20, 65, size=n).astype(float)
    female = rng.binomial(1, 0.52, n).astype(float)
    educ = rng.integers(7, 18, size=n).astype(float)
    hhninc = np.exp(rng.normal(3.0, 0.5, n))
    public = rng.binomial(1, 0.85, n).astype(float)
    addon = rng.binomial(1, 0.04, n).astype(float)

    log_mu = (
        0.5
        + 0.02 * age
        + 0.3 * female
        - 0.05 * educ
        - 0.1 * np.log(hhninc)
        + 0.4 * public
        + 0.2 * addon
        + rng.normal(0, 0.5, n)
    )
    mu = np.exp(log_mu)
    docvis = rng.poisson(mu).astype(float)

    return pd.DataFrame(
        {
            "docvis": docvis,
            "age": age,
            "female": female,
            "educ": educ,
            "hhninc": hhninc,
            "public": public,
            "addon": addon,
        }
    )


def _generate_job_satisfaction(rng: np.random.Generator, n: int = 1000) -> pd.DataFrame:
    """Ordered satisfaction scale 1–5.

    Ordered outcome : ``satisfaction`` (1 = very dissatisfied, 5 = very satisfied)
    Regressors      : ``age``, ``female``, ``educ``, ``tenure``, ``log_wage``
    """
    age = rng.integers(20, 60, size=n).astype(float)
    female = rng.binomial(1, 0.5, n).astype(float)
    educ = rng.integers(8, 18, size=n).astype(float)
    tenure = np.clip(rng.exponential(5, n), 0, 30)
    log_wage = 1.5 + 0.05 * educ + 0.01 * tenure + rng.normal(0, 0.3, n)

    latent = (
        -2.0
        + 0.02 * educ
        + 0.03 * tenure
        + 0.5 * log_wage
        - 0.005 * age
        + 0.1 * female
        + rng.normal(0, 1, n)
    )

    # Cut-points for 5 categories
    cuts = [-1.5, -0.5, 0.5, 1.5]
    satisfaction = np.ones(n, dtype=int)
    for k, c in enumerate(cuts, start=2):
        satisfaction = np.where(latent >= c, k, satisfaction)

    return pd.DataFrame(
        {
            "satisfaction": satisfaction.astype(float),
            "age": age,
            "female": female,
            "educ": educ,
            "tenure": tenure,
            "log_wage": log_wage,
        }
    )
