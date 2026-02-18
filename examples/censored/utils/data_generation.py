"""
Data generation functions for censored and selection models tutorials.

Each function generates a synthetic dataset with realistic properties
for demonstrating Tobit and Heckman selection models.
"""

import numpy as np
import pandas as pd


def generate_labor_supply(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate cross-sectional labor supply data with left-censoring at zero.

    The data generating process follows a standard labor supply model where
    the latent desired hours are a function of wages, education, experience,
    and household characteristics. Observed hours are censored at zero for
    non-participants.

    Parameters
    ----------
    n : int, default 500
        Number of observations.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: hours, wage, education, experience,
        experience_sq, age, children, married, non_labor_income.
    """
    rng = np.random.default_rng(seed)

    education = rng.integers(8, 21, size=n)
    age = rng.integers(25, 60, size=n)
    experience = np.clip(age - education - 6 + rng.normal(0, 2, n), 0, None)
    experience_sq = experience**2
    children = rng.poisson(0.8, size=n)
    married = rng.binomial(1, 0.6, size=n)
    non_labor_income = np.abs(rng.normal(20, 15, n))
    wage = np.exp(
        0.8 + 0.07 * education + 0.03 * experience - 0.0005 * experience_sq + rng.normal(0, 0.4, n)
    )

    # Latent hours: positive relationship with wage, negative with non-labor income
    latent_hours = (
        10
        + 3.0 * np.log(wage)
        + 0.5 * education
        + 0.8 * experience
        - 0.01 * experience_sq
        - 2.5 * children
        - 0.1 * non_labor_income
        + rng.normal(0, 8, n)
    )

    # Censor at zero
    hours = np.maximum(latent_hours, 0.0)

    return pd.DataFrame(
        {
            "hours": np.round(hours, 1),
            "wage": np.round(wage, 2),
            "education": education,
            "experience": np.round(experience, 1),
            "experience_sq": np.round(experience_sq, 1),
            "age": age,
            "children": children,
            "married": married,
            "non_labor_income": np.round(non_labor_income, 2),
        }
    )


def generate_health_panel(n: int = 500, t: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Generate panel data on health expenditures with left-censoring.

    Individual random effects drive both health expenditure levels and
    the probability of any expenditure. Health expenditures are left-censored
    at zero (individuals who do not seek care).

    Parameters
    ----------
    n : int, default 500
        Number of individuals.
    t : int, default 4
        Number of time periods.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel DataFrame with columns: id, time, expenditure, income, age,
        chronic, insurance, female, bmi.
    """
    rng = np.random.default_rng(seed)
    N = n * t

    ids = np.repeat(np.arange(1, n + 1), t)
    times = np.tile(np.arange(1, t + 1), n)

    # Time-invariant characteristics
    female = np.repeat(rng.binomial(1, 0.52, size=n), t)
    base_age = np.repeat(rng.integers(20, 70, size=n), t)
    age = base_age + np.tile(np.arange(t), n)

    # Individual random effect
    alpha_i = np.repeat(rng.normal(0, 2.5, size=n), t)

    # Time-varying characteristics
    chronic = rng.poisson(0.5 + 0.02 * age, size=N)
    insurance = rng.binomial(1, 0.7, size=N)
    income = np.abs(rng.normal(40, 20, N) + 0.5 * (age - 40))
    bmi = np.clip(rng.normal(25, 5, N), 15, 45)

    # Latent expenditure
    latent_exp = (
        alpha_i
        - 5.0
        + 0.02 * income
        + 0.05 * age
        + 3.0 * chronic
        + 2.0 * insurance
        + 0.5 * female
        + 0.1 * (bmi - 25)
        + rng.normal(0, 4, N)
    )

    expenditure = np.maximum(latent_exp, 0.0)

    return pd.DataFrame(
        {
            "id": ids,
            "time": times,
            "expenditure": np.round(expenditure, 2),
            "income": np.round(income, 2),
            "age": age,
            "chronic": chronic,
            "insurance": insurance,
            "female": female,
            "bmi": np.round(bmi, 1),
        }
    )


def generate_consumer_durables(n: int = 200, t: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Generate panel data on consumer durable goods spending with left-censoring.

    Durable goods purchases are lumpy: many households spend zero in a given
    period. The data generating process includes individual fixed effects,
    making it suitable for the Honoré trimmed LAD estimator.

    Parameters
    ----------
    n : int, default 200
        Number of households.
    t : int, default 5
        Number of time periods.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Panel DataFrame with columns: id, time, spending, income, wealth,
        household_size, homeowner, urban, credit_score.
    """
    rng = np.random.default_rng(seed)
    N = n * t

    ids = np.repeat(np.arange(1, n + 1), t)
    times = np.tile(np.arange(1, t + 1), n)

    # Time-invariant
    homeowner = np.repeat(rng.binomial(1, 0.65, size=n), t)
    urban = np.repeat(rng.binomial(1, 0.7, size=n), t)

    # Individual fixed effect (substantial heterogeneity)
    alpha_i = np.repeat(rng.normal(0, 4.0, size=n), t)

    # Time-varying
    income = np.abs(rng.normal(50, 25, N) + 2.0 * np.tile(np.arange(t), n))
    wealth = np.abs(rng.normal(100, 60, N) + 5.0 * np.tile(np.arange(t), n))
    household_size = rng.integers(1, 6, size=N)
    credit_score = rng.normal(0, 1, N)

    # Latent spending (large variance -> many zeros)
    latent_spending = (
        alpha_i
        - 8.0
        + 0.05 * income
        + 0.02 * wealth
        + 1.5 * household_size
        + 3.0 * homeowner
        + 1.0 * urban
        + 2.0 * credit_score
        + rng.normal(0, 6, N)
    )

    spending = np.maximum(latent_spending, 0.0)

    return pd.DataFrame(
        {
            "id": ids,
            "time": times,
            "spending": np.round(spending, 2),
            "income": np.round(income, 2),
            "wealth": np.round(wealth, 2),
            "household_size": household_size,
            "homeowner": homeowner,
            "urban": urban,
            "credit_score": np.round(credit_score, 3),
        }
    )


def generate_mroz_data(n: int = 753, seed: int = 42) -> pd.DataFrame:
    """
    Generate simulated data modeled after Mroz (1987) labor force participation.

    Women decide whether to participate in the labor force (selection equation),
    and wages are observed only for participants (outcome equation). The husband's
    income and number of young children serve as exclusion restrictions.

    Parameters
    ----------
    n : int, default 753
        Number of observations.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: lfp, hours, wage, education, experience,
        experience_sq, age, children_lt6, children_6_18, husband_income.
    """
    rng = np.random.default_rng(seed)

    education = rng.integers(8, 18, size=n)
    age = rng.integers(25, 55, size=n)
    experience = np.clip(age - education - 6 + rng.normal(0, 3, n), 0, None)
    experience_sq = experience**2
    children_lt6 = rng.poisson(0.4, size=n)
    children_6_18 = rng.poisson(0.8, size=n)
    husband_income = np.abs(rng.normal(30, 15, n))

    # Selection equation: labor force participation
    z_star = (
        -1.5
        + 0.12 * education
        + 0.04 * experience
        - 0.001 * experience_sq
        - 0.02 * age
        - 0.8 * children_lt6
        - 0.15 * children_6_18
        - 0.02 * husband_income
        + rng.normal(0, 1, n)
    )
    lfp = (z_star > 0).astype(int)

    # Outcome equation: log wages (with correlation to selection)
    u = rng.normal(0, 1, n)
    rho = 0.5  # correlation between selection and outcome errors
    eps = rho * (z_star - z_star.mean()) / z_star.std() + np.sqrt(1 - rho**2) * u

    log_wage = 0.5 + 0.08 * education + 0.04 * experience - 0.0008 * experience_sq + 0.3 * eps
    wage = np.exp(log_wage)

    # Wages observed only for participants
    wage = np.where(lfp == 1, np.round(wage, 2), np.nan)

    # Hours (observed only for participants)
    hours = np.where(
        lfp == 1,
        np.round(np.clip(rng.normal(1600, 400, n), 100, 3500), 0),
        0.0,
    )

    return pd.DataFrame(
        {
            "lfp": lfp,
            "hours": hours,
            "wage": wage,
            "education": education,
            "experience": np.round(experience, 1),
            "experience_sq": np.round(experience_sq, 1),
            "age": age,
            "children_lt6": children_lt6,
            "children_6_18": children_6_18,
            "husband_income": np.round(husband_income, 2),
        }
    )


def generate_college_wage(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Generate data on college attendance decisions and post-college wages.

    Individuals decide whether to attend college (selection), and wages
    are observed only for college graduates. Distance to nearest college
    and local tuition serve as exclusion restrictions (instruments that
    affect college attendance but not wages directly).

    Parameters
    ----------
    n : int, default 600
        Number of observations.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: college, wage, ability, parent_education,
        family_income, distance_college, tuition, urban, female.
    """
    rng = np.random.default_rng(seed)

    ability = rng.normal(0, 1, n)
    parent_education = rng.integers(8, 20, size=n)
    family_income = np.abs(rng.normal(50, 25, n))
    distance_college = np.abs(rng.normal(20, 15, n))
    tuition = np.abs(rng.normal(8, 4, n))
    urban = rng.binomial(1, 0.6, size=n)
    female = rng.binomial(1, 0.5, size=n)

    # Selection: college attendance
    college_star = (
        -0.5
        + 0.5 * ability
        + 0.08 * parent_education
        + 0.01 * family_income
        - 0.02 * distance_college  # exclusion restriction
        - 0.06 * tuition  # exclusion restriction
        + 0.3 * urban
        + rng.normal(0, 1, n)
    )
    college = (college_star > 0).astype(int)

    # Outcome: log wages (correlated errors with selection)
    v = rng.normal(0, 1, n)
    rho = 0.4
    eta = rho * (college_star - college_star.mean()) / college_star.std() + np.sqrt(1 - rho**2) * v

    log_wage = (
        2.0
        + 0.3 * ability
        + 0.02 * parent_education
        + 0.005 * family_income
        + 0.1 * urban
        - 0.05 * female
        + 0.25 * eta
    )
    wage = np.exp(log_wage)

    # Wages observed only for college attendees
    wage = np.where(college == 1, np.round(wage, 2), np.nan)

    return pd.DataFrame(
        {
            "college": college,
            "wage": wage,
            "ability": np.round(ability, 3),
            "parent_education": parent_education,
            "family_income": np.round(family_income, 2),
            "distance_college": np.round(distance_college, 1),
            "tuition": np.round(tuition, 2),
            "urban": urban,
            "female": female,
        }
    )


if __name__ == "__main__":
    import os

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    datasets = {
        "labor_supply.csv": generate_labor_supply(),
        "health_expenditure_panel.csv": generate_health_panel(),
        "consumer_durables_panel.csv": generate_consumer_durables(),
        "mroz_1987.csv": generate_mroz_data(),
        "college_wage.csv": generate_college_wage(),
    }

    for filename, df in datasets.items():
        path = os.path.join(data_dir, filename)
        df.to_csv(path, index=False)
        print(f"Generated {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        print()
