"""
Data generation functions for GMM tutorial simulated datasets.

Provides DGP (Data Generating Process) functions for creating pedagogical
datasets that demonstrate key concepts in GMM estimation.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_nickell_bias_data(
    output_path: str | None = None,
    seed: int = 42,
    N: int = 500,
    rho_values: list[float] = [0.3, 0.5, 0.8],
    T_values: list[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Generate panel data demonstrating Nickell bias.

    DGP: y_{it} = rho * y_{i,t-1} + mu_i + eps_{it}

    Parameters
    ----------
    output_path : str, optional
        Path to save the CSV file.
    seed : int
        Random seed for reproducibility.
    N : int
        Number of entities per (rho, T) combination.
    rho_values : list of float
        True autoregressive parameters.
    T_values : list of int
        Panel lengths.

    Returns
    -------
    pd.DataFrame
        Panel data with columns: entity, time, y, rho, T.
    """
    rng = np.random.default_rng(seed)
    frames = []

    for rho in rho_values:
        for T in T_values:
            # Fixed effects
            mu = rng.normal(0, 1, N)
            # Error terms (extra burn-in periods)
            burn_in = 50
            total_T = T + burn_in

            y = np.zeros((N, total_T))
            y[:, 0] = mu / (1 - rho) + rng.normal(0, 1, N)

            for t in range(1, total_T):
                eps = rng.normal(0, 1, N)
                y[:, t] = rho * y[:, t - 1] + mu + eps

            # Keep only last T periods
            y = y[:, burn_in:]

            for i in range(N):
                for t in range(T):
                    frames.append(
                        {
                            "entity": i + 1,
                            "time": t + 1,
                            "y": y[i, t],
                            "rho": rho,
                            "T": T,
                        }
                    )

    df = pd.DataFrame(frames)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def generate_weak_instruments_data(
    output_path: str | None = None,
    seed: int = 123,
    N: int = 200,
    T: int = 10,
    rho: float = 0.95,
) -> pd.DataFrame:
    """
    Generate panel data with weak instruments (near unit root).

    DGP: y_{it} = rho * y_{i,t-1} + beta * x_{it} + mu_i + eps_{it}
    with rho close to 1, making lagged differences weak instruments.

    Parameters
    ----------
    output_path : str, optional
        Path to save the CSV file.
    seed : int
        Random seed for reproducibility.
    N : int
        Number of entities.
    T : int
        Number of time periods.
    rho : float
        Autoregressive parameter (close to 1 for weak instruments).

    Returns
    -------
    pd.DataFrame
        Panel data with columns: entity, time, y, x.
    """
    rng = np.random.default_rng(seed)
    beta = 0.5
    burn_in = 50

    mu = rng.normal(0, 2, N)
    frames = []

    for i in range(N):
        total_T = T + burn_in
        x = rng.normal(0, 1, total_T)
        y = np.zeros(total_T)
        y[0] = mu[i] / (1 - rho) + rng.normal(0, 1)

        for t in range(1, total_T):
            eps = rng.normal(0, 1)
            y[t] = rho * y[t - 1] + beta * x[t] + mu[i] + eps

        # Keep last T periods
        for t in range(T):
            frames.append(
                {
                    "entity": i + 1,
                    "time": t + 1,
                    "y": y[burn_in + t],
                    "x": x[burn_in + t],
                }
            )

    df = pd.DataFrame(frames)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def generate_bad_specification_data(
    output_path: str | None = None,
    seed: int = 456,
    N: int = 300,
    T: int = 8,
) -> pd.DataFrame:
    """
    Generate panel data with a correlated omitted variable.

    DGP: y_{it} = rho * y_{i,t-1} + beta1 * x1_{it} + beta2 * x2_{it} + mu_i + eps_{it}
    where x2 is correlated with instruments, causing Hansen J to reject.

    Parameters
    ----------
    output_path : str, optional
        Path to save the CSV file.
    seed : int
        Random seed for reproducibility.
    N : int
        Number of entities.
    T : int
        Number of time periods.

    Returns
    -------
    pd.DataFrame
        Panel data with columns: entity, time, y, x1, x2_omitted.
    """
    rng = np.random.default_rng(seed)
    rho = 0.5
    beta1 = 1.0
    beta2 = 0.8
    burn_in = 50

    mu = rng.normal(0, 1, N)
    frames = []

    for i in range(N):
        total_T = T + burn_in
        x1 = rng.normal(0, 1, total_T)

        # x2 is correlated with lagged y (violates instrument exogeneity)
        y = np.zeros(total_T)
        x2 = np.zeros(total_T)
        y[0] = mu[i] / (1 - rho) + rng.normal(0, 1)
        x2[0] = 0.6 * y[0] + rng.normal(0, 0.5)

        for t in range(1, total_T):
            eps = rng.normal(0, 1)
            x2[t] = 0.6 * y[t - 1] + rng.normal(0, 0.5)
            y[t] = rho * y[t - 1] + beta1 * x1[t] + beta2 * x2[t] + mu[i] + eps

        for t in range(T):
            frames.append(
                {
                    "entity": i + 1,
                    "time": t + 1,
                    "y": y[burn_in + t],
                    "x1": x1[burn_in + t],
                    "x2_omitted": x2[burn_in + t],
                }
            )

    df = pd.DataFrame(frames)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def generate_medium_panel_data(
    output_path: str | None = None,
    seed: int = 789,
    N: int = 200,
    T: int = 15,
    rho: float = 0.7,
) -> pd.DataFrame:
    """
    Generate medium-sized panel for bias correction comparison.

    DGP: y_{it} = rho * y_{i,t-1} + beta * x_{it} + mu_i + eps_{it}
    with moderate N and T, suitable for comparing two-step and CUE estimators.

    Parameters
    ----------
    output_path : str, optional
        Path to save the CSV file.
    seed : int
        Random seed for reproducibility.
    N : int
        Number of entities.
    T : int
        Number of time periods.
    rho : float
        True autoregressive parameter.

    Returns
    -------
    pd.DataFrame
        Panel data with columns: entity, time, y, x.
    """
    rng = np.random.default_rng(seed)
    beta = 0.3
    burn_in = 50

    mu = rng.normal(0, 1, N)
    frames = []

    for i in range(N):
        total_T = T + burn_in
        x = rng.normal(0, 1, total_T)
        y = np.zeros(total_T)
        y[0] = mu[i] / (1 - rho) + rng.normal(0, 1)

        for t in range(1, total_T):
            eps = rng.normal(0, 1)
            y[t] = rho * y[t - 1] + beta * x[t] + mu[i] + eps

        for t in range(T):
            frames.append(
                {
                    "entity": i + 1,
                    "time": t + 1,
                    "y": y[burn_in + t],
                    "x": x[burn_in + t],
                }
            )

    df = pd.DataFrame(frames)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def monte_carlo_simulation(
    dgp_func,
    estimator_func,
    n_simulations: int = 1000,
    seed: int = 42,
    **dgp_kwargs,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for a given DGP and estimator.

    Parameters
    ----------
    dgp_func : callable
        Function that generates panel data (returns DataFrame).
    estimator_func : callable
        Function that takes a DataFrame and returns estimated parameters.
    n_simulations : int
        Number of Monte Carlo replications.
    seed : int
        Base random seed (incremented per replication).
    **dgp_kwargs
        Additional arguments passed to dgp_func.

    Returns
    -------
    pd.DataFrame
        Results from each replication.
    """
    results = []
    for s in range(n_simulations):
        data = dgp_func(seed=seed + s, **dgp_kwargs)
        estimates = estimator_func(data)
        estimates["simulation"] = s + 1
        results.append(estimates)

    return pd.DataFrame(results)


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    print("Generating simulated datasets...")
    generate_nickell_bias_data(output_path=data_dir / "dgp_nickell_bias.csv")
    print("  -> dgp_nickell_bias.csv")

    generate_weak_instruments_data(output_path=data_dir / "weak_instruments.csv")
    print("  -> weak_instruments.csv")

    generate_bad_specification_data(output_path=data_dir / "bad_specification.csv")
    print("  -> bad_specification.csv")

    generate_medium_panel_data(output_path=data_dir / "medium_panel_bias.csv")
    print("  -> medium_panel_bias.csv")

    print("Done!")
