# tests/var/fixtures/var_test_data.py

import numpy as np
import pandas as pd


def generate_panel_var_data(
    n_entities: int = 50, n_periods: int = 20, n_vars: int = 3, var_lags: int = 2, seed: int = 42
) -> pd.DataFrame:
    """
    Gera dados de painel com DGP VAR(p) conhecido.

    DGP:
        y1_t = 0.5*y1_{t-1} + 0.2*y2_{t-1} + e1_t
        y2_t = 0.3*y1_{t-1} + 0.4*y2_{t-1} + e2_t
        y3_t = 0.1*y1_{t-1} + 0.1*y2_{t-1} + 0.6*y3_{t-1} + e3_t

    Onde e_t ~ N(0, Sigma) com:
        Sigma = [[1.0, 0.3, 0.1],
                 [0.3, 1.0, 0.2],
                 [0.1, 0.2, 1.0]]

    Parameters
    ----------
    n_entities : int, default=50
        Number of panel entities
    n_periods : int, default=20
        Number of time periods per entity
    n_vars : int, default=3
        Number of variables in the VAR system
    var_lags : int, default=2
        Number of lags (not used in current DGP but kept for compatibility)
    seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Panel data with columns ['entity', 'time', 'y1', 'y2', 'y3']
    """
    np.random.seed(seed)

    # Coeficientes verdadeiros (para validação)
    # A1_true é a matriz de coeficientes autoregressivos
    # Cada linha representa uma equação, cada coluna um lag da variável correspondente
    A1_true = np.array(
        [
            [0.5, 0.2, 0.0],  # y1_t equation
            [0.3, 0.4, 0.0],  # y2_t equation
            [0.1, 0.1, 0.6],  # y3_t equation
        ]
    )

    # Matriz de covariância dos erros
    Sigma_true = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])

    # Gerar dados
    data_list = []
    for entity in range(n_entities):
        # Valores iniciais
        y = np.zeros((n_periods, n_vars))
        y[0] = np.random.multivariate_normal(np.zeros(n_vars), Sigma_true)

        # Gerar série temporal VAR(1)
        for t in range(1, n_periods):
            epsilon = np.random.multivariate_normal(np.zeros(n_vars), Sigma_true)
            y[t] = A1_true @ y[t - 1] + epsilon

        # Criar DataFrame para esta entidade
        df_entity = pd.DataFrame(y, columns=["y1", "y2", "y3"])
        df_entity["entity"] = entity
        df_entity["time"] = range(n_periods)
        data_list.append(df_entity)

    # Concatenar todos os dados
    df = pd.concat(data_list, ignore_index=True)

    # Reordenar colunas para ter entity e time primeiro
    df = df[["entity", "time", "y1", "y2", "y3"]]

    return df


# Parâmetros verdadeiros para validação
TRUE_PARAMS = {
    "A1": np.array([[0.5, 0.2, 0.0], [0.3, 0.4, 0.0], [0.1, 0.1, 0.6]]),
    "Sigma": np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]]),
}
