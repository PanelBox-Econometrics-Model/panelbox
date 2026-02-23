"""Custom Hypothesis strategies for panel data testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def balanced_panels(
    draw,
    min_entities=3,
    max_entities=50,
    min_periods=3,
    max_periods=30,
    n_regressors=None,
    min_regressors=1,
    max_regressors=5,
):
    """Generate a balanced panel DataFrame with random but valid data."""
    n_entities = draw(st.integers(min_value=min_entities, max_value=max_entities))
    n_periods = draw(st.integers(min_value=min_periods, max_value=max_periods))

    if n_regressors is None:
        n_regressors = draw(st.integers(min_value=min_regressors, max_value=max_regressors))

    n_obs = n_entities * n_periods

    # Entity and time indices
    entities = np.repeat(np.arange(n_entities), n_periods)
    times = np.tile(np.arange(n_periods), n_entities)

    # Generate X columns with reasonable values
    X_data = {}
    for i in range(n_regressors):
        col = draw(
            arrays(
                dtype=np.float64,
                shape=(n_obs,),
                elements=st.floats(
                    min_value=-100,
                    max_value=100,
                    allow_nan=False,
                    allow_infinity=False,
                ),
            )
        )
        X_data[f"x{i + 1}"] = col

    # Generate y as linear combination + noise
    beta = draw(
        arrays(
            dtype=np.float64,
            shape=(n_regressors,),
            elements=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
        )
    )
    X_matrix = np.column_stack(list(X_data.values()))
    noise = draw(
        arrays(
            dtype=np.float64,
            shape=(n_obs,),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    y = X_matrix @ beta + noise

    # Ensure the design matrix [1, X] has full column rank and is well-conditioned.
    # This catches: all-zero columns, collinear columns, constant columns,
    # and near-singular matrices that cause NaN in robust SE.
    X_with_intercept = np.column_stack([np.ones(n_obs), X_matrix])
    assume(np.linalg.matrix_rank(X_with_intercept) == n_regressors + 1)
    cond = np.linalg.cond(X_with_intercept)
    assume(np.isfinite(cond) and cond < 1e8)
    # Also ensure y has variation (avoid degenerate regressions with zero SE)
    assume(np.std(y) > 1e-6)

    df = pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": y,
            **X_data,
        }
    )

    formula = "y ~ " + " + ".join(X_data.keys())
    return df, formula, "entity", "time"


@st.composite
def unbalanced_panels(draw, min_entities=3, max_entities=20):
    """Generate an unbalanced panel with varying periods per entity."""
    n_entities = draw(st.integers(min_value=min_entities, max_value=max_entities))

    dfs = []
    for i in range(n_entities):
        n_periods = draw(st.integers(min_value=2, max_value=15))
        start_period = draw(st.integers(min_value=0, max_value=5))
        periods = np.arange(start_period, start_period + n_periods)

        x1 = draw(
            arrays(
                np.float64,
                shape=(n_periods,),
                elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
            )
        )
        y = 2.0 * x1 + draw(
            arrays(
                np.float64,
                shape=(n_periods,),
                elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            )
        )

        dfs.append(
            pd.DataFrame(
                {
                    "entity": i,
                    "time": periods,
                    "y": y,
                    "x1": x1,
                }
            )
        )

    result_df = pd.concat(dfs, ignore_index=True)
    # Ensure x1 has sufficient variation
    assume(np.std(result_df["x1"].values) > 1e-6)
    return result_df, "y ~ x1", "entity", "time"


@st.composite
def positive_definite_matrices(draw, size=None, min_size=2, max_size=10):
    """Generate a symmetric positive definite matrix."""
    if size is None:
        size = draw(st.integers(min_value=min_size, max_value=max_size))

    A = draw(
        arrays(
            dtype=np.float64,
            shape=(size, size),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        )
    )
    return A @ A.T + np.eye(size) * 0.1
