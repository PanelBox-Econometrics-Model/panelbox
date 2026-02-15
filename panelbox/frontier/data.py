"""
Data validation and enumerations for Stochastic Frontier Analysis.

This module defines the core data types, enumerations, and validation
functions used throughout the frontier analysis module.
"""

from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class FrontierType(str, Enum):
    """Type of frontier to estimate.

    Attributes:
        PRODUCTION: Production frontier where inefficiency reduces output
                   (y = f(x) * exp(-u) * exp(v))
        COST: Cost frontier where inefficiency increases cost
             (y = f(x) * exp(u) * exp(v))
    """

    PRODUCTION = "production"
    COST = "cost"


class DistributionType(str, Enum):
    """Distribution assumed for the inefficiency term u.

    Attributes:
        HALF_NORMAL: Half-normal distribution (Aigner et al. 1977)
        EXPONENTIAL: Exponential distribution (Meeusen & van den Broeck 1977)
        TRUNCATED_NORMAL: Truncated normal distribution with location parameter μ
        GAMMA: Gamma distribution (Greene 1990)
    """

    HALF_NORMAL = "half_normal"
    EXPONENTIAL = "exponential"
    TRUNCATED_NORMAL = "truncated_normal"
    GAMMA = "gamma"


class ModelType(str, Enum):
    """Type of SFA model structure.

    Attributes:
        CROSS_SECTION: Cross-sectional model (no panel structure)
        POOLED: Pooled panel model (ignores time dimension)
        PITT_LEE: Pitt & Lee (1981) panel model (time-invariant inefficiency)
        BATTESE_COELLI_92: Battese & Coelli (1992) time-varying model
        BATTESE_COELLI_95: Battese & Coelli (1995) with heterogeneity
        CSS: Cornwell, Schmidt & Sickles (1990) flexible time pattern
        TRUE_FIXED_EFFECTS: Greene (2005) true fixed effects model
        TRUE_RANDOM_EFFECTS: Greene (2005) true random effects model
    """

    CROSS_SECTION = "cross_section"
    POOLED = "pooled"
    PITT_LEE = "pitt_lee"
    BATTESE_COELLI_92 = "bc92"
    BATTESE_COELLI_95 = "bc95"
    CSS = "css"
    TRUE_FIXED_EFFECTS = "tfe"
    TRUE_RANDOM_EFFECTS = "tre"


def validate_frontier_data(
    data: pd.DataFrame,
    depvar: str,
    exog: List[str],
    entity: Optional[str] = None,
    time: Optional[str] = None,
    inefficiency_vars: Optional[List[str]] = None,
    het_vars: Optional[List[str]] = None,
) -> dict:
    """Validate data for frontier estimation.

    Performs comprehensive validation of input data for SFA models,
    checking for missing values, variable existence, and data types.

    Parameters:
        data: Input DataFrame
        depvar: Dependent variable name
        exog: List of exogenous variable names
        entity: Entity identifier (for panel data)
        time: Time identifier (for panel data)
        inefficiency_vars: Variables for inefficiency mean (BC95)
        het_vars: Variables for heteroskedasticity

    Returns:
        Dictionary with validation results and processed data info

    Raises:
        ValueError: If validation fails
        KeyError: If required variables not found in data
    """
    # Check DataFrame is not empty
    if data.empty:
        raise ValueError("Input DataFrame is empty")

    # Check all required variables exist
    all_vars = [depvar] + exog
    if inefficiency_vars:
        all_vars.extend(inefficiency_vars)
    if het_vars:
        all_vars.extend(het_vars)
    if entity:
        all_vars.append(entity)
    if time:
        all_vars.append(time)

    missing_vars = set(all_vars) - set(data.columns)
    if missing_vars:
        raise KeyError(f"Variables not found in DataFrame: {missing_vars}")

    # Check for missing values
    check_cols = [depvar] + exog
    if inefficiency_vars:
        check_cols.extend(inefficiency_vars)
    if het_vars:
        check_cols.extend(het_vars)

    missing_counts = data[check_cols].isnull().sum()
    if missing_counts.any():
        missing_info = missing_counts[missing_counts > 0]
        raise ValueError(
            f"Missing values detected:\n{missing_info}\n"
            "SFA estimation requires complete cases. "
            "Please handle missing data before estimation."
        )

    # Check numeric types
    for var in check_cols:
        if not pd.api.types.is_numeric_dtype(data[var]):
            raise ValueError(
                f"Variable '{var}' is not numeric (dtype: {data[var].dtype}). "
                "All variables must be numeric for SFA estimation."
            )

    # Check for infinite values
    inf_counts = np.isinf(data[check_cols]).sum()
    if inf_counts.any():
        inf_info = inf_counts[inf_counts > 0]
        raise ValueError(
            f"Infinite values detected:\n{inf_info}\n" "Please remove or replace infinite values."
        )

    # Validate panel structure if applicable
    n_obs = len(data)
    n_entities = None
    n_periods = None
    is_balanced = None

    if entity and time:
        # Check panel structure
        entity_counts = data.groupby(entity)[time].count()
        n_entities = data[entity].nunique()
        n_periods = data[time].nunique()
        is_balanced = entity_counts.std() == 0

        if not is_balanced:
            min_periods = entity_counts.min()
            max_periods = entity_counts.max()
            print(
                f"Warning: Unbalanced panel detected. "
                f"Periods per entity: {min_periods} to {max_periods}"
            )

    # Check for collinearity (warn only)
    try:
        from numpy.linalg import matrix_rank

        X = data[exog].values
        if matrix_rank(X) < len(exog):
            print(
                f"Warning: Potential collinearity detected. "
                f"Matrix rank ({matrix_rank(X)}) < number of variables ({len(exog)})"
            )
    except Exception:
        pass  # If check fails, continue anyway

    return {
        "n_obs": n_obs,
        "n_entities": n_entities,
        "n_periods": n_periods,
        "is_balanced": is_balanced,
        "n_exog": len(exog),
        "has_panel_structure": entity is not None and time is not None,
        "validation_passed": True,
    }


def prepare_panel_index(
    data: pd.DataFrame, entity: Optional[str] = None, time: Optional[str] = None
) -> pd.DataFrame:
    """Prepare panel data with proper multi-index.

    Creates a properly indexed copy of the data with MultiIndex
    (entity, time) for panel models, or simple integer index for
    cross-sectional models.

    Parameters:
        data: Input DataFrame
        entity: Entity identifier column name
        time: Time identifier column name

    Returns:
        DataFrame with proper index
    """
    data = data.copy()

    if entity and time:
        # Create multi-index for panel data
        if not isinstance(data.index, pd.MultiIndex):
            data = data.set_index([entity, time])
            data.index.names = ["entity", "time"]
        return data
    elif entity:
        # Single entity index (pooled cross-section)
        if data.index.name != entity:
            data = data.set_index(entity)
            data.index.name = "entity"
        return data
    else:
        # Cross-sectional data - ensure simple integer index
        if not isinstance(data.index, pd.RangeIndex):
            data = data.reset_index(drop=True)
        return data


def check_distribution_compatibility(
    dist: DistributionType, model_type: ModelType, inefficiency_vars: Optional[List[str]] = None
) -> None:
    """Check if distribution is compatible with model type.

    Parameters:
        dist: Distribution type for inefficiency
        model_type: Type of SFA model
        inefficiency_vars: Variables for heterogeneity in inefficiency

    Raises:
        ValueError: If incompatible combination detected
    """
    # BC95 requires truncated normal
    if inefficiency_vars and dist != DistributionType.TRUNCATED_NORMAL:
        raise ValueError(
            "Battese-Coelli (1995) heterogeneity specification requires "
            "truncated_normal distribution. "
            f"Got: {dist}"
        )

    # Check model-specific restrictions
    if model_type == ModelType.BATTESE_COELLI_95:
        if dist != DistributionType.TRUNCATED_NORMAL:
            raise ValueError("Battese-Coelli (1995) model requires truncated_normal distribution")

    # Gamma is computationally intensive - warn for large panels
    if dist == DistributionType.GAMMA:
        if model_type not in [ModelType.CROSS_SECTION, ModelType.POOLED]:
            print(
                "Warning: Gamma distribution with panel models can be "
                "computationally intensive. Consider half_normal or exponential "
                "for faster estimation."
            )
