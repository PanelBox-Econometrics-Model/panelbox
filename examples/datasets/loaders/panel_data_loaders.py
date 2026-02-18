"""
Panel Data Loading Utilities

This module provides standardized loading functions for tutorial datasets.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Determine base path for datasets
_BASE_PATH = Path(__file__).parent.parent / "panel"


def _validate_panel_data(
    data: pd.DataFrame, entity_col: str, time_col: str, name: str
) -> pd.DataFrame:
    """
    Validate panel data structure.

    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
    entity_col : str
        Entity identifier column
    time_col : str
        Time identifier column
    name : str
        Dataset name for error messages

    Returns
    -------
    pd.DataFrame
        Validated data

    Raises
    ------
    ValueError
        If required columns are missing
    """
    if entity_col not in data.columns:
        raise ValueError(f"{name}: Entity column '{entity_col}' not found")
    if time_col not in data.columns:
        raise ValueError(f"{name}: Time column '{time_col}' not found")

    # Check for missing entity or time identifiers
    if data[entity_col].isna().any():
        raise ValueError(f"{name}: Missing values in entity column '{entity_col}'")
    if data[time_col].isna().any():
        raise ValueError(f"{name}: Missing values in time column '{time_col}'")

    return data


def load_grunfeld(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load Grunfeld investment data.

    Classic panel dataset on corporate investment decisions for 10 large
    U.S. manufacturing firms (1935-1954).

    Parameters
    ----------
    path : str, optional
        Path to grunfeld.csv. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Grunfeld data with columns: firm, year, invest, value, capital

    Examples
    --------
    >>> data = load_grunfeld()
    >>> print(data.shape)
    (200, 5)
    >>> print(data.head())
    """
    if path is None:
        path = _BASE_PATH / "grunfeld.csv"

    data = pd.read_csv(path)

    # Validate
    data = _validate_panel_data(data, "firm", "year", "Grunfeld")

    # Ensure proper types
    data["firm"] = data["firm"].astype(str)
    data["year"] = data["year"].astype(int)

    # Convert numeric columns to float
    numeric_cols = ["invest", "value", "capital"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def load_wage_panel(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load wage panel data.

    Simulated/subset wage panel data based on NLSY/PSID structure.
    Contains wage and demographic information for 500 individuals over 7 years.

    Parameters
    ----------
    path : str, optional
        Path to wage_panel.csv. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Wage panel data

    Examples
    --------
    >>> data = load_wage_panel()
    >>> print(data.columns)
    """
    if path is None:
        path = _BASE_PATH / "wage_panel.csv"

    if not Path(path).exists():
        raise FileNotFoundError(
            f"wage_panel.csv not found at {path}. "
            "This dataset will be created in future development."
        )

    data = pd.read_csv(path)

    # Validate
    data = _validate_panel_data(data, "person_id", "year", "Wage Panel")

    # Ensure proper types
    data["person_id"] = data["person_id"].astype(str)
    data["year"] = data["year"].astype(int)

    # Numeric columns
    numeric_cols = ["lwage", "hours", "exper"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Binary/categorical columns
    binary_cols = ["union", "married", "black", "hisp"]
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].astype(int)

    return data


def load_country_growth(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load country growth data.

    Macroeconomic panel data for 80 countries over 30 years,
    based on Penn World Tables structure.

    Parameters
    ----------
    path : str, optional
        Path to country_growth.csv. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Country growth data

    Examples
    --------
    >>> data = load_country_growth()
    >>> print(data.columns)
    """
    if path is None:
        path = _BASE_PATH / "country_growth.csv"

    if not Path(path).exists():
        raise FileNotFoundError(
            f"country_growth.csv not found at {path}. "
            "This dataset will be created in future development."
        )

    data = pd.read_csv(path)

    # Validate
    data = _validate_panel_data(data, "country_code", "year", "Country Growth")

    # Ensure proper types
    data["country_code"] = data["country_code"].astype(str)
    data["year"] = data["year"].astype(int)

    # Numeric columns
    numeric_cols = [
        "gdp_growth",
        "invest_rate",
        "school_years",
        "pop_growth",
        "trade_openness",
        "govt_consumption",
        "inflation",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def load_firm_productivity(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load firm productivity data.

    Manufacturing firm productivity panel with 300 firms over 12 years.
    **Unbalanced** (some firms exit/enter).

    Parameters
    ----------
    path : str, optional
        Path to firm_productivity.csv. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Firm productivity data

    Examples
    --------
    >>> data = load_firm_productivity()
    >>> print(data.columns)
    """
    if path is None:
        path = _BASE_PATH / "firm_productivity.csv"

    if not Path(path).exists():
        raise FileNotFoundError(
            f"firm_productivity.csv not found at {path}. "
            "This dataset will be created in future development."
        )

    data = pd.read_csv(path)

    # Validate
    data = _validate_panel_data(data, "firm_id", "year", "Firm Productivity")

    # Ensure proper types
    data["firm_id"] = data["firm_id"].astype(str)
    data["year"] = data["year"].astype(int)

    # Numeric columns (logs)
    numeric_cols = ["log_output", "log_labor", "log_capital", "log_materials", "age"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Binary columns
    binary_cols = ["export", "foreign_owned"]
    for col in binary_cols:
        if col in data.columns:
            data[col] = data[col].astype(int)

    # Industry code
    if "industry" in data.columns:
        data["industry"] = data["industry"].astype(str)

    return data


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of dataset: 'grunfeld', 'wage_panel', 'country_growth', 'firm_productivity'

    Returns
    -------
    dict
        Dictionary with dataset metadata

    Examples
    --------
    >>> info = get_dataset_info('grunfeld')
    >>> print(info['description'])
    """
    info_dict = {
        "grunfeld": {
            "name": "Grunfeld Investment Data",
            "entities": 10,
            "periods": 20,
            "total_obs": 200,
            "balanced": True,
            "entity_col": "firm",
            "time_col": "year",
            "description": "Corporate investment for 10 U.S. firms (1935-1954)",
            "source": "Grunfeld (1958)",
            "loader": load_grunfeld,
        },
        "wage_panel": {
            "name": "Wage Panel Data",
            "entities": 500,
            "periods": 7,
            "total_obs": 3500,
            "balanced": True,
            "entity_col": "person_id",
            "time_col": "year",
            "description": "Wage and demographics for 500 individuals over 7 years",
            "source": "Simulated (NLSY/PSID structure)",
            "loader": load_wage_panel,
        },
        "country_growth": {
            "name": "Country Growth Data",
            "entities": 80,
            "periods": 30,
            "total_obs": 2400,
            "balanced": True,
            "entity_col": "country_code",
            "time_col": "year",
            "description": "Macroeconomic panel for 80 countries over 30 years",
            "source": "Penn World Tables (adapted)",
            "loader": load_country_growth,
        },
        "firm_productivity": {
            "name": "Firm Productivity Data",
            "entities": 300,
            "periods": 12,
            "total_obs": 3200,
            "balanced": False,
            "entity_col": "firm_id",
            "time_col": "year",
            "description": "Manufacturing firm productivity (unbalanced)",
            "source": "Simulated",
            "loader": load_firm_productivity,
        },
    }

    if dataset_name not in info_dict:
        available = ", ".join(info_dict.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    return info_dict[dataset_name]


def list_available_datasets() -> pd.DataFrame:
    """
    List all available datasets with metadata.

    Returns
    -------
    pd.DataFrame
        Table of available datasets

    Examples
    --------
    >>> datasets = list_available_datasets()
    >>> print(datasets)
    """
    datasets = ["grunfeld", "wage_panel", "country_growth", "firm_productivity"]
    info_list = []

    for dataset in datasets:
        info = get_dataset_info(dataset)
        info_list.append(
            {
                "Dataset": dataset,
                "Name": info["name"],
                "N": info["entities"],
                "T": info["periods"],
                "Total Obs": info["total_obs"],
                "Balanced": info["balanced"],
                "Source": info["source"],
            }
        )

    return pd.DataFrame(info_list)
