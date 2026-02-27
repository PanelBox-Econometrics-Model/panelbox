"""
Dataset Loading Functions.

==========================

Functions for loading example panel datasets bundled with PanelBox.

Datasets are organized in categories (count, gmm, spatial, etc.)
and can be loaded by name using ``load_dataset()``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from panelbox.core.panel_data import PanelData

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _get_data_path() -> str:
    """Get the path to the data directory."""
    return _DATA_DIR


def _find_dataset(name: str, category: str | None = None) -> str | None:
    """Find a dataset CSV file by name, optionally within a category.

    Parameters
    ----------
    name : str
        Dataset name (without .csv extension).
    category : str or None
        Subdirectory to search in (e.g. "count", "gmm").
        If None, searches all subdirectories.

    Returns
    -------
    str or None
        Full path to the CSV file, or None if not found.
    """
    filename = f"{name}.csv"

    # 1. Check root data directory
    root_path = os.path.join(_DATA_DIR, filename)
    if category is None and os.path.isfile(root_path):
        return root_path

    # 2. Check specific category
    if category is not None:
        cat_path = os.path.join(_DATA_DIR, category, filename)
        if os.path.isfile(cat_path):
            return cat_path
        return None

    # 3. Search all subdirectories
    for entry in sorted(os.listdir(_DATA_DIR)):
        subdir = os.path.join(_DATA_DIR, entry)
        if os.path.isdir(subdir):
            candidate = os.path.join(subdir, filename)
            if os.path.isfile(candidate):
                return candidate

    return None


def load_grunfeld(return_panel_data: bool = False) -> pd.DataFrame | PanelData:
    """
    Load Grunfeld investment data.

    Classic panel dataset on investment behavior of large US corporations.
    10 firms, 20 years (1935-1954), 200 observations (balanced).

    Parameters
    ----------
    return_panel_data : bool, default=False
        If True, returns a PanelData object instead of DataFrame.

    Returns
    -------
    pd.DataFrame or PanelData

    Examples
    --------
    >>> from panelbox.datasets import load_grunfeld
    >>> data = load_grunfeld()
    >>> print(data.shape)
    (200, 5)
    """
    data_path = os.path.join(_DATA_DIR, "grunfeld.csv")
    df = pd.read_csv(data_path)

    if return_panel_data:
        from panelbox.core.panel_data import PanelData

        return PanelData(df, entity_col="firm", time_col="year")

    return df


def load_abdata(return_panel_data: bool = False) -> pd.DataFrame | PanelData | None:
    """
    Load Arellano-Bond employment data.

    UK company employment data used in Arellano & Bond (1991).
    ~140 firms, 7-9 years (1976-1984), ~1000 observations (unbalanced).

    Parameters
    ----------
    return_panel_data : bool, default=False
        If True, returns a PanelData object instead of DataFrame.

    Returns
    -------
    pd.DataFrame or PanelData or None

    Examples
    --------
    >>> from panelbox.datasets import load_abdata
    >>> data = load_abdata()
    """
    data_path = os.path.join(_DATA_DIR, "abdata.csv")

    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)

    if return_panel_data:
        from panelbox.core.panel_data import PanelData

        entity_col = "id" if "id" in df.columns else df.columns[0]
        time_col = "year" if "year" in df.columns else df.columns[1]
        return PanelData(df, entity_col=entity_col, time_col=time_col)

    return df


def list_datasets(category: str | None = None) -> list[str]:
    """
    List all available datasets.

    Parameters
    ----------
    category : str or None
        If provided, list only datasets in that category.
        If None, list all datasets across all categories.

    Returns
    -------
    list of str
        Dataset names. For categorized datasets, returns "category/name".

    Examples
    --------
    >>> from panelbox.datasets import list_datasets
    >>> all_ds = list_datasets()
    >>> count_ds = list_datasets("count")
    """
    datasets: list[str] = []

    if not os.path.exists(_DATA_DIR):
        return datasets

    if category is not None:
        cat_dir = os.path.join(_DATA_DIR, category)
        if os.path.isdir(cat_dir):
            for f in os.listdir(cat_dir):
                if f.endswith(".csv"):
                    datasets.append(f[:-4])
        return sorted(datasets)

    # Root-level datasets
    for f in os.listdir(_DATA_DIR):
        if f.endswith(".csv"):
            datasets.append(f[:-4])

    # Categorized datasets
    for entry in sorted(os.listdir(_DATA_DIR)):
        subdir = os.path.join(_DATA_DIR, entry)
        if os.path.isdir(subdir):
            for f in os.listdir(subdir):
                if f.endswith(".csv"):
                    datasets.append(f"{entry}/{f[:-4]}")

    return sorted(datasets)


def list_categories() -> list[str]:
    """
    List available dataset categories.

    Returns
    -------
    list of str
        Category names (subdirectories in the data folder).

    Examples
    --------
    >>> from panelbox.datasets import list_categories
    >>> print(list_categories())
    ['censored', 'count', 'diagnostics', ...]
    """
    categories: list[str] = []
    if os.path.exists(_DATA_DIR):
        for entry in sorted(os.listdir(_DATA_DIR)):
            if os.path.isdir(os.path.join(_DATA_DIR, entry)):
                categories.append(entry)
    return categories


def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    """
    Get information about a specific dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'grunfeld', 'healthcare_visits').

    Returns
    -------
    dict
        Dictionary with dataset metadata and statistics.
    """
    known_info: dict[str, dict[str, str]] = {
        "grunfeld": {
            "name": "Grunfeld Investment Data",
            "description": "Investment data for 10 US manufacturing firms (1935-1954)",
            "source": "Grunfeld (1958)",
            "entity_col": "firm",
            "time_col": "year",
        },
        "abdata": {
            "name": "Arellano-Bond Employment Data",
            "description": "UK company employment data (1976-1984)",
            "source": "Arellano & Bond (1991)",
            "entity_col": "id",
            "time_col": "year",
        },
    }

    base_info: dict[str, Any] = dict(
        known_info.get(
            dataset_name,
            {"name": dataset_name, "description": "", "source": ""},
        )
    )

    # Try to load and add statistics
    try:
        df = load_dataset(dataset_name)
        if df is not None:
            base_info["n_obs"] = len(df)
            base_info["variables"] = list(df.columns)

            # Check if panel is balanced
            entity_col = base_info.get("entity_col")
            time_col = base_info.get("time_col")
            if entity_col and time_col and entity_col in df.columns and time_col in df.columns:
                counts = df.groupby(entity_col)[time_col].count()
                base_info["balanced"] = bool(counts.min() == counts.max())
    except Exception as e:
        base_info["error"] = str(e)

    return base_info


def load_dataset(name: str, category: str | None = None) -> pd.DataFrame | None:
    """
    Load a dataset by name.

    Searches all bundled datasets. Use ``list_datasets()`` to see available
    names.

    Parameters
    ----------
    name : str
        Dataset name (without .csv). Examples: "grunfeld", "healthcare_visits",
        "bilateral_trade", "firm_investment".
    category : str or None
        Optional category to disambiguate duplicates.
        Examples: "count", "gmm", "spatial".

    Returns
    -------
    pd.DataFrame or None
        The dataset as a DataFrame, or None if not found.

    Examples
    --------
    >>> from panelbox.datasets import load_dataset
    >>> data = load_dataset("healthcare_visits")
    >>> data = load_dataset("macro_panel", category="var")
    """
    # Shortcut for named loaders
    if name == "grunfeld" and category is None:
        return load_grunfeld()
    if name == "abdata" and category is None:
        return load_abdata()

    # Handle "category/name" format
    if "/" in name and category is None:
        category, name = name.split("/", 1)

    path = _find_dataset(name, category)
    if path is not None:
        return pd.read_csv(path)

    logger.warning("Dataset '%s' not found.", name)
    logger.warning("Available datasets: %s", ", ".join(list_datasets()))
    return None
