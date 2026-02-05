"""
Dataset Loading Functions
==========================

Functions for loading example panel datasets.

Each dataset includes:
- Description of the data source
- Variable definitions
- Example usage
- Citation information
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pandas as pd

if TYPE_CHECKING:
    from panelbox.core.panel_data import PanelData


def _get_data_path() -> str:
    """Get the path to the data directory."""
    return os.path.join(os.path.dirname(__file__), "data")


def load_grunfeld(return_panel_data: bool = False) -> Union[pd.DataFrame, "PanelData"]:
    """
    Load Grunfeld investment data.

    Classic panel dataset on investment behavior of large US corporations.

    Parameters
    ----------
    return_panel_data : bool, default=False
        If True, returns a PanelData object instead of DataFrame

    Returns
    -------
    pd.DataFrame or PanelData
        Panel dataset with firm-year observations

    Notes
    -----
    **Dataset Description:**

    The Grunfeld data contains observations on 10 large US manufacturing firms
    over the period 1935-1954 (20 years). It has been widely used to illustrate
    panel data econometric methods.

    **Variables:**
    - `firm` : Firm identifier (1-10)
    - `year` : Year (1935-1954)
    - `invest` : Gross investment (millions of dollars)
    - `value` : Market value of the firm (millions of dollars)
    - `capital` : Stock of plant and equipment (millions of dollars)

    **Sample Size:**
    - Entities (N): 10 firms
    - Time periods (T): 20 years
    - Total observations: 200

    **Panel Structure:**
    - Balanced panel (all firms observed in all years)

    **Common Uses:**
    - Fixed effects estimation
    - Between vs. within variation
    - Dynamic panel models

    **Citation:**
    Grunfeld, Y. (1958). The determinants of corporate investment.
    Unpublished Ph.D. dissertation, University of Chicago.

    **Source:**
    Standard dataset in econometrics, available in Stata (`webuse grunfeld`)
    and R (`plm` package).

    Examples
    --------
    >>> import panelbox as pb
    >>>
    >>> # Load data
    >>> data = pb.load_grunfeld()
    >>> print(data.head())
    >>>
    >>> # Panel structure
    >>> print(f"Firms: {data['firm'].nunique()}")
    >>> print(f"Years: {data['year'].nunique()}")
    >>> print(f"Total obs: {len(data)}")
    >>>
    >>> # Estimate fixed effects
    >>> fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    >>> results = fe.fit()
    >>> print(results.summary())

    See Also
    --------
    FixedEffects : Fixed Effects estimator
    RandomEffects : Random Effects estimator
    DifferenceGMM : Difference GMM estimator
    SystemGMM : System GMM estimator
    """
    data_path = os.path.join(_get_data_path(), "grunfeld.csv")
    df = pd.read_csv(data_path)

    if return_panel_data:
        from panelbox.core.panel_data import PanelData

        return PanelData(df, entity_col="firm", time_col="year")

    return df


def load_abdata(return_panel_data: bool = False) -> Optional[Union[pd.DataFrame, "PanelData"]]:
    """
    Load Arellano-Bond employment data.

    Panel dataset on UK company employment used in Arellano & Bond (1991).

    Parameters
    ----------
    return_panel_data : bool, default=False
        If True, returns a PanelData object instead of DataFrame

    Returns
    -------
    pd.DataFrame or PanelData or None
        Panel dataset with firm-year observations, or None if not found

    Notes
    -----
    **Dataset Description:**

    This is the employment dataset used in the seminal Arellano-Bond (1991)
    paper on dynamic panel GMM estimation. It contains data on UK companies.

    **Variables (typical):**
    - `id` : Company identifier
    - `year` : Year
    - `n` or `emp` : Employment (number of employees)
    - `w` or `wage` : Real wage
    - `k` or `capital` : Gross capital stock
    - `ys` or `output` : Industry output

    **Sample Size:**
    - Entities (N): ~140 firms
    - Time periods (T): 7-9 years (1976-1984)
    - Total observations: ~1,000 (unbalanced)

    **Panel Structure:**
    - Unbalanced panel (not all firms observed in all years)

    **Common Uses:**
    - Dynamic panel GMM estimation
    - Arellano-Bond Difference GMM
    - Blundell-Bond System GMM
    - Testing for serial correlation in errors

    **Citation:**
    Arellano, M., & Bond, S. (1991). Some tests of specification for panel data:
    Monte Carlo evidence and an application to employment equations.
    Review of Economic Studies, 58(2), 277-297.

    Examples
    --------
    >>> import panelbox as pb
    >>>
    >>> # Load data
    >>> data = pb.load_abdata()
    >>> if data is not None:
    ...     # Estimate Difference GMM
    ...     gmm = pb.DifferenceGMM(
    ...         data=data,
    ...         dep_var='n',
    ...         lags=1,
    ...         exog_vars=['w', 'k'],
    ...         id_var='id',
    ...         time_var='year'
    ...     )
    ...     results = gmm.fit()
    """
    data_path = os.path.join(_get_data_path(), "abdata.csv")

    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)

    if return_panel_data:
        from panelbox.core.panel_data import PanelData

        # Try to infer entity and time columns
        entity_col = "id" if "id" in df.columns else df.columns[0]
        time_col = "year" if "year" in df.columns else df.columns[1]
        return PanelData(df, entity_col=entity_col, time_col=time_col)

    return df


def list_datasets() -> List[str]:
    """
    List all available datasets.

    Returns
    -------
    list of str
        Names of available datasets

    Examples
    --------
    >>> import panelbox as pb
    >>> datasets = pb.list_datasets()
    >>> print("Available datasets:")
    >>> for ds in datasets:
    ...     print(f"  - {ds}")
    """
    datasets = []
    data_path = _get_data_path()

    if os.path.exists(data_path):
        for filename in os.listdir(data_path):
            if filename.endswith(".csv"):
                dataset_name = filename[:-4]  # Remove .csv extension
                datasets.append(dataset_name)

    return sorted(datasets)


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a specific dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'grunfeld', 'abdata')

    Returns
    -------
    dict
        Dictionary containing dataset information:
        - name: Dataset name
        - description: Brief description
        - n_entities: Number of entities (if loaded)
        - n_periods: Number of time periods (if loaded)
        - n_obs: Total observations (if loaded)
        - variables: List of variables (if loaded)
        - balanced: Whether panel is balanced (if loaded)
        - source: Data source/citation

    Examples
    --------
    >>> import panelbox as pb
    >>> info = pb.get_dataset_info('grunfeld')
    >>> print(f"Dataset: {info['name']}")
    >>> print(f"Description: {info['description']}")
    >>> print(f"Variables: {', '.join(info['variables'])}")
    """
    dataset_info = {
        "grunfeld": {
            "name": "Grunfeld Investment Data",
            "description": "Investment data for 10 US manufacturing firms (1935-1954)",
            "source": "Grunfeld (1958)",
            "citation": "Grunfeld, Y. (1958). The determinants of corporate investment.",
            "entity_col": "firm",
            "time_col": "year",
        },
        "abdata": {
            "name": "Arellano-Bond Employment Data",
            "description": "UK company employment data (1976-1984)",
            "source": "Arellano & Bond (1991)",
            "citation": "Arellano, M., & Bond, S. (1991). Review of Economic Studies, 58(2), 277-297.",
            "entity_col": "id",
            "time_col": "year",
        },
    }

    base_info: Dict[str, Any] = dict(
        dataset_info.get(
            dataset_name,
            {
                "name": dataset_name,
                "description": "Unknown dataset",
                "source": "Unknown",
            },
        )
    )

    # Try to load dataset and add statistics
    df: Optional[pd.DataFrame] = None
    try:
        if dataset_name == "grunfeld":
            df = load_grunfeld()
        elif dataset_name == "abdata":
            df = load_abdata()
        else:
            data_path = os.path.join(_get_data_path(), f"{dataset_name}.csv")
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
            else:
                return base_info

        if df is not None:
            entity_col = base_info.get("entity_col", df.columns[0])
            time_col = base_info.get("time_col", df.columns[1])

            base_info["n_entities"] = df[entity_col].nunique()
            base_info["n_periods"] = df[time_col].nunique()
            base_info["n_obs"] = len(df)
            base_info["variables"] = list(df.columns)

            # Check if balanced
            obs_per_entity = df.groupby(entity_col).size()
            base_info["balanced"] = (obs_per_entity == obs_per_entity.iloc[0]).all()

    except Exception as e:
        base_info["error"] = str(e)

    return base_info


# Convenience function for backwards compatibility
def load_dataset(name: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        Name of the dataset
    **kwargs
        Additional arguments passed to the specific load function

    Returns
    -------
    pd.DataFrame or None
        The requested dataset, or None if not found
    """
    if name == "grunfeld":
        return load_grunfeld(**kwargs)
    elif name == "abdata":
        return load_abdata(**kwargs)
    else:
        # Try to load from file
        data_path = os.path.join(_get_data_path(), f"{name}.csv")
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            print(f"Dataset '{name}' not found.")
            print(f"Available datasets: {', '.join(list_datasets())}")
            return None
