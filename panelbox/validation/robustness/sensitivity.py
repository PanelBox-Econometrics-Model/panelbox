"""
Sensitivity Analysis for Panel Data Models.

This module provides tools for assessing the robustness of panel data estimation
results through various sensitivity analysis methods including:
- Leave-one-out analysis (entities and periods)
- Subsample sensitivity analysis
- Visualization of sensitivity results

Author: PanelBox Development Team
Date: 2026-01-22
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = None

from panelbox.core.results import PanelResults


@dataclass
class SensitivityResults:
    """
    Container for sensitivity analysis results.

    Attributes
    ----------
    method : str
        Type of sensitivity analysis performed
    estimates : pd.DataFrame
        Parameter estimates for each subsample
    std_errors : pd.DataFrame
        Standard errors for each subsample
    statistics : Dict
        Summary statistics (max deviation, mean estimate, etc.)
    influential_units : List
        List of influential units (entities or periods)
    subsample_info : pd.DataFrame
        Information about each subsample used
    """

    method: str
    estimates: pd.DataFrame
    std_errors: pd.DataFrame
    statistics: Dict
    influential_units: List
    subsample_info: pd.DataFrame


class SensitivityAnalysis:
    """
    Sensitivity Analysis for Panel Data Models.

    This class provides comprehensive tools for assessing the sensitivity of
    panel data model estimates to various changes in the sample composition.

    Parameters
    ----------
    results : PanelResults
        Fitted panel model results object
    show_progress : bool, default=False
        Whether to display progress bar during estimation

    Attributes
    ----------
    results : PanelResults
        Original fitted model results
    model : PanelModel
        Original panel model object
    params : pd.Series
        Original parameter estimates
    std_errors : pd.Series
        Original standard errors

    Examples
    --------
    >>> import panelbox as pb
    >>>
    >>> # Fit model
    >>> fe = pb.FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = fe.fit()
    >>>
    >>> # Sensitivity analysis
    >>> sensitivity = pb.SensitivityAnalysis(results)
    >>>
    >>> # Leave-one-out analysis
    >>> loo_entities = sensitivity.leave_one_out_entities()
    >>> loo_periods = sensitivity.leave_one_out_periods()
    >>>
    >>> # Subset sensitivity
    >>> subset_results = sensitivity.subset_sensitivity(n_subsamples=20)
    >>>
    >>> # Visualize
    >>> fig = sensitivity.plot_sensitivity(loo_entities)
    >>> plt.show()
    """

    def __init__(self, results: PanelResults, show_progress: bool = False):
        """Initialize sensitivity analysis."""
        self.results = results

        # Get model from results
        if results._model is None:
            raise ValueError(
                "Results object must contain a reference to the original model. "
                "Ensure the model stores a reference to itself in results._model"
            )

        self.model = results._model
        self.params = results.params
        self.std_errors = results.std_errors
        self.show_progress = show_progress

        # Store original data info
        self.entity_col = self.model.data.entity_col
        self.time_col = self.model.data.time_col
        self.data = self.model.data.data.copy()

        # Get unique entities and time periods
        self.entities = sorted(self.data[self.entity_col].unique())
        self.time_periods = sorted(self.data[self.time_col].unique())

        self.n_entities = len(self.entities)
        self.n_periods = len(self.time_periods)

    def leave_one_out_entities(self, influence_threshold: float = 2.0) -> SensitivityResults:
        """
        Leave-one-out analysis by entities.

        Removes one entity at a time and re-estimates the model to assess
        the influence of each entity on parameter estimates.

        Parameters
        ----------
        influence_threshold : float, default=2.0
            Threshold for identifying influential entities (in standard deviations)

        Returns
        -------
        SensitivityResults
            Results containing estimates for each entity left out

        Notes
        -----
        An entity is considered influential if removing it causes parameter
        estimates to deviate by more than `influence_threshold` standard
        deviations from the original estimates.

        Examples
        --------
        >>> sensitivity = pb.SensitivityAnalysis(results)
        >>> loo_results = sensitivity.leave_one_out_entities()
        >>> print(loo_results.statistics)
        >>> print(loo_results.influential_units)
        """
        if self.show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(self.entities, desc="LOO Entities")
            except ImportError:
                iterator = self.entities
                warnings.warn("Install tqdm for progress bars: pip install tqdm")
        else:
            iterator = self.entities

        estimates_list = []
        std_errors_list = []
        subsample_info = []

        for entity in iterator:
            # Create subsample excluding this entity
            subsample = self.data[self.data[self.entity_col] != entity].copy()

            try:
                # Refit model on subsample
                subsample_model = self._create_model(subsample)
                subsample_results = subsample_model.fit()

                estimates_list.append(subsample_results.params.values)
                std_errors_list.append(subsample_results.std_errors.values)

                subsample_info.append(
                    {"excluded": entity, "n_obs": len(subsample), "converged": True}
                )

            except Exception as e:
                # If estimation fails, use NaN
                estimates_list.append(np.full(len(self.params), np.nan))
                std_errors_list.append(np.full(len(self.params), np.nan))

                subsample_info.append(
                    {"excluded": entity, "n_obs": len(subsample), "converged": False}
                )

                if self.show_progress:
                    warnings.warn(f"Failed to estimate without entity {entity}: {e}")

        # Convert to DataFrames
        estimates_df = pd.DataFrame(
            estimates_list, index=[f"excl_{e}" for e in self.entities], columns=self.params.index
        )

        std_errors_df = pd.DataFrame(
            std_errors_list, index=[f"excl_{e}" for e in self.entities], columns=self.params.index
        )

        subsample_info_df = pd.DataFrame(subsample_info)

        # Calculate statistics
        statistics = self._calculate_statistics(estimates_df, influence_threshold)

        # Identify influential entities
        influential_units = self._identify_influential_units(estimates_df, influence_threshold)

        return SensitivityResults(
            method="leave_one_out_entities",
            estimates=estimates_df,
            std_errors=std_errors_df,
            statistics=statistics,
            influential_units=influential_units,
            subsample_info=subsample_info_df,
        )

    def leave_one_out_periods(self, influence_threshold: float = 2.0) -> SensitivityResults:
        """
        Leave-one-out analysis by time periods.

        Removes one time period at a time and re-estimates the model to assess
        the influence of each time period on parameter estimates.

        Parameters
        ----------
        influence_threshold : float, default=2.0
            Threshold for identifying influential periods (in standard deviations)

        Returns
        -------
        SensitivityResults
            Results containing estimates for each period left out

        Notes
        -----
        A time period is considered influential if removing it causes parameter
        estimates to deviate by more than `influence_threshold` standard
        deviations from the original estimates.

        Examples
        --------
        >>> sensitivity = pb.SensitivityAnalysis(results)
        >>> loo_results = sensitivity.leave_one_out_periods()
        >>> print(loo_results.statistics)
        >>> print(loo_results.influential_units)
        """
        if self.show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(self.time_periods, desc="LOO Periods")
            except ImportError:
                iterator = self.time_periods
                warnings.warn("Install tqdm for progress bars: pip install tqdm")
        else:
            iterator = self.time_periods

        estimates_list = []
        std_errors_list = []
        subsample_info = []

        for period in iterator:
            # Create subsample excluding this period
            subsample = self.data[self.data[self.time_col] != period].copy()

            try:
                # Refit model on subsample
                subsample_model = self._create_model(subsample)
                subsample_results = subsample_model.fit()

                estimates_list.append(subsample_results.params.values)
                std_errors_list.append(subsample_results.std_errors.values)

                subsample_info.append(
                    {"excluded": period, "n_obs": len(subsample), "converged": True}
                )

            except Exception as e:
                # If estimation fails, use NaN
                estimates_list.append(np.full(len(self.params), np.nan))
                std_errors_list.append(np.full(len(self.params), np.nan))

                subsample_info.append(
                    {"excluded": period, "n_obs": len(subsample), "converged": False}
                )

                if self.show_progress:
                    warnings.warn(f"Failed to estimate without period {period}: {e}")

        # Convert to DataFrames
        estimates_df = pd.DataFrame(
            estimates_list,
            index=[f"excl_{t}" for t in self.time_periods],
            columns=self.params.index,
        )

        std_errors_df = pd.DataFrame(
            std_errors_list,
            index=[f"excl_{t}" for t in self.time_periods],
            columns=self.params.index,
        )

        subsample_info_df = pd.DataFrame(subsample_info)

        # Calculate statistics
        statistics = self._calculate_statistics(estimates_df, influence_threshold)

        # Identify influential periods
        influential_units = self._identify_influential_units(estimates_df, influence_threshold)

        return SensitivityResults(
            method="leave_one_out_periods",
            estimates=estimates_df,
            std_errors=std_errors_df,
            statistics=statistics,
            influential_units=influential_units,
            subsample_info=subsample_info_df,
        )

    def subset_sensitivity(
        self,
        n_subsamples: int = 20,
        subsample_size: float = 0.8,
        stratify: bool = True,
        random_state: Optional[int] = None,
    ) -> SensitivityResults:
        """
        Subsample sensitivity analysis.

        Randomly draws multiple subsamples and re-estimates the model on each
        to assess the stability of parameter estimates across different samples.

        Parameters
        ----------
        n_subsamples : int, default=20
            Number of random subsamples to draw
        subsample_size : float, default=0.8
            Fraction of entities to include in each subsample (0 < size < 1)
        stratify : bool, default=True
            Whether to stratify sampling to maintain temporal balance
        random_state : int, optional
            Random seed for reproducibility

        Returns
        -------
        SensitivityResults
            Results containing estimates for each subsample

        Notes
        -----
        Stratified sampling ensures each subsample maintains the same temporal
        structure by randomly selecting a fraction of entities while keeping
        all time periods for selected entities.

        Examples
        --------
        >>> sensitivity = pb.SensitivityAnalysis(results)
        >>> subset_results = sensitivity.subset_sensitivity(
        ...     n_subsamples=50,
        ...     subsample_size=0.75
        ... )
        >>> print(subset_results.statistics)
        """
        if not (0 < subsample_size < 1):
            raise ValueError("subsample_size must be between 0 and 1")

        if n_subsamples < 2:
            raise ValueError("n_subsamples must be at least 2")

        rng = np.random.RandomState(random_state)

        if self.show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(range(n_subsamples), desc="Subsamples")
            except ImportError:
                iterator = range(n_subsamples)
                warnings.warn("Install tqdm for progress bars: pip install tqdm")
        else:
            iterator = range(n_subsamples)

        estimates_list = []
        std_errors_list = []
        subsample_info = []

        n_entities_subsample = max(2, int(self.n_entities * subsample_size))

        for i in iterator:
            # Sample entities
            sampled_entities = rng.choice(self.entities, size=n_entities_subsample, replace=False)

            # Create subsample
            subsample = self.data[self.data[self.entity_col].isin(sampled_entities)].copy()

            try:
                # Refit model on subsample
                subsample_model = self._create_model(subsample)
                subsample_results = subsample_model.fit()

                estimates_list.append(subsample_results.params.values)
                std_errors_list.append(subsample_results.std_errors.values)

                subsample_info.append(
                    {
                        "subsample_id": i,
                        "n_entities": len(sampled_entities),
                        "n_obs": len(subsample),
                        "converged": True,
                    }
                )

            except Exception as e:
                # If estimation fails, use NaN
                estimates_list.append(np.full(len(self.params), np.nan))
                std_errors_list.append(np.full(len(self.params), np.nan))

                subsample_info.append(
                    {
                        "subsample_id": i,
                        "n_entities": len(sampled_entities),
                        "n_obs": len(subsample),
                        "converged": False,
                    }
                )

                if self.show_progress:
                    warnings.warn(f"Failed to estimate subsample {i}: {e}")

        # Convert to DataFrames
        estimates_df = pd.DataFrame(
            estimates_list,
            index=[f"subsample_{i}" for i in range(n_subsamples)],
            columns=self.params.index,
        )

        std_errors_df = pd.DataFrame(
            std_errors_list,
            index=[f"subsample_{i}" for i in range(n_subsamples)],
            columns=self.params.index,
        )

        subsample_info_df = pd.DataFrame(subsample_info)

        # Calculate statistics
        statistics = self._calculate_statistics(estimates_df, threshold=2.0)

        return SensitivityResults(
            method="subset_sensitivity",
            estimates=estimates_df,
            std_errors=std_errors_df,
            statistics=statistics,
            influential_units=[],  # Not applicable for subset analysis
            subsample_info=subsample_info_df,
        )

    def plot_sensitivity(
        self,
        sensitivity_results: SensitivityResults,
        params: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12, 6),
        reference_line: bool = True,
        confidence_band: bool = True,
        **kwargs,
    ) -> Figure:
        """
        Plot sensitivity analysis results.

        Creates visualization showing how parameter estimates vary across
        different subsamples or leave-one-out analyses.

        Parameters
        ----------
        sensitivity_results : SensitivityResults
            Results from sensitivity analysis
        params : List[str], optional
            List of parameters to plot. If None, plots all parameters
        figsize : Tuple[float, float], default=(12, 6)
            Figure size (width, height)
        reference_line : bool, default=True
            Whether to show reference line at original estimate
        confidence_band : bool, default=True
            Whether to show confidence band (mean ± 1.96 * std)
        **kwargs
            Additional keyword arguments passed to plt.subplots

        Returns
        -------
        Figure
            Matplotlib figure object

        Examples
        --------
        >>> sensitivity = pb.SensitivityAnalysis(results)
        >>> loo_results = sensitivity.leave_one_out_entities()
        >>> fig = sensitivity.plot_sensitivity(loo_results)
        >>> plt.show()
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "Matplotlib is required for plotting. " "Install it with: pip install matplotlib"
            )

        if params is None:
            params = list(self.params.index)

        n_params = len(params)

        # Create subplots
        fig, axes = plt.subplots(1, n_params, figsize=figsize, squeeze=False, **kwargs)
        axes = axes.flatten()

        for idx, param in enumerate(params):
            ax = axes[idx]

            # Get estimates for this parameter
            estimates = sensitivity_results.estimates[param].dropna()

            # Plot estimates
            x = range(len(estimates))
            ax.scatter(x, estimates, alpha=0.6, s=30)

            # Reference line (original estimate)
            if reference_line:
                original_value = self.params[param]
                ax.axhline(
                    original_value,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="Original",
                    alpha=0.7,
                )

            # Confidence band
            if confidence_band:
                mean_est = estimates.mean()
                std_est = estimates.std()
                ax.axhline(mean_est, color="blue", linestyle="-", alpha=0.5, label="Mean")
                ax.fill_between(
                    x,
                    mean_est - 1.96 * std_est,
                    mean_est + 1.96 * std_est,
                    alpha=0.2,
                    color="blue",
                    label="95% Band",
                )

            ax.set_xlabel("Subsample/Exclusion")
            ax.set_ylabel("Estimate")
            ax.set_title(f"{param}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Overall title
        method_title = sensitivity_results.method.replace("_", " ").title()
        fig.suptitle(f"Sensitivity Analysis: {method_title}", fontsize=14, y=1.02)

        plt.tight_layout()

        return fig

    def summary(self, sensitivity_results: SensitivityResults) -> pd.DataFrame:
        """
        Generate summary table of sensitivity analysis results.

        Parameters
        ----------
        sensitivity_results : SensitivityResults
            Results from sensitivity analysis

        Returns
        -------
        pd.DataFrame
            Summary statistics for each parameter

        Examples
        --------
        >>> sensitivity = pb.SensitivityAnalysis(results)
        >>> loo_results = sensitivity.leave_one_out_entities()
        >>> summary = sensitivity.summary(loo_results)
        >>> print(summary)
        """
        estimates = sensitivity_results.estimates

        summary_data = []

        for param in estimates.columns:
            param_estimates = estimates[param].dropna()

            original = self.params[param]
            mean_est = param_estimates.mean()
            std_est = param_estimates.std()
            min_est = param_estimates.min()
            max_est = param_estimates.max()

            # Max deviation from original (in standard deviations)
            max_dev = np.abs(param_estimates - original).max()
            max_dev_std = max_dev / self.std_errors[param]

            summary_data.append(
                {
                    "Parameter": param,
                    "Original": original,
                    "Mean": mean_est,
                    "Std": std_est,
                    "Min": min_est,
                    "Max": max_est,
                    "Range": max_est - min_est,
                    "Max Deviation": max_dev,
                    "Max Dev (SE)": max_dev_std,
                    "N Valid": len(param_estimates),
                }
            )

        return pd.DataFrame(summary_data)

    def _create_model(self, data: pd.DataFrame):
        """Create a new model instance with given data."""
        # Get model class and parameters
        model_class = type(self.model)

        # Reconstruct model with new data
        formula = self.model.formula_parser.formula

        # Create new model
        new_model = model_class(
            formula=formula, data=data, entity_col=self.entity_col, time_col=self.time_col
        )

        return new_model

    def _calculate_statistics(self, estimates_df: pd.DataFrame, threshold: float) -> Dict:
        """Calculate summary statistics from estimates."""
        statistics = {}

        for param in estimates_df.columns:
            param_estimates = estimates_df[param].dropna()

            original = self.params[param]
            original_se = self.std_errors[param]

            # Deviations from original
            deviations = param_estimates - original
            abs_deviations = np.abs(deviations)

            # Standardized deviations
            std_deviations = deviations / original_se

            statistics[param] = {
                "mean": param_estimates.mean(),
                "std": param_estimates.std(),
                "min": param_estimates.min(),
                "max": param_estimates.max(),
                "range": param_estimates.max() - param_estimates.min(),
                "max_abs_deviation": abs_deviations.max(),
                "mean_abs_deviation": abs_deviations.mean(),
                "max_std_deviation": np.abs(std_deviations).max(),
                "n_beyond_threshold": (np.abs(std_deviations) > threshold).sum(),
                "pct_beyond_threshold": (np.abs(std_deviations) > threshold).mean() * 100,
            }

        return statistics

    def _identify_influential_units(self, estimates_df: pd.DataFrame, threshold: float) -> List:
        """Identify influential units based on threshold."""
        influential = []

        for idx in estimates_df.index:
            # Check if any parameter estimate exceeds threshold
            is_influential = False

            for param in estimates_df.columns:
                estimate = estimates_df.loc[idx, param]

                if np.isnan(estimate):
                    continue

                original = self.params[param]
                original_se = self.std_errors[param]

                # Standardized deviation
                std_dev = np.abs(estimate - original) / original_se

                if std_dev > threshold:
                    is_influential = True
                    break

            if is_influential:
                influential.append(idx)

        return influential


def dfbetas(
    results: PanelResults, entity_col: Optional[str] = None, time_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate DFBETAS influence statistics.

    DFBETAS measures the change in parameter estimates when each observation
    is deleted, standardized by the standard error.

    Parameters
    ----------
    results : PanelResults
        Fitted panel model results
    entity_col : str, optional
        Entity column name (inferred from model if not provided)
    time_col : str, optional
        Time column name (inferred from model if not provided)

    Returns
    -------
    pd.DataFrame
        DFBETAS statistics for each observation and parameter

    Notes
    -----
    DFBETAS_i = (beta - beta_{-i}) / SE_{-i}

    where beta is the full sample estimate and beta_{-i} is the estimate
    with observation i deleted.

    Observations with |DFBETAS| > 2/sqrt(n) are considered influential.

    Examples
    --------
    >>> dfbetas_stats = pb.dfbetas(results)
    >>> influential = dfbetas_stats[dfbetas_stats.abs() > 2/np.sqrt(len(data))]
    >>> print(influential)
    """
    # This is a placeholder for future implementation
    # Full DFBETAS requires refitting N times (computationally expensive)
    raise NotImplementedError(
        "DFBETAS calculation not yet implemented. "
        "Use SensitivityAnalysis.leave_one_out_entities() for entity-level influence."
    )
