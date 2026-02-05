"""
Jackknife inference for panel data models.

This module implements jackknife resampling for panel data, providing
alternative estimates of bias and variance. The jackknife is particularly
useful for small samples and provides influence diagnostics.

References
----------
Efron, B., & Tibshirani, R. J. (1994). An Introduction to the Bootstrap.
    Chapman and Hall/CRC.
Shao, J., & Tu, D. (1995). The Jackknife and Bootstrap.
    Springer Science & Business Media.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from panelbox.core.results import PanelResults


@dataclass
class JackknifeResults:
    """
    Container for jackknife results.

    Attributes
    ----------
    jackknife_estimates : pd.DataFrame
        Parameter estimates for each jackknife sample (N x n_params)
    original_estimates : pd.Series
        Original parameter estimates
    jackknife_mean : pd.Series
        Mean of jackknife estimates
    jackknife_bias : pd.Series
        Jackknife bias estimates
    jackknife_se : pd.Series
        Jackknife standard errors
    influence : pd.DataFrame
        Influence values for each entity
    n_jackknife : int
        Number of jackknife samples (entities)
    """

    jackknife_estimates: pd.DataFrame
    original_estimates: pd.Series
    jackknife_mean: pd.Series
    jackknife_bias: pd.Series
    jackknife_se: pd.Series
    influence: pd.DataFrame
    n_jackknife: int

    def summary(self) -> str:
        """Generate summary of jackknife results."""
        lines = []
        lines.append("Jackknife Results")
        lines.append("=" * 70)
        lines.append(f"Number of jackknife samples: {self.n_jackknife}")
        lines.append("")

        lines.append("Parameter Estimates and Bias:")
        lines.append("-" * 70)
        lines.append(
            f"{'Parameter':<15} {'Original':>12} {'Jackknife':>12} " f"{'Bias':>12} {'SE (JK)':>12}"
        )
        lines.append("-" * 70)

        for param in self.original_estimates.index:
            lines.append(
                f"{param:<15} {self.original_estimates[param]:>12.6f} "
                f"{self.jackknife_mean[param]:>12.6f} "
                f"{self.jackknife_bias[param]:>12.6f} "
                f"{self.jackknife_se[param]:>12.6f}"
            )

        lines.append("")
        lines.append("Influential Entities:")
        lines.append("-" * 70)

        # Find most influential entities (highest absolute influence)
        max_influence = self.influence.abs().max(axis=1)
        top_influential = max_influence.nlargest(5)

        if len(top_influential) > 0:
            lines.append(f"{'Entity':>10} {'Max Influence':>15}")
            lines.append("-" * 70)
            for entity, infl in top_influential.items():
                lines.append(f"{entity:>10} {infl:>15.6f}")
        else:
            lines.append("No influential entities detected")

        return "\n".join(lines)


class PanelJackknife:
    """
    Jackknife inference for panel data models.

    The jackknife resampling method systematically leaves out one entity
    at a time and re-estimates the model. This provides estimates of:
    - Bias in parameter estimates
    - Standard errors
    - Influence of individual entities

    Parameters
    ----------
    results : PanelResults
        Fitted model results to jackknife
    verbose : bool, default=True
        Whether to print progress information

    Attributes
    ----------
    jackknife_results_ : JackknifeResults
        Jackknife results after calling run()

    Examples
    --------
    >>> import panelbox as pb
    >>> import pandas as pd
    >>>
    >>> # Fit model
    >>> data = pd.read_csv('panel_data.csv')
    >>> fe = pb.FixedEffects("y ~ x1 + x2", data, "entity_id", "time")
    >>> results = fe.fit()
    >>>
    >>> # Jackknife inference
    >>> jackknife = pb.PanelJackknife(results)
    >>> jk_results = jackknife.run()
    >>>
    >>> # View results
    >>> print(jk_results.summary())
    >>>
    >>> # Get bias-corrected estimates
    >>> bias_corrected = jackknife.bias_corrected_estimates()
    >>> print(bias_corrected)

    Notes
    -----
    - Jackknife is less computationally intensive than bootstrap
    - Provides good estimates for variance and bias
    - Each jackknife sample excludes one entity (all its time periods)
    - For N entities, requires N model re-estimations
    """

    def __init__(self, results: PanelResults, verbose: bool = True):
        self.results = results
        self.verbose = verbose

        # Extract model information
        self.model = results._model
        assert self.model is not None, "Results must have a model reference for jackknife"
        self.formula = results.formula
        self.entity_col = self.model.data.entity_col
        self.time_col = self.model.data.time_col

        # Get original data
        self.data = self.model.data.data

        # Get entities
        self.entities = sorted(self.data[self.entity_col].unique())
        self.n_entities = len(self.entities)

        # Results storage
        self.jackknife_results_: Optional[JackknifeResults] = None

    def run(self) -> JackknifeResults:
        """
        Run jackknife procedure.

        Returns
        -------
        jackknife_results : JackknifeResults
            Jackknife results containing estimates, bias, and standard errors

        Notes
        -----
        The jackknife procedure:

        1. For each entity i:
           - Remove entity i from dataset
           - Re-estimate model on remaining N-1 entities
           - Store parameter estimates

        2. Compute jackknife statistics:
           - Mean of jackknife estimates
           - Bias: (N-1) * (jackknife_mean - original)
           - SE: sqrt((N-1)/N * sum((theta_i - mean)^2))
           - Influence: (N-1) * (original - theta_i)
        """
        if self.verbose:
            print("Starting jackknife procedure...")
            print(f"Total entities: {self.n_entities}")
            print("")

        # Storage for jackknife estimates
        jackknife_estimates = []
        failed_samples = []

        # Original estimates
        original_estimates = self.results.params

        # Perform leave-one-out
        for i, entity in enumerate(self.entities, 1):
            if self.verbose:
                print(f"Jackknife sample {i}/{self.n_entities}: " f"Excluding entity {entity}")

            try:
                # Remove entity i
                jackknife_data = self.data[self.data[self.entity_col] != entity].copy()

                # Re-estimate model
                model_class = type(self.model)
                jackknife_model = model_class(
                    self.formula, jackknife_data, self.entity_col, self.time_col
                )
                jackknife_result = jackknife_model.fit(cov_type=self.results.cov_type)

                # Store estimates
                jackknife_estimates.append(
                    {"entity_excluded": entity, **jackknife_result.params.to_dict()}
                )

            except Exception as e:
                warnings.warn(f"Jackknife sample {i} (entity {entity}) failed: {str(e)}")
                failed_samples.append(entity)
                continue

        # Check if we have any successful samples
        if not jackknife_estimates:
            raise RuntimeError("All jackknife samples failed")

        if self.verbose and failed_samples:
            print(f"\nWarning: {len(failed_samples)} samples failed")
            print(f"Successfully completed: {len(jackknife_estimates)}/{self.n_entities}")

        # Convert to DataFrame
        jackknife_df = pd.DataFrame(jackknife_estimates)
        entity_col_jk = jackknife_df["entity_excluded"]
        jackknife_df = jackknife_df.drop("entity_excluded", axis=1)

        # Compute jackknife statistics
        N = len(jackknife_estimates)

        # Mean of jackknife estimates
        jackknife_mean = jackknife_df.mean()

        # Jackknife bias: (N-1) * (mean_jackknife - theta_original)
        jackknife_bias = (N - 1) * (jackknife_mean - original_estimates)

        # Jackknife standard error: sqrt((N-1)/N * sum((theta_i - mean)^2))
        deviations = jackknife_df - jackknife_mean
        jackknife_variance = ((N - 1) / N) * (deviations**2).sum()
        jackknife_se = np.sqrt(jackknife_variance)

        # Influence: (N-1) * (theta_original - theta_(-i))
        influence_df = pd.DataFrame(
            (N - 1) * (original_estimates.values - jackknife_df.values),
            columns=original_estimates.index,
            index=entity_col_jk,
        )

        # Create results object
        self.jackknife_results_ = JackknifeResults(
            jackknife_estimates=jackknife_df,
            original_estimates=original_estimates,
            jackknife_mean=jackknife_mean,
            jackknife_bias=jackknife_bias,
            jackknife_se=jackknife_se,
            influence=influence_df,
            n_jackknife=N,
        )

        if self.verbose:
            print("\nJackknife Complete!")
            print(f"Successful samples: {N}/{self.n_entities}")

        return self.jackknife_results_

    def bias_corrected_estimates(self) -> pd.Series:
        """
        Compute bias-corrected parameter estimates.

        Returns
        -------
        bias_corrected : pd.Series
            Bias-corrected estimates: original - bias

        Raises
        ------
        RuntimeError
            If run() has not been called yet

        Notes
        -----
        Bias correction formula:
            theta_corrected = theta_original - bias
            where bias = (N-1) * (mean_jackknife - theta_original)

        This is equivalent to:
            theta_corrected = N * theta_original - (N-1) * mean_jackknife
        """
        if self.jackknife_results_ is None:
            raise RuntimeError("Must call run() before bias_corrected_estimates()")

        bias_corrected = (
            self.jackknife_results_.original_estimates - self.jackknife_results_.jackknife_bias
        )

        return bias_corrected

    def confidence_intervals(self, alpha: float = 0.05, method: str = "normal") -> pd.DataFrame:
        """
        Compute confidence intervals using jackknife standard errors.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (e.g., 0.05 for 95% CI)
        method : {'normal', 'percentile'}, default='normal'
            Method for computing confidence intervals:

            - 'normal': Normal approximation using jackknife SE
            - 'percentile': Percentile method using jackknife distribution

        Returns
        -------
        ci : pd.DataFrame
            Confidence intervals with columns 'lower' and 'upper'

        Raises
        ------
        RuntimeError
            If run() has not been called yet
        """
        if self.jackknife_results_ is None:
            raise RuntimeError("Must call run() before confidence_intervals()")

        if method == "normal":
            # Normal approximation
            from scipy import stats

            z = stats.norm.ppf(1 - alpha / 2)

            lower = (
                self.jackknife_results_.original_estimates
                - z * self.jackknife_results_.jackknife_se
            )
            upper = (
                self.jackknife_results_.original_estimates
                + z * self.jackknife_results_.jackknife_se
            )

        elif method == "percentile":
            # Percentile method
            lower = self.jackknife_results_.jackknife_estimates.quantile(alpha / 2)
            upper = self.jackknife_results_.jackknife_estimates.quantile(1 - alpha / 2)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'normal' or 'percentile'")

        ci = pd.DataFrame({"lower": lower, "upper": upper})

        return ci

    def influential_entities(self, threshold: float = 2.0, metric: str = "max") -> pd.DataFrame:
        """
        Identify influential entities based on jackknife influence.

        Parameters
        ----------
        threshold : float, default=2.0
            Threshold for influence (in units of mean absolute influence)
        metric : {'max', 'mean', 'sum'}, default='max'
            How to aggregate influence across parameters:

            - 'max': Maximum absolute influence across parameters
            - 'mean': Mean absolute influence across parameters
            - 'sum': Sum of absolute influences across parameters

        Returns
        -------
        influential : pd.DataFrame
            DataFrame of influential entities with their influence measures

        Raises
        ------
        RuntimeError
            If run() has not been called yet
        """
        if self.jackknife_results_ is None:
            raise RuntimeError("Must call run() before influential_entities()")

        influence = self.jackknife_results_.influence

        # Compute aggregate influence
        if metric == "max":
            aggregate_influence = influence.abs().max(axis=1)
        elif metric == "mean":
            aggregate_influence = influence.abs().mean(axis=1)
        elif metric == "sum":
            aggregate_influence = influence.abs().sum(axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'max', 'mean', or 'sum'")

        # Determine threshold
        mean_influence = aggregate_influence.mean()
        influence_threshold = threshold * mean_influence

        # Filter influential entities
        influential_mask = aggregate_influence > influence_threshold
        influential = pd.DataFrame(
            {
                "entity": aggregate_influence[influential_mask].index,
                "influence": aggregate_influence[influential_mask].values,
                "threshold": influence_threshold,
            }
        )

        return influential

    def summary(self) -> str:
        """
        Generate summary of jackknife results.

        Returns
        -------
        summary_str : str
            Formatted summary string

        Raises
        ------
        RuntimeError
            If run() has not been called yet
        """
        if self.jackknife_results_ is None:
            raise RuntimeError("Must call run() before summary()")

        return self.jackknife_results_.summary()
