"""
Outlier detection and leverage diagnostics for panel data models.

This module implements various methods for detecting outliers and high-leverage
points in panel data, including:
- Univariate methods (IQR, Z-score)
- Multivariate methods (Mahalanobis distance)
- Regression diagnostics (standardized residuals, studentized residuals)
- Leverage diagnostics (hat values)

References
----------
Cook, R. D., & Weisberg, S. (1982). Residuals and Influence in Regression.
    Chapman and Hall.
Rousseeuw, P. J., & Leroy, A. M. (1987). Robust Regression and Outlier Detection.
    John Wiley & Sons.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.core.results import PanelResults


@dataclass
class OutlierResults:
    """
    Container for outlier detection results.

    Attributes
    ----------
    outliers : pd.DataFrame
        DataFrame with outlier flags and diagnostic statistics
    method : str
        Method used for detection
    threshold : float
        Threshold used for detection
    n_outliers : int
        Number of outliers detected
    """

    outliers: pd.DataFrame
    method: str
    threshold: float
    n_outliers: int

    def summary(self) -> str:
        """Generate summary of outlier detection."""
        lines = []
        lines.append("Outlier Detection Results")
        lines.append("=" * 70)
        lines.append(f"Method: {self.method}")
        lines.append(f"Threshold: {self.threshold}")
        lines.append(f"Outliers detected: {self.n_outliers} / {len(self.outliers)}")
        lines.append(f"Percentage: {100 * self.n_outliers / len(self.outliers):.2f}%")

        if self.n_outliers > 0:
            lines.append("")
            lines.append("Top 10 outliers:")
            lines.append("-" * 70)
            top_outliers = self.outliers[self.outliers["is_outlier"]].head(10)
            lines.append(top_outliers.to_string())

        return "\n".join(lines)


class OutlierDetector:
    """
    Outlier detection for panel data models.

    This class provides various methods for detecting outliers and
    high-leverage points in panel data regression models.

    Parameters
    ----------
    results : PanelResults
        Fitted model results to analyze
    verbose : bool, default=True
        Whether to print progress information

    Attributes
    ----------
    outlier_results_ : OutlierResults
        Results after calling detection methods

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
    >>> # Detect outliers
    >>> detector = pb.OutlierDetector(results)
    >>>
    >>> # Univariate methods
    >>> outliers_iqr = detector.detect_outliers_univariate(method='iqr')
    >>> outliers_zscore = detector.detect_outliers_univariate(method='zscore')
    >>>
    >>> # Multivariate method
    >>> outliers_mahal = detector.detect_outliers_multivariate()
    >>>
    >>> # Regression diagnostics
    >>> outliers_resid = detector.detect_outliers_residuals(method='standardized')
    >>>
    >>> # Leverage points
    >>> leverage = detector.detect_leverage_points()
    >>>
    >>> # Plot diagnostics
    >>> detector.plot_diagnostics()

    Notes
    -----
    - Different methods may identify different outliers
    - Combine multiple methods for robust detection
    - Outliers should be investigated, not automatically removed
    """

    def __init__(self, results: PanelResults, verbose: bool = True):
        self.results = results
        self.verbose = verbose

        # Extract model information
        self.model = results._model
        self.data = self.model.data.data

        # Get entity and time columns
        self.entity_col = self.model.data.entity_col
        self.time_col = self.model.data.time_col

        # Results storage
        self.outlier_results_: Optional[OutlierResults] = None

    def detect_outliers_univariate(
        self, variable: Optional[str] = None, method: str = "iqr", threshold: float = 1.5
    ) -> OutlierResults:
        """
        Detect outliers using univariate methods.

        Parameters
        ----------
        variable : str, optional
            Variable to check for outliers. If None, uses residuals.
        method : {'iqr', 'zscore'}, default='iqr'
            Detection method:

            - 'iqr': Interquartile range method (Q1 - k*IQR, Q3 + k*IQR)
            - 'zscore': Z-score method (|z| > threshold)
        threshold : float, default=1.5
            Threshold parameter:

            - For IQR: multiplier for IQR (typically 1.5 or 3.0)
            - For Z-score: threshold for |z| (typically 2.5 or 3.0)

        Returns
        -------
        outlier_results : OutlierResults
            Outlier detection results
        """
        if variable is None:
            # Use residuals
            values = self.results.resid
            var_name = "residuals"
        else:
            values = self.data[variable].values
            var_name = variable

        if method == "iqr":
            # IQR method
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            is_outlier = (values < lower_bound) | (values > upper_bound)
            distance = np.minimum(np.abs(values - lower_bound), np.abs(values - upper_bound))

            method_name = f"IQR (k={threshold})"

        elif method == "zscore":
            # Z-score method
            mean = np.mean(values)
            std = np.std(values)
            z_scores = (values - mean) / std

            is_outlier = np.abs(z_scores) > threshold
            distance = np.abs(z_scores)

            method_name = f"Z-score (threshold={threshold})"

        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")

        # Create results DataFrame
        outliers_df = pd.DataFrame(
            {
                "entity": self.data[self.entity_col].values,
                "time": self.data[self.time_col].values,
                "value": values,
                "is_outlier": is_outlier,
                "distance": distance,
            }
        )

        n_outliers = is_outlier.sum()

        self.outlier_results_ = OutlierResults(
            outliers=outliers_df,
            method=f"{method_name} on {var_name}",
            threshold=threshold,
            n_outliers=n_outliers,
        )

        if self.verbose:
            print(f"Detected {n_outliers} outliers using {method_name}")

        return self.outlier_results_

    def detect_outliers_multivariate(self, threshold: float = 3.0) -> OutlierResults:
        """
        Detect outliers using Mahalanobis distance.

        Parameters
        ----------
        threshold : float, default=3.0
            Threshold for Mahalanobis distance (in units of chi-square quantile)

        Returns
        -------
        outlier_results : OutlierResults
            Outlier detection results

        Notes
        -----
        Mahalanobis distance accounts for correlations between variables
        and is more appropriate for multivariate outlier detection than
        univariate methods.
        """
        # Get design matrix (X)
        from patsy import dmatrix

        formula_rhs = self.results.formula.split("~")[1].strip()
        X = dmatrix(formula_rhs, self.data, return_type="dataframe")

        # Compute Mahalanobis distance
        mean = X.mean().values
        # Use covariance matrix, handling potential singularity
        try:
            cov = np.cov(X.values.T)
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            warnings.warn("Covariance matrix is singular, using pseudo-inverse")
            cov_inv = np.linalg.pinv(np.cov(X.values.T))

        diff = X.values - mean
        mahal_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

        # Threshold based on chi-square distribution
        df = X.shape[1]
        chi2_threshold = stats.chi2.ppf(0.975, df)  # 97.5th percentile
        threshold_value = np.sqrt(chi2_threshold) * threshold

        is_outlier = mahal_dist > threshold_value

        # Create results DataFrame
        outliers_df = pd.DataFrame(
            {
                "entity": self.data[self.entity_col].values,
                "time": self.data[self.time_col].values,
                "mahalanobis_distance": mahal_dist,
                "is_outlier": is_outlier,
                "distance": mahal_dist,
            }
        )

        n_outliers = is_outlier.sum()

        self.outlier_results_ = OutlierResults(
            outliers=outliers_df,
            method=f"Mahalanobis distance",
            threshold=threshold_value,
            n_outliers=n_outliers,
        )

        if self.verbose:
            print(f"Detected {n_outliers} outliers using Mahalanobis distance")

        return self.outlier_results_

    def detect_outliers_residuals(
        self, method: str = "standardized", threshold: float = 2.5
    ) -> OutlierResults:
        """
        Detect outliers using residual-based methods.

        Parameters
        ----------
        method : {'standardized', 'studentized'}, default='standardized'
            Type of residuals:

            - 'standardized': Residuals / sqrt(MSE)
            - 'studentized': Residuals / sqrt(MSE * (1 - h_ii))
        threshold : float, default=2.5
            Threshold for absolute residual value

        Returns
        -------
        outlier_results : OutlierResults
            Outlier detection results
        """
        residuals = self.results.resid

        if method == "standardized":
            # Standardized residuals: r / sqrt(MSE)
            mse = np.sum(residuals**2) / self.results.df_resid
            std_residuals = residuals / np.sqrt(mse)
            is_outlier = np.abs(std_residuals) > threshold

            outliers_df = pd.DataFrame(
                {
                    "entity": self.data[self.entity_col].values,
                    "time": self.data[self.time_col].values,
                    "residual": residuals,
                    "standardized_residual": std_residuals,
                    "is_outlier": is_outlier,
                    "distance": np.abs(std_residuals),
                }
            )

        elif method == "studentized":
            # Studentized residuals require leverage values
            # For panel data, this is approximate
            mse = np.sum(residuals**2) / self.results.df_resid

            # Approximate leverage (would need full hat matrix for exact)
            n = len(residuals)
            k = len(self.results.params)
            approx_leverage = k / n  # Average leverage

            studentized_residuals = residuals / np.sqrt(mse * (1 - approx_leverage))
            is_outlier = np.abs(studentized_residuals) > threshold

            outliers_df = pd.DataFrame(
                {
                    "entity": self.data[self.entity_col].values,
                    "time": self.data[self.time_col].values,
                    "residual": residuals,
                    "studentized_residual": studentized_residuals,
                    "is_outlier": is_outlier,
                    "distance": np.abs(studentized_residuals),
                }
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        n_outliers = is_outlier.sum()

        self.outlier_results_ = OutlierResults(
            outliers=outliers_df,
            method=f"{method.capitalize()} residuals",
            threshold=threshold,
            n_outliers=n_outliers,
        )

        if self.verbose:
            print(f"Detected {n_outliers} outliers using {method} residuals")

        return self.outlier_results_

    def detect_leverage_points(self, threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Detect high-leverage points.

        Parameters
        ----------
        threshold : float, optional
            Threshold for leverage. If None, uses 2*k/n (common rule of thumb)
            where k is number of parameters and n is number of observations

        Returns
        -------
        leverage_df : pd.DataFrame
            DataFrame with leverage values and flags

        Notes
        -----
        For panel data with fixed effects, exact leverage calculation
        requires the full hat matrix, which can be memory-intensive.
        This implementation provides an approximation.
        """
        n = len(self.results.resid)
        k = len(self.results.params)

        if threshold is None:
            threshold = 2 * k / n

        # For panel FE models, this is an approximation
        # True leverage would require hat matrix: H = X(X'X)^-1 X'
        # We approximate using distance from means

        from patsy import dmatrix

        formula_rhs = self.results.formula.split("~")[1].strip()
        X = dmatrix(formula_rhs, self.data, return_type="dataframe")

        # Approximate leverage using Mahalanobis distance
        mean = X.mean().values
        try:
            cov = np.cov(X.values.T)
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            warnings.warn("Using pseudo-inverse for leverage calculation")
            cov_inv = np.linalg.pinv(np.cov(X.values.T))

        diff = X.values - mean
        mahal_dist_sq = np.sum(diff @ cov_inv * diff, axis=1)

        # Convert to approximate leverage (0 to 1 scale)
        leverage = mahal_dist_sq / (n - 1) + 1 / n

        is_high_leverage = leverage > threshold

        leverage_df = pd.DataFrame(
            {
                "entity": self.data[self.entity_col].values,
                "time": self.data[self.time_col].values,
                "leverage": leverage,
                "is_high_leverage": is_high_leverage,
            }
        )

        n_high_leverage = is_high_leverage.sum()

        if self.verbose:
            print(f"Detected {n_high_leverage} high-leverage points (threshold={threshold:.4f})")

        return leverage_df

    def plot_diagnostics(self, save_path: Optional[str] = None):
        """
        Plot diagnostic plots for outlier detection.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot. If None, displays the plot.

        Raises
        ------
        ImportError
            If matplotlib is not installed
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. " "Install with: pip install matplotlib"
            )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        residuals = self.results.resid
        fitted = self.results.fittedvalues

        # Plot 1: Residuals vs Fitted
        ax1 = axes[0, 0]
        ax1.scatter(fitted, residuals, alpha=0.5, s=20)
        ax1.axhline(y=0, color="r", linestyle="--", linewidth=1)
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Fitted")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Q-Q plot
        ax2 = axes[0, 1]
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title("Normal Q-Q Plot")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Scale-Location (sqrt of standardized residuals vs fitted)
        ax3 = axes[1, 0]
        mse = np.sum(residuals**2) / self.results.df_resid
        std_residuals = residuals / np.sqrt(mse)
        ax3.scatter(fitted, np.sqrt(np.abs(std_residuals)), alpha=0.5, s=20)
        ax3.set_xlabel("Fitted Values")
        ax3.set_ylabel("√|Standardized Residuals|")
        ax3.set_title("Scale-Location Plot")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Histogram of residuals
        ax4 = axes[1, 1]
        ax4.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor="black")

        # Overlay normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax4.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal")
        ax4.set_xlabel("Residuals")
        ax4.set_ylabel("Density")
        ax4.set_title("Distribution of Residuals")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.verbose:
                print(f"Plot saved to {save_path}")
        else:
            plt.show()
