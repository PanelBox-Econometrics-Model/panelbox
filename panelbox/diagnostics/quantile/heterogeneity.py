"""
Tests for heterogeneous effects across quantiles.

This module provides statistical tests to assess whether covariate effects
vary across the conditional distribution, justifying the use of quantile
regression over mean-based methods.
"""

from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class HeterogeneityTests:
    """
    Tests for heterogeneous effects across quantiles.
    """

    def __init__(self, result):
        """
        Parameters
        ----------
        result : QuantilePanelResult
            Fitted QR model with multiple quantiles
        """
        self.result = result
        self.tau_list = sorted(result.results.keys())

        if len(self.tau_list) < 2:
            raise ValueError("Need at least 2 quantiles for heterogeneity tests")

    def test_slope_equality(
        self,
        var_idx: Optional[Union[int, List[int]]] = None,
        tau_pairs: Optional[List[Tuple[float, float]]] = None,
    ) -> "SlopeEqualityTestResult":
        """
        Test equality of slopes across quantiles.

        H0: β_j(τ₁) = β_j(τ₂) for specified variable j and quantile pairs

        Parameters
        ----------
        var_idx : int or list, optional
            Variable index(es) to test. If None, test all jointly.
        tau_pairs : list of tuples, optional
            Pairs of quantiles to compare. If None, test all adjacent.

        Returns
        -------
        SlopeEqualityTestResult
        """
        if tau_pairs is None:
            # Test adjacent quantiles
            tau_pairs = [
                (self.tau_list[i], self.tau_list[i + 1]) for i in range(len(self.tau_list) - 1)
            ]

        if var_idx is None:
            # Joint test for all variables
            n_params = len(self.result.results[self.tau_list[0]].params)
            var_idx = list(range(n_params))
        elif np.isscalar(var_idx):
            var_idx = [var_idx]

        # Collect test statistics
        test_stats = []
        dof_total = 0

        for tau1, tau2 in tau_pairs:
            res1 = self.result.results[tau1]
            res2 = self.result.results[tau2]

            # Difference in coefficients
            diff = res1.params[var_idx] - res2.params[var_idx]

            # Covariance of difference (assuming independence)
            V1 = res1.cov_matrix[np.ix_(var_idx, var_idx)]
            V2 = res2.cov_matrix[np.ix_(var_idx, var_idx)]
            V_diff = V1 + V2

            # Wald statistic
            try:
                wald = diff @ np.linalg.inv(V_diff) @ diff
            except np.linalg.LinAlgError:
                wald = diff @ np.linalg.pinv(V_diff) @ diff

            test_stats.append(wald)
            dof_total += len(var_idx)

        # Combined test statistic
        combined_stat = np.sum(test_stats)
        p_value = 1 - stats.chi2.cdf(combined_stat, dof_total)

        return SlopeEqualityTestResult(
            statistic=combined_stat,
            p_value=p_value,
            df=dof_total,
            tau_pairs=tau_pairs,
            var_idx=var_idx,
            individual_stats=test_stats,
        )

    def test_joint_equality(
        self, tau_subset: Optional[List[float]] = None
    ) -> "JointEqualityTestResult":
        """
        Joint test that all slope coefficients are equal across quantiles.

        H0: β(τ) = β for all τ (no heterogeneity)

        Parameters
        ----------
        tau_subset : list, optional
            Subset of quantiles to test. If None, use all.

        Returns
        -------
        JointEqualityTestResult
        """
        if tau_subset is None:
            tau_subset = self.tau_list

        # Stack coefficients
        coef_matrix = np.array([self.result.results[tau].params for tau in tau_subset])

        # Use median as reference
        ref_idx = len(tau_subset) // 2
        ref_coef = coef_matrix[ref_idx]

        # Test statistic
        test_stat = 0
        for i, tau in enumerate(tau_subset):
            if i == ref_idx:
                continue

            diff = coef_matrix[i] - ref_coef
            V_sum = (
                self.result.results[tau_subset[i]].cov_matrix
                + self.result.results[tau_subset[ref_idx]].cov_matrix
            )

            try:
                test_stat += diff @ np.linalg.inv(V_sum) @ diff
            except:
                test_stat += diff @ np.linalg.pinv(V_sum) @ diff

        # Degrees of freedom
        n_params = len(ref_coef)
        df = (len(tau_subset) - 1) * n_params

        # P-value
        p_value = 1 - stats.chi2.cdf(test_stat, df)

        return JointEqualityTestResult(
            statistic=test_stat,
            p_value=p_value,
            df=df,
            tau_subset=tau_subset,
            coef_matrix=coef_matrix,
        )

    def test_monotonicity(self, var_idx: int) -> "MonotonicityTestResult":
        """
        Test if coefficient is monotonic in τ.

        Tests both increasing and decreasing patterns.

        Parameters
        ----------
        var_idx : int
            Variable index to test

        Returns
        -------
        MonotonicityTestResult
        """
        # Extract coefficient path
        coef_path = np.array([self.result.results[tau].params[var_idx] for tau in self.tau_list])

        # Check monotonicity
        is_increasing = all(coef_path[i] <= coef_path[i + 1] for i in range(len(coef_path) - 1))
        is_decreasing = all(coef_path[i] >= coef_path[i + 1] for i in range(len(coef_path) - 1))

        # Test against null of no systematic pattern
        # Using Spearman correlation with quantile levels
        correlation, p_value = stats.spearmanr(self.tau_list, coef_path)

        return MonotonicityTestResult(
            correlation=correlation,
            p_value=p_value,
            is_increasing=is_increasing,
            is_decreasing=is_decreasing,
            coef_path=coef_path,
            tau_list=self.tau_list,
            var_idx=var_idx,
        )

    def interquantile_range_test(self) -> Tuple[float, float]:
        """
        Test if interquantile range varies with covariates.

        Compares Q(0.75) - Q(0.25) regression to constant model.
        """
        # Ensure we have the needed quantiles
        if 0.25 not in self.tau_list or 0.75 not in self.tau_list:
            raise ValueError("Need quantiles 0.25 and 0.75 for IQR test")

        # IQR coefficients
        coef_75 = self.result.results[0.75].params
        coef_25 = self.result.results[0.25].params
        iqr_coef = coef_75 - coef_25

        # Test if IQR coefficients are jointly zero (except intercept)
        test_idx = list(range(1, len(iqr_coef)))  # Exclude intercept
        iqr_test = iqr_coef[test_idx]

        # Approximate covariance
        V_75 = self.result.results[0.75].cov_matrix[np.ix_(test_idx, test_idx)]
        V_25 = self.result.results[0.25].cov_matrix[np.ix_(test_idx, test_idx)]
        V_iqr = V_75 + V_25

        # Wald test
        try:
            test_stat = iqr_test @ np.linalg.inv(V_iqr) @ iqr_test
        except:
            test_stat = iqr_test @ np.linalg.pinv(V_iqr) @ iqr_test

        df = len(test_idx)
        p_value = 1 - stats.chi2.cdf(test_stat, df)

        print("\nInterquantile Range Test")
        print("=" * 50)
        print("H0: IQR does not vary with covariates")
        print(f"Test Statistic: {test_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        if p_value < 0.05:
            print("Conclusion: Reject H0 - heteroskedasticity detected")
        else:
            print("Conclusion: Cannot reject H0 - constant IQR")

        return test_stat, p_value

    def plot_coefficient_paths(
        self, var_names: Optional[List[str]] = None, confidence_bands: bool = True
    ):
        """
        Plot how coefficients change across quantiles.

        Parameters
        ----------
        var_names : list, optional
            Variable names for labels
        confidence_bands : bool
            Whether to include 95% confidence bands
        """
        n_params = len(self.result.results[self.tau_list[0]].params)

        if var_names is None:
            var_names = [f"β{i+1}" for i in range(n_params)]

        # Create subplots
        n_cols = 2
        n_rows = (n_params + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(n_params):
            ax = axes[i]

            # Extract coefficients and standard errors
            coefs = []
            lower_bounds = []
            upper_bounds = []

            for tau in self.tau_list:
                result_tau = self.result.results[tau]
                coef = result_tau.params[i]
                coefs.append(coef)

                if confidence_bands:
                    if hasattr(result_tau, "bse"):
                        se = result_tau.bse[i]
                    else:
                        se = np.sqrt(result_tau.cov_matrix[i, i])
                    lower_bounds.append(coef - 1.96 * se)
                    upper_bounds.append(coef + 1.96 * se)

            # Plot coefficient path
            ax.plot(self.tau_list, coefs, "o-", linewidth=2, markersize=6, label=var_names[i])

            # Add confidence bands
            if confidence_bands:
                ax.fill_between(self.tau_list, lower_bounds, upper_bounds, alpha=0.3)

            # Add reference line at mean effect (optional)
            mean_coef = np.mean(coefs)
            ax.axhline(mean_coef, color="red", linestyle="--", alpha=0.5, label="Mean effect")

            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel("Coefficient Value")
            ax.set_title(f"{var_names[i]} across Quantiles")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Coefficient Heterogeneity Across Quantiles", fontsize=14)
        plt.tight_layout()
        return fig


class SlopeEqualityTestResult:
    """Results for slope equality test."""

    def __init__(
        self,
        statistic: float,
        p_value: float,
        df: int,
        tau_pairs: List[Tuple[float, float]],
        var_idx: List[int],
        individual_stats: List[float],
    ):
        self.statistic = statistic
        self.p_value = p_value
        self.df = df
        self.tau_pairs = tau_pairs
        self.var_idx = var_idx
        self.individual_stats = individual_stats

    def summary(self):
        """Print test summary."""
        print("\nSlope Equality Test")
        print("=" * 50)
        print("H0: Slopes are equal across quantile pairs")
        print(f"Variables tested: {self.var_idx}")
        print(f"Quantile pairs: {self.tau_pairs}")
        print(f"\nOverall Test Statistic: {self.statistic:.4f}")
        print(f"Degrees of Freedom: {self.df}")
        print(f"P-value: {self.p_value:.4f}")

        if len(self.individual_stats) > 1:
            print("\nIndividual pair tests:")
            for (tau1, tau2), stat in zip(self.tau_pairs, self.individual_stats):
                print(f"  τ={tau1} vs τ={tau2}: χ² = {stat:.4f}")

        if self.p_value < 0.05:
            print("\nConclusion: REJECT equality (heterogeneous effects)")
        else:
            print("\nConclusion: Cannot reject equality")


class JointEqualityTestResult:
    """Results for joint equality test."""

    def __init__(
        self,
        statistic: float,
        p_value: float,
        df: int,
        tau_subset: List[float],
        coef_matrix: np.ndarray,
    ):
        self.statistic = statistic
        self.p_value = p_value
        self.df = df
        self.tau_subset = tau_subset
        self.coef_matrix = coef_matrix

    def summary(self):
        """Print test summary."""
        print("\nJoint Equality Test")
        print("=" * 50)
        print("H0: All coefficients are equal across quantiles")
        print(f"Quantiles tested: {self.tau_subset}")
        print(f"\nTest Statistic: {self.statistic:.4f}")
        print(f"Degrees of Freedom: {self.df}")
        print(f"P-value: {self.p_value:.4f}")

        if self.p_value < 0.05:
            print("\nConclusion: REJECT H0 - significant heterogeneity detected")
            print("Quantile regression is justified over mean regression.")
        else:
            print("\nConclusion: Cannot reject H0 - no significant heterogeneity")
            print("Mean regression may be sufficient.")


class MonotonicityTestResult:
    """Results for monotonicity test."""

    def __init__(
        self,
        correlation: float,
        p_value: float,
        is_increasing: bool,
        is_decreasing: bool,
        coef_path: np.ndarray,
        tau_list: List[float],
        var_idx: int,
    ):
        self.correlation = correlation
        self.p_value = p_value
        self.is_increasing = is_increasing
        self.is_decreasing = is_decreasing
        self.coef_path = coef_path
        self.tau_list = tau_list
        self.var_idx = var_idx

    def summary(self):
        """Print test summary."""
        print(f"\nMonotonicity Test for Variable {self.var_idx + 1}")
        print("=" * 50)
        print(f"Spearman correlation: {self.correlation:.4f}")
        print(f"P-value: {self.p_value:.4f}")

        if self.is_increasing:
            print("Pattern: Strictly Increasing")
        elif self.is_decreasing:
            print("Pattern: Strictly Decreasing")
        else:
            print("Pattern: Non-monotonic")

        if abs(self.correlation) > 0.7 and self.p_value < 0.05:
            direction = "increasing" if self.correlation > 0 else "decreasing"
            print(f"\nConclusion: Strong {direction} trend detected")
        else:
            print("\nConclusion: No clear monotonic pattern")

    def plot(self):
        """Visualize monotonicity pattern."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Coefficient path
        ax1.plot(self.tau_list, self.coef_path, "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("Quantile (τ)")
        ax1.set_ylabel(f"Coefficient β{self.var_idx+1}")
        ax1.set_title(f"Coefficient Path: Variable {self.var_idx+1}")
        ax1.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(self.tau_list, self.coef_path, 1)
        p = np.poly1d(z)
        ax1.plot(
            self.tau_list,
            p(self.tau_list),
            "r--",
            alpha=0.5,
            label=f"Linear Trend (ρ={self.correlation:.3f})",
        )
        ax1.legend()

        # First differences
        diffs = np.diff(self.coef_path)
        tau_mid = [
            (self.tau_list[i] + self.tau_list[i + 1]) / 2 for i in range(len(self.tau_list) - 1)
        ]

        ax2.bar(tau_mid, diffs, width=0.05, alpha=0.7)
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_xlabel("Quantile (τ)")
        ax2.set_ylabel("First Difference")
        ax2.set_title("Changes in Coefficient")
        ax2.grid(True, alpha=0.3)

        # Add pattern text
        pattern = "Pattern: "
        if self.is_increasing:
            pattern += "Strictly Increasing"
        elif self.is_decreasing:
            pattern += "Strictly Decreasing"
        else:
            pattern += "Non-monotonic"

        ax1.text(
            0.05,
            0.95,
            pattern,
            transform=ax1.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle("Monotonicity Analysis", fontsize=14)
        plt.tight_layout()
        return fig
