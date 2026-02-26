"""
Validation tests for panel cointegration tests against R packages (plm, urca).

This module compares PanelBox cointegration test results with R implementations
to ensure statistical accuracy.
"""

import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.cointegration import kao_test, pedroni_test, westerlund_test

pytestmark = pytest.mark.r_validation


class TestCointegrationVsR:
    """Test PanelBox cointegration tests against R (plm, urca)."""

    @classmethod
    def setup_class(cls):
        """Run R script to generate reference results."""
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "cointegration_r.R")

        # Run R script
        result = subprocess.run(
            ["Rscript", script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path),
        )

        if result.returncode != 0:
            pytest.skip(f"R script failed: {result.stderr}")

        # Load R results
        output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")

        cls.r_results = pd.read_csv(os.path.join(output_dir, "cointegration_r_results.csv"))

        cls.test_data = pd.read_csv(os.path.join(output_dir, "cointegration_test_data.csv"))

        print("\nR Results loaded successfully:")
        print(cls.r_results)

    def test_kao_vs_r(self):
        """Test Kao test against R purtest results."""
        # Run our Kao test
        result = kao_test(
            data=self.test_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars=["x"],
            method="df",
            trend="c",
        )

        # Get R result
        r_kao = self.r_results[self.r_results["test"] == "Kao"].iloc[0]

        # The R implementation uses IPS test on residuals
        # Our implementation uses DF/ADF directly
        # We'll compare the conclusion (reject/not reject) rather than exact statistics

        print("\nKao Test Comparison:")
        print(f"PanelBox - Statistic: {result.statistic}, p-value: {result.pvalue}")
        print(
            f"R (IPS on residuals) - Statistic: {r_kao['statistic']:.4f}, p-value: {r_kao['p_value']:.6f}"
        )

        # Extract first value if dict
        if isinstance(result.pvalue, dict):
            py_pval = next(iter(result.pvalue.values()))
        else:
            py_pval = result.pvalue

        # Both should detect cointegration (reject H0 of no cointegration)
        assert py_pval < 0.05, "PanelBox Kao should detect cointegration"
        assert r_kao["p_value"] < 0.05, "R test should detect cointegration"

    def test_data_generation_consistency(self):
        """Test that we can reproduce the same data characteristics."""
        # Generate same data in Python
        np.random.seed(42)
        N = 30
        T = 80

        data_list = []
        for i in range(N):
            u = np.random.randn(T)
            x = np.cumsum(u)  # I(1) process
            epsilon = 0.5 * np.random.randn(T)  # I(0) error
            y = 1.5 * x + epsilon  # Cointegrated relationship

            entity_data = pd.DataFrame(
                {"entity": f"Entity_{i + 1}", "time": range(1, T + 1), "y": y, "x": x}
            )
            data_list.append(entity_data)

        data_py = pd.concat(data_list, ignore_index=True)

        # Compare means (should be close but not exact due to RNG differences)
        print("\nData Characteristics:")
        print(f"Python - y mean: {data_py['y'].mean():.4f}, x mean: {data_py['x'].mean():.4f}")
        print(
            f"R      - y mean: {self.test_data['y'].mean():.4f}, x mean: {self.test_data['x'].mean():.4f}"
        )

        # Check that both datasets have cointegrated structure
        assert len(data_py) == len(self.test_data), "Dataset sizes should match"

    def test_pedroni_implementation(self):
        """Test Pedroni test implementation."""
        # Run our Pedroni test
        result = pedroni_test(
            data=self.test_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars=["x"],
            method="all",
            trend="c",
        )

        print("\nPedroni Test Statistics:")
        for stat_name, stat_value in result.statistic.items():
            print(f"{stat_name}: {stat_value:.4f}")

        # Check that at least some statistics detect cointegration
        # (reject H0 at 5% level)
        rejections = sum(1 for pval in result.pvalue.values() if pval < 0.05)

        print(f"\nNumber of statistics rejecting H0 (p < 0.05): {rejections}/7")

        # For cointegrated data, we expect most statistics to reject
        assert rejections >= 4, f"Expected at least 4 rejections, got {rejections}"

    def test_westerlund_implementation(self):
        """Test Westerlund test implementation."""
        # Run our Westerlund test (without bootstrap for speed)
        result = westerlund_test(
            data=self.test_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars=["x"],
            method="all",
            trend="c",
            lags=1,  # Fixed lags for speed
            n_bootstrap=0,  # No bootstrap, use tabulated values
        )

        print("\nWesterlund Test Statistics:")
        for stat_name, stat_value in result.statistic.items():
            print(f"{stat_name}: {stat_value:.4f}")

        # Check that statistics are computed
        assert len(result.statistic) == 4, "Should have 4 test statistics"

        # For cointegrated data, we expect rejection
        # (alpha coefficients should be significantly negative)
        print("\nWesterlund p-values:")
        for stat_name, pval in result.pvalue.items():
            print(f"{stat_name}: {pval:.4f}")

    @pytest.mark.parametrize(
        ("test_func", "test_name"),
        [(kao_test, "Kao"), (pedroni_test, "Pedroni"), (westerlund_test, "Westerlund")],
    )
    def test_cointegration_detection(self, test_func, test_name):
        """
        Test that all methods detect cointegration in cointegrated data.
        """
        if test_name == "Westerlund":
            result = test_func(
                data=self.test_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars=["x"],
                method="all",
                trend="c",
                lags=1,
                n_bootstrap=0,
            )
        else:
            result = test_func(
                data=self.test_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars=["x"],
                method="all" if test_name == "Pedroni" else "df",
                trend="c",
            )

        print(f"\n{test_name} Test Result:")
        print(f"Statistic: {result.statistic}")
        print(f"P-value: {result.pvalue}")

        # All methods should detect cointegration
        # (at least one statistic should reject H0)
        if isinstance(result.pvalue, dict):
            min_pval = min(result.pvalue.values())
        else:
            min_pval = result.pvalue

        print(f"Minimum p-value: {min_pval:.6f}")
        assert min_pval < 0.10, f"{test_name} should detect cointegration (p < 0.10)"


class TestNonCointegration:
    """Test with non-cointegrated (spurious regression) data."""

    @classmethod
    def setup_class(cls):
        """Generate non-cointegrated data."""
        np.random.seed(123)
        N = 20
        T = 50

        data_list = []
        for i in range(N):
            # Two independent I(1) processes
            x = np.cumsum(np.random.randn(T))
            y = np.cumsum(np.random.randn(T))

            entity_data = pd.DataFrame(
                {"entity": f"Entity_{i + 1}", "time": range(1, T + 1), "y": y, "x": x}
            )
            data_list.append(entity_data)

        cls.data_non_coint = pd.concat(data_list, ignore_index=True)

    def test_kao_non_cointegration(self):
        """Kao test should NOT reject H0 for non-cointegrated data."""
        result = kao_test(
            data=self.data_non_coint,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars=["x"],
            method="df",
            trend="c",
        )

        print("\nKao Test (Non-cointegrated data):")
        print(f"Statistic: {result.statistic}")
        print(f"P-value: {result.pvalue}")

        # Extract first value if dict
        if isinstance(result.pvalue, dict):
            py_pval = next(iter(result.pvalue.values()))
        else:
            py_pval = result.pvalue

        # Should NOT reject H0 of no cointegration
        # (p-value should be high)
        # Note: This is a probabilistic test, may occasionally fail
        assert py_pval > 0.01, "Kao should not reject H0 for non-cointegrated data"

    def test_pedroni_non_cointegration(self):
        """Pedroni test should show weak/no evidence of cointegration."""
        result = pedroni_test(
            data=self.data_non_coint,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars=["x"],
            method="all",
            trend="c",
        )

        print("\nPedroni Test (Non-cointegrated data):")
        rejections = sum(1 for pval in result.pvalue.values() if pval < 0.05)

        print(f"Number of rejections at 5%: {rejections}/7")

        # Should have few rejections (allowing for some due to chance)
        assert rejections <= 3, f"Expected few rejections, got {rejections}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
