"""
Validation tests for Moran's I and LISA against R spdep package.

This module validates the Python implementation of Moran's I and Local Moran's I (LISA)
by comparing results against reference results from R's spdep package.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import pytest
from statsmodels.regression.linear_model import OLS

from panelbox.diagnostics.spatial_tests import (
    LISAResult,
    LocalMoranI,
    MoranIPanelTest,
    MoranIResult,
)

FIXTURES_PATH = Path(__file__).parent.parent / "spatial" / "fixtures"


@pytest.fixture
def spatial_test_data():
    """Load spatial test data."""
    df = pd.read_csv(FIXTURES_PATH / "spatial_test_data.csv")
    W = np.loadtxt(FIXTURES_PATH / "spatial_weights.csv", delimiter=",")
    return df, W


@pytest.fixture
def r_morans_results():
    """Load R Moran's I results."""
    results_file = FIXTURES_PATH / "r_morans_i_results.json"
    if not results_file.exists():
        pytest.skip("R validation results not found. Run r_morans_i_validation.R first.")
    with open(results_file, "r") as f:
        return json.load(f)


@pytest.fixture
def ols_residuals(spatial_test_data):
    """Fit OLS model and return residuals."""
    df, W = spatial_test_data

    # Fit pooled OLS
    y, X = patsy.dmatrices("y ~ x1 + x2 + x3", data=df, return_type="dataframe")
    ols_result = OLS(y.values.flatten(), X.values).fit()

    return ols_result.resid, df


class TestMoransIValidation:
    """Validate Moran's I against R spdep."""

    def test_morans_i_pooled_statistic(self, ols_residuals, spatial_test_data, r_morans_results):
        """Test Moran's I pooled method statistic matches R."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        # Moran's I test
        test = MoranIPanelTest(
            residuals=residuals,
            W=W,
            entity_ids=df["entity"].values,
            time_ids=df["time"].values,
        )

        result = test.run(method="pooled")

        # Compare with R
        r_stat = r_morans_results["moran_pooled"]["statistic"]

        assert np.isclose(
            result.statistic, r_stat, rtol=0.10
        ), f"Moran's I statistic: {result.statistic:.6f} vs R: {r_stat:.6f}"

    def test_morans_i_pooled_expected_value(
        self, ols_residuals, spatial_test_data, r_morans_results
    ):
        """Test Moran's I expected value matches R."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        test = MoranIPanelTest(
            residuals=residuals,
            W=W,
            entity_ids=df["entity"].values,
            time_ids=df["time"].values,
        )

        result = test.run(method="pooled")
        r_expected = r_morans_results["moran_pooled"]["expected"]

        assert np.isclose(
            result.expected_value, r_expected, rtol=0.01
        ), f"Expected value: {result.expected_value:.6f} vs R: {r_expected:.6f}"

    def test_morans_i_pooled_variance(self, ols_residuals, spatial_test_data, r_morans_results):
        """Test Moran's I variance matches R."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        test = MoranIPanelTest(
            residuals=residuals,
            W=W,
            entity_ids=df["entity"].values,
            time_ids=df["time"].values,
        )

        result = test.run(method="pooled")
        r_variance = r_morans_results["moran_pooled"]["variance"]

        assert np.isclose(
            result.variance, r_variance, rtol=0.15
        ), f"Variance: {result.variance:.8f} vs R: {r_variance:.8f}"

    def test_morans_i_pooled_pvalue(self, ols_residuals, spatial_test_data, r_morans_results):
        """Test Moran's I p-value matches R."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        test = MoranIPanelTest(
            residuals=residuals,
            W=W,
            entity_ids=df["entity"].values,
            time_ids=df["time"].values,
        )

        result = test.run(method="pooled")
        r_pvalue = r_morans_results["moran_pooled"]["pvalue"]

        # P-values may vary more, use looser tolerance
        assert np.isclose(
            result.pvalue, r_pvalue, rtol=0.20
        ), f"Moran's I p-value: {result.pvalue:.6f} vs R: {r_pvalue:.6f}"

    def test_morans_i_by_period(self, ols_residuals, spatial_test_data, r_morans_results):
        """Test Moran's I by period matches R."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        test = MoranIPanelTest(
            residuals=residuals,
            W=W,
            entity_ids=df["entity"].values,
            time_ids=df["time"].values,
        )

        results = test.run(method="by_period")
        r_by_period = r_morans_results["moran_by_period"]

        # Check at least one period
        first_period = list(r_by_period.keys())[0]

        assert first_period in results, f"Period {first_period} not in results"

        py_stat = results[first_period].statistic
        r_stat = r_by_period[first_period]["statistic"]

        assert np.isclose(
            py_stat, r_stat, rtol=0.15
        ), f"Moran's I period {first_period}: {py_stat:.6f} vs R: {r_stat:.6f}"

    def test_morans_i_average_method(self, ols_residuals, spatial_test_data):
        """Test Moran's I average method runs without error."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        test = MoranIPanelTest(
            residuals=residuals,
            W=W,
            entity_ids=df["entity"].values,
            time_ids=df["time"].values,
        )

        result = test.run(method="average")

        assert isinstance(result, MoranIResult)
        assert result.additional_info["method"] == "average"
        assert result.additional_info["n_periods"] > 0

    def test_morans_i_conclusion(self, ols_residuals, spatial_test_data):
        """Test Moran's I conclusion is set correctly."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        test = MoranIPanelTest(
            residuals=residuals,
            W=W,
            entity_ids=df["entity"].values,
            time_ids=df["time"].values,
        )

        result = test.run(method="pooled")

        assert isinstance(result.conclusion, str)
        assert len(result.conclusion) > 0


class TestLISAValidation:
    """Validate LISA against R spdep."""

    def test_lisa_local_statistics_correlation(
        self, ols_residuals, spatial_test_data, r_morans_results
    ):
        """Test LISA local statistics correlate highly with R."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        # Compute time-averaged residuals by entity
        df_resid = df.copy()
        df_resid["residual"] = residuals
        avg_resid = df_resid.groupby("entity")["residual"].mean().values

        # LISA
        lisa = LocalMoranI(values=avg_resid, W=W, entity_ids=np.unique(df["entity"].values))

        result = lisa.run(permutations=999)

        # Compare with R
        r_local_i = np.array(r_morans_results["lisa"]["local_i"])

        # Check correlation of local statistics
        correlation = np.corrcoef(result.local_i, r_local_i)[0, 1]

        assert correlation > 0.95, f"LISA local statistics correlation too low: {correlation:.4f}"

    def test_lisa_standardized_values(self, ols_residuals, spatial_test_data, r_morans_results):
        """Test LISA standardized values match R."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        df_resid = df.copy()
        df_resid["residual"] = residuals
        avg_resid = df_resid.groupby("entity")["residual"].mean().values

        lisa = LocalMoranI(values=avg_resid, W=W, entity_ids=np.unique(df["entity"].values))

        result = lisa.run(permutations=999)

        # Compare standardized values with R
        r_z_values = np.array(r_morans_results["lisa"]["z_values"])

        correlation = np.corrcoef(result.z_values, r_z_values)[0, 1]

        assert correlation > 0.99, f"Z-values correlation too low: {correlation:.4f}"

    def test_lisa_cluster_types(self, ols_residuals, spatial_test_data, r_morans_results):
        """Test LISA cluster classification."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        df_resid = df.copy()
        df_resid["residual"] = residuals
        avg_resid = df_resid.groupby("entity")["residual"].mean().values

        lisa = LocalMoranI(values=avg_resid, W=W, entity_ids=np.unique(df["entity"].values))

        result = lisa.run(permutations=999)
        clusters = result.get_clusters(alpha=0.05)

        # Count cluster types
        py_counts = clusters["cluster_type"].value_counts().to_dict()
        r_counts = r_morans_results["lisa"]["cluster_counts"]

        # Check that we have similar proportions
        # (exact counts may vary due to randomness in permutation)
        for cluster_type in ["HH", "LL", "HL", "LH", "Not significant"]:
            py_count = py_counts.get(cluster_type, 0)
            r_count = r_counts.get(cluster_type, 0)

            # Allow for some variation due to permutation randomness
            if r_count > 0:
                ratio = py_count / r_count
                assert (
                    0.5 < ratio < 2.0
                ), f"Cluster type {cluster_type}: Python {py_count} vs R {r_count} (ratio: {ratio:.2f})"

    def test_lisa_pvalues_reasonable(self, ols_residuals, spatial_test_data):
        """Test LISA p-values are in valid range."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        df_resid = df.copy()
        df_resid["residual"] = residuals
        avg_resid = df_resid.groupby("entity")["residual"].mean().values

        lisa = LocalMoranI(values=avg_resid, W=W, entity_ids=np.unique(df["entity"].values))

        result = lisa.run(permutations=999)

        # All p-values should be between 0 and 1
        assert np.all(result.pvalues >= 0)
        assert np.all(result.pvalues <= 1)

        # Minimum p-value should be > 0
        assert np.min(result.pvalues) > 0

    def test_lisa_summary(self, ols_residuals, spatial_test_data):
        """Test LISA summary method."""
        residuals, df = ols_residuals
        _, W = spatial_test_data

        df_resid = df.copy()
        df_resid["residual"] = residuals
        avg_resid = df_resid.groupby("entity")["residual"].mean().values

        lisa = LocalMoranI(values=avg_resid, W=W, entity_ids=np.unique(df["entity"].values))

        result = lisa.run(permutations=999)
        summary = result.summary(alpha=0.05)

        assert isinstance(summary, str)
        assert "Local Moran's I" in summary
        assert "Total observations" in summary


class TestMoransIUnit:
    """Unit tests for Moran's I."""

    def test_positive_autocorrelation(self):
        """Moran's I should detect positive autocorrelation."""
        np.random.seed(42)
        N, T = 50, 5

        # Create clustered data with circular neighbors
        W = np.zeros((N, N))
        for i in range(N):
            W[i, (i + 1) % N] = 0.5
            W[i, (i - 1) % N] = 0.5

        # Create clusters of similar values (positive spatial autocorrelation)
        # Group entities in clusters of size 5
        residuals = np.zeros(N * T)
        for i in range(N):
            # Cluster: entities 0-4 have value 5, 5-9 have value -5, 10-14 have value 5, etc.
            cluster_val = 5.0 if (i // 5) % 2 == 0 else -5.0
            for t in range(T):
                residuals[i * T + t] = cluster_val + np.random.randn() * 0.1

        entity_ids = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)

        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
        result = test.run()

        # With clustered data, we should detect positive spatial autocorrelation
        assert (
            result.statistic > 0
        ), f"Should detect positive autocorrelation, got {result.statistic}"
        assert result.pvalue < 0.05, f"Should be significant, got p-value {result.pvalue}"

    def test_no_autocorrelation(self):
        """Moran's I should not reject for random data."""
        np.random.seed(42)
        N, T = 50, 5

        W = np.zeros((N, N))
        for i in range(N):
            W[i, (i + 1) % N] = 0.5
            W[i, (i - 1) % N] = 0.5

        residuals = np.random.randn(N * T)
        entity_ids = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)

        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
        result = test.run()

        # Should not strongly reject (allowing for some randomness)
        assert result.pvalue > 0.01 or np.abs(result.statistic) < 0.3

    def test_negative_autocorrelation(self):
        """Moran's I should detect negative autocorrelation."""
        np.random.seed(42)
        N, T = 50, 5

        # Create checkerboard pattern
        W = np.zeros((N, N))
        for i in range(N):
            W[i, (i + 1) % N] = 0.5
            W[i, (i - 1) % N] = 0.5

        # Opposite values for neighbors
        residuals = np.zeros(N * T)
        for i in range(N):
            base_val = 1.0 if i % 2 == 0 else -1.0
            for t in range(T):
                residuals[i * T + t] = base_val + np.random.randn() * 0.1

        entity_ids = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)

        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
        result = test.run()

        # Should detect negative autocorrelation
        assert result.statistic < 0, "Should detect negative autocorrelation"

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        N, T = 10, 5
        W = np.eye(N)
        residuals = np.random.randn(N * T)
        entity_ids = np.repeat(np.arange(N), T)
        time_ids = np.tile(np.arange(T), N)

        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)

        with pytest.raises(ValueError, match="Unknown method"):
            test.run(method="invalid_method")


class TestLISAUnit:
    """Unit tests for LISA."""

    def test_lisa_hot_spot_detection(self):
        """Test LISA detects hot spots (HH clusters)."""
        np.random.seed(42)
        N = 50

        # Create spatial weights
        W = np.zeros((N, N))
        for i in range(N):
            W[i, (i + 1) % N] = 0.5
            W[i, (i - 1) % N] = 0.5

        # Create data with one hot spot region
        values = np.random.randn(N) * 0.1
        # Hot spot: entities 10-15
        values[10:16] += 3.0

        lisa = LocalMoranI(values=values, W=W, entity_ids=np.arange(N))
        result = lisa.run(permutations=999)
        clusters = result.get_clusters(alpha=0.05)

        # Should detect some HH clusters in the hot spot region
        hh_clusters = clusters[clusters["cluster_type"] == "HH"]
        assert len(hh_clusters) > 0, "Should detect hot spots"

    def test_lisa_cold_spot_detection(self):
        """Test LISA detects cold spots (LL clusters)."""
        np.random.seed(42)
        N = 50

        W = np.zeros((N, N))
        for i in range(N):
            W[i, (i + 1) % N] = 0.5
            W[i, (i - 1) % N] = 0.5

        # Create data with one cold spot region
        values = np.random.randn(N) * 0.1
        # Cold spot: entities 20-25
        values[20:26] -= 3.0

        lisa = LocalMoranI(values=values, W=W, entity_ids=np.arange(N))
        result = lisa.run(permutations=999)
        clusters = result.get_clusters(alpha=0.05)

        # Should detect some LL clusters
        ll_clusters = clusters[clusters["cluster_type"] == "LL"]
        assert len(ll_clusters) > 0, "Should detect cold spots"

    def test_lisa_reproducibility(self):
        """Test LISA gives same results with same seed."""
        N = 30
        W = np.zeros((N, N))
        for i in range(N):
            W[i, (i + 1) % N] = 0.5
            W[i, (i - 1) % N] = 0.5

        values = np.random.randn(N)

        # Run twice - should give same results due to fixed seed
        lisa1 = LocalMoranI(values=values, W=W, entity_ids=np.arange(N))
        result1 = lisa1.run(permutations=999)

        lisa2 = LocalMoranI(values=values, W=W, entity_ids=np.arange(N))
        result2 = lisa2.run(permutations=999)

        np.testing.assert_array_equal(result1.local_i, result2.local_i)
        np.testing.assert_array_equal(result1.pvalues, result2.pvalues)

    def test_lisa_permutations_effect(self):
        """Test that more permutations give more stable results."""
        N = 30
        W = np.zeros((N, N))
        for i in range(N):
            W[i, (i + 1) % N] = 0.5
            W[i, (i - 1) % N] = 0.5

        values = np.random.randn(N)

        # Local I should be same regardless of permutations
        lisa_99 = LocalMoranI(values=values, W=W, entity_ids=np.arange(N))
        result_99 = lisa_99.run(permutations=99)

        lisa_999 = LocalMoranI(values=values, W=W, entity_ids=np.arange(N))
        result_999 = lisa_999.run(permutations=999)

        # Local I values should be identical
        np.testing.assert_array_almost_equal(result_99.local_i, result_999.local_i)

        # P-values can differ slightly but should be correlated
        correlation = np.corrcoef(result_99.pvalues, result_999.pvalues)[0, 1]
        assert correlation > 0.8, "P-values should be correlated"
