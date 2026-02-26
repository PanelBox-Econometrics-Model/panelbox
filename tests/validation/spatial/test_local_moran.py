"""
Tests for LocalMoranI (LISA) spatial cluster detection.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from panelbox.validation.spatial.local_moran import LocalMoranI


def _create_rook_weights(nrow, ncol):
    """Create a rook contiguity weight matrix for an nrow x ncol grid."""
    N = nrow * ncol
    W = np.zeros((N, N))
    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            # Right neighbor
            if j + 1 < ncol:
                W[idx, idx + 1] = 1.0
                W[idx + 1, idx] = 1.0
            # Bottom neighbor
            if i + 1 < nrow:
                W[idx, idx + ncol] = 1.0
                W[idx + ncol, idx] = 1.0
    return W


class TestLocalMoranIInit:
    """Test LocalMoranI initialization."""

    def test_init_cross_section(self):
        """Test initialization with cross-section data."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N)
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)

        assert lisa.N == N
        assert lisa.is_panel is False
        assert lisa.threshold == 0.05
        assert len(lisa.cluster_types) == 5

    def test_init_panel_data(self):
        """Test initialization with panel data."""
        np.random.seed(42)
        N, T = 25, 5
        NT = N * T
        variable = np.random.randn(NT)
        W = _create_rook_weights(5, 5)
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)

        lisa = LocalMoranI(variable, W, entity_index=entity_index, time_index=time_index)

        assert lisa.N == N
        assert lisa.is_panel is True

    def test_init_custom_threshold(self):
        """Test initialization with custom significance threshold."""
        np.random.seed(42)
        N = 9
        variable = np.random.randn(N)
        W = _create_rook_weights(3, 3)

        lisa = LocalMoranI(variable, W, threshold=0.10)

        assert lisa.threshold == 0.10

    def test_init_incompatible_dimensions(self):
        """Test error on incompatible W and variable dimensions."""
        variable = np.random.randn(10)
        W = _create_rook_weights(5, 5)  # 25x25

        with pytest.raises(ValueError, match="incompatible"):
            LocalMoranI(variable, W)


class TestLocalMoranIRun:
    """Test LocalMoranI run method."""

    def test_run_returns_dataframe(self):
        """Test that run returns a DataFrame with correct columns."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N)
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)

        assert isinstance(results, pd.DataFrame)
        expected_cols = ["entity", "value", "Ii", "EIi", "VIi", "z_score", "pvalue", "cluster_type"]
        for col in expected_cols:
            assert col in results.columns
        assert len(results) == N

    def test_run_reproducible_with_seed(self):
        """Test that results are reproducible with same seed."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N)
        W = _create_rook_weights(5, 5)

        lisa1 = LocalMoranI(variable, W)
        results1 = lisa1.run(n_permutations=99, seed=123)

        lisa2 = LocalMoranI(variable, W)
        results2 = lisa2.run(n_permutations=99, seed=123)

        np.testing.assert_allclose(results1["Ii"].values, results2["Ii"].values)
        np.testing.assert_allclose(results1["pvalue"].values, results2["pvalue"].values)
        assert list(results1["cluster_type"]) == list(results2["cluster_type"])

    def test_run_pvalues_in_range(self):
        """Test that p-values are in [0, 1]."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N) * 2
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)

        assert (results["pvalue"] >= 0).all()
        assert (results["pvalue"] <= 1).all()

    def test_run_expected_values_correct(self):
        """Test that expected values are computed correctly."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N)
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)

        # Expected value should be -1/(N-1) for all valid observations
        expected_EI = -1.0 / (N - 1)
        np.testing.assert_allclose(results["EIi"].values, expected_EI, rtol=1e-10)

    def test_run_variance_positive(self):
        """Test that variance is positive for all valid observations."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N) * 3 + 1
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)

        assert (results["VIi"] > 0).all()


class TestLocalMoranIClusterClassification:
    """Test cluster classification logic."""

    def test_hh_cluster_positive_Ii(self):
        """Test that High-High cluster regions produce positive local Moran's I."""
        np.random.seed(42)
        N = 25
        W = _create_rook_weights(5, 5)

        # Create data with high values clustered in center
        variable = np.zeros(N)
        center_indices = [6, 7, 8, 11, 12, 13, 16, 17, 18]
        for idx in center_indices:
            variable[idx] = 5.0
        # Add small noise
        variable += np.random.randn(N) * 0.1

        lisa = LocalMoranI(variable, W, threshold=0.10)
        results = lisa.run(n_permutations=99, seed=42)

        # Center nodes surrounded by other high values should have positive Ii
        # (positive spatial autocorrelation: similar values near each other)
        center_Ii = results.loc[results["entity"].isin(center_indices), "Ii"]
        assert (center_Ii > 0).all(), "Center cluster nodes should have positive Ii"

        # Center nodes should be above the global mean
        mean_val = results["value"].mean()
        center_vals = results.loc[results["entity"].isin(center_indices), "value"]
        assert (center_vals > mean_val).all(), "Center cluster values should be above mean"

    def test_ll_cluster_positive_Ii(self):
        """Test that Low-Low cluster regions produce positive local Moran's I."""
        np.random.seed(42)
        N = 25
        W = _create_rook_weights(5, 5)

        # Create data with low values clustered in a corner
        variable = np.zeros(N) + 3.0  # High baseline
        cold_indices = [0, 1, 5, 6]
        for idx in cold_indices:
            variable[idx] = -3.0
        variable += np.random.randn(N) * 0.1

        lisa = LocalMoranI(variable, W, threshold=0.10)
        results = lisa.run(n_permutations=99, seed=42)

        # Cold cluster nodes surrounded by other low values should have positive Ii
        # (positive spatial autocorrelation: similar low values near each other)
        cold_Ii = results.loc[results["entity"].isin(cold_indices), "Ii"]
        assert (cold_Ii > 0).all(), "Cold cluster nodes should have positive Ii"

        # Cold cluster nodes should be below the global mean
        mean_val = results["value"].mean()
        cold_vals = results.loc[results["entity"].isin(cold_indices), "value"]
        assert (cold_vals < mean_val).all(), "Cold cluster values should be below mean"

    def test_classify_clusters_logic(self):
        """Test that _classify_clusters correctly labels HH, LL, HL, LH."""
        np.random.seed(42)
        N = 9
        W = _create_rook_weights(3, 3)

        variable = np.array([10, 10, 10, 0, 0, 0, -10, -10, -10], dtype=float)
        lisa = LocalMoranI(variable, W)

        mean_value = variable.mean()
        valid_mask = np.ones(N, dtype=bool)

        # Positive Ii + above mean -> HH
        # Positive Ii + below mean -> LL
        # Negative Ii + above mean -> HL
        # Negative Ii + below mean -> LH
        Ii_pos = np.array([1.0] * N)
        Ii_neg = np.array([-1.0] * N)
        pvalues_sig = np.array([0.01] * N)
        pvalues_ns = np.array([0.99] * N)

        # All significant, positive Ii
        labels = lisa._classify_clusters(variable, mean_value, Ii_pos, pvalues_sig, valid_mask)
        # Entities 0,1,2 (value=10 > mean=0) with positive Ii -> HH
        assert labels[0] == "HH"
        assert labels[1] == "HH"
        assert labels[2] == "HH"
        # Entities 6,7,8 (value=-10 < mean=0) with positive Ii -> LL
        assert labels[6] == "LL"
        assert labels[7] == "LL"
        assert labels[8] == "LL"

        # All significant, negative Ii
        labels = lisa._classify_clusters(variable, mean_value, Ii_neg, pvalues_sig, valid_mask)
        # Entities 0,1,2 (value=10 > mean=0) with negative Ii -> HL
        assert labels[0] == "HL"
        # Entities 6,7,8 (value=-10 < mean=0) with negative Ii -> LH
        assert labels[6] == "LH"

        # Not significant
        labels = lisa._classify_clusters(variable, mean_value, Ii_pos, pvalues_ns, valid_mask)
        assert all(l == "Not significant" for l in labels)

    def test_all_cluster_types_valid(self):
        """Test that only valid cluster types are returned."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N) * 3
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)

        valid_types = {"HH", "LL", "HL", "LH", "Not significant"}
        for ct in results["cluster_type"]:
            assert ct in valid_types

    def test_no_significant_clusters_random_data(self):
        """Test that random data produces mostly 'Not significant' clusters."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N)
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W, threshold=0.05)
        results = lisa.run(n_permutations=999, seed=42)

        not_sig = np.sum(results["cluster_type"] == "Not significant")
        # At least 60% should be not significant for random data
        assert not_sig / N >= 0.60


class TestLocalMoranIEdgeCases:
    """Test edge cases."""

    def test_insufficient_data(self):
        """Test with fewer than 3 valid observations."""
        variable = np.array([1.0, 2.0])
        W = np.array([[0, 1], [1, 0]], dtype=float)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)

        # Should return DataFrame with nan values
        assert isinstance(results, pd.DataFrame)
        assert results["cluster_type"].iloc[0] == "Not significant"

    def test_constant_variable(self):
        """Test with constant variable (zero std)."""
        np.random.seed(42)
        N = 9
        variable = np.ones(N) * 5.0
        W = _create_rook_weights(3, 3)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == N

    def test_panel_data_uses_last_period(self):
        """Test that panel data uses last period by default."""
        np.random.seed(42)
        N, T = 9, 3
        NT = N * T
        variable = np.random.randn(NT)
        W = _create_rook_weights(3, 3)
        entity_index = np.repeat(np.arange(N), T)
        time_index = np.tile(np.arange(T), N)

        lisa = LocalMoranI(variable, W, entity_index=entity_index, time_index=time_index)
        results = lisa.run(n_permutations=99, seed=42)

        assert len(results) == N
        assert lisa.is_panel is True

    def test_no_neighbors(self):
        """Test with isolated node (no neighbors)."""
        N = 4
        # Only connect 0-1 and 2-3, leave them disconnected from each other
        W = np.zeros((N, N))
        W[0, 1] = 1
        W[1, 0] = 1
        W[2, 3] = 1
        W[3, 2] = 1

        variable = np.array([10.0, 9.0, -10.0, -9.0])

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == N


class TestLocalMoranISummary:
    """Test summary method."""

    def test_summary_returns_dataframe(self):
        """Test that summary returns a DataFrame."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N) * 3
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)
        summary = lisa.summary(results)

        assert isinstance(summary, pd.DataFrame)
        assert "Count" in summary.columns
        assert "Mean_Ii" in summary.columns
        assert "Std_Ii" in summary.columns
        assert "Mean_pvalue" in summary.columns
        assert "Percentage" in summary.columns

    def test_summary_counts_sum_to_N(self):
        """Test that cluster counts sum to N."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N) * 3
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)
        summary = lisa.summary(results)

        assert summary["Count"].sum() == N

    def test_summary_percentages_sum_to_100(self):
        """Test that percentages sum to 100."""
        np.random.seed(42)
        N = 25
        variable = np.random.randn(N) * 3
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)
        summary = lisa.summary(results)

        np.testing.assert_allclose(summary["Percentage"].sum(), 100.0, atol=0.01)


class TestLocalMoranIPlot:
    """Test plot_clusters method."""

    def test_plot_matplotlib(self):
        """Test matplotlib plotting."""
        import matplotlib.pyplot as plt

        np.random.seed(42)
        N = 25
        variable = np.random.randn(N) * 3
        W = _create_rook_weights(5, 5)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)
        fig = lisa.plot_clusters(results, backend="matplotlib")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_matplotlib_few_entities(self):
        """Test matplotlib plot with <= 30 entities (shows labels)."""
        import matplotlib.pyplot as plt

        np.random.seed(42)
        N = 9
        variable = np.random.randn(N) * 3
        W = _create_rook_weights(3, 3)

        lisa = LocalMoranI(variable, W)
        results = lisa.run(n_permutations=99, seed=42)
        fig = lisa.plot_clusters(results, backend="matplotlib")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
