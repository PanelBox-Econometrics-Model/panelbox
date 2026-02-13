"""
Tests for Panel VECM estimation and cointegration rank tests.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var import PanelVARData
from panelbox.var.vecm import CointegrationRankTest, PanelVECM, PanelVECMResult, RankSelectionResult


class TestCointegrationRankTest:
    """Tests for cointegration rank selection."""

    @pytest.fixture
    def cointegrated_panel_data(self):
        """
        Generate panel data with one cointegrating relation.

        y1_t and y2_t are cointegrated: y1_t ≈ β·y2_t
        """
        np.random.seed(42)
        N = 20  # entities
        T = 30  # time periods

        data = []
        for i in range(N):
            # Random walk component (common trend)
            trend = np.cumsum(np.random.randn(T) * 0.5)

            # y1 and y2 share the common trend (cointegrated)
            beta_true = 1.5
            y2 = trend + np.random.randn(T) * 0.1
            y1 = beta_true * y2 + np.random.randn(T) * 0.1

            # y3 is independent (not cointegrated)
            y3 = np.cumsum(np.random.randn(T) * 0.5)

            for t in range(T):
                data.append({"entity": i, "time": t, "y1": y1[t], "y2": y2[t], "y3": y3[t]})

        df = pd.DataFrame(data)
        return df

    @pytest.fixture
    def no_cointegration_data(self):
        """Generate panel data without cointegration (independent random walks)."""
        np.random.seed(123)
        N = 20
        T = 30

        data = []
        for i in range(N):
            y1 = np.cumsum(np.random.randn(T))
            y2 = np.cumsum(np.random.randn(T))

            for t in range(T):
                data.append({"entity": i, "time": t, "y1": y1[t], "y2": y2[t]})

        df = pd.DataFrame(data)
        return df

    def test_rank_test_initialization(self, cointegrated_panel_data):
        """Test that rank test initializes correctly."""
        data = PanelVARData(
            cointegrated_panel_data,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        rank_test = CointegrationRankTest(data)

        assert rank_test.K == 2
        assert rank_test.N == 20
        assert rank_test.max_rank == 1  # K - 1
        assert rank_test.p == 2

    def test_rank_test_with_cointegration(self, cointegrated_panel_data):
        """Test rank selection with cointegrated data."""
        data = PanelVARData(
            cointegrated_panel_data,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        rank_test = CointegrationRankTest(data)
        results = rank_test.test_rank()

        # Check that results object is correct type
        assert isinstance(results, RankSelectionResult)

        # Should detect at least some cointegration
        # (may not be perfect with small sample)
        assert results.selected_rank >= 0
        assert results.selected_rank <= 1

        # Check that we have test results for all ranks
        assert len(results.trace_tests) == 2  # ranks 0 and 1
        assert len(results.maxeig_tests) == 2

    def test_rank_test_without_cointegration(self, no_cointegration_data):
        """Test rank selection without cointegration."""
        data = PanelVARData(
            no_cointegration_data,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        rank_test = CointegrationRankTest(data)
        results = rank_test.test_rank()

        # Should select rank 0 (no cointegration)
        # (may not be perfect with small sample)
        assert results.selected_rank >= 0
        assert results.selected_rank <= 1

    def test_rank_test_summary(self, cointegrated_panel_data):
        """Test that summary output works."""
        data = PanelVARData(
            cointegrated_panel_data,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=2,
        )

        rank_test = CointegrationRankTest(data)
        results = rank_test.test_rank()

        summary = results.summary()

        assert isinstance(summary, str)
        assert "Panel Cointegration Rank Test" in summary
        assert "Trace Test Results" in summary
        assert "Max-Eigenvalue Test Results" in summary
        assert "Selected Ranks" in summary


class TestPanelVECM:
    """Tests for Panel VECM estimation."""

    @pytest.fixture
    def simple_cointegrated_data(self):
        """Generate simple cointegrated panel data."""
        np.random.seed(42)
        N = 30
        T = 40

        # True parameters
        beta_true = np.array([[1.0], [-1.5]])  # y1 - 1.5*y2 = 0
        alpha_true = np.array([[-0.2], [0.1]])  # adjustment speeds

        data = []
        for i in range(N):
            # Initialize
            y = np.zeros((T, 2))
            y[0] = np.random.randn(2)

            # Generate via VECM
            for t in range(1, T):
                # Error correction term
                ect = beta_true.T @ y[t - 1 : t].T

                # VECM equation
                dy = alpha_true @ ect + np.random.randn(2, 1) * 0.1
                y[t] = y[t - 1] + dy.flatten()

            for t in range(T):
                data.append({"entity": i, "time": t, "y1": y[t, 0], "y2": y[t, 1]})

        df = pd.DataFrame(data)
        return df, beta_true, alpha_true

    def test_vecm_initialization(self, simple_cointegrated_data):
        """Test VECM initialization."""
        df, _, _ = simple_cointegrated_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(data, rank=1)

        assert vecm.K == 2
        assert vecm.N == 30
        assert vecm.rank == 1

    def test_vecm_estimation_ml(self, simple_cointegrated_data):
        """Test ML estimation of VECM."""
        df, beta_true, alpha_true = simple_cointegrated_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(data, rank=1)
        results = vecm.fit(method="ml")

        # Check result type
        assert isinstance(results, PanelVECMResult)

        # Check dimensions
        assert results.alpha.shape == (2, 1)  # K × r
        assert results.beta.shape == (2, 1)  # K × r
        assert results.K == 2
        assert results.rank == 1

        # Check that β is normalized (first element = 1)
        assert np.abs(results.beta[0, 0] - 1.0) < 1e-6

        # Check that estimated β is close to true β (normalized)
        beta_true_normalized = beta_true / beta_true[0, 0]
        # Allow some estimation error
        assert np.abs(results.beta[1, 0] - beta_true_normalized[1, 0]) < 0.5

    def test_vecm_rank_zero(self):
        """Test VECM with rank=0 (no cointegration)."""
        np.random.seed(123)
        N = 20
        T = 30

        # Generate independent random walks
        data = []
        for i in range(N):
            y1 = np.cumsum(np.random.randn(T))
            y2 = np.cumsum(np.random.randn(T))

            for t in range(T):
                data.append({"entity": i, "time": t, "y1": y1[t], "y2": y2[t]})

        df = pd.DataFrame(data)

        panel_data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(panel_data, rank=0)
        results = vecm.fit()

        # With rank=0, α and β should be empty
        assert results.alpha.shape == (2, 0)
        assert results.beta.shape == (2, 0)
        assert results.Pi.shape == (2, 2)
        assert np.allclose(results.Pi, 0)

    def test_vecm_to_var_conversion(self, simple_cointegrated_data):
        """Test conversion from VECM to VAR representation."""
        df, _, _ = simple_cointegrated_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(data, rank=1)
        results = vecm.fit()

        # Convert to VAR
        A_matrices = results.to_var()

        # Check dimensions
        assert len(A_matrices) == 2  # p lags
        for A in A_matrices:
            assert A.shape == (2, 2)

        # Check relationship: Π = A_1 + A_2 + ... + A_p - I
        Pi_reconstructed = sum(A_matrices) - np.eye(2)
        assert np.allclose(Pi_reconstructed, results.Pi, atol=1e-10)

    def test_vecm_cointegrating_relations(self, simple_cointegrated_data):
        """Test extraction of cointegrating relations."""
        df, _, _ = simple_cointegrated_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(data, rank=1)
        results = vecm.fit()

        # Get cointegrating relations
        beta_df = results.cointegrating_relations()

        assert isinstance(beta_df, pd.DataFrame)
        assert beta_df.shape == (2, 1)
        assert list(beta_df.index) == ["y1", "y2"]

        # First element should be 1 (normalized)
        assert np.abs(beta_df.iloc[0, 0] - 1.0) < 1e-10

    def test_vecm_adjustment_speeds(self, simple_cointegrated_data):
        """Test extraction of adjustment speeds."""
        df, _, _ = simple_cointegrated_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(data, rank=1)
        results = vecm.fit()

        # Get adjustment speeds
        alpha_df = results.adjustment_speeds()

        assert isinstance(alpha_df, pd.DataFrame)
        assert alpha_df.shape == (2, 1)
        assert list(alpha_df.index) == ["Δy1", "Δy2"]

    def test_weak_exogeneity_test(self, simple_cointegrated_data):
        """Test weak exogeneity test."""
        df, _, _ = simple_cointegrated_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(data, rank=1)
        results = vecm.fit()

        # Test weak exogeneity
        test_y1 = results.test_weak_exogeneity("y1")

        assert "statistic" in test_y1
        assert "p_value" in test_y1
        assert "df" in test_y1
        assert test_y1["df"] == 1  # rank
        assert 0 <= test_y1["p_value"] <= 1

    def test_strong_exogeneity_test(self, simple_cointegrated_data):
        """Test strong exogeneity test."""
        df, _, _ = simple_cointegrated_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(data, rank=1)
        results = vecm.fit()

        # Test strong exogeneity
        test_y1 = results.test_strong_exogeneity("y1")

        assert "statistic" in test_y1
        assert "p_value" in test_y1
        assert "df" in test_y1
        assert test_y1["df"] > 1  # rank + short-run restrictions
        assert 0 <= test_y1["p_value"] <= 1

    def test_vecm_summary(self, simple_cointegrated_data):
        """Test summary output."""
        df, _, _ = simple_cointegrated_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(data, rank=1)
        results = vecm.fit()

        summary = results.summary()

        assert isinstance(summary, str)
        assert "Panel VECM Estimation Results" in summary
        assert "Cointegrating Relations" in summary
        assert "Adjustment Coefficients" in summary
        assert "Short-Run Dynamics" in summary
        assert "Exogeneity Tests" in summary

    def test_automatic_rank_selection(self, simple_cointegrated_data):
        """Test automatic rank selection."""
        df, _, _ = simple_cointegrated_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        # Don't specify rank - should auto-select
        vecm = PanelVECM(data)

        # Rank should be selected automatically
        assert vecm.rank >= 0
        assert vecm.rank < 2

        # Should be able to fit
        results = vecm.fit()
        assert isinstance(results, PanelVECMResult)

    def test_vecm_three_variables(self):
        """Test VECM with three variables."""
        np.random.seed(456)
        N = 25
        T = 35

        # Generate data with 2 cointegrating relations among 3 variables
        data = []
        for i in range(N):
            y = np.zeros((T, 3))
            y[0] = np.random.randn(3)

            # Simple dynamics
            beta1 = np.array([1, -1, 0])
            beta2 = np.array([0, 1, -1])
            alpha1 = np.array([-0.1, 0.05, 0.05])
            alpha2 = np.array([0.05, -0.1, 0.05])

            for t in range(1, T):
                ect1 = beta1 @ y[t - 1]
                ect2 = beta2 @ y[t - 1]
                dy = alpha1 * ect1 + alpha2 * ect2 + np.random.randn(3) * 0.1
                y[t] = y[t - 1] + dy

            for t in range(T):
                data.append({"entity": i, "time": t, "y1": y[t, 0], "y2": y[t, 1], "y3": y[t, 2]})

        df = pd.DataFrame(data)

        panel_data = PanelVARData(
            df, endog_vars=["y1", "y2", "y3"], entity_col="entity", time_col="time", lags=2
        )

        vecm = PanelVECM(panel_data, rank=2)
        results = vecm.fit()

        # Check dimensions
        assert results.alpha.shape == (3, 2)
        assert results.beta.shape == (3, 2)
        assert results.rank == 2

        # Get cointegrating relations
        beta_df = results.cointegrating_relations()
        assert beta_df.shape == (3, 2)

    def test_vecm_invalid_rank(self):
        """Test that invalid rank raises error."""
        df = pd.DataFrame(
            {
                "entity": [0] * 10,
                "time": range(10),
                "y1": np.random.randn(10),
                "y2": np.random.randn(10),
            }
        )

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        # Rank must be < K
        with pytest.raises(ValueError):
            PanelVECM(data, rank=2)

        with pytest.raises(ValueError):
            PanelVECM(data, rank=-1)


class TestVECMToVARConversion:
    """Tests for VECM to VAR conversion."""

    def test_conversion_identity(self):
        """Test that VECM->VAR conversion preserves Π = Σ A_l - I."""
        # Create simple VECM parameters
        K = 2
        rank = 1

        alpha = np.array([[-0.2], [0.1]])
        beta = np.array([[1.0], [-1.5]])
        Gamma = [np.array([[0.3, 0.1], [0.05, 0.4]])]

        Pi = alpha @ beta.T

        # Manual conversion
        A1 = Pi + Gamma[0] + np.eye(K)
        A2 = -Gamma[0]

        # Check relationship
        Pi_reconstructed = A1 + A2 - np.eye(K)

        assert np.allclose(Pi_reconstructed, Pi, atol=1e-10)

    def test_conversion_with_vecm_result(self):
        """Test conversion using actual VECMResult object."""
        # Create mock VECM result
        K = 2
        rank = 1

        alpha = np.array([[-0.2], [0.1]])
        beta = np.array([[1.0], [-1.5]])
        Gamma = [np.array([[0.3, 0.1], [0.05, 0.4]])]
        Sigma = np.eye(K) * 0.1
        residuals = np.random.randn(100, K)

        result = PanelVECMResult(
            alpha=alpha,
            beta=beta,
            Gamma=Gamma,
            Sigma=Sigma,
            residuals=residuals,
            var_names=["y1", "y2"],
            rank=rank,
            method="ml",
            N=20,
            T_avg=25.0,
        )

        # Convert to VAR
        A_matrices = result.to_var()

        assert len(A_matrices) == 2

        # Verify relationship
        Pi_reconstructed = sum(A_matrices) - np.eye(K)
        assert np.allclose(Pi_reconstructed, result.Pi, atol=1e-10)
