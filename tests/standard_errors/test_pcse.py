"""
Tests for Panel-Corrected Standard Errors (PCSE).
"""

import numpy as np
import pytest

from panelbox.standard_errors.pcse import PanelCorrectedStandardErrors, PCSEResult, pcse


@pytest.fixture
def simple_panel_data():
    """Create simple balanced panel data for testing."""
    # 3 entities, 5 time periods (T > N)
    n_entities = 3
    n_periods = 5
    n_params = 2

    # Create entity and time IDs
    entity_ids = np.repeat(np.arange(n_entities), n_periods)
    time_ids = np.tile(np.arange(n_periods), n_entities)

    # Create design matrix
    np.random.seed(42)
    X = np.random.randn(n_entities * n_periods, n_params)

    # Create residuals with some structure
    resid = np.random.randn(n_entities * n_periods) * 0.5

    return X, resid, entity_ids, time_ids


@pytest.fixture
def large_panel_data():
    """Create larger panel for more realistic testing."""
    # 5 entities, 20 time periods
    n_entities = 5
    n_periods = 20
    n_params = 3

    entity_ids = np.repeat(np.arange(n_entities), n_periods)
    time_ids = np.tile(np.arange(n_periods), n_entities)

    np.random.seed(123)
    X = np.random.randn(n_entities * n_periods, n_params)
    resid = np.random.randn(n_entities * n_periods) * 0.8

    return X, resid, entity_ids, time_ids


class TestPCSEResult:
    """Test PCSEResult dataclass."""

    def test_result_creation(self):
        """Test creating PCSEResult."""
        cov_matrix = np.array([[1.0, 0.1], [0.1, 2.0]])
        std_errors = np.array([1.0, 1.414])
        sigma_matrix = np.array([[0.5, 0.2, 0.1], [0.2, 0.6, 0.15], [0.1, 0.15, 0.55]])

        result = PCSEResult(
            cov_matrix=cov_matrix,
            std_errors=std_errors,
            sigma_matrix=sigma_matrix,
            n_obs=15,
            n_params=2,
            n_entities=3,
            n_periods=5,
        )

        assert result.n_obs == 15
        assert result.n_params == 2
        assert result.n_entities == 3
        assert result.n_periods == 5
        assert result.cov_matrix.shape == (2, 2)
        assert result.std_errors.shape == (2,)
        assert result.sigma_matrix.shape == (3, 3)


class TestPanelCorrectedStandardErrors:
    """Test PanelCorrectedStandardErrors class."""

    def test_initialization(self, simple_panel_data):
        """Test PCSE initialization."""
        X, resid, entity_ids, time_ids = simple_panel_data

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)

        assert pcse_est.n_obs == 15
        assert pcse_est.n_params == 2
        assert pcse_est.n_entities == 3
        assert pcse_est.n_periods == 5
        assert pcse_est.X.shape == (15, 2)
        assert pcse_est.resid.shape == (15,)

    def test_dimension_validation_entity_ids(self, simple_panel_data):
        """Test that entity_ids dimension mismatch raises error."""
        X, resid, entity_ids, time_ids = simple_panel_data

        with pytest.raises(ValueError, match="entity_ids dimension mismatch"):
            PanelCorrectedStandardErrors(X, resid, entity_ids[:-1], time_ids)

    def test_dimension_validation_time_ids(self, simple_panel_data):
        """Test that time_ids dimension mismatch raises error."""
        X, resid, entity_ids, time_ids = simple_panel_data

        with pytest.raises(ValueError, match="time_ids dimension mismatch"):
            PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids[:-2])

    def test_t_greater_than_n_requirement(self, simple_panel_data):
        """Test that T > N is satisfied."""
        X, resid, entity_ids, time_ids = simple_panel_data

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)

        assert pcse_est.n_periods > pcse_est.n_entities

    def test_warning_when_t_leq_n(self):
        """Test warning when T <= N."""
        # Create panel with T <= N (2 entities, 2 time periods)
        n_entities = 3
        n_periods = 2  # T < N
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        X = np.random.randn(n_entities * n_periods, 2)
        resid = np.random.randn(n_entities * n_periods)

        with pytest.warns(UserWarning, match="PCSE requires T > N"):
            PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)

    def test_reshape_panel(self, simple_panel_data):
        """Test reshaping residuals to panel format."""
        X, resid, entity_ids, time_ids = simple_panel_data

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        resid_matrix = pcse_est._reshape_panel()

        assert resid_matrix.shape == (3, 5)  # (N x T)
        # Should not have NaN for balanced panel
        assert not np.any(np.isnan(resid_matrix))

    def test_estimate_sigma(self, simple_panel_data):
        """Test estimation of contemporaneous covariance matrix."""
        X, resid, entity_ids, time_ids = simple_panel_data

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        sigma = pcse_est._estimate_sigma()

        assert sigma.shape == (3, 3)  # (N x N)
        # Should be symmetric
        assert np.allclose(sigma, sigma.T)
        # Diagonal should be positive
        assert np.all(np.diag(sigma) > 0)

    def test_compute_basic(self, simple_panel_data):
        """Test basic PCSE computation."""
        X, resid, entity_ids, time_ids = simple_panel_data

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        result = pcse_est.compute()

        assert isinstance(result, PCSEResult)
        assert result.cov_matrix.shape == (2, 2)
        assert result.std_errors.shape == (2,)
        assert result.sigma_matrix.shape == (3, 3)
        assert np.all(result.std_errors > 0)  # Standard errors should be positive

    def test_compute_larger_panel(self, large_panel_data):
        """Test PCSE with larger panel."""
        X, resid, entity_ids, time_ids = large_panel_data

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        result = pcse_est.compute()

        assert result.cov_matrix.shape == (3, 3)
        assert result.std_errors.shape == (3,)
        assert result.n_entities == 5
        assert result.n_periods == 20
        assert np.all(result.std_errors > 0)

    def test_covariance_matrix_symmetry(self, simple_panel_data):
        """Test that covariance matrix is symmetric."""
        X, resid, entity_ids, time_ids = simple_panel_data

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        result = pcse_est.compute()

        assert np.allclose(result.cov_matrix, result.cov_matrix.T)

    def test_std_errors_from_diagonal(self, simple_panel_data):
        """Test that standard errors are sqrt of diagonal elements."""
        X, resid, entity_ids, time_ids = simple_panel_data

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        result = pcse_est.compute()

        expected_se = np.sqrt(np.diag(result.cov_matrix))
        assert np.allclose(result.std_errors, expected_se)

    def test_singular_sigma_warning(self):
        """Test warning when sigma matrix is singular."""
        # Create perfectly collinear residuals to get singular sigma
        n_entities = 3
        n_periods = 5
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        X = np.random.randn(n_entities * n_periods, 2)
        # Make residuals identical across entities (will make sigma singular)
        base_resid = np.random.randn(n_periods)
        resid = np.tile(base_resid, n_entities)

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)

        with pytest.warns(UserWarning, match="singular"):
            result = pcse_est.compute()

        # Should still return a result using pseudoinverse
        assert isinstance(result, PCSEResult)

    def test_diagnostic_summary(self, simple_panel_data):
        """Test diagnostic summary generation."""
        X, resid, entity_ids, time_ids = simple_panel_data

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        summary = pcse_est.diagnostic_summary()

        assert "Panel-Corrected Standard Errors Diagnostics" in summary
        assert "Number of observations: 15" in summary
        assert "Number of entities (N): 3" in summary
        assert "Number of time periods (T): 5" in summary
        # T=5, N=3 so T < 2N, will show WARNING instead of T/N ratio
        assert "WARNING: T < 2N" in summary

    def test_diagnostic_summary_warning_t_leq_n(self):
        """Test diagnostic summary when T <= N."""
        n_entities = 5
        n_periods = 3  # T < N
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        X = np.random.randn(n_entities * n_periods, 2)
        resid = np.random.randn(n_entities * n_periods)

        with pytest.warns(UserWarning):
            pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)

        summary = pcse_est.diagnostic_summary()

        assert "CRITICAL: T ≤ N" in summary
        assert "PCSE requires T > N" in summary

    def test_diagnostic_summary_warning_t_less_2n(self):
        """Test diagnostic summary when T < 2N."""
        n_entities = 5
        n_periods = 8  # T > N but T < 2N
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        X = np.random.randn(n_entities * n_periods, 2)
        resid = np.random.randn(n_entities * n_periods)

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        summary = pcse_est.diagnostic_summary()

        assert "WARNING: T < 2N" in summary

    def test_string_entity_time_ids(self):
        """Test PCSE works with string IDs."""
        n_entities = 3
        n_periods = 5
        entity_ids = np.repeat(["A", "B", "C"], n_periods)
        time_ids = np.tile(["2020", "2021", "2022", "2023", "2024"], n_entities)

        X = np.random.randn(n_entities * n_periods, 2)
        resid = np.random.randn(n_entities * n_periods)

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        result = pcse_est.compute()

        assert isinstance(result, PCSEResult)
        assert result.n_entities == 3
        assert result.n_periods == 5


class TestPCSEConvenienceFunction:
    """Test the pcse() convenience function."""

    def test_convenience_function(self, simple_panel_data):
        """Test pcse() convenience function."""
        X, resid, entity_ids, time_ids = simple_panel_data

        result = pcse(X, resid, entity_ids, time_ids)

        assert isinstance(result, PCSEResult)
        assert result.cov_matrix.shape == (2, 2)
        assert result.std_errors.shape == (2,)

    def test_convenience_equals_class_method(self, simple_panel_data):
        """Test that convenience function gives same result as class method."""
        X, resid, entity_ids, time_ids = simple_panel_data

        # Using convenience function
        result1 = pcse(X, resid, entity_ids, time_ids)

        # Using class method
        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)
        result2 = pcse_est.compute()

        assert np.allclose(result1.std_errors, result2.std_errors)
        assert np.allclose(result1.cov_matrix, result2.cov_matrix)


class TestUnbalancedPanel:
    """Test PCSE with unbalanced panels."""

    def test_unbalanced_panel(self):
        """Test PCSE with missing observations."""
        # Create unbalanced panel (some entity-time combinations missing)
        entity_ids = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        time_ids = np.array([0, 1, 2, 3, 0, 1, 3, 0, 1, 2, 3])

        X = np.random.randn(11, 2)
        resid = np.random.randn(11)

        pcse_est = PanelCorrectedStandardErrors(X, resid, entity_ids, time_ids)

        # Should handle unbalanced panel
        resid_matrix = pcse_est._reshape_panel()
        assert resid_matrix.shape == (3, 4)  # 3 entities, 4 time periods

        # Should have NaN for missing observations
        assert np.any(np.isnan(resid_matrix))

        # Should still compute (though may have issues with NaN)
        result = pcse_est.compute()
        assert isinstance(result, PCSEResult)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_parameter(self):
        """Test with single parameter (k=1)."""
        n_entities = 3
        n_periods = 6
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        X = np.random.randn(n_entities * n_periods, 1)
        resid = np.random.randn(n_entities * n_periods)

        result = pcse(X, resid, entity_ids, time_ids)

        assert result.cov_matrix.shape == (1, 1)
        assert result.std_errors.shape == (1,)

    def test_many_parameters(self):
        """Test with many parameters."""
        n_entities = 4
        n_periods = 15
        n_params = 8
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        X = np.random.randn(n_entities * n_periods, n_params)
        resid = np.random.randn(n_entities * n_periods)

        result = pcse(X, resid, entity_ids, time_ids)

        assert result.cov_matrix.shape == (n_params, n_params)
        assert result.std_errors.shape == (n_params,)

    def test_small_residuals(self):
        """Test with very small residuals."""
        n_entities = 3
        n_periods = 6
        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)

        X = np.random.randn(n_entities * n_periods, 2)
        resid = np.random.randn(n_entities * n_periods) * 1e-10

        result = pcse(X, resid, entity_ids, time_ids)

        # Should still work with very small residuals
        assert np.all(result.std_errors > 0)
        assert np.all(result.std_errors < 1e-5)  # Should be small
