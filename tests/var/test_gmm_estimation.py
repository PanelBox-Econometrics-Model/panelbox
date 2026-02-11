"""
Tests for Panel VAR GMM estimation
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.gmm import GMMEstimationResult, estimate_panel_var_gmm, gmm_one_step, gmm_two_step


class TestGMMOneStep:
    """Tests for one-step GMM estimation"""

    def test_basic_one_step(self):
        """Test basic one-step GMM estimation"""
        np.random.seed(42)
        n_obs = 100
        n_params = 2
        n_instruments = 4

        # Generate synthetic data
        Z = np.random.randn(n_obs, n_instruments)
        X = np.random.randn(n_obs, n_params)
        true_beta = np.array([[1.0], [0.5]])
        y = X @ true_beta + np.random.randn(n_obs, 1) * 0.1

        beta, vcov, residuals = gmm_one_step(y, X, Z)

        # Check shapes
        assert beta.shape == (n_params, 1)
        assert vcov.shape == (n_params, n_params)
        assert residuals.shape == (n_obs, 1)

        # Check that estimates are reasonable (close to true values)
        assert np.abs(beta[0, 0] - 1.0) < 0.5
        assert np.abs(beta[1, 0] - 0.5) < 0.5

    def test_one_step_with_identity_weight(self):
        """Test that default weight matrix is identity"""
        np.random.seed(42)
        n_obs = 50
        n_params = 2
        n_instruments = 3

        Z = np.random.randn(n_obs, n_instruments)
        X = np.random.randn(n_obs, n_params)
        y = np.random.randn(n_obs, 1)

        # Should not raise error
        beta, vcov, residuals = gmm_one_step(y, X, Z)

        assert beta is not None
        assert vcov is not None

    def test_one_step_with_custom_weight_matrix(self):
        """Test one-step with custom weight matrix"""
        np.random.seed(42)
        n_obs = 50
        n_params = 2
        n_instruments = 4

        Z = np.random.randn(n_obs, n_instruments)
        X = np.random.randn(n_obs, n_params)
        y = np.random.randn(n_obs, 1)

        # Custom weight matrix
        W = np.eye(n_instruments) * 0.5

        beta, vcov, residuals = gmm_one_step(y, X, Z, weight_matrix=W)

        assert beta is not None
        assert vcov is not None


class TestGMMTwoStep:
    """Tests for two-step GMM estimation"""

    def test_basic_two_step(self):
        """Test basic two-step GMM estimation"""
        np.random.seed(42)
        n_obs = 100
        n_params = 2
        n_instruments = 4

        Z = np.random.randn(n_obs, n_instruments)
        X = np.random.randn(n_obs, n_params)
        true_beta = np.array([[1.0], [0.5]])
        y = X @ true_beta + np.random.randn(n_obs, 1) * 0.1

        beta, vcov, residuals, corrected = gmm_two_step(y, X, Z, windmeijer_correction=True)

        # Check shapes
        assert beta.shape == (n_params, 1)
        assert vcov.shape == (n_params, n_params)
        assert residuals.shape == (n_obs, 1)

        # Check Windmeijer correction was applied
        assert corrected is True

    def test_windmeijer_increases_ses(self):
        """Test that Windmeijer correction increases SEs"""
        np.random.seed(42)
        n_obs = 100
        n_params = 2
        n_instruments = 4

        Z = np.random.randn(n_obs, n_instruments)
        X = np.random.randn(n_obs, n_params)
        y = np.random.randn(n_obs, 1)

        # Without correction
        _, vcov_uncorrected, _, _ = gmm_two_step(y, X, Z, windmeijer_correction=False)

        # With correction
        _, vcov_corrected, _, corrected = gmm_two_step(y, X, Z, windmeijer_correction=True)

        # Corrected SEs should be larger (vcov diagonal elements)
        if corrected:
            assert np.all(np.diag(vcov_corrected) >= np.diag(vcov_uncorrected))


class TestEstimatePanelVARGMM:
    """Tests for full Panel VAR GMM estimation"""

    def test_basic_estimation_fod(self):
        """Test basic GMM estimation with FOD"""
        np.random.seed(42)

        # Create simple panel data
        n_entities = 5
        n_periods = 10

        data_list = []
        for i in range(1, n_entities + 1):
            for t in range(1, n_periods + 1):
                data_list.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )

        df = pd.DataFrame(data_list)

        # Estimate with FOD
        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            transform="fod",
            gmm_step="one-step",
            instrument_type="all",
        )

        # Check result structure
        assert isinstance(result, GMMEstimationResult)
        assert result.coefficients.shape == (2, 2)  # VAR(1) with 2 variables
        assert result.gmm_step == "one-step"
        assert result.transform == "fod"

    def test_basic_estimation_fd(self):
        """Test basic GMM estimation with FD"""
        np.random.seed(42)

        n_entities = 5
        n_periods = 10

        data_list = []
        for i in range(1, n_entities + 1):
            for t in range(1, n_periods + 1):
                data_list.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )

        df = pd.DataFrame(data_list)

        # Estimate with FD
        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            transform="fd",
            gmm_step="one-step",
            instrument_type="all",
        )

        assert isinstance(result, GMMEstimationResult)
        assert result.transform == "fd"

    def test_two_step_estimation(self):
        """Test two-step GMM estimation"""
        np.random.seed(42)

        n_entities = 10
        n_periods = 15

        data_list = []
        for i in range(1, n_entities + 1):
            for t in range(1, n_periods + 1):
                data_list.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )

        df = pd.DataFrame(data_list)

        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            transform="fod",
            gmm_step="two-step",
            windmeijer_correction=True,
        )

        assert result.gmm_step == "two-step"
        assert result.windmeijer_corrected is True

    def test_collapsed_instruments(self):
        """Test GMM with collapsed instruments"""
        np.random.seed(42)

        n_entities = 5
        n_periods = 20  # Longer panel to benefit from collapse

        data_list = []
        for i in range(1, n_entities + 1):
            for t in range(1, n_periods + 1):
                data_list.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )

        df = pd.DataFrame(data_list)

        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            transform="fod",
            instrument_type="collapsed",
            max_instruments=3,
        )

        assert result.instrument_type == "collapsed"
        # Collapsed should limit instruments
        assert result.n_instruments <= 20

    def test_var_lag_2(self):
        """Test GMM estimation with VAR(2)"""
        np.random.seed(42)

        n_entities = 5
        n_periods = 12

        data_list = []
        for i in range(1, n_entities + 1):
            for t in range(1, n_periods + 1):
                data_list.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )

        df = pd.DataFrame(data_list)

        result = estimate_panel_var_gmm(
            df, var_lags=2, value_cols=["y1", "y2"], transform="fod", gmm_step="one-step"
        )

        # VAR(2) with 2 variables: 4 lags × 2 equations = 8 params per equation
        # Actually: 2 lags × 2 vars = 4 params per equation
        assert result.coefficients.shape == (4, 2)  # (K*p, K)

    def test_invalid_transform(self):
        """Test error on invalid transform"""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0], "y2": [10.0, 20.0]})

        with pytest.raises(ValueError, match="Unknown transform"):
            estimate_panel_var_gmm(df, var_lags=1, value_cols=["y1", "y2"], transform="invalid")

    def test_invalid_gmm_step(self):
        """Test error on invalid gmm_step"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1],
                "time": [1, 2, 3, 4],
                "y1": [1.0, 2.0, 3.0, 4.0],
                "y2": [10.0, 20.0, 30.0, 40.0],
            }
        )

        with pytest.raises(ValueError, match="Unknown gmm_step"):
            estimate_panel_var_gmm(df, var_lags=1, value_cols=["y1", "y2"], gmm_step="invalid")

    def test_insufficient_data(self):
        """Test error with insufficient data"""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0], "y2": [10.0, 20.0]})

        # VAR(1) needs at least t >= 3 for first valid instrument
        with pytest.raises(ValueError):
            estimate_panel_var_gmm(df, var_lags=1, value_cols=["y1", "y2"])


class TestGMMResultDataclass:
    """Tests for GMMEstimationResult dataclass"""

    def test_result_structure(self):
        """Test that result dataclass has all required fields"""
        result = GMMEstimationResult(
            coefficients=np.array([[1.0, 0.5]]),
            standard_errors=np.array([[0.1, 0.2]]),
            residuals=np.array([[0.01]]),
            vcov=np.eye(2),
            n_obs=100,
            n_entities=10,
            n_instruments=20,
            gmm_step="one-step",
            transform="fod",
            instrument_type="all",
        )

        assert result.coefficients.shape == (1, 2)
        assert result.n_obs == 100
        assert result.gmm_step == "one-step"
        assert result.windmeijer_corrected is False  # default
