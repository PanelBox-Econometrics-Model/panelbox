"""
Comprehensive tests for Panel IV (Instrumental Variables) models.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.core.panel_data import PanelData
from panelbox.models.iv.panel_iv import PanelIV


@pytest.fixture
def simple_iv_data():
    """Create simple panel data with an endogenous regressor and instrument."""
    np.random.seed(42)
    n_entities = 50
    n_periods = 10

    # Create panel structure
    entities = np.repeat(range(n_entities), n_periods)
    time = np.tile(range(n_periods), n_entities)

    # Generate data with endogeneity
    z = np.random.randn(n_entities * n_periods)
    epsilon = np.random.randn(n_entities * n_periods)
    x = 2 * z + epsilon + np.random.randn(n_entities * n_periods)
    y = 1 + 3 * x + epsilon
    w = np.random.randn(n_entities * n_periods)

    df = pd.DataFrame({"entity": entities, "time": time, "y": y, "x": x, "z": z, "w": w})

    return df


class TestPanelIVInitialization:
    """Test PanelIV initialization and validation."""

    def test_init_basic_pooled(self, simple_iv_data):
        """Test basic pooled IV initialization."""
        iv = PanelIV(
            "y ~ w + x | w + z",
            simple_iv_data,
            entity_col="entity",
            time_col="time",
            model_type="pooled",
        )

        assert iv.model_type_iv == "pooled"
        assert "z" in iv.instruments
        assert "w" in iv.instruments

    def test_init_fixed_effects(self, simple_iv_data):
        """Test FE IV initialization."""
        iv = PanelIV(
            "y ~ w + x | w + z",
            simple_iv_data,
            entity_col="entity",
            time_col="time",
            model_type="fe",
        )

        assert iv.model_type_iv == "fe"

    def test_init_random_effects(self, simple_iv_data):
        """Test RE IV initialization."""
        iv = PanelIV(
            "y ~ w + x | w + z",
            simple_iv_data,
            entity_col="entity",
            time_col="time",
            model_type="re",
        )

        assert iv.model_type_iv == "re"

    def test_missing_pipe_separator(self, simple_iv_data):
        """Test error when IV separator is missing."""
        with pytest.raises(ValueError, match="must contain '\\|' separator"):
            PanelIV("y ~ x + z", simple_iv_data, entity_col="entity", time_col="time")

    def test_invalid_model_type(self, simple_iv_data):
        """Test error for invalid model type."""
        with pytest.raises(ValueError, match="model_type must be"):
            PanelIV(
                "y ~ x | z",
                simple_iv_data,
                entity_col="entity",
                time_col="time",
                model_type="invalid",
            )

    def test_instrument_not_in_data(self, simple_iv_data):
        """Test error when instrument is not in data."""
        with pytest.raises(ValueError, match=r"Instrument .* not found"):
            PanelIV("y ~ x | nonexistent_var", simple_iv_data, entity_col="entity", time_col="time")

    def test_no_endogenous_variables(self, simple_iv_data):
        """Test error when no endogenous variables are identified."""
        with pytest.raises(ValueError, match="No endogenous variables"):  # noqa: PT012
            iv = PanelIV("y ~ x | x", simple_iv_data, entity_col="entity", time_col="time")
            iv.fit()

    def test_under_identification(self, simple_iv_data):
        """Test error for under-identified model."""
        simple_iv_data["x2"] = np.random.randn(len(simple_iv_data))

        with pytest.raises(ValueError, match="under-identified"):  # noqa: PT012
            iv = PanelIV("y ~ x + x2 | z", simple_iv_data, entity_col="entity", time_col="time")
            iv.fit()


class TestPanelIVEstimation:
    """Test IV estimation for different specifications."""

    def test_pooled_iv_fit(self, simple_iv_data):
        """Test pooled IV estimation."""
        iv = PanelIV(
            "y ~ w + x | w + z",
            simple_iv_data,
            entity_col="entity",
            time_col="time",
            model_type="pooled",
        )

        results = iv.fit(cov_type="nonrobust")

        assert results is not None
        assert len(results.params) == 3
        assert "x" in results.params.index
        assert hasattr(results, "first_stage_results")
        assert "x" in results.first_stage_results

    def test_fixed_effects_iv_fit(self, simple_iv_data):
        """Test FE IV estimation."""
        iv = PanelIV(
            "y ~ w + x | w + z",
            simple_iv_data,
            entity_col="entity",
            time_col="time",
            model_type="fe",
        )

        results = iv.fit(cov_type="nonrobust")

        assert results is not None
        assert len(results.params) == 2
        assert "x" in results.params.index

    def test_random_effects_iv_fit(self, simple_iv_data):
        """Test RE IV estimation."""
        iv = PanelIV(
            "y ~ w + x | w + z",
            simple_iv_data,
            entity_col="entity",
            time_col="time",
            model_type="re",
        )

        results = iv.fit(cov_type="nonrobust")

        assert results is not None
        assert len(results.params) == 3

    def test_weak_instruments_warning(self, simple_iv_data):
        """Test weak instruments warning."""
        simple_iv_data["weak_z"] = np.random.randn(len(simple_iv_data))

        iv = PanelIV("y ~ x | weak_z", simple_iv_data, entity_col="entity", time_col="time")

        with pytest.warns(UserWarning, match="weak instruments"):
            iv.fit()

        assert iv.weak_instruments is True


class TestPanelIVCovarianceTypes:
    """Test different covariance estimators."""

    def test_nonrobust_covariance(self, simple_iv_data):
        """Test nonrobust standard errors."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit(cov_type="nonrobust")

        assert results is not None
        assert results.cov_type == "nonrobust"

    def test_hc0_covariance(self, simple_iv_data):
        """Test HC0 standard errors."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit(cov_type="HC0")
        assert results is not None
        assert results.cov_type == "HC0"

    def test_hc1_covariance(self, simple_iv_data):
        """Test HC1 standard errors."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit(cov_type="HC1")
        assert results is not None

    def test_invalid_cov_type(self, simple_iv_data):
        """Test error for invalid covariance type."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        with pytest.raises(ValueError, match="Unknown covariance type"):
            iv.fit(cov_type="invalid")


class TestPanelIVResults:
    """Test IV results properties and statistics."""

    def test_results_attributes(self, simple_iv_data):
        """Test that results have expected attributes."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit()

        assert hasattr(results, "params")
        assert hasattr(results, "std_errors")
        assert hasattr(results, "cov_params")
        assert hasattr(results, "resid")
        assert hasattr(results, "fittedvalues")

    def test_parameter_names_pooled(self, simple_iv_data):
        """Test parameter names for pooled model."""
        iv = PanelIV(
            "y ~ w + x | w + z",
            simple_iv_data,
            entity_col="entity",
            time_col="time",
            model_type="pooled",
        )

        results = iv.fit()

        assert "Intercept" in results.params.index
        assert "w" in results.params.index
        assert "x" in results.params.index

    def test_parameter_names_fe(self, simple_iv_data):
        """Test parameter names for FE model."""
        iv = PanelIV(
            "y ~ w + x | w + z",
            simple_iv_data,
            entity_col="entity",
            time_col="time",
            model_type="fe",
        )

        results = iv.fit()

        assert "Intercept" not in results.params.index
        assert "w" in results.params.index
        assert "x" in results.params.index


class TestPanelIVEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_endogenous_single_instrument(self, simple_iv_data):
        """Test exactly identified model with one of each."""
        iv = PanelIV("y ~ x | z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit()

        assert len(results.endogenous_vars) == 1
        assert results.n_instruments == 1

    def test_with_weights(self, simple_iv_data):
        """Test IV with observation weights."""
        np.random.seed(42)
        weights = np.random.uniform(0.5, 1.5, len(simple_iv_data))

        iv = PanelIV(
            "y ~ w + x | w + z",
            simple_iv_data,
            entity_col="entity",
            time_col="time",
            weights=weights,
        )

        results = iv.fit()
        assert results is not None

    def test_estimate_coefficients_not_implemented(self, simple_iv_data):
        """Test that _estimate_coefficients raises NotImplementedError."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        with pytest.raises(NotImplementedError, match="Use fit"):
            iv._estimate_coefficients()

    def test_overidentified_model(self, simple_iv_data):
        """Test overidentified model (more instruments than endogenous)."""
        simple_iv_data["z2"] = np.random.randn(len(simple_iv_data))

        iv = PanelIV("y ~ w + x | w + z + z2", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit()

        assert results.n_instruments > results.n_endogenous
        assert len(results.params) == 3


class TestPanelIVWithPanelData:
    """Test IV using PanelData objects."""

    def test_with_panel_data_object(self, simple_iv_data):
        """Test IV with PanelData instead of DataFrame."""
        panel_data = PanelData(simple_iv_data, entity_col="entity", time_col="time")

        iv = PanelIV("y ~ w + x | w + z", panel_data, entity_col="entity", time_col="time")

        results = iv.fit()
        assert results is not None

    def test_get_dataframe_from_panel_data(self, simple_iv_data):
        """Test _get_dataframe works with PanelData."""
        panel_data = PanelData(simple_iv_data, entity_col="entity", time_col="time")

        iv = PanelIV("y ~ w + x | w + z", panel_data, entity_col="entity", time_col="time")

        df = iv._get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(simple_iv_data)


class TestPanelIVUncoveredBranches:
    """Tests covering previously uncovered branches in panel_iv.py."""

    def test_clustered_covariance(self, simple_iv_data):
        """Test clustered covariance estimator."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit(cov_type="clustered")

        assert results is not None
        assert results.cov_type == "clustered"
        assert results.params is not None
        assert all(np.isfinite(results.std_errors))

    def test_clustered_covariance_custom_cluster(self, simple_iv_data):
        """Test clustered covariance with custom cluster variable."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        # Use time as an alternative cluster variable
        cluster_id = simple_iv_data["time"].values
        results = iv.fit(cov_type="clustered", cluster=cluster_id)

        assert results is not None
        assert results.cov_type == "clustered"

    def test_twoway_covariance(self, simple_iv_data):
        """Test two-way clustering covariance estimator."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit(cov_type="twoway")

        assert results is not None
        assert results.cov_type == "twoway"
        assert all(np.isfinite(results.std_errors))

    def test_driscoll_kraay_covariance(self, simple_iv_data):
        """Test Driscoll-Kraay covariance estimator."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit(cov_type="driscoll_kraay")

        assert results is not None
        assert results.cov_type == "driscoll_kraay"
        assert all(np.isfinite(results.std_errors))

    def test_driscoll_kraay_with_maxlags(self, simple_iv_data):
        """Test Driscoll-Kraay covariance with explicit maxlags."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit(cov_type="driscoll_kraay", maxlags=2)

        assert results is not None
        assert results.cov_type == "driscoll_kraay"

    def test_xtx_inv_pinv_fallback(self, simple_iv_data):
        """Test that pinv fallback is used when X'X is singular."""
        import unittest.mock

        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        original_inv = np.linalg.inv

        call_count = {"n": 0}

        def mock_inv(M):
            call_count["n"] += 1
            # Raise LinAlgError on the first call (XtX inversion in _compute_covariance)
            if call_count["n"] == 1:
                raise np.linalg.LinAlgError("Singular matrix")
            return original_inv(M)

        with unittest.mock.patch("numpy.linalg.inv", side_effect=mock_inv):
            results = iv.fit(cov_type="nonrobust")

        assert results is not None
        assert results.params is not None

    def test_model_info_attributes(self, simple_iv_data):
        """Test that IV-specific model info attributes are set on results."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit()

        assert results.endogenous_vars == ["x"]
        assert results.exogenous_vars == ["w"]
        assert "z" in results.instruments
        assert "w" in results.instruments
        assert results.n_instruments == 2
        assert results.n_endogenous == 1
        assert isinstance(results.weak_instruments, bool)

    def test_first_stage_f_statistic(self, simple_iv_data):
        """Test first stage F-statistic is computed."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit()

        assert "x" in results.first_stage_results
        fs = results.first_stage_results["x"]
        assert "f_statistic" in fs
        assert "rsquared" in fs
        assert "fitted" in fs
        assert "residuals" in fs
        assert "gamma" in fs

    def test_rsquared_adjusted(self, simple_iv_data):
        """Test R-squared and adjusted R-squared computation."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        results = iv.fit()

        assert hasattr(results, "rsquared")
        assert np.isfinite(results.rsquared)

    def test_get_dataframe_fallback_to_self_data(self, simple_iv_data):
        """Test _get_dataframe fallback when data has no .data attribute."""
        iv = PanelIV("y ~ w + x | w + z", simple_iv_data, entity_col="entity", time_col="time")

        # Temporarily replace data with a plain DataFrame (without .data attribute)
        original_data = iv.data
        iv.data = simple_iv_data  # DataFrame has no .data attribute
        df = iv._get_dataframe()
        assert isinstance(df, pd.DataFrame)

        # Restore
        iv.data = original_data
