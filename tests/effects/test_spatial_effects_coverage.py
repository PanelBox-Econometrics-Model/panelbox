"""Tests for panelbox.effects.spatial_effects module.

Covers compute_spatial_effects, _simulation_inference, _delta_method_inference,
SpatialEffectsResult, and _compute_pvalue to raise coverage to 85%+.
"""

import matplotlib

matplotlib.use("Agg")

import warnings
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelbox.effects.spatial_effects import (
    SpatialEffectsResult,
    _compute_pvalue,
    _delta_method_inference,
    _simulation_inference,
    compute_spatial_effects,
    spatial_impact_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _make_W(n=5):
    """Create a simple row-normalized spatial weight matrix."""
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i, j] = 1.0
    # Row-normalize
    row_sums = W.sum(axis=1, keepdims=True)
    W = W / row_sums
    return W


def _make_mock_result(model_type="SAR", n=5, with_theta=True):
    """Create a mock SpatialPanelResult for testing.

    Parameters
    ----------
    model_type : str
        'SAR' or 'SDM'
    n : int
        Size of spatial weight matrix
    with_theta : bool
        Whether to include W*x1 parameter for SDM
    """
    W = _make_W(n)

    # Build parameter names
    param_names = ["x1", "x2", "rho"]
    param_values = [0.5, -0.3, 0.4]

    if model_type == "SDM" and with_theta:
        param_names = ["x1", "x2", "W*x1", "W*x2", "rho"]
        param_values = [0.5, -0.3, 0.2, -0.1, 0.4]

    params = pd.Series(param_values, index=param_names)
    k = len(param_names)

    # Create a positive definite covariance matrix
    rng = np.random.default_rng(42)
    A = rng.standard_normal((k, k))
    cov_params = A @ A.T * 0.01  # small variances

    mock_result = MagicMock()
    mock_result.params = params
    mock_result.cov_params = cov_params
    mock_result.W = W
    mock_result.model.spatial_model_type = model_type

    return mock_result


def _make_effects_dict(var_names=("x1",)):
    """Create a minimal effects dict for testing inference functions."""
    effects = {}
    for var in var_names:
        effects[var] = {
            "direct": 0.6,
            "indirect": 0.2,
            "total": 0.8,
            "impact_matrix": np.eye(5) * 0.6,
        }
    return effects


# ---------------------------------------------------------------------------
# Tests for spatial_impact_matrix
# ---------------------------------------------------------------------------


class TestSpatialImpactMatrix:
    def test_sar(self):
        """Test impact matrix computation for SAR model."""
        W = _make_W(3)
        mat = spatial_impact_matrix(rho=0.3, beta=1.0, theta=None, W=W, model_type="SAR")
        assert mat.shape == (3, 3)
        # Direct effects on diagonal should be larger than off-diagonal
        assert np.mean(np.diag(mat)) > np.mean(mat - np.diag(np.diag(mat)))

    def test_sdm(self):
        """Test impact matrix computation for SDM model."""
        W = _make_W(3)
        mat = spatial_impact_matrix(rho=0.3, beta=1.0, theta=0.5, W=W, model_type="SDM")
        assert mat.shape == (3, 3)

    def test_sdm_no_theta_raises(self):
        """Test SDM without theta raises ValueError."""
        W = _make_W(3)
        with pytest.raises(ValueError, match="SDM requires theta"):
            spatial_impact_matrix(rho=0.3, beta=1.0, theta=None, W=W, model_type="SDM")

    def test_unknown_model_type_raises(self):
        """Test unknown model type raises ValueError."""
        W = _make_W(3)
        with pytest.raises(ValueError, match="Unknown model type"):
            spatial_impact_matrix(rho=0.3, beta=1.0, theta=None, W=W, model_type="SEM")


# ---------------------------------------------------------------------------
# Tests for compute_spatial_effects
# ---------------------------------------------------------------------------


class TestComputeSpatialEffects:
    def test_sar(self):
        """Test compute_spatial_effects for SAR model."""
        mock_result = _make_mock_result("SAR")
        np.random.seed(123)
        res = compute_spatial_effects(
            mock_result,
            variables=["x1"],
            n_simulations=50,
            method="simulation",
        )
        assert isinstance(res, SpatialEffectsResult)
        assert "x1" in res.effects
        assert "direct" in res.effects["x1"]
        assert "indirect" in res.effects["x1"]
        assert "total" in res.effects["x1"]

    def test_sdm(self):
        """Test compute_spatial_effects for SDM model with theta."""
        mock_result = _make_mock_result("SDM")
        np.random.seed(123)
        res = compute_spatial_effects(
            mock_result,
            variables=["x1"],
            n_simulations=50,
            method="simulation",
        )
        assert isinstance(res, SpatialEffectsResult)
        assert "x1" in res.effects

    def test_invalid_method(self):
        """Test compute_spatial_effects raises for unknown method."""
        mock_result = _make_mock_result("SAR")
        with pytest.raises(ValueError, match="Unknown inference method"):
            compute_spatial_effects(mock_result, method="bootstrap")

    def test_no_spatial_model_attribute(self):
        """Test raises when result is not from spatial model."""
        mock_result = MagicMock()
        del mock_result.model.spatial_model_type
        with pytest.raises(ValueError, match="Result must be from a spatial model"):
            compute_spatial_effects(mock_result)

    def test_invalid_model_type(self):
        """Test raises for unsupported spatial model type (e.g., SEM)."""
        mock_result = MagicMock()
        mock_result.model.spatial_model_type = "SEM"
        with pytest.raises(
            ValueError, match="Effects decomposition only available for SAR and SDM"
        ):
            compute_spatial_effects(mock_result)

    def test_variables_none_uses_all(self):
        """Test that variables=None picks all exogenous variables."""
        mock_result = _make_mock_result("SAR")
        np.random.seed(42)
        res = compute_spatial_effects(
            mock_result,
            variables=None,
            n_simulations=50,
            method="simulation",
        )
        # Should include x1 and x2 (not rho, not sigma_*)
        assert "x1" in res.effects
        assert "x2" in res.effects
        assert "rho" not in res.effects

    def test_variables_string(self):
        """Test that a single string variable is handled."""
        mock_result = _make_mock_result("SAR")
        np.random.seed(42)
        res = compute_spatial_effects(
            mock_result,
            variables="x1",
            n_simulations=50,
            method="simulation",
        )
        assert "x1" in res.effects
        assert len(res.effects) == 1

    def test_variable_not_found_warns(self):
        """Test that a missing variable emits a warning."""
        mock_result = _make_mock_result("SAR")
        np.random.seed(42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_spatial_effects(
                mock_result,
                variables=["nonexistent"],
                n_simulations=50,
                method="simulation",
            )
            assert any("not found" in str(warning.message) for warning in w)

    def test_delta_method(self):
        """Test compute_spatial_effects with delta method inference."""
        mock_result = _make_mock_result("SAR")
        res = compute_spatial_effects(
            mock_result,
            variables=["x1"],
            method="delta",
        )
        assert isinstance(res, SpatialEffectsResult)
        assert res.method == "delta"

    def test_sdm_theta_missing_defaults_zero(self):
        """Test SDM model when W*var is not in params defaults theta to 0."""
        mock_result = _make_mock_result("SDM", with_theta=False)
        # Manually set model type to SDM but without W* params
        mock_result.model.spatial_model_type = "SDM"
        # params: x1, x2, rho (no W*x1, W*x2)
        mock_result.params = pd.Series([0.5, -0.3, 0.4], index=["x1", "x2", "rho"])
        k = 3
        rng = np.random.default_rng(42)
        A = rng.standard_normal((k, k))
        mock_result.cov_params = A @ A.T * 0.01

        np.random.seed(42)
        res = compute_spatial_effects(
            mock_result,
            variables=["x1"],
            n_simulations=50,
            method="simulation",
        )
        assert "x1" in res.effects


# ---------------------------------------------------------------------------
# Tests for _simulation_inference
# ---------------------------------------------------------------------------


class TestSimulationInference:
    def test_basic(self):
        """Test simulation-based inference produces expected keys."""
        mock_result = _make_mock_result("SAR")
        effects = _make_effects_dict(["x1"])
        np.random.seed(42)
        effects = _simulation_inference(
            mock_result, effects, ["x1"], n_simulations=100, confidence_level=0.95
        )
        assert "direct_se" in effects["x1"]
        assert "direct_ci" in effects["x1"]
        assert "direct_pvalue" in effects["x1"]
        assert "indirect_se" in effects["x1"]
        assert "total_se" in effects["x1"]
        # SE should be positive
        assert effects["x1"]["direct_se"] > 0

    def test_cholesky_fallback(self):
        """Test SVD fallback when covariance is not positive definite."""
        mock_result = _make_mock_result("SAR")
        # Make covariance matrix NOT positive definite
        k = len(mock_result.params)
        bad_cov = -np.eye(k) * 0.01  # Negative diagonal
        mock_result.cov_params = bad_cov

        effects = _make_effects_dict(["x1"])
        np.random.seed(42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            effects = _simulation_inference(
                mock_result, effects, ["x1"], n_simulations=50, confidence_level=0.95
            )
            assert any("not positive definite" in str(warning.message) for warning in w)

    def test_rho_bounds(self):
        """Test rho bounds checking: rho is clipped when abs(rho_sim) >= 0.99."""
        mock_result = _make_mock_result("SAR")
        # Set rho very close to 1 so draws exceed bounds
        mock_result.params = mock_result.params.copy()
        mock_result.params["rho"] = 0.98

        # Make large variance for rho so draws exceed 0.99
        k = len(mock_result.params)
        rng = np.random.default_rng(10)
        A = rng.standard_normal((k, k))
        cov = A @ A.T * 0.1  # Larger variance
        mock_result.cov_params = cov

        effects = _make_effects_dict(["x1"])
        np.random.seed(42)
        effects = _simulation_inference(
            mock_result, effects, ["x1"], n_simulations=200, confidence_level=0.95
        )
        # Should still produce valid results
        assert "direct_se" in effects["x1"]
        assert not np.isnan(effects["x1"]["direct_se"])

    def test_sdm_simulation(self):
        """Test simulation inference for SDM model."""
        mock_result = _make_mock_result("SDM")
        effects = _make_effects_dict(["x1"])
        np.random.seed(42)
        effects = _simulation_inference(
            mock_result, effects, ["x1"], n_simulations=100, confidence_level=0.95
        )
        assert "direct_se" in effects["x1"]
        assert effects["x1"]["direct_se"] > 0

    def test_variable_not_in_effects_skipped(self):
        """Test that variables not in effects dict are skipped."""
        mock_result = _make_mock_result("SAR")
        effects = _make_effects_dict(["x1"])
        np.random.seed(42)
        # Pass "x2" in variables list but it is not in effects dict
        effects = _simulation_inference(
            mock_result, effects, ["x1", "nonexistent"], n_simulations=50, confidence_level=0.95
        )
        # Should still work for x1
        assert "direct_se" in effects["x1"]


# ---------------------------------------------------------------------------
# Tests for _delta_method_inference
# ---------------------------------------------------------------------------


class TestDeltaMethodInference:
    def test_sar(self):
        """Test delta method for SAR model effects."""
        mock_result = _make_mock_result("SAR")
        effects = _make_effects_dict(["x1"])
        effects = _delta_method_inference(mock_result, effects, ["x1"], confidence_level=0.95)
        assert "direct_se" in effects["x1"]
        assert effects["x1"]["direct_se"] > 0
        assert "direct_ci" in effects["x1"]
        ci = effects["x1"]["direct_ci"]
        # CI should bracket the point estimate
        assert ci[0] <= effects["x1"]["direct"]
        assert ci[1] >= effects["x1"]["direct"]
        # P-value should be between 0 and 1
        assert 0 <= effects["x1"]["direct_pvalue"] <= 1

    def test_sdm(self):
        """Test delta method for SDM model with theta."""
        mock_result = _make_mock_result("SDM")
        effects = _make_effects_dict(["x1"])
        effects = _delta_method_inference(mock_result, effects, ["x1"], confidence_level=0.95)
        assert "direct_se" in effects["x1"]
        assert effects["x1"]["direct_se"] > 0

    def test_sdm_without_theta_param(self):
        """Test delta method for SDM when W*var not in params."""
        mock_result = _make_mock_result("SDM", with_theta=False)
        mock_result.model.spatial_model_type = "SDM"
        mock_result.params = pd.Series([0.5, -0.3, 0.4], index=["x1", "x2", "rho"])
        k = 3
        rng = np.random.default_rng(42)
        A = rng.standard_normal((k, k))
        mock_result.cov_params = A @ A.T * 0.01

        effects = _make_effects_dict(["x1"])
        effects = _delta_method_inference(mock_result, effects, ["x1"], confidence_level=0.95)
        assert "direct_se" in effects["x1"]
        assert effects["x1"]["direct_se"] > 0

    def test_delta_zero_se(self):
        """Test delta method handles zero SE (produces nan p-value)."""
        mock_result = _make_mock_result("SAR")
        # Zero covariance => zero SE
        k = len(mock_result.params)
        mock_result.cov_params = np.zeros((k, k))

        effects = _make_effects_dict(["x1"])
        effects = _delta_method_inference(mock_result, effects, ["x1"], confidence_level=0.95)
        assert np.isnan(effects["x1"]["direct_pvalue"])


# ---------------------------------------------------------------------------
# Tests for SpatialEffectsResult
# ---------------------------------------------------------------------------


def _make_spatial_effects_result(method="simulation"):
    """Create a SpatialEffectsResult for testing."""
    mock_result = _make_mock_result("SAR")
    effects = {
        "x1": {
            "direct": 0.6,
            "indirect": 0.2,
            "total": 0.8,
            "impact_matrix": np.eye(5) * 0.6,
            "direct_se": 0.1,
            "indirect_se": 0.05,
            "total_se": 0.12,
            "direct_ci": (0.4, 0.8),
            "indirect_ci": (0.1, 0.3),
            "total_ci": (0.56, 1.04),
            "direct_pvalue": 0.001,
            "indirect_pvalue": 0.02,
            "total_pvalue": 0.0005,
        },
    }
    return SpatialEffectsResult(
        effects=effects,
        model_result=mock_result,
        method=method,
        n_simulations=1000 if method == "simulation" else None,
        confidence_level=0.95,
    )


class TestSpatialEffectsResult:
    def test_summary(self):
        """Test SpatialEffectsResult.summary() returns DataFrame."""
        ser = _make_spatial_effects_result()
        df = ser.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # direct, indirect, total for x1
        assert "Variable" in df.columns
        assert "Effect" in df.columns
        assert "Estimate" in df.columns
        assert "P-value" in df.columns

    def test_summary_no_pvalues(self):
        """Test summary without p-values."""
        ser = _make_spatial_effects_result()
        df = ser.summary(show_pvalues=False)
        assert isinstance(df, pd.DataFrame)
        assert "P-value" not in df.columns

    def test_print_summary(self, capsys):
        """Test print_summary formatted output."""
        ser = _make_spatial_effects_result()
        ser.print_summary()
        captured = capsys.readouterr()
        assert "Spatial Effects Decomposition" in captured.out
        assert "SAR" in captured.out
        assert "simulation" in captured.out
        assert "1000" in captured.out
        assert "95.0%" in captured.out

    def test_print_summary_no_simulations(self, capsys):
        """Test print_summary when n_simulations is None (delta method)."""
        ser = _make_spatial_effects_result(method="delta")
        ser.print_summary()
        captured = capsys.readouterr()
        assert "Spatial Effects Decomposition" in captured.out
        assert "delta" in captured.out
        # Should NOT print "Simulations:" line
        assert "Simulations:" not in captured.out

    def test_plot_plotly(self):
        """Test plot with plotly backend."""
        pytest.importorskip("plotly")
        ser = _make_spatial_effects_result()
        fig = ser.plot(backend="plotly")
        assert fig is not None

    def test_plot_plotly_no_ci(self):
        """Test plotly plot without confidence intervals."""
        pytest.importorskip("plotly")
        ser = _make_spatial_effects_result()
        fig = ser.plot(backend="plotly", show_ci=False)
        assert fig is not None

    def test_plot_matplotlib(self):
        """Test plot with matplotlib backend."""
        ser = _make_spatial_effects_result()
        fig = ser.plot(backend="matplotlib")
        assert fig is not None

    def test_plot_matplotlib_no_ci(self):
        """Test matplotlib plot without CI (show_ci=False)."""
        ser = _make_spatial_effects_result()
        # Remove SE keys to test no-CI branch
        for var in ser.effects.values():
            for key in list(var.keys()):
                if key.endswith("_se"):
                    del var[key]
        fig = ser.plot(backend="matplotlib", show_ci=True)
        assert fig is not None

    def test_plot_invalid_backend(self):
        """Test plot raises for invalid backend."""
        ser = _make_spatial_effects_result()
        with pytest.raises(ValueError, match="Unknown backend"):
            ser.plot(backend="seaborn")

    def test_to_latex(self):
        """Test to_latex export."""
        ser = _make_spatial_effects_result()
        latex = ser.to_latex()
        assert isinstance(latex, str)
        assert "\\begin{tabular}" in latex
        assert "x1" in latex

    def test_to_latex_with_file(self, tmp_path):
        """Test to_latex with filename saves file."""
        ser = _make_spatial_effects_result()
        filepath = str(tmp_path / "effects.tex")
        latex = ser.to_latex(filename=filepath)
        assert isinstance(latex, str)
        with open(filepath) as f:
            content = f.read()
        assert "\\begin{tabular}" in content


# ---------------------------------------------------------------------------
# Tests for _compute_pvalue
# ---------------------------------------------------------------------------


class TestComputePvalue:
    def test_positive_estimate(self):
        """Test p-value for positive estimate."""
        # Simulated distribution mostly positive
        rng = np.random.default_rng(42)
        simulated = rng.normal(loc=2.0, scale=0.5, size=1000)
        pval = _compute_pvalue(2.0, simulated)
        assert 0 <= pval <= 1
        # Most values are > 0, so p-value should be small
        assert pval < 0.1

    def test_negative_estimate(self):
        """Test p-value for negative estimate."""
        # Simulated distribution mostly negative
        rng = np.random.default_rng(42)
        simulated = rng.normal(loc=-2.0, scale=0.5, size=1000)
        pval = _compute_pvalue(-2.0, simulated)
        assert 0 <= pval <= 1
        assert pval < 0.1

    def test_pvalue_capped_at_one(self):
        """Test p-value is capped at 1.0."""
        # Centered distribution around zero: p-value should be large
        rng = np.random.default_rng(42)
        simulated = rng.normal(loc=0.0, scale=1.0, size=1000)
        pval = _compute_pvalue(0.01, simulated)
        assert pval <= 1.0
