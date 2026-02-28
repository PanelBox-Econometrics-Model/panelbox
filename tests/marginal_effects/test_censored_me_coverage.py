"""
Deep coverage tests for panelbox.marginal_effects.censored_me.

Targets uncovered lines: 252-267 (probability branch in AME), 280 (cov_params
fallback), 397 (var not in exog_names), 412-418 (probability branch in MEM),
420-435 (MEM me_func probability), 446 (no cov_params fallback).
"""

import numpy as np
import pytest

from panelbox.marginal_effects.censored_me import (
    _inverse_mills_ratio,
    _mills_ratio_derivative,
    compute_tobit_ame,
    compute_tobit_mem,
)
from panelbox.marginal_effects.discrete_me import MarginalEffectsResult

# ---------------------------------------------------------------------------
# Helper to create a minimal Tobit-like model mock
# ---------------------------------------------------------------------------


class _MinimalTobitModel:
    """Minimal Tobit model for testing marginal effects without full estimation."""

    def __init__(self, X, beta, sigma, censoring_point=0.0, censoring_type="left"):
        self.exog = X
        self.beta = np.asarray(beta)
        self.sigma = sigma
        self.censoring_point = censoring_point
        self.censoring_type = censoring_type
        self.exog_names = [f"x{i}" for i in range(X.shape[1])]
        # Fake covariance: (K+1) x (K+1) for [beta..., sigma]
        n_params = len(beta) + 1
        self.cov_params = np.eye(n_params) * 0.01


class _MinimalTobitResult:
    """Minimal result wrapping a model."""

    def __init__(self, model):
        self.model = model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tobit_model_and_result():
    """Create a minimal Tobit model and result for testing."""
    np.random.seed(42)
    N = 200
    X = np.column_stack([np.ones(N), np.random.randn(N), np.random.randn(N)])
    beta = np.array([1.0, 0.5, -0.3])
    sigma = 1.5
    model = _MinimalTobitModel(X, beta, sigma, censoring_point=0.0)
    model.exog_names = ["const", "x1", "x2"]
    result = _MinimalTobitResult(model)
    return model, result


@pytest.fixture
def tobit_model_no_cov():
    """Tobit model without cov_params (triggers NaN SE fallback)."""
    np.random.seed(42)
    N = 100
    X = np.column_stack([np.ones(N), np.random.randn(N)])
    beta = np.array([0.5, 0.8])
    sigma = 1.0
    model = _MinimalTobitModel(X, beta, sigma, censoring_point=0.0)
    model.exog_names = ["const", "x1"]
    del model.cov_params  # Remove to trigger NaN fallback
    result = _MinimalTobitResult(model)
    return model, result


# ---------------------------------------------------------------------------
# Tests for helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Test inverse Mills ratio and its derivative."""

    def test_inverse_mills_ratio_normal_values(self):
        """Test IMR for normal z values."""
        z = np.array([0.0, 1.0, 2.0])
        imr = _inverse_mills_ratio(z)
        assert imr.shape == (3,)
        assert np.all(np.isfinite(imr))
        assert np.all(imr > 0)

    def test_inverse_mills_ratio_very_negative(self):
        """Cover the asymptotic branch for very negative z."""
        z = np.array([-50.0, -100.0])
        imr = _inverse_mills_ratio(z)
        # For very negative z, IMR ~ -z
        assert np.allclose(imr, -z, rtol=0.1)

    def test_mills_ratio_derivative(self):
        """Test derivative of IMR."""
        z = np.array([0.0, 1.0, -1.0])
        deriv = _mills_ratio_derivative(z)
        assert deriv.shape == (3,)
        # Derivative should be negative (IMR is decreasing)
        assert np.all(deriv < 0)


# ---------------------------------------------------------------------------
# compute_tobit_ame
# ---------------------------------------------------------------------------


class TestComputeTobitAME:
    """Cover all branches in compute_tobit_ame."""

    def test_ame_unconditional(self, tobit_model_and_result):
        """Cover lines 228-233: unconditional AME."""
        _, result = tobit_model_and_result
        ame = compute_tobit_ame(result, which="unconditional")
        assert isinstance(ame, MarginalEffectsResult)
        assert "x1" in ame.marginal_effects.index
        assert np.isfinite(ame.marginal_effects["x1"])

    def test_ame_conditional(self, tobit_model_and_result):
        """Cover lines 235-240: conditional AME."""
        _, result = tobit_model_and_result
        ame = compute_tobit_ame(result, which="conditional")
        assert isinstance(ame, MarginalEffectsResult)
        assert "x1" in ame.marginal_effects.index

    def test_ame_probability(self, tobit_model_and_result):
        """Cover lines 242-247 (252-267 in me_func): probability AME."""
        _, result = tobit_model_and_result
        ame = compute_tobit_ame(result, which="probability")
        assert isinstance(ame, MarginalEffectsResult)
        assert "x1" in ame.marginal_effects.index
        # Probability ME should be positive for positive beta
        assert ame.marginal_effects["x1"] > 0

    def test_ame_invalid_which(self, tobit_model_and_result):
        """Cover line 174: ValueError for invalid 'which'."""
        _, result = tobit_model_and_result
        with pytest.raises(ValueError, match="which must be"):
            compute_tobit_ame(result, which="invalid")

    def test_ame_with_varlist(self, tobit_model_and_result):
        """Cover varlist filtering and line 222-223 (var not in exog_names)."""
        _, result = tobit_model_and_result
        ame = compute_tobit_ame(result, which="unconditional", varlist=["x1", "ghost"])
        assert "x1" in ame.marginal_effects.index
        assert "ghost" not in ame.marginal_effects.index

    def test_ame_no_cov_params(self, tobit_model_no_cov):
        """Cover lines 284-285: NaN SE when model has no cov_params."""
        _, result = tobit_model_no_cov
        ame = compute_tobit_ame(result, which="unconditional")
        assert "x1" in ame.marginal_effects.index
        assert np.isnan(ame.std_errors["x1"])

    def test_ame_no_model_attr(self):
        """Cover line 178: result without .model attribute."""
        np.random.seed(50)
        N = 50
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        model = _MinimalTobitModel(X, [0.5, 0.3], 1.0)
        model.exog_names = ["const", "x1"]
        # Pass model directly as result (no .model attribute)
        ame = compute_tobit_ame(model, which="unconditional")
        assert isinstance(ame, MarginalEffectsResult)

    def test_ame_no_beta_raises(self):
        """Cover line 185: ValueError when model has no beta."""

        class NoFitModel:
            pass

        model = NoFitModel()
        result = _MinimalTobitResult(model)
        with pytest.raises(ValueError, match="Model must be fitted"):
            compute_tobit_ame(result, which="conditional")

    def test_ame_right_censoring_raises(self):
        """Cover line 204: NotImplementedError for non-left censoring."""
        np.random.seed(60)
        N = 50
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        model = _MinimalTobitModel(X, [0.5, 0.3], 1.0, censoring_type="right")
        model.exog_names = ["const", "x1"]
        result = _MinimalTobitResult(model)
        with pytest.raises(NotImplementedError, match="left censoring"):
            compute_tobit_ame(result, which="conditional")


# ---------------------------------------------------------------------------
# compute_tobit_mem
# ---------------------------------------------------------------------------


class TestComputeTobitMEM:
    """Cover all branches in compute_tobit_mem."""

    def test_mem_unconditional(self, tobit_model_and_result):
        """Cover lines 402-405: unconditional MEM."""
        _, result = tobit_model_and_result
        mem = compute_tobit_mem(result, which="unconditional")
        assert isinstance(mem, MarginalEffectsResult)
        assert mem.at_values is not None
        assert "x1" in mem.marginal_effects.index

    def test_mem_conditional(self, tobit_model_and_result):
        """Cover lines 407-410: conditional MEM."""
        _, result = tobit_model_and_result
        mem = compute_tobit_mem(result, which="conditional")
        assert isinstance(mem, MarginalEffectsResult)
        assert "x1" in mem.marginal_effects.index

    def test_mem_probability(self, tobit_model_and_result):
        """Cover lines 412-415 and 420-435: probability MEM and me_func."""
        _, result = tobit_model_and_result
        mem = compute_tobit_mem(result, which="probability")
        assert isinstance(mem, MarginalEffectsResult)
        assert "x1" in mem.marginal_effects.index
        # Probability ME should be positive for positive beta
        assert mem.marginal_effects["x1"] > 0

    def test_mem_invalid_which(self, tobit_model_and_result):
        """Cover line 344: ValueError for invalid 'which'."""
        _, result = tobit_model_and_result
        with pytest.raises(ValueError, match="which must be"):
            compute_tobit_mem(result, which="bogus")

    def test_mem_with_varlist(self, tobit_model_and_result):
        """Cover varlist filtering and line 396-397."""
        _, result = tobit_model_and_result
        mem = compute_tobit_mem(result, which="conditional", varlist=["x2", "phantom"])
        assert "x2" in mem.marginal_effects.index
        assert "phantom" not in mem.marginal_effects.index

    def test_mem_no_cov_params(self, tobit_model_no_cov):
        """Cover line 446/450: NaN SE when model has no cov_params."""
        _, result = tobit_model_no_cov
        mem = compute_tobit_mem(result, which="unconditional")
        assert "x1" in mem.marginal_effects.index
        assert np.isnan(mem.std_errors["x1"])

    def test_mem_no_model_attr(self):
        """Cover line 349: result without .model attribute."""
        np.random.seed(70)
        N = 50
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        model = _MinimalTobitModel(X, [0.5, 0.3], 1.0)
        model.exog_names = ["const", "x1"]
        mem = compute_tobit_mem(model, which="unconditional")
        assert isinstance(mem, MarginalEffectsResult)

    def test_mem_no_beta_raises(self):
        """Cover line 356: ValueError when model has no beta."""

        class NoFitModel:
            pass

        model = NoFitModel()
        result = _MinimalTobitResult(model)
        with pytest.raises(ValueError, match="Model must be fitted"):
            compute_tobit_mem(result, which="conditional")

    def test_mem_right_censoring_raises(self):
        """Cover line 374-376: NotImplementedError for non-left censoring."""
        np.random.seed(60)
        N = 50
        X = np.column_stack([np.ones(N), np.random.randn(N)])
        model = _MinimalTobitModel(X, [0.5, 0.3], 1.0, censoring_type="right")
        model.exog_names = ["const", "x1"]
        result = _MinimalTobitResult(model)
        with pytest.raises(NotImplementedError, match="left censoring"):
            compute_tobit_mem(result, which="conditional")
