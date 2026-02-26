"""
Tests for quantile regression base class (base.py).

Targets uncovered lines 43-51, 109-180.
"""

import numpy as np
import pytest

from panelbox.core.panel_data import PanelData
from panelbox.models.quantile.base import QuantilePanelModel, QuantilePanelResult


class ConcreteQuantileModel(QuantilePanelModel):
    """Concrete subclass for testing the ABC."""

    def __init__(self, data, formula=None, tau=0.5, **kwargs):
        super().__init__(data, formula, tau, **kwargs)
        # Set up X, y from data
        if formula:
            self.y = data.data[self.dependent_var].values
            self.X = data.data[self.independent_vars].values
            self.X = np.column_stack([np.ones(len(self.y)), self.X])
        else:
            self.y = data.data.iloc[:, 0].values
            self.X = data.data.iloc[:, 1:].values
        self.k_exog = self.X.shape[1]

    def _objective(self, params, tau):
        residuals = self.y - self.X @ params
        return np.sum(self.check_loss(residuals, tau))


@pytest.fixture
def panel_data():
    """Create panel data for tests."""
    import pandas as pd

    np.random.seed(42)
    n_entities = 20
    n_time = 10
    n = n_entities * n_time

    entity_ids = np.repeat(np.arange(n_entities), n_time)
    time_ids = np.tile(np.arange(n_time), n_entities)
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)
    y = 1 + 2 * X1 - X2 + np.random.randn(n)

    df = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "X1": X1, "X2": X2})
    return PanelData(df, entity_col="entity", time_col="time")


class TestQuantilePanelModel:
    """Tests for the QuantilePanelModel base class."""

    def test_invalid_tau_raises(self, panel_data):
        """Test that tau outside (0,1) raises ValueError (lines 42-43)."""
        with pytest.raises(ValueError, match="Quantile levels tau must be in"):
            ConcreteQuantileModel(panel_data, formula="y ~ X1 + X2", tau=0.0)
        with pytest.raises(ValueError, match="Quantile levels tau must be in"):
            ConcreteQuantileModel(panel_data, formula="y ~ X1 + X2", tau=1.0)
        with pytest.raises(ValueError, match="Quantile levels tau must be in"):
            ConcreteQuantileModel(panel_data, formula="y ~ X1 + X2", tau=-0.1)

    def test_formula_parsing(self, panel_data):
        """Test _parse_formula (lines 46-47, 53-62)."""
        model = ConcreteQuantileModel(panel_data, formula="y ~ X1 + X2", tau=0.5)
        assert model.dependent_var == "y"
        assert model.independent_vars == ["X1", "X2"]

    def test_invalid_formula_raises(self, panel_data):
        """Test invalid formula raises ValueError (lines 61-62)."""
        with pytest.raises(ValueError, match="Invalid formula format"):
            ConcreteQuantileModel(panel_data, formula="y X1 X2", tau=0.5)

    def test_kwargs_stored_as_attributes(self, panel_data):
        """Test that extra kwargs become attributes (lines 49-51)."""
        model = ConcreteQuantileModel(
            panel_data, formula="y ~ X1 + X2", tau=0.5, custom_param=42, another_param="test"
        )
        assert model.custom_param == 42
        assert model.another_param == "test"

    def test_multiple_quantiles(self, panel_data):
        """Test initialization with multiple quantiles."""
        model = ConcreteQuantileModel(panel_data, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75])
        assert model.n_quantiles == 3
        np.testing.assert_array_equal(model.tau, [0.25, 0.5, 0.75])

    def test_check_loss_static(self):
        """Test check_loss static method (lines 64-71)."""
        u = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        tau = 0.5
        loss = QuantilePanelModel.check_loss(u, tau)
        # ρ_τ(u) = u*(τ - 1{u<0}) = 0.5*|u| for τ=0.5
        expected = 0.5 * np.abs(u)
        np.testing.assert_allclose(loss, expected)

    def test_check_loss_gradient_static(self):
        """Test check_loss_gradient static method (lines 73-80)."""
        u = np.array([-2.0, -1.0, 0.5, 1.0, 2.0])
        tau = 0.25
        grad = QuantilePanelModel.check_loss_gradient(u, tau)
        # τ - 1{u<0}: for negative u → 0.25-1 = -0.75, for positive u → 0.25
        expected = np.where(u < 0, tau - 1, tau)
        np.testing.assert_allclose(grad, expected)


class TestQuantilePanelResult:
    """Tests for the QuantilePanelResult class."""

    def test_summary_output(self, panel_data, capsys):
        """Test QuantilePanelResult.summary() (lines 162-180)."""
        model = ConcreteQuantileModel(panel_data, formula="y ~ X1 + X2", tau=0.5)
        results = {
            0.5: {
                "params": np.array([1.0, 2.0, -1.0]),
                "converged": True,
                "iterations": 10,
            }
        }
        result_obj = QuantilePanelResult(model, results)
        result_obj.summary()
        captured = capsys.readouterr()
        assert "Quantile Regression Results" in captured.out
        assert "τ = 0.500" in captured.out
        assert "Converged: True" in captured.out
        assert "Iterations: 10" in captured.out

    def test_summary_multiple_quantiles(self, panel_data, capsys):
        """Test summary with multiple quantiles (lines 168-178)."""
        model = ConcreteQuantileModel(panel_data, formula="y ~ X1 + X2", tau=[0.25, 0.75])
        results = {
            0.25: {"params": np.array([0.5, 1.5, -0.5]), "converged": True},
            0.75: {"params": np.array([1.5, 2.5, -1.5]), "converged": True},
        }
        result_obj = QuantilePanelResult(model, results)
        result_obj.summary()
        captured = capsys.readouterr()
        assert "τ = 0.250" in captured.out
        assert "τ = 0.750" in captured.out

    def test_summary_partial_info(self, panel_data, capsys):
        """Test summary with partial result info (missing keys, lines 173-178)."""
        model = ConcreteQuantileModel(panel_data, formula="y ~ X1 + X2", tau=0.5)
        # Only params, no converged or iterations
        results = {0.5: {"params": np.array([1.0, 2.0])}}
        result_obj = QuantilePanelResult(model, results)
        result_obj.summary()
        captured = capsys.readouterr()
        assert "Quantile Regression Results" in captured.out
        # "Converged" should not appear since it's not in the dict
        assert "Converged" not in captured.out

    def test_result_attributes(self, panel_data):
        """Test QuantilePanelResult attributes (lines 158-160)."""
        model = ConcreteQuantileModel(panel_data, formula="y ~ X1 + X2", tau=0.5)
        results = {0.5: {"params": np.array([1.0])}}
        result_obj = QuantilePanelResult(model, results)
        assert result_obj.model is model
        assert result_obj.results is results
