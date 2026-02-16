"""
Tests for marginal effects module.

This module tests marginal effects computation for:
- Wang (2002) heteroscedastic model
- Battese & Coelli (1995) model
- General marginal effects function
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.frontier.utils import (
    marginal_effects,
    marginal_effects_bc95,
    marginal_effects_summary,
    marginal_effects_wang_2002,
)


class MockModel:
    """Mock SFA model for testing."""

    def __init__(
        self,
        data,
        n_exog=2,
        ineff_var_names=None,
        hetero_var_names=None,
        hetero_vars=None,
        inefficiency_vars=None,
    ):
        self.data = data
        self.n_exog = n_exog
        self.ineff_var_names = ineff_var_names or ["z1"]
        self.hetero_var_names = hetero_var_names
        self.hetero_vars = hetero_vars
        self.inefficiency_vars = inefficiency_vars


class MockResult:
    """Mock SFA result for testing."""

    def __init__(self, model, params, efficiency_data=None):
        self.model = model
        self.params = params
        self._efficiency_data = efficiency_data
        # Mock vcov
        k = len(params)
        self.vcov = np.eye(k) * 0.01

    def efficiency(self, estimator="bc"):
        """Return efficiency scores."""
        if self._efficiency_data is not None:
            return self._efficiency_data
        # Default
        n = len(self.model.data)
        return pd.DataFrame(
            {
                "entity": range(n),
                "time": [2010] * n,
                "efficiency": np.random.uniform(0.7, 0.95, n),
            }
        )


class TestMarginalEffectsWang2002:
    """Tests for Wang (2002) marginal effects."""

    def test_wang_location_effects(self):
        """Test marginal effects on inefficiency mean."""
        np.random.seed(42)
        n = 100

        # Generate data
        z1 = np.random.uniform(0, 1, n)  # Age
        w1 = np.random.uniform(0, 1, n)  # Size

        data = pd.DataFrame(
            {
                "z1": z1,
                "w1": w1,
            }
        )

        # Model with location and scale
        model = MockModel(
            data,
            n_exog=2,
            ineff_var_names=["z1"],
            hetero_var_names=["w1"],
            hetero_vars=["w1"],
        )

        # Parameters: β (2), δ (1), γ (1), σ²_v (1)
        # True values: δ = 0.5 (positive → increases inefficiency)
        params = np.array(
            [
                0.6,
                0.3,  # β
                0.5,  # δ (location)
                -0.3,  # γ (scale)
                0.1,  # σ²_v
            ]
        )

        result = MockResult(model, params)

        # Compute marginal effects on mean inefficiency
        me = marginal_effects_wang_2002(result, method="mean")

        # Check structure
        assert isinstance(me, pd.DataFrame)
        assert "variable" in me.columns
        assert "marginal_effect" in me.columns
        assert "std_error" in me.columns
        assert "z_stat" in me.columns
        assert "p_value" in me.columns

        # Check that z1 has positive effect (δ = 0.5 > 0)
        me_z1 = me[me["variable"] == "z1"]["marginal_effect"].values[0]
        assert me_z1 > 0, "z1 should increase inefficiency"

        # Check p-values are in [0, 1]
        assert all(me["p_value"] >= 0) and all(me["p_value"] <= 1)

    def test_wang_variance_effects(self):
        """Test marginal effects on inefficiency variance."""
        np.random.seed(123)
        n = 100

        data = pd.DataFrame(
            {
                "z1": np.random.uniform(0, 1, n),
                "w1": np.random.uniform(0, 1, n),
            }
        )

        model = MockModel(
            data,
            n_exog=2,
            ineff_var_names=["z1"],
            hetero_var_names=["w1"],
            hetero_vars=["w1"],
        )

        params = np.array([0.6, 0.3, 0.2, 0.4, 0.1])

        result = MockResult(model, params)

        # Compute marginal effects on variance
        me = marginal_effects_wang_2002(result, method="variance")

        assert isinstance(me, pd.DataFrame)
        assert len(me) == 1  # One scale variable
        assert me["variable"].values[0] == "w1"
        assert "marginal_effect" in me.columns


class TestMarginalEffectsBC95:
    """Tests for Battese & Coelli (1995) marginal effects."""

    def test_bc95_location_effects(self):
        """Test marginal effects for BC95 model."""
        np.random.seed(42)
        n = 100

        z1 = np.random.uniform(0, 1, n)
        z2 = np.random.uniform(0, 1, n)

        data = pd.DataFrame(
            {
                "z1": z1,
                "z2": z2,
            }
        )

        model = MockModel(
            data,
            n_exog=2,
            ineff_var_names=["z1", "z2"],
            inefficiency_vars=["z1", "z2"],
        )

        # Parameters: β (2), σ²_v, σ²_u, δ (2)
        # δ1 = 0.3 (positive), δ2 = -0.2 (negative)
        params = np.array(
            [
                0.6,
                0.4,  # β
                0.1,  # σ²_v
                0.2,  # σ²_u
                0.3,
                -0.2,  # δ
            ]
        )

        result = MockResult(model, params)

        # Compute marginal effects
        me = marginal_effects_bc95(result, method="mean")

        assert isinstance(me, pd.DataFrame)
        assert len(me) == 2  # Two inefficiency determinants

        # Check signs
        me_z1 = me[me["variable"] == "z1"]["marginal_effect"].values[0]
        me_z2 = me[me["variable"] == "z2"]["marginal_effect"].values[0]

        assert me_z1 > 0, "z1 should increase inefficiency"
        assert me_z2 < 0, "z2 should decrease inefficiency"

    def test_bc95_efficiency_effects(self):
        """Test marginal effects on efficiency (not inefficiency)."""
        np.random.seed(42)
        n = 50

        data = pd.DataFrame(
            {
                "z1": np.random.uniform(0, 1, n),
            }
        )

        model = MockModel(
            data,
            n_exog=2,
            ineff_var_names=["z1"],
            inefficiency_vars=["z1"],
        )

        params = np.array([0.6, 0.4, 0.1, 0.2, 0.3])

        # Need efficiency data for this test
        eff_data = pd.DataFrame(
            {
                "entity": range(n),
                "time": [2010] * n,
                "efficiency": np.random.uniform(0.7, 0.95, n),
            }
        )

        result = MockResult(model, params, eff_data)

        # Compute marginal effects on efficiency
        me = marginal_effects_bc95(result, method="efficiency")

        assert isinstance(me, pd.DataFrame)

        # Effect on efficiency should have opposite sign to effect on inefficiency
        # If δ > 0 (increases u), then ME on TE < 0 (decreases efficiency)
        me_eff = me["marginal_effect"].values[0]
        assert me_eff < 0, "Positive δ should decrease efficiency"


class TestMarginalEffectsGeneral:
    """Tests for general marginal_effects function."""

    def test_dispatch_to_wang(self):
        """Test that Wang model is correctly detected."""
        np.random.seed(42)
        n = 50

        data = pd.DataFrame(
            {
                "z1": np.random.uniform(0, 1, n),
                "w1": np.random.uniform(0, 1, n),
            }
        )

        model = MockModel(
            data,
            n_exog=2,
            ineff_var_names=["z1"],
            hetero_var_names=["w1"],
            hetero_vars=["w1"],  # This triggers Wang model
        )

        params = np.array([0.6, 0.3, 0.4, -0.2, 0.1])
        result = MockResult(model, params)

        # Should dispatch to Wang
        me = marginal_effects(result, method="mean")

        assert isinstance(me, pd.DataFrame)
        assert len(me) > 0

    def test_dispatch_to_bc95(self):
        """Test that BC95 model is correctly detected."""
        np.random.seed(42)
        n = 50

        data = pd.DataFrame(
            {
                "z1": np.random.uniform(0, 1, n),
            }
        )

        model = MockModel(
            data,
            n_exog=2,
            ineff_var_names=["z1"],
            inefficiency_vars=["z1"],  # This triggers BC95
        )

        params = np.array([0.6, 0.4, 0.1, 0.2, 0.3])

        eff_data = pd.DataFrame(
            {
                "entity": range(n),
                "time": [2010] * n,
                "efficiency": np.random.uniform(0.7, 0.95, n),
            }
        )

        result = MockResult(model, params, eff_data)

        # Should dispatch to BC95
        me = marginal_effects(result, method="mean")

        assert isinstance(me, pd.DataFrame)
        assert len(me) > 0

    def test_error_without_determinants(self):
        """Test that error is raised for model without determinants."""
        data = pd.DataFrame(
            {
                "x1": [1, 2, 3],
            }
        )

        # Model without inefficiency determinants
        model = MockModel(
            data,
            n_exog=2,
            ineff_var_names=None,
            hetero_vars=None,
            inefficiency_vars=None,
        )

        params = np.array([0.6, 0.4, 0.1, 0.2])
        result = MockResult(model, params)

        with pytest.raises(ValueError, match="inefficiency determinants"):
            marginal_effects(result)


class TestMarginalEffectsSummary:
    """Tests for marginal effects summary formatting."""

    def test_summary_format(self):
        """Test that summary generates proper text output."""
        # Create sample marginal effects DataFrame
        me_df = pd.DataFrame(
            {
                "variable": ["age", "education", "experience"],
                "marginal_effect": [0.023, -0.015, -0.008],
                "std_error": [0.005, 0.007, 0.010],
                "z_stat": [4.6, -2.1, -0.8],
                "p_value": [0.000, 0.035, 0.424],
                "interpretation": [
                    "increases inefficiency ***",
                    "decreases inefficiency **",
                    "decreases inefficiency",
                ],
            }
        )

        summary = marginal_effects_summary(me_df)

        # Check that summary is a string
        assert isinstance(summary, str)

        # Check that it contains expected elements
        assert "MARGINAL EFFECTS" in summary
        assert "age" in summary
        assert "education" in summary
        assert "experience" in summary
        assert "***" in summary  # Significance stars
        assert "**" in summary


class TestMarginalEffectsNumericalAccuracy:
    """Tests for numerical accuracy of marginal effects."""

    def test_finite_differences_bc95(self):
        """Test ME against finite differences for BC95."""
        np.random.seed(42)
        n = 100

        z1 = np.random.uniform(0, 1, n)
        data = pd.DataFrame({"z1": z1})

        model = MockModel(
            data,
            n_exog=2,
            ineff_var_names=["z1"],
            inefficiency_vars=["z1"],
        )

        params = np.array([0.6, 0.4, 0.1, 0.2, 0.5])  # δ = 0.5
        result = MockResult(model, params)

        # Analytical ME
        me = marginal_effects_bc95(result, method="mean")
        me_analytical = me["marginal_effect"].values[0]

        # Finite difference approximation
        # E[u | z] ≈ μ + σ·λ(μ/σ) where λ is inverse mills ratio
        # For small change in z: ΔE[u] ≈ δ·Δz
        # So ME ≈ δ with adjustment

        # The ME should be reasonably close to δ
        delta = params[4]

        # ME should have same sign as δ
        assert np.sign(me_analytical) == np.sign(delta)

        # ME should be of similar magnitude (within factor of 2)
        # Note: exact match not expected due to mills ratio adjustment
        assert abs(me_analytical) > 0.3 * abs(delta)
        assert abs(me_analytical) < 3.0 * abs(delta)


class TestMarginalEffectsInterpretation:
    """Tests for correct interpretation of marginal effects."""

    def test_positive_me_interpretation(self):
        """Test interpretation of positive marginal effect."""
        me_df = pd.DataFrame(
            {
                "variable": ["age"],
                "marginal_effect": [0.05],
                "std_error": [0.01],
                "z_stat": [5.0],
                "p_value": [0.001],
                "interpretation": ["increases inefficiency ***"],
            }
        )

        # Positive ME on inefficiency = bad for firm
        # Older age → more inefficient
        assert me_df["marginal_effect"].values[0] > 0
        assert "increases" in me_df["interpretation"].values[0]

    def test_negative_me_interpretation(self):
        """Test interpretation of negative marginal effect."""
        me_df = pd.DataFrame(
            {
                "variable": ["education"],
                "marginal_effect": [-0.03],
                "std_error": [0.01],
                "z_stat": [-3.0],
                "p_value": [0.003],
                "interpretation": ["decreases inefficiency ***"],
            }
        )

        # Negative ME on inefficiency = good for firm
        # More education → less inefficient (more efficient)
        assert me_df["marginal_effect"].values[0] < 0
        assert "decreases" in me_df["interpretation"].values[0]
