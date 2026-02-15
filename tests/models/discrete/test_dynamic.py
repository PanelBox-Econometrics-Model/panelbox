"""
Tests for dynamic binary panel models.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.models.discrete.dynamic import DynamicBinaryPanel


class TestDynamicBinaryPanel:
    """Test suite for dynamic binary panel models."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)

        # Generate panel data
        self.n_entities = 100
        self.n_periods = 10
        self.n_vars = 2

        # Create panel structure
        self.entity = np.repeat(np.arange(self.n_entities), self.n_periods)
        self.time = np.tile(np.arange(self.n_periods), self.n_entities)

        # Generate exogenous variables
        self.X = np.random.randn(self.n_entities * self.n_periods, self.n_vars)

        # True parameters
        self.beta_true = np.array([0.5, -0.3])
        self.gamma_true = 0.4  # Lag coefficient
        self.sigma_u = 0.5

        # Generate dependent variable with dynamics
        self.y = np.zeros(self.n_entities * self.n_periods)

        for i in range(self.n_entities):
            # Random effect
            u_i = np.random.normal(0, self.sigma_u)

            for t in range(self.n_periods):
                idx = i * self.n_periods + t

                # Linear index
                linear_idx = self.X[idx] @ self.beta_true + u_i

                # Add lag if not first period
                if t > 0:
                    prev_idx = i * self.n_periods + t - 1
                    linear_idx += self.gamma_true * self.y[prev_idx]

                # Generate binary outcome
                self.y[idx] = int(stats.norm.cdf(linear_idx) > np.random.rand())

    def test_wooldridge_initial_conditions(self):
        """Test Wooldridge approach for initial conditions."""
        model = DynamicBinaryPanel(
            self.y,
            self.X,
            self.entity,
            self.time,
            initial_conditions="wooldridge",
            effects="pooled",
        )

        result = model.fit()

        assert result.converged
        assert hasattr(result, "gamma")  # Lag coefficient
        assert hasattr(result, "delta_y0")  # Initial value coefficient
        assert hasattr(result, "delta_xbar")  # X averages coefficients

        # Check that lag coefficient is positive (state dependence)
        assert result.gamma > 0

    def test_simple_initial_conditions(self):
        """Test simple approach (dropping first period)."""
        model = DynamicBinaryPanel(
            self.y, self.X, self.entity, self.time, initial_conditions="simple", effects="pooled"
        )

        result = model.fit()

        assert result.converged
        assert hasattr(result, "gamma")

        # Simple approach should have fewer parameters
        assert len(result.params) < 10  # beta + gamma only

    def test_random_effects(self):
        """Test random effects specification."""
        model = DynamicBinaryPanel(
            self.y,
            self.X,
            self.entity,
            self.time,
            initial_conditions="wooldridge",
            effects="random",
        )

        result = model.fit()

        assert result.converged
        assert hasattr(result, "sigma_u")
        assert result.sigma_u > 0  # Variance should be positive

    def test_state_dependence(self):
        """Test that model captures state dependence."""
        model = DynamicBinaryPanel(
            self.y,
            self.X,
            self.entity,
            self.time,
            initial_conditions="wooldridge",
            effects="pooled",
        )

        result = model.fit()

        # State dependence coefficient should be significant
        assert result.gamma > 0.1  # Reasonable threshold for simulation
        assert result.gamma < 1.0  # Should not be too large

    def test_predict(self):
        """Test prediction functionality."""
        model = DynamicBinaryPanel(
            self.y, self.X, self.entity, self.time, initial_conditions="simple", effects="pooled"
        )

        result = model.fit()
        predictions = result.predict()

        assert len(predictions) > 0
        assert np.all((predictions >= 0) & (predictions <= 1))

    def test_marginal_effects(self):
        """Test marginal effects calculation."""
        model = DynamicBinaryPanel(
            self.y, self.X, self.entity, self.time, initial_conditions="simple", effects="pooled"
        )

        result = model.fit()
        me = result.marginal_effects()

        assert len(me) == self.n_vars + 1  # Variables + lag
        assert np.all(np.isfinite(me))

    def test_summary(self):
        """Test summary output."""
        model = DynamicBinaryPanel(
            self.y,
            self.X,
            self.entity,
            self.time,
            initial_conditions="wooldridge",
            effects="random",
        )

        result = model.fit()
        summary = result.summary()

        assert "Dynamic Binary Panel Model" in summary
        assert "Initial Conditions: wooldridge" in summary
        assert "γ (lag)" in summary
        assert "σ_u" in summary

    def test_missing_identifiers(self):
        """Test error handling for missing entity/time identifiers."""
        with pytest.raises(ValueError, match="Entity and time identifiers required"):
            model = DynamicBinaryPanel(self.y, self.X, entity=None, time=None)
            model.fit()

    def test_data_preparation(self):
        """Test data preparation with lags."""
        model = DynamicBinaryPanel(
            self.y, self.X, self.entity, self.time, initial_conditions="simple"
        )

        model._prepare_data()

        # Check lagged variable creation
        assert model.endog_lagged is not None
        assert len(model.endog_lagged) == len(self.y)

        # First period should have NaN lags
        for i in range(self.n_entities):
            first_idx = i * self.n_periods
            assert np.isnan(model.endog_lagged[first_idx])

            # Other periods should have previous value
            for t in range(1, self.n_periods):
                idx = i * self.n_periods + t
                prev_idx = idx - 1
                assert model.endog_lagged[idx] == self.y[prev_idx]
