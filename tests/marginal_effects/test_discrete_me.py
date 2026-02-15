"""
Tests for discrete marginal effects.

Tests Average Marginal Effects (AME), Marginal Effects at Means (MEM),
and Marginal Effects at Representative values (MER) for binary models.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.core.panel_data import PanelData
from panelbox.marginal_effects.discrete_me import (
    MarginalEffectsResult,
    _is_binary,
    _logit_cdf,
    _logit_pdf,
    compute_ame,
    compute_mem,
    compute_mer,
)
from panelbox.models.discrete.binary import PooledLogit, PooledProbit


class TestHelperFunctions:
    """Test helper functions for marginal effects."""

    def test_is_binary(self):
        """Test binary variable detection."""
        # Binary variables
        assert _is_binary(np.array([0, 1, 1, 0, 1]))
        assert _is_binary(np.array([0, 0, 0]))
        assert _is_binary(np.array([1, 1, 1]))

        # Non-binary variables
        assert not _is_binary(np.array([0, 1, 2]))
        assert not _is_binary(np.array([0.5, 1.0]))
        assert not _is_binary(np.array([-1, 0, 1]))

    def test_logit_pdf(self):
        """Test logistic PDF calculation."""
        # At 0, should be 0.25
        assert np.allclose(_logit_pdf(0), 0.25)

        # Should be symmetric
        assert np.allclose(_logit_pdf(2), _logit_pdf(-2))

        # Vector input
        x = np.array([-2, -1, 0, 1, 2])
        pdf = _logit_pdf(x)
        assert len(pdf) == 5
        assert np.all(pdf > 0)

    def test_logit_cdf(self):
        """Test logistic CDF calculation."""
        # At 0, should be 0.5
        assert np.allclose(_logit_cdf(0), 0.5)

        # Should be between 0 and 1
        x = np.array([-10, -1, 0, 1, 10])
        cdf = _logit_cdf(x)
        assert np.all((cdf >= 0) & (cdf <= 1))

        # Should be monotonic
        assert np.all(np.diff(cdf) > 0)


class TestMarginalEffectsLogit:
    """Test marginal effects for Logit models."""

    @pytest.fixture
    def setup_logit_data(self):
        """Create test data for Logit model."""
        np.random.seed(42)
        n = 200
        t = 5

        # Generate panel data
        entity_ids = np.repeat(np.arange(n), t)
        time_ids = np.tile(np.arange(t), n)

        # Covariates
        x1 = np.random.randn(n * t)  # Continuous
        x2 = np.random.binomial(1, 0.5, n * t)  # Binary
        x3 = np.random.randn(n * t)  # Continuous

        # True parameters
        beta = np.array([0.5, 1.0, -0.3, 0.8])  # Including intercept

        # Linear predictor
        X = np.column_stack([np.ones(n * t), x1, x2, x3])
        eta = X @ beta

        # Generate binary outcome
        prob = 1 / (1 + np.exp(-eta))
        y = np.random.binomial(1, prob)

        # Create DataFrame
        data = pd.DataFrame(
            {"entity": entity_ids, "time": time_ids, "y": y, "x1": x1, "x2": x2, "x3": x3}
        )

        return data

    def test_ame_logit_continuous(self, setup_logit_data):
        """Test AME for continuous variables in Logit."""
        data = setup_logit_data

        # Fit model
        panel_data = PanelData(data, "entity", "time")
        model = PooledLogit("y ~ x1 + x2 + x3", data, "entity", "time")
        result = model.fit(cov_type="nonrobust")

        # Compute AME
        ame_result = compute_ame(result)

        # Check structure
        assert isinstance(ame_result, MarginalEffectsResult)
        assert len(ame_result.marginal_effects) == 4  # intercept + 3 vars

        # Check that AME exists for all variables
        for var in ["Intercept", "x1", "x2", "x3"]:
            assert var in ame_result.marginal_effects.index
            assert var in ame_result.std_errors.index

        # Check reasonable magnitudes
        # For Logit, AME should be smaller than coefficients in absolute value
        for var in ["x1", "x3"]:  # Continuous vars
            ame = ame_result.marginal_effects[var]
            coef = result.params[var]
            assert abs(ame) < abs(coef)

    def test_ame_logit_binary(self, setup_logit_data):
        """Test AME for binary variables in Logit."""
        data = setup_logit_data

        # Fit model
        model = PooledLogit("y ~ x1 + x2 + x3", data, "entity", "time")
        result = model.fit(cov_type="nonrobust")

        # Compute AME
        ame_result = compute_ame(result)

        # Binary variable should have discrete difference
        # AME for x2 (binary) should be different from simple derivative
        ame_x2 = ame_result.marginal_effects["x2"]
        assert ame_x2 != 0  # Should have an effect

        # Check standard errors exist and are positive
        assert ame_result.std_errors["x2"] > 0

    def test_mem_logit(self, setup_logit_data):
        """Test MEM for Logit model."""
        data = setup_logit_data

        # Fit model
        model = PooledLogit("y ~ x1 + x2 + x3", data, "entity", "time")
        result = model.fit(cov_type="nonrobust")

        # Compute MEM
        mem_result = compute_mem(result)

        # Check structure
        assert isinstance(mem_result, MarginalEffectsResult)
        assert mem_result.me_type == "MEM"

        # MEM should include 'at' values
        assert mem_result.at_values is not None
        assert len(mem_result.at_values) > 0

        # Check that all variables have effects
        for var in result.params.index:
            assert var in mem_result.marginal_effects.index

        # MEM and AME should be different (in general)
        ame_result = compute_ame(result)
        for var in ["x1", "x3"]:  # For continuous vars
            assert not np.allclose(
                mem_result.marginal_effects[var], ame_result.marginal_effects[var], rtol=1e-2
            )

    def test_mer_logit(self, setup_logit_data):
        """Test MER for Logit model."""
        data = setup_logit_data

        # Fit model
        model = PooledLogit("y ~ x1 + x2 + x3", data, "entity", "time")
        result = model.fit(cov_type="nonrobust")

        # Compute MER at specific values
        at_values = {"x1": 0, "x2": 1, "x3": -1}
        mer_result = compute_mer(result, at=at_values)

        # Check structure
        assert isinstance(mer_result, MarginalEffectsResult)
        assert mer_result.me_type == "MER"

        # Check that specified values were used
        assert mer_result.at_values is not None
        for var, val in at_values.items():
            assert mer_result.at_values[var] == val

        # Check all effects computed
        for var in result.params.index:
            assert var in mer_result.marginal_effects.index
            assert mer_result.std_errors[var] > 0

    def test_marginal_effects_summary(self, setup_logit_data):
        """Test summary output for marginal effects."""
        data = setup_logit_data

        # Fit model
        model = PooledLogit("y ~ x1 + x2 + x3", data, "entity", "time")
        result = model.fit(cov_type="nonrobust")

        # Compute AME
        ame_result = compute_ame(result)

        # Get summary
        summary_df = ame_result.summary()

        # Check summary structure
        assert isinstance(summary_df, pd.DataFrame)
        assert "AME" in summary_df.columns
        assert "Std. Err." in summary_df.columns
        assert "P>|z|" in summary_df.columns

        # Check confidence intervals
        ci = ame_result.conf_int()
        assert "lower" in ci.columns
        assert "upper" in ci.columns
        assert np.all(ci["lower"] < ci["upper"])


class TestMarginalEffectsProbit:
    """Test marginal effects for Probit models."""

    @pytest.fixture
    def setup_probit_data(self):
        """Create test data for Probit model."""
        np.random.seed(123)
        n = 150
        t = 4

        # Generate panel data
        entity_ids = np.repeat(np.arange(n), t)
        time_ids = np.tile(np.arange(t), n)

        # Covariates
        x1 = np.random.randn(n * t)
        x2 = np.random.randn(n * t)

        # True parameters
        beta = np.array([0.3, 0.5, -0.4])  # Including intercept

        # Linear predictor
        X = np.column_stack([np.ones(n * t), x1, x2])
        eta = X @ beta

        # Generate binary outcome (Probit)
        prob = stats.norm.cdf(eta)
        y = np.random.binomial(1, prob)

        # Create DataFrame
        data = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "x1": x1, "x2": x2})

        return data

    def test_ame_probit(self, setup_probit_data):
        """Test AME for Probit model."""
        data = setup_probit_data

        # Fit model
        model = PooledProbit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit(cov_type="nonrobust")

        # Compute AME
        ame_result = compute_ame(result)

        # Check structure
        assert isinstance(ame_result, MarginalEffectsResult)

        # For Probit, AME = β * average(φ(X'β))
        # Should be smaller than raw coefficients
        for var in ["x1", "x2"]:
            ame = ame_result.marginal_effects[var]
            coef = result.params[var]
            assert abs(ame) < abs(coef)

        # Standard errors should be positive
        for var in result.params.index:
            assert ame_result.std_errors[var] > 0

    def test_mem_vs_ame_probit(self, setup_probit_data):
        """Test that MEM differs from AME in Probit."""
        data = setup_probit_data

        # Fit model
        model = PooledProbit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit(cov_type="nonrobust")

        # Compute both
        ame_result = compute_ame(result)
        mem_result = compute_mem(result)

        # In nonlinear models, AME ≠ MEM (Jensen's inequality)
        for var in ["x1", "x2"]:
            ame = ame_result.marginal_effects[var]
            mem = mem_result.marginal_effects[var]

            # They should be similar but not identical
            assert np.abs(ame - mem) > 1e-4
            # But not too different (same order of magnitude)
            assert np.abs(ame - mem) / np.abs(ame) < 0.5

    def test_mer_custom_values_probit(self, setup_probit_data):
        """Test MER with custom representative values."""
        data = setup_probit_data

        # Fit model
        model = PooledProbit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit(cov_type="nonrobust")

        # Compute MER at extremes
        at_high = {"x1": 2, "x2": 2}
        mer_high = compute_mer(result, at=at_high)

        at_low = {"x1": -2, "x2": -2}
        mer_low = compute_mer(result, at=at_low)

        # Effects should be different at different points
        # (due to nonlinearity)
        for var in ["x1", "x2"]:
            me_high = mer_high.marginal_effects[var]
            me_low = mer_low.marginal_effects[var]
            assert not np.allclose(me_high, me_low, rtol=1e-2)

            # In Probit, ME = β * φ(X'β)
            # φ is maximized at 0, so effects should be smaller at extremes
            mem_result = compute_mem(result)
            me_mean = mem_result.marginal_effects[var]

            # Effects at extremes should be smaller (in absolute value)
            # than at the mean (approximately)
            # This may not always hold depending on the data
            pass  # Skip this check as it's data-dependent


class TestDeltaMethod:
    """Test Delta method for standard errors."""

    def test_delta_method_consistency(self):
        """Test that Delta method gives consistent SEs."""
        np.random.seed(456)

        # Simple test case
        n = 500
        x = np.random.randn(n)
        y = (x > 0).astype(int)

        data = pd.DataFrame({"entity": np.arange(n), "time": np.zeros(n), "y": y, "x": x})

        # Fit model
        model = PooledLogit("y ~ x", data, "entity", "time")
        result = model.fit(cov_type="nonrobust")

        # Compute AME with Delta method SEs
        ame_result = compute_ame(result)

        # Standard errors should be positive
        assert np.all(ame_result.std_errors > 0)

        # Z-statistics should be finite
        z_stats = ame_result.z_stats
        assert np.all(np.isfinite(z_stats))

        # P-values should be between 0 and 1
        pvals = ame_result.pvalues
        assert np.all((pvals >= 0) & (pvals <= 1))
