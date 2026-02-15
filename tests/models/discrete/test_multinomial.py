"""
Tests for multinomial logit panel models.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.models.discrete.multinomial import MultinomialLogit


class TestMultinomialLogit:
    """Test suite for multinomial logit models."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)

        # Generate panel data
        self.n_entities = 50
        self.n_periods = 5
        self.n_vars = 2
        self.n_alternatives = 3

        # Create panel structure
        self.entity = np.repeat(np.arange(self.n_entities), self.n_periods)
        self.time = np.tile(np.arange(self.n_periods), self.n_entities)
        self.n_obs = self.n_entities * self.n_periods

        # Generate exogenous variables
        self.X = np.random.randn(self.n_obs, self.n_vars)

        # True parameters (J-1 sets for J alternatives)
        self.beta_true = np.array(
            [[0.5, -0.3], [0.2, 0.4]]  # Alternative 1 vs base  # Alternative 2 vs base
        )

        # Generate choices
        self.y = np.zeros(self.n_obs, dtype=int)
        for i in range(self.n_obs):
            # Compute utilities
            util_0 = 0  # Base alternative
            util_1 = self.X[i] @ self.beta_true[0]
            util_2 = self.X[i] @ self.beta_true[1]

            # Probabilities via softmax
            utils = np.array([util_0, util_1, util_2])
            exp_utils = np.exp(utils - utils.max())  # Numerical stability
            probs = exp_utils / exp_utils.sum()

            # Generate choice
            self.y[i] = np.random.choice(self.n_alternatives, p=probs)

    def test_basic_estimation(self):
        """Test basic multinomial logit estimation."""
        model = MultinomialLogit(self.y, self.X, n_alternatives=self.n_alternatives)

        result = model.fit()

        assert result.converged
        assert hasattr(result, "params")
        assert len(result.params) == (self.n_alternatives - 1) * self.n_vars

    def test_predictions(self):
        """Test prediction functionality."""
        model = MultinomialLogit(self.y, self.X, n_alternatives=self.n_alternatives)

        result = model.fit()
        probs = result.predict_proba()

        # Check shape
        assert probs.shape == (self.n_obs, self.n_alternatives)

        # Check probabilities sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)

        # Check all probabilities in [0, 1]
        assert np.all((probs >= 0) & (probs <= 1))

    def test_marginal_effects(self):
        """Test marginal effects calculation."""
        model = MultinomialLogit(self.y, self.X, n_alternatives=self.n_alternatives)

        result = model.fit()
        me = result.marginal_effects()

        # Check shape
        assert me.shape == (self.n_alternatives, self.n_vars)

        # Marginal effects should sum to zero across alternatives
        assert np.allclose(me.sum(axis=0), 0)

    def test_different_base_alternative(self):
        """Test using different base alternative."""
        model = MultinomialLogit(
            self.y,
            self.X,
            n_alternatives=self.n_alternatives,
            base_alternative=1,  # Use alternative 1 as base
        )

        result = model.fit()
        assert result.converged

    def test_summary(self):
        """Test summary output."""
        model = MultinomialLogit(self.y, self.X, n_alternatives=self.n_alternatives)

        result = model.fit()
        summary = result.summary()

        assert "Multinomial Logit Results" in summary
        assert "Log-likelihood" in summary
        assert "Alternative" in summary

    def test_with_pandas_input(self):
        """Test with pandas DataFrame input."""
        df = pd.DataFrame({"choice": self.y, "x1": self.X[:, 0], "x2": self.X[:, 1]})

        model = MultinomialLogit(df["choice"], df[["x1", "x2"]], n_alternatives=self.n_alternatives)

        result = model.fit()
        assert result.converged

    def test_alternative_inference(self):
        """Test automatic inference of alternatives."""
        model = MultinomialLogit(
            self.y,
            self.X,
            # n_alternatives not specified
        )

        # Should infer from data
        assert model.n_alternatives == self.n_alternatives

    def test_invalid_choices(self):
        """Test error handling for invalid choice values."""
        y_invalid = self.y.copy()
        y_invalid[0] = 10  # Invalid choice

        with pytest.raises(ValueError):
            model = MultinomialLogit(y_invalid, self.X, n_alternatives=self.n_alternatives)
            model.fit()
