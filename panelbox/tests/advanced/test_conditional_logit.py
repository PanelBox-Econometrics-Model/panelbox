"""Tests for Conditional Logit."""

import numpy as np
import pandas as pd
import pytest


class TestConditionalLogit:
    """Test ConditionalLogit class."""

    def test_init(self, choice_data):
        """Test initialization."""
        from panelbox.models.discrete import ConditionalLogit

        model = ConditionalLogit(
            data=choice_data,
            choice_col="trip_id",
            alt_col="mode",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )

        assert model.n_choices == 100
        assert model.n_alts == 3
        assert model.n_params == 2

    def test_validate_data_missing_cols(self, choice_data):
        """Test validation catches missing columns."""
        from panelbox.models.discrete import ConditionalLogit

        bad_data = choice_data.drop(columns=["cost"])

        with pytest.raises(ValueError, match="Missing columns"):
            ConditionalLogit(
                data=bad_data,
                choice_col="trip_id",
                alt_col="mode",
                chosen_col="chosen",
                alt_varying_vars=["cost", "time"],
            )

    def test_validate_data_invalid_choice(self, choice_data):
        """Test validation catches invalid choices."""
        from panelbox.models.discrete import ConditionalLogit

        # Make a choice occasion with 2 chosen
        bad_data = choice_data.copy()
        bad_data.loc[1, "chosen"] = 1

        with pytest.raises(ValueError, match="exactly one chosen"):
            ConditionalLogit(
                data=bad_data,
                choice_col="trip_id",
                alt_col="mode",
                chosen_col="chosen",
                alt_varying_vars=["cost", "time"],
            )

    def test_fit(self, choice_data):
        """Test fitting."""
        from panelbox.models.discrete import ConditionalLogit

        model = ConditionalLogit(
            data=choice_data,
            choice_col="trip_id",
            alt_col="mode",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )

        result = model.fit()

        assert result.converged
        assert len(result.params) == 2

    def test_parameter_signs(self, choice_data):
        """Test that estimated signs are correct."""
        from panelbox.models.discrete import ConditionalLogit

        model = ConditionalLogit(
            data=choice_data,
            choice_col="trip_id",
            alt_col="mode",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )

        result = model.fit()

        # Both should be negative (higher cost/time = lower utility)
        assert result.params[0] < 0, "Cost coefficient should be negative"
        assert result.params[1] < 0, "Time coefficient should be negative"

    def test_predict_proba(self, choice_data):
        """Test probability predictions."""
        from panelbox.models.discrete import ConditionalLogit

        model = ConditionalLogit(
            data=choice_data,
            choice_col="trip_id",
            alt_col="mode",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )

        result = model.fit()
        probs = result.predict_proba()

        # Shape check
        assert probs.shape == (100, 3)

        # Probabilities sum to 1
        np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(100))

        # All in [0, 1]
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_fit_statistics(self, choice_data):
        """Test fit statistics computation."""
        from panelbox.models.discrete import ConditionalLogit

        model = ConditionalLogit(
            data=choice_data,
            choice_col="trip_id",
            alt_col="mode",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )

        result = model.fit()

        assert hasattr(result, "llf")
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")
        assert hasattr(result, "pseudo_r2")
        assert hasattr(result, "accuracy")

        assert result.pseudo_r2 >= 0
        assert result.accuracy >= 0 and result.accuracy <= 1

    def test_summary(self, choice_data):
        """Test summary output."""
        from panelbox.models.discrete import ConditionalLogit

        model = ConditionalLogit(
            data=choice_data,
            choice_col="trip_id",
            alt_col="mode",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )

        result = model.fit()
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Conditional Logit" in summary
        assert "cost" in summary
        assert "time" in summary
