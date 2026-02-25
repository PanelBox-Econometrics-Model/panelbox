"""
Tests for panelbox.models.base module.

Covers uncovered lines:
  - 219-223: NonlinearPanelModel.fit() default start_params (zeros fallback)
  - 243-244: vcov fallback to pinv when inv raises LinAlgError
  - 277: PanelModelResults exog_names from model.exog.columns
  - 338: PanelModelResults.predict() with DataFrame exog but no exog_names
  - 343: PanelModelResults.predict() with exog=None
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from panelbox.models.base import NonlinearPanelModel, PanelModel, PanelModelResults

# ---------------------------------------------------------------------------
# Concrete subclasses for testing abstract base classes
# ---------------------------------------------------------------------------


class ConcreteLinearModel(PanelModel):
    """Minimal concrete PanelModel for testing."""

    def fit(self, **kwargs):
        self.fitted = True
        self.params = np.zeros(self.n_params)
        return self

    def predict(self, exog=None, **kwargs):
        if exog is not None:
            X = exog
        else:
            X = self.exog
        return X @ np.ones(X.shape[1])


class ConcreteNonlinearModel(NonlinearPanelModel):
    """Minimal NonlinearPanelModel with a simple log-likelihood.

    Uses a simple sum-of-squares likelihood so optimization converges quickly.
    Log-likelihood: -0.5 * sum((y - X @ params)^2)
    """

    def _log_likelihood(self, params):
        residuals = self.endog - self.exog @ params
        return -0.5 * np.sum(residuals**2)

    def predict(self, exog=None, **kwargs):
        if exog is not None:
            return exog @ self.params
        return self.exog @ self.params


class ConcreteNonlinearModelWithStartParams(ConcreteNonlinearModel):
    """NonlinearPanelModel subclass that provides _get_start_params."""

    def _get_start_params(self):
        return np.ones(self.n_params) * 0.5


# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_panel():
    """Simple panel data: 5 entities, 4 periods, 1 regressor."""
    np.random.seed(42)
    n_entities = 5
    n_periods = 4
    n_obs = n_entities * n_periods

    entity_id = np.repeat(np.arange(n_entities), n_periods)
    time_id = np.tile(np.arange(n_periods), n_entities)
    X = np.random.randn(n_obs, 1) * 2 + 5
    true_beta = np.array([3.0])
    y = X @ true_beta + np.random.randn(n_obs) * 0.5

    return y, X, entity_id, time_id


@pytest.fixture
def simple_panel_2d(simple_panel):
    """Same as simple_panel but with 2 regressors."""
    y_orig, X_orig, entity_id, time_id = simple_panel
    np.random.seed(99)
    n_obs = len(y_orig)
    X = np.column_stack([X_orig, np.random.randn(n_obs) * 3 + 10])
    true_beta = np.array([3.0, -1.0])
    y = X @ true_beta + np.random.randn(n_obs) * 0.5
    return y, X, entity_id, time_id


# ---------------------------------------------------------------------------
# Tests for NonlinearPanelModel.fit() — lines 219-223
# ---------------------------------------------------------------------------


class TestNonlinearFitDefaultStartParams:
    """Test NonlinearPanelModel.fit() when start_params is None."""

    def test_fit_defaults_to_zeros_no_get_start_params(self, simple_panel):
        """Lines 221-223: when no _get_start_params, defaults to zeros.

        ConcreteNonlinearModel does NOT have _get_start_params, so the
        fallback path (lines 221-223) creates np.zeros(n_params).
        """
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModel(y, X, entity_id, time_id)

        # fit without start_params — triggers lines 217-223
        result = model.fit(start_params=None, maxiter=500)

        assert isinstance(result, PanelModelResults)
        assert len(result.params) == 1
        # Optimization from zeros should converge near true beta ~3.0
        assert abs(result.params[0] - 3.0) < 1.0

    def test_fit_uses_get_start_params_when_available(self, simple_panel):
        """Lines 219-220: when _get_start_params exists, it is used."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModelWithStartParams(y, X, entity_id, time_id)

        result = model.fit(start_params=None, maxiter=500)

        assert isinstance(result, PanelModelResults)
        assert len(result.params) == 1

    def test_fit_with_explicit_start_params(self, simple_panel):
        """When start_params is provided explicitly, it is used (lines 217 false)."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModel(y, X, entity_id, time_id)

        result = model.fit(start_params=np.array([1.0]), maxiter=500)

        assert isinstance(result, PanelModelResults)
        assert len(result.params) == 1


# ---------------------------------------------------------------------------
# Tests for vcov fallback to pinv — lines 243-244
# ---------------------------------------------------------------------------


class TestNonlinearFitVcovFallback:
    """Test NonlinearPanelModel.fit() vcov pinv fallback."""

    def test_vcov_pinv_fallback_on_singular_hessian(self, simple_panel):
        """Lines 243-244: when np.linalg.inv raises LinAlgError, pinv is used."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModel(y, X, entity_id, time_id)

        # First fit normally to get params via optimization
        model.fit(maxiter=500)

        # Now patch np.linalg.inv to raise LinAlgError during fit

        def raising_inv(x):
            raise np.linalg.LinAlgError("Singular matrix")

        with patch("numpy.linalg.inv", side_effect=raising_inv):
            model2 = ConcreteNonlinearModel(y, X, entity_id, time_id)
            result_fallback = model2.fit(maxiter=500)

        assert isinstance(result_fallback, PanelModelResults)
        # vcov should still be computed via pinv
        assert result_fallback.vcov is not None
        assert result_fallback.vcov.shape == (1, 1)


# ---------------------------------------------------------------------------
# Tests for PanelModelResults — line 277, 338, 343
# ---------------------------------------------------------------------------


class TestPanelModelResultsExogNames:
    """Test PanelModelResults exog_names extraction from model.exog.columns."""

    def test_exog_names_from_dataframe_columns(self, simple_panel):
        """Line 277: exog_names extracted from model.exog.columns when it has columns attr."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteLinearModel(y, X, entity_id, time_id)
        model.fit()

        # Monkey-patch model.exog to be a DataFrame with columns attribute
        model.exog = pd.DataFrame(X, columns=["x1"])

        params = np.array([3.0])
        vcov = np.array([[0.1]])

        result = PanelModelResults(model, params, vcov)

        assert result.exog_names == ["x1"]

    def test_exog_names_none_when_no_columns(self, simple_panel):
        """When model.exog has no columns attribute, exog_names remains None."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteLinearModel(y, X, entity_id, time_id)
        model.fit()
        # model.exog is already ndarray (no .columns)

        params = np.array([3.0])
        vcov = np.array([[0.1]])

        result = PanelModelResults(model, params, vcov)

        assert result.exog_names is None

    def test_exog_names_from_model_attribute(self, simple_panel):
        """When model has exog_names attribute, it is used directly (line 274)."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteLinearModel(y, X, entity_id, time_id)
        model.fit()
        model.exog_names = ["regressor1"]

        params = np.array([3.0])
        vcov = np.array([[0.1]])

        result = PanelModelResults(model, params, vcov)

        assert result.exog_names == ["regressor1"]


class TestPanelModelResultsPredict:
    """Test PanelModelResults.predict() branches."""

    def test_predict_with_dataframe_no_exog_names(self, simple_panel):
        """Line 338: exog is DataFrame but exog_names is None -> exog.values."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModel(y, X, entity_id, time_id)
        result = model.fit(maxiter=500)

        # Ensure exog_names is None
        result.exog_names = None

        new_data = pd.DataFrame(X, columns=["x1"])
        predictions = result.predict(exog=new_data)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_with_dataframe_and_exog_names(self, simple_panel):
        """Lines 332-336: exog is DataFrame with matching exog_names."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModel(y, X, entity_id, time_id)
        result = model.fit(maxiter=500)
        result.exog_names = ["x1"]

        new_data = pd.DataFrame(X, columns=["x1"])
        predictions = result.predict(exog=new_data)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_with_dataframe_missing_columns(self, simple_panel):
        """Lines 333-335: DataFrame missing required columns raises ValueError."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModel(y, X, entity_id, time_id)
        result = model.fit(maxiter=500)
        result.exog_names = ["x1"]

        new_data = pd.DataFrame({"wrong_col": np.ones(5)})

        with pytest.raises(ValueError, match="Missing columns"):
            result.predict(exog=new_data)

    def test_predict_with_none_exog(self, simple_panel):
        """Line 343: exog=None calls model.predict() without exog."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModel(y, X, entity_id, time_id)
        result = model.fit(maxiter=500)

        predictions = result.predict(exog=None)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_with_ndarray_exog(self, simple_panel):
        """Lines 340-341: exog is ndarray passed through to model.predict()."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModel(y, X, entity_id, time_id)
        result = model.fit(maxiter=500)

        predictions = result.predict(exog=X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)


class TestPanelModelResultsSummary:
    """Test PanelModelResults.summary() method."""

    def test_summary_returns_string(self, simple_panel):
        """Test that summary() returns a formatted string."""
        y, X, entity_id, time_id = simple_panel
        model = ConcreteNonlinearModel(y, X, entity_id, time_id)
        result = model.fit(maxiter=500)

        summary = result.summary()

        assert isinstance(summary, str)
        assert "Model Results" in summary
        assert "Number of Obs" in summary
        assert "Param 0" in summary
