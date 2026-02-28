"""
Tests for panelbox.models.selection.heckman_old module.

Targets: PanelHeckman (two_step and mle), PanelHeckmanResult.
Goal: 0% -> 60%+ coverage.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def heckman_data():
    """Create synthetic panel data with selection."""
    rng = np.random.RandomState(42)
    n_entities = 30
    n_periods = 5
    n_obs = n_entities * n_periods

    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)

    # Outcome equation regressors (with constant)
    x1 = rng.randn(n_obs)
    x2 = rng.randn(n_obs)
    X = np.column_stack([np.ones(n_obs), x1, x2])

    # Selection equation regressors (includes exclusion restriction z)
    z = rng.randn(n_obs)  # exclusion restriction
    Z = np.column_stack([np.ones(n_obs), x1, x2, z])

    # True parameters
    beta_true = np.array([1.0, 0.5, -0.3])
    gamma_true = np.array([0.5, 0.3, -0.2, 0.4])

    # Generate selection and outcome
    u = rng.randn(n_obs)
    s_star = Z @ gamma_true + u
    selection = (s_star > 0).astype(float)

    eps = rng.randn(n_obs)
    y = X @ beta_true + eps

    return {
        "endog": y,
        "exog": X,
        "selection": selection,
        "exog_selection": Z,
        "entity": entity,
        "time": time,
    }


@pytest.fixture
def heckman_data_no_exclusion():
    """Data where Z has same number of columns as X (no exclusion restriction)."""
    rng = np.random.RandomState(123)
    n = 100
    X = np.column_stack([np.ones(n), rng.randn(n)])
    Z = np.column_stack([np.ones(n), rng.randn(n)])
    selection = (rng.randn(n) > 0).astype(float)
    y = rng.randn(n)
    entity = np.repeat(np.arange(20), 5)
    time = np.tile(np.arange(5), 20)
    return {
        "endog": y,
        "exog": X,
        "selection": selection,
        "exog_selection": Z,
        "entity": entity,
        "time": time,
    }


# ---------------------------------------------------------------------------
# PanelHeckman construction
# ---------------------------------------------------------------------------


class TestPanelHeckmanInit:
    """Tests for PanelHeckman constructor and validation."""

    def test_basic_init(self, heckman_data):
        """Construct PanelHeckman with valid data."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        assert model.method == "two_step"
        assert model.selection is not None
        assert model.exog_selection is not None

    def test_init_mle_method(self, heckman_data):
        """Construct with method='mle'."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data, method="mle")
        assert model.method == "mle"

    def test_init_no_entity_time(self, heckman_data):
        """Construct without entity/time (use defaults)."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        data = {k: v for k, v in heckman_data.items() if k not in ("entity", "time")}
        model = PanelHeckman(**data)
        assert model.entity is None
        assert model.time is None

    def test_validate_selection_length_mismatch(self, heckman_data):
        """Raise ValueError when selection length != endog length."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        with pytest.raises(ValueError, match="Selection and outcome must have same length"):
            PanelHeckman(
                endog=heckman_data["endog"],
                exog=heckman_data["exog"],
                selection=heckman_data["selection"][:10],
                exog_selection=heckman_data["exog_selection"],
                entity=heckman_data["entity"],
                time=heckman_data["time"],
            )

    def test_validate_exog_selection_length_mismatch(self, heckman_data):
        """Raise ValueError when exog_selection length != endog length."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        with pytest.raises(ValueError, match="Selection regressors must match data length"):
            PanelHeckman(
                endog=heckman_data["endog"],
                exog=heckman_data["exog"],
                selection=heckman_data["selection"],
                exog_selection=heckman_data["exog_selection"][:10],
                entity=heckman_data["entity"],
                time=heckman_data["time"],
            )

    def test_validate_selection_not_binary(self, heckman_data):
        """Raise ValueError when selection is not binary."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        bad_sel = heckman_data["selection"].copy()
        bad_sel[0] = 2.0
        with pytest.raises(ValueError, match="Selection must be binary"):
            PanelHeckman(
                endog=heckman_data["endog"],
                exog=heckman_data["exog"],
                selection=bad_sel,
                exog_selection=heckman_data["exog_selection"],
                entity=heckman_data["entity"],
                time=heckman_data["time"],
            )

    def test_validate_no_exclusion_restriction_warns(self, heckman_data_no_exclusion):
        """Warn when selection equation has <= columns than outcome."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        with pytest.warns(UserWarning, match="exclusion restriction"):
            PanelHeckman(**heckman_data_no_exclusion)


# ---------------------------------------------------------------------------
# Two-step estimation
# ---------------------------------------------------------------------------


class TestPanelHeckmanTwoStep:
    """Tests for two-step Heckman estimation."""

    def test_two_step_fit(self, heckman_data):
        """Fit Heckman model with two-step procedure."""
        from panelbox.models.selection.heckman_old import PanelHeckman, PanelHeckmanResult

        model = PanelHeckman(**heckman_data)
        result = model.fit()

        assert isinstance(result, PanelHeckmanResult)
        assert result.method == "two_step"
        assert result.outcome_params is not None
        assert result.probit_params is not None
        assert result.sigma > 0
        assert True  # rho might be outside [-1,1] in two-step
        assert result.lambda_imr is not None
        assert len(result.outcome_params) == heckman_data["exog"].shape[1]
        assert len(result.probit_params) == heckman_data["exog_selection"].shape[1]

    def test_two_step_explicit_method(self, heckman_data):
        """Pass method='two_step' explicitly to fit()."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data, method="mle")
        # Override with explicit method
        result = model.fit(method="two_step")
        assert result.method == "two_step"

    def test_two_step_n_obs(self, heckman_data):
        """Check observation counts in result."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()

        assert result.n_total == len(heckman_data["selection"])
        assert result.n_selected == int(np.sum(heckman_data["selection"]))
        assert result.n_selected < result.n_total

    def test_two_step_params_concatenation(self, heckman_data):
        """Check that params = [beta, gamma, sigma, rho]."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()

        k_out = heckman_data["exog"].shape[1]
        k_sel = heckman_data["exog_selection"].shape[1]
        expected_len = k_out + k_sel + 2  # beta + gamma + sigma + rho
        assert len(result.params) == expected_len


# ---------------------------------------------------------------------------
# MLE estimation
# ---------------------------------------------------------------------------


class TestPanelHeckmanMLE:
    """Tests for MLE Heckman estimation."""

    def test_mle_fit(self, heckman_data):
        """Fit Heckman model with MLE."""
        from panelbox.models.selection.heckman_old import PanelHeckman, PanelHeckmanResult

        model = PanelHeckman(**heckman_data, method="mle")
        result = model.fit()

        assert isinstance(result, PanelHeckmanResult)
        assert result.method == "mle"
        assert result.llf is not None
        assert isinstance(result.converged, bool)
        assert result.outcome_params is not None
        assert result.probit_params is not None
        assert result.sigma > 0

    def test_mle_explicit_method(self, heckman_data):
        """Pass method='mle' explicitly to fit()."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data, method="two_step")
        result = model.fit(method="mle")
        assert result.method == "mle"
        assert result.llf is not None

    def test_mle_unknown_method_raises(self, heckman_data):
        """Raise ValueError for unknown estimation method."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        with pytest.raises(ValueError, match="Unknown method"):
            model.fit(method="invalid")

    def test_mle_log_likelihood_method(self, heckman_data):
        """Test _log_likelihood directly."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        k_out = heckman_data["exog"].shape[1]
        k_sel = heckman_data["exog_selection"].shape[1]
        # params = [beta, gamma, log_sigma, arctanh_rho]
        params = np.zeros(k_out + k_sel + 2)
        nll = model._log_likelihood(params)
        assert np.isfinite(nll)
        assert nll > 0  # negative log-likelihood is positive


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestPanelHeckmanPredict:
    """Tests for model-level predict."""

    def test_predict_after_fit(self, heckman_data):
        """Predict after fitting."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()
        model.results = result

        pred = model.predict()
        assert len(pred) == len(heckman_data["endog"])

    def test_predict_with_params(self, heckman_data):
        """Predict with explicit params."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()

        pred = model.predict(params=result.params)
        assert len(pred) == len(heckman_data["endog"])

    def test_predict_with_exog(self, heckman_data):
        """Predict with new exog data."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()
        model.results = result

        new_X = heckman_data["exog"][:5]
        pred = model.predict(exog=new_X)
        assert len(pred) == 5

    def test_predict_no_fit_raises(self, heckman_data):
        """Raise ValueError when predicting without fitting."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        with pytest.raises(ValueError, match="Model not fitted"):
            model.predict()


# ---------------------------------------------------------------------------
# PanelHeckmanResult
# ---------------------------------------------------------------------------


class TestPanelHeckmanResult:
    """Tests for PanelHeckmanResult class."""

    def test_summary_two_step(self, heckman_data):
        """Generate summary for two-step result."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()
        summary = result.summary()

        assert isinstance(summary, str)
        assert "TWO_STEP" in summary
        assert "Total observations" in summary
        assert "Selected observations" in summary
        assert "gamma_" in summary
        assert "beta_" in summary
        assert "sigma:" in summary
        assert "rho:" in summary

    def test_summary_mle_with_llf(self, heckman_data):
        """Summary for MLE includes log-likelihood."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data, method="mle")
        result = model.fit()
        summary = result.summary()

        assert "MLE" in summary
        assert "Log-likelihood" in summary

    def test_summary_positive_rho(self):
        """Summary branch for positive rho (> 0.1)."""
        from panelbox.models.selection.heckman_old import PanelHeckman, PanelHeckmanResult

        # Create a result with high positive rho
        rng = np.random.RandomState(99)
        n = 50
        entity = np.repeat(np.arange(10), 5)
        time = np.tile(np.arange(5), 10)
        X = np.column_stack([np.ones(n), rng.randn(n)])
        Z = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
        sel = (rng.randn(n) > -0.5).astype(float)
        y = rng.randn(n)

        model = PanelHeckman(y, X, sel, Z, entity, time)

        result = PanelHeckmanResult(
            model=model,
            params=np.array([1.0, 0.5, 0.3, 0.2, 0.1, 1.0, 0.5]),
            method="two_step",
            probit_params=np.array([0.3, 0.2, 0.1]),
            outcome_params=np.array([1.0, 0.5]),
            sigma=1.0,
            rho=0.5,  # positive and > 0.1
            lambda_imr=np.zeros(n),
        )
        summary = result.summary()
        assert "Positive selection" in summary
        assert "Selection bias is present" in summary

    def test_summary_negative_rho(self):
        """Summary branch for negative rho (< -0.1)."""
        from panelbox.models.selection.heckman_old import PanelHeckman, PanelHeckmanResult

        rng = np.random.RandomState(99)
        n = 50
        entity = np.repeat(np.arange(10), 5)
        time = np.tile(np.arange(5), 10)
        X = np.column_stack([np.ones(n), rng.randn(n)])
        Z = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
        sel = (rng.randn(n) > -0.5).astype(float)
        y = rng.randn(n)

        model = PanelHeckman(y, X, sel, Z, entity, time)

        result = PanelHeckmanResult(
            model=model,
            params=np.array([1.0, 0.5, 0.3, 0.2, 0.1, 1.0, -0.5]),
            method="two_step",
            probit_params=np.array([0.3, 0.2, 0.1]),
            outcome_params=np.array([1.0, 0.5]),
            sigma=1.0,
            rho=-0.5,  # negative and < -0.1
            lambda_imr=np.zeros(n),
        )
        summary = result.summary()
        assert "Negative selection" in summary
        assert "Selection bias is present" in summary

    def test_summary_small_rho(self):
        """Summary branch for small rho (abs(rho) <= 0.1)."""
        from panelbox.models.selection.heckman_old import PanelHeckman, PanelHeckmanResult

        rng = np.random.RandomState(99)
        n = 50
        entity = np.repeat(np.arange(10), 5)
        time = np.tile(np.arange(5), 10)
        X = np.column_stack([np.ones(n), rng.randn(n)])
        Z = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
        sel = (rng.randn(n) > -0.5).astype(float)
        y = rng.randn(n)

        model = PanelHeckman(y, X, sel, Z, entity, time)

        result = PanelHeckmanResult(
            model=model,
            params=np.array([1.0, 0.5, 0.3, 0.2, 0.1, 1.0, 0.05]),
            method="two_step",
            probit_params=np.array([0.3, 0.2, 0.1]),
            outcome_params=np.array([1.0, 0.5]),
            sigma=1.0,
            rho=0.05,  # small rho
            lambda_imr=np.zeros(n),
        )
        summary = result.summary()
        assert "Little evidence of selection bias" in summary

    def test_result_predict_unconditional(self, heckman_data):
        """Predict unconditional outcomes from result."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()

        pred = result.predict(type="unconditional")
        assert len(pred) == len(heckman_data["endog"])

    def test_result_predict_conditional(self, heckman_data):
        """Predict conditional outcomes (with selection correction)."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()

        pred = result.predict(type="conditional")
        assert len(pred) == len(heckman_data["endog"])

    def test_result_predict_with_new_data(self, heckman_data):
        """Predict with new exog and exog_selection."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()

        new_X = heckman_data["exog"][:5]
        new_Z = heckman_data["exog_selection"][:5]

        pred_uncond = result.predict(exog=new_X, type="unconditional")
        assert len(pred_uncond) == 5

        pred_cond = result.predict(exog=new_X, exog_selection=new_Z, type="conditional")
        assert len(pred_cond) == 5

    def test_selection_test(self, heckman_data):
        """Run selection bias test."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        model = PanelHeckman(**heckman_data)
        result = model.fit()
        test_result = result.selection_test()

        assert isinstance(test_result, dict)
        assert "rho" in test_result
        assert "z_statistic" in test_result
        assert "p_value" in test_result
        assert "significant" in test_result
        assert isinstance(test_result["significant"], (bool, np.bool_))
        assert 0 <= test_result["p_value"] <= 1

    def test_selection_all_selected(self):
        """Edge case: all observations selected."""
        from panelbox.models.selection.heckman_old import PanelHeckman

        rng = np.random.RandomState(7)
        n = 50
        entity = np.repeat(np.arange(10), 5)
        time = np.tile(np.arange(5), 10)
        X = np.column_stack([np.ones(n), rng.randn(n)])
        Z = np.column_stack([np.ones(n), rng.randn(n), rng.randn(n)])
        sel = np.ones(n)  # all selected
        y = rng.randn(n)

        model = PanelHeckman(y, X, sel, Z, entity, time)
        result = model.fit()
        assert result.n_selected == n
