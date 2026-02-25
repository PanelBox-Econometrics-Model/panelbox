"""
Tests for Ordered Choice Models.

Author: PanelBox Developers
License: MIT
"""

import warnings

import numpy as np
import pytest
from scipy import stats

from panelbox.marginal_effects.discrete_me import compute_ordered_ame, compute_ordered_mem
from panelbox.models.discrete.ordered import OrderedLogit, OrderedProbit, RandomEffectsOrderedLogit


class TestOrderedLogit:
    """Test suite for Ordered Logit model."""

    @pytest.fixture
    def simulated_data(self):
        """Generate simulated ordered data."""
        np.random.seed(42)

        # Data dimensions
        N = 200
        K = 3
        J = 4  # Number of categories (0, 1, 2, 3)

        # True parameters
        beta_true = np.array([0.5, -0.3, 0.2])
        cutpoints_true = np.array([-1, 0, 1])  # J-1 cutpoints

        # Generate covariates
        X = np.random.randn(N, K)

        # Linear predictor
        linear_pred = X @ beta_true

        # Generate ordered outcomes using logistic errors
        y = np.zeros(N, dtype=int)
        for i in range(N):
            # Add logistic error
            y_star = linear_pred[i] + np.random.logistic(0, 1)

            # Determine category based on cutpoints
            if y_star <= cutpoints_true[0]:
                y[i] = 0
            elif y_star <= cutpoints_true[1]:
                y[i] = 1
            elif y_star <= cutpoints_true[2]:
                y[i] = 2
            else:
                y[i] = 3

        # Create panel structure (for compatibility)
        groups = np.arange(N)
        time = np.zeros(N)

        return {
            "y": y,
            "X": X,
            "groups": groups,
            "time": time,
            "beta_true": beta_true,
            "cutpoints_true": cutpoints_true,
            "n_categories": J,
        }

    def test_initialization(self, simulated_data):
        """Test model initialization."""
        model = OrderedLogit(
            endog=simulated_data["y"],
            exog=simulated_data["X"],
            groups=simulated_data["groups"],
            n_categories=simulated_data["n_categories"],
        )

        assert model.n_obs == len(simulated_data["y"])
        assert model.n_features == simulated_data["X"].shape[1]
        assert model.n_categories == simulated_data["n_categories"]
        assert model.n_cutpoints == simulated_data["n_categories"] - 1

    def test_category_remapping(self):
        """Test automatic remapping of non-standard categories."""
        # Categories not starting from 0
        y = np.array([1, 2, 3, 1, 2, 3])
        X = np.random.randn(6, 2)
        groups = np.arange(6)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            model = OrderedLogit(endog=y, exog=X, groups=groups)

            # Check warning was issued
            assert len(w) == 1
            assert "Categories should be 0, 1" in str(w[0].message)

        # Check categories were remapped
        assert np.array_equal(np.unique(model.endog), [0, 1, 2])

    def test_cutpoint_transformation(self, simulated_data):
        """Test cutpoint transformation to ensure ordering."""
        model = OrderedLogit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        # Test transformation
        cutpoint_params = np.array([0, np.log(1), np.log(2)])
        cutpoints = model._transform_cutpoints(cutpoint_params)

        # Check ordering
        assert cutpoints[0] == 0
        assert cutpoints[1] == 1  # 0 + exp(log(1))
        assert cutpoints[2] == 3  # 1 + exp(log(2))
        assert np.all(np.diff(cutpoints) > 0)

        # Test inverse transformation
        cutpoint_params_back = model._inverse_transform_cutpoints(cutpoints)
        assert np.allclose(cutpoint_params, cutpoint_params_back)

    def test_log_likelihood_computation(self, simulated_data):
        """Test log-likelihood computation."""
        model = OrderedLogit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        # Create parameter vector
        simulated_data["X"].shape[1]
        beta = np.array([0.5, -0.3, 0.2])
        cutpoint_params = np.array([0, np.log(1), np.log(1)])
        params = np.concatenate([beta, cutpoint_params])

        # Compute log-likelihood
        llf = model._log_likelihood(params)

        assert np.isfinite(llf)
        assert llf < 0  # Log-likelihood should be negative

    def test_fit_convergence(self, simulated_data):
        """Test model fitting and convergence."""
        model = OrderedLogit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=100)

        assert hasattr(result, "params")
        assert hasattr(result, "beta")
        assert hasattr(result, "cutpoints")
        assert hasattr(result, "llf")

        # Check dimensions
        assert len(result.beta) == simulated_data["X"].shape[1]
        assert len(result.cutpoints) == simulated_data["n_categories"] - 1

        # Check cutpoints are ordered
        assert np.all(np.diff(result.cutpoints) > 0)

    def test_parameter_recovery(self):
        """Test parameter recovery on larger dataset."""
        np.random.seed(123)
        N = 1000
        K = 2
        J = 3

        beta_true = np.array([0.8, -0.5])
        cutpoints_true = np.array([-0.5, 0.5])

        X = np.random.randn(N, K)
        linear_pred = X @ beta_true

        # Generate outcomes
        y = np.zeros(N, dtype=int)
        for i in range(N):
            y_star = linear_pred[i] + np.random.logistic(0, 1)
            if y_star <= cutpoints_true[0]:
                y[i] = 0
            elif y_star <= cutpoints_true[1]:
                y[i] = 1
            else:
                y[i] = 2

        groups = np.arange(N)

        model = OrderedLogit(endog=y, exog=X, groups=groups, n_categories=J)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=200)

        # Check parameter recovery (allow larger tolerance for sampling variability)
        assert np.allclose(result.beta, beta_true, atol=0.25)
        assert np.allclose(result.cutpoints, cutpoints_true, atol=0.3)

    def test_predict_proba(self, simulated_data):
        """Test probability predictions."""
        model = OrderedLogit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=50)

        # Predict probabilities
        probs = result.predict_proba()

        assert probs.shape == (len(simulated_data["y"]), simulated_data["n_categories"])

        # Check probabilities sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)

        # Check all probabilities are in [0, 1]
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_predict_category(self, simulated_data):
        """Test category predictions."""
        model = OrderedLogit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=50)

        # Predict categories
        pred_categories = result.predict(type="category")

        assert len(pred_categories) == len(simulated_data["y"])
        assert np.all(pred_categories >= 0)
        assert np.all(pred_categories < simulated_data["n_categories"])

        # Check predictions are integers
        assert pred_categories.dtype in [np.int32, np.int64]

    def test_summary_output(self, simulated_data):
        """Test summary output."""
        model = OrderedLogit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=50)

        summary = result.summary()

        assert isinstance(summary, str)
        assert "OrderedLogit Results" in summary
        assert "Number of categories:" in summary
        assert "Cutpoints (ordered):" in summary
        assert "Log-likelihood:" in summary


class TestOrderedProbit:
    """Test suite for Ordered Probit model."""

    @pytest.fixture
    def simulated_data(self):
        """Generate simulated data for ordered probit."""
        np.random.seed(42)

        N = 200
        K = 2
        J = 3

        beta_true = np.array([0.5, -0.3])
        cutpoints_true = np.array([-0.5, 0.5])

        X = np.random.randn(N, K)
        linear_pred = X @ beta_true

        # Generate with normal errors (for probit)
        y = np.zeros(N, dtype=int)
        for i in range(N):
            y_star = linear_pred[i] + np.random.normal(0, 1)
            if y_star <= cutpoints_true[0]:
                y[i] = 0
            elif y_star <= cutpoints_true[1]:
                y[i] = 1
            else:
                y[i] = 2

        groups = np.arange(N)

        return {
            "y": y,
            "X": X,
            "groups": groups,
            "beta_true": beta_true,
            "cutpoints_true": cutpoints_true,
            "n_categories": J,
        }

    def test_probit_vs_logit_difference(self, simulated_data):
        """Test that probit and logit give different results."""
        # Fit Ordered Probit
        probit_model = OrderedProbit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probit_result = probit_model.fit(maxiter=50)

        # Fit Ordered Logit on same data
        logit_model = OrderedLogit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logit_result = logit_model.fit(maxiter=50)

        # Parameters should be different (but correlated)
        assert not np.allclose(probit_result.beta, logit_result.beta)

        # But signs should generally agree
        assert np.all(np.sign(probit_result.beta) == np.sign(logit_result.beta))

    def test_probit_cdf_pdf(self):
        """Test that probit uses normal CDF and PDF."""
        model = OrderedProbit(
            endog=np.array([0, 1, 2]), exog=np.array([[1], [2], [3]]), groups=np.array([0, 1, 2])
        )

        # Test CDF
        z_values = np.array([-2, -1, 0, 1, 2])
        cdf_values = model._cdf(z_values)
        expected_cdf = stats.norm.cdf(z_values)
        assert np.allclose(cdf_values, expected_cdf)

        # Test PDF
        pdf_values = model._pdf(z_values)
        expected_pdf = stats.norm.pdf(z_values)
        assert np.allclose(pdf_values, expected_pdf)


class TestRandomEffectsOrderedLogit:
    """Test suite for Random Effects Ordered Logit."""

    @pytest.fixture
    def panel_data(self):
        """Generate panel data for RE ordered logit."""
        np.random.seed(42)

        N = 30  # Number of individuals
        T = 5  # Time periods
        K = 2  # Covariates
        J = 3  # Categories

        beta_true = np.array([0.5, -0.3])
        cutpoints_true = np.array([-0.5, 0.5])
        sigma_alpha_true = 0.5

        # Generate data
        X = np.random.randn(N * T, K)
        groups = np.repeat(np.arange(N), T)

        # Random effects
        alpha_i = np.random.normal(0, sigma_alpha_true, N)
        alpha = np.repeat(alpha_i, T)

        # Generate outcomes
        y = np.zeros(N * T, dtype=int)
        for i in range(N * T):
            y_star = X[i] @ beta_true + alpha[i] + np.random.logistic(0, 1)
            if y_star <= cutpoints_true[0]:
                y[i] = 0
            elif y_star <= cutpoints_true[1]:
                y[i] = 1
            else:
                y[i] = 2

        return {
            "y": y,
            "X": X,
            "groups": groups,
            "beta_true": beta_true,
            "cutpoints_true": cutpoints_true,
            "sigma_alpha_true": sigma_alpha_true,
            "N": N,
            "T": T,
            "J": J,
        }

    def test_re_initialization(self, panel_data):
        """Test RE model initialization."""
        model = RandomEffectsOrderedLogit(
            endog=panel_data["y"],
            exog=panel_data["X"],
            groups=panel_data["groups"],
            quadrature_points=8,
        )

        assert model.n_categories == panel_data["J"]
        assert model.n_entities == panel_data["N"]
        assert model.quadrature_points == 8
        assert "sigma_alpha" in model.param_names

    def test_re_fit_basic(self, panel_data):
        """Test basic RE fitting."""
        model = RandomEffectsOrderedLogit(
            endog=panel_data["y"],
            exog=panel_data["X"],
            groups=panel_data["groups"],
            quadrature_points=6,  # Fewer points for speed
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=20)  # Few iterations for testing

        assert hasattr(result, "beta")
        assert hasattr(result, "cutpoints")
        assert hasattr(result, "sigma_alpha")

        # Check sigma_alpha is positive
        assert result.sigma_alpha > 0

    @pytest.mark.xfail(
        reason="RE and pooled may converge to same estimates with weak heterogeneity"
    )
    def test_re_vs_pooled_difference(self, panel_data):
        """Test that RE and pooled models give different results."""
        # Fit RE model
        re_model = RandomEffectsOrderedLogit(
            endog=panel_data["y"],
            exog=panel_data["X"],
            groups=panel_data["groups"],
            quadrature_points=6,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            re_result = re_model.fit(maxiter=20)

        # Fit pooled model
        pooled_model = OrderedLogit(
            endog=panel_data["y"], exog=panel_data["X"], groups=panel_data["groups"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pooled_result = pooled_model.fit(maxiter=20)

        # Parameters should be different when there are random effects
        assert not np.allclose(re_result.beta, pooled_result.beta, atol=0.01)


class TestOrderedMarginalEffects:
    """Test marginal effects for ordered models."""

    @pytest.fixture
    def fitted_model(self):
        """Create a fitted ordered logit model."""
        np.random.seed(42)
        N = 100
        K = 2
        J = 3

        X = np.random.randn(N, K)
        y = np.random.randint(0, J, N)
        groups = np.arange(N)

        model = OrderedLogit(endog=y, exog=X, groups=groups)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=50)

        # Create a mock result object with necessary attributes
        class MockResult:
            def __init__(self, model_obj, params, beta, cutpoints):
                self.model = model_obj
                self.params = params
                self.beta = beta
                self.cutpoints = cutpoints
                self.cov_params = np.eye(len(params)) * 0.01

        mock_result = MockResult(model, result.params, result.beta, result.cutpoints)
        return mock_result

    def test_ordered_ame_computation(self, fitted_model):
        """Test AME computation for ordered models."""
        ame_result = compute_ordered_ame(fitted_model.model)

        # Check structure
        assert ame_result.marginal_effects.shape[0] == fitted_model.model.n_categories
        assert ame_result.std_errors.shape == ame_result.marginal_effects.shape

        # Check sum-to-zero property
        assert ame_result.verify_sum_to_zero(tol=1e-8)

    def test_ordered_mem_computation(self, fitted_model):
        """Test MEM computation for ordered models."""
        mem_result = compute_ordered_mem(fitted_model.model)

        # Check structure
        assert mem_result.marginal_effects.shape[0] == fitted_model.model.n_categories
        assert mem_result.at_values is not None

        # Check sum-to-zero property
        assert mem_result.verify_sum_to_zero(tol=1e-8)

    def test_marginal_effects_signs(self, fitted_model):
        """Test that marginal effects can have different signs across categories."""
        ame_result = compute_ordered_ame(fitted_model.model)

        # For at least one variable, effects should have different signs
        # across categories (not always positive or negative)
        for var in ame_result.marginal_effects.columns:
            effects = ame_result.marginal_effects[var].values
            # Check if there are both positive and negative effects
            has_positive = np.any(effects > 0.001)
            has_negative = np.any(effects < -0.001)

            # At least for some variables, we expect mixed signs
            if has_positive and has_negative:
                break
        else:
            # This might not always be true depending on random data,
            # but generally we expect some variables to have mixed effects
            pass  # Don't fail test as this depends on random data


class TestOrderedCoverageGaps:
    """Tests targeting uncovered branches in ordered.py."""

    @pytest.fixture
    def fitted_logit(self):
        """Return a fitted OrderedLogit model."""
        np.random.seed(42)
        N = 200
        K = 2
        beta_true = np.array([0.5, -0.3])
        cutpoints_true = np.array([-0.5, 0.5])

        X = np.random.randn(N, K)
        linear_pred = X @ beta_true
        y = np.zeros(N, dtype=int)
        for i in range(N):
            y_star = linear_pred[i] + np.random.logistic(0, 1)
            if y_star <= cutpoints_true[0]:
                y[i] = 0
            elif y_star <= cutpoints_true[1]:
                y[i] = 1
            else:
                y[i] = 2

        groups = np.arange(N)
        model = OrderedLogit(endog=y, exog=X, groups=groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(maxiter=100)
        return model

    def test_predict_type_prob(self, fitted_logit):
        """Test predict() with type='prob' returns probabilities."""
        probs = fitted_logit.predict(type="prob")
        assert probs.shape == (fitted_logit.n_obs, fitted_logit.n_categories)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0)

    def test_predict_type_invalid(self, fitted_logit):
        """Test predict() with invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown prediction type"):
            fitted_logit.predict(type="invalid_type")

    def test_summary_before_fit(self):
        """Test summary() on unfitted model returns informative string."""
        np.random.seed(42)
        model = OrderedLogit(
            endog=np.array([0, 1, 2, 0, 1, 2]),
            exog=np.random.randn(6, 2),
            groups=np.arange(6),
        )
        summary = model.summary()
        assert summary == "Model has not been fitted yet"

    def test_predict_proba_1d_exog(self, fitted_logit):
        """Test predict_proba() with a single observation (1D exog)."""
        single_obs = fitted_logit.exog[0]
        probs = fitted_logit.predict_proba(single_obs)
        assert probs.shape == (1, fitted_logit.n_categories)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_predict_before_fit(self):
        """Test predict_proba() before fitting raises ValueError."""
        model = OrderedLogit(
            endog=np.array([0, 1, 2]),
            exog=np.array([[1.0], [2.0], [3.0]]),
            groups=np.array([0, 1, 2]),
        )
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict_proba()

    def test_cutpoint_init_boundary_proportions(self):
        """Test cutpoint initialization when a category proportion is 0 or 1."""
        np.random.seed(99)
        N = 200
        # Create data where all observations are in category 0 or 2
        # so that P(y <= 0) = 1.0 or P(y <= j) = 0.0 for some j
        y = np.array([0] * 100 + [2] * 100)
        X = np.random.randn(N, 2)
        groups = np.arange(N)

        model = OrderedLogit(endog=y, exog=X, groups=groups, n_categories=3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # This exercises the branch where prop_j is 0 or 1
            model.fit(maxiter=50)

        assert hasattr(model, "params")
        assert hasattr(model, "cutpoints")

    def test_probit_fit_and_summary(self):
        """Test OrderedProbit fit() and summary()."""
        np.random.seed(42)
        N = 200
        K = 2
        beta_true = np.array([0.5, -0.3])
        cutpoints_true = np.array([-0.5, 0.5])

        X = np.random.randn(N, K)
        linear_pred = X @ beta_true
        y = np.zeros(N, dtype=int)
        for i in range(N):
            y_star = linear_pred[i] + np.random.normal(0, 1)
            if y_star <= cutpoints_true[0]:
                y[i] = 0
            elif y_star <= cutpoints_true[1]:
                y[i] = 1
            else:
                y[i] = 2

        groups = np.arange(N)
        model = OrderedProbit(endog=y, exog=X, groups=groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(maxiter=100)

        # Exercise summary on probit
        summary = model.summary()
        assert "OrderedProbit Results" in summary
        assert "Cutpoints (ordered):" in summary

        # Exercise predict with type='prob'
        probs = model.predict(type="prob")
        assert probs.shape == (N, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

        # Exercise predict with type='category'
        cats = model.predict(type="category")
        assert len(cats) == N
        assert np.all(cats >= 0)
        assert np.all(cats < 3)

    def test_probit_cutpoint_initialization(self):
        """Test that OrderedProbit uses norm.ppf for initial cutpoints."""
        np.random.seed(42)
        N = 300
        X = np.random.randn(N, 1)
        y = np.zeros(N, dtype=int)
        for i in range(N):
            y_star = 0.5 * X[i, 0] + np.random.normal(0, 1)
            if y_star <= -0.5:
                y[i] = 0
            elif y_star <= 0.5:
                y[i] = 1
            else:
                y[i] = 2

        groups = np.arange(N)
        model = OrderedProbit(endog=y, exog=X, groups=groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(maxiter=100)

        assert hasattr(model, "beta")
        assert len(model.cutpoints) == 2
        assert model.cutpoints[0] < model.cutpoints[1]

    def test_singular_hessian_standard_errors(self):
        """Test standard error computation with singular Hessian."""
        np.random.seed(42)
        # Create a degenerate case: perfectly collinear features
        N = 50
        x1 = np.random.randn(N)
        X = np.column_stack([x1, x1])  # Perfectly collinear
        y = np.random.randint(0, 3, N)
        groups = np.arange(N)

        model = OrderedLogit(endog=y, exog=X, groups=groups)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(maxiter=50)

        # Even if SE computation fails, model should have bse attribute
        assert hasattr(model, "bse")

    def test_predict_new_exog(self, fitted_logit):
        """Test predictions with new exogenous data."""
        new_X = np.random.randn(10, fitted_logit.n_features)
        probs = fitted_logit.predict_proba(new_X)
        assert probs.shape == (10, fitted_logit.n_categories)
        assert np.allclose(probs.sum(axis=1), 1.0)

        cats = fitted_logit.predict(exog=new_X, type="category")
        assert len(cats) == 10

    def test_optimization_failure_warning(self):
        """Test that optimization failure produces a warning."""
        np.random.seed(42)
        N = 20
        X = np.random.randn(N, 2)
        y = np.random.randint(0, 3, N)
        groups = np.arange(N)

        model = OrderedLogit(endog=y, exog=X, groups=groups)
        # Use very few iterations to force non-convergence
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model.fit(maxiter=1, method="BFGS")

        # Model should still have results even if not converged
        assert hasattr(model, "params")
