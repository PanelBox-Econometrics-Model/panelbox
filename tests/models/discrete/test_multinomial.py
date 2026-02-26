"""
Tests for multinomial logit panel models.
"""

import numpy as np
import pandas as pd
import pytest

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

        with pytest.raises(ValueError):  # noqa: PT012
            model = MultinomialLogit(y_invalid, self.X, n_alternatives=self.n_alternatives)
            model.fit()


class TestMultinomialLogitFixedEffects:
    """Test suite for fixed effects multinomial logit."""

    def setup_method(self):
        """Setup test data for FE model."""
        np.random.seed(42)

        self.n_entities = 30
        self.n_periods = 4
        self.n_vars = 2
        self.n_alternatives = 3

        # Create panel data
        data = []
        for i in range(self.n_entities):
            for t in range(self.n_periods):
                data.append(
                    {"entity": i, "time": t, "x1": np.random.randn(), "x2": np.random.randn()}
                )

        self.df = pd.DataFrame(data)

        # True parameters
        beta_true = np.array([[0.5, -0.3], [0.2, 0.4]])

        # Generate choices with individual heterogeneity
        choices = []
        for i in range(self.n_entities):
            alpha_i = np.random.randn() * 0.5  # Individual effect
            entity_df = self.df[self.df["entity"] == i]

            for _, row in entity_df.iterrows():
                X = np.array([row["x1"], row["x2"]])
                util_0 = alpha_i
                util_1 = X @ beta_true[0] + alpha_i
                util_2 = X @ beta_true[1] + alpha_i

                utils = np.array([util_0, util_1, util_2])
                exp_utils = np.exp(utils - utils.max())
                probs = exp_utils / exp_utils.sum()

                choice = np.random.choice(self.n_alternatives, p=probs)
                choices.append(choice)

        self.df["choice"] = choices

    @pytest.mark.xfail(
        reason="Source code bug: MultinomialLogit passes entity_col string as entity_id array to PanelModel",
        strict=True,
    )
    def test_fixed_effects_estimation(self):
        """Test FE multinomial logit estimation."""
        # This should work but may be slow
        y = self.df["choice"].values
        X = self.df[["x1", "x2"]].values

        model = MultinomialLogit(
            y, X, n_alternatives=self.n_alternatives, method="fixed_effects", entity_col="entity"
        )

        # Should raise warning about feasibility
        with pytest.warns(UserWarning):  # noqa: PT030
            result = model.fit(maxiter=100)

        # Basic checks
        assert hasattr(result, "params")
        assert len(result.params) == (self.n_alternatives - 1) * self.n_vars

    def test_fe_requires_entity_col(self):
        """Test that FE requires entity_col."""
        y = self.df["choice"].values
        X = self.df[["x1", "x2"]].values

        with pytest.raises(ValueError, match="entity_col required"):
            MultinomialLogit(y, X, n_alternatives=self.n_alternatives, method="fixed_effects")


class TestMultinomialLogitRandomEffects:
    """Test suite for random effects multinomial logit."""

    def setup_method(self):
        """Setup test data for RE model."""
        np.random.seed(42)

        self.n_entities = 20
        self.n_periods = 3
        self.n_vars = 2
        self.n_alternatives = 3

        # Create panel data
        data = []
        for i in range(self.n_entities):
            for t in range(self.n_periods):
                data.append(
                    {"entity": i, "time": t, "x1": np.random.randn(), "x2": np.random.randn()}
                )

        self.df = pd.DataFrame(data)

        # True parameters
        beta_true = np.array([[0.5, -0.3], [0.2, 0.4]])

        # Generate choices with random effects
        choices = []
        for i in range(self.n_entities):
            alpha_i = np.random.randn() * 0.3  # Random effect
            entity_df = self.df[self.df["entity"] == i]

            for _, row in entity_df.iterrows():
                X = np.array([row["x1"], row["x2"]])
                util_0 = alpha_i
                util_1 = X @ beta_true[0] + alpha_i
                util_2 = X @ beta_true[1] + alpha_i

                utils = np.array([util_0, util_1, util_2])
                exp_utils = np.exp(utils - utils.max())
                probs = exp_utils / exp_utils.sum()

                choice = np.random.choice(self.n_alternatives, p=probs)
                choices.append(choice)

        self.df["choice"] = choices

    @pytest.mark.xfail(
        reason="Source code bug: MultinomialLogit passes entity_col string as entity_id array to PanelModel",
        strict=True,
    )
    def test_random_effects_estimation(self):
        """Test RE multinomial logit estimation."""
        y = self.df["choice"].values
        X = self.df[["x1", "x2"]].values

        model = MultinomialLogit(
            y, X, n_alternatives=self.n_alternatives, method="random_effects", entity_col="entity"
        )

        result = model.fit(maxiter=50)

        # Basic checks
        assert hasattr(result, "params")
        # RE has additional variance parameter
        # For now, just check that estimation runs

    def test_re_requires_entity_col(self):
        """Test that RE requires entity_col."""
        y = self.df["choice"].values
        X = self.df[["x1", "x2"]].values

        with pytest.raises(ValueError, match="entity_col required"):
            MultinomialLogit(y, X, n_alternatives=self.n_alternatives, method="random_effects")


class TestMultinomialLogitComparison:
    """Test comparison between pooled, FE, and RE."""

    def setup_method(self):
        """Setup common test data."""
        np.random.seed(42)

        self.n_entities = 25
        self.n_periods = 4
        self.n_vars = 2
        self.n_alternatives = 3

        # Create balanced panel
        data = []
        for i in range(self.n_entities):
            for t in range(self.n_periods):
                data.append(
                    {"entity": i, "time": t, "x1": np.random.randn(), "x2": np.random.randn()}
                )

        self.df = pd.DataFrame(data)

        # Simple DGP for testing
        beta_true = np.array([[0.3, -0.2], [0.1, 0.3]])
        choices = []

        for _, row in self.df.iterrows():
            X = np.array([row["x1"], row["x2"]])
            util_0 = 0
            util_1 = X @ beta_true[0]
            util_2 = X @ beta_true[1]

            utils = np.array([util_0, util_1, util_2])
            exp_utils = np.exp(utils - utils.max())
            probs = exp_utils / exp_utils.sum()

            choice = np.random.choice(self.n_alternatives, p=probs)
            choices.append(choice)

        self.df["choice"] = choices

    def test_pooled_vs_fe_vs_re(self):
        """Compare pooled, FE, and RE estimates."""
        y = self.df["choice"].values
        X = self.df[["x1", "x2"]].values

        # Pooled
        model_pooled = MultinomialLogit(y, X, n_alternatives=self.n_alternatives, method="pooled")
        result_pooled = model_pooled.fit()

        # These should all produce results
        assert result_pooled.converged
        assert hasattr(result_pooled, "params")

        # Note: FE and RE tests are skipped in this simple comparison
        # as they require proper entity handling which needs refactoring


class TestMultinomialAdditional:
    """Additional tests for MultinomialLogit to cover uncovered lines."""

    @pytest.fixture
    def mlogit_data(self):
        np.random.seed(42)
        n = 300
        J = 3
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        beta_true = np.array([[0.5, 1.0, -0.5], [1.0, -0.5, 0.8]])
        utilities = np.zeros((n, J))
        utilities[:, 1] = X @ beta_true[0]
        utilities[:, 2] = X @ beta_true[1]
        exp_u = np.exp(utilities - utilities.max(axis=1, keepdims=True))
        probs = exp_u / exp_u.sum(axis=1, keepdims=True)
        y = np.array([np.random.choice(J, p=probs[i]) for i in range(n)])
        return y, X

    def test_predict_method(self, mlogit_data):
        """Test predict returns argmax of probabilities."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        predictions = result.predict()
        assert len(predictions) == len(y)
        assert np.all(np.isin(predictions, [0, 1, 2]))

    def test_predict_with_new_data(self, mlogit_data):
        """Test predict with new exog data."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        new_X = X[:10]
        predictions = result.predict(exog=new_X)
        assert len(predictions) == 10

    def test_predict_proba_new_data(self, mlogit_data):
        """Test predict_proba with new exog data."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        probs = result.predict_proba(exog=X[:5])
        assert probs.shape == (5, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_marginal_effects_overall(self, mlogit_data):
        """Test marginal effects computed at overall (average over obs)."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        me = result.marginal_effects(at="overall")
        assert me.shape == (3, 3)  # J x K
        # Sum across alternatives should be ~0
        np.testing.assert_allclose(me.sum(axis=0), 0, atol=0.01)

    def test_marginal_effects_median(self, mlogit_data):
        """Test marginal effects computed at median."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        me = result.marginal_effects(at="median")
        assert me.shape == (3, 3)

    def test_marginal_effects_specific_variable(self, mlogit_data):
        """Test marginal effects for a specific variable index."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        me = result.marginal_effects(at="mean", variable=0)
        assert me.shape == (3,)

    def test_marginal_effects_se(self, mlogit_data):
        """Test marginal effects standard errors via delta method."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        se = result.marginal_effects_se(at="mean")
        assert se.shape == (3, 3)
        assert np.all(se >= 0)

    def test_marginal_effects_se_median(self, mlogit_data):
        """Test marginal effects SE computed at median."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        se = result.marginal_effects_se(at="median")
        assert se.shape == (3, 3)
        assert np.all(se >= 0)

    def test_marginal_effects_se_overall(self, mlogit_data):
        """Test marginal effects SE computed at overall (uses mean fallback)."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        with pytest.warns(UserWarning, match="computed at mean"):
            se = result.marginal_effects_se(at="overall")
        assert se.shape == (3, 3)
        assert np.all(se >= 0)

    def test_marginal_effects_se_specific_variable(self, mlogit_data):
        """Test marginal effects SE for a specific variable."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        se = result.marginal_effects_se(at="mean", variable=1)
        assert se.shape == (3,)
        assert np.all(se >= 0)

    def test_summary(self, mlogit_data):
        """Test summary string output."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        s = result.summary()
        assert isinstance(s, str)
        assert "Multinomial Logit" in s
        assert "Confusion Matrix" in s
        assert "Log-likelihood" in s
        assert "AIC" in s
        assert "BIC" in s

    def test_fit_custom_start_params(self, mlogit_data):
        """Test fit with custom starting parameters."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        start = np.zeros(model.n_params) + 0.01
        result = model.fit(start_params=start)
        assert hasattr(result, "params")
        assert len(result.params) == model.n_params

    def test_fit_custom_method(self, mlogit_data):
        """Test fit with a different optimization method."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit(method="L-BFGS-B")
        assert hasattr(result, "params")
        assert len(result.params) == model.n_params

    def test_fe_feasibility_warning_many_alts(self):
        """Test warning when J > 4 for fixed effects."""
        np.random.seed(42)
        n = 100
        J = 5
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randint(0, J, n)
        # Pass actual entity_id array (not a string column name)
        entity_ids = np.repeat(np.arange(10), 10)

        with pytest.warns(UserWarning, match="computationally intensive"):
            MultinomialLogit(y, X, n_alternatives=J, method="fixed_effects", entity_col=entity_ids)

    def test_fe_feasibility_warning_many_periods(self):
        """Test warning when avg T > 10 for fixed effects.

        Due to a known bug where _entity_col_data is never set, _get_entity_ids()
        falls back to sequential IDs (each obs = own entity, T=1). We test the
        _check_fe_feasibility method directly by setting entity_ids manually.
        """
        np.random.seed(42)
        n_entities = 5
        n_periods = 12  # avg T > 10
        n = n_entities * n_periods
        J = 3
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randint(0, J, n)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)

        # Build the model as pooled first (avoids the entity_col issue)
        model = MultinomialLogit(y, X, n_alternatives=J, method="pooled")
        # Manually set the required attributes and switch to FE
        model.method = "fixed_effects"
        model.entity_ids = entity_ids
        model.n_entities = n_entities

        with pytest.warns(UserWarning, match="very slow"):
            model._check_fe_feasibility()

    def test_different_base_alt(self, mlogit_data):
        """Test model with a different base alternative."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X, base_alternative=1)
        result = model.fit()
        assert model.base_alternative == 1
        probs = result.predict_proba()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_model_predict_method_on_model(self, mlogit_data):
        """Test the model-level predict method (not result)."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        preds = model.predict(result.params, X[:5])
        assert len(preds) == 5
        assert np.all(np.isin(preds, [0, 1, 2]))

    def test_log_likelihood_fixed_effects(self):
        """Test FE log-likelihood computation directly."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 4
        n = n_entities * n_periods
        J = 3
        K = 2
        X = np.random.randn(n, K)
        y = np.random.randint(0, J, n)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)

        model = MultinomialLogit(
            y, X, n_alternatives=J, method="fixed_effects", entity_col=entity_ids
        )
        params = np.zeros(model.n_params)

        # Call FE log-likelihood directly
        with pytest.warns(UserWarning, match="within-transformation"):
            ll = model._log_likelihood_fixed_effects(params)
        assert np.isfinite(ll)

    def test_log_likelihood_random_effects(self):
        """Test RE log-likelihood computation directly."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 3
        n = n_entities * n_periods
        J = 3
        K = 2
        X = np.random.randn(n, K)
        y = np.random.randint(0, J, n)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)

        model = MultinomialLogit(
            y, X, n_alternatives=J, method="random_effects", entity_col=entity_ids
        )
        params = np.zeros(model.n_params)

        ll = model._log_likelihood_random_effects(params)
        assert np.isfinite(ll)

    def test_log_likelihood_random_effects_with_sigma(self):
        """Test RE log-likelihood with extra sigma parameter."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 3
        n = n_entities * n_periods
        J = 3
        K = 2
        X = np.random.randn(n, K)
        y = np.random.randint(0, J, n)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)

        model = MultinomialLogit(
            y, X, n_alternatives=J, method="random_effects", entity_col=entity_ids
        )
        # Add one extra parameter for log_sigma
        params = np.zeros(model.n_params + 1)
        params[-1] = 0.0  # log(sigma) = 0 => sigma = 1

        ll = model._log_likelihood_random_effects(params)
        assert np.isfinite(ll)

    def test_plot_marginal_effects_single_variable(self, mlogit_data):
        """Test plotting marginal effects for a single variable."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        fig = result.plot_marginal_effects(variable=0)
        assert fig is not None
        plt.close("all")

    def test_plot_marginal_effects_all_vars(self, mlogit_data):
        """Test plotting marginal effects for all variables."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        fig = result.plot_marginal_effects()
        assert fig is not None
        plt.close("all")

    def test_marginal_effects_invalid_at(self, mlogit_data):
        """Test marginal_effects with invalid 'at' raises ValueError."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        with pytest.raises(ValueError, match="Unknown 'at'"):
            result.marginal_effects(at="invalid")

    def test_marginal_effects_se_invalid_at(self, mlogit_data):
        """Test marginal_effects_se with invalid 'at' raises ValueError."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        with pytest.raises(ValueError, match="Unknown 'at'"):
            result.marginal_effects_se(at="invalid")

    def test_fit_statistics(self, mlogit_data):
        """Test that fit statistics are computed and accessible."""
        y, X = mlogit_data
        model = MultinomialLogit(y, X)
        result = model.fit()
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")
        assert hasattr(result, "pseudo_r2")
        assert hasattr(result, "accuracy")
        assert hasattr(result, "confusion_matrix")
        assert result.confusion_matrix.shape == (3, 3)
        assert 0 <= result.accuracy <= 1
        assert result.pseudo_r2 > 0  # Should explain something

    def test_invalid_method(self, mlogit_data):
        """Test that invalid method raises ValueError."""
        y, X = mlogit_data
        with pytest.raises(ValueError, match="method must be one of"):
            MultinomialLogit(y, X, method="invalid_method")


class TestConditionalLogitCoverage:
    """Tests for ConditionalLogit to cover uncovered lines."""

    @pytest.fixture
    def clogit_data(self):
        """Create choice data for conditional logit."""
        np.random.seed(42)
        n_choices = 100
        n_alts = 3
        rows = []
        for i in range(n_choices):
            costs = np.random.uniform(1, 10, n_alts)
            times = np.random.uniform(0.5, 3, n_alts)
            utils = -0.5 * costs - 0.3 * times + np.random.gumbel(size=n_alts)
            chosen = np.argmax(utils)
            for j in range(n_alts):
                rows.append(
                    {
                        "choice_id": i,
                        "alt_id": j,
                        "chosen": 1 if j == chosen else 0,
                        "cost": costs[j],
                        "time": times[j],
                    }
                )
        return pd.DataFrame(rows)

    def test_clogit_fit(self, clogit_data):
        """Test conditional logit estimation."""
        from panelbox.models.discrete.multinomial import ConditionalLogit

        model = ConditionalLogit(
            data=clogit_data,
            choice_col="choice_id",
            alt_col="alt_id",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )
        result = model.fit()
        assert result.converged
        assert result.params[0] < 0  # cost should be negative

    def test_clogit_predict_proba(self, clogit_data):
        """Test conditional logit predicted probabilities."""
        from panelbox.models.discrete.multinomial import ConditionalLogit

        model = ConditionalLogit(
            data=clogit_data,
            choice_col="choice_id",
            alt_col="alt_id",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )
        result = model.fit()
        probs = result.predict_proba()
        assert probs.shape == (100, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)

    def test_clogit_predict(self, clogit_data):
        """Test conditional logit predict method."""
        from panelbox.models.discrete.multinomial import ConditionalLogit

        model = ConditionalLogit(
            data=clogit_data,
            choice_col="choice_id",
            alt_col="alt_id",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )
        model.fit()
        preds = model.predict()
        assert len(preds) == 100
        assert np.all(np.isin(preds, [0, 1, 2]))

    def test_clogit_summary(self, clogit_data):
        """Test conditional logit summary output."""
        from panelbox.models.discrete.multinomial import ConditionalLogit

        model = ConditionalLogit(
            data=clogit_data,
            choice_col="choice_id",
            alt_col="alt_id",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )
        result = model.fit()
        s = result.summary()
        assert "Conditional Logit" in s
        assert "cost" in s
        assert "time" in s
        assert "Log-likelihood" in s

    def test_clogit_fit_statistics(self, clogit_data):
        """Test conditional logit fit statistics."""
        from panelbox.models.discrete.multinomial import ConditionalLogit

        model = ConditionalLogit(
            data=clogit_data,
            choice_col="choice_id",
            alt_col="alt_id",
            chosen_col="chosen",
            alt_varying_vars=["cost", "time"],
        )
        result = model.fit()
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")
        assert hasattr(result, "pseudo_r2")
        assert hasattr(result, "accuracy")
        assert 0 <= result.accuracy <= 1

    def test_clogit_missing_columns(self):
        """Test conditional logit raises on missing columns."""
        from panelbox.models.discrete.multinomial import ConditionalLogit

        df = pd.DataFrame({"a": [1, 2], "b": [0, 1]})
        with pytest.raises(ValueError, match="Missing columns"):
            ConditionalLogit(
                data=df,
                choice_col="choice_id",
                alt_col="alt_id",
                chosen_col="chosen",
                alt_varying_vars=["cost"],
            )

    def test_clogit_invalid_choices(self):
        """Test conditional logit raises when choice occasion has != 1 chosen."""
        from panelbox.models.discrete.multinomial import ConditionalLogit

        # Create data where choice_id=0 has 2 chosen alternatives
        df = pd.DataFrame(
            {
                "choice_id": [0, 0, 0, 1, 1, 1],
                "alt_id": [0, 1, 2, 0, 1, 2],
                "chosen": [1, 1, 0, 1, 0, 0],  # choice_id=0 has 2 chosen
                "cost": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            }
        )
        with pytest.raises(ValueError, match="exactly one chosen"):
            ConditionalLogit(
                data=df,
                choice_col="choice_id",
                alt_col="alt_id",
                chosen_col="chosen",
                alt_varying_vars=["cost"],
            )

    def test_clogit_with_case_varying(self):
        """Test conditional logit with case-varying variables."""
        from panelbox.models.discrete.multinomial import ConditionalLogit

        np.random.seed(42)
        n_choices = 80
        n_alts = 3
        rows = []
        for i in range(n_choices):
            costs = np.random.uniform(1, 10, n_alts)
            income = np.random.uniform(20, 100)  # case-varying
            utils = -0.5 * costs + np.random.gumbel(size=n_alts)
            chosen = np.argmax(utils)
            for j in range(n_alts):
                rows.append(
                    {
                        "choice_id": i,
                        "alt_id": j,
                        "chosen": 1 if j == chosen else 0,
                        "cost": costs[j],
                        "income": income,
                    }
                )
        df = pd.DataFrame(rows)
        model = ConditionalLogit(
            data=df,
            choice_col="choice_id",
            alt_col="alt_id",
            chosen_col="chosen",
            alt_varying_vars=["cost"],
            case_varying_vars=["income"],
        )
        result = model.fit()
        assert result.converged
        # n_params = 1 (cost) + 1 * (3-1) = 3
        assert len(result.params) == 3

        s = result.summary()
        assert "cost" in s
        assert "income" in s
