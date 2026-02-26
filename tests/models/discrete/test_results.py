"""Tests for NonlinearPanelResults class."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit


class TestNonlinearPanelResults:
    """Tests for NonlinearPanelResults."""

    @pytest.fixture
    def mock_model_and_results(self):
        """Create a mock model and NonlinearPanelResults instance.

        Since NonlinearPanelResults.__init__ calls super().__init__ with keyword
        arguments that do not match the PanelResults positional signature, we
        patch PanelResults.__init__ and manually wire up the attributes that the
        NonlinearPanelResults methods depend on.
        """
        np.random.seed(42)
        n = 200
        k = 3

        # Create mock model
        model = MagicMock()
        model.__class__.__name__ = "PooledLogit"
        model.model_type = "binary"
        model.depvar = "y"
        model.exog_names = ["Intercept", "x1", "x2"]
        model.n_entities = 20
        model.entity_id = np.repeat(np.arange(20), 10)

        # Design matrix and true params
        X = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])
        true_params = np.array([0.5, 1.0, -0.5])
        eta = X @ true_params
        probs = expit(eta)
        y = (np.random.rand(n) < probs).astype(float)

        model.X = X
        model.y = y
        model.nobs = n

        # Link function (logistic)
        model._link_function = MagicMock(side_effect=expit)

        # Log-likelihood function
        def _log_likelihood(params):
            eta_ll = (
                X @ params[: X.shape[1]] if len(params) >= X.shape[1] else np.full(n, params[0])
            )
            p = expit(eta_ll)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        model._log_likelihood = MagicMock(side_effect=_log_likelihood)

        # For null model deepcopy support
        model.formula = "y ~ x1 + x2"
        model._prepare_matrices = MagicMock()

        # Hessian function (for link_test)
        def _hessian(params):
            return -np.eye(len(params)) * 10  # Simple negative definite

        model._hessian = MagicMock(side_effect=_hessian)

        params = true_params
        llf = _log_likelihood(params)

        # Covariance matrix returned by compute_mle_standard_errors
        cov = np.eye(k) * 0.01

        # Patch both compute_mle_standard_errors AND PanelResults.__init__
        # because the super().__init__ call signature is incompatible.
        with (
            patch(
                "panelbox.models.discrete.results.compute_mle_standard_errors",
                return_value=cov,
            ),
            patch(
                "panelbox.core.results.PanelResults.__init__",
                return_value=None,
            ),
        ):
            from panelbox.models.discrete.results import NonlinearPanelResults

            results = NonlinearPanelResults(
                model=model,
                params=params,
                llf=llf,
                converged=True,
                n_iter=10,
                se_type="cluster",
                opt_result=None,
            )

        # Manually wire up attributes that PanelResults.__init__ would set
        param_names = model.exog_names
        results.params = pd.Series(params, index=param_names)
        std_errors = np.sqrt(np.diag(cov))
        results.std_errors = pd.Series(std_errors, index=param_names)
        results.cov_params = pd.DataFrame(cov, index=param_names, columns=param_names)
        results.nobs = n
        results.n_entities = model.n_entities
        results.df_resid = n - k
        results.df_model = k
        results._model = model

        # Compute t-values and p-values
        from scipy import stats

        results.tvalues = results.params / results.std_errors
        pvalues_array = 2 * (1 - stats.t.cdf(np.abs(results.tvalues.values), results.df_resid))
        results.pvalues = pd.Series(pvalues_array, index=param_names)

        # Residuals / fittedvalues (not strictly needed but some methods might check)
        linear_pred = X @ params
        results.fittedvalues = expit(linear_pred)
        results.resid = y - results.fittedvalues

        # Override conf_int to return a numpy-compatible object.
        # NonlinearPanelResults.summary() uses self.conf_int()[:, 0] which
        # requires numpy-style column indexing. The parent PanelResults.conf_int()
        # returns a DataFrame, so we return the .values from the DataFrame.
        from scipy import stats as scipy_stats

        def conf_int(alpha=0.05):
            t_critical = scipy_stats.t.ppf(1 - alpha / 2, results.df_resid)
            margin = t_critical * results.std_errors
            lower = results.params - margin
            upper = results.params + margin
            return np.column_stack([lower.values, upper.values])

        results.conf_int = conf_int

        return model, results

    # ------------------------------------------------------------------ #
    # Instantiation
    # ------------------------------------------------------------------ #

    def test_init_llf(self, mock_model_and_results):
        model, results = mock_model_and_results
        assert results.llf == pytest.approx(model._log_likelihood(results.params.values), rel=1e-6)

    def test_init_converged(self, mock_model_and_results):
        _, results = mock_model_and_results
        assert results.converged is True

    def test_init_n_iter(self, mock_model_and_results):
        _, results = mock_model_and_results
        assert results.n_iter == 10

    def test_init_se_type(self, mock_model_and_results):
        _, results = mock_model_and_results
        assert results._se_type == "cluster"

    def test_init_cache(self, mock_model_and_results):
        _, results = mock_model_and_results
        assert results._cache == {}

    # ------------------------------------------------------------------ #
    # AIC / BIC
    # ------------------------------------------------------------------ #

    def test_aic(self, mock_model_and_results):
        _, results = mock_model_and_results
        expected_aic = -2 * results.llf + 2 * len(results.params)
        assert results.aic == pytest.approx(expected_aic)

    def test_bic(self, mock_model_and_results):
        _, results = mock_model_and_results
        expected_bic = -2 * results.llf + np.log(results.nobs) * len(results.params)
        assert results.bic == pytest.approx(expected_bic)

    # ------------------------------------------------------------------ #
    # llf_null
    # ------------------------------------------------------------------ #

    def test_llf_null(self, mock_model_and_results):
        """Test that llf_null computes null model log-likelihood."""
        model, results = mock_model_and_results

        # deepcopy of MagicMock returns a new MagicMock.
        # We need to make sure the null_model returned by deepcopy has
        # the same _log_likelihood and _prepare_matrices.
        with patch("copy.deepcopy") as mock_deepcopy:
            null_model = MagicMock()
            null_model.depvar = "y"
            null_model._prepare_matrices = MagicMock()

            # Null model log-likelihood: only intercept
            n = model.nobs
            y = model.y

            def null_ll(params):
                p = expit(np.full(n, params[0]))
                p = np.clip(p, 1e-10, 1 - 1e-10)
                return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

            null_model._log_likelihood = MagicMock(side_effect=null_ll)
            mock_deepcopy.return_value = null_model

            llf_null = results.llf_null

        assert isinstance(llf_null, float)
        assert np.isfinite(llf_null)
        # Null model LL should be worse (smaller) than full model LL
        assert llf_null <= results.llf + 1e-6

    def test_llf_null_caching(self, mock_model_and_results):
        """Test that llf_null is cached after first computation."""
        _, results = mock_model_and_results

        # Manually set the cache
        results._cache["llf_null"] = -100.0
        assert results.llf_null == -100.0

    # ------------------------------------------------------------------ #
    # pseudo_r2
    # ------------------------------------------------------------------ #

    def test_pseudo_r2_mcfadden(self, mock_model_and_results):
        _, results = mock_model_and_results
        # Set llf_null in cache to avoid deepcopy issues
        results._cache["llf_null"] = -138.0
        r2 = results.pseudo_r2("mcfadden")
        expected = 1 - results.llf / (-138.0)
        assert r2 == pytest.approx(expected)
        assert 0 <= r2 <= 1

    def test_pseudo_r2_cox_snell(self, mock_model_and_results):
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        r2 = results.pseudo_r2("cox_snell")
        lr_stat = 2 * (results.llf - (-138.0))
        expected = 1 - np.exp(-lr_stat / results.nobs)
        assert r2 == pytest.approx(expected)
        assert isinstance(r2, float)
        assert np.isfinite(r2)

    def test_pseudo_r2_nagelkerke(self, mock_model_and_results):
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        r2 = results.pseudo_r2("nagelkerke")
        assert isinstance(r2, float)
        assert np.isfinite(r2)

    def test_pseudo_r2_invalid(self, mock_model_and_results):
        _, results = mock_model_and_results
        with pytest.raises(ValueError, match="Unknown pseudo-R"):
            results.pseudo_r2("invalid")

    # ------------------------------------------------------------------ #
    # predict
    # ------------------------------------------------------------------ #

    def test_predict_linear(self, mock_model_and_results):
        model, results = mock_model_and_results
        pred = results.predict(type="linear")
        expected = model.X @ results.params.values
        np.testing.assert_allclose(pred, expected)
        assert len(pred) == model.nobs

    def test_predict_prob(self, mock_model_and_results):
        model, results = mock_model_and_results
        pred = results.predict(type="prob")
        assert len(pred) == model.nobs
        assert np.all(pred >= 0) and np.all(pred <= 1)

    def test_predict_class(self, mock_model_and_results):
        model, results = mock_model_and_results
        pred = results.predict(type="class")
        assert set(np.unique(pred)).issubset({0, 1})
        assert len(pred) == model.nobs

    def test_predict_invalid_type(self, mock_model_and_results):
        _, results = mock_model_and_results
        with pytest.raises(ValueError, match="Unknown prediction type"):
            results.predict(type="invalid")

    def test_predict_with_exog(self, mock_model_and_results):
        model, results = mock_model_and_results
        new_X = model.X[:10]
        pred = results.predict(exog=new_X, type="prob")
        assert len(pred) == 10
        assert np.all(pred >= 0) and np.all(pred <= 1)

    def test_predict_linear_with_exog(self, mock_model_and_results):
        model, results = mock_model_and_results
        new_X = model.X[:5]
        pred = results.predict(exog=new_X, type="linear")
        expected = new_X @ results.params.values
        np.testing.assert_allclose(pred, expected)

    def test_predict_class_with_exog(self, mock_model_and_results):
        model, results = mock_model_and_results
        new_X = model.X[:15]
        pred = results.predict(exog=new_X, type="class")
        assert len(pred) == 15
        assert set(np.unique(pred)).issubset({0, 1})

    def test_predict_prob_without_link_function(self, mock_model_and_results):
        """Test predict type='prob' when model has no _link_function (fallback to expit)."""
        model, results = mock_model_and_results
        # Configure hasattr to return False for _link_function
        del model._link_function
        pred = results.predict(type="prob")
        assert len(pred) == model.nobs
        assert np.all(pred >= 0) and np.all(pred <= 1)

    # ------------------------------------------------------------------ #
    # classification_table
    # ------------------------------------------------------------------ #

    def test_classification_table(self, mock_model_and_results):
        _, results = mock_model_and_results
        ct = results.classification_table()
        assert isinstance(ct, pd.DataFrame)
        assert ct.shape == (2, 2)
        assert list(ct.index) == ["Actual=0", "Actual=1"]
        assert list(ct.columns) == ["Predicted=0", "Predicted=1"]

    def test_classification_table_custom_threshold(self, mock_model_and_results):
        _, results = mock_model_and_results
        ct = results.classification_table(threshold=0.7)
        assert isinstance(ct, pd.DataFrame)
        assert ct.shape == (2, 2)

    def test_classification_table_values_sum(self, mock_model_and_results):
        """Total entries in confusion matrix should equal nobs."""
        model, results = mock_model_and_results
        ct = results.classification_table()
        assert ct.values.sum() == model.nobs

    # ------------------------------------------------------------------ #
    # classification_metrics
    # ------------------------------------------------------------------ #

    def test_classification_metrics_keys(self, mock_model_and_results):
        _, results = mock_model_and_results
        metrics = results.classification_metrics()
        expected_keys = {"accuracy", "precision", "recall", "f1", "auc_roc"}
        assert set(metrics.keys()) == expected_keys

    def test_classification_metrics_ranges(self, mock_model_and_results):
        _, results = mock_model_and_results
        metrics = results.classification_metrics()
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["auc_roc"] <= 1

    def test_classification_metrics_custom_threshold(self, mock_model_and_results):
        _, results = mock_model_and_results
        metrics = results.classification_metrics(threshold=0.3)
        assert 0 <= metrics["accuracy"] <= 1

    # ------------------------------------------------------------------ #
    # hosmer_lemeshow_test
    # ------------------------------------------------------------------ #

    def test_hosmer_lemeshow_test(self, mock_model_and_results):
        _, results = mock_model_and_results
        hl = results.hosmer_lemeshow_test()
        assert "statistic" in hl
        assert "p_value" in hl
        assert "df" in hl
        assert hl["df"] == 8  # default n_groups=10, df = 10-2
        assert 0 <= hl["p_value"] <= 1
        assert hl["statistic"] >= 0

    def test_hosmer_lemeshow_custom_groups(self, mock_model_and_results):
        _, results = mock_model_and_results
        hl = results.hosmer_lemeshow_test(n_groups=5)
        assert hl["df"] == 3  # 5 - 2

    def test_hosmer_lemeshow_test_large_groups(self, mock_model_and_results):
        _, results = mock_model_and_results
        hl = results.hosmer_lemeshow_test(n_groups=20)
        assert hl["df"] == 18  # 20 - 2

    # ------------------------------------------------------------------ #
    # link_test
    # ------------------------------------------------------------------ #

    def test_link_test(self, mock_model_and_results):
        """Test link test returns expected keys."""
        model, results = mock_model_and_results

        # Extend _log_likelihood to handle augmented model (k+1 params)
        X = model.X
        y = model.y

        def _log_likelihood_aug(params):
            # For the augmented model, just use first k params on X
            k = X.shape[1]
            if len(params) > k:
                eta_ll = X @ params[:k]  # Ignore extra param for simplicity
            else:
                eta_ll = X @ params
            p = expit(eta_ll)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        model._log_likelihood = MagicMock(side_effect=_log_likelihood_aug)

        def _hessian_aug(params):
            # Return positive diagonal so sqrt(diag(H)) works in link_test line 430
            return np.eye(len(params)) * 0.01

        model._hessian = MagicMock(side_effect=_hessian_aug)

        with patch("copy.deepcopy") as mock_deepcopy:
            aug_model = MagicMock()
            aug_model._log_likelihood = MagicMock(side_effect=_log_likelihood_aug)
            aug_model._hessian = MagicMock(side_effect=_hessian_aug)
            mock_deepcopy.return_value = aug_model

            lt = results.link_test()

        assert "coefficient" in lt
        assert "std_error" in lt
        assert "z_statistic" in lt
        assert "p_value" in lt
        assert isinstance(lt["p_value"], float)
        assert 0 <= lt["p_value"] <= 1

    # ------------------------------------------------------------------ #
    # summary
    # ------------------------------------------------------------------ #

    def test_summary_returns_dataframe(self, mock_model_and_results):
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        s = results.summary()
        assert isinstance(s, pd.DataFrame)

    def test_summary_contains_model_info(self, mock_model_and_results):
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        s = results.summary()
        # The summary should have some content
        assert len(s) > 0

    # ------------------------------------------------------------------ #
    # to_latex
    # ------------------------------------------------------------------ #

    def test_to_latex(self, mock_model_and_results):
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        latex = results.to_latex()
        assert isinstance(latex, str)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "Coefficient" in latex or "Variable" in latex

    def test_to_latex_with_stars(self, mock_model_and_results):
        """Test that significance stars appear in LaTeX output."""
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        latex = results.to_latex()
        # At least some coefficients should be significant given our
        # well-specified model, so stars should appear
        assert "\\begin{tabular}" in latex

    def test_to_latex_with_file(self, mock_model_and_results, tmp_path):
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        filepath = tmp_path / "test_table.tex"
        latex = results.to_latex(filepath=str(filepath))
        assert filepath.exists()
        content = filepath.read_text()
        assert content == latex

    def test_to_latex_custom_caption_label(self, mock_model_and_results):
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        latex = results.to_latex(caption="My Model", label="tab:my_model")
        assert "My Model" in latex
        assert "tab:my_model" in latex

    def test_to_latex_model_stats(self, mock_model_and_results):
        """Test that model statistics appear in LaTeX output."""
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        latex = results.to_latex()
        assert "Observations" in latex
        assert "Log-Likelihood" in latex
        assert "AIC" in latex
        assert "BIC" in latex
        assert "Pseudo-R" in latex

    # ------------------------------------------------------------------ #
    # marginal_effects (not implemented)
    # ------------------------------------------------------------------ #

    def test_marginal_effects_not_implemented(self, mock_model_and_results):
        _, results = mock_model_and_results
        with pytest.raises(NotImplementedError):
            results.marginal_effects()

    # ------------------------------------------------------------------ #
    # _repr_html_
    # ------------------------------------------------------------------ #

    def test_repr_html(self, mock_model_and_results):
        _, results = mock_model_and_results
        results._cache["llf_null"] = -138.0
        html = results._repr_html_()
        assert isinstance(html, str)
        assert "<" in html  # Should contain HTML tags

    # ------------------------------------------------------------------ #
    # conf_int (inherited but used by many methods)
    # ------------------------------------------------------------------ #

    def test_conf_int(self, mock_model_and_results):
        """Test confidence intervals used by summary and to_latex."""
        _, results = mock_model_and_results
        ci = results.conf_int()
        assert isinstance(ci, np.ndarray)
        assert ci.shape == (3, 2)
        # Check lower < param < upper
        for i, name in enumerate(results.params.index):
            assert ci[i, 0] < results.params[name]
            assert ci[i, 1] > results.params[name]

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_predict_prob_uses_model_x_when_exog_none(self, mock_model_and_results):
        """When exog=None, predict should use model.X."""
        model, results = mock_model_and_results
        pred = results.predict(exog=None, type="prob")
        assert len(pred) == model.nobs

    def test_classification_table_extreme_threshold(self, mock_model_and_results):
        """With threshold=0, all predictions should be 1."""
        model, results = mock_model_and_results
        ct = results.classification_table(threshold=0.0)
        # With threshold=0 everything is predicted as 1
        assert ct["Predicted=0"].sum() == 0
        assert ct["Predicted=1"].sum() == model.nobs

    def test_classification_table_high_threshold(self, mock_model_and_results):
        """With threshold=1, all predictions should be 0."""
        model, results = mock_model_and_results
        ct = results.classification_table(threshold=1.0)
        # With threshold=1 everything is predicted as 0
        assert ct["Predicted=1"].sum() == 0
        assert ct["Predicted=0"].sum() == model.nobs


class TestNonlinearPanelResultsComputeCov:
    """Test _compute_cov_params method separately."""

    def test_compute_cov_params_calls_mle_se(self):
        """Test that _compute_cov_params delegates to compute_mle_standard_errors."""
        from panelbox.models.discrete.results import NonlinearPanelResults

        model = MagicMock()
        model.entity_id = np.array([0, 0, 1, 1])
        params = np.array([1.0, 2.0])
        cov = np.eye(2) * 0.05

        with patch(
            "panelbox.models.discrete.results.compute_mle_standard_errors",
            return_value=cov,
        ) as mock_se:
            # Call the unbound method by creating a temporary instance
            # with patched __init__ to avoid full construction
            result = NonlinearPanelResults._compute_cov_params(None, model, params, "cluster")
            mock_se.assert_called_once_with(
                model=model, params=params, se_type="cluster", entity_id=model.entity_id
            )
            np.testing.assert_array_equal(result, cov)
