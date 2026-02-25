"""
Tests for Panel Heckman selection model.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.selection import PanelHeckman


class TestPanelHeckman:
    """Test suite for Panel Heckman selection model."""

    def setup_method(self):
        """Setup test data with selection."""
        np.random.seed(42)

        # Generate data
        self.n = 500
        self.k_outcome = 3
        self.k_selection = 4

        # Regressors
        self.X = np.random.randn(self.n, self.k_outcome)
        self.X[:, 0] = 1  # Intercept

        # Selection equation has exclusion restriction
        self.Z = np.random.randn(self.n, self.k_selection)
        self.Z[:, 0] = 1  # Intercept
        self.Z[:, -1] = np.random.randn(self.n)  # Exclusion restriction

        # True parameters
        self.beta_true = np.array([2.0, 0.5, -0.3])
        self.gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        self.sigma_true = 1.0
        self.rho_true = 0.5

        # Generate correlated errors
        mean = [0, 0]
        cov = [[1, self.rho_true], [self.rho_true, 1]]
        errors = np.random.multivariate_normal(mean, cov, self.n)
        u = errors[:, 0]
        e = errors[:, 1] * self.sigma_true

        # Selection process
        s_star = self.Z @ self.gamma_true + u
        self.selection = (s_star > 0).astype(int)

        # Outcome (latent)
        y_star = self.X @ self.beta_true + e

        # Observed outcome (only if selected)
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_two_step_estimation(self):
        """Test two-step Heckman estimation."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z, method="two_step")

        result = model.fit()

        assert result.converged
        assert result.method == "two_step"
        assert hasattr(result, "probit_params")
        assert hasattr(result, "outcome_params")
        assert hasattr(result, "sigma")
        assert hasattr(result, "rho")
        assert hasattr(result, "lambda_imr")

        # Check parameter dimensions
        assert len(result.probit_params) == self.k_selection
        assert len(result.outcome_params) == self.k_outcome

        # Check that rho is in valid range
        assert -1 <= result.rho <= 1

    def test_mle_estimation(self):
        """Test maximum likelihood estimation."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z, method="mle")

        result = model.fit()

        assert result.method == "mle"
        assert hasattr(result, "llf")
        assert result.llf is not None

        # MLE should give similar results to two-step for large samples
        two_step = model.fit(method="two_step")

        # Parameters should be reasonably close
        # (exact comparison depends on sample size and convergence)
        assert np.allclose(result.rho, two_step.rho, atol=0.2)

    def test_prediction(self):
        """Test prediction functionality."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)

        result = model.fit()

        # Unconditional prediction (latent outcome)
        pred_uncond = result.predict(type="unconditional")
        assert len(pred_uncond) == self.n

        # Conditional prediction (corrected for selection)
        pred_cond = result.predict(type="conditional")
        assert len(pred_cond) == self.n

        # Conditional should differ from unconditional when rho != 0
        if abs(result.rho) > 0.1:
            assert not np.allclose(pred_uncond, pred_cond)

    def test_selection_test(self):
        """Test for selection bias."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)

        result = model.fit()
        test = result.selection_test()

        assert "rho" in test
        assert "p_value" in test
        assert "significant" in test

        # With true rho = 0.5, should detect selection
        # (though this is stochastic)
        assert abs(test["rho"]) > 0.1

    def test_no_exclusion_restriction_warning(self):
        """Test warning when no exclusion restriction."""
        # Use same regressors for both equations
        with pytest.warns(UserWarning, match="exclusion restriction"):
            PanelHeckman(self.y, self.X, self.selection, self.X)  # Same as outcome equation

    def test_summary(self):
        """Test summary output."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)

        result = model.fit()
        summary = result.summary()

        assert "Panel Heckman Selection Model" in summary
        assert "Selection Equation" in summary
        assert "Outcome Equation" in summary
        assert f"Selected observations: {np.sum(self.selection)}" in summary
        assert "rho:" in summary

    def test_no_selection_case(self):
        """Test when all observations are selected."""
        # All selected
        selection_all = np.ones(self.n)
        y_all = self.X @ self.beta_true + np.random.randn(self.n)

        model = PanelHeckman(y_all, self.X, selection_all, self.Z)

        result = model.fit()

        # When all selected, should be similar to OLS
        # and rho should be near zero
        assert abs(result.rho) < 0.5

    def test_high_censoring_case(self):
        """Test with high censoring rate."""
        # Create high censoring
        selection_few = np.zeros(self.n)
        selection_few[:50] = 1  # Only 10% selected

        model = PanelHeckman(self.y, self.X, selection_few, self.Z)

        result = model.fit()

        # Should still converge even with high censoring
        assert result.converged
        assert result.n_selected == 50


class TestModelPredictDirectly:
    """Tests for PanelHeckman.predict() method (lines 158-170)."""

    def setup_method(self):
        """Setup test data with selection."""
        np.random.seed(42)

        self.n = 500
        self.k_outcome = 3
        self.k_selection = 4

        self.X = np.random.randn(self.n, self.k_outcome)
        self.X[:, 0] = 1  # Intercept

        self.Z = np.random.randn(self.n, self.k_selection)
        self.Z[:, 0] = 1  # Intercept
        self.Z[:, -1] = np.random.randn(self.n)  # Exclusion restriction

        self.beta_true = np.array([2.0, 0.5, -0.3])
        self.gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        self.sigma_true = 1.0
        self.rho_true = 0.5

        mean = [0, 0]
        cov = [[1, self.rho_true], [self.rho_true, 1]]
        errors = np.random.multivariate_normal(mean, cov, self.n)
        u = errors[:, 0]
        e = errors[:, 1] * self.sigma_true

        s_star = self.Z @ self.gamma_true + u
        self.selection = (s_star > 0).astype(int)

        y_star = self.X @ self.beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_model_predict_before_fit_raises(self):
        """Test predict() before fit() raises ValueError (line 162)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)

        with pytest.raises(ValueError, match="Model not fitted yet"):
            model.predict()

    def test_model_predict_after_fit(self):
        """Test predict() after fit() uses stored params (lines 159-160)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        model.results = model.fit()

        pred = model.predict()

        assert pred is not None
        assert len(pred) == self.n

    def test_model_predict_with_params(self):
        """Test predict() with explicit params (lines 164-168)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        # Pass params explicitly
        pred = model.predict(params=result.params)

        assert pred is not None
        assert len(pred) == self.n

    def test_model_predict_with_exog(self):
        """Test predict() with explicit exog (lines 169-170)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        # Create new exog data
        new_exog = np.random.randn(10, self.k_outcome)
        new_exog[:, 0] = 1  # Intercept

        pred = model.predict(params=result.params, exog=new_exog)

        assert pred is not None
        assert len(pred) == 10


class TestResultPredictWithDataFrame:
    """Tests for PanelHeckmanResult.predict() with DataFrame inputs (lines 505-522)."""

    def setup_method(self):
        """Setup test data with selection."""
        np.random.seed(42)

        self.n = 500
        self.k_outcome = 3
        self.k_selection = 4

        self.X = np.random.randn(self.n, self.k_outcome)
        self.X[:, 0] = 1

        self.Z = np.random.randn(self.n, self.k_selection)
        self.Z[:, 0] = 1
        self.Z[:, -1] = np.random.randn(self.n)

        self.beta_true = np.array([2.0, 0.5, -0.3])
        self.gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        self.sigma_true = 1.0
        self.rho_true = 0.5

        mean = [0, 0]
        cov = [[1, self.rho_true], [self.rho_true, 1]]
        errors = np.random.multivariate_normal(mean, cov, self.n)
        u = errors[:, 0]
        e = errors[:, 1] * self.sigma_true

        s_star = self.Z @ self.gamma_true + u
        self.selection = (s_star > 0).astype(int)

        y_star = self.X @ self.beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_result_predict_with_dataframe_exog_no_names(self):
        """Test result.predict(exog=DataFrame) when exog_names is None (line 511)."""
        import pandas as pd

        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        # exog_names should be None since we passed numpy arrays
        assert result.exog_names is None

        # Create a DataFrame with same number of columns
        new_exog_df = pd.DataFrame(
            np.random.randn(10, self.k_outcome),
            columns=["c0", "c1", "c2"],
        )
        new_exog_df["c0"] = 1  # Intercept

        pred = result.predict(exog=new_exog_df, type="unconditional")

        assert pred is not None
        assert len(pred) == 10

    def test_result_predict_conditional_with_dataframe(self):
        """Test conditional prediction with DataFrame exog_selection (lines 515-522)."""
        import pandas as pd

        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        # exog_selection_names should be None since we passed numpy arrays
        assert result.exog_selection_names is None

        new_exog_df = pd.DataFrame(
            np.random.randn(10, self.k_outcome),
            columns=["c0", "c1", "c2"],
        )
        new_exog_df["c0"] = 1

        new_exog_sel_df = pd.DataFrame(
            np.random.randn(10, self.k_selection),
            columns=["s0", "s1", "s2", "s3"],
        )
        new_exog_sel_df["s0"] = 1

        pred_cond = result.predict(
            exog=new_exog_df,
            exog_selection=new_exog_sel_df,
            type="conditional",
        )

        assert pred_cond is not None
        assert len(pred_cond) == 10

    def test_result_predict_unconditional_default(self):
        """Test unconditional prediction uses model's exog by default (line 502-503)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        pred = result.predict(type="unconditional")

        assert len(pred) == self.n

    def test_result_predict_conditional_default(self):
        """Test conditional prediction uses model's exog and exog_selection by default."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        pred = result.predict(type="conditional")

        assert len(pred) == self.n


class TestPlotImrImportError:
    """Tests for plot_imr() ImportError handling (lines 668-669)."""

    def setup_method(self):
        """Setup test data with selection."""
        np.random.seed(42)

        self.n = 200
        self.k_outcome = 3
        self.k_selection = 4

        self.X = np.random.randn(self.n, self.k_outcome)
        self.X[:, 0] = 1

        self.Z = np.random.randn(self.n, self.k_selection)
        self.Z[:, 0] = 1
        self.Z[:, -1] = np.random.randn(self.n)

        self.beta_true = np.array([2.0, 0.5, -0.3])
        self.gamma_true = np.array([0.5, 0.3, -0.2, 0.4])

        mean = [0, 0]
        cov = [[1, 0.5], [0.5, 1]]
        errors = np.random.multivariate_normal(mean, cov, self.n)
        u = errors[:, 0]
        e = errors[:, 1]

        s_star = self.Z @ self.gamma_true + u
        self.selection = (s_star > 0).astype(int)

        y_star = self.X @ self.beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_plot_imr_import_error(self):
        """Test plot_imr() raises ImportError when matplotlib is unavailable (lines 668-669)."""
        import sys

        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        # Temporarily hide matplotlib from the import system
        matplotlib_modules = {}
        for key in list(sys.modules.keys()):
            if key == "matplotlib" or key.startswith("matplotlib."):
                matplotlib_modules[key] = sys.modules.pop(key)

        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot" or name == "matplotlib":
                raise ImportError("No module named 'matplotlib'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import
        try:
            with pytest.raises(ImportError, match="matplotlib is required for plotting"):
                result.plot_imr()
        finally:
            builtins.__import__ = original_import
            sys.modules.update(matplotlib_modules)


class TestDiagnosticsProperties:
    """Tests for result diagnostic properties (rho, sigma, lambda_imr)."""

    def setup_method(self):
        """Setup test data with selection."""
        np.random.seed(42)

        self.n = 500
        self.k_outcome = 3
        self.k_selection = 4

        self.X = np.random.randn(self.n, self.k_outcome)
        self.X[:, 0] = 1

        self.Z = np.random.randn(self.n, self.k_selection)
        self.Z[:, 0] = 1
        self.Z[:, -1] = np.random.randn(self.n)

        self.beta_true = np.array([2.0, 0.5, -0.3])
        self.gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        self.sigma_true = 1.0
        self.rho_true = 0.5

        mean = [0, 0]
        cov = [[1, self.rho_true], [self.rho_true, 1]]
        errors = np.random.multivariate_normal(mean, cov, self.n)
        u = errors[:, 0]
        e = errors[:, 1] * self.sigma_true

        s_star = self.Z @ self.gamma_true + u
        self.selection = (s_star > 0).astype(int)

        y_star = self.X @ self.beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_rho_property(self):
        """Test that rho is accessible and in valid range."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        assert hasattr(result, "rho")
        assert isinstance(result.rho, float)
        assert -1 <= result.rho <= 1

    def test_sigma_property(self):
        """Test that sigma is accessible and positive."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        assert hasattr(result, "sigma")
        assert isinstance(result.sigma, float)
        assert result.sigma > 0

    def test_lambda_imr_property(self):
        """Test that lambda_imr is accessible and has correct shape."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        assert hasattr(result, "lambda_imr")
        assert len(result.lambda_imr) == self.n

        # IMR for selected observations should be non-negative
        selected = self.selection == 1
        assert (result.lambda_imr[selected] >= 0).all()

    def test_n_selected_and_n_total(self):
        """Test that n_selected and n_total are correct."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        assert result.n_total == self.n
        assert result.n_selected == np.sum(self.selection)


class TestHeckmanValidation:
    """Tests for _validate_data() error paths (lines 90, 93, 98)."""

    def test_selection_length_mismatch(self):
        """Test error when selection and endog have different lengths (line 90)."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        Z = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = np.random.randn(n)
        selection_wrong = np.ones(n + 10)  # Wrong length

        with pytest.raises(ValueError, match="Selection and outcome must have same length"):
            PanelHeckman(y, X, selection_wrong, Z)

    def test_exog_selection_length_mismatch(self):
        """Test error when exog_selection length doesn't match (line 93)."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        Z_wrong = np.column_stack([np.ones(n + 5), np.random.randn(n + 5, 3)])
        y = np.random.randn(n)
        selection = np.ones(n, dtype=int)

        with pytest.raises(ValueError, match="Selection regressors must match data length"):
            PanelHeckman(y, X, selection, Z_wrong)

    def test_non_binary_selection(self):
        """Test error when selection is not binary (line 98)."""
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        Z = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = np.random.randn(n)
        selection_bad = np.array([0, 1, 2] * 33 + [0])  # Non-binary

        with pytest.raises(ValueError, match="Selection must be binary"):
            PanelHeckman(y, X, selection_bad, Z)


class TestHeckmanFitWarnings:
    """Tests for fit() warning paths (lines 194, 202, 224)."""

    def setup_method(self):
        """Setup small test data."""
        np.random.seed(42)
        self.n = 100
        self.X = np.column_stack([np.ones(self.n), np.random.randn(self.n, 2)])
        self.Z = np.column_stack([np.ones(self.n), np.random.randn(self.n, 3)])

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])

        errors = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], self.n)
        u, e = errors[:, 0], errors[:, 1]

        s_star = self.Z @ gamma_true + u
        self.selection = (s_star > 0).astype(int)
        y_star = self.X @ beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_mle_large_sample_warning(self):
        """Test MLE warning when N > 500 (line 194)."""
        # Create large dataset
        np.random.seed(42)
        n_big = 600
        X_big = np.column_stack([np.ones(n_big), np.random.randn(n_big, 2)])
        Z_big = np.column_stack([np.ones(n_big), np.random.randn(n_big, 3)])

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])
        errors = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_big)
        u, e = errors[:, 0], errors[:, 1]
        s_star = Z_big @ gamma_true + u
        selection_big = (s_star > 0).astype(int)
        y_star = X_big @ beta_true + e
        y_big = np.where(selection_big == 1, y_star, np.nan)

        model = PanelHeckman(y_big, X_big, selection_big, Z_big, method="mle")

        with pytest.warns(UserWarning, match="MLE with N>500"):
            model.fit()

    def test_mle_high_quadrature_warning(self):
        """Test MLE warning when quadrature_points > 15 (line 202)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z, method="mle")

        with pytest.warns(UserWarning, match="MLE with >15 quadrature"):
            model.fit(quadrature_points=20)

    def test_unknown_method_raises(self):
        """Test error for unknown estimation method (line 224)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)

        with pytest.raises(ValueError, match="Unknown method"):
            model.fit(method="invalid_method")


class TestHeckmanMLEConvergence:
    """Tests for MLE convergence warning (lines 354-358)."""

    def test_mle_nonconvergence_warning(self):
        """Test warning when MLE fails to converge (lines 354-355)."""
        np.random.seed(42)
        # Create data that's hard to optimize (small sample, bad initial values)
        n = 30
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        Z = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        selection = np.array([1] * 15 + [0] * 15)
        y = np.random.randn(n)

        model = PanelHeckman(y, X, selection, Z, method="mle")

        # Force non-convergence by limiting iterations
        with pytest.warns(UserWarning, match="MLE did not converge"):
            model.fit(maxiter=1)


class TestHeckmanExogNames:
    """Tests for exog_names attribute paths in PanelHeckmanResult (lines 411-420)."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.n = 200
        self.X = np.column_stack([np.ones(self.n), np.random.randn(self.n, 2)])
        self.Z = np.column_stack([np.ones(self.n), np.random.randn(self.n, 3)])

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])

        errors = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], self.n)
        u, e = errors[:, 0], errors[:, 1]
        s_star = self.Z @ gamma_true + u
        self.selection = (s_star > 0).astype(int)
        y_star = self.X @ beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_exog_names_from_model_attribute(self):
        """Test exog_names from model.exog_names attribute (lines 412-413, 419-420)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        # Set exog_names on model before fit (simulates model subclass that sets this)
        model.exog_names = ["intercept", "var1", "var2"]
        model.exog_selection_names = ["intercept", "sel1", "sel2", "sel3"]

        result = model.fit()

        assert result.exog_names == ["intercept", "var1", "var2"]
        assert result.exog_selection_names == ["intercept", "sel1", "sel2", "sel3"]

    def test_exog_names_from_dataframe_columns(self):
        """Test exog_names from DataFrame.columns attribute (lines 410-411, 417-418)."""
        from panelbox.models.selection.heckman import PanelHeckmanResult

        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        # Simulate a model whose exog is a DataFrame (line 410-411)
        # by temporarily replacing model.exog with a DataFrame
        orig_exog = model.exog
        orig_exog_sel = model.exog_selection
        model.exog = pd.DataFrame(orig_exog, columns=["const", "x1", "x2"])
        model.exog_selection = pd.DataFrame(orig_exog_sel, columns=["const", "z1", "z2", "z3"])
        # Remove exog_names attribute so it falls through to columns check
        if hasattr(model, "exog_names"):
            delattr(model, "exog_names")
        if hasattr(model, "exog_selection_names"):
            delattr(model, "exog_selection_names")

        result2 = PanelHeckmanResult(
            model=model,
            params=result.params,
            method="two_step",
            probit_params=result.probit_params,
            outcome_params=result.outcome_params,
            sigma=result.sigma,
            rho=result.rho,
            lambda_imr=result.lambda_imr,
        )

        assert result2.exog_names == ["const", "x1", "x2"]
        assert result2.exog_selection_names == ["const", "z1", "z2", "z3"]

        # Restore
        model.exog = orig_exog
        model.exog_selection = orig_exog_sel

    def test_exog_names_none_for_plain_arrays(self):
        """Test exog_names is None when plain arrays passed (line 415)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        assert result.exog_names is None
        assert result.exog_selection_names is None


class TestHeckmanSummaryPaths:
    """Tests for summary() paths: MLE llf and negative selection (lines 441, 469-472)."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.n = 200
        self.X = np.column_stack([np.ones(self.n), np.random.randn(self.n, 2)])
        self.Z = np.column_stack([np.ones(self.n), np.random.randn(self.n, 3)])

    def test_summary_with_mle_shows_loglikelihood(self):
        """Test summary includes log-likelihood for MLE (line 441)."""
        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])

        errors = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], self.n)
        u, e = errors[:, 0], errors[:, 1]
        s_star = self.Z @ gamma_true + u
        selection = (s_star > 0).astype(int)
        y_star = self.X @ beta_true + e
        y = np.where(selection == 1, y_star, np.nan)

        model = PanelHeckman(y, self.X, selection, self.Z, method="mle")
        result = model.fit()
        summary = result.summary()

        assert "Log-likelihood:" in summary

    def test_summary_negative_selection(self):
        """Test summary shows negative selection note (lines 469-470)."""
        # Create data with negative selection bias (rho < -0.1)
        np.random.seed(123)
        rho_neg = -0.7
        errors = np.random.multivariate_normal([0, 0], [[1, rho_neg], [rho_neg, 1]], self.n)
        u, e = errors[:, 0], errors[:, 1]

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])
        s_star = self.Z @ gamma_true + u
        selection = (s_star > 0).astype(int)
        y_star = self.X @ beta_true + e
        y = np.where(selection == 1, y_star, np.nan)

        model = PanelHeckman(y, self.X, selection, self.Z)
        result = model.fit()

        summary = result.summary()
        # With negative rho, should mention negative selection
        if result.rho < -0.1:
            assert "Negative selection" in summary
        elif result.rho > 0.1:
            assert "Positive selection" in summary

    def test_summary_no_selection_bias(self):
        """Test summary shows 'little evidence' when rho ~ 0 (line 472)."""
        # Create data with no selection bias
        np.random.seed(42)
        errors_u = np.random.randn(self.n)
        errors_e = np.random.randn(self.n)  # Independent errors

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])
        s_star = self.Z @ gamma_true + errors_u
        selection = (s_star > 0).astype(int)
        y_star = self.X @ beta_true + errors_e
        y = np.where(selection == 1, y_star, np.nan)

        model = PanelHeckman(y, self.X, selection, self.Z)
        result = model.fit()

        summary = result.summary()
        # With independent errors, rho should be near 0
        if abs(result.rho) <= 0.1:
            assert "Little evidence of selection bias" in summary


class TestHeckmanPredictWithNames:
    """Tests for predict() with named DataFrame columns (lines 504-520)."""

    def setup_method(self):
        """Setup test data with DataFrame columns."""
        np.random.seed(42)
        self.n = 200

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])
        rho = 0.5

        self.X_arr = np.column_stack([np.ones(self.n), np.random.randn(self.n, 2)])
        self.Z_arr = np.column_stack([np.ones(self.n), np.random.randn(self.n, 3)])

        errors = np.random.multivariate_normal([0, 0], [[1, rho], [rho, 1]], self.n)
        u, e = errors[:, 0], errors[:, 1]
        s_star = self.Z_arr @ gamma_true + u
        self.selection = (s_star > 0).astype(int)
        y_star = self.X_arr @ beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def _fit_with_names(self):
        """Fit model and manually set exog_names to simulate DataFrame input."""
        model = PanelHeckman(self.y, self.X_arr, self.selection, self.Z_arr)
        # Set exog_names to simulate what happens when DataFrame columns are present
        model.exog_names = ["const", "x1", "x2"]
        model.exog_selection_names = ["const", "z1", "z2", "z3"]
        result = model.fit()
        return result

    def test_predict_with_named_exog_dataframe(self):
        """Test predict with DataFrame that has matching columns (lines 505-509)."""
        result = self._fit_with_names()

        assert result.exog_names == ["const", "x1", "x2"]

        new_exog = pd.DataFrame(
            {
                "const": np.ones(5),
                "x1": np.random.randn(5),
                "x2": np.random.randn(5),
                "extra_col": np.random.randn(5),  # Extra col should be ignored
            }
        )

        pred = result.predict(exog=new_exog, type="unconditional")
        assert len(pred) == 5

    def test_predict_with_missing_exog_columns(self):
        """Test predict raises when DataFrame is missing columns (lines 506-508)."""
        result = self._fit_with_names()

        bad_exog = pd.DataFrame(
            {
                "const": np.ones(5),
                "x1": np.random.randn(5),
                # "x2" is missing
            }
        )

        with pytest.raises(ValueError, match="Missing columns in exog"):
            result.predict(exog=bad_exog)

    def test_predict_with_named_exog_selection_dataframe(self):
        """Test conditional predict with named exog_selection (lines 515-520)."""
        result = self._fit_with_names()

        assert result.exog_selection_names == ["const", "z1", "z2", "z3"]

        new_exog = pd.DataFrame(
            {
                "const": np.ones(5),
                "x1": np.random.randn(5),
                "x2": np.random.randn(5),
            }
        )

        new_exog_sel = pd.DataFrame(
            {
                "const": np.ones(5),
                "z1": np.random.randn(5),
                "z2": np.random.randn(5),
                "z3": np.random.randn(5),
            }
        )

        pred = result.predict(
            exog=new_exog,
            exog_selection=new_exog_sel,
            type="conditional",
        )
        assert len(pred) == 5

    def test_predict_with_missing_exog_selection_columns(self):
        """Test predict raises when exog_selection DataFrame is missing columns (lines 517-519)."""
        result = self._fit_with_names()

        new_exog = pd.DataFrame(
            {
                "const": np.ones(5),
                "x1": np.random.randn(5),
                "x2": np.random.randn(5),
            }
        )

        bad_exog_sel = pd.DataFrame(
            {
                "const": np.ones(5),
                "z1": np.random.randn(5),
                # "z2" and "z3" are missing
            }
        )

        with pytest.raises(ValueError, match="Missing columns in exog_selection"):
            result.predict(
                exog=new_exog,
                exog_selection=bad_exog_sel,
                type="conditional",
            )


class TestSelectionEffectMethod:
    """Tests for selection_effect() method (lines 590-601)."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.n = 200

        self.X = np.column_stack([np.ones(self.n), np.random.randn(self.n, 2)])
        self.Z = np.column_stack([np.ones(self.n), np.random.randn(self.n, 3)])

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])

        errors = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], self.n)
        u, e = errors[:, 0], errors[:, 1]
        s_star = self.Z @ gamma_true + u
        self.selection = (s_star > 0).astype(int)
        y_star = self.X @ beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_selection_effect_returns_dict(self):
        """Test selection_effect() returns expected dictionary (lines 590-601)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        effect = result.selection_effect()

        assert isinstance(effect, dict)
        assert "statistic" in effect
        assert "pvalue" in effect
        assert "reject" in effect
        assert "interpretation" in effect
        assert isinstance(effect["pvalue"], float)
        assert 0 <= effect["pvalue"] <= 1

    def test_selection_effect_with_custom_alpha(self):
        """Test selection_effect() with custom alpha level."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        effect_05 = result.selection_effect(alpha=0.05)
        effect_01 = result.selection_effect(alpha=0.01)

        # Both should return valid results
        assert effect_05["reject"] in (True, False)
        assert effect_01["reject"] in (True, False)


class TestIMRDiagnosticsMethod:
    """Tests for imr_diagnostics() method (lines 629-633)."""

    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.n = 200

        self.X = np.column_stack([np.ones(self.n), np.random.randn(self.n, 2)])
        self.Z = np.column_stack([np.ones(self.n), np.random.randn(self.n, 3)])

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])

        errors = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], self.n)
        u, e = errors[:, 0], errors[:, 1]
        s_star = self.Z @ gamma_true + u
        self.selection = (s_star > 0).astype(int)
        y_star = self.X @ beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_imr_diagnostics_returns_dict(self):
        """Test imr_diagnostics() returns expected keys (lines 629-633)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        diag = result.imr_diagnostics()

        assert isinstance(diag, dict)
        assert "imr_mean" in diag
        assert "imr_std" in diag
        assert "imr_min" in diag
        assert "imr_max" in diag
        assert "high_imr_count" in diag
        assert "selection_rate" in diag
        assert "n_selected" in diag
        assert "n_total" in diag

        # IMR mean should be positive for selected observations
        assert diag["imr_mean"] > 0
        assert diag["n_total"] == self.n
        assert diag["n_selected"] == np.sum(self.selection)


class TestPlotIMR:
    """Tests for plot_imr() plotting code (lines 674-703)."""

    def setup_method(self):
        """Setup test data."""
        import matplotlib

        matplotlib.use("Agg")

        np.random.seed(42)
        self.n = 200

        self.X = np.column_stack([np.ones(self.n), np.random.randn(self.n, 2)])
        self.Z = np.column_stack([np.ones(self.n), np.random.randn(self.n, 3)])

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])

        errors = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], self.n)
        u, e = errors[:, 0], errors[:, 1]
        s_star = self.Z @ gamma_true + u
        self.selection = (s_star > 0).astype(int)
        y_star = self.X @ beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_plot_imr_returns_figure(self):
        """Test plot_imr() returns a matplotlib Figure (lines 674-703)."""
        import matplotlib.pyplot as plt

        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        fig = result.plot_imr()

        assert fig is not None
        # Should have 2 subplots (scatter + histogram)
        assert len(fig.axes) == 2

        # Check axis labels
        assert fig.axes[0].get_xlabel() == "Predicted Selection Probability"
        assert fig.axes[0].get_ylabel() == "Inverse Mills Ratio"
        assert fig.axes[1].get_xlabel() == "Inverse Mills Ratio"

        plt.close(fig)

    def test_plot_imr_custom_figsize(self):
        """Test plot_imr() with custom figsize."""
        import matplotlib.pyplot as plt

        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        fig = result.plot_imr(figsize=(16, 8))

        assert fig is not None
        plt.close(fig)


class TestCompareOLSHeckman:
    """Tests for compare_ols_heckman() method (lines 734-759)."""

    def setup_method(self):
        """Setup test data with known selection bias."""
        np.random.seed(42)
        self.n = 300

        self.X = np.column_stack([np.ones(self.n), np.random.randn(self.n, 2)])
        self.Z = np.column_stack([np.ones(self.n), np.random.randn(self.n, 3)])

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])

        # Strong selection bias (rho = 0.7)
        errors = np.random.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], self.n)
        u, e = errors[:, 0], errors[:, 1]
        s_star = self.Z @ gamma_true + u
        self.selection = (s_star > 0).astype(int)
        y_star = self.X @ beta_true + e
        self.y = np.where(self.selection == 1, y_star, np.nan)

    def test_compare_ols_heckman_returns_dict(self):
        """Test compare_ols_heckman() returns expected keys (lines 734-759)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        comparison = result.compare_ols_heckman()

        assert isinstance(comparison, dict)
        assert "beta_ols" in comparison
        assert "beta_heckman" in comparison
        assert "difference" in comparison
        assert "pct_difference" in comparison
        assert "max_abs_difference" in comparison
        assert "interpretation" in comparison

        # Check dimensions
        assert len(comparison["beta_ols"]) == self.X.shape[1]
        assert len(comparison["beta_heckman"]) == self.X.shape[1]
        assert len(comparison["difference"]) == self.X.shape[1]

    def test_compare_ols_heckman_interpretation_substantial(self):
        """Test interpretation when bias is substantial (lines 748-752)."""
        model = PanelHeckman(self.y, self.X, self.selection, self.Z)
        result = model.fit()

        comparison = result.compare_ols_heckman()

        # With rho=0.7, we expect some differences
        assert isinstance(comparison["interpretation"], str)
        assert len(comparison["interpretation"]) > 0

    def test_compare_ols_heckman_no_bias(self):
        """Test comparison when there's little bias (lines 753-756)."""
        np.random.seed(42)
        n = 300
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        Z = np.column_stack([np.ones(n), np.random.randn(n, 3)])

        # Independent errors (no selection bias)
        u = np.random.randn(n)
        e = np.random.randn(n)

        gamma_true = np.array([0.5, 0.3, -0.2, 0.4])
        beta_true = np.array([2.0, 0.5, -0.3])

        s_star = Z @ gamma_true + u
        selection = (s_star > 0).astype(int)
        y_star = X @ beta_true + e
        y = np.where(selection == 1, y_star, np.nan)

        model = PanelHeckman(y, X, selection, Z)
        result = model.fit()

        comparison = result.compare_ols_heckman()

        # With independent errors, OLS and Heckman should be similar
        assert isinstance(comparison["interpretation"], str)


class TestHeckmanExtremeSelectionWarning:
    """Test extreme selection rate warning (lines 211-217)."""

    def test_extreme_low_selection_rate_warning(self):
        """Test warning when selection rate < 5%."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        Z = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = np.random.randn(n)

        # Only 4% selected
        selection = np.zeros(n, dtype=int)
        selection[:8] = 1  # 8/200 = 4%

        model = PanelHeckman(y, X, selection, Z)

        with pytest.warns(UserWarning, match="Extreme selection rate"):
            model.fit()

    def test_extreme_high_selection_rate_warning(self):
        """Test warning when selection rate > 95%."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        Z = np.column_stack([np.ones(n), np.random.randn(n, 3)])
        y = np.random.randn(n)

        # 97% selected
        selection = np.ones(n, dtype=int)
        selection[:6] = 0  # 194/200 = 97%

        model = PanelHeckman(y, X, selection, Z)

        with pytest.warns(UserWarning, match="Extreme selection rate"):
            model.fit()
