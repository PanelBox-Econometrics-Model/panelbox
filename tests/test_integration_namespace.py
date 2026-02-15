"""
Integration tests for PanelBox namespace imports.

Tests that all advanced methods are properly exposed through the
top-level panelbox namespace.
"""

import pytest


class TestNamespaceImports:
    """Test that all expected classes and functions are accessible from panelbox namespace."""

    def test_gmm_advanced_imports(self):
        """Test advanced GMM estimator imports."""
        import panelbox

        assert hasattr(panelbox, "ContinuousUpdatedGMM")
        assert hasattr(panelbox, "BiasCorrectedGMM")
        assert hasattr(panelbox, "GMMDiagnostics")

    def test_selection_model_imports(self):
        """Test selection model imports."""
        import panelbox

        assert hasattr(panelbox, "PanelHeckman")
        assert hasattr(panelbox, "PanelHeckmanResult")
        assert hasattr(panelbox, "compute_imr")
        assert hasattr(panelbox, "imr_derivative")
        assert hasattr(panelbox, "imr_diagnostics")
        assert hasattr(panelbox, "test_selection_effect")

    def test_cointegration_test_imports(self):
        """Test cointegration test imports."""
        import panelbox

        assert hasattr(panelbox, "westerlund_test")
        assert hasattr(panelbox, "WesterlundResult")
        assert hasattr(panelbox, "pedroni_test")
        assert hasattr(panelbox, "PedroniResult")
        assert hasattr(panelbox, "kao_test")
        assert hasattr(panelbox, "KaoResult")

    def test_unit_root_test_imports(self):
        """Test unit root test imports."""
        import panelbox

        assert hasattr(panelbox, "hadri_test")
        assert hasattr(panelbox, "HadriResult")
        assert hasattr(panelbox, "breitung_test")
        assert hasattr(panelbox, "BreitungResult")
        assert hasattr(panelbox, "panel_unit_root_test")
        assert hasattr(panelbox, "PanelUnitRootResult")

    def test_specification_test_imports(self):
        """Test specification test imports."""
        import panelbox

        assert hasattr(panelbox, "j_test")
        assert hasattr(panelbox, "JTestResult")
        assert hasattr(panelbox, "cox_test")
        assert hasattr(panelbox, "wald_encompassing_test")
        assert hasattr(panelbox, "likelihood_ratio_test")
        assert hasattr(panelbox, "EncompassingResult")

    def test_discrete_choice_imports(self):
        """Test discrete choice model imports."""
        import panelbox

        assert hasattr(panelbox, "MultinomialLogit")
        assert hasattr(panelbox, "MultinomialLogitResult")
        assert hasattr(panelbox, "ConditionalLogit")
        assert hasattr(panelbox, "PooledLogit")
        assert hasattr(panelbox, "PooledProbit")
        assert hasattr(panelbox, "FixedEffectsLogit")
        assert hasattr(panelbox, "RandomEffectsProbit")

    def test_count_model_imports(self):
        """Test count model imports."""
        import panelbox

        assert hasattr(panelbox, "PPML")
        assert hasattr(panelbox, "PPMLResult")
        assert hasattr(panelbox, "PooledPoisson")
        assert hasattr(panelbox, "PoissonFixedEffects")
        assert hasattr(panelbox, "RandomEffectsPoisson")
        assert hasattr(panelbox, "PoissonQML")

    def test_quantile_regression_imports(self):
        """Test quantile regression imports."""
        import panelbox

        assert hasattr(panelbox, "PooledQuantile")
        assert hasattr(panelbox, "PooledQuantileResults")
        assert hasattr(panelbox, "QuantileBootstrap")
        assert hasattr(panelbox, "BootstrapResult")
        assert hasattr(panelbox, "QuantileRegressionDiagnostics")


class TestGlobalNamespaceUsage:
    """Test that classes can be instantiated using global namespace imports."""

    def test_cue_gmm_instantiation(self):
        """Test CUE-GMM can be accessed via global namespace."""
        import panelbox

        # Should not raise AttributeError
        cls = panelbox.ContinuousUpdatedGMM
        assert cls.__name__ == "ContinuousUpdatedGMM"

    def test_panel_heckman_instantiation(self):
        """Test Panel Heckman can be accessed via global namespace."""
        import panelbox

        cls = panelbox.PanelHeckman
        assert cls.__name__ == "PanelHeckman"

    def test_westerlund_test_callable(self):
        """Test westerlund_test can be called via global namespace."""
        import panelbox

        func = panelbox.westerlund_test
        assert callable(func)

    def test_j_test_callable(self):
        """Test j_test can be called via global namespace."""
        import panelbox

        func = panelbox.j_test
        assert callable(func)

    def test_multinomial_logit_instantiation(self):
        """Test MultinomialLogit can be accessed via global namespace."""
        import panelbox

        cls = panelbox.MultinomialLogit
        assert cls.__name__ == "MultinomialLogit"

    def test_ppml_instantiation(self):
        """Test PPML can be accessed via global namespace."""
        import panelbox

        cls = panelbox.PPML
        assert cls.__name__ == "PPML"


class TestAllExports:
    """Test that __all__ contains all expected exports."""

    def test_all_contains_advanced_gmm(self):
        """Test __all__ exports advanced GMM classes."""
        import panelbox

        assert "ContinuousUpdatedGMM" in panelbox.__all__
        assert "BiasCorrectedGMM" in panelbox.__all__
        assert "GMMDiagnostics" in panelbox.__all__

    def test_all_contains_selection_models(self):
        """Test __all__ exports selection model classes."""
        import panelbox

        assert "PanelHeckman" in panelbox.__all__
        assert "PanelHeckmanResult" in panelbox.__all__

    def test_all_contains_cointegration_tests(self):
        """Test __all__ exports cointegration test functions."""
        import panelbox

        assert "westerlund_test" in panelbox.__all__
        assert "pedroni_test" in panelbox.__all__
        assert "kao_test" in panelbox.__all__

    def test_all_contains_unit_root_tests(self):
        """Test __all__ exports unit root test functions."""
        import panelbox

        assert "hadri_test" in panelbox.__all__
        assert "breitung_test" in panelbox.__all__
        assert "panel_unit_root_test" in panelbox.__all__

    def test_all_contains_specification_tests(self):
        """Test __all__ exports specification test functions."""
        import panelbox

        assert "j_test" in panelbox.__all__

    def test_all_contains_discrete_choice(self):
        """Test __all__ exports discrete choice models."""
        import panelbox

        assert "MultinomialLogit" in panelbox.__all__
        assert "MultinomialLogitResult" in panelbox.__all__

    def test_all_contains_count_models(self):
        """Test __all__ exports count models."""
        import panelbox

        assert "PPML" in panelbox.__all__
        assert "PPMLResult" in panelbox.__all__
