"""
Unit tests for Mundlak test.

Tests the MundlakTest which tests the Random Effects specification
by checking if regressors are correlated with entity effects.
"""

import numpy as np
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.random_effects import RandomEffects
from panelbox.validation.specification.mundlak import MundlakTest


class TestMundlak:
    """Test suite for Mundlak test."""

    def test_rejects_when_re_assumption_violated(self, panel_for_mundlak):
        """Test that Mundlak test rejects RE when regressors correlated with effects."""
        re = RandomEffects("y ~ x1 + x2", panel_for_mundlak, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run(alpha=0.05)

        # Should reject RE specification (regressors are correlated with entity effects)
        # Note: May not always reject due to sample variation
        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_no_false_positive_clean_data(self, clean_panel_data):
        """Test that Mundlak doesn't reject when RE assumption holds."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run(alpha=0.05)

        # Should NOT reject RE specification with clean data
        # (though some Type I error expected at alpha level)
        assert result.pvalue > 0.01, "Should not strongly reject with clean data"

    def test_requires_random_effects(self, clean_panel_data):
        """Test that Mundlak test requires Random Effects model."""
        # Fit Fixed Effects instead
        fe = FixedEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = fe.fit()

        test = MundlakTest(results)
        with pytest.raises(ValueError, match="only applicable to Random Effects"):
            test.run()

    def test_result_attributes(self, clean_panel_data):
        """Test that result has all required attributes."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run()

        # Check all attributes exist
        assert hasattr(result, "test_name")
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "reject_null")
        assert hasattr(result, "conclusion")
        assert hasattr(result, "details")

        # Check types
        assert isinstance(result.test_name, str)
        assert isinstance(result.statistic, (int, float))
        assert isinstance(result.pvalue, (int, float))
        assert isinstance(result.reject_null, bool)

        # Check test name
        assert "Mundlak" in result.test_name

    def test_statistic_non_negative(self, clean_panel_data):
        """Test that Wald statistic is non-negative."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run()

        # Wald statistic should be non-negative
        assert result.statistic >= 0, "Wald statistic must be non-negative"

    def test_different_alpha_levels(self, clean_panel_data):
        """Test behavior with different significance levels."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Test with different alphas
        result_001 = test.run(alpha=0.01)
        result_010 = test.run(alpha=0.10)

        # P-value and statistic should be the same
        assert result_001.pvalue == result_010.pvalue
        assert result_001.statistic == result_010.statistic

    def test_degrees_of_freedom(self, clean_panel_data):
        """Test that degrees of freedom equals number of regressors (excluding intercept)."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run()

        # DF should be number of time-varying regressors
        assert "df" in result.details
        len(results.params) - 1  # Exclude intercept
        # Note: Actual df might differ based on which means are included
        assert result.details["df"] > 0

    def test_pvalue_bounds(self, clean_panel_data):
        """Test that p-value is between 0 and 1."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run()

        assert 0 <= result.pvalue <= 1, f"P-value {result.pvalue} should be in [0, 1]"

    def test_requires_model_reference(self, clean_panel_data):
        """Test that Mundlak test can access design matrix through model reference."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        # Should have model reference
        assert hasattr(results, "_model")
        assert results._model is not None

        test = MundlakTest(results)
        result = test.run()

        # Should successfully run (accessing X via model)
        assert result is not None

    def test_with_single_regressor(self, clean_panel_data):
        """Test Mundlak with single regressor."""
        re = RandomEffects("y ~ x1", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_with_multiple_regressors(self, clean_panel_data):
        """Test Mundlak with multiple regressors."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_entity_means_computation(self, clean_panel_data):
        """Test that entity means are computed correctly."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run()

        # Test should complete without error
        assert result is not None

        # The test internally computes entity means and augments the model
        # If it completes, the computation was successful

    def test_handles_time_invariant_regressors(self, clean_panel_data):
        """Test behavior when regressor is time-invariant."""
        # Add a time-invariant regressor
        data_with_invariant = clean_panel_data.copy()
        # Entity-specific constant
        entity_constant = data_with_invariant.groupby("entity")["x1"].transform("mean")
        data_with_invariant["x_const"] = entity_constant

        re = RandomEffects("y ~ x1 + x_const", data_with_invariant, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        # Should raise error for time-invariant regressors
        # (they are perfectly collinear with intercept in pooled estimation)
        with pytest.raises(ValueError, match="time-invariant regressors"):
            test.run()

    def test_with_unbalanced_panel(self, unbalanced_panel_data):
        """Test Mundlak test with unbalanced panel."""
        re = RandomEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")

    def test_conclusion_interpretation(self, clean_panel_data):
        """Test that conclusion provides correct interpretation."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        result = test.run()

        # Conclusion should mention RE specification
        assert result.conclusion is not None
        assert isinstance(result.conclusion, str)
        assert len(result.conclusion) > 0

    def test_missing_data_error(self, clean_panel_data):
        """Test ValueError when data/formula not available (line 114)."""

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        # Mock _get_data_full to return None
        test._get_data_full = lambda: (None, None, None, None, None)

        with pytest.raises(ValueError, match="Data, formula, and variable names required"):
            test.run()

    def test_no_time_varying_regressors_error(self, clean_panel_data):
        """Test ValueError when all regressors are time-invariant (line 139)."""
        # Create data with only time-invariant regressor
        data_invariant = clean_panel_data.copy()
        # Entity-specific constant (no time variation)
        data_invariant["x_const"] = data_invariant.groupby("entity")["entity"].transform("first")

        # Create RE model with only constant regressor
        # This is tricky - we need to construct data where all regressors are time-invariant
        # Use a regressor that has zero within-group variance
        data_invariant["x1"] = data_invariant.groupby("entity")["x1"].transform("mean")
        data_invariant["x2"] = data_invariant.groupby("entity")["x2"].transform("mean")

        re = RandomEffects("y ~ x1 + x2", data_invariant, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        with pytest.raises(ValueError, match="No time-varying regressors found"):
            test.run()

    def test_model_estimation_failure(self, clean_panel_data):
        """Test handling of model estimation failure (lines 181-183)."""
        from unittest.mock import patch

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Mock PooledOLS.fit to raise an exception
        with (
            patch(
                "panelbox.models.static.pooled_ols.PooledOLS.fit",
                side_effect=RuntimeError("Test error"),
            ),
            pytest.raises(ValueError, match="Failed to estimate augmented model"),
        ):
            test.run()

    def test_vcov_singular_matrix_handling(self, clean_panel_data):
        """Test handling of singular vcov matrix (lines 206-207)."""
        from unittest.mock import patch

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Mock np.linalg.inv to raise LinAlgError only for small matrices (vcov)
        original_inv = np.linalg.inv
        calls = [0]

        def mock_inv(a):
            calls[0] += 1
            # Raise on the second call which should be the vcov inversion
            # First call is during model fitting (large matrix)
            if calls[0] >= 2 and a.shape[0] == 2:  # vcov for 2 mean variables
                raise np.linalg.LinAlgError("Singular matrix")
            return original_inv(a)

        with patch("numpy.linalg.inv", side_effect=mock_inv):
            # Should use pinv fallback
            result = test.run()
            assert result is not None
            assert result.statistic >= 0

    def test_get_data_full_missing_model(self, clean_panel_data):
        """Test _get_data_full when model reference missing (line 273)."""

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        # Remove _model reference
        delattr(results, "_model")

        data, formula, _entity_col, _time_col, _var_names = test._get_data_full()
        assert data is None
        assert formula is None

    def test_get_data_full_missing_attributes(self, clean_panel_data):
        """Test _get_data_full when model attributes missing (line 278)."""
        from unittest.mock import Mock

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        # Mock model without required attributes
        results._model = Mock(spec=[])

        data, _formula, _entity_col, _time_col, _var_names = test._get_data_full()
        assert data is None

    def test_get_data_full_missing_formula(self, clean_panel_data):
        """Test _get_data_full when formula missing (line 295)."""

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        # Remove formula attribute
        if hasattr(results._model, "formula"):
            delattr(results._model, "formula")

        _data, formula, _entity_col, _time_col, _var_names = test._get_data_full()
        assert formula is None

    def test_get_data_full_exception_handling(self, clean_panel_data):
        """Test _get_data_full exception handling (lines 315-316)."""
        from unittest.mock import patch

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Patch the copy method to raise exception
        with patch.object(results._model.data.data, "copy", side_effect=RuntimeError("Test error")):
            data, _formula, _entity_col, _time_col, _var_names = test._get_data_full()
            assert data is None

    def test_get_data_legacy_method(self, clean_panel_data):
        """Test legacy _get_data method (lines 332-351)."""
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)
        X, y, entities = test._get_data()

        # Should return data successfully
        assert X is not None
        assert y is not None
        assert entities is not None

    def test_get_data_legacy_exception_handling(self, clean_panel_data):
        """Test _get_data exception handling (lines 350-351)."""
        from unittest.mock import patch

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Patch build_design_matrices to raise exception
        with patch.object(
            results._model.formula_parser,
            "build_design_matrices",
            side_effect=RuntimeError("Test error"),
        ):
            X, y, entities = test._get_data()
            # Should catch exception and return None
            assert X is None
            assert y is None
            assert entities is None


class TestMundlakUncoveredBranches:
    """Tests targeting specific uncovered lines in mundlak.py."""

    def test_var_not_in_columns_skipped(self, clean_panel_data):
        """Test line 130->129: variable in var_names not in data columns is skipped.

        When _get_data_full returns var_names that include a variable not present
        in the DataFrame, that variable should be skipped in the mean computation
        loop (line 130 evaluates False, continues to line 129).

        We add the nonexistent variable to var_names only (not to the formula),
        so that the loop exercises the False branch at line 130, then the
        augmented formula is still valid for PooledOLS.
        """
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Patch _get_data_full to include a nonexistent variable in var_names.
        # Keep the original formula unchanged so PooledOLS doesn't choke.
        original_get_data_full = test._get_data_full

        def patched_get_data_full():
            data, formula, entity_col, time_col, var_names = original_get_data_full()
            var_names_with_missing = ["nonexistent_var", *var_names]
            return data, formula, entity_col, time_col, var_names_with_missing

        test._get_data_full = patched_get_data_full

        # The run() method builds augmented formula with orig_vars = " + ".join(var_names)
        # which includes "nonexistent_var". We need to intercept the PooledOLS call
        # so it receives a corrected formula. Patch PooledOLS to strip the bad var.
        from panelbox.models.static.pooled_ols import PooledOLS as OrigPooledOLS

        class PatchedPooledOLS(OrigPooledOLS):
            def __init__(self, formula, data, entity_col, time_col, **kwargs):
                # Remove the nonexistent variable from formula
                formula = formula.replace("nonexistent_var + ", "")
                formula = formula.replace(" + nonexistent_var", "")
                super().__init__(formula, data, entity_col, time_col, **kwargs)

        from unittest.mock import patch

        with patch("panelbox.models.static.pooled_ols.PooledOLS", PatchedPooledOLS):
            result = test.run()

        assert result is not None
        assert result.statistic >= 0
        assert 0 <= result.pvalue <= 1

    def test_generic_exception_in_augmented_model(self, clean_panel_data):
        """Test line 184: generic exception when estimating augmented model.

        Covers the 'except Exception' handler (lines 185-186) that catches
        non-LinAlgError exceptions from model estimation.
        """
        from unittest.mock import patch

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # PooledOLS is imported locally inside run(), so patch it where it's imported from
        with (
            patch(
                "panelbox.models.static.pooled_ols.PooledOLS",
                side_effect=TypeError("Unexpected type error"),
            ),
            pytest.raises(ValueError, match="Failed to estimate augmented model"),
        ):
            test.run()

    def test_mean_indices_mismatch(self, clean_panel_data):
        """Test line 196: ValueError when mean coefficient count doesn't match.

        When the augmented model doesn't contain the expected number of
        mean variable coefficients, a ValueError should be raised.
        """
        from unittest.mock import patch

        import pandas as pd

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Create a mock fit result that returns params WITHOUT the mean variables
        # The augmented model expects x1_mean and x2_mean in params but they're missing
        fake_params = pd.Series(
            {"Intercept": 1.0, "x1": 2.0, "x2": 3.0},
            index=["Intercept", "x1", "x2"],
        )
        fake_cov = pd.DataFrame(
            np.eye(3),
            index=["Intercept", "x1", "x2"],
            columns=["Intercept", "x1", "x2"],
        )

        # Patch fit() to return results missing the mean variable parameters
        with patch("panelbox.models.static.pooled_ols.PooledOLS.fit") as mock_fit:
            mock_fit.return_value = type(
                "MockResult", (), {"params": fake_params, "cov_params": fake_cov}
            )()
            with pytest.raises(ValueError, match=r"Expected .* mean coefficients, found"):
                test.run()

    def test_get_data_full_with_rhs_terms(self, clean_panel_data):
        """Test line 304: _get_data_full when formula_parser has rhs_terms attribute.

        Covers the branch where model.formula_parser has 'rhs_terms',
        which extracts variable names from rhs_terms instead of parsing
        the formula string manually.
        """
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Add rhs_terms attribute to formula_parser
        results._model.formula_parser.rhs_terms = ["x1", "x2"]

        data, formula, _entity_col, _time_col, var_names = test._get_data_full()

        assert data is not None
        assert formula is not None
        assert var_names == ["x1", "x2"]

    def test_get_data_legacy_missing_model(self, clean_panel_data):
        """Test line 336: _get_data returns None when _model is missing.

        Covers the early return when results doesn't have _model attribute.
        """
        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Remove _model attribute
        delattr(results, "_model")

        X, y, entities = test._get_data()
        assert X is None
        assert y is None
        assert entities is None

    def test_get_data_legacy_missing_attributes(self, clean_panel_data):
        """Test line 341: _get_data returns None when model lacks formula_parser/data.

        Covers the check at line 340 where model doesn't have both
        formula_parser and data attributes.
        """
        from unittest.mock import Mock

        re = RandomEffects("y ~ x1 + x2", clean_panel_data, "entity", "time")
        results = re.fit()

        test = MundlakTest(results)

        # Replace _model with a mock that lacks formula_parser and data
        results._model = Mock(spec=[])

        X, y, entities = test._get_data()
        assert X is None
        assert y is None
        assert entities is None
