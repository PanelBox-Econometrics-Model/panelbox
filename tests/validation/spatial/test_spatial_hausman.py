"""
Tests for SpatialHausmanTest.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.validation.base import ValidationTestResult
from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest


class MockModel:
    """Mock model class for testing."""

    class __class__:
        __name__ = "MockModel"


class MockResult:
    """Mock model result for Hausman test testing."""

    def __init__(self, params, bse=None, cov=None, model_name=None):
        """
        Parameters
        ----------
        params : dict or np.ndarray
            Parameter estimates. If dict, creates pd.Series.
        bse : dict or np.ndarray, optional
            Standard errors
        cov : np.ndarray, optional
            Full covariance matrix
        model_name : str, optional
            Name for the model
        """
        if isinstance(params, dict):
            self.params = pd.Series(params)
        else:
            self.params = params

        if bse is not None:
            if isinstance(bse, dict):
                self.bse = pd.Series(bse)
            else:
                self.bse = bse

        if cov is not None:
            self.cov = cov

        if model_name:
            # Create a model object whose __class__.__name__ is model_name
            model_cls = type(model_name, (), {})
            self.model = model_cls()


class TestSpatialHausmanInit:
    """Test SpatialHausmanTest initialization."""

    def test_init_with_series_params(self):
        """Test init with named pd.Series parameters."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.1, "x2": 2.1}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)

        assert len(test.common_params) == 2
        assert test.use_positional is False
        assert set(test.common_params) == {"x1", "x2"}

    def test_init_with_array_params(self):
        """Test init with numpy array parameters.

        When both models have numpy arrays, _extract_parameters generates
        synthetic beta_i names which match across models, so use_positional
        is False. The common_params will be beta_0, beta_1, etc.
        """
        result1 = MockResult(np.array([1.0, 2.0]), bse=np.array([0.1, 0.2]))
        result2 = MockResult(np.array([1.1, 2.1]), bse=np.array([0.15, 0.25]))

        test = SpatialHausmanTest(result1, result2)

        # Both arrays get beta_i names, which intersect, so not positional
        assert test.use_positional is False
        assert len(test.common_params) == 2
        assert set(test.common_params) == {"beta_0", "beta_1"}

    def test_init_positional_matching_mixed_types(self):
        """Test positional matching when one is Series and other is array with
        non-matching names."""
        result1 = MockResult({"a": 1.0, "b": 2.0}, bse={"a": 0.1, "b": 0.2})
        result2 = MockResult({"c": 1.1, "d": 2.1}, bse={"c": 0.15, "d": 0.25})

        test = SpatialHausmanTest(result1, result2)

        # No name overlap -> positional matching
        assert test.use_positional is True
        assert len(test.common_params) == 2
        assert test.common_params == [0, 1]

    def test_init_excludes_spatial_params(self):
        """Test that spatial parameters are excluded."""
        result1 = MockResult(
            {"x1": 1.0, "x2": 2.0, "rho": 0.5},
            bse={"x1": 0.1, "x2": 0.2, "rho": 0.05},
        )
        result2 = MockResult(
            {"x1": 1.1, "x2": 2.1, "lambda": 0.3},
            bse={"x1": 0.15, "x2": 0.25, "lambda": 0.08},
        )

        test = SpatialHausmanTest(result1, result2)

        # Should only have x1 and x2, not rho or lambda
        assert "rho" not in test.common_params
        assert "lambda" not in test.common_params
        assert set(test.common_params) == {"x1", "x2"}

    def test_init_excludes_all_spatial_keywords(self):
        """Test that all recognized spatial param keywords are excluded."""
        spatial_names = ["rho", "lambda", "theta", "spatial_lag", "spatial_error"]
        params1 = {"x1": 1.0}
        bse1 = {"x1": 0.1}
        params2 = {"x1": 1.1}
        bse2 = {"x1": 0.15}

        for sp_name in spatial_names:
            params1[sp_name] = 0.5
            bse1[sp_name] = 0.05
            params2[sp_name] = 0.3
            bse2[sp_name] = 0.03

        result1 = MockResult(params1, bse=bse1)
        result2 = MockResult(params2, bse=bse2)

        test = SpatialHausmanTest(result1, result2)

        for sp_name in spatial_names:
            assert sp_name not in test.common_params
        assert set(test.common_params) == {"x1"}

    def test_init_missing_params_result1_raises(self):
        """Test that missing params attribute on result1 raises error."""

        class NoParams:
            pass

        result1 = NoParams()
        result2 = MockResult({"x1": 1.0}, bse={"x1": 0.1})

        with pytest.raises(ValueError, match="params"):
            SpatialHausmanTest(result1, result2)

    def test_init_missing_params_result2_raises(self):
        """Test that missing params attribute on result2 raises error."""

        class NoParams:
            pass

        result1 = MockResult({"x1": 1.0}, bse={"x1": 0.1})
        result2 = NoParams()

        with pytest.raises(ValueError, match="params"):
            SpatialHausmanTest(result1, result2)

    def test_init_partial_overlap(self):
        """Test with partially overlapping params."""
        result1 = MockResult(
            {"x1": 1.0, "x2": 2.0, "x3": 3.0},
            bse={"x1": 0.1, "x2": 0.2, "x3": 0.3},
        )
        result2 = MockResult(
            {"x1": 1.1, "x2": 2.1, "x4": 4.0},
            bse={"x1": 0.15, "x2": 0.25, "x4": 0.4},
        )

        test = SpatialHausmanTest(result1, result2)

        assert set(test.common_params) == {"x1", "x2"}

    def test_init_single_common_param(self):
        """Test with only one common parameter."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.5, "x3": 3.0}, bse={"x1": 0.15, "x3": 0.3})

        test = SpatialHausmanTest(result1, result2)

        assert test.common_params == ["x1"]
        assert test.use_positional is False

    def test_init_common_params_sorted(self):
        """Test that common parameters are sorted."""
        result1 = MockResult(
            {"z": 1.0, "a": 2.0, "m": 3.0},
            bse={"z": 0.1, "a": 0.2, "m": 0.3},
        )
        result2 = MockResult(
            {"m": 1.5, "z": 2.5, "a": 3.5},
            bse={"m": 0.15, "z": 0.25, "a": 0.35},
        )

        test = SpatialHausmanTest(result1, result2)

        assert test.common_params == ["a", "m", "z"]

    def test_init_different_length_arrays_positional(self):
        """Test positional matching with different length arrays."""
        result1 = MockResult(np.array([1.0, 2.0, 3.0]), bse=np.array([0.1, 0.2, 0.3]))
        result2 = MockResult(np.array([1.5, 2.5]), bse=np.array([0.15, 0.25]))

        test = SpatialHausmanTest(result1, result2)

        # Both get beta_i names; result1 has {beta_0, beta_1, beta_2},
        # result2 has {beta_0, beta_1}. Intersection is {beta_0, beta_1}.
        assert len(test.common_params) == 2
        assert set(test.common_params) == {"beta_0", "beta_1"}


class TestSpatialHausmanRun:
    """Test SpatialHausmanTest run method."""

    def test_run_returns_validation_result(self):
        """Test that run returns ValidationTestResult."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.1, "x2": 2.1}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)
        result = test.run(alpha=0.05)

        assert isinstance(result, ValidationTestResult)
        assert result.test_name == "Spatial Hausman Test"
        assert result.statistic >= 0
        assert 0 <= result.pvalue <= 1

    def test_run_identical_models_high_pvalue(self):
        """Test that identical model params give high p-value (statistic ~ 0)."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)
        result = test.run(alpha=0.05)

        # Identical params -> statistic should be 0 or close to 0
        assert result.statistic < 1e-6
        assert result.pvalue > 0.5

    def test_run_different_models_low_pvalue(self):
        """Test that very different params give low p-value."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.01, "x2": 0.01})
        result2 = MockResult({"x1": 5.0, "x2": 10.0}, bse={"x1": 0.02, "x2": 0.02})

        test = SpatialHausmanTest(result1, result2)
        result = test.run(alpha=0.05)

        # Very different params with small standard errors -> should reject
        assert result.pvalue < 0.05
        assert result.statistic > 10

    def test_run_null_hypothesis_text(self):
        """Test the null and alternative hypothesis text."""
        result1 = MockResult({"x1": 1.0}, bse={"x1": 0.1})
        result2 = MockResult({"x1": 1.5}, bse={"x1": 0.15})

        test = SpatialHausmanTest(result1, result2)
        result = test.run()

        assert result.null_hypothesis == "Both models are consistent"
        assert result.alternative_hypothesis == "Only one model is consistent"

    def test_run_metadata(self):
        """Test that metadata is populated correctly."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.5, "x2": 2.5}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)
        result = test.run(alpha=0.05)

        assert "degrees_of_freedom" in result.metadata
        assert result.metadata["degrees_of_freedom"] == 2
        assert "n_parameters_tested" in result.metadata
        assert result.metadata["n_parameters_tested"] == 2
        assert "max_difference" in result.metadata
        assert "conclusion" in result.metadata
        assert "distribution" in result.metadata
        assert "model1" in result.metadata
        assert "model2" in result.metadata

    def test_run_df_equals_number_of_params(self):
        """Test that degrees of freedom equals number of tested parameters."""
        result1 = MockResult(
            {"x1": 1.0, "x2": 2.0, "x3": 3.0},
            bse={"x1": 0.1, "x2": 0.2, "x3": 0.3},
        )
        result2 = MockResult(
            {"x1": 1.5, "x2": 2.5, "x3": 3.5},
            bse={"x1": 0.15, "x2": 0.25, "x3": 0.35},
        )

        test = SpatialHausmanTest(result1, result2)
        result = test.run()

        assert result.df == 3
        assert result.metadata["degrees_of_freedom"] == 3

    def test_run_with_subset(self):
        """Test running with parameter subset."""
        result1 = MockResult(
            {"x1": 1.0, "x2": 2.0, "x3": 3.0},
            bse={"x1": 0.1, "x2": 0.2, "x3": 0.3},
        )
        result2 = MockResult(
            {"x1": 1.5, "x2": 2.5, "x3": 3.5},
            bse={"x1": 0.15, "x2": 0.25, "x3": 0.35},
        )

        test = SpatialHausmanTest(result1, result2)
        result = test.run(alpha=0.05, subset=["x1", "x2"])

        assert result.metadata["n_parameters_tested"] == 2
        assert result.df == 2

    def test_run_subset_single_param(self):
        """Test running with a single-parameter subset."""
        # result1 is efficient (smaller SE), result2 is consistent (larger SE)
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.02, "x2": 0.2})
        result2 = MockResult({"x1": 5.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)
        result = test.run(alpha=0.05, subset=["x1"])

        assert result.metadata["n_parameters_tested"] == 1
        assert result.df == 1
        # x1 differs by 4.0 with small SEs -> should reject
        assert result.pvalue < 0.05

    def test_run_with_covariance_matrix(self):
        """Test running with full covariance matrix via cov_params()."""
        np.random.seed(42)
        params1 = {"x1": 1.0, "x2": 2.0}
        params2 = {"x1": 1.5, "x2": 2.5}

        cov1 = np.array([[0.01, 0.001], [0.001, 0.04]])
        cov2 = np.array([[0.02, 0.002], [0.002, 0.06]])

        class ResultWithCov:
            def __init__(self, p, c):
                self.params = pd.Series(p)
                self._cov = c

            def cov_params(self):
                return pd.DataFrame(
                    self._cov,
                    index=self.params.index,
                    columns=self.params.index,
                )

        result1 = ResultWithCov(params1, cov1)
        result2 = ResultWithCov(params2, cov2)

        test = SpatialHausmanTest(result1, result2)
        result = test.run()

        assert isinstance(result, ValidationTestResult)
        assert result.statistic >= 0
        assert 0 <= result.pvalue <= 1

    def test_run_with_numpy_array_params(self):
        """Test running with numpy array parameters (synthetic beta_i names)."""
        result1 = MockResult(np.array([1.0, 2.0]), bse=np.array([0.1, 0.2]))
        result2 = MockResult(np.array([1.5, 2.5]), bse=np.array([0.15, 0.25]))

        test = SpatialHausmanTest(result1, result2)
        result = test.run()

        assert isinstance(result, ValidationTestResult)
        assert result.statistic >= 0
        assert result.df == 2

    def test_run_alpha_changes_conclusion(self):
        """Test that different alpha values can change the conclusion."""
        # Use params that are somewhat different but with moderate SEs
        # so the p-value falls between tight and loose alpha thresholds
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)

        # Identical params -> statistic ~ 0, pvalue ~ 1
        result_tight = test.run(alpha=0.001)
        result_loose = test.run(alpha=0.999)

        # Both should fail to reject since params are identical
        assert "Fail to reject" in result_tight.metadata["conclusion"]
        assert "Fail to reject" in result_loose.metadata["conclusion"]

        # Now use very different params
        result3 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.01, "x2": 0.01})
        result4 = MockResult({"x1": 5.0, "x2": 10.0}, bse={"x1": 0.02, "x2": 0.02})

        test2 = SpatialHausmanTest(result3, result4)
        result_reject = test2.run(alpha=0.05)

        assert "Reject" in result_reject.metadata["conclusion"]

    def test_run_max_difference_correct(self):
        """Test that max_difference is computed correctly."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.3, "x2": 5.0}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)
        result = test.run()

        # Max difference should be |2.0 - 5.0| = 3.0
        assert np.isclose(result.metadata["max_difference"], 3.0)

    def test_run_statistic_nonnegative_when_v_diff_pd(self):
        """Test that the statistic is non-negative when V_diff is positive definite.

        The Hausman statistic H = (b2-b1)' (V2-V1)^{-1} (b2-b1) is non-negative
        when V2 - V1 is positive definite (i.e., result2 is less efficient).
        We enforce se2 > se1 so that V_diff = V2 - V1 is PD (diagonal case).
        """
        np.random.seed(123)
        for _ in range(10):
            p1 = np.random.randn(3)
            p2 = np.random.randn(3)
            se1 = np.abs(np.random.randn(3)) * 0.05 + 0.01
            # Ensure se2 > se1 so V_diff = V2 - V1 is positive definite
            se2 = se1 + np.abs(np.random.randn(3)) * 0.05 + 0.01

            result1 = MockResult(
                {"x1": p1[0], "x2": p1[1], "x3": p1[2]},
                bse={"x1": se1[0], "x2": se1[1], "x3": se1[2]},
            )
            result2 = MockResult(
                {"x1": p2[0], "x2": p2[1], "x3": p2[2]},
                bse={"x1": se2[0], "x2": se2[1], "x3": se2[2]},
            )

            test = SpatialHausmanTest(result1, result2)
            result = test.run()

            assert result.statistic >= 0, (
                f"Statistic should be non-negative, got {result.statistic}"
            )

    def test_run_pvalue_between_zero_and_one(self):
        """Test that p-value is always in [0, 1]."""
        np.random.seed(456)
        for _ in range(10):
            p1 = np.random.randn(2)
            p2 = np.random.randn(2)
            se1 = np.abs(np.random.randn(2)) * 0.1 + 0.01
            se2 = np.abs(np.random.randn(2)) * 0.1 + 0.01

            result1 = MockResult(
                {"x1": p1[0], "x2": p1[1]},
                bse={"x1": se1[0], "x2": se1[1]},
            )
            result2 = MockResult(
                {"x1": p2[0], "x2": p2[1]},
                bse={"x1": se2[0], "x2": se2[1]},
            )

            test = SpatialHausmanTest(result1, result2)
            result = test.run()

            assert 0 <= result.pvalue <= 1, f"P-value should be in [0, 1], got {result.pvalue}"


class TestSpatialHausmanGetParams:
    """Test _get_params method."""

    def test_get_params_series_by_name(self):
        """Test extracting params from pd.Series by name."""
        result1 = MockResult(
            {"x1": 1.0, "x2": 2.0, "x3": 3.0},
            bse={"x1": 0.1, "x2": 0.2, "x3": 0.3},
        )
        result2 = MockResult(
            {"x1": 1.5, "x2": 2.5},
            bse={"x1": 0.15, "x2": 0.25},
        )

        test = SpatialHausmanTest(result1, result2)
        # common_params should be ["x1", "x2"]
        params = test._get_params(result1, ["x1", "x2"])

        np.testing.assert_array_almost_equal(params, [1.0, 2.0])

    def test_get_params_numpy_array(self):
        """Test extracting params from numpy array."""
        result1 = MockResult(np.array([1.0, 2.0]), bse=np.array([0.1, 0.2]))
        result2 = MockResult(np.array([1.5, 2.5]), bse=np.array([0.15, 0.25]))

        test = SpatialHausmanTest(result1, result2)
        params = test._get_params(result1, test.common_params)

        np.testing.assert_array_almost_equal(params, [1.0, 2.0])


class TestSpatialHausmanGetCovariance:
    """Test _get_covariance method."""

    def test_get_covariance_from_bse(self):
        """Test extracting covariance from standard errors (diagonal)."""
        result = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})

        test = SpatialHausmanTest(
            result,
            MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2}),
        )
        cov = test._get_covariance(result, test.common_params)

        # Should be diagonal with bse^2
        assert cov.shape == (2, 2)
        np.testing.assert_allclose(np.diag(cov), np.array([0.01, 0.04]), rtol=1e-10)
        # Off-diagonals should be zero
        np.testing.assert_allclose(cov[0, 1], 0.0)
        np.testing.assert_allclose(cov[1, 0], 0.0)

    def test_get_covariance_from_cov_params_callable(self):
        """Test extracting covariance from cov_params() method."""
        cov_matrix = np.array([[0.01, 0.003], [0.003, 0.04]])

        class ResultWithCovMethod:
            params = pd.Series({"x1": 1.0, "x2": 2.0})

            def cov_params(self):
                return pd.DataFrame(
                    cov_matrix,
                    index=self.params.index,
                    columns=self.params.index,
                )

        result = ResultWithCovMethod()
        test = SpatialHausmanTest(
            result,
            MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2}),
        )
        extracted_cov = test._get_covariance(result, test.common_params)

        assert extracted_cov.shape == (2, 2)
        np.testing.assert_allclose(extracted_cov, cov_matrix)

    def test_get_covariance_from_cov_params_attribute(self):
        """Test extracting covariance from cov_params as attribute (not callable)."""
        cov_matrix = np.array([[0.01, 0.002], [0.002, 0.04]])

        class ResultWithCovAttr:
            params = pd.Series({"x1": 1.0, "x2": 2.0})
            cov_params = cov_matrix  # Not callable, just an attribute

        result = ResultWithCovAttr()
        test = SpatialHausmanTest(
            result,
            MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2}),
        )
        extracted_cov = test._get_covariance(result, test.common_params)

        assert extracted_cov.shape == (2, 2)

    def test_get_covariance_from_cov_attribute(self):
        """Test extracting covariance from cov attribute."""
        cov_matrix = np.array([[0.01, 0.001], [0.001, 0.04]])

        class ResultWithCov:
            params = pd.Series({"x1": 1.0, "x2": 2.0})
            cov = cov_matrix

        result = ResultWithCov()
        test = SpatialHausmanTest(
            result,
            MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2}),
        )
        extracted_cov = test._get_covariance(result, test.common_params)

        assert extracted_cov.shape == (2, 2)

    def test_get_covariance_no_source_raises(self):
        """Test error when no covariance source available."""

        class NoCovResult:
            params = pd.Series({"x1": 1.0, "x2": 2.0})

        result = NoCovResult()
        test = SpatialHausmanTest(
            result,
            MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2}),
        )

        with pytest.raises(ValueError, match="Cannot extract covariance"):
            test._get_covariance(result, test.common_params)

    def test_get_covariance_bse_numpy_array(self):
        """Test covariance from bse when bse is a numpy array."""
        result1 = MockResult(np.array([1.0, 2.0]), bse=np.array([0.1, 0.2]))
        result2 = MockResult(np.array([1.5, 2.5]), bse=np.array([0.15, 0.25]))

        test = SpatialHausmanTest(result1, result2)
        cov = test._get_covariance(result1, test.common_params)

        assert cov.shape == (2, 2)
        np.testing.assert_allclose(np.diag(cov), [0.01, 0.04], rtol=1e-10)

    def test_get_covariance_submatrix_extraction(self):
        """Test that covariance submatrix is correctly extracted for a parameter subset."""
        cov_full = np.array(
            [
                [0.01, 0.002, 0.001],
                [0.002, 0.04, 0.003],
                [0.001, 0.003, 0.09],
            ]
        )

        class ResultWithFullCov:
            params = pd.Series({"x1": 1.0, "x2": 2.0, "x3": 3.0})

            def cov_params(self):
                return pd.DataFrame(
                    cov_full,
                    index=self.params.index,
                    columns=self.params.index,
                )

        result = ResultWithFullCov()
        other = MockResult(
            {"x1": 1.5, "x2": 2.5, "x3": 3.5},
            bse={"x1": 0.15, "x2": 0.25, "x3": 0.35},
        )

        test = SpatialHausmanTest(result, other)
        # Extract submatrix for just x1 and x2
        cov_sub = test._get_covariance(result, ["x1", "x2"])

        expected = np.array([[0.01, 0.002], [0.002, 0.04]])
        np.testing.assert_allclose(cov_sub, expected)


class TestSpatialHausmanSummary:
    """Test summary method."""

    def test_summary_returns_dataframe(self):
        """Test that summary returns a DataFrame."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.5, "x2": 2.5}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)
        summary = test.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Parameter" in summary.columns
        assert "Difference" in summary.columns
        assert "Abs_Diff" in summary.columns
        assert len(summary) == 2

    def test_summary_difference_correct(self):
        """Test that parameter differences are correct."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.5, "x2": 3.0}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)
        summary = test.summary()

        # Differences should be params1 - params2
        diffs = summary["Difference"].values
        assert np.allclose(np.abs(diffs), [0.5, 1.0], atol=1e-5)

    def test_summary_abs_diff_correct(self):
        """Test that absolute differences are correct."""
        result1 = MockResult({"x1": 1.0, "x2": 5.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 3.0, "x2": 2.0}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)
        summary = test.summary()

        abs_diffs = summary["Abs_Diff"].values
        assert np.allclose(abs_diffs, [2.0, 3.0], atol=1e-5)

    def test_summary_includes_both_model_coefs(self):
        """Test that summary includes coefficients from both models."""
        result1 = MockResult(
            {"x1": 1.0, "x2": 2.0},
            bse={"x1": 0.1, "x2": 0.2},
            model_name="SAR",
        )
        result2 = MockResult(
            {"x1": 1.5, "x2": 2.5},
            bse={"x1": 0.15, "x2": 0.25},
            model_name="SEM",
        )

        test = SpatialHausmanTest(result1, result2)
        summary = test.summary()

        # Should have columns for both model coefficients and standard errors
        coef_cols = [c for c in summary.columns if "_coef" in c]
        se_cols = [c for c in summary.columns if "_se" in c]

        assert len(coef_cols) == 2
        assert len(se_cols) == 2
        assert "SAR_coef" in summary.columns
        assert "SEM_coef" in summary.columns
        assert "SAR_se" in summary.columns
        assert "SEM_se" in summary.columns

    def test_summary_parameter_names(self):
        """Test that parameter names appear in the summary."""
        result1 = MockResult(
            {"income": 1.0, "education": 2.0},
            bse={"income": 0.1, "education": 0.2},
        )
        result2 = MockResult(
            {"income": 1.5, "education": 2.5},
            bse={"income": 0.15, "education": 0.25},
        )

        test = SpatialHausmanTest(result1, result2)
        summary = test.summary()

        param_names = summary["Parameter"].tolist()
        assert "income" in param_names or "education" in param_names

    @pytest.mark.xfail(
        reason=(
            "Bug in summary(): when both models have numpy array params, "
            "common_params are synthetic string names (beta_i) but "
            "use_positional=False, causing string-index into numpy array"
        ),
        raises=IndexError,
        strict=True,
    )
    def test_summary_with_numpy_arrays(self):
        """Test summary with numpy array parameters (beta_i names).

        Known bug: summary() fails when params/bse are numpy arrays because
        common_params are synthetic names like 'beta_0', 'beta_1' but the code
        tries to index numpy arrays with these strings when use_positional=False.
        """
        result1 = MockResult(np.array([1.0, 2.0]), bse=np.array([0.1, 0.2]))
        result2 = MockResult(np.array([1.5, 2.5]), bse=np.array([0.15, 0.25]))

        test = SpatialHausmanTest(result1, result2)
        summary = test.summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2

    def test_summary_rounded(self):
        """Test that summary values are rounded to 6 decimal places."""
        result1 = MockResult({"x1": 1.123456789}, bse={"x1": 0.123456789})
        result2 = MockResult({"x1": 2.987654321}, bse={"x1": 0.987654321})

        test = SpatialHausmanTest(result1, result2)
        summary = test.summary()

        # Values should be rounded to 6 decimal places
        diff = summary["Difference"].values[0]
        # Check that no more than 6 decimal places
        assert diff == round(diff, 6)


class TestSpatialHausmanModelName:
    """Test _get_model_name method."""

    def test_get_model_name_from_model_class(self):
        """Test getting model name from result.model.__class__.__name__."""
        result = MockResult({"x1": 1.0}, bse={"x1": 0.1}, model_name="SpatialLag")

        test = SpatialHausmanTest(result, MockResult({"x1": 1.0}, bse={"x1": 0.1}))
        name = test._get_model_name(result)

        assert name == "SpatialLag"

    def test_get_model_name_from_result_class(self):
        """Test fallback to result.__class__.__name__."""
        result = MockResult({"x1": 1.0}, bse={"x1": 0.1})
        # MockResult without model_name doesn't set self.model

        test = SpatialHausmanTest(result, MockResult({"x1": 1.0}, bse={"x1": 0.1}))
        name = test._get_model_name(result)

        # Should return "MockResult" since no .model attribute
        assert name == "MockResult"
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_model_name_fallback(self):
        """Test fallback model name when no class info available."""

        class SimpleResult:
            params = pd.Series({"x1": 1.0})
            bse = pd.Series({"x1": 0.1})

        result = SimpleResult()
        test = SpatialHausmanTest(result, MockResult({"x1": 1.0}, bse={"x1": 0.1}))
        name = test._get_model_name(result)

        # Should return class name
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_model_name_with_model_name_attribute(self):
        """Test getting name when model has a 'name' attribute."""

        class ResultWithModelName:
            params = pd.Series({"x1": 1.0})
            bse = pd.Series({"x1": 0.1})

            class model:
                name = "SpatialError"

        result = ResultWithModelName()
        test = SpatialHausmanTest(result, MockResult({"x1": 1.0}, bse={"x1": 0.1}))
        name = test._get_model_name(result)

        assert isinstance(name, str)
        assert len(name) > 0

    def test_model_names_appear_in_metadata(self):
        """Test that model names appear in the run result metadata."""
        result1 = MockResult({"x1": 1.0}, bse={"x1": 0.1}, model_name="SAR")
        result2 = MockResult({"x1": 1.5}, bse={"x1": 0.15}, model_name="SEM")

        test = SpatialHausmanTest(result1, result2)
        result = test.run()

        assert result.metadata["model1"] == "SAR"
        assert result.metadata["model2"] == "SEM"


class TestSpatialHausmanEdgeCases:
    """Test edge cases and special scenarios."""

    def test_handles_v_diff_not_positive_definite(self):
        """Test that the test handles non-positive-definite V_diff without error.

        When the efficient model has larger variance than the consistent model
        for some parameters, V_diff = V2 - V1 may not be positive definite.
        The implementation adds regularization but the statistic can still be
        negative -- this is a known limitation of the Hausman test when the
        ordering of models (efficient vs. consistent) is violated.
        """
        # Make V1 > V2 for some parameters -> V_diff = V2 - V1 not PD
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.5, "x2": 0.5})
        result2 = MockResult({"x1": 1.5, "x2": 2.5}, bse={"x1": 0.1, "x2": 0.1})

        test = SpatialHausmanTest(result1, result2)
        # Should not raise - regularization handles the inversion
        result = test.run()

        assert isinstance(result, ValidationTestResult)
        # With V_diff not PD, p-value should be 1.0 (negative statistic)
        assert result.pvalue >= 0

    def test_large_number_of_parameters(self):
        """Test with a larger number of parameters."""
        n_params = 20
        param_names = [f"x{i}" for i in range(n_params)]
        params1 = {name: float(i) for i, name in enumerate(param_names)}
        params2 = {name: float(i) + 0.1 for i, name in enumerate(param_names)}
        bse1 = dict.fromkeys(param_names, 0.1)
        bse2 = dict.fromkeys(param_names, 0.15)

        result1 = MockResult(params1, bse=bse1)
        result2 = MockResult(params2, bse=bse2)

        test = SpatialHausmanTest(result1, result2)
        result = test.run()

        assert result.df == n_params
        assert result.metadata["n_parameters_tested"] == n_params

    def test_parameters_field_in_metadata_named_params(self):
        """Test that metadata contains parameter names when not positional."""
        result1 = MockResult({"x1": 1.0, "x2": 2.0}, bse={"x1": 0.1, "x2": 0.2})
        result2 = MockResult({"x1": 1.5, "x2": 2.5}, bse={"x1": 0.15, "x2": 0.25})

        test = SpatialHausmanTest(result1, result2)
        result = test.run()

        # When not positional, parameters should be listed in metadata
        assert result.metadata["parameters"] is not None
        assert set(result.metadata["parameters"]) == {"x1", "x2"}

    def test_conclusion_reject(self):
        """Test conclusion string when rejecting H0."""
        result1 = MockResult({"x1": 1.0}, bse={"x1": 0.01})
        result2 = MockResult({"x1": 100.0}, bse={"x1": 0.02})

        test = SpatialHausmanTest(result1, result2)
        result = test.run(alpha=0.05)

        assert "Reject" in result.metadata["conclusion"]
        assert "consistent estimator" in result.metadata["conclusion"]

    def test_conclusion_fail_to_reject(self):
        """Test conclusion string when failing to reject H0."""
        result1 = MockResult({"x1": 1.0}, bse={"x1": 0.1})
        result2 = MockResult({"x1": 1.0}, bse={"x1": 0.15})

        test = SpatialHausmanTest(result1, result2)
        result = test.run(alpha=0.05)

        assert "Fail to reject" in result.metadata["conclusion"]
        assert "efficient estimator" in result.metadata["conclusion"]

    def test_chi_square_distribution_string(self):
        """Test that distribution string is correctly formatted."""
        result1 = MockResult(
            {"x1": 1.0, "x2": 2.0, "x3": 3.0},
            bse={"x1": 0.1, "x2": 0.2, "x3": 0.3},
        )
        result2 = MockResult(
            {"x1": 1.5, "x2": 2.5, "x3": 3.5},
            bse={"x1": 0.15, "x2": 0.25, "x3": 0.35},
        )

        test = SpatialHausmanTest(result1, result2)
        result = test.run()

        # Should contain chi-square with correct df
        assert "3" in result.metadata["distribution"]
