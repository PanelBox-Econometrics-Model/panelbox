"""
Tests for FormulaParser class.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.core.formula_parser import FormulaParser, parse_formula


class TestFormulaParserInitialization:
    """Tests for FormulaParser initialization."""

    def test_init_valid_formula(self):
        """Test initialization with valid formula."""
        parser = FormulaParser("y ~ x1 + x2")
        assert parser.formula == "y ~ x1 + x2"
        assert not parser._parsed

    def test_init_invalid_type(self):
        """Test that non-string formula raises TypeError."""
        with pytest.raises(TypeError, match="formula must be a string"):
            FormulaParser(123)

    def test_init_missing_tilde(self):
        """Test that formula without ~ raises ValueError."""
        with pytest.raises(ValueError, match="formula must contain '~'"):
            FormulaParser("y = x1 + x2")


class TestBasicParsing:
    """Tests for basic formula parsing."""

    def test_simple_formula(self):
        """Test parsing simple formula."""
        parser = FormulaParser("y ~ x1 + x2").parse()

        assert parser.dependent == "y"
        assert "x1" in parser.regressors
        assert "x2" in parser.regressors
        assert parser.has_intercept is True

    def test_formula_without_intercept(self):
        """Test formula with -1 (no intercept)."""
        parser = FormulaParser("y ~ x1 + x2 - 1").parse()

        assert parser.has_intercept is False
        assert parser.dependent == "y"

    def test_formula_with_spaces(self):
        """Test that parser handles extra spaces."""
        parser = FormulaParser("  y  ~  x1  +  x2  ").parse()

        assert parser.dependent == "y"
        assert "x1" in parser.regressors
        assert "x2" in parser.regressors

    def test_single_regressor(self):
        """Test formula with single regressor."""
        parser = FormulaParser("y ~ x1").parse()

        assert parser.dependent == "y"
        assert "x1" in parser.regressors
        assert len(parser.regressors) == 1


class TestInteractions:
    """Tests for interaction terms."""

    def test_interaction_colon(self):
        """Test interaction with : operator."""
        parser = FormulaParser("y ~ x1:x2").parse()

        assert "x1" in parser.regressors
        assert "x2" in parser.regressors

    def test_interaction_star(self):
        """Test interaction with * operator (includes main effects)."""
        parser = FormulaParser("y ~ x1 * x2").parse()

        assert "x1" in parser.regressors
        assert "x2" in parser.regressors


class TestTransformations:
    """Tests for variable transformations."""

    def test_log_transformation(self):
        """Test log transformation."""
        parser = FormulaParser("y ~ log(x1)").parse()

        assert "x1" in parser.regressors

    def test_i_transformation(self):
        """Test I() transformation."""
        parser = FormulaParser("y ~ I(x1**2)").parse()

        assert "x1" in parser.regressors

    def test_multiple_transformations(self):
        """Test multiple transformations."""
        parser = FormulaParser("y ~ log(x1) + I(x2**2)").parse()

        assert "x1" in parser.regressors
        assert "x2" in parser.regressors


class TestDesignMatrices:
    """Tests for building design matrices."""

    def test_build_simple_design_matrix(self):
        """Test building design matrix for simple formula."""
        data = pd.DataFrame(
            {"y": [1, 2, 3, 4, 5], "x1": [10, 20, 30, 40, 50], "x2": [5, 10, 15, 20, 25]}
        )

        parser = FormulaParser("y ~ x1 + x2").parse()
        y, X = parser.build_design_matrices(data, return_type="dataframe")

        assert len(y) == 5
        assert "Intercept" in X.columns
        assert "x1" in X.columns
        assert "x2" in X.columns
        assert X.shape == (5, 3)  # 5 rows, 3 columns (Intercept, x1, x2)

    def test_build_without_intercept(self):
        """Test building design matrix without intercept."""
        data = pd.DataFrame(
            {"y": [1, 2, 3, 4, 5], "x1": [10, 20, 30, 40, 50], "x2": [5, 10, 15, 20, 25]}
        )

        parser = FormulaParser("y ~ x1 + x2 - 1").parse()
        y, X = parser.build_design_matrices(data, return_type="dataframe")

        assert "Intercept" not in X.columns
        assert "x1" in X.columns
        assert "x2" in X.columns
        assert X.shape == (5, 2)  # No intercept

    def test_build_with_transformation(self):
        """Test building design matrix with transformations."""
        data = pd.DataFrame(
            {
                "y": [1, 2, 3, 4, 5],
                "x1": [1, 2, 3, 4, 5],
            }
        )

        parser = FormulaParser("y ~ I(x1**2)").parse()
        y, X = parser.build_design_matrices(data, return_type="dataframe")

        # Check that x1**2 was computed correctly
        expected = data["x1"] ** 2
        np.testing.assert_array_almost_equal(X["I(x1 ** 2)"].values, expected.values)

    def test_build_with_interaction(self):
        """Test building design matrix with interaction."""
        data = pd.DataFrame({"y": [1, 2, 3, 4, 5], "x1": [1, 2, 3, 4, 5], "x2": [2, 3, 4, 5, 6]})

        parser = FormulaParser("y ~ x1 * x2").parse()
        y, X = parser.build_design_matrices(data, return_type="dataframe")

        # Should have Intercept, x1, x2, and x1:x2
        assert "Intercept" in X.columns
        assert "x1" in X.columns
        assert "x2" in X.columns
        assert "x1:x2" in X.columns

        # Check interaction was computed correctly
        expected_interaction = data["x1"] * data["x2"]
        np.testing.assert_array_almost_equal(X["x1:x2"].values, expected_interaction.values)

    def test_build_return_types(self):
        """Test different return types."""
        data = pd.DataFrame(
            {
                "y": [1, 2, 3, 4, 5],
                "x1": [10, 20, 30, 40, 50],
            }
        )

        parser = FormulaParser("y ~ x1").parse()

        # Test dataframe return
        y_df, X_df = parser.build_design_matrices(data, return_type="dataframe")
        assert isinstance(y_df, pd.Series)
        assert isinstance(X_df, pd.DataFrame)

        # Test array return
        y_arr, X_arr = parser.build_design_matrices(data, return_type="array")
        assert isinstance(y_arr, np.ndarray)
        assert isinstance(X_arr, np.ndarray)
        assert y_arr.ndim == 1
        assert X_arr.ndim == 2

    def test_invalid_return_type(self):
        """Test that invalid return_type raises ValueError."""
        data = pd.DataFrame(
            {
                "y": [1, 2, 3],
                "x1": [10, 20, 30],
            }
        )

        parser = FormulaParser("y ~ x1").parse()

        with pytest.raises(ValueError, match="return_type must be"):
            parser.build_design_matrices(data, return_type="invalid")


class TestVariableNames:
    """Tests for getting variable names."""

    def test_get_variable_names(self):
        """Test getting variable names from design matrix."""
        data = pd.DataFrame(
            {"y": [1, 2, 3, 4, 5], "x1": [10, 20, 30, 40, 50], "x2": [5, 10, 15, 20, 25]}
        )

        parser = FormulaParser("y ~ x1 + x2").parse()
        var_names = parser.get_variable_names(data)

        assert "Intercept" in var_names
        assert "x1" in var_names
        assert "x2" in var_names
        assert len(var_names) == 3

    def test_get_variable_names_with_interaction(self):
        """Test getting variable names with interaction."""
        data = pd.DataFrame({"y": [1, 2, 3, 4, 5], "x1": [1, 2, 3, 4, 5], "x2": [2, 3, 4, 5, 6]})

        parser = FormulaParser("y ~ x1 * x2").parse()
        var_names = parser.get_variable_names(data)

        assert "Intercept" in var_names
        assert "x1" in var_names
        assert "x2" in var_names
        assert "x1:x2" in var_names


class TestConvenienceFunction:
    """Tests for parse_formula convenience function."""

    def test_parse_formula(self):
        """Test parse_formula convenience function."""
        parser = parse_formula("y ~ x1 + x2")

        assert parser._parsed is True
        assert parser.dependent == "y"
        assert "x1" in parser.regressors


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_unparsed(self):
        """Test repr for unparsed formula."""
        parser = FormulaParser("y ~ x1 + x2")
        repr_str = repr(parser)

        assert "FormulaParser" in repr_str
        assert "unparsed" in repr_str

    def test_repr_parsed(self):
        """Test repr for parsed formula."""
        parser = FormulaParser("y ~ x1 + x2").parse()
        repr_str = repr(parser)

        assert "FormulaParser" in repr_str
        assert "dependent='y'" in repr_str
        assert "y ~ x1 + x2" in repr_str
