"""
Tests for formatting utilities.
"""

import pandas as pd
import pytest

from panelbox.utils.formatting import (
    format_coefficient_table,
    format_number,
    format_pvalue,
    significance_stars,
)


class TestFormatPvalue:
    """Test format_pvalue function."""

    def test_very_small_pvalue(self):
        """Test formatting very small p-value."""
        result = format_pvalue(0.00005)
        assert result == "<0.0001"

    def test_pvalue_below_threshold(self):
        """Test p-value below threshold."""
        result = format_pvalue(0.00001)
        assert result == "<0.0001"

    def test_pvalue_at_threshold(self):
        """Test p-value at threshold."""
        result = format_pvalue(0.0001)
        assert result == "0.0001"  # At threshold, not below it

    def test_normal_pvalue(self):
        """Test normal p-value formatting."""
        result = format_pvalue(0.1234)
        assert result == "0.1234"

    def test_large_pvalue(self):
        """Test large p-value."""
        result = format_pvalue(0.9876)
        assert result == "0.9876"

    def test_custom_digits(self):
        """Test with custom number of digits."""
        result = format_pvalue(0.12345, digits=2)
        assert result == "0.12"

    def test_custom_digits_below_threshold(self):
        """Test with custom digits for small p-value."""
        result = format_pvalue(0.001, digits=2)
        assert result == "<0.01"

    def test_zero_pvalue(self):
        """Test zero p-value."""
        result = format_pvalue(0.0)
        assert result == "<0.0001"

    def test_one_pvalue(self):
        """Test p-value of 1.0."""
        result = format_pvalue(1.0)
        assert result == "1.0000"


class TestFormatNumber:
    """Test format_number function."""

    def test_format_integer(self):
        """Test formatting integer."""
        result = format_number(1000)
        assert "1,000" in result or "1000" in result  # Depends on locale
        assert len(result) >= 10  # Should be right-aligned to width 10

    def test_format_float(self):
        """Test formatting float."""
        result = format_number(1234.5678)
        assert "1234.5678" in result

    def test_format_float_with_decimals(self):
        """Test formatting float with custom decimals."""
        result = format_number(1234.5678, decimals=2)
        assert "1234.57" in result

    def test_format_negative_number(self):
        """Test formatting negative number."""
        result = format_number(-123.456)
        assert "-123.4560" in result

    def test_format_with_custom_width(self):
        """Test formatting with custom width."""
        result = format_number(123.45, width=20)
        assert len(result) >= 15  # Should be right-aligned

    def test_format_zero(self):
        """Test formatting zero."""
        result = format_number(0.0)
        assert "0.0000" in result

    def test_format_large_integer(self):
        """Test formatting large integer."""
        result = format_number(1000000)
        # Should have comma separators (locale-dependent)
        assert "1000000" in result or "1,000,000" in result


class TestSignificanceStars:
    """Test significance_stars function."""

    def test_highly_significant(self):
        """Test p < 0.001."""
        assert significance_stars(0.0001) == "***"
        assert significance_stars(0.0005) == "***"

    def test_very_significant(self):
        """Test p < 0.01."""
        assert significance_stars(0.005) == "**"
        assert significance_stars(0.009) == "**"

    def test_significant(self):
        """Test p < 0.05."""
        assert significance_stars(0.02) == "*"
        assert significance_stars(0.04) == "*"

    def test_marginally_significant(self):
        """Test p < 0.10."""
        assert significance_stars(0.06) == "."
        assert significance_stars(0.09) == "."

    def test_not_significant(self):
        """Test p >= 0.10."""
        assert significance_stars(0.10) == ""
        assert significance_stars(0.20) == ""
        assert significance_stars(0.50) == ""

    def test_boundary_values(self):
        """Test boundary values."""
        assert significance_stars(0.001) == "**"  # Just above 0.001
        assert significance_stars(0.01) == "*"  # Just above 0.01
        assert significance_stars(0.05) == "."  # Just above 0.05

    def test_exact_threshold_values(self):
        """Test exact threshold values."""
        assert significance_stars(0.0009) == "***"
        assert significance_stars(0.0099) == "**"
        assert significance_stars(0.0499) == "*"
        assert significance_stars(0.0999) == "."


class TestFormatCoefficientTable:
    """Test format_coefficient_table function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample coefficient data."""
        params = pd.Series({"x1": 1.234, "x2": -0.567, "x3": 0.123})
        std_errors = pd.Series({"x1": 0.045, "x2": 0.089, "x3": 0.056})
        tvalues = pd.Series({"x1": 27.4, "x2": -6.37, "x3": 2.20})
        pvalues = pd.Series({"x1": 0.0001, "x2": 0.012, "x3": 0.045})
        return params, std_errors, tvalues, pvalues

    def test_basic_table(self, sample_data):
        """Test basic coefficient table without confidence intervals."""
        params, std_errors, tvalues, pvalues = sample_data

        table = format_coefficient_table(params, std_errors, tvalues, pvalues)

        assert "Variable" in table
        assert "Coef." in table
        assert "Std.Err." in table
        assert "x1" in table
        assert "x2" in table
        assert "x3" in table
        assert "1.2340" in table
        assert "-0.5670" in table
        assert "***" in table  # x1 is highly significant
        assert "**" in table  # x2 is very significant
        assert "*" in table  # x3 is significant

    def test_table_with_confidence_intervals(self, sample_data):
        """Test coefficient table with confidence intervals."""
        params, std_errors, tvalues, pvalues = sample_data
        conf_int = pd.DataFrame(
            {
                "lower": {"x1": 1.145, "x2": -0.741, "x3": 0.013},
                "upper": {"x1": 1.323, "x2": -0.393, "x3": 0.233},
            }
        )

        table = format_coefficient_table(params, std_errors, tvalues, pvalues, conf_int)

        assert "[0.025" in table
        assert "0.975]" in table
        assert "1.1450" in table
        assert "1.3230" in table

    def test_table_alignment(self, sample_data):
        """Test that table has proper alignment."""
        params, std_errors, tvalues, pvalues = sample_data

        table = format_coefficient_table(params, std_errors, tvalues, pvalues)
        lines = table.split("\n")

        # Should have header, separator, and rows
        assert len(lines) >= 5  # Header + separator + 3 variables
        assert "-" * 50 in table  # Should have separator line

    def test_empty_table(self):
        """Test with empty data."""
        params = pd.Series(dtype=float)
        std_errors = pd.Series(dtype=float)
        tvalues = pd.Series(dtype=float)
        pvalues = pd.Series(dtype=float)

        table = format_coefficient_table(params, std_errors, tvalues, pvalues)

        # Should still have header
        assert "Variable" in table

    def test_single_variable(self):
        """Test with single variable."""
        params = pd.Series({"x1": 1.234})
        std_errors = pd.Series({"x1": 0.045})
        tvalues = pd.Series({"x1": 27.4})
        pvalues = pd.Series({"x1": 0.0001})

        table = format_coefficient_table(params, std_errors, tvalues, pvalues)

        assert "x1" in table
        assert "1.2340" in table

    def test_mixed_significance_levels(self):
        """Test with variables at different significance levels."""
        params = pd.Series(
            {
                "highly_sig": 1.0,
                "very_sig": 1.0,
                "sig": 1.0,
                "marginal": 1.0,
                "not_sig": 1.0,
            }
        )
        std_errors = pd.Series(
            {
                "highly_sig": 0.1,
                "very_sig": 0.1,
                "sig": 0.1,
                "marginal": 0.1,
                "not_sig": 0.1,
            }
        )
        tvalues = pd.Series(
            {
                "highly_sig": 10.0,
                "very_sig": 5.0,
                "sig": 2.5,
                "marginal": 1.8,
                "not_sig": 1.0,
            }
        )
        pvalues = pd.Series(
            {
                "highly_sig": 0.0001,
                "very_sig": 0.005,
                "sig": 0.02,
                "marginal": 0.07,
                "not_sig": 0.30,
            }
        )

        table = format_coefficient_table(params, std_errors, tvalues, pvalues)

        # Count stars
        assert table.count("***") >= 1
        assert table.count("**") >= 1
        assert table.count("*") >= 2  # Both * and **
        assert table.count(".") >= 1


class TestIntegration:
    """Integration tests for formatting functions."""

    def test_format_pvalue_with_stars(self):
        """Test combining format_pvalue and significance_stars."""
        pvalue = 0.001
        formatted = format_pvalue(pvalue)
        stars = significance_stars(pvalue)

        assert formatted == "0.0010"
        assert stars == "**"

    def test_table_formatting_workflow(self):
        """Test complete table formatting workflow."""
        # Simulate regression results
        params = pd.Series({"const": 5.123, "x1": 2.456, "x2": -1.789})
        std_errors = pd.Series({"const": 0.234, "x1": 0.123, "x2": 0.456})
        tvalues = pd.Series({"const": 21.9, "x1": 20.0, "x2": -3.92})
        pvalues = pd.Series({"const": 0.0, "x1": 0.0001, "x2": 0.001})

        table = format_coefficient_table(params, std_errors, tvalues, pvalues)

        # Verify structure
        assert isinstance(table, str)
        assert len(table) > 0
        assert "const" in table
        assert "x1" in table
        assert "x2" in table

        # Verify all highly significant
        assert table.count("***") >= 2
