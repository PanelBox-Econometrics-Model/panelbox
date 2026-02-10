"""
Tests for visualization custom exceptions.
"""

import pytest

from panelbox.visualization.exceptions import (
    ChartNotFoundError,
    DataTransformError,
    ExportError,
    InvalidDataStructureError,
    InvalidThemeError,
    MissingDataError,
    PerformanceWarning,
    VisualizationError,
    did_you_mean,
    raise_if_missing_data,
    validate_data_structure,
)


class TestVisualizationError:
    """Test base VisualizationError class."""

    def test_init_message_only(self):
        """Test initialization with message only."""
        error = VisualizationError("Test error message")
        assert "Test error message" in str(error)

    def test_init_with_suggestion(self):
        """Test initialization with suggestion."""
        error = VisualizationError("Test error", suggestion="Try this instead")
        error_str = str(error)
        assert "Test error" in error_str
        assert "Suggestion" in error_str
        assert "Try this instead" in error_str

    def test_has_message_attribute(self):
        """Test that error has message attribute."""
        error = VisualizationError("Test message")
        assert error.message == "Test message"

    def test_has_suggestion_attribute(self):
        """Test that error has suggestion attribute."""
        error = VisualizationError("Test", suggestion="Suggestion text")
        assert error.suggestion == "Suggestion text"

    def test_suggestion_none(self):
        """Test that suggestion can be None."""
        error = VisualizationError("Test")
        assert error.suggestion is None

    def test_inherits_from_exception(self):
        """Test that VisualizationError inherits from Exception."""
        assert issubclass(VisualizationError, Exception)

    def test_can_be_raised_and_caught(self):
        """Test that error can be raised and caught."""
        with pytest.raises(VisualizationError) as exc_info:
            raise VisualizationError("Test error")
        assert "Test error" in str(exc_info.value)


class TestChartNotFoundError:
    """Test ChartNotFoundError class."""

    def test_init_chart_type_only(self):
        """Test initialization with chart type only."""
        error = ChartNotFoundError("bar_chart")
        error_str = str(error)
        assert "bar_chart" in error_str
        assert "not found" in error_str.lower()

    def test_init_with_available_charts(self):
        """Test initialization with available charts list."""
        available = ["line_chart", "bar_chart", "scatter_plot"]
        error = ChartNotFoundError("pie_chart", available_charts=available)
        error_str = str(error)
        assert "line_chart" in error_str
        assert "bar_chart" in error_str
        assert "scatter_plot" in error_str

    def test_truncates_long_list(self):
        """Test that long lists are truncated."""
        available = [f"chart_{i}" for i in range(20)]
        error = ChartNotFoundError("test", available_charts=available)
        error_str = str(error)
        assert "and 10 more" in error_str

    def test_suggestion_with_no_charts(self):
        """Test suggestion when no charts provided."""
        error = ChartNotFoundError("test")
        error_str = str(error)
        assert "ChartRegistry.list_charts()" in error_str

    def test_inherits_from_visualization_error(self):
        """Test that it inherits from VisualizationError."""
        assert issubclass(ChartNotFoundError, VisualizationError)


class TestMissingDataError:
    """Test MissingDataError class."""

    def test_init_single_key(self):
        """Test initialization with single missing key."""
        error = MissingDataError(["x_data"], "line_chart")
        error_str = str(error)
        assert "x_data" in error_str
        assert "line_chart" in error_str

    def test_init_multiple_keys(self):
        """Test initialization with multiple missing keys."""
        error = MissingDataError(["x_data", "y_data"], "scatter")
        error_str = str(error)
        assert "x_data" in error_str
        assert "y_data" in error_str

    def test_suggestion_includes_chart_type(self):
        """Test that suggestion includes chart type."""
        error = MissingDataError(["data"], "test_chart")
        error_str = str(error)
        assert "test_chart" in error_str

    def test_inherits_from_visualization_error(self):
        """Test inheritance."""
        assert issubclass(MissingDataError, VisualizationError)


class TestInvalidThemeError:
    """Test InvalidThemeError class."""

    def test_init_theme_name_only(self):
        """Test initialization with theme name only."""
        error = InvalidThemeError("custom_theme")
        error_str = str(error)
        assert "custom_theme" in error_str
        assert "Invalid theme" in error_str

    def test_init_with_available_themes(self):
        """Test initialization with available themes."""
        available = ["professional", "academic", "presentation"]
        error = InvalidThemeError("custom", available_themes=available)
        error_str = str(error)
        assert "professional" in error_str
        assert "academic" in error_str
        assert "presentation" in error_str

    def test_default_suggestion(self):
        """Test default suggestion without available themes."""
        error = InvalidThemeError("test")
        error_str = str(error)
        assert "professional" in error_str
        assert "academic" in error_str

    def test_inherits_from_visualization_error(self):
        """Test inheritance."""
        assert issubclass(InvalidThemeError, VisualizationError)


class TestDataTransformError:
    """Test DataTransformError class."""

    def test_init_basic(self):
        """Test basic initialization."""
        error = DataTransformError("PanelTransformer", "Missing column")
        error_str = str(error)
        assert "PanelTransformer" in error_str
        assert "Missing column" in error_str

    def test_suggestion_mentions_common_causes(self):
        """Test that suggestion mentions common causes."""
        error = DataTransformError("test", "reason")
        error_str = str(error)
        assert "PanelResults" in error_str or "DataFrame" in error_str

    def test_inherits_from_visualization_error(self):
        """Test inheritance."""
        assert issubclass(DataTransformError, VisualizationError)


class TestExportError:
    """Test ExportError class."""

    def test_init_basic(self):
        """Test basic initialization."""
        error = ExportError("png", "kaleido not found")
        error_str = str(error)
        assert "png" in error_str.lower()
        assert "kaleido not found" in error_str

    def test_png_suggestion(self):
        """Test PNG-specific suggestion."""
        error = ExportError("png", "test")
        error_str = str(error)
        assert "kaleido" in error_str.lower()

    def test_svg_suggestion(self):
        """Test SVG-specific suggestion."""
        error = ExportError("svg", "test")
        error_str = str(error)
        assert "kaleido" in error_str.lower()

    def test_pdf_suggestion(self):
        """Test PDF-specific suggestion."""
        error = ExportError("pdf", "test")
        error_str = str(error)
        assert "kaleido" in error_str.lower()

    def test_html_suggestion(self):
        """Test HTML-specific suggestion."""
        error = ExportError("html", "test")
        error_str = str(error)
        assert "without additional dependencies" in error_str

    def test_json_suggestion(self):
        """Test JSON-specific suggestion."""
        error = ExportError("json", "test")
        error_str = str(error)
        assert "without additional dependencies" in error_str

    def test_unknown_format_suggestion(self):
        """Test suggestion for unknown format."""
        error = ExportError("xyz", "test")
        error_str = str(error)
        assert "dependencies" in error_str.lower()

    def test_inherits_from_visualization_error(self):
        """Test inheritance."""
        assert issubclass(ExportError, VisualizationError)


class TestInvalidDataStructureError:
    """Test InvalidDataStructureError class."""

    def test_init_basic(self):
        """Test basic initialization."""
        error = InvalidDataStructureError("DataFrame", "dict")
        error_str = str(error)
        assert "DataFrame" in error_str
        assert "dict" in error_str

    def test_suggestion_mentions_common_causes(self):
        """Test that suggestion lists common causes."""
        error = InvalidDataStructureError("expected", "received")
        error_str = str(error)
        assert "MultiIndex" in error_str
        assert "fit()" in error_str

    def test_inherits_from_visualization_error(self):
        """Test inheritance."""
        assert issubclass(InvalidDataStructureError, VisualizationError)


class TestThemeLoadError:
    """Test ThemeLoadError class."""

    def test_init_basic(self):
        """Test basic initialization."""
        from panelbox.visualization.exceptions import ThemeLoadError

        error = ThemeLoadError("/path/to/theme.yaml", "File not found")
        error_str = str(error)
        assert "/path/to/theme.yaml" in error_str
        assert "File not found" in error_str

    def test_suggestion_mentions_yaml_json(self):
        """Test that suggestion mentions YAML/JSON."""
        from panelbox.visualization.exceptions import ThemeLoadError

        error = ThemeLoadError("test.yaml", "reason")
        error_str = str(error)
        assert "YAML" in error_str or "JSON" in error_str

    def test_suggestion_lists_required_fields(self):
        """Test that suggestion lists required fields."""
        from panelbox.visualization.exceptions import ThemeLoadError

        error = ThemeLoadError("test.yaml", "reason")
        error_str = str(error)
        assert "name" in error_str
        assert "colors" in error_str
        assert "font_family" in error_str


class TestPerformanceWarning:
    """Test PerformanceWarning class."""

    def test_inherits_from_user_warning(self):
        """Test that it inherits from UserWarning."""
        assert issubclass(PerformanceWarning, UserWarning)

    def test_can_be_issued(self):
        """Test that warning can be issued."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.warn("Performance issue", PerformanceWarning)
            assert len(w) == 1
            assert issubclass(w[0].category, PerformanceWarning)


class TestRaiseIfMissingData:
    """Test raise_if_missing_data helper function."""

    def test_no_missing_data(self):
        """Test that no error is raised when all data present."""
        data = {"x": [1, 2, 3], "y": [4, 5, 6]}
        required = ["x", "y"]

        # Should not raise
        raise_if_missing_data(data, required, "test_chart")

    def test_missing_single_key(self):
        """Test that error is raised for missing key."""
        data = {"x": [1, 2, 3]}
        required = ["x", "y"]

        with pytest.raises(MissingDataError) as exc_info:
            raise_if_missing_data(data, required, "test_chart")

        assert "y" in str(exc_info.value)

    def test_missing_multiple_keys(self):
        """Test that error is raised for multiple missing keys."""
        data = {"x": [1, 2, 3]}
        required = ["x", "y", "z"]

        with pytest.raises(MissingDataError) as exc_info:
            raise_if_missing_data(data, required, "test_chart")

        error_str = str(exc_info.value)
        assert "y" in error_str
        assert "z" in error_str

    def test_none_value_treated_as_missing(self):
        """Test that None values are treated as missing."""
        data = {"x": [1, 2, 3], "y": None}
        required = ["x", "y"]

        with pytest.raises(MissingDataError):
            raise_if_missing_data(data, required, "test_chart")


class TestValidateDataStructure:
    """Test validate_data_structure helper function."""

    def test_valid_type(self):
        """Test that no error is raised for valid type."""
        import pandas as pd

        data = pd.DataFrame()
        validate_data_structure(data, pd.DataFrame)

    def test_invalid_type(self):
        """Test that error is raised for invalid type."""
        import pandas as pd

        data = {"key": "value"}

        with pytest.raises(InvalidDataStructureError):
            validate_data_structure(data, pd.DataFrame)

    def test_error_message_includes_types(self):
        """Test that error message includes expected and received types."""
        import pandas as pd

        data = "string"

        with pytest.raises(InvalidDataStructureError) as exc_info:
            validate_data_structure(data, pd.DataFrame)

        error_str = str(exc_info.value)
        assert "DataFrame" in error_str
        assert "str" in error_str

    def test_custom_param_name(self):
        """Test with custom parameter name."""
        data = 123

        with pytest.raises(InvalidDataStructureError):
            validate_data_structure(data, str, param_name="text_input")


class TestDidYouMean:
    """Test did_you_mean helper function."""

    def test_exact_match_distance_zero(self):
        """Test with exact match."""
        result = did_you_mean("test", ["test", "best", "rest"])
        assert result == "test"

    def test_close_match(self):
        """Test with close match."""
        result = did_you_mean("tset", ["test", "best", "rest"])
        assert result == "test"

    def test_no_match_beyond_threshold(self):
        """Test that None returned when distance exceeds threshold."""
        result = did_you_mean("xyz", ["abc", "def"], threshold=1)
        assert result is None

    def test_finds_closest_match(self):
        """Test that closest match is returned."""
        result = did_you_mean("profesional", ["professional", "presentation"])
        assert result == "professional"

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        result = did_you_mean("TEST", ["test", "best"])
        assert result == "test"

    def test_returns_none_for_empty_list(self):
        """Test that None returned for empty valid names list."""
        result = did_you_mean("test", [])
        assert result is None

    def test_custom_threshold(self):
        """Test with custom threshold."""
        # With threshold=1, "test" and "best" are within threshold
        result = did_you_mean("test", ["best"], threshold=1)
        assert result == "best"

        # With threshold=0, only exact matches
        result = did_you_mean("test", ["best"], threshold=0)
        assert result is None

    def test_levenshtein_distance_symmetric(self):
        """Test that Levenshtein distance is symmetric."""
        result1 = did_you_mean("abc", ["xyz"])
        result2 = did_you_mean("xyz", ["abc"])
        # Both should be None if distance too large, or both should match
        assert (result1 is None and result2 is None) or (
            result1 is not None and result2 is not None
        )

    def test_empty_string(self):
        """Test with empty string."""
        result = did_you_mean("", ["a", "ab", "abc"])
        # Empty string distance to "a" is 1
        assert result in ["a", None]


class TestIntegration:
    """Integration tests combining multiple exceptions."""

    def test_exception_chain(self):
        """Test raising different exceptions in a workflow."""
        # Simulate a workflow that might raise different errors

        # 1. Chart not found
        with pytest.raises(ChartNotFoundError):
            raise ChartNotFoundError("invalid_chart", ["valid_chart"])

        # 2. Missing data
        with pytest.raises(MissingDataError):
            raise_if_missing_data({"x": [1, 2]}, ["x", "y"], "chart")

        # 3. Invalid theme
        with pytest.raises(InvalidThemeError):
            raise InvalidThemeError("bad_theme")

    def test_error_messages_are_helpful(self):
        """Test that all error messages contain helpful information."""
        errors = [
            ChartNotFoundError("test", ["valid"]),
            MissingDataError(["key"], "chart"),
            InvalidThemeError("theme"),
            DataTransformError("transformer", "reason"),
            ExportError("png", "reason"),
            InvalidDataStructureError("expected", "received"),
        ]

        for error in errors:
            error_str = str(error)
            # All should have a suggestion
            assert "Suggestion" in error_str or "suggestion" in error_str.lower()
            # All should have emoji or clear formatting
            assert len(error_str) > 10
