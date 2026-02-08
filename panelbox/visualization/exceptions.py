"""
Custom exceptions for PanelBox visualization system.

This module provides specialized exceptions with helpful error messages
and suggestions for common visualization errors.
"""

from typing import List, Optional


class VisualizationError(Exception):
    """Base exception for all visualization-related errors."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        """
        Initialize visualization error.

        Parameters
        ----------
        message : str
            Error message
        suggestion : str, optional
            Helpful suggestion for fixing the error
        """
        self.message = message
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format error message with suggestion."""
        msg = f"❌ {self.message}"
        if self.suggestion:
            msg += f"\n\n💡 Suggestion: {self.suggestion}"
        return msg


class ChartNotFoundError(VisualizationError):
    """Raised when a chart type is not found in the registry."""

    def __init__(self, chart_type: str, available_charts: Optional[List[str]] = None):
        """
        Initialize chart not found error.

        Parameters
        ----------
        chart_type : str
            The chart type that was not found
        available_charts : list of str, optional
            List of available chart types
        """
        message = f"Chart type '{chart_type}' not found in registry."

        if available_charts:
            suggestion = "Available chart types:\n" + "\n".join(
                f"  • {ct}" for ct in sorted(available_charts)[:10]
            )
            if len(available_charts) > 10:
                suggestion += f"\n  ... and {len(available_charts) - 10} more"
        else:
            suggestion = "Use ChartRegistry.list_charts() to see available types."

        super().__init__(message, suggestion)


class MissingDataError(VisualizationError):
    """Raised when required data is missing for chart creation."""

    def __init__(self, missing_keys: List[str], chart_type: str):
        """
        Initialize missing data error.

        Parameters
        ----------
        missing_keys : list of str
            List of missing required keys
        chart_type : str
            The chart type being created
        """
        message = f"Missing required data for '{chart_type}' chart: " + ", ".join(
            f"'{k}'" for k in missing_keys
        )

        suggestion = (
            f"The '{chart_type}' chart requires these data fields. "
            "Please ensure your data dict includes all required keys."
        )

        super().__init__(message, suggestion)


class InvalidThemeError(VisualizationError):
    """Raised when an invalid theme is specified."""

    def __init__(self, theme_name: str, available_themes: Optional[List[str]] = None):
        """
        Initialize invalid theme error.

        Parameters
        ----------
        theme_name : str
            The invalid theme name
        available_themes : list of str, optional
            List of available theme names
        """
        message = f"Invalid theme: '{theme_name}'"

        if available_themes:
            suggestion = (
                f"Available themes: {', '.join(available_themes)}\n"
                "You can also pass a Theme object directly or load from YAML/JSON."
            )
        else:
            suggestion = (
                "Available themes: 'professional', 'academic', 'presentation'\n"
                "You can also pass a Theme object directly."
            )

        super().__init__(message, suggestion)


class DataTransformError(VisualizationError):
    """Raised when data transformation fails."""

    def __init__(self, transformer_name: str, reason: str):
        """
        Initialize data transform error.

        Parameters
        ----------
        transformer_name : str
            Name of the transformer that failed
        reason : str
            Reason for the failure
        """
        message = f"Data transformation failed in {transformer_name}: {reason}"

        suggestion = (
            "Please check that your input data matches the expected format. "
            "For PanelResults objects, ensure the model has been fitted. "
            "For DataFrames, ensure proper MultiIndex structure (entity, time)."
        )

        super().__init__(message, suggestion)


class ExportError(VisualizationError):
    """Raised when chart export fails."""

    def __init__(self, export_format: str, reason: str):
        """
        Initialize export error.

        Parameters
        ----------
        export_format : str
            The export format that failed
        reason : str
            Reason for the failure
        """
        message = f"Failed to export chart to {export_format}: {reason}"

        suggestions_by_format = {
            "png": "PNG export requires kaleido package: pip install kaleido",
            "svg": "SVG export requires kaleido package: pip install kaleido",
            "pdf": "PDF export requires kaleido package: pip install kaleido",
            "html": "HTML export should work without additional dependencies.",
            "json": "JSON export should work without additional dependencies.",
        }

        suggestion = suggestions_by_format.get(
            export_format.lower(), "Check that all required dependencies are installed."
        )

        super().__init__(message, suggestion)


class InvalidDataStructureError(VisualizationError):
    """Raised when data structure is invalid for the operation."""

    def __init__(self, expected: str, received: str):
        """
        Initialize invalid data structure error.

        Parameters
        ----------
        expected : str
            Description of expected data structure
        received : str
            Description of received data structure
        """
        message = f"Invalid data structure. Expected: {expected}, Received: {received}"

        suggestion = (
            "Common causes:\n"
            "  • DataFrame missing MultiIndex for panel data\n"
            "  • Missing required columns\n"
            "  • Wrong data type (expected DataFrame, got dict or vice versa)\n"
            "  • Results object not fitted (call .fit() first)"
        )

        super().__init__(message, suggestion)


class ThemeLoadError(VisualizationError):
    """Raised when theme loading from file fails."""

    def __init__(self, file_path: str, reason: str):
        """
        Initialize theme load error.

        Parameters
        ----------
        file_path : str
            Path to theme file that failed to load
        reason : str
            Reason for the failure
        """
        message = f"Failed to load theme from '{file_path}': {reason}"

        suggestion = (
            "Theme files should be valid YAML or JSON with required fields:\n"
            "  • name: str\n"
            "  • colors: list of hex color codes\n"
            "  • font_family: str\n"
            "  • font_size: int\n"
            "See documentation for complete theme schema."
        )

        super().__init__(message, suggestion)


class PerformanceWarning(UserWarning):
    """Warning for performance-related issues."""

    pass


def raise_if_missing_data(data: dict, required_keys: List[str], chart_type: str):
    """
    Raise MissingDataError if required keys are missing from data.

    Parameters
    ----------
    data : dict
        Data dictionary to check
    required_keys : list of str
        List of required keys
    chart_type : str
        Name of the chart type

    Raises
    ------
    MissingDataError
        If any required keys are missing
    """
    missing = [k for k in required_keys if k not in data or data[k] is None]
    if missing:
        raise MissingDataError(missing, chart_type)


def validate_data_structure(data, expected_type: type, param_name: str = "data") -> None:
    """
    Validate that data matches expected type.

    Parameters
    ----------
    data : any
        Data to validate
    expected_type : type
        Expected type
    param_name : str, default 'data'
        Parameter name for error message

    Raises
    ------
    InvalidDataStructureError
        If data type doesn't match expected
    """
    if not isinstance(data, expected_type):
        raise InvalidDataStructureError(expected=str(expected_type), received=str(type(data)))


# Helper function for better error messages
def did_you_mean(invalid_name: str, valid_names: List[str], threshold: int = 3) -> Optional[str]:
    """
    Suggest a valid name similar to the invalid one.

    Parameters
    ----------
    invalid_name : str
        The invalid name provided
    valid_names : list of str
        List of valid names
    threshold : int, default 3
        Maximum Levenshtein distance for suggestion

    Returns
    -------
    str or None
        Suggested valid name if found, None otherwise
    """

    def levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    # Find closest match
    min_distance = float("inf")
    closest_match = None

    for valid_name in valid_names:
        distance = levenshtein_distance(invalid_name.lower(), valid_name.lower())
        if distance < min_distance and distance <= threshold:
            min_distance = distance
            closest_match = valid_name

    return closest_match
