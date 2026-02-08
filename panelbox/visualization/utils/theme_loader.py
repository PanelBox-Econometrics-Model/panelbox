"""
Custom Theme Loader - YAML/JSON Support.

This module provides functionality to load custom themes from YAML or JSON files,
allowing users to create and share custom visualization themes.

Features:
- Load themes from YAML/JSON files
- Validate theme structure
- Convert to Theme objects
- Save themes to files
- Merge with base themes

Examples:
    Load theme from file:
    >>> from panelbox.visualization.utils import load_theme
    >>> theme = load_theme('my_theme.yaml')
    >>> chart = ChartFactory.create('bar_chart', data, theme=theme)

    Create and save custom theme:
    >>> from panelbox.visualization import Theme
    >>> from panelbox.visualization.utils import save_theme
    >>> custom_theme = Theme(name='My Theme', colors=['#FF5733', ...])
    >>> save_theme(custom_theme, 'my_theme.yaml')
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Union

import yaml

from ..exceptions import InvalidThemeError, ThemeLoadError
from ..themes import Theme

# Theme schema for validation
THEME_SCHEMA = {
    "required_fields": [
        "name",
        "colors",
        "font_family",
        "font_size",
    ],
    "optional_fields": [
        "background_color",
        "text_color",
        "grid_color",
        "success_color",
        "warning_color",
        "danger_color",
        "info_color",
        "axis_line_color",
        "title_font_size",
        "subtitle_font_size",
        "axis_label_font_size",
        "legend_font_size",
        "annotation_font_size",
        "marker_size",
        "line_width",
        "border_width",
        "corner_radius",
        "spacing",
        "height",
        "width",
        "margin",
    ],
    "types": {
        "name": str,
        "colors": list,
        "font_family": str,
        "font_size": int,
        "background_color": str,
        "text_color": str,
        "grid_color": str,
        "success_color": str,
        "warning_color": str,
        "danger_color": str,
        "info_color": str,
        "title_font_size": int,
        "subtitle_font_size": int,
        "height": int,
        "width": int,
        "marker_size": int,
        "line_width": (int, float),
        "border_width": int,
        "spacing": int,
    },
}


def load_theme(file_path: Union[str, Path], validate: bool = True) -> Theme:
    """
    Load a custom theme from YAML or JSON file.

    Parameters
    ----------
    file_path : str or Path
        Path to theme file (.yaml, .yml, or .json)
    validate : bool, default True
        Whether to validate theme structure

    Returns
    -------
    Theme
        Loaded theme object

    Raises
    ------
    ThemeLoadError
        If file cannot be loaded or theme is invalid
    FileNotFoundError
        If file does not exist

    Examples
    --------
    >>> from panelbox.visualization.utils import load_theme
    >>> theme = load_theme('custom_theme.yaml')
    >>> print(theme.name)
    'My Custom Theme'

    >>> # Use with chart creation
    >>> from panelbox.visualization import ChartFactory
    >>> chart = ChartFactory.create('bar_chart', data, theme=theme)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Theme file not found: {file_path}")

    try:
        # Load file based on extension
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.suffix in [".yaml", ".yml"]:
                theme_data = yaml.safe_load(f)
            elif file_path.suffix == ".json":
                theme_data = json.load(f)
            else:
                raise ThemeLoadError(
                    str(file_path),
                    f"Unsupported file format: {file_path.suffix}. Use .yaml, .yml, or .json",
                )

        if theme_data is None:
            raise ThemeLoadError(str(file_path), "File is empty or invalid")

        # Validate if requested
        if validate:
            _validate_theme_data(theme_data, str(file_path))

        # Create Theme object
        theme = _create_theme_from_dict(theme_data)
        return theme

    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ThemeLoadError(str(file_path), f"Parse error: {str(e)}")
    except Exception as e:
        raise ThemeLoadError(str(file_path), str(e))


def save_theme(
    theme: Theme, file_path: Union[str, Path], format: str = "yaml", include_defaults: bool = False
) -> None:
    """
    Save a theme to YAML or JSON file.

    Parameters
    ----------
    theme : Theme
        Theme object to save
    file_path : str or Path
        Output file path
    format : str, default 'yaml'
        Output format ('yaml' or 'json')
    include_defaults : bool, default False
        Whether to include default values in output

    Examples
    --------
    >>> from panelbox.visualization import PROFESSIONAL_THEME
    >>> from panelbox.visualization.utils import save_theme
    >>> save_theme(PROFESSIONAL_THEME, 'my_theme.yaml')

    >>> # Create custom theme and save
    >>> custom_theme = Theme(
    ...     name='Dark Mode',
    ...     colors=['#FF5733', '#33FF57', '#3357FF'],
    ...     font_family='Arial',
    ...     font_size=12,
    ...     background_color='#1a1a1a',
    ...     text_color='#ffffff'
    ... )
    >>> save_theme(custom_theme, 'dark_theme.json', format='json')
    """
    file_path = Path(file_path)

    # Convert theme to dict
    theme_dict = asdict(theme)

    # Remove defaults if not requested
    if not include_defaults:
        theme_dict = {k: v for k, v in theme_dict.items() if v is not None}

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            if format.lower() in ["yaml", "yml"]:
                yaml.dump(theme_dict, f, default_flow_style=False, sort_keys=False)
            elif format.lower() == "json":
                json.dump(theme_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")

        print(f"✅ Theme saved to: {file_path}")

    except Exception as e:
        raise ThemeLoadError(str(file_path), f"Failed to save: {str(e)}")


def merge_themes(base_theme: Theme, overrides: Dict) -> Theme:
    """
    Create a new theme by merging a base theme with overrides.

    Parameters
    ----------
    base_theme : Theme
        Base theme to start from
    overrides : dict
        Dictionary of fields to override

    Returns
    -------
    Theme
        New theme with merged values

    Examples
    --------
    >>> from panelbox.visualization import PROFESSIONAL_THEME
    >>> from panelbox.visualization.utils import merge_themes
    >>> custom = merge_themes(
    ...     PROFESSIONAL_THEME,
    ...     {'name': 'My Custom', 'background_color': '#f0f0f0'}
    ... )
    """
    # Convert base theme to dict
    theme_dict = asdict(base_theme)

    # Apply overrides
    theme_dict.update(overrides)

    # Create new theme
    return _create_theme_from_dict(theme_dict)


def _validate_theme_data(theme_data: Dict, file_path: str) -> None:
    """
    Validate theme data structure.

    Parameters
    ----------
    theme_data : dict
        Theme data to validate
    file_path : str
        File path for error messages

    Raises
    ------
    ThemeLoadError
        If validation fails
    """
    # Check required fields
    missing_fields = [field for field in THEME_SCHEMA["required_fields"] if field not in theme_data]

    if missing_fields:
        raise ThemeLoadError(file_path, f"Missing required fields: {', '.join(missing_fields)}")

    # Validate field types
    for field, value in theme_data.items():
        if field in THEME_SCHEMA["types"]:
            expected_type = THEME_SCHEMA["types"][field]

            # Handle union types (e.g., int or float)
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    raise ThemeLoadError(
                        file_path,
                        f"Field '{field}' has invalid type. "
                        f"Expected one of {expected_type}, got {type(value)}",
                    )
            else:
                if not isinstance(value, expected_type):
                    raise ThemeLoadError(
                        file_path,
                        f"Field '{field}' has invalid type. "
                        f"Expected {expected_type}, got {type(value)}",
                    )

    # Validate colors array
    if "colors" in theme_data:
        colors = theme_data["colors"]
        if not isinstance(colors, list) or len(colors) < 3:
            raise ThemeLoadError(
                file_path, "Field 'colors' must be a list with at least 3 color values"
            )

        # Validate hex colors
        for color in colors:
            if not isinstance(color, str) or not color.startswith("#"):
                raise ThemeLoadError(
                    file_path,
                    f"Invalid color value: '{color}'. Colors must be hex strings (e.g., '#FF5733')",
                )


def _create_theme_from_dict(theme_data: Dict) -> Theme:
    """
    Create Theme object from dictionary.

    Parameters
    ----------
    theme_data : dict
        Theme data

    Returns
    -------
    Theme
        Theme object
    """
    # Filter to only valid Theme fields
    valid_fields = THEME_SCHEMA["required_fields"] + THEME_SCHEMA["optional_fields"]

    filtered_data = {k: v for k, v in theme_data.items() if k in valid_fields}

    try:
        return Theme(**filtered_data)
    except TypeError as e:
        raise InvalidThemeError(
            theme_data.get("name", "unknown"), f"Invalid theme structure: {str(e)}"
        )


def create_theme_template(output_path: Union[str, Path], format: str = "yaml") -> None:
    """
    Create a template theme file with all available options.

    Parameters
    ----------
    output_path : str or Path
        Output file path
    format : str, default 'yaml'
        Output format ('yaml' or 'json')

    Examples
    --------
    >>> from panelbox.visualization.utils import create_theme_template
    >>> create_theme_template('my_theme_template.yaml')
    >>> # Edit the template and load it
    >>> theme = load_theme('my_theme_template.yaml')
    """
    template = {
        "name": "My Custom Theme",
        "colors": [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow-green
            "#17becf",  # Cyan
        ],
        "font_family": "Inter, system-ui, -apple-system, sans-serif",
        "font_size": 12,
        "background_color": "#ffffff",
        "text_color": "#333333",
        "grid_color": "#e0e0e0",
        "success_color": "#10b981",
        "warning_color": "#f59e0b",
        "danger_color": "#ef4444",
        "info_color": "#3b82f6",
        "axis_line_color": "#666666",
        "title_font_size": 20,
        "subtitle_font_size": 14,
        "axis_label_font_size": 11,
        "legend_font_size": 11,
        "annotation_font_size": 10,
        "marker_size": 8,
        "line_width": 2.0,
        "border_width": 1,
        "corner_radius": 4,
        "spacing": 10,
        "height": 500,
        "width": 800,
        "margin": {"l": 80, "r": 40, "t": 80, "b": 60},
    }

    output_path = Path(output_path)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            if format.lower() in ["yaml", "yml"]:
                # Add comments for YAML
                f.write("# PanelBox Custom Theme Template\n")
                f.write("# Edit values below to create your custom theme\n\n")
                yaml.dump(template, f, default_flow_style=False, sort_keys=False)
            elif format.lower() == "json":
                json.dump(template, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

        print(f"✅ Theme template created at: {output_path}")
        print("\n💡 Edit the file and load it with:")
        print("   from panelbox.visualization.utils import load_theme")
        print(f"   theme = load_theme('{output_path}')")

    except Exception as e:
        raise ThemeLoadError(str(output_path), f"Failed to create template: {str(e)}")


# Convenience function for listing built-in themes
def list_builtin_themes() -> List[str]:
    """
    List all built-in theme names.

    Returns
    -------
    list of str
        Names of built-in themes

    Examples
    --------
    >>> from panelbox.visualization.utils import list_builtin_themes
    >>> themes = list_builtin_themes()
    >>> print(themes)
    ['professional', 'academic', 'presentation']
    """
    return ["professional", "academic", "presentation"]


def get_theme_colors(theme_name_or_path: Union[str, Path]) -> List[str]:
    """
    Get color palette from a theme.

    Parameters
    ----------
    theme_name_or_path : str or Path
        Built-in theme name or path to theme file

    Returns
    -------
    list of str
        List of color hex codes

    Examples
    --------
    >>> from panelbox.visualization.utils import get_theme_colors
    >>> colors = get_theme_colors('professional')
    >>> print(colors[:3])
    ['#2563eb', '#dc2626', '#059669']
    """
    from ..themes import get_theme

    # Try as built-in theme first
    if isinstance(theme_name_or_path, str) and theme_name_or_path in list_builtin_themes():
        theme = get_theme(theme_name_or_path)
    else:
        # Try loading from file
        theme = load_theme(theme_name_or_path)

    return theme.colors


if __name__ == "__main__":
    # Create template when run as script
    import sys

    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = "custom_theme_template.yaml"

    create_theme_template(output_file)
