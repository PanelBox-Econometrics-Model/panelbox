"""
Tests for theming system.
"""

import pytest

from panelbox.visualization.themes import (
    ACADEMIC_THEME,
    PRESENTATION_THEME,
    PROFESSIONAL_THEME,
    Theme,
    get_theme,
    list_themes,
    register_theme,
)


def test_professional_theme():
    """Test professional theme attributes."""
    assert PROFESSIONAL_THEME.name == "professional"
    assert len(PROFESSIONAL_THEME.color_scheme) > 0
    assert PROFESSIONAL_THEME.font_config["family"]
    assert PROFESSIONAL_THEME.layout_config["paper_bgcolor"]


def test_academic_theme():
    """Test academic theme attributes."""
    assert ACADEMIC_THEME.name == "academic"
    assert ACADEMIC_THEME.plotly_template == "simple_white"


def test_presentation_theme():
    """Test presentation theme attributes."""
    assert PRESENTATION_THEME.name == "presentation"
    assert PRESENTATION_THEME.font_config["size"] > PROFESSIONAL_THEME.font_config["size"]


def test_theme_get_color():
    """Test getting color from theme."""
    theme = PROFESSIONAL_THEME
    color = theme.get_color(0)

    assert color == theme.color_scheme[0]

    # Test wrapping
    wrapped_color = theme.get_color(len(theme.color_scheme) + 1)
    assert wrapped_color == theme.color_scheme[1]


def test_theme_to_dict():
    """Test converting theme to dictionary."""
    theme_dict = PROFESSIONAL_THEME.to_dict()

    assert theme_dict["name"] == "professional"
    assert "color_scheme" in theme_dict
    assert "font_config" in theme_dict


def test_get_theme_by_name():
    """Test getting theme by name."""
    theme = get_theme("professional")
    assert theme == PROFESSIONAL_THEME

    theme = get_theme("academic")
    assert theme == ACADEMIC_THEME


def test_get_theme_case_insensitive():
    """Test theme name is case insensitive."""
    theme = get_theme("PROFESSIONAL")
    assert theme == PROFESSIONAL_THEME


def test_get_theme_with_theme_object():
    """Test passing Theme object to get_theme."""
    theme = get_theme(PROFESSIONAL_THEME)
    assert theme == PROFESSIONAL_THEME


def test_get_theme_invalid():
    """Test invalid theme name."""
    with pytest.raises(ValueError, match="not found"):
        get_theme("nonexistent")


def test_get_theme_invalid_type():
    """Test invalid theme type."""
    with pytest.raises(TypeError, match="must be str or Theme"):
        get_theme(123)


def test_register_custom_theme():
    """Test registering custom theme."""
    custom = Theme(
        name="custom",
        color_scheme=["#FF0000", "#00FF00"],
        font_config={"family": "Arial", "size": 12, "color": "#000"},
        layout_config={"paper_bgcolor": "#FFF"},
    )

    register_theme(custom)

    retrieved = get_theme("custom")
    assert retrieved == custom


def test_register_theme_invalid():
    """Test registering invalid theme."""
    with pytest.raises(TypeError, match="must be a Theme object"):
        register_theme("not a theme")


def test_list_themes():
    """Test listing themes."""
    themes = list_themes()

    assert "professional" in themes
    assert "academic" in themes
    assert "presentation" in themes
    assert len(themes) >= 3
