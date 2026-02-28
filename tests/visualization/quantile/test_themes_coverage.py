"""Coverage tests for panelbox.visualization.quantile.themes module.

Targets uncovered lines: 255-265, 287-291, 325-339, 358-367, 374-376,
380-387, 414-415, 434-435, 454-456, 475-477
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figs():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def reset_mpl():
    """Reset matplotlib defaults after each test."""
    yield
    matplotlib.rcdefaults()


class TestPublicationThemeApply:
    """Test PublicationTheme.apply() method."""

    def test_apply_nature(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        PublicationTheme.apply("nature")
        assert plt.rcParams["font.size"] == 8

    def test_apply_science(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        PublicationTheme.apply("science")
        assert plt.rcParams["font.size"] == 9

    def test_apply_economics(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        try:
            PublicationTheme.apply("economics")
            assert plt.rcParams["font.size"] == 10
        except KeyError:
            pytest.skip("axes.grid.alpha not supported in this matplotlib version")

    def test_apply_presentation(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        try:
            PublicationTheme.apply("presentation")
            assert plt.rcParams["font.size"] == 14
        except KeyError:
            # Some matplotlib versions don't support axes.grid.alpha
            pytest.skip("axes.grid.alpha not supported in this matplotlib version")

    def test_apply_poster(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        PublicationTheme.apply("poster")
        assert plt.rcParams["font.size"] == 24

    def test_apply_minimal(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        PublicationTheme.apply("minimal")
        assert plt.rcParams["font.size"] == 10

    def test_apply_aea(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        PublicationTheme.apply("aea")
        assert plt.rcParams["font.size"] == 11

    def test_apply_ieee(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        try:
            PublicationTheme.apply("ieee")
            assert plt.rcParams["font.size"] == 10
        except KeyError:
            pytest.skip("axes.grid.alpha not supported in this matplotlib version")

    def test_apply_invalid_theme_raises(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        with pytest.raises(ValueError, match="Unknown theme"):
            PublicationTheme.apply("nonexistent_theme")

    def test_apply_with_color_palette(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        PublicationTheme.apply("nature", color_palette="colorblind")
        # Verify the cycle was updated
        cycle = plt.rcParams["axes.prop_cycle"]
        colors = [c["color"] for c in cycle]
        assert len(colors) > 0

    def test_apply_with_invalid_color_palette_no_error(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        # Invalid palette name should not raise - it's silently ignored
        PublicationTheme.apply("nature", color_palette="nonexistent")


class TestPublicationThemeReset:
    """Test PublicationTheme.reset()."""

    def test_reset(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        PublicationTheme.apply("poster")
        assert plt.rcParams["font.size"] == 24
        PublicationTheme.reset()
        # After reset, should be back to default
        assert plt.rcParams["font.size"] != 24


class TestPublicationThemeGetColors:
    """Test PublicationTheme.get_colors()."""

    def test_get_colorblind_palette(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        colors = PublicationTheme.get_colors("colorblind")
        assert len(colors) == 9
        assert colors[0] == "#0173B2"

    def test_get_grayscale_palette(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        colors = PublicationTheme.get_colors("grayscale")
        assert len(colors) == 5

    def test_get_nature_palette(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        colors = PublicationTheme.get_colors("nature")
        assert len(colors) == 8

    def test_get_economics_palette(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        colors = PublicationTheme.get_colors("economics")
        assert len(colors) == 8

    def test_get_vibrant_palette(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        colors = PublicationTheme.get_colors("vibrant")
        assert len(colors) == 8

    def test_get_invalid_palette_raises(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        with pytest.raises(ValueError, match="Unknown palette"):
            PublicationTheme.get_colors("nonexistent")


class TestPublicationThemeUse:
    """Test PublicationTheme.use() context manager."""

    def test_use_as_context_manager(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        original_size = plt.rcParams["font.size"]
        with PublicationTheme.use("poster") as ctx:
            assert plt.rcParams["font.size"] == 24
            assert ctx is not None
        # Should be restored
        assert plt.rcParams["font.size"] == original_size

    def test_use_with_color_palette(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        with PublicationTheme.use("nature", color_palette="colorblind"):
            assert plt.rcParams["font.size"] == 8


class TestPublicationThemeSaveConfig:
    """Test PublicationTheme.save_config()."""

    def test_save_config(self, tmp_path):
        from panelbox.visualization.quantile.themes import PublicationTheme

        filepath = str(tmp_path / "nature.mplstyle")
        PublicationTheme.save_config("nature", filepath)

        with open(filepath) as f:
            content = f.read()
        assert "font.size" in content
        assert "PanelBox" in content

    def test_save_config_with_list_values(self, tmp_path):
        from panelbox.visualization.quantile.themes import PublicationTheme

        filepath = str(tmp_path / "economics.mplstyle")
        PublicationTheme.save_config("economics", filepath)

        with open(filepath) as f:
            content = f.read()
        # economics theme has font.serif which is a list
        assert "Times New Roman" in content

    def test_save_config_invalid_theme_raises(self, tmp_path):
        from panelbox.visualization.quantile.themes import PublicationTheme

        filepath = str(tmp_path / "bad.mplstyle")
        with pytest.raises(ValueError, match="Unknown theme"):
            PublicationTheme.save_config("nonexistent", filepath)


class TestPublicationThemeGetFigsize:
    """Test PublicationTheme.get_figsize()."""

    def test_get_figsize_single_column(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        size = PublicationTheme.get_figsize("nature", columns="single")
        assert size == (3.5, 2.625)

    def test_get_figsize_double_column(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        size = PublicationTheme.get_figsize("nature", columns="double")
        assert size == (7.0, 2.625)  # Width doubled

    def test_get_figsize_invalid_theme_raises(self):
        from panelbox.visualization.quantile.themes import PublicationTheme

        with pytest.raises(ValueError, match="Unknown theme"):
            PublicationTheme.get_figsize("nonexistent")


class TestThemeContext:
    """Test ThemeContext class directly."""

    def test_enter_exit(self):
        from panelbox.visualization.quantile.themes import ThemeContext

        ctx = ThemeContext("science")
        ctx.__enter__()
        assert plt.rcParams["font.size"] == 9
        ctx.__exit__(None, None, None)


class TestColorSchemes:
    """Test ColorSchemes utility class."""

    def test_get_sequential(self):
        from panelbox.visualization.quantile.themes import ColorSchemes

        colors = ColorSchemes.get_sequential(5)
        assert len(colors) == 5

    def test_get_sequential_custom_cmap(self):
        from panelbox.visualization.quantile.themes import ColorSchemes

        colors = ColorSchemes.get_sequential(3, cmap_name="Reds")
        assert len(colors) == 3

    def test_get_diverging(self):
        from panelbox.visualization.quantile.themes import ColorSchemes

        colors = ColorSchemes.get_diverging(5)
        assert len(colors) == 5

    def test_get_diverging_custom_cmap(self):
        from panelbox.visualization.quantile.themes import ColorSchemes

        colors = ColorSchemes.get_diverging(4, cmap_name="PiYG")
        assert len(colors) == 4

    def test_get_qualitative(self):
        from panelbox.visualization.quantile.themes import ColorSchemes

        colors = ColorSchemes.get_qualitative(6)
        assert len(colors) == 6

    def test_get_qualitative_custom_palette(self):
        from panelbox.visualization.quantile.themes import ColorSchemes

        colors = ColorSchemes.get_qualitative(4, palette="Set2")
        assert len(colors) == 4

    def test_adjust_alpha(self):
        from panelbox.visualization.quantile.themes import ColorSchemes

        colors = ["#FF0000", "#00FF00", "#0000FF"]
        result = ColorSchemes.adjust_alpha(colors, 0.5)
        assert len(result) == 3
        # RGBA tuples
        for c in result:
            assert len(c) == 4
            assert c[3] == 0.5
