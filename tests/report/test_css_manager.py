"""
Tests for CSSManager.
"""

import pytest

from panelbox.report.asset_manager import AssetManager
from panelbox.report.css_manager import CSSManager


class TestCSSManagerInit:
    """Test CSSManager initialization."""

    def test_init_default(self):
        """Test default initialization creates AssetManager."""
        cm = CSSManager()
        assert cm.asset_manager is not None
        assert isinstance(cm.asset_manager, AssetManager)

    def test_init_with_asset_manager(self):
        """Test initialization with custom AssetManager."""
        am = AssetManager()
        cm = CSSManager(asset_manager=am)
        assert cm.asset_manager is am

    def test_init_with_minify(self):
        """Test initialization with minify enabled."""
        cm = CSSManager(minify=True)
        assert cm.minify is True

    def test_init_default_layers(self):
        """Test that default layers are created."""
        cm = CSSManager()
        assert "base" in cm.layers
        assert "components" in cm.layers
        assert "custom" in cm.layers

    def test_init_empty_custom_css(self):
        """Test that custom_css starts empty."""
        cm = CSSManager()
        assert cm.custom_css == []


class TestCSSManagerLayers:
    """Test layer management."""

    def test_add_layer(self):
        """Test adding a new CSS layer."""
        cm = CSSManager()
        cm.add_layer("theme", files=["dark-theme.css"], priority=5)
        assert "theme" in cm.layers
        assert cm.layers["theme"].priority == 5

    def test_add_css_to_layer(self):
        """Test adding CSS file to existing layer."""
        cm = CSSManager()
        cm.add_css_to_layer("custom", "my-styles.css")
        assert "my-styles.css" in cm.layers["custom"].files

    def test_add_css_to_nonexistent_layer_raises(self):
        """Test adding CSS to nonexistent layer raises ValueError."""
        cm = CSSManager()
        with pytest.raises(ValueError, match="does not exist"):
            cm.add_css_to_layer("nonexistent_layer", "body.css")

    def test_add_css_to_layer_no_duplicates(self):
        """Test adding duplicate CSS file is ignored."""
        cm = CSSManager()
        cm.add_css_to_layer("custom", "my-styles.css")
        cm.add_css_to_layer("custom", "my-styles.css")
        assert cm.layers["custom"].files.count("my-styles.css") == 1

    def test_remove_css_from_layer(self):
        """Test removing CSS file from layer."""
        cm = CSSManager()
        cm.add_css_to_layer("custom", "old-styles.css")
        cm.remove_css_from_layer("custom", "old-styles.css")
        assert "old-styles.css" not in cm.layers["custom"].files

    def test_remove_css_from_nonexistent_layer_raises(self):
        """Test removing CSS from nonexistent layer raises ValueError."""
        cm = CSSManager()
        with pytest.raises(ValueError, match="does not exist"):
            cm.remove_css_from_layer("nonexistent", "file.css")

    def test_remove_css_nonexistent_file_noop(self):
        """Test removing nonexistent file from layer is a no-op."""
        cm = CSSManager()
        # Should not raise
        cm.remove_css_from_layer("custom", "nonexistent.css")

    def test_add_custom_css_convenience(self):
        """Test add_custom_css convenience method."""
        cm = CSSManager()
        cm.add_custom_css("validation-custom.css")
        assert "validation-custom.css" in cm.layers["custom"].files

    def test_reset_to_defaults(self):
        """Test resetting layers to defaults."""
        cm = CSSManager()
        cm.add_layer("theme", files=["dark.css"], priority=5)
        cm.add_inline_css(".custom { color: red; }")
        cm.reset_to_defaults()
        assert "theme" not in cm.layers
        assert "base" in cm.layers
        assert "components" in cm.layers
        assert "custom" in cm.layers
        assert cm.custom_css == []


class TestCSSManagerCompile:
    """Test CSS compilation."""

    def test_compile_returns_string(self):
        """Test compile returns CSS string."""
        cm = CSSManager()
        css = cm.compile()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_compile_cache_reuse(self):
        """Test that compile cache is reused on second call."""
        cm = CSSManager()
        css1 = cm.compile()
        css2 = cm.compile()
        assert css1 == css2
        assert cm._cache_valid is True

    def test_compile_force_recompile(self):
        """Test force recompilation."""
        cm = CSSManager()
        cm.compile()
        css2 = cm.compile(force=True)
        assert isinstance(css2, str)

    def test_compile_includes_layer_headers(self):
        """Test compiled CSS includes layer headers."""
        cm = CSSManager()
        css = cm.compile()
        assert "Layer: BASE" in css
        assert "Layer: COMPONENTS" in css

    def test_add_inline_css(self):
        """Test adding inline CSS appears in compiled output."""
        cm = CSSManager()
        cm.add_inline_css("body { color: blue; }")
        compiled = cm.compile()
        assert "color: blue" in compiled

    def test_compile_with_inline_css_header(self):
        """Test compile includes inline CSS section header."""
        cm = CSSManager()
        cm.add_inline_css(".test { margin: 0; }")
        result = cm.compile()
        assert "INLINE CUSTOM CSS" in result
        assert ".test" in result

    def test_invalidate_cache_on_add(self):
        """Test cache is invalidated when adding CSS."""
        cm = CSSManager()
        cm.compile()
        assert cm._cache_valid is True
        cm.add_inline_css(".new { padding: 0; }")
        assert cm._cache_valid is False


class TestCSSManagerReportType:
    """Test report-type compilation."""

    def test_compile_for_report_type(self):
        """Test compile for specific report type returns string."""
        cm = CSSManager()
        result = cm.compile_for_report_type("regression")
        assert isinstance(result, str)

    def test_compile_for_report_type_comparison(self):
        """Test compile for comparison report type includes comparison CSS."""
        cm = CSSManager()
        result = cm.compile_for_report_type("comparison")
        assert isinstance(result, str)

    def test_compile_for_report_type_restores_custom(self):
        """Test that compile_for_report_type restores original custom files."""
        cm = CSSManager()
        original_custom = cm.layers["custom"].files.copy()
        cm.compile_for_report_type("regression")
        assert cm.layers["custom"].files == original_custom


class TestCSSManagerUtils:
    """Test utility methods."""

    def test_get_layer_info(self):
        """Test getting layer information."""
        cm = CSSManager()
        info = cm.get_layer_info()
        assert isinstance(info, dict)
        assert "base" in info
        assert "priority" in info["base"]
        assert "files" in info["base"]
        assert "file_count" in info["base"]

    def test_list_available_css(self):
        """Test listing available CSS files."""
        cm = CSSManager()
        files = cm.list_available_css()
        assert isinstance(files, dict)
        assert "css" in files

    def test_get_size_estimate(self):
        """Test getting CSS size estimate."""
        cm = CSSManager()
        size = cm.get_size_estimate()
        assert isinstance(size, dict)
        assert "total" in size
        assert "total_kb" in size
        assert size["total"] > 0

    def test_validate_layers(self):
        """Test layer validation returns missing files dict."""
        cm = CSSManager()
        result = cm.validate_layers()
        assert isinstance(result, dict)
        # Default layers should have no missing files
        assert result["base"] == []
        assert result["components"] == []

    def test_validate_layers_with_missing(self):
        """Test layer validation detects missing files."""
        cm = CSSManager()
        cm.add_css_to_layer("custom", "nonexistent_file.css")
        result = cm.validate_layers()
        assert "nonexistent_file.css" in result["custom"]

    def test_clear_cache(self):
        """Test clearing compilation cache."""
        cm = CSSManager()
        cm.compile()
        cm.clear_cache()
        assert cm._cache_valid is False
        assert cm._compiled_css is None

    def test_repr(self):
        """Test string representation."""
        cm = CSSManager()
        repr_str = repr(cm)
        assert "CSSManager" in repr_str
        assert "layers=" in repr_str
        assert "minify=" in repr_str
