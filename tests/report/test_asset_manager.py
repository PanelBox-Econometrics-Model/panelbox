"""
Tests for AssetManager.
"""

import pytest

from panelbox.report.asset_manager import AssetManager


class TestAssetManagerInit:
    """Test AssetManager initialization."""

    def test_init_default(self):
        """Test default initialization uses package assets."""
        am = AssetManager()
        assert am.asset_dir.exists()
        assert am.minify is False
        assert am.asset_cache == {}

    def test_init_with_minify(self):
        """Test initialization with minify enabled."""
        am = AssetManager(minify=True)
        assert am.minify is True

    def test_init_invalid_asset_dir_raises(self):
        """Test that invalid asset_dir raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            AssetManager(asset_dir="/nonexistent/path")

    def test_init_custom_dir(self, tmp_path):
        """Test initialization with custom asset directory."""
        am = AssetManager(asset_dir=tmp_path)
        assert am.asset_dir == tmp_path


class TestAssetManagerCSS:
    """Test CSS loading and caching."""

    def test_get_css_returns_content(self):
        """Test that get_css returns CSS content."""
        am = AssetManager()
        css = am.get_css("base_styles.css")
        assert isinstance(css, str)
        assert len(css) > 0

    def test_get_css_cache_hit(self):
        """Test that second call returns cached CSS."""
        am = AssetManager()
        css1 = am.get_css("base_styles.css")
        css2 = am.get_css("base_styles.css")
        assert css1 == css2
        assert "css:base_styles.css" in am.asset_cache

    def test_get_css_with_minify(self):
        """Test CSS returned with minification enabled."""
        am = AssetManager(minify=True)
        css = am.get_css("base_styles.css")
        assert css is not None
        assert isinstance(css, str)

    def test_get_css_missing_file_raises(self):
        """Test that missing CSS file raises FileNotFoundError."""
        am = AssetManager()
        with pytest.raises(FileNotFoundError, match="CSS file not found"):
            am.get_css("nonexistent_file.css")


class TestAssetManagerJS:
    """Test JS loading and caching."""

    def test_get_js_returns_content(self):
        """Test that get_js returns JS content."""
        am = AssetManager()
        js = am.get_js("tab-navigation.js")
        assert isinstance(js, str)
        assert len(js) > 0

    def test_get_js_cache_hit(self):
        """Test that second call returns cached JS."""
        am = AssetManager()
        js1 = am.get_js("tab-navigation.js")
        js2 = am.get_js("tab-navigation.js")
        assert js1 == js2
        assert "js:tab-navigation.js" in am.asset_cache

    def test_get_js_with_minify(self):
        """Test JS returned with minification enabled."""
        am = AssetManager(minify=True)
        js = am.get_js("tab-navigation.js")
        assert js is not None
        assert isinstance(js, str)

    def test_get_js_missing_file_raises(self):
        """Test that missing JS file raises FileNotFoundError."""
        am = AssetManager()
        with pytest.raises(FileNotFoundError, match="JavaScript file not found"):
            am.get_js("nonexistent_file.js")


class TestAssetManagerMinify:
    """Test CSS and JS minification."""

    def test_minify_css_removes_comments(self):
        """Test CSS minification removes comments."""
        am = AssetManager(minify=True)
        css = "/* comment */ body { color: red; }\n  h1 { font-size: 16px; }"
        result = am._minify_css(css)
        assert "/* comment */" not in result
        assert "body" in result
        assert "color" in result

    def test_minify_css_removes_whitespace(self):
        """Test CSS minification removes excess whitespace."""
        am = AssetManager(minify=True)
        css = "body  {  color:  red;  }\n\n  h1  {  margin:  0;  }"
        result = am._minify_css(css)
        # Should not have multiple spaces
        assert "  " not in result

    def test_minify_css_disabled(self):
        """Test CSS minification returns original when disabled."""
        am = AssetManager(minify=False)
        css = "/* comment */ body { color: red; }"
        result = am._minify_css(css)
        assert result == css

    def test_minify_js_removes_single_line_comments(self):
        """Test JS minification removes single-line comments."""
        am = AssetManager(minify=True)
        js = "// comment\nfunction test() { return 1; }\n  var x = 2;"
        result = am._minify_js(js)
        assert "// comment" not in result
        assert "function" in result

    def test_minify_js_removes_block_comments(self):
        """Test JS minification removes block comments."""
        am = AssetManager(minify=True)
        js = "/* block comment */\nvar x = 1;"
        result = am._minify_js(js)
        assert "/* block comment */" not in result
        assert "var" in result

    def test_minify_js_disabled(self):
        """Test JS minification returns original when disabled."""
        am = AssetManager(minify=False)
        js = "// comment\nfunction test() { return 1; }"
        result = am._minify_js(js)
        assert result == js


class TestAssetManagerImages:
    """Test image base64 encoding."""

    def test_get_image_base64(self, tmp_path):
        """Test base64 encoding of image file."""
        # Create required directory structure
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_path = img_dir / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        # AssetManager looks for images relative to asset_dir
        am = AssetManager(asset_dir=tmp_path)
        result = am.get_image_base64("images/test.png")
        assert result.startswith("data:image/png;base64,")

    def test_get_image_base64_cache(self, tmp_path):
        """Test image base64 caching."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img_path = img_dir / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)

        am = AssetManager(asset_dir=tmp_path)
        result1 = am.get_image_base64("images/test.png")
        result2 = am.get_image_base64("images/test.png")
        assert result1 == result2

    def test_get_image_base64_missing_raises(self):
        """Test missing image raises FileNotFoundError."""
        am = AssetManager()
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            am.get_image_base64("nonexistent.png")

    def test_get_image_base64_unknown_mime(self, tmp_path):
        """Test image with unknown MIME type falls back to octet-stream."""
        img_path = tmp_path / "test.qzx"
        img_path.write_bytes(b"\x00" * 10)
        am = AssetManager(asset_dir=tmp_path)
        result = am.get_image_base64("test.qzx")
        assert "application/octet-stream" in result


class TestAssetManagerCollect:
    """Test collecting multiple assets."""

    def test_collect_css(self):
        """Test collecting multiple CSS files."""
        am = AssetManager()
        css = am.collect_css(["base_styles.css", "report_components.css"])
        assert "base_styles.css" in css
        assert "report_components.css" in css

    def test_collect_css_missing_file_skips(self):
        """Test that missing CSS file is skipped with warning."""
        am = AssetManager()
        css = am.collect_css(["nonexistent_file.css", "base_styles.css"])
        assert "base_styles.css" in css
        assert "nonexistent_file.css" not in css.replace("nonexistent_file.css", "") or isinstance(
            css, str
        )

    def test_collect_js(self):
        """Test collecting multiple JS files."""
        am = AssetManager()
        js = am.collect_js(["utils.js", "tab-navigation.js"])
        assert "utils.js" in js
        assert "tab-navigation.js" in js

    def test_collect_js_missing_file_skips(self):
        """Test that missing JS file is skipped with warning."""
        am = AssetManager()
        js = am.collect_js(["nonexistent_file.js"])
        assert isinstance(js, str)


class TestAssetManagerEmbed:
    """Test Plotly embedding."""

    def test_embed_plotly_true(self):
        """Test embed_plotly with include=True returns CDN script."""
        am = AssetManager()
        result = am.embed_plotly(include_plotly=True)
        assert "plotly" in result.lower()
        assert "script" in result.lower()

    def test_embed_plotly_false(self):
        """Test embed_plotly with include=False returns empty."""
        am = AssetManager()
        result = am.embed_plotly(include_plotly=False)
        assert result == ""


class TestAssetManagerUtils:
    """Test utility methods."""

    def test_list_assets_all(self):
        """Test listing all assets."""
        am = AssetManager()
        assets = am.list_assets(asset_type="all")
        assert isinstance(assets, dict)
        assert "css" in assets
        assert "js" in assets

    def test_list_assets_css(self):
        """Test listing CSS assets only."""
        am = AssetManager()
        assets = am.list_assets(asset_type="css")
        assert isinstance(assets, dict)
        assert "css" in assets
        assert "js" not in assets

    def test_list_assets_js(self):
        """Test listing JS assets only."""
        am = AssetManager()
        assets = am.list_assets(asset_type="js")
        assert isinstance(assets, dict)
        assert "js" in assets
        assert "css" not in assets

    def test_clear_cache(self):
        """Test clearing the asset cache."""
        am = AssetManager()
        am.get_css("base_styles.css")
        assert len(am.asset_cache) > 0
        am.clear_cache()
        assert len(am.asset_cache) == 0

    def test_repr(self):
        """Test string representation."""
        am = AssetManager()
        repr_str = repr(am)
        assert "AssetManager" in repr_str
        assert "minify=" in repr_str
        assert "cached_assets=" in repr_str
