"""
Asset Manager for PanelBox Reports.

Manages collection, minification, and embedding of assets (CSS, JS, images).
"""

import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional


class AssetManager:
    """
    Manages assets for report generation.

    Handles CSS, JavaScript, and image files. Can embed assets
    inline for self-contained HTML reports.

    Parameters
    ----------
    asset_dir : str or Path, optional
        Directory containing assets. If None, uses package assets.
    minify : bool, default=False
        Enable minification of CSS/JS (requires additional packages)

    Attributes
    ----------
    asset_dir : Path
        Directory containing assets
    minify : bool
        Whether to minify assets
    asset_cache : dict
        Cache for loaded assets

    Examples
    --------
    >>> manager = AssetManager()
    >>> css = manager.get_css('base_styles.css')
    >>> js = manager.get_js('tab-navigation.js')
    """

    def __init__(self, asset_dir: Optional[Path] = None, minify: bool = False):
        """Initialize Asset Manager."""
        # Determine asset directory
        if asset_dir is None:
            # Use package assets
            package_dir = Path(__file__).parent.parent
            asset_dir = package_dir / "templates" / "assets"
        else:
            asset_dir = Path(asset_dir)

        if not asset_dir.exists():
            raise ValueError(f"Asset directory does not exist: {asset_dir}")

        self.asset_dir = asset_dir
        self.minify = minify
        self.asset_cache: Dict[str, str] = {}

    def get_css(self, css_path: str) -> str:
        """
        Load CSS file content.

        Parameters
        ----------
        css_path : str
            Relative path to CSS file from assets/css directory.
            Example: 'base_styles.css' or 'components/buttons.css'

        Returns
        -------
        str
            CSS content

        Examples
        --------
        >>> css = manager.get_css('base_styles.css')
        >>> css = manager.get_css('components/buttons.css')
        """
        cache_key = f"css:{css_path}"

        if cache_key in self.asset_cache:
            return self.asset_cache[cache_key]

        css_file = self.asset_dir / "css" / css_path

        if not css_file.exists():
            raise FileNotFoundError(f"CSS file not found: {css_file}")

        content = css_file.read_text(encoding="utf-8")

        if self.minify:
            content = self._minify_css(content)

        self.asset_cache[cache_key] = content
        return content

    def get_js(self, js_path: str) -> str:
        """
        Load JavaScript file content.

        Parameters
        ----------
        js_path : str
            Relative path to JS file from assets/js directory.
            Example: 'tab-navigation.js' or 'components/charts.js'

        Returns
        -------
        str
            JavaScript content

        Examples
        --------
        >>> js = manager.get_js('tab-navigation.js')
        >>> js = manager.get_js('components/charts.js')
        """
        cache_key = f"js:{js_path}"

        if cache_key in self.asset_cache:
            return self.asset_cache[cache_key]

        js_file = self.asset_dir / "js" / js_path

        if not js_file.exists():
            raise FileNotFoundError(f"JavaScript file not found: {js_file}")

        content = js_file.read_text(encoding="utf-8")

        if self.minify:
            content = self._minify_js(content)

        self.asset_cache[cache_key] = content
        return content

    def get_image_base64(self, image_path: str) -> str:
        """
        Load image and encode as base64 data URI.

        Parameters
        ----------
        image_path : str
            Relative path to image from assets directory

        Returns
        -------
        str
            Base64-encoded data URI

        Examples
        --------
        >>> data_uri = manager.get_image_base64('images/logo.png')
        >>> # Returns: 'data:image/png;base64,iVBORw0KG...'
        """
        cache_key = f"img:{image_path}"

        if cache_key in self.asset_cache:
            return self.asset_cache[cache_key]

        img_file = self.asset_dir / image_path

        if not img_file.exists():
            raise FileNotFoundError(f"Image file not found: {img_file}")

        # Read binary data
        img_data = img_file.read_bytes()

        # Encode as base64
        b64_data = base64.b64encode(img_data).decode("utf-8")

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(img_file))
        if mime_type is None:
            mime_type = "application/octet-stream"

        # Create data URI
        data_uri = f"data:{mime_type};base64,{b64_data}"

        self.asset_cache[cache_key] = data_uri
        return data_uri

    def collect_css(self, css_files: List[str]) -> str:
        """
        Collect multiple CSS files into one string.

        Parameters
        ----------
        css_files : list of str
            List of CSS file paths to collect

        Returns
        -------
        str
            Combined CSS content

        Examples
        --------
        >>> css = manager.collect_css([
        ...     'base_styles.css',
        ...     'report_components.css',
        ...     'components/tables.css'
        ... ])
        """
        css_parts = []

        for css_file in css_files:
            try:
                content = self.get_css(css_file)
                css_parts.append(f"/* ========== {css_file} ========== */\n")
                css_parts.append(content)
                css_parts.append("\n\n")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

        return "".join(css_parts)

    def collect_js(self, js_files: List[str]) -> str:
        """
        Collect multiple JavaScript files into one string.

        Parameters
        ----------
        js_files : list of str
            List of JS file paths to collect

        Returns
        -------
        str
            Combined JavaScript content

        Examples
        --------
        >>> js = manager.collect_js([
        ...     'utils.js',
        ...     'tab-navigation.js'
        ... ])
        """
        js_parts = []

        for js_file in js_files:
            try:
                content = self.get_js(js_file)
                js_parts.append(f"/* ========== {js_file} ========== */\n")
                js_parts.append(content)
                js_parts.append("\n\n")
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

        return "".join(js_parts)

    def embed_plotly(self, include_plotly: bool = True) -> str:
        """
        Get Plotly.js library for embedding.

        Parameters
        ----------
        include_plotly : bool, default=True
            Whether to include full Plotly library

        Returns
        -------
        str
            Plotly.js library content or CDN link

        Examples
        --------
        >>> plotly_js = manager.embed_plotly()
        """
        if not include_plotly:
            return ""

        # Use CDN link for now
        # In production, you might want to embed the full library
        return '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" ' 'charset="utf-8"></script>'

    def _minify_css(self, css: str) -> str:
        """
        Minify CSS content.

        Basic minification without external dependencies.
        For production, consider using cssmin package.

        Parameters
        ----------
        css : str
            CSS content to minify

        Returns
        -------
        str
            Minified CSS
        """
        if not self.minify:
            return css

        # Basic minification
        # Remove comments
        import re

        css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)

        # Remove extra whitespace
        css = re.sub(r"\s+", " ", css)
        css = re.sub(r"\s*([{}:;,])\s*", r"\1", css)

        return css.strip()

    def _minify_js(self, js: str) -> str:
        """
        Minify JavaScript content.

        Basic minification without external dependencies.
        For production, consider using jsmin package.

        Parameters
        ----------
        js : str
            JavaScript content to minify

        Returns
        -------
        str
            Minified JavaScript
        """
        if not self.minify:
            return js

        # Basic minification
        # Remove comments (simple approach)
        import re

        js = re.sub(r"//.*?$", "", js, flags=re.MULTILINE)
        js = re.sub(r"/\*.*?\*/", "", js, flags=re.DOTALL)

        # Remove extra whitespace (but preserve newlines in strings)
        # This is a very basic approach
        js = re.sub(r"\s+", " ", js)

        return js.strip()

    def clear_cache(self) -> None:
        """Clear asset cache."""
        self.asset_cache.clear()

    def list_assets(self, asset_type: str = "all") -> Dict[str, List[str]]:
        """
        List available assets.

        Parameters
        ----------
        asset_type : str, default="all"
            Type of assets to list: "css", "js", "images", or "all"

        Returns
        -------
        dict
            Dictionary mapping asset type to list of paths

        Examples
        --------
        >>> assets = manager.list_assets()
        >>> print(assets['css'])
        ['base_styles.css', 'report_components.css', ...]
        """
        assets = {}

        if asset_type in ("css", "all"):
            css_dir = self.asset_dir / "css"
            if css_dir.exists():
                css_files = [str(p.relative_to(css_dir)) for p in css_dir.rglob("*.css")]
                assets["css"] = sorted(css_files)

        if asset_type in ("js", "all"):
            js_dir = self.asset_dir / "js"
            if js_dir.exists():
                js_files = [str(p.relative_to(js_dir)) for p in js_dir.rglob("*.js")]
                assets["js"] = sorted(js_files)

        if asset_type in ("images", "all"):
            img_dir = self.asset_dir / "images"
            if img_dir.exists():
                img_files = [
                    str(p.relative_to(self.asset_dir)) for p in img_dir.rglob("*") if p.is_file()
                ]
                assets["images"] = sorted(img_files)

        return assets

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AssetManager(asset_dir={self.asset_dir}, "
            f"minify={self.minify}, "
            f"cached_assets={len(self.asset_cache)})"
        )
