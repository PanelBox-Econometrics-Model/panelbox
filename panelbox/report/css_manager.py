"""
CSS Manager for PanelBox Reports.

Manages compilation and layering of CSS styles with 3-layer architecture.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from .asset_manager import AssetManager


@dataclass
class CSSLayer:
    """
    Represents a CSS layer in the compilation pipeline.

    Attributes
    ----------
    name : str
        Layer name (e.g., 'base', 'components', 'custom')
    files : list of str
        CSS files to include in this layer
    priority : int
        Layer priority (lower = earlier in output)
    """
    name: str
    files: List[str]
    priority: int = 0


class CSSManager:
    """
    Manages CSS compilation with 3-layer architecture.

    Implements a layered CSS system inspired by modern design systems:
    - Layer 1 (Base): Design tokens, reset, utilities
    - Layer 2 (Components): Reusable UI components
    - Layer 3 (Custom): Report-specific overrides

    Parameters
    ----------
    asset_manager : AssetManager, optional
        Asset manager for loading CSS files. If None, creates default.
    minify : bool, default=False
        Enable CSS minification

    Attributes
    ----------
    asset_manager : AssetManager
        Asset manager instance
    layers : dict
        Registered CSS layers
    minify : bool
        Whether to minify output

    Examples
    --------
    >>> css_mgr = CSSManager(minify=False)
    >>> css = css_mgr.compile()
    >>> css_mgr.add_custom_css('custom-styles.css')
    >>> css = css_mgr.compile()
    """

    # Default layer configuration
    DEFAULT_LAYERS = {
        'base': CSSLayer(
            name='base',
            files=['base_styles.css'],
            priority=0
        ),
        'components': CSSLayer(
            name='components',
            files=['report_components.css'],
            priority=10
        ),
        'custom': CSSLayer(
            name='custom',
            files=[],
            priority=20
        )
    }

    def __init__(
        self,
        asset_manager: Optional[AssetManager] = None,
        minify: bool = False
    ):
        """Initialize CSS Manager."""
        if asset_manager is None:
            asset_manager = AssetManager(minify=minify)

        self.asset_manager = asset_manager
        self.minify = minify

        # Initialize layers with defaults
        self.layers: Dict[str, CSSLayer] = {}
        for name, layer in self.DEFAULT_LAYERS.items():
            self.layers[name] = CSSLayer(
                name=layer.name,
                files=layer.files.copy(),
                priority=layer.priority
            )

        # Track custom CSS snippets
        self.custom_css: List[str] = []

        # Compilation cache
        self._compiled_css: Optional[str] = None
        self._cache_valid = False

    def add_layer(
        self,
        name: str,
        files: List[str],
        priority: int
    ) -> None:
        """
        Add a new CSS layer.

        Parameters
        ----------
        name : str
            Layer name
        files : list of str
            CSS files to include
        priority : int
            Layer priority (lower = earlier in output)

        Examples
        --------
        >>> css_mgr.add_layer('theme', ['dark-theme.css'], priority=5)
        """
        self.layers[name] = CSSLayer(
            name=name,
            files=files,
            priority=priority
        )
        self._invalidate_cache()

    def add_css_to_layer(self, layer_name: str, css_file: str) -> None:
        """
        Add a CSS file to an existing layer.

        Parameters
        ----------
        layer_name : str
            Name of the layer
        css_file : str
            CSS file to add

        Examples
        --------
        >>> css_mgr.add_css_to_layer('custom', 'my-styles.css')
        """
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' does not exist")

        if css_file not in self.layers[layer_name].files:
            self.layers[layer_name].files.append(css_file)
            self._invalidate_cache()

    def remove_css_from_layer(self, layer_name: str, css_file: str) -> None:
        """
        Remove a CSS file from a layer.

        Parameters
        ----------
        layer_name : str
            Name of the layer
        css_file : str
            CSS file to remove

        Examples
        --------
        >>> css_mgr.remove_css_from_layer('custom', 'old-styles.css')
        """
        if layer_name not in self.layers:
            raise ValueError(f"Layer '{layer_name}' does not exist")

        if css_file in self.layers[layer_name].files:
            self.layers[layer_name].files.remove(css_file)
            self._invalidate_cache()

    def add_custom_css(self, css_file: str) -> None:
        """
        Add a custom CSS file to the custom layer.

        Convenience method for adding files to the custom layer.

        Parameters
        ----------
        css_file : str
            CSS file to add

        Examples
        --------
        >>> css_mgr.add_custom_css('validation-custom.css')
        """
        self.add_css_to_layer('custom', css_file)

    def add_inline_css(self, css_content: str) -> None:
        """
        Add inline CSS snippet.

        Parameters
        ----------
        css_content : str
            CSS content to add

        Examples
        --------
        >>> css_mgr.add_inline_css('.my-class { color: red; }')
        """
        self.custom_css.append(css_content)
        self._invalidate_cache()

    def compile(self, force: bool = False) -> str:
        """
        Compile all CSS layers into a single string.

        Parameters
        ----------
        force : bool, default=False
            Force recompilation even if cache is valid

        Returns
        -------
        str
            Compiled CSS content

        Examples
        --------
        >>> css = css_mgr.compile()
        >>> # Use in template
        >>> '<style>{{ css }}</style>'
        """
        # Return cached version if valid
        if self._cache_valid and not force and self._compiled_css is not None:
            return self._compiled_css

        css_parts = []

        # Sort layers by priority
        sorted_layers = sorted(
            self.layers.values(),
            key=lambda layer: layer.priority
        )

        # Compile each layer
        for layer in sorted_layers:
            if not layer.files:
                continue

            # Add layer header
            css_parts.append(
                f"/* ========================================\n"
                f" * Layer: {layer.name.upper()} (Priority: {layer.priority})\n"
                f" * ======================================== */\n"
            )

            # Collect CSS files for this layer
            layer_css = self.asset_manager.collect_css(layer.files)
            css_parts.append(layer_css)
            css_parts.append("\n")

        # Add custom inline CSS
        if self.custom_css:
            css_parts.append(
                f"/* ========================================\n"
                f" * INLINE CUSTOM CSS\n"
                f" * ======================================== */\n"
            )
            for custom in self.custom_css:
                css_parts.append(custom)
                css_parts.append("\n")

        # Join all parts
        compiled = "".join(css_parts)

        # Cache result
        self._compiled_css = compiled
        self._cache_valid = True

        return compiled

    def compile_for_report_type(self, report_type: str) -> str:
        """
        Compile CSS with report-type-specific styles.

        Automatically adds report-type CSS if available.

        Parameters
        ----------
        report_type : str
            Report type (e.g., 'validation', 'regression', 'gmm')

        Returns
        -------
        str
            Compiled CSS content

        Examples
        --------
        >>> css = css_mgr.compile_for_report_type('validation')
        """
        # Check if report-type CSS exists
        report_css_file = f"{report_type}_report.css"

        # Temporarily add to custom layer
        original_custom_files = self.layers['custom'].files.copy()

        try:
            # Add report-type CSS if it exists
            self.add_custom_css(report_css_file)
        except FileNotFoundError:
            # File doesn't exist, that's okay
            pass

        # Compile
        css = self.compile(force=True)

        # Restore original custom files
        self.layers['custom'].files = original_custom_files
        self._invalidate_cache()

        return css

    def get_layer_info(self) -> Dict[str, Dict]:
        """
        Get information about all layers.

        Returns
        -------
        dict
            Dictionary mapping layer name to layer info

        Examples
        --------
        >>> info = css_mgr.get_layer_info()
        >>> print(info['base']['files'])
        ['base_styles.css']
        """
        return {
            name: {
                'priority': layer.priority,
                'files': layer.files.copy(),
                'file_count': len(layer.files)
            }
            for name, layer in self.layers.items()
        }

    def reset_to_defaults(self) -> None:
        """
        Reset CSS layers to default configuration.

        Examples
        --------
        >>> css_mgr.add_custom_css('temp.css')
        >>> css_mgr.reset_to_defaults()
        >>> # All custom CSS removed
        """
        self.layers.clear()
        for name, layer in self.DEFAULT_LAYERS.items():
            self.layers[name] = CSSLayer(
                name=layer.name,
                files=layer.files.copy(),
                priority=layer.priority
            )

        self.custom_css.clear()
        self._invalidate_cache()

    def list_available_css(self) -> Dict[str, List[str]]:
        """
        List all available CSS files from asset manager.

        Returns
        -------
        dict
            Dictionary with 'css' key containing list of available files

        Examples
        --------
        >>> files = css_mgr.list_available_css()
        >>> print(files['css'])
        ['base_styles.css', 'report_components.css', ...]
        """
        return self.asset_manager.list_assets(asset_type='css')

    def get_size_estimate(self) -> Dict[str, int]:
        """
        Estimate size of compiled CSS.

        Returns
        -------
        dict
            Dictionary with size estimates in bytes

        Examples
        --------
        >>> sizes = css_mgr.get_size_estimate()
        >>> print(f"Total size: {sizes['total'] / 1024:.1f} KB")
        """
        css = self.compile()

        sizes = {
            'total': len(css.encode('utf-8')),
            'total_kb': len(css.encode('utf-8')) / 1024
        }

        # Estimate per layer
        for name, layer in self.layers.items():
            if not layer.files:
                continue

            layer_css = self.asset_manager.collect_css(layer.files)
            sizes[f'{name}_layer'] = len(layer_css.encode('utf-8'))

        return sizes

    def validate_layers(self) -> Dict[str, List[str]]:
        """
        Validate that all CSS files in layers exist.

        Returns
        -------
        dict
            Dictionary mapping layer name to list of missing files

        Examples
        --------
        >>> missing = css_mgr.validate_layers()
        >>> if missing['custom']:
        ...     print(f"Missing files: {missing['custom']}")
        """
        missing = {}

        for name, layer in self.layers.items():
            missing[name] = []
            for css_file in layer.files:
                try:
                    self.asset_manager.get_css(css_file)
                except FileNotFoundError:
                    missing[name].append(css_file)

        return missing

    def clear_cache(self) -> None:
        """Clear compilation cache."""
        self._invalidate_cache()
        self.asset_manager.clear_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate compilation cache."""
        self._cache_valid = False
        self._compiled_css = None

    def __repr__(self) -> str:
        """String representation."""
        layer_count = len(self.layers)
        file_count = sum(len(layer.files) for layer in self.layers.values())
        custom_count = len(self.custom_css)

        return (
            f"CSSManager("
            f"layers={layer_count}, "
            f"files={file_count}, "
            f"custom_snippets={custom_count}, "
            f"minify={self.minify})"
        )
