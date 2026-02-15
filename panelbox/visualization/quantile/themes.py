"""
Publication-ready themes for quantile regression visualizations.

This module provides predefined themes optimized for different publication
venues including academic journals, presentations, and reports.
"""

from typing import Any, Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt


class PublicationTheme:
    """
    Predefined themes for publication-quality plots.

    Provides optimized settings for various publication venues including
    Nature, Science, economics journals, and presentations.

    Examples
    --------
    >>> PublicationTheme.apply('nature')
    >>> # Now all plots will use Nature's style guidelines

    >>> # Or use as context manager
    >>> with PublicationTheme.use('economics'):
    ...     fig = plt.figure()
    ...     # Create plot with economics journal style
    """

    THEMES = {
        "nature": {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1,
            "lines.markersize": 3,
            "figure.figsize": (3.5, 2.625),  # Single column
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
        },
        "science": {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.75,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "figure.figsize": (3.25, 2.5),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
        "economics": {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.linewidth": 1,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "figure.figsize": (4.5, 3.5),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": True,
            "axes.grid.alpha": 0.3,
            "grid.linestyle": ":",
            "legend.frameon": True,
            "legend.shadow": True,
        },
        "presentation": {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 14,
            "axes.linewidth": 2,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 3,
            "lines.markersize": 8,
            "figure.figsize": (10, 7),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "axes.grid": True,
            "axes.grid.alpha": 0.3,
            "grid.linestyle": "--",
        },
        "poster": {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 24,
            "axes.linewidth": 3,
            "axes.labelsize": 28,
            "axes.titlesize": 32,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
            "lines.linewidth": 4,
            "lines.markersize": 12,
            "figure.figsize": (16, 12),
            "figure.dpi": 100,
            "savefig.dpi": 150,
        },
        "minimal": {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "figure.figsize": (6, 4),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.grid": False,
            "legend.frameon": False,
        },
        "aea": {  # American Economic Association
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 1.2,
            "figure.figsize": (6.5, 4.5),  # Full width
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": False,
            "legend.frameon": True,
            "mathtext.fontset": "cm",
        },
        "ieee": {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.linewidth": 0.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1,
            "figure.figsize": (3.5, 2.625),  # Single column
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "axes.grid": True,
            "axes.grid.alpha": 0.3,
            "grid.linestyle": ":",
        },
    }

    # Color palettes optimized for different use cases
    COLOR_PALETTES = {
        "colorblind": [
            "#0173B2",
            "#DE8F05",
            "#029E73",
            "#CC78BC",
            "#ECE133",
            "#56B4E9",
            "#F0E442",
            "#D55E00",
            "#009E73",
        ],
        "grayscale": ["#000000", "#404040", "#808080", "#BFBFBF", "#DFDFDF"],
        "nature": [
            "#E64B35",
            "#4DBBD5",
            "#00A087",
            "#3C5488",
            "#F39B7F",
            "#8491B4",
            "#91D1C2",
            "#DC0000",
        ],
        "economics": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ],
        "vibrant": [
            "#FFC107",
            "#004D40",
            "#DD2C00",
            "#6200EA",
            "#00B8D4",
            "#64DD17",
            "#FF6D00",
            "#304FFE",
        ],
    }

    @classmethod
    def apply(cls, theme_name: str, color_palette: Optional[str] = None) -> None:
        """
        Apply a predefined theme.

        Parameters
        ----------
        theme_name : str
            Name of the theme to apply
        color_palette : str, optional
            Name of color palette to use

        Raises
        ------
        ValueError
            If theme_name is not recognized
        """
        if theme_name not in cls.THEMES:
            available = ", ".join(cls.THEMES.keys())
            raise ValueError(f"Unknown theme: {theme_name}. Available: {available}")

        # Apply theme
        plt.rcParams.update(cls.THEMES[theme_name])

        # Set color palette if specified
        if color_palette and color_palette in cls.COLOR_PALETTES:
            colors = cls.COLOR_PALETTES[color_palette]
            plt.rcParams["axes.prop_cycle"] = plt.cycler("color", colors)

    @classmethod
    def reset(cls) -> None:
        """Reset matplotlib to default settings."""
        mpl.rcdefaults()

    @classmethod
    def get_colors(cls, palette_name: str) -> list:
        """
        Get color palette.

        Parameters
        ----------
        palette_name : str
            Name of the color palette

        Returns
        -------
        list
            List of color hex codes
        """
        if palette_name not in cls.COLOR_PALETTES:
            available = ", ".join(cls.COLOR_PALETTES.keys())
            raise ValueError(f"Unknown palette: {palette_name}. Available: {available}")

        return cls.COLOR_PALETTES[palette_name]

    @classmethod
    def use(cls, theme_name: str, color_palette: Optional[str] = None):
        """
        Context manager for temporary theme application.

        Parameters
        ----------
        theme_name : str
            Name of the theme to apply
        color_palette : str, optional
            Name of color palette to use

        Examples
        --------
        >>> with PublicationTheme.use('nature'):
        ...     plt.plot([1, 2, 3], [1, 4, 9])
        ...     plt.savefig('figure.pdf')
        """
        return ThemeContext(theme_name, color_palette)

    @classmethod
    def save_config(cls, theme_name: str, filename: str) -> None:
        """
        Save theme configuration to file.

        Parameters
        ----------
        theme_name : str
            Name of the theme to save
        filename : str
            Output filename (should end with .mplstyle)
        """
        if theme_name not in cls.THEMES:
            raise ValueError(f"Unknown theme: {theme_name}")

        config = cls.THEMES[theme_name]

        with open(filename, "w") as f:
            f.write(f"# Matplotlib style for {theme_name}\n")
            f.write("# Generated by PanelBox\n\n")

            for key, value in config.items():
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                f.write(f"{key}: {value}\n")

        print(f"Theme configuration saved to {filename}")

    @classmethod
    def get_figsize(cls, theme_name: str, columns: str = "single") -> tuple:
        """
        Get appropriate figure size for a journal.

        Parameters
        ----------
        theme_name : str
            Name of the theme
        columns : str
            'single' or 'double' column width

        Returns
        -------
        tuple
            (width, height) in inches
        """
        if theme_name not in cls.THEMES:
            raise ValueError(f"Unknown theme: {theme_name}")

        base_size = cls.THEMES[theme_name].get("figure.figsize", (6, 4))

        if columns == "double":
            # Double the width for double-column figures
            return (base_size[0] * 2, base_size[1])
        else:
            return base_size


class ThemeContext:
    """Context manager for temporary theme application."""

    def __init__(self, theme_name: str, color_palette: Optional[str] = None):
        self.theme_name = theme_name
        self.color_palette = color_palette
        self.original_rcparams = {}

    def __enter__(self):
        # Store original rcParams
        theme_keys = PublicationTheme.THEMES[self.theme_name].keys()
        self.original_rcparams = {
            key: plt.rcParams[key] for key in theme_keys if key in plt.rcParams
        }

        # Apply new theme
        PublicationTheme.apply(self.theme_name, self.color_palette)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original rcParams
        plt.rcParams.update(self.original_rcparams)


class ColorSchemes:
    """
    Utility class for managing color schemes in visualizations.
    """

    @staticmethod
    def get_sequential(n: int, cmap_name: str = "Blues") -> list:
        """
        Get sequential colors from a colormap.

        Parameters
        ----------
        n : int
            Number of colors needed
        cmap_name : str
            Name of matplotlib colormap

        Returns
        -------
        list
            List of colors
        """
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i / (n - 1)) for i in range(n)]

    @staticmethod
    def get_diverging(n: int, cmap_name: str = "RdBu_r") -> list:
        """
        Get diverging colors from a colormap.

        Parameters
        ----------
        n : int
            Number of colors needed
        cmap_name : str
            Name of matplotlib colormap

        Returns
        -------
        list
            List of colors
        """
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i / (n - 1)) for i in range(n)]

    @staticmethod
    def get_qualitative(n: int, palette: str = "husl") -> list:
        """
        Get qualitative colors using seaborn palettes.

        Parameters
        ----------
        n : int
            Number of colors needed
        palette : str
            Name of seaborn palette

        Returns
        -------
        list
            List of colors
        """
        import seaborn as sns

        return sns.color_palette(palette, n)

    @staticmethod
    def adjust_alpha(colors: list, alpha: float) -> list:
        """
        Adjust alpha (transparency) of colors.

        Parameters
        ----------
        colors : list
            List of colors (can be hex, rgb, etc.)
        alpha : float
            Alpha value (0 = transparent, 1 = opaque)

        Returns
        -------
        list
            List of RGBA colors
        """
        from matplotlib.colors import to_rgba

        return [to_rgba(c, alpha) for c in colors]
