"""
Utility functions for Diagnostics tutorials.

This module provides helper functions for:
- Data generation (reproducible simulated datasets)
- Visualization (consistent plotting across tutorials)
- Unit root analysis (comparison and interpretation)
- Cointegration analysis (comparison and visualization)
- Spatial analysis (weight matrices, Moran's I, LISA)
- Formatting (LaTeX tables, HTML reports)

All functions are designed to work seamlessly with PanelBox and the
tutorial notebooks.
"""

__version__ = "1.0.0"

# Data generators
# Cointegration helpers
from .cointegration_helpers import (
    compare_cointegration_methods,
    compute_half_lives,
    extract_cointegration_vectors,
    plot_cointegration_residuals,
)
from .data_generators import (
    generate_eu_regions,
    generate_firm_productivity,
    generate_interest_rates,
    generate_nlswork,
    generate_oecd_macro,
    generate_penn_world_table,
    generate_ppp_data,
    generate_prices_panel,
    generate_trade_panel,
    generate_us_counties,
)

# General utilities
from .diagnostics_utils import (
    create_results_table,
    export_to_latex,
    format_test_results,
    save_diagnostic_report,
)

# Spatial helpers
from .spatial_helpers import (
    build_weight_matrix,
    lm_decision_tree_summary,
    plot_lisa_map,
    plot_moran_scatterplot,
)

# Unit root helpers
from .unit_root_helpers import (
    compare_unit_root_tests,
    interpret_results,
    plot_levels_vs_differences,
    recommend_transformation,
)

# Visualization
from .visualization_helpers import (
    create_diagnostic_dashboard,
    plot_test_comparison,
    plot_time_series_grid,
    set_diagnostics_style,
)

__all__ = [
    # Spatial
    "build_weight_matrix",
    # Cointegration
    "compare_cointegration_methods",
    # Unit root
    "compare_unit_root_tests",
    "compute_half_lives",
    "create_diagnostic_dashboard",
    "create_results_table",
    "export_to_latex",
    "extract_cointegration_vectors",
    # General utilities
    "format_test_results",
    "generate_eu_regions",
    "generate_firm_productivity",
    "generate_interest_rates",
    "generate_nlswork",
    "generate_oecd_macro",
    # Data generators
    "generate_penn_world_table",
    "generate_ppp_data",
    "generate_prices_panel",
    "generate_trade_panel",
    "generate_us_counties",
    "interpret_results",
    "lm_decision_tree_summary",
    "plot_cointegration_residuals",
    "plot_levels_vs_differences",
    "plot_lisa_map",
    "plot_moran_scatterplot",
    # Visualization
    "plot_test_comparison",
    "plot_time_series_grid",
    "recommend_transformation",
    "save_diagnostic_report",
    "set_diagnostics_style",
]
