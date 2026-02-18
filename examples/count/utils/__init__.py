"""
Utility functions for count models tutorials.

This module provides helper functions for:
- Data generation (reproducible simulated datasets)
- Visualization (consistent plotting across tutorials)
- Diagnostics (common statistical tests)

All functions are designed to work seamlessly with PanelBox and the tutorial notebooks.
"""

__version__ = "1.0.0"

from .data_generators import (
    generate_crime_data,
    generate_healthcare_data,
    generate_innovation_data,
    generate_patent_data,
    generate_policy_impact_data,
    generate_trade_data,
    generate_zinb_healthcare_data,
)
from .diagnostics_helpers import (
    compute_overdispersion_index,
    compute_rootogram_data,
    detect_outliers_count,
    hausman_test_summary,
    overdispersion_test,
    vuong_test_summary,
)
from .visualization_helpers import (
    compare_models_plot,
    plot_irr_forest,
    plot_marginal_effects,
    plot_panel_trends,
    plot_rootogram,
    plot_variance_mean,
    plot_zero_inflation,
)

__all__ = [
    # Data generators
    "generate_healthcare_data",
    "generate_patent_data",
    "generate_crime_data",
    "generate_trade_data",
    "generate_zinb_healthcare_data",
    "generate_policy_impact_data",
    "generate_innovation_data",
    # Visualization
    "plot_rootogram",
    "plot_variance_mean",
    "plot_marginal_effects",
    "plot_irr_forest",
    "compare_models_plot",
    "plot_panel_trends",
    "plot_zero_inflation",
    # Diagnostics
    "compute_overdispersion_index",
    "overdispersion_test",
    "vuong_test_summary",
    "compute_rootogram_data",
    "detect_outliers_count",
    "hausman_test_summary",
]
