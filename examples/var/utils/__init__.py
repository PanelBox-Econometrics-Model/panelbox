"""VAR tutorial utilities.

This module provides helper functions for:
- Data generation (reproducible simulated panel datasets)
- Visualization (consistent, publication-quality plotting)
- VAR simulation (pedagogical VAR process simulation)
- Diagnostics (statistical tests and model comparison)
"""

__version__ = "1.0.0"

from .data_generators import (
    generate_dynamic_panel,
    generate_energy_panel,
    generate_finance_panel,
    generate_interest_parity_panel,
    generate_macro_panel,
    generate_monetary_policy_panel,
    generate_ppp_panel,
    generate_trade_panel,
)
from .diagnostic_tools import (
    forecast_evaluation,
    granger_causality_summary,
    model_comparison_table,
    residual_diagnostics,
)
from .var_simulation import (
    check_stability,
    companion_matrix,
    simulate_panel_var,
    simulate_var,
    theoretical_irf,
)
from .visualization_helpers import (
    plot_coefficient_heatmap,
    plot_fevd_stacked,
    plot_forecast_fan,
    plot_irf_comparison,
    plot_irf_grid,
    plot_stability_diagram,
    set_academic_style,
)

__all__ = [
    "check_stability",
    "companion_matrix",
    "forecast_evaluation",
    "generate_dynamic_panel",
    "generate_energy_panel",
    "generate_finance_panel",
    "generate_interest_parity_panel",
    # Data generators
    "generate_macro_panel",
    "generate_monetary_policy_panel",
    "generate_ppp_panel",
    "generate_trade_panel",
    "granger_causality_summary",
    "model_comparison_table",
    "plot_coefficient_heatmap",
    "plot_fevd_stacked",
    "plot_forecast_fan",
    "plot_irf_comparison",
    # Visualization
    "plot_irf_grid",
    "plot_stability_diagram",
    # Diagnostics
    "residual_diagnostics",
    "set_academic_style",
    "simulate_panel_var",
    # VAR simulation
    "simulate_var",
    "theoretical_irf",
]
