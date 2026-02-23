"""Quantile regression tutorial utilities."""

__version__ = "1.0.0"

from .comparison_helpers import (
    compare_fe_methods,
    compare_qr_ols,
    create_summary_table,
    inter_quantile_test,
    pseudo_r2_table,
    timing_benchmark,
)
from .plot_helpers import (
    plot_bootstrap_distribution,
    plot_check_loss,
    plot_coefficient_grid,
    plot_coefficient_path,
    plot_crossing_detection,
    plot_qte_comparison,
    plot_quantile_fan_chart,
    plot_residual_diagnostics,
    set_quantile_style,
)
from .simulation_helpers import (
    generate_card_education,
    generate_crossing_example,
    generate_financial_returns,
    generate_firm_production,
    generate_heteroskedastic,
    generate_labor_program,
    generate_location_shift,
    generate_treatment_effects,
)

__all__ = [
    "compare_fe_methods",
    # Comparison
    "compare_qr_ols",
    "create_summary_table",
    # Simulation
    "generate_card_education",
    "generate_crossing_example",
    "generate_financial_returns",
    "generate_firm_production",
    "generate_heteroskedastic",
    "generate_labor_program",
    "generate_location_shift",
    "generate_treatment_effects",
    "inter_quantile_test",
    "plot_bootstrap_distribution",
    "plot_check_loss",
    "plot_coefficient_grid",
    # Plotting
    "plot_coefficient_path",
    "plot_crossing_detection",
    "plot_qte_comparison",
    "plot_quantile_fan_chart",
    "plot_residual_diagnostics",
    "pseudo_r2_table",
    "set_quantile_style",
    "timing_benchmark",
]
