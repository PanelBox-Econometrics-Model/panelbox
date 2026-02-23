"""Frontier/SFA tutorial utilities."""

# Data generators
from .data_generation import (
    generate_airline_panel,
    generate_bank_panel,
    generate_brazilian_firms,
    generate_dairy_farm_data,
    generate_electricity_panel,
    generate_farm_data,
    generate_hospital_data,
    generate_hospital_panel,
    generate_manufacturing_panel,
    generate_school_panel,
    generate_telecom_panel,
)

# Visualization
from .plotting_helpers import (
    plot_efficiency_evolution,
    plot_efficiency_histogram,
    plot_efficiency_ranking,
    plot_frontier_2d,
    plot_model_comparison,
    plot_variance_decomposition,
    set_sfa_style,
)

# Report templates
from .report_templates import (
    efficiency_ranking_table,
    estimation_table_latex,
    generate_efficiency_report,
    model_comparison_table,
)

# Validation and diagnostics
from .validation_helpers import (
    bootstrap_efficiency_ci,
    efficiency_ranking_stability,
    model_selection_workflow,
    validate_frontier_assumptions,
)

__all__ = [
    "bootstrap_efficiency_ci",
    "efficiency_ranking_stability",
    "efficiency_ranking_table",
    "estimation_table_latex",
    "generate_airline_panel",
    "generate_bank_panel",
    "generate_brazilian_firms",
    "generate_dairy_farm_data",
    # Reports
    "generate_efficiency_report",
    "generate_electricity_panel",
    "generate_farm_data",
    # Data generators
    "generate_hospital_data",
    "generate_hospital_panel",
    "generate_manufacturing_panel",
    "generate_school_panel",
    "generate_telecom_panel",
    "model_comparison_table",
    "model_selection_workflow",
    "plot_efficiency_evolution",
    "plot_efficiency_histogram",
    "plot_efficiency_ranking",
    # Visualization
    "plot_frontier_2d",
    "plot_model_comparison",
    "plot_variance_decomposition",
    "set_sfa_style",
    # Validation
    "validate_frontier_assumptions",
]
