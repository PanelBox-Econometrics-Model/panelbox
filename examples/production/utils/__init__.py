"""Production & Deployment tutorial utilities."""

from .data_generators import (
    generate_bank_lgd,
    generate_firm_panel,
    generate_future_macro,
    generate_macro_quarterly,
    generate_new_bank_data,
    generate_new_firms,
)
from .evaluation_helpers import (
    direction_accuracy,
    forecast_evaluation_table,
    mae,
    mape,
    prediction_interval_coverage,
    rmse,
    theil_u,
)
from .visualization_helpers import (
    plot_actual_vs_predicted,
    plot_coefficient_drift,
    plot_forecast_trajectory,
    plot_model_comparison,
    plot_prediction_errors,
    set_production_style,
)

__all__ = [
    "direction_accuracy",
    "forecast_evaluation_table",
    "generate_bank_lgd",
    # Data generators
    "generate_firm_panel",
    "generate_future_macro",
    "generate_macro_quarterly",
    "generate_new_bank_data",
    "generate_new_firms",
    "mae",
    "mape",
    # Visualization
    "plot_actual_vs_predicted",
    "plot_coefficient_drift",
    "plot_forecast_trajectory",
    "plot_model_comparison",
    "plot_prediction_errors",
    "prediction_interval_coverage",
    # Evaluation metrics
    "rmse",
    "set_production_style",
    "theil_u",
]
