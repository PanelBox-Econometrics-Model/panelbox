"""
validation — panelbox validation tutorial package.

Exposes utils helpers at package level for convenient notebook imports::

    from validation.utils import generate_firmdata, plot_residuals_by_entity
"""

from .utils import (
    generate_firmdata,
    generate_macro_panel,
    generate_macro_ts_panel,
    generate_panel_comprehensive,
    generate_panel_unbalanced,
    generate_panel_with_outliers,
    generate_real_firms,
    generate_sales_panel,
    generate_small_panel,
    load_dataset,
    plot_acf_panel,
    plot_bootstrap_distribution,
    plot_correlation_heatmap,
    plot_cv_predictions,
    plot_forest_plot,
    plot_influence_index,
    plot_residuals_by_entity,
)

__all__ = [
    "generate_firmdata",
    "generate_macro_panel",
    "generate_macro_ts_panel",
    "generate_panel_comprehensive",
    "generate_panel_unbalanced",
    "generate_panel_with_outliers",
    "generate_real_firms",
    "generate_sales_panel",
    "generate_small_panel",
    "load_dataset",
    "plot_acf_panel",
    "plot_bootstrap_distribution",
    "plot_correlation_heatmap",
    "plot_cv_predictions",
    "plot_forest_plot",
    "plot_influence_index",
    "plot_residuals_by_entity",
]
