"""Chart builders for PanelBox HTML reports.

This module provides Plotly-based chart builders that generate
embeddable HTML ``<div>`` elements for inclusion in report templates.
Each builder accepts a transformer output dict and returns a
``dict[str, str]`` mapping chart names to HTML strings.
"""

from panelbox.report.charts._base import BaseReportChartBuilder
from panelbox.report.charts._utils import (
    PANELBOX_COLORS,
    PLOTLY_LAYOUT_DEFAULTS,
    fig_to_html,
    format_pvalue,
    significance_color,
)
from panelbox.report.charts.discrete_charts import DiscreteChartBuilder
from panelbox.report.charts.gmm_charts import GMMChartBuilder
from panelbox.report.charts.quantile_charts import QuantileChartBuilder
from panelbox.report.charts.regression_charts import RegressionChartBuilder
from panelbox.report.charts.sfa_charts import SFAChartBuilder
from panelbox.report.charts.var_charts import VARChartBuilder

__all__ = [
    "PANELBOX_COLORS",
    "PLOTLY_LAYOUT_DEFAULTS",
    "BaseReportChartBuilder",
    "DiscreteChartBuilder",
    "GMMChartBuilder",
    "QuantileChartBuilder",
    "RegressionChartBuilder",
    "SFAChartBuilder",
    "VARChartBuilder",
    "fig_to_html",
    "format_pvalue",
    "significance_color",
]
