"""
Econometric test visualizations.

This module provides specialized visualizations for econometric tests,
including serial correlation (ACF/PACF), unit root tests, cointegration,
and cross-sectional dependence.

Chart Types
-----------
- ACFPACFPlot: Autocorrelation and partial autocorrelation plots
- UnitRootTestPlot: Unit root test results visualization
- CointegrationHeatmap: Cointegration test results heatmap
- CrossSectionalDependencePlot: Cross-sectional dependence visualization
- HausmanTestPlot: Hausman test results visualization

Examples
--------
>>> from panelbox.visualization import create_acf_pacf_plot
>>>
>>> chart = create_acf_pacf_plot(
...     residuals,
...     max_lags=20,
...     confidence_level=0.95,
...     theme='academic'
... )
>>> chart.show()
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import PlotlyChartBase
from ..registry import register_chart
from ..themes import Theme


def _get_font_config(theme: Theme, size_modifier: int = 0) -> dict:
    """Helper to extract font configuration from theme."""
    return dict(
        family=theme.font_config.get("family", "Arial"),
        size=theme.font_config.get("size", 12) + size_modifier,
        color=theme.font_config.get("color", "#2c3e50"),
    )


def calculate_acf(residuals: np.ndarray, max_lags: int) -> np.ndarray:
    """
    Calculate autocorrelation function (ACF).

    Parameters
    ----------
    residuals : np.ndarray
        Residuals or time series data
    max_lags : int
        Maximum number of lags to compute

    Returns
    -------
    np.ndarray
        ACF values for lags 0 to max_lags

    Notes
    -----
    ACF(k) = Corr(εₜ, εₜ₋ₖ) = Cov(εₜ, εₜ₋ₖ) / Var(εₜ)
    """
    residuals = np.asarray(residuals).flatten()
    n = len(residuals)

    # Demean
    residuals = residuals - np.mean(residuals)

    # Calculate ACF
    acf = np.zeros(max_lags + 1)
    variance = np.dot(residuals, residuals) / n

    for lag in range(max_lags + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.dot(residuals[:-lag], residuals[lag:]) / (n * variance)

    return acf


def calculate_pacf(residuals: np.ndarray, max_lags: int) -> np.ndarray:
    """
    Calculate partial autocorrelation function (PACF) using Yule-Walker equations.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals or time series data
    max_lags : int
        Maximum number of lags to compute

    Returns
    -------
    np.ndarray
        PACF values for lags 0 to max_lags

    Notes
    -----
    PACF uses Durbin-Levinson algorithm to solve Yule-Walker equations.
    """
    residuals = np.asarray(residuals).flatten()
    acf = calculate_acf(residuals, max_lags)

    pacf = np.zeros(max_lags + 1)
    pacf[0] = 1.0

    if max_lags == 0:
        return pacf

    pacf[1] = acf[1]

    # Durbin-Levinson recursion
    for lag in range(2, max_lags + 1):
        # Build correlation matrix
        r = acf[1:lag]
        R = np.zeros((lag - 1, lag - 1))

        for i in range(lag - 1):
            for j in range(lag - 1):
                R[i, j] = acf[abs(i - j)]

        # Solve for PACF
        try:
            phi = np.linalg.solve(R, r)
            pacf[lag] = (acf[lag] - np.dot(r, phi)) / (1 - np.dot(r, phi))
        except np.linalg.LinAlgError:
            pacf[lag] = 0.0

    return pacf


def ljung_box_test(residuals: np.ndarray, max_lags: int) -> Dict[str, float]:
    """
    Ljung-Box test for autocorrelation.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals to test
    max_lags : int
        Number of lags to include in test

    Returns
    -------
    dict
        Dictionary with 'statistic' and 'pvalue'

    Notes
    -----
    Q = T(T+2) Σ(ACF²ₖ/(T-k)) for k=1 to m
    H₀: No autocorrelation up to lag m
    """
    from scipy.stats import chi2

    residuals = np.asarray(residuals).flatten()
    n = len(residuals)
    acf_vals = calculate_acf(residuals, max_lags)

    # Ljung-Box statistic
    q_stat = 0.0
    for k in range(1, max_lags + 1):
        q_stat += (acf_vals[k] ** 2) / (n - k)

    q_stat *= n * (n + 2)

    # P-value from chi-squared distribution
    pvalue = 1 - chi2.cdf(q_stat, max_lags)

    return {"statistic": q_stat, "pvalue": pvalue, "df": max_lags}


@register_chart("acf_pacf_plot")
class ACFPACFPlot(PlotlyChartBase):
    """
    Autocorrelation (ACF) and Partial Autocorrelation (PACF) plot.

    Displays ACF and PACF plots in a two-panel subplot to diagnose
    serial correlation in residuals or time series data.

    Parameters
    ----------
    theme : Theme, optional
        Visual theme for the chart

    Examples
    --------
    >>> from panelbox.visualization import ChartFactory
    >>>
    >>> chart = ChartFactory.create(
    ...     'acf_pacf_plot',
    ...     data={'residuals': residuals, 'max_lags': 20},
    ...     theme='academic'
    ... )
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """
        Create ACF/PACF plot.

        Parameters
        ----------
        data : dict
            Dictionary with keys:
            - 'residuals': array-like of residuals
            - 'max_lags': int, maximum lags (optional, default=20)
            - 'confidence_level': float, confidence level (optional, default=0.95)
            - 'show_ljung_box': bool, show Ljung-Box test (optional, default=True)
        **kwargs
            Additional chart options

        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Extract data
        residuals = np.asarray(data.get("residuals", [])).flatten()
        max_lags = data.get("max_lags", min(20, len(residuals) // 4))
        confidence_level = data.get("confidence_level", 0.95)
        show_ljung_box = data.get("show_ljung_box", True)

        if len(residuals) == 0:
            raise ValueError("Residuals cannot be empty")

        # Calculate ACF and PACF
        acf = calculate_acf(residuals, max_lags)
        pacf = calculate_pacf(residuals, max_lags)

        # Calculate confidence bands
        n = len(residuals)
        if confidence_level == 0.95:
            z_critical = 1.96
        elif confidence_level == 0.99:
            z_critical = 2.576
        else:
            from scipy.stats import norm

            z_critical = norm.ppf(1 - (1 - confidence_level) / 2)

        confidence_band = z_critical / np.sqrt(n)

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Autocorrelation Function (ACF)",
                "Partial Autocorrelation Function (PACF)",
            ),
            vertical_spacing=0.12,
        )

        # Lag values (exclude lag 0 from display)
        lags = np.arange(1, max_lags + 1)

        # ACF plot (subplot 1)
        fig.add_trace(
            go.Bar(
                x=lags,
                y=acf[1:],
                marker=dict(
                    color=[
                        (
                            self.theme.success_color
                            if abs(val) > confidence_band
                            else self.theme.get_color(0)
                        )
                        for val in acf[1:]
                    ],
                    line=dict(color=self.theme.get_color(0), width=1),
                ),
                name="ACF",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # PACF plot (subplot 2)
        fig.add_trace(
            go.Bar(
                x=lags,
                y=pacf[1:],
                marker=dict(
                    color=[
                        (
                            self.theme.success_color
                            if abs(val) > confidence_band
                            else self.theme.get_color(1)
                        )
                        for val in pacf[1:]
                    ],
                    line=dict(color=self.theme.get_color(1), width=1),
                ),
                name="PACF",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Add confidence bands for both subplots
        for row in [1, 2]:
            # Upper band
            fig.add_hline(
                y=confidence_band,
                line_dash="dash",
                line_color="rgba(255, 0, 0, 0.5)",
                line_width=1,
                row=row,
                col=1,
            )
            # Lower band
            fig.add_hline(
                y=-confidence_band,
                line_dash="dash",
                line_color="rgba(255, 0, 0, 0.5)",
                line_width=1,
                row=row,
                col=1,
            )
            # Zero line
            fig.add_hline(y=0, line_color="rgba(0, 0, 0, 0.3)", line_width=1, row=row, col=1)

        # Add Ljung-Box test annotation
        if show_ljung_box:
            lb_result = ljung_box_test(residuals, max_lags)
            annotation_text = (
                f"Ljung-Box Test (lag={max_lags})<br>"
                f"Q-statistic: {lb_result['statistic']:.2f}<br>"
                f"p-value: {lb_result['pvalue']:.4f}<br>"
                f"{'✓ No autocorrelation (α=0.05)' if lb_result['pvalue'] > 0.05 else '✗ Autocorrelation detected'}"
            )

            fig.add_annotation(
                text=annotation_text,
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.98,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=self.theme.get_color(0),
                borderwidth=1,
                borderpad=10,
                font=_get_font_config(self.theme, -2),
            )

        # Update layout
        title = kwargs.get("title", "ACF and PACF - Serial Correlation Analysis")

        fig.update_xaxes(title_text="Lag", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=2, col=1)
        fig.update_yaxes(title_text="ACF", row=1, col=1)
        fig.update_yaxes(title_text="PACF", row=2, col=1)

        fig.update_layout(
            title=dict(text=title, font=_get_font_config(self.theme, 4)),
            height=700,
            showlegend=False,
            hovermode="x unified",
        )

        # Apply theme layout
        fig.update_layout(**self.theme.layout_config)

        return fig


@register_chart("unit_root_test_plot")
class UnitRootTestPlot(PlotlyChartBase):
    """
    Unit root test results visualization.

    Displays results from unit root tests (ADF, PP, KPSS, etc.) with
    test statistics vs critical values and optional time series overlay.

    Parameters
    ----------
    theme : Theme, optional
        Visual theme for the chart

    Examples
    --------
    >>> from panelbox.visualization import ChartFactory
    >>>
    >>> chart = ChartFactory.create(
    ...     'unit_root_test_plot',
    ...     data={
    ...         'test_names': ['ADF', 'PP', 'KPSS'],
    ...         'test_stats': [-3.5, -3.8, 0.3],
    ...         'critical_values': {'1%': -3.96, '5%': -3.41, '10%': -3.13},
    ...         'pvalues': [0.008, 0.003, 0.15]
    ...     },
    ...     theme='professional'
    ... )
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """
        Create unit root test visualization.

        Parameters
        ----------
        data : dict
            Dictionary with keys:
            - 'test_names': list of test names
            - 'test_stats': list of test statistics
            - 'critical_values': dict with critical values (e.g., {'1%': -3.96, '5%': -3.41})
            - 'pvalues': list of p-values
            - 'series': optional time series data for overlay
            - 'time_index': optional time index for series
        **kwargs
            Additional chart options

        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Extract data
        test_names = data.get("test_names", [])
        test_stats = data.get("test_stats", [])
        critical_values = data.get("critical_values", {})
        pvalues = data.get("pvalues", [])
        series = data.get("series", None)
        time_index = data.get("time_index", None)

        if len(test_names) == 0:
            raise ValueError("Test names cannot be empty")

        # Determine if we need subplots (series overlay)
        if series is not None and time_index is not None:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("Unit Root Test Statistics", "Time Series"),
                vertical_spacing=0.15,
                row_heights=[0.6, 0.4],
            )
            has_series = True
        else:
            fig = go.Figure()
            has_series = False

        # Color code by significance (assuming H0: unit root exists for ADF/PP)
        colors = []
        for i, pval in enumerate(pvalues):
            if pval < 0.01:
                colors.append(self.theme.success_color)  # Strong rejection
            elif pval < 0.05:
                colors.append(self.theme.get_color(2))  # Moderate rejection
            elif pval < 0.10:
                colors.append(self.theme.warning_color)  # Weak rejection
            else:
                colors.append(self.theme.danger_color)  # Cannot reject

        # Test statistics bar chart
        trace_stats = go.Bar(
            x=test_names,
            y=test_stats,
            marker=dict(color=colors, line=dict(color=self.theme.get_color(0), width=1)),
            name="Test Statistic",
            text=[f"{stat:.3f}<br>p={pval:.4f}" for stat, pval in zip(test_stats, pvalues)],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Statistic: %{y:.3f}<extra></extra>",
        )

        if has_series:
            fig.add_trace(trace_stats, row=1, col=1)
        else:
            fig.add_trace(trace_stats)

        # Add critical value lines
        if critical_values:
            for level, value in critical_values.items():
                row_val = 1 if has_series else None
                col_val = 1 if has_series else None

                fig.add_hline(
                    y=value,
                    line_dash="dash",
                    line_color=self.theme.get_color(3),
                    line_width=2,
                    annotation_text=f"{level} critical value",
                    annotation_position="right",
                    row=row_val,
                    col=col_val,
                )

        # Add time series if provided
        if has_series:
            fig.add_trace(
                go.Scatter(
                    x=time_index,
                    y=series,
                    mode="lines",
                    line=dict(color=self.theme.get_color(0), width=2),
                    name="Series",
                    hovertemplate="Time: %{x}<br>Value: %{y:.3f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Create legend for significance levels
        legend_text = (
            "<b>Significance Levels:</b><br>"
            f"<span style='color:{self.theme.success_color}'>■</span> p < 0.01 (Strong)<br>"
            f"<span style='color:{self.theme.get_color(2)}'>■</span> 0.01 ≤ p < 0.05 (Moderate)<br>"
            f"<span style='color:{self.theme.warning_color}'>■</span> 0.05 ≤ p < 0.10 (Weak)<br>"
            f"<span style='color:{self.theme.danger_color}'>■</span> p ≥ 0.10 (Non-significant)"
        )

        fig.add_annotation(
            text=legend_text,
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.98 if not has_series else 0.55,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=self.theme.get_color(0),
            borderwidth=1,
            borderpad=10,
            font=_get_font_config(self.theme, -2),
            align="left",
        )

        # Update layout
        title = kwargs.get("title", "Unit Root Test Results")

        if has_series:
            fig.update_xaxes(title_text="Test", row=1, col=1)
            fig.update_yaxes(title_text="Test Statistic", row=1, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Value", row=2, col=1)
        else:
            fig.update_xaxes(title_text="Test")
            fig.update_yaxes(title_text="Test Statistic")

        fig.update_layout(
            title=dict(text=title, font=_get_font_config(self.theme, 4)),
            height=700 if has_series else 500,
            showlegend=False,
            hovermode="closest",
        )

        # Apply theme layout
        fig.update_layout(**self.theme.layout_config)

        return fig


@register_chart("cointegration_heatmap")
class CointegrationHeatmap(PlotlyChartBase):
    """
    Cointegration test results heatmap.

    Displays pairwise cointegration test results in a heatmap format,
    useful for identifying cointegrated relationships among multiple series.

    Parameters
    ----------
    theme : Theme, optional
        Visual theme for the chart

    Examples
    --------
    >>> from panelbox.visualization import ChartFactory
    >>>
    >>> chart = ChartFactory.create(
    ...     'cointegration_heatmap',
    ...     data={
    ...         'variables': ['GDP', 'Consumption', 'Investment'],
    ...         'pvalues': [[1.0, 0.02, 0.15],
    ...                     [0.02, 1.0, 0.08],
    ...                     [0.15, 0.08, 1.0]],
    ...         'test_name': 'Engle-Granger'
    ...     },
    ...     theme='academic'
    ... )
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """
        Create cointegration heatmap.

        Parameters
        ----------
        data : dict
            Dictionary with keys:
            - 'variables': list of variable names
            - 'pvalues': 2D array of p-values (n x n matrix)
            - 'test_name': name of cointegration test (optional)
            - 'test_stats': optional 2D array of test statistics
        **kwargs
            Additional chart options

        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Extract data
        variables = data.get("variables", [])
        pvalues = np.array(data.get("pvalues", []))
        test_name = data.get("test_name", "Cointegration Test")
        test_stats = data.get("test_stats", None)
        if test_stats is not None:
            test_stats = np.array(test_stats)

        if len(variables) == 0:
            raise ValueError("Variables cannot be empty")

        n = len(variables)

        # Create text annotations
        text_annotations = []
        for i in range(n):
            row_text = []
            for j in range(n):
                if i == j:
                    row_text.append("—")  # Diagonal (self)
                else:
                    pval = pvalues[i, j]
                    if test_stats is not None:
                        stat = test_stats[i, j]
                        row_text.append(f"p={pval:.3f}<br>stat={stat:.2f}")
                    else:
                        row_text.append(f"p={pval:.3f}")
            text_annotations.append(row_text)

        # Create custom colorscale based on significance
        # Low p-value (strong cointegration) = green
        # High p-value (no cointegration) = red
        colorscale = [
            [0.0, self.theme.success_color],  # p=0 (strong)
            [0.01, self.theme.success_color],
            [0.05, self.theme.get_color(2)],  # p=0.05 (moderate)
            [0.10, self.theme.warning_color],  # p=0.10 (weak)
            [1.0, self.theme.danger_color],  # p=1.0 (none)
        ]

        # Mask diagonal (self-cointegration is meaningless)
        masked_pvalues = pvalues.copy()
        np.fill_diagonal(masked_pvalues, np.nan)

        # Create heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=masked_pvalues,
                x=variables,
                y=variables,
                text=text_annotations,
                texttemplate="%{text}",
                textfont={"size": 10},
                colorscale=colorscale,
                zmin=0,
                zmax=1,
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>p-value: %{z:.4f}<extra></extra>",
                colorbar=dict(
                    title="p-value",
                    tickvals=[0, 0.01, 0.05, 0.10, 0.5, 1.0],
                    ticktext=["0.00<br>(Strong)", "0.01", "0.05", "0.10", "0.50", "1.00<br>(None)"],
                ),
            )
        )

        # Add significance level reference
        legend_text = (
            "<b>Interpretation:</b><br>"
            f"<span style='color:{self.theme.success_color}'>■</span> p < 0.01: Strong cointegration<br>"
            f"<span style='color:{self.theme.get_color(2)}'>■</span> 0.01 ≤ p < 0.05: Moderate<br>"
            f"<span style='color:{self.theme.warning_color}'>■</span> 0.05 ≤ p < 0.10: Weak<br>"
            f"<span style='color:{self.theme.danger_color}'>■</span> p ≥ 0.10: No cointegration"
        )

        fig.add_annotation(
            text=legend_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=self.theme.get_color(0),
            borderwidth=1,
            borderpad=10,
            font=_get_font_config(self.theme, -2),
            align="left",
        )

        # Update layout
        title = kwargs.get("title", f"{test_name} - Pairwise Cointegration Tests")

        fig.update_layout(
            title=dict(text=title, font=_get_font_config(self.theme, 4)),
            xaxis=dict(title="Variable", side="bottom", tickangle=-45),
            yaxis=dict(title="Variable", autorange="reversed"),  # Top to bottom
            height=600,
            width=700,
        )

        # Apply theme layout
        fig.update_layout(**self.theme.layout_config)

        return fig


@register_chart("cross_sectional_dependence_plot")
class CrossSectionalDependencePlot(PlotlyChartBase):
    """
    Cross-sectional dependence test visualization.

    Displays CD test statistics and correlation patterns across entities.

    Parameters
    ----------
    theme : Theme, optional
        Visual theme for the chart

    Examples
    --------
    >>> from panelbox.visualization import ChartFactory
    >>>
    >>> chart = ChartFactory.create(
    ...     'cross_sectional_dependence_plot',
    ...     data={
    ...         'cd_statistic': 5.23,
    ...         'pvalue': 0.001,
    ...         'avg_correlation': 0.42,
    ...         'entity_correlations': [0.3, 0.5, 0.6, 0.2]
    ...     },
    ...     theme='professional'
    ... )
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """
        Create cross-sectional dependence visualization.

        Parameters
        ----------
        data : dict
            Dictionary with keys:
            - 'cd_statistic': Pesaran CD test statistic
            - 'pvalue': p-value
            - 'avg_correlation': average absolute correlation
            - 'entity_correlations': optional list of entity-level correlations
            - 'correlation_matrix': optional full correlation matrix
        **kwargs
            Additional chart options

        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Extract data
        cd_stat = data.get("cd_statistic", None)
        pvalue = data.get("pvalue", None)
        avg_corr = data.get("avg_correlation", None)
        entity_corrs = data.get("entity_correlations", None)
        corr_matrix = data.get("correlation_matrix", None)

        if cd_stat is None:
            raise ValueError("CD statistic is required")

        # Create figure with subplots if entity data available
        if entity_corrs is not None and len(entity_corrs) > 0:
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("CD Test Result", "Entity-Level Correlations"),
                column_widths=[0.4, 0.6],
                specs=[[{"type": "indicator"}, {"type": "bar"}]],
            )
            has_entity_data = True
        else:
            fig = go.Figure()
            has_entity_data = False

        # Determine color based on significance
        if pvalue is not None:
            if pvalue < 0.01:
                color = self.theme.danger_color  # Strong CD (bad for independence)
            elif pvalue < 0.05:
                color = self.theme.warning_color
            else:
                color = self.theme.success_color  # No CD (good)
        else:
            color = self.theme.get_color(0)

        # CD statistic indicator
        indicator = go.Indicator(
            mode="number+delta+gauge",
            value=cd_stat,
            title=dict(text="Pesaran CD Statistic"),
            delta=dict(reference=0, increasing=dict(color=self.theme.danger_color)),
            gauge=dict(
                axis=dict(range=[-5, 5]),
                bar=dict(color=color),
                threshold=dict(
                    line=dict(color="red", width=4), thickness=0.75, value=1.96  # 5% critical value
                ),
                steps=[
                    dict(range=[-5, -1.96], color="lightgray"),
                    dict(range=[-1.96, 1.96], color="lightgreen"),
                    dict(range=[1.96, 5], color="lightcoral"),
                ],
            ),
            domain=dict(x=[0, 1], y=[0.3, 1]),
        )

        if has_entity_data:
            fig.add_trace(indicator, row=1, col=1)
        else:
            fig.add_trace(indicator)

        # Add text annotations for statistics
        stats_text = (
            f"<b>CD Statistic:</b> {cd_stat:.3f}<br>" f"<b>p-value:</b> {pvalue:.4f}<br>"
            if pvalue is not None
            else (
                "" f"<b>Avg |ρ|:</b> {avg_corr:.3f}<br>"
                if avg_corr is not None
                else ""
                f"<b>Interpretation:</b> {'Strong CD' if pvalue and pvalue < 0.05 else 'Weak/No CD'}"
            )
        )

        fig.add_annotation(
            text=stats_text,
            xref="paper",
            yref="paper",
            x=0.5 if not has_entity_data else 0.2,
            y=0.15,
            xanchor="center",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor=self.theme.get_color(0),
            borderwidth=1,
            borderpad=10,
            font=_get_font_config(self.theme, -1),
            align="left",
        )

        # Entity-level correlations bar chart
        if has_entity_data:
            entity_names = [f"Entity {i+1}" for i in range(len(entity_corrs))]

            fig.add_trace(
                go.Bar(
                    x=entity_names,
                    y=entity_corrs,
                    marker=dict(
                        color=entity_corrs,
                        colorscale="RdYlGn_r",  # Red (high) to Green (low)
                        cmin=0,
                        cmax=1,
                        colorbar=dict(title="Correlation", x=1.15),
                    ),
                    name="Correlation",
                    hovertemplate="<b>%{x}</b><br>Avg Correlation: %{y:.3f}<extra></extra>",
                ),
                row=1,
                col=2,
            )

            fig.update_xaxes(title_text="Entity", tickangle=-45, row=1, col=2)
            fig.update_yaxes(title_text="Avg Absolute Correlation", row=1, col=2)

        # Update layout
        title = kwargs.get("title", "Cross-Sectional Dependence Test (Pesaran CD)")

        fig.update_layout(
            title=dict(text=title, font=_get_font_config(self.theme, 4)),
            height=500,
            showlegend=False,
        )

        # Apply theme layout
        fig.update_layout(**self.theme.layout_config)

        return fig


# Export helper to avoid circular imports
__all__ = [
    "ACFPACFPlot",
    "UnitRootTestPlot",
    "CointegrationHeatmap",
    "CrossSectionalDependencePlot",
    "calculate_acf",
    "calculate_pacf",
    "ljung_box_test",
]
