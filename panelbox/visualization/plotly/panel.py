"""
Panel-specific visualization charts for panel data analysis.

This module provides specialized charts for analyzing panel data structures,
including entity effects, time effects, between-within variance decomposition,
and panel structure analysis.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base import PlotlyChartBase
from ..registry import register_chart
from ..config.chart_config import ChartConfig


def _get_font_config(theme, size_modifier=0):
    """Helper to extract font configuration from theme."""
    return dict(
        family=theme.font_config.get('family', 'Arial'),
        size=theme.font_config.get('size', 12) + size_modifier,
        color=theme.font_config.get('color', '#2c3e50')
    )


@register_chart('panel_entity_effects')
class EntityEffectsPlot(PlotlyChartBase):
    """
    Visualize entity fixed or random effects in panel models.

    This chart displays the estimated effects for each entity (αᵢ) with
    confidence intervals, allowing identification of entities with significantly
    positive or negative effects.

    Parameters
    ----------
    data : dict or PanelResults
        Either a dictionary with keys:
        - 'entity_id': list of entity identifiers
        - 'effect': list of effect estimates
        - 'std_error': list of standard errors (optional)
        Or a PanelResults object with entity effects

    Examples
    --------
    >>> from panelbox.visualization import ChartFactory
    >>> data = {
    ...     'entity_id': ['Firm_A', 'Firm_B', 'Firm_C'],
    ...     'effect': [0.5, -0.3, 0.8],
    ...     'std_error': [0.1, 0.15, 0.12]
    ... }
    >>> chart = ChartFactory.create('panel_entity_effects', data=data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create entity effects visualization."""
        # Extract data
        if isinstance(data, dict):
            transformed_data = data
        else:
            transformed_data = self._prepare_data()

        entity_ids = transformed_data['entity_id']
        effects = transformed_data['effect']
        std_errors = transformed_data.get('std_error')

        # Get config
        sort_by = kwargs.get('sort_by', self.config.get('sort_by', 'magnitude'))
        show_confidence = kwargs.get('show_confidence', self.config.get('show_confidence', True))
        significance_level = kwargs.get('significance_level', self.config.get('significance_level', 0.05))
        max_entities = kwargs.get('max_entities', self.config.get('max_entities', None))

        # Sort data
        df = pd.DataFrame({
            'entity_id': entity_ids,
            'effect': effects,
            'std_error': std_errors if std_errors is not None else [0] * len(effects)
        })

        if sort_by == 'magnitude':
            df = df.sort_values('effect', ascending=True)
        elif sort_by == 'alphabetical':
            df = df.sort_values('entity_id')
        # 'significance' will be handled below

        # Limit entities if needed
        if max_entities and len(df) > max_entities:
            # Sample extremes for better representation
            n_half = max_entities // 2
            df_sorted = df.sort_values('effect')
            df = pd.concat([
                df_sorted.head(n_half),
                df_sorted.tail(max_entities - n_half)
            ]).sort_values('effect', ascending=True)

        # Calculate confidence intervals
        if show_confidence and std_errors is not None:
            z_score = 1.96 if significance_level == 0.05 else 2.576  # 95% or 99%
            df['ci_lower'] = df['effect'] - z_score * df['std_error']
            df['ci_upper'] = df['effect'] + z_score * df['std_error']
            df['significant'] = ~((df['ci_lower'] <= 0) & (df['ci_upper'] >= 0))
        else:
            df['significant'] = False

        # Color by significance
        colors = [
            self.theme.success_color if sig else self.theme.get_color(5)  # Brown/neutral
            for sig in df['significant']
        ]

        # Create figure
        fig = go.Figure()

        # Add error bars if available
        if show_confidence and std_errors is not None:
            error_x = dict(
                type='data',
                symmetric=False,
                array=df['ci_upper'] - df['effect'],
                arrayminus=df['effect'] - df['ci_lower'],
                color='rgba(0,0,0,0.3)',
                thickness=1.5
            )
        else:
            error_x = None

        # Add bars
        fig.add_trace(go.Bar(
            y=df['entity_id'],
            x=df['effect'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            error_x=error_x,
            text=[f"{e:.3f}" for e in df['effect']],
            textposition='auto',
            hovertemplate=(
                '<b>%{y}</b><br>' +
                'Effect: %{x:.4f}<br>' +
                '<extra></extra>'
            )
        ))

        # Add reference line at zero
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="rgba(0,0,0,0.4)",
            line_width=2,
            annotation_text="Zero Effect",
            annotation_position="top"
        )

        # Update layout
        title = self.config.get('title', 'Entity Fixed Effects')
        fig.update_layout(
            title=dict(text=title, font=_get_font_config(self.theme, 4)),
            xaxis_title='Effect Size',
            yaxis_title='Entity',
            showlegend=False,
            hovermode='closest'
        )
        # Apply theme layout (may override above)
        fig.update_layout(**self.theme.layout_config)

        return fig

    def _prepare_data(self) -> Dict[str, Any]:
        """Prepare data for plotting."""
        if isinstance(self.data, dict):
            # Already in correct format
            return self.data
        else:
            # Assume PanelResults object
            # This will be implemented when we integrate with PanelResults
            try:
                from ...transformers.panel import PanelDataTransformer
                return PanelDataTransformer.extract_entity_effects(self.data)
            except (ImportError, AttributeError):
                # Fallback: treat as dict
                return self.data


@register_chart('panel_time_effects')
class TimeEffectsPlot(PlotlyChartBase):
    """
    Visualize time effects in panel models.

    This chart displays the estimated time effects (λₜ) as a line chart
    with confidence bands, useful for identifying temporal patterns and
    structural breaks.

    Parameters
    ----------
    data : dict or PanelResults
        Either a dictionary with keys:
        - 'time': list of time periods
        - 'effect': list of effect estimates
        - 'std_error': list of standard errors (optional)
        Or a PanelResults object with time effects

    Examples
    --------
    >>> data = {
    ...     'time': [2000, 2001, 2002, 2003],
    ...     'effect': [0.1, 0.3, -0.2, 0.5],
    ...     'std_error': [0.05, 0.06, 0.04, 0.07]
    ... }
    >>> chart = ChartFactory.create('panel_time_effects', data=data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create time effects visualization."""
        # Extract data
        if isinstance(data, dict):
            transformed_data = data
        else:
            transformed_data = self._prepare_data()

        time_periods = transformed_data['time']
        effects = transformed_data['effect']
        std_errors = transformed_data.get('std_error')

        # Get config
        show_confidence = kwargs.get('show_confidence', self.config.get('show_confidence', True))
        highlight_significant = kwargs.get('highlight_significant', self.config.get('highlight_significant', True))
        significance_level = kwargs.get('significance_level', self.config.get('significance_level', 0.05))

        # Create DataFrame
        df = pd.DataFrame({
            'time': time_periods,
            'effect': effects,
            'std_error': std_errors if std_errors is not None else [0] * len(effects)
        }).sort_values('time')

        # Calculate confidence intervals
        if show_confidence and std_errors is not None:
            z_score = 1.96 if significance_level == 0.05 else 2.576
            df['ci_lower'] = df['effect'] - z_score * df['std_error']
            df['ci_upper'] = df['effect'] + z_score * df['std_error']
        else:
            df['ci_lower'] = df['effect']
            df['ci_upper'] = df['effect']

        # Create figure
        fig = go.Figure()

        # Add confidence band
        if show_confidence and std_errors is not None:
            fig.add_trace(go.Scatter(
                x=list(df['time']) + list(df['time'][::-1]),
                y=list(df['ci_upper']) + list(df['ci_lower'][::-1]),
                fill='toself',
                fillcolor='rgba(100, 100, 100, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{int((1-significance_level)*100)}% CI',
                hoverinfo='skip',
                showlegend=True
            ))

        # Add main line
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['effect'],
            mode='lines+markers',
            name='Time Effect',
            line=dict(color=self.theme.get_color(0), width=3),  # Primary color
            marker=dict(size=8, color=self.theme.get_color(0)),
            hovertemplate=(
                '<b>Time: %{x}</b><br>' +
                'Effect: %{y:.4f}<br>' +
                '<extra></extra>'
            )
        ))

        # Add reference line at zero
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="rgba(0,0,0,0.4)",
            line_width=2,
            annotation_text="Zero Effect",
            annotation_position="right"
        )

        # Highlight significant periods
        if highlight_significant and std_errors is not None:
            significant_periods = df[~((df['ci_lower'] <= 0) & (df['ci_upper'] >= 0))]
            if len(significant_periods) > 0:
                fig.add_trace(go.Scatter(
                    x=significant_periods['time'],
                    y=significant_periods['effect'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=self.theme.warning_color,
                        symbol='star',
                        line=dict(color='white', width=1)
                    ),
                    name='Significant',
                    hoverinfo='skip',
                    showlegend=True
                ))

        # Update layout
        title = self.config.get('title', 'Time Fixed Effects')
        fig.update_layout(
            title=dict(text=title, font=_get_font_config(self.theme, 4)),
            xaxis_title='Time Period',
            yaxis_title='Effect Size',
            hovermode='x unified'
        )
        fig.update_layout(**self.theme.layout_config)

        return fig

    def _prepare_data(self) -> Dict[str, Any]:
        """Prepare data for plotting."""
        if isinstance(self.data, dict):
            return self.data
        else:
            try:
                from ...transformers.panel import PanelDataTransformer
                return PanelDataTransformer.extract_time_effects(self.data)
            except (ImportError, AttributeError):
                return self.data


@register_chart('panel_between_within')
class BetweenWithinPlot(PlotlyChartBase):
    """
    Visualize between and within variance decomposition.

    This chart decomposes the total variance into between-entity and
    within-entity components, helping to understand the relative importance
    of cross-sectional vs. time-series variation.

    Formulas:
    - Between Variation: σ²_b = Var(ȳᵢ)
    - Within Variation: σ²_w = Var(yᵢₜ - ȳᵢ)
    - Total Variation: σ²_t = σ²_b + σ²_w

    Parameters
    ----------
    data : dict or PanelData
        Either a dictionary with keys:
        - 'variables': list of variable names
        - 'between_var': list of between variances
        - 'within_var': list of within variances
        Or a PanelData object

    Examples
    --------
    >>> data = {
    ...     'variables': ['wage', 'education', 'experience'],
    ...     'between_var': [10.5, 5.2, 8.3],
    ...     'within_var': [3.2, 1.8, 2.1]
    ... }
    >>> chart = ChartFactory.create('panel_between_within', data=data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create between-within variance decomposition chart."""
        # Extract data
        if isinstance(data, dict):
            transformed_data = data
        else:
            transformed_data = self._prepare_data()

        variables = transformed_data['variables']
        between_var = np.array(transformed_data['between_var'])
        within_var = np.array(transformed_data['within_var'])

        # Get config - use 'style' instead of 'chart_type' to avoid conflict
        style = kwargs.get('chart_type', kwargs.get('style', self.config.get('chart_type', self.config.get('style', 'stacked'))))
        show_percentages = kwargs.get('show_percentages', self.config.get('show_percentages', True))

        # Calculate percentages
        total_var = between_var + within_var
        between_pct = (between_var / total_var * 100)
        within_pct = (within_var / total_var * 100)

        # Create figure based on style
        if style == 'stacked':
            fig = self._create_stacked_chart(
                variables, between_var, within_var,
                between_pct, within_pct, show_percentages
            )
        elif style == 'side_by_side':
            fig = self._create_side_by_side_chart(
                variables, between_var, within_var,
                between_pct, within_pct, show_percentages
            )
        else:  # scatter
            fig = self._create_scatter_chart(
                variables, between_var, within_var,
                between_pct, within_pct
            )

        # Update layout
        title = self.config.get('title', 'Between-Within Variance Decomposition')
        fig.update_layout(title=dict(text=title, font=_get_font_config(self.theme, 4)))
        fig.update_layout(**self.theme.layout_config)

        return fig

    def _create_stacked_chart(self, variables, between_var, within_var,
                             between_pct, within_pct, show_percentages):
        """Create stacked bar chart."""
        fig = go.Figure()

        # Between variance
        text_between = [f"{p:.1f}%" if show_percentages else f"{v:.2f}"
                       for p, v in zip(between_pct, between_var)]

        fig.add_trace(go.Bar(
            x=variables,
            y=between_var,
            name='Between Variance',
            marker_color=self.theme.get_color(0),  # Primary
            text=text_between,
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>Between: %{y:.4f}<extra></extra>'
        ))

        # Within variance
        text_within = [f"{p:.1f}%" if show_percentages else f"{v:.2f}"
                      for p, v in zip(within_pct, within_var)]

        fig.add_trace(go.Bar(
            x=variables,
            y=within_var,
            name='Within Variance',
            marker_color=self.theme.get_color(1),  # Secondary
            text=text_within,
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>Within: %{y:.4f}<extra></extra>'
        ))

        fig.update_layout(
            barmode='stack',
            xaxis_title='Variable',
            yaxis_title='Variance'
        )

        return fig

    def _create_side_by_side_chart(self, variables, between_var, within_var,
                                   between_pct, within_pct, show_percentages):
        """Create side-by-side bar chart."""
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=variables,
            y=between_var,
            name='Between Variance',
            marker_color=self.theme.get_color(0),  # Primary
            text=[f"{p:.1f}%" if show_percentages else ""
                  for p in between_pct],
            textposition='outside',
        ))

        fig.add_trace(go.Bar(
            x=variables,
            y=within_var,
            name='Within Variance',
            marker_color=self.theme.get_color(1),  # Secondary
            text=[f"{p:.1f}%" if show_percentages else ""
                  for p in within_pct],
            textposition='outside',
        ))

        fig.update_layout(
            barmode='group',
            xaxis_title='Variable',
            yaxis_title='Variance'
        )

        return fig

    def _create_scatter_chart(self, variables, between_var, within_var,
                             between_pct, within_pct):
        """Create scatter plot of between vs within variance."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=between_var,
            y=within_var,
            mode='markers+text',
            marker=dict(
                size=15,
                color=self.theme.get_color(0),  # Primary
                line=dict(color='white', width=2)
            ),
            text=variables,
            textposition='top center',
            hovertemplate=(
                '<b>%{text}</b><br>' +
                'Between: %{x:.4f}<br>' +
                'Within: %{y:.4f}<br>' +
                '<extra></extra>'
            )
        ))

        # Add diagonal reference line
        max_val = max(max(between_var), max(within_var))
        fig.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', dash='dash'),
            name='Equal Variance',
            showlegend=True
        ))

        fig.update_layout(
            xaxis_title='Between Variance',
            yaxis_title='Within Variance'
        )

        return fig

    def _prepare_data(self) -> Dict[str, Any]:
        """Prepare data for plotting."""
        if isinstance(self.data, dict):
            return self.data
        else:
            try:
                from ...transformers.panel import PanelDataTransformer
                return PanelDataTransformer.calculate_between_within(self.data)
            except (ImportError, AttributeError):
                return self.data


@register_chart('panel_structure')
class PanelStructurePlot(PlotlyChartBase):
    """
    Visualize panel data structure and balance.

    This chart creates a heatmap showing the presence/absence of observations
    for each entity-time combination, along with balance statistics.

    Parameters
    ----------
    data : dict or PanelData
        Either a dictionary with keys:
        - 'entities': list of entity IDs
        - 'time_periods': list of time periods
        - 'presence_matrix': 2D array (entities × time) of 0/1
        Or a PanelData object

    Examples
    --------
    >>> data = {
    ...     'entities': ['A', 'B', 'C'],
    ...     'time_periods': [2000, 2001, 2002],
    ...     'presence_matrix': [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    ... }
    >>> chart = ChartFactory.create('panel_structure', data=data)
    >>> chart.show()
    """

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create panel structure visualization."""
        # Extract data
        if isinstance(data, dict):
            transformed_data = data
        else:
            transformed_data = self._prepare_data()

        entities = transformed_data['entities']
        time_periods = transformed_data['time_periods']
        presence_matrix = np.array(transformed_data['presence_matrix'])

        # Get config
        show_statistics = kwargs.get('show_statistics', self.config.get('show_statistics', True))
        highlight_complete = kwargs.get('highlight_complete', self.config.get('highlight_complete', True))

        # Calculate statistics
        n_entities = len(entities)
        n_periods = len(time_periods)
        total_cells = n_entities * n_periods
        present_cells = np.sum(presence_matrix)
        balance_pct = (present_cells / total_cells) * 100

        # Identify balanced entities (all periods present)
        entity_completeness = np.sum(presence_matrix, axis=1) == n_periods
        complete_entities = [e for e, c in zip(entities, entity_completeness) if c]

        # Create heatmap
        fig = go.Figure()

        # Custom colorscale: missing = red, present = green
        colorscale = [
            [0, self.theme.danger_color],  # Missing
            [1, self.theme.success_color]  # Present
        ]

        fig.add_trace(go.Heatmap(
            z=presence_matrix,
            x=time_periods,
            y=entities,
            colorscale=colorscale,
            showscale=True,
            hovertemplate=(
                '<b>Entity: %{y}</b><br>' +
                'Time: %{x}<br>' +
                'Status: %{z}<br>' +
                '<extra></extra>'
            ),
            colorbar=dict(
                title='Observation',
                tickvals=[0, 1],
                ticktext=['Missing', 'Present']
            )
        ))

        # Add annotations for statistics
        annotations = []
        if show_statistics:
            stats_text = (
                f"<b>Panel Structure Statistics</b><br>"
                f"Entities: {n_entities}<br>"
                f"Periods: {n_periods}<br>"
                f"Balance: {balance_pct:.1f}%<br>"
                f"Complete Entities: {len(complete_entities)}/{n_entities}"
            )

            annotations.append(dict(
                text=stats_text,
                xref="paper",
                yref="paper",
                x=1.15,
                y=0.5,
                showarrow=False,
                align="left",
                font=dict(size=10),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor='#ddd',
                borderwidth=1
            ))

        # Update layout
        title = self.config.get('title', 'Panel Data Structure')
        fig.update_layout(
            title=dict(text=title, font=_get_font_config(self.theme, 4)),
            xaxis_title='Time Period',
            yaxis_title='Entity',
            annotations=annotations
        )
        fig.update_layout(**self.theme.layout_config)

        return fig

    def _prepare_data(self) -> Dict[str, Any]:
        """Prepare data for plotting."""
        if isinstance(self.data, dict):
            return self.data
        else:
            try:
                from ...transformers.panel import PanelDataTransformer
                return PanelDataTransformer.analyze_panel_structure(self.data)
            except (ImportError, AttributeError):
                return self.data
