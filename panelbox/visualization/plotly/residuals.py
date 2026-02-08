"""
Residual diagnostic chart implementations.

This module provides interactive Plotly charts for visualizing
residual diagnostics from panel data models.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from ..base import PlotlyChartBase
from ..registry import register_chart


@register_chart("residual_qq_plot")
class QQPlot(PlotlyChartBase):
    """
    Q-Q (Quantile-Quantile) plot for normality testing.

    Compares the distribution of residuals against a theoretical
    normal distribution. Points should fall along the diagonal line
    if residuals are normally distributed.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'residuals': array-like - Model residuals

        Optional:
        - 'standardized': bool - Use standardized residuals (default: True)
        - 'show_confidence': bool - Show confidence bands (default: True)
        - 'confidence_level': float - Confidence level (default: 0.95)

    Examples
    --------
    >>> chart = QQPlot()
    >>> chart.create(data={'residuals': model.resid})
    >>> html = chart.to_html()

    Notes
    -----
    - Deviations from the diagonal indicate non-normality
    - S-shaped patterns suggest heavy or light tails
    - Confidence bands help assess significance of deviations
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate Q-Q plot data."""
        super()._validate_data(data)

        if 'residuals' not in data:
            raise ValueError("Q-Q plot data must contain 'residuals'")

        residuals = np.asarray(data['residuals'])
        if len(residuals) == 0:
            raise ValueError("Residuals array cannot be empty")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        processed.setdefault('standardized', True)
        processed.setdefault('show_confidence', True)
        processed.setdefault('confidence_level', 0.95)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create Q-Q plot."""
        residuals = np.asarray(data['residuals'])
        standardized = data['standardized']
        show_confidence = data['show_confidence']
        confidence_level = data['confidence_level']

        # Standardize residuals if requested
        if standardized:
            residuals = (residuals - np.mean(residuals)) / np.std(residuals, ddof=1)

        # Sort residuals
        sorted_resid = np.sort(residuals)
        n = len(sorted_resid)

        # Theoretical quantiles (normal distribution)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

        # Create figure
        fig = go.Figure()

        # Add scatter plot of residuals
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_resid,
            mode='markers',
            name='Residuals',
            marker=dict(
                color=self.theme.color_scheme[0],
                size=6,
                opacity=0.6
            ),
            hovertemplate=(
                '<b>Theoretical Quantile:</b> %{x:.3f}<br>'
                '<b>Sample Quantile:</b> %{y:.3f}<br>'
                '<extra></extra>'
            )
        ))

        # Add diagonal reference line (y = x)
        line_min = min(theoretical_quantiles.min(), sorted_resid.min())
        line_max = max(theoretical_quantiles.max(), sorted_resid.max())

        fig.add_trace(go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Perfect Normal</b><extra></extra>'
        ))

        # Add confidence bands if requested
        if show_confidence:
            # Calculate confidence bands using order statistics
            alpha = 1 - confidence_level
            se = (1 / stats.norm.pdf(theoretical_quantiles)) * np.sqrt(
                (np.arange(1, n + 1) / (n + 1)) * (1 - np.arange(1, n + 1) / (n + 1)) / n
            )

            lower_band = theoretical_quantiles - stats.norm.ppf(1 - alpha / 2) * se
            upper_band = theoretical_quantiles + stats.norm.ppf(1 - alpha / 2) * se

            # Add confidence bands
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=upper_band,
                mode='lines',
                name=f'{confidence_level*100:.0f}% Confidence',
                line=dict(color='rgba(128, 128, 128, 0.3)', width=1),
                showlegend=True,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=lower_band,
                mode='lines',
                name='',
                line=dict(color='rgba(128, 128, 128, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(128, 128, 128, 0.1)',
                showlegend=False,
                hoverinfo='skip'
            ))

        # Update layout
        fig.update_layout(
            title=self.config.get('title', 'Q-Q Plot'),
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            hovermode='closest'
        )

        return fig


@register_chart("residual_vs_fitted")
class ResidualVsFittedPlot(PlotlyChartBase):
    """
    Residuals vs Fitted Values plot.

    Helps detect non-linearity, heteroskedasticity, and outliers.
    Residuals should be randomly scattered around zero with constant spread.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'fitted': array-like - Fitted values from model
        - 'residuals': array-like - Model residuals

        Optional:
        - 'add_lowess': bool - Add LOWESS smoothing line (default: True)
        - 'add_reference': bool - Add horizontal reference at y=0 (default: True)

    Examples
    --------
    >>> chart = ResidualVsFittedPlot()
    >>> chart.create(data={
    ...     'fitted': model.fittedvalues,
    ...     'residuals': model.resid
    ... })

    Notes
    -----
    - Horizontal band around zero suggests linearity assumption holds
    - Funnel shape indicates heteroskedasticity
    - Curved pattern suggests non-linearity
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate residual vs fitted data."""
        super()._validate_data(data)

        required = ['fitted', 'residuals']
        for field in required:
            if field not in data:
                raise ValueError(f"Residual vs fitted data must contain '{field}'")

        fitted = np.asarray(data['fitted'])
        residuals = np.asarray(data['residuals'])

        if len(fitted) != len(residuals):
            raise ValueError("fitted and residuals must have same length")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        processed.setdefault('add_lowess', True)
        processed.setdefault('add_reference', True)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create residual vs fitted plot."""
        fitted = np.asarray(data['fitted'])
        residuals = np.asarray(data['residuals'])
        add_lowess = data['add_lowess']
        add_reference = data['add_reference']

        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=fitted,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color=self.theme.color_scheme[0],
                size=6,
                opacity=0.6
            ),
            hovertemplate=(
                '<b>Fitted:</b> %{x:.3f}<br>'
                '<b>Residual:</b> %{y:.3f}<br>'
                '<extra></extra>'
            )
        ))

        # Add horizontal reference line at y=0
        if add_reference:
            fig.add_hline(
                y=0,
                line_dash='dash',
                line_color='red',
                line_width=2,
                annotation_text='y=0',
                annotation_position='right'
            )

        # Add LOWESS smoothing
        if add_lowess:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess

                # Compute LOWESS
                smoothed = lowess(residuals, fitted, frac=0.2)

                fig.add_trace(go.Scatter(
                    x=smoothed[:, 0],
                    y=smoothed[:, 1],
                    mode='lines',
                    name='LOWESS Smooth',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Smoothed Trend</b><extra></extra>'
                ))
            except ImportError:
                # statsmodels not available, skip LOWESS
                pass

        # Update layout
        fig.update_layout(
            title=self.config.get('title', 'Residuals vs Fitted Values'),
            xaxis_title='Fitted Values',
            yaxis_title='Residuals',
            hovermode='closest'
        )

        return fig


@register_chart("residual_scale_location")
class ScaleLocationPlot(PlotlyChartBase):
    """
    Scale-Location plot (Spread-Location plot).

    Checks homoscedasticity assumption. Plot shows square root of
    standardized residuals vs fitted values. Points should be
    randomly scattered with constant spread.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'fitted': array-like - Fitted values from model
        - 'residuals': array-like - Model residuals

        Optional:
        - 'add_lowess': bool - Add LOWESS smoothing line (default: True)

    Examples
    --------
    >>> chart = ScaleLocationPlot()
    >>> chart.create(data={
    ...     'fitted': model.fittedvalues,
    ...     'residuals': model.resid
    ... })

    Notes
    -----
    - Horizontal line suggests equal variance (homoscedasticity)
    - Upward/downward trend indicates heteroskedasticity
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate scale-location data."""
        super()._validate_data(data)

        required = ['fitted', 'residuals']
        for field in required:
            if field not in data:
                raise ValueError(f"Scale-location data must contain '{field}'")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        processed.setdefault('add_lowess', True)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create scale-location plot."""
        fitted = np.asarray(data['fitted'])
        residuals = np.asarray(data['residuals'])
        add_lowess = data['add_lowess']

        # Standardize residuals
        std_residuals = residuals / np.std(residuals, ddof=1)

        # Square root of absolute standardized residuals
        sqrt_abs_std_resid = np.sqrt(np.abs(std_residuals))

        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=fitted,
            y=sqrt_abs_std_resid,
            mode='markers',
            name='Residuals',
            marker=dict(
                color=self.theme.color_scheme[0],
                size=6,
                opacity=0.6
            ),
            hovertemplate=(
                '<b>Fitted:</b> %{x:.3f}<br>'
                '<b>√|Std. Residual|:</b> %{y:.3f}<br>'
                '<extra></extra>'
            )
        ))

        # Add LOWESS smoothing
        if add_lowess:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess

                smoothed = lowess(sqrt_abs_std_resid, fitted, frac=0.2)

                fig.add_trace(go.Scatter(
                    x=smoothed[:, 0],
                    y=smoothed[:, 1],
                    mode='lines',
                    name='LOWESS Smooth',
                    line=dict(color='red', width=2),
                    hovertemplate='<b>Smoothed Trend</b><extra></extra>'
                ))
            except ImportError:
                pass

        # Update layout
        fig.update_layout(
            title=self.config.get('title', 'Scale-Location Plot'),
            xaxis_title='Fitted Values',
            yaxis_title='√|Standardized Residuals|',
            hovermode='closest'
        )

        return fig


@register_chart("residual_vs_leverage")
class ResidualVsLeveragePlot(PlotlyChartBase):
    """
    Residuals vs Leverage plot (Influence plot).

    Identifies influential observations. Points outside Cook's distance
    contours have high influence on the model.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'residuals': array-like - Standardized residuals
        - 'leverage': array-like - Leverage (hat) values

        Optional:
        - 'cooks_d': array-like - Cook's distance values
        - 'show_contours': bool - Show Cook's distance contours (default: True)
        - 'labels': array-like - Observation labels for outliers

    Examples
    --------
    >>> chart = ResidualVsLeveragePlot()
    >>> chart.create(data={
    ...     'residuals': std_residuals,
    ...     'leverage': leverage_values,
    ...     'cooks_d': cooks_distance
    ... })

    Notes
    -----
    - High leverage + large residual = influential observation
    - Cook's distance > 0.5 suggests influential point
    - Cook's distance > 1.0 strongly suggests influential point
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate leverage data."""
        super()._validate_data(data)

        required = ['residuals', 'leverage']
        for field in required:
            if field not in data:
                raise ValueError(f"Leverage plot data must contain '{field}'")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        processed.setdefault('show_contours', True)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create residuals vs leverage plot."""
        residuals = np.asarray(data['residuals'])
        leverage = np.asarray(data['leverage'])
        show_contours = data['show_contours']
        cooks_d = data.get('cooks_d')
        labels = data.get('labels')

        fig = go.Figure()

        # Determine point colors based on Cook's distance if available
        if cooks_d is not None:
            cooks_d = np.asarray(cooks_d)
            # Color by Cook's distance
            colors = cooks_d
            colorbar_title = "Cook's D"
        else:
            colors = self.theme.color_scheme[0]
            colorbar_title = None

        # Add scatter plot
        scatter_kwargs = {
            'x': leverage,
            'y': residuals,
            'mode': 'markers',
            'name': 'Observations',
            'marker': dict(
                size=6,
                opacity=0.6,
                color=colors,
                colorscale='Reds' if cooks_d is not None else None,
                showscale=cooks_d is not None,
                colorbar=dict(title=colorbar_title) if cooks_d is not None else None
            ),
            'hovertemplate': (
                '<b>Leverage:</b> %{x:.3f}<br>'
                '<b>Std. Residual:</b> %{y:.3f}<br>'
                + ("<b>Cook's D:</b> %{marker.color:.3f}<br>" if cooks_d is not None else "")
                + '<extra></extra>'
            )
        }

        if labels is not None:
            scatter_kwargs['text'] = labels
            scatter_kwargs['textposition'] = 'top center'

        fig.add_trace(go.Scatter(**scatter_kwargs))

        # Add Cook's distance contours if requested
        if show_contours and cooks_d is not None:
            # Compute theoretical Cook's distance contours
            # Cook's D = (resid^2 / p) * (leverage / (1 - leverage)^2)
            # where p is number of parameters

            p = len(data.get('params', [3]))  # Estimate if not provided
            lev_range = np.linspace(0, leverage.max(), 100)

            for d_level in [0.5, 1.0]:
                # Solve for residual at each leverage for this Cook's D
                resid_contour = np.sqrt(d_level * p * (1 - lev_range)**2 / lev_range)
                resid_contour = np.where(np.isfinite(resid_contour), resid_contour, np.nan)

                # Add positive and negative contours
                for sign, name_suffix in [(1, ''), (-1, '')]:
                    fig.add_trace(go.Scatter(
                        x=lev_range,
                        y=sign * resid_contour,
                        mode='lines',
                        name=f"Cook's D = {d_level}" if sign == 1 else '',
                        line=dict(
                            color='red' if d_level == 0.5 else 'darkred',
                            width=1,
                            dash='dash'
                        ),
                        showlegend=(sign == 1),
                        hoverinfo='skip'
                    ))

        # Update layout
        fig.update_layout(
            title=self.config.get('title', 'Residuals vs Leverage'),
            xaxis_title='Leverage',
            yaxis_title='Standardized Residuals',
            hovermode='closest'
        )

        return fig


@register_chart("residual_timeseries")
class ResidualTimeSeriesPlot(PlotlyChartBase):
    """
    Residual time series plot.

    Visualizes residuals over time to detect serial correlation,
    temporal patterns, and structural breaks.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'residuals': array-like - Model residuals
        
        Optional:
        - 'time_index': array-like - Time index (default: range)
        - 'add_bands': bool - Add ±2σ bands (default: True)
        - 'entity_id': array-like - Entity identifiers for faceting

    Examples
    --------
    >>> chart = ResidualTimeSeriesPlot()
    >>> chart.create(data={
    ...     'residuals': model.resid,
    ...     'time_index': dates
    ... })

    Notes
    -----
    - Points should be randomly scattered around zero
    - Patterns or clusters suggest serial correlation
    - Points outside ±2σ bands may be outliers
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate time series data."""
        super()._validate_data(data)

        if 'residuals' not in data:
            raise ValueError("Time series data must contain 'residuals'")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        residuals = np.asarray(processed['residuals'])
        
        if 'time_index' not in processed:
            processed['time_index'] = np.arange(len(residuals))
        
        processed.setdefault('add_bands', True)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create residual time series plot."""
        residuals = np.asarray(data['residuals'])
        time_index = np.asarray(data['time_index'])
        add_bands = data['add_bands']

        fig = go.Figure()

        # Add residual line plot
        fig.add_trace(go.Scatter(
            x=time_index,
            y=residuals,
            mode='lines+markers',
            name='Residuals',
            line=dict(color=self.theme.color_scheme[0], width=1),
            marker=dict(size=4, opacity=0.6),
            hovertemplate=(
                '<b>Time:</b> %{x}<br>'
                '<b>Residual:</b> %{y:.3f}<br>'
                '<extra></extra>'
            )
        ))

        # Add zero reference line
        fig.add_hline(
            y=0,
            line_dash='dash',
            line_color='red',
            line_width=1,
            annotation_text='y=0'
        )

        # Add ±2σ bands
        if add_bands:
            sigma = np.std(residuals, ddof=1)
            
            fig.add_hline(
                y=2*sigma,
                line_dash='dot',
                line_color='gray',
                line_width=1,
                annotation_text='+2σ',
                annotation_position='right'
            )
            
            fig.add_hline(
                y=-2*sigma,
                line_dash='dot',
                line_color='gray',
                line_width=1,
                annotation_text='-2σ',
                annotation_position='right'
            )

        # Update layout
        fig.update_layout(
            title=self.config.get('title', 'Residuals Over Time'),
            xaxis_title='Time',
            yaxis_title='Residuals',
            hovermode='x'
        )

        return fig


@register_chart("residual_distribution")
class ResidualDistributionPlot(PlotlyChartBase):
    """
    Residual distribution plot (histogram + KDE).

    Shows the empirical distribution of residuals overlaid with
    theoretical normal distribution for normality assessment.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'residuals': array-like - Model residuals
        
        Optional:
        - 'bins': int - Number of histogram bins (default: 'auto')
        - 'show_kde': bool - Show KDE curve (default: True)
        - 'show_normal': bool - Show theoretical normal (default: True)

    Examples
    --------
    >>> chart = ResidualDistributionPlot()
    >>> chart.create(data={'residuals': model.resid})

    Notes
    -----
    - Bell-shaped distribution centered at zero suggests normality
    - Skewness or multiple peaks indicate non-normality
    - Heavy tails suggest outliers or non-normal errors
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate distribution data."""
        super()._validate_data(data)

        if 'residuals' not in data:
            raise ValueError("Distribution data must contain 'residuals'")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        processed.setdefault('bins', 'auto')
        processed.setdefault('show_kde', True)
        processed.setdefault('show_normal', True)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create residual distribution plot."""
        residuals = np.asarray(data['residuals'])
        bins = data['bins']
        show_kde = data['show_kde']
        show_normal = data['show_normal']

        fig = go.Figure()

        # Add histogram
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=bins if isinstance(bins, int) else None,
            name='Residuals',
            marker_color=self.theme.color_scheme[0],
            opacity=0.7,
            histnorm='probability density'
        ))

        # Add KDE if requested
        if show_kde:
            try:
                from scipy.stats import gaussian_kde
                
                kde = gaussian_kde(residuals)
                x_range = np.linspace(residuals.min(), residuals.max(), 100)
                kde_values = kde(x_range)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde_values,
                    mode='lines',
                    name='KDE',
                    line=dict(color='blue', width=2),
                    hovertemplate='<b>Density:</b> %{y:.4f}<extra></extra>'
                ))
            except ImportError:
                pass

        # Add theoretical normal distribution
        if show_normal:
            mean = np.mean(residuals)
            std = np.std(residuals, ddof=1)
            x_range = np.linspace(residuals.min(), residuals.max(), 100)
            normal_pdf = stats.norm.pdf(x_range, mean, std)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=normal_pdf,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>Normal PDF:</b> %{y:.4f}<extra></extra>'
            ))

        # Update layout
        fig.update_layout(
            title=self.config.get('title', 'Residual Distribution'),
            xaxis_title='Residuals',
            yaxis_title='Density',
            barmode='overlay',
            hovermode='x'
        )

        return fig


@register_chart("residual_partial_regression")
class PartialRegressionPlot(PlotlyChartBase):
    """
    Partial regression (added-variable) plot.

    Shows the relationship between dependent variable and a specific
    predictor after controlling for other predictors. Helps identify
    influence of individual variables and detect outliers.

    Data Format
    -----------
    data : dict
        Must contain:
        - 'y_resid': array-like - Residuals of y on other X's
        - 'x_resid': array-like - Residuals of focal X on other X's
        
        Optional:
        - 'variable_name': str - Name of focal variable
        - 'add_regression_line': bool - Add fitted line (default: True)
        - 'add_confidence': bool - Add confidence band (default: True)

    Examples
    --------
    >>> chart = PartialRegressionPlot()
    >>> chart.create(data={
    ...     'y_resid': y_residuals,
    ...     'x_resid': x_residuals,
    ...     'variable_name': 'GDP'
    ... })

    Notes
    -----
    - Slope represents the partial effect of the variable
    - Points far from the line may be influential
    - Used for assessing individual variable contribution
    """

    def _validate_data(self, data: Dict[str, Any]) -> None:
        """Validate partial regression data."""
        super()._validate_data(data)

        required = ['y_resid', 'x_resid']
        for field in required:
            if field not in data:
                raise ValueError(f"Partial regression data must contain '{field}'")

    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess data."""
        processed = data.copy()
        processed.setdefault('variable_name', 'X')
        processed.setdefault('add_regression_line', True)
        processed.setdefault('add_confidence', True)
        return processed

    def _create_figure(self, data: Dict[str, Any], **kwargs) -> go.Figure:
        """Create partial regression plot."""
        y_resid = np.asarray(data['y_resid'])
        x_resid = np.asarray(data['x_resid'])
        variable_name = data['variable_name']
        add_regression_line = data['add_regression_line']
        add_confidence = data['add_confidence']

        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=x_resid,
            y=y_resid,
            mode='markers',
            name='Observations',
            marker=dict(
                color=self.theme.color_scheme[0],
                size=6,
                opacity=0.6
            ),
            hovertemplate=(
                f'<b>{variable_name} Residual:</b> %{{x:.3f}}<br>'
                '<b>y Residual:</b> %{y:.3f}<br>'
                '<extra></extra>'
            )
        ))

        # Add regression line if requested
        if add_regression_line:
            # Compute OLS fit
            from numpy.polynomial import Polynomial
            
            # Fit line (y_resid = slope * x_resid)
            slope = np.sum(x_resid * y_resid) / np.sum(x_resid ** 2)
            
            x_range = np.array([x_resid.min(), x_resid.max()])
            y_fit = slope * x_range
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_fit,
                mode='lines',
                name='Regression Line',
                line=dict(color='red', width=2),
                hovertemplate=f'<b>Slope:</b> {slope:.4f}<extra></extra>'
            ))
            
            # Add confidence band if requested
            if add_confidence:
                # Compute standard error
                n = len(x_resid)
                residuals_fit = y_resid - slope * x_resid
                mse = np.sum(residuals_fit ** 2) / (n - 1)
                se = np.sqrt(mse / np.sum((x_resid - np.mean(x_resid)) ** 2))
                
                # 95% confidence interval
                t_val = stats.t.ppf(0.975, n - 1)
                margin = t_val * se * np.sqrt(1 + 1/n + (x_range - np.mean(x_resid))**2 / np.sum((x_resid - np.mean(x_resid))**2))
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_fit + margin,
                    mode='lines',
                    name='95% CI',
                    line=dict(color='rgba(128, 128, 128, 0.3)', width=1),
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_fit - margin,
                    mode='lines',
                    name='',
                    line=dict(color='rgba(128, 128, 128, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.1)',
                    showlegend=False,
                    hoverinfo='skip'
                ))

        # Update layout
        fig.update_layout(
            title=self.config.get('title', f'Partial Regression: {variable_name}'),
            xaxis_title=f'{variable_name} | Others',
            yaxis_title='y | Others',
            hovermode='closest'
        )

        return fig
