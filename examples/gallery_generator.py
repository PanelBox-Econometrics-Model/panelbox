"""
Chart Gallery Generator for PanelBox.

This script generates a comprehensive gallery of all available charts
with synthetic data examples. Can be run as a script or imported into
a Jupyter notebook.

Usage:
    As script:
    $ python examples/gallery_generator.py

    In notebook:
    >>> from gallery_generator import generate_gallery
    >>> generate_gallery()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


class ChartGallery:
    """Generator for comprehensive chart gallery."""

    def __init__(self):
        """Initialize gallery generator."""
        self.examples = []

    def generate_all(self) -> List[Dict[str, Any]]:
        """
        Generate all chart examples.

        Returns
        -------
        list of dict
            List of chart examples with metadata
        """
        print("="*80)
        print("PanelBox Chart Gallery Generator")
        print("="*80)
        print("\nGenerating examples for all 32+ chart types...\n")

        # Generate examples by category
        self._generate_residual_diagnostics()
        self._generate_validation_charts()
        self._generate_comparison_charts()
        self._generate_panel_specific()
        self._generate_econometric_tests()
        self._generate_distribution_charts()
        self._generate_correlation_charts()
        self._generate_timeseries_charts()

        print(f"\n✅ Generated {len(self.examples)} chart examples")
        return self.examples

    def _add_example(
        self,
        category: str,
        name: str,
        chart_type: str,
        description: str,
        code: str,
        data_generator: callable
    ):
        """Add example to gallery."""
        self.examples.append({
            'category': category,
            'name': name,
            'chart_type': chart_type,
            'description': description,
            'code': code,
            'data': data_generator()
        })
        print(f"  ✓ {category}: {name}")

    # ========================================================================
    # RESIDUAL DIAGNOSTICS
    # ========================================================================

    def _generate_residual_diagnostics(self):
        """Generate residual diagnostic examples."""
        print("\n1. Residual Diagnostics")
        print("-" * 60)

        # QQ Plot
        self._add_example(
            category="Residual Diagnostics",
            name="Q-Q Plot",
            chart_type="residual_qq_plot",
            description="Assess normality of residuals using quantile-quantile plot",
            code="""from panelbox.visualization import create_residual_diagnostics

# Generate residuals (normally distributed)
residuals = np.random.randn(100)

# Create diagnostics
diagnostics = create_residual_diagnostics(
    {'residuals': residuals, 'fitted': np.random.randn(100)},
    charts=['qq_plot'],
    theme='academic'
)

diagnostics['qq_plot'].show()""",
            data_generator=lambda: {
                'residuals': np.random.randn(100),
                'fitted': np.random.randn(100) * 2
            }
        )

        # Residual vs Fitted
        self._add_example(
            category="Residual Diagnostics",
            name="Residual vs Fitted",
            chart_type="residual_vs_fitted",
            description="Detect heteroskedasticity and non-linear patterns",
            code="""from panelbox.visualization import create_residual_diagnostics

# Create diagnostics
diagnostics = create_residual_diagnostics(
    results,
    charts=['residual_vs_fitted'],
    theme='professional'
)

diagnostics['residual_vs_fitted'].show()""",
            data_generator=lambda: {
                'residuals': np.random.randn(150) * (1 + np.linspace(0, 1, 150)),
                'fitted': np.linspace(-2, 2, 150)
            }
        )

        # Scale-Location
        self._add_example(
            category="Residual Diagnostics",
            name="Scale-Location Plot",
            chart_type="scale_location",
            description="Check homoskedasticity assumption",
            code="""diagnostics = create_residual_diagnostics(
    results,
    charts=['scale_location'],
    theme='academic'
)

diagnostics['scale_location'].show()""",
            data_generator=lambda: {
                'residuals': np.random.randn(120),
                'fitted': np.random.randn(120) * 1.5
            }
        )

        # ACF/PACF
        self._add_example(
            category="Residual Diagnostics",
            name="ACF/PACF Plot",
            chart_type="acf_pacf_plot",
            description="Detect serial correlation in residuals",
            code="""from panelbox.visualization import create_acf_pacf_plot

# Generate AR(1) process
residuals = np.zeros(200)
residuals[0] = np.random.randn()
for t in range(1, 200):
    residuals[t] = 0.7 * residuals[t-1] + np.random.randn()

chart = create_acf_pacf_plot(
    residuals,
    max_lags=20,
    confidence_level=0.95,
    show_ljung_box=True,
    theme='academic'
)

chart.show()""",
            data_generator=lambda: self._generate_ar1_process(200, 0.7)
        )

    # ========================================================================
    # PANEL-SPECIFIC CHARTS
    # ========================================================================

    def _generate_panel_specific(self):
        """Generate panel-specific chart examples."""
        print("\n4. Panel-Specific Charts")
        print("-" * 60)

        # Entity Effects
        self._add_example(
            category="Panel-Specific",
            name="Entity Effects Plot",
            chart_type="entity_effects_plot",
            description="Visualize entity-specific fixed effects",
            code="""from panelbox.visualization import create_entity_effects_plot

# Entity effects data
data = {
    'entity_id': ['Firm A', 'Firm B', 'Firm C', 'Firm D', 'Firm E'],
    'effect': [0.5, -0.3, 0.8, -0.1, 0.2],
    'std_error': [0.15, 0.12, 0.18, 0.10, 0.14]
}

chart = create_entity_effects_plot(
    data,
    theme='professional',
    sort_by='effect'
)

chart.show()""",
            data_generator=lambda: {
                'entity_id': [f'Entity_{i}' for i in range(1, 11)],
                'effect': np.random.randn(10) * 0.5,
                'std_error': np.random.uniform(0.1, 0.2, 10)
            }
        )

        # Time Effects
        self._add_example(
            category="Panel-Specific",
            name="Time Effects Plot",
            chart_type="time_effects_plot",
            description="Visualize time-period fixed effects",
            code="""from panelbox.visualization import create_time_effects_plot

# Time effects data
data = {
    'time_id': list(range(2010, 2021)),
    'effect': np.cumsum(np.random.randn(11) * 0.1),
    'std_error': np.random.uniform(0.05, 0.15, 11)
}

chart = create_time_effects_plot(
    data,
    theme='academic',
    show_trend=True
)

chart.show()""",
            data_generator=lambda: {
                'time_id': list(range(2010, 2021)),
                'effect': np.cumsum(np.random.randn(11) * 0.1),
                'std_error': np.random.uniform(0.05, 0.15, 11)
            }
        )

        # Between-Within
        self._add_example(
            category="Panel-Specific",
            name="Between-Within Variation",
            chart_type="between_within_plot",
            description="Decompose variance into between and within components",
            code="""from panelbox.visualization import create_between_within_plot

# Generate panel data
panel_data = pd.DataFrame({
    'entity': np.repeat(range(1, 51), 20),
    'time': np.tile(range(1, 21), 50),
    'capital': np.random.randn(1000),
    'labor': np.random.randn(1000),
    'output': np.random.randn(1000)
})

chart = create_between_within_plot(
    panel_data.set_index(['entity', 'time']),
    variables=['capital', 'labor', 'output'],
    theme='professional',
    style='stacked'
)

chart.show()""",
            data_generator=lambda: {
                'variables': ['Variable 1', 'Variable 2', 'Variable 3'],
                'between_var': [2.5, 3.0, 1.5],
                'within_var': [1.5, 2.0, 2.5],
                'total_var': [4.0, 5.0, 4.0]
            }
        )

        # Panel Structure
        self._add_example(
            category="Panel-Specific",
            name="Panel Structure Plot",
            chart_type="panel_structure_plot",
            description="Visualize panel balance and missing data patterns",
            code="""from panelbox.visualization import create_panel_structure_plot

# Panel data
panel_data = pd.DataFrame({
    'firm': np.repeat(range(1, 21), 30),
    'year': np.tile(range(1, 31), 20),
    'value': np.random.randn(600)
})

chart = create_panel_structure_plot(
    panel_data.set_index(['firm', 'year']),
    theme='professional'
)

chart.show()""",
            data_generator=lambda: self._generate_panel_structure_data(20, 30, 0.15)
        )

    # ========================================================================
    # ECONOMETRIC TESTS
    # ========================================================================

    def _generate_econometric_tests(self):
        """Generate econometric test chart examples."""
        print("\n5. Econometric Tests")
        print("-" * 60)

        # Unit Root Test
        self._add_example(
            category="Econometric Tests",
            name="Unit Root Test Plot",
            chart_type="unit_root_test_plot",
            description="Visualize stationarity test results",
            code="""from panelbox.visualization import create_unit_root_test_plot

results = {
    'test_names': ['ADF', 'PP', 'KPSS', 'DF-GLS'],
    'test_stats': [-3.5, -3.8, 0.3, -2.9],
    'critical_values': {'1%': -3.96, '5%': -3.41, '10%': -3.13},
    'pvalues': [0.008, 0.003, 0.15, 0.04]
}

chart = create_unit_root_test_plot(
    results,
    theme='professional'
)

chart.show()""",
            data_generator=lambda: {
                'test_names': ['ADF', 'PP', 'KPSS', 'DF-GLS'],
                'test_stats': [-3.5, -3.8, 0.3, -2.9],
                'critical_values': {'1%': -3.96, '5%': -3.41, '10%': -3.13},
                'pvalues': [0.008, 0.003, 0.15, 0.04]
            }
        )

        # Cointegration Heatmap
        self._add_example(
            category="Econometric Tests",
            name="Cointegration Heatmap",
            chart_type="cointegration_heatmap",
            description="Visualize pairwise cointegration relationships",
            code="""from panelbox.visualization import create_cointegration_heatmap

results = {
    'variables': ['GDP', 'Consumption', 'Investment', 'Exports'],
    'pvalues': [
        [1.0, 0.02, 0.15, 0.08],
        [0.02, 1.0, 0.08, 0.12],
        [0.15, 0.08, 1.0, 0.05],
        [0.08, 0.12, 0.05, 1.0]
    ],
    'test_name': 'Engle-Granger'
}

chart = create_cointegration_heatmap(
    results,
    theme='academic'
)

chart.show()""",
            data_generator=lambda: self._generate_cointegration_data(5)
        )

        # Cross-Sectional Dependence
        self._add_example(
            category="Econometric Tests",
            name="Cross-Sectional Dependence",
            chart_type="cross_sectional_dependence_plot",
            description="Pesaran CD test visualization",
            code="""from panelbox.visualization import create_cross_sectional_dependence_plot

results = {
    'cd_statistic': 3.45,
    'pvalue': 0.003,
    'avg_correlation': 0.28,
    'entity_correlations': [0.15, 0.32, 0.45, 0.21, 0.38, 0.29]
}

chart = create_cross_sectional_dependence_plot(
    results,
    theme='professional'
)

chart.show()""",
            data_generator=lambda: {
                'cd_statistic': 3.45,
                'pvalue': 0.003,
                'avg_correlation': 0.28,
                'entity_correlations': np.random.uniform(0.1, 0.5, 10).tolist()
            }
        )

    # ========================================================================
    # VALIDATION CHARTS
    # ========================================================================

    def _generate_validation_charts(self):
        """Generate validation chart examples."""
        print("\n2. Validation Charts")
        print("-" * 60)

        self._add_example(
            category="Validation",
            name="Validation Dashboard",
            chart_type="validation_dashboard",
            description="Complete validation overview",
            code="""from panelbox.visualization import create_validation_charts

charts = create_validation_charts(
    validation_report,
    theme='professional'
)

charts['dashboard'].show()""",
            data_generator=lambda: {}  # Placeholder
        )

    # ========================================================================
    # COMPARISON CHARTS
    # ========================================================================

    def _generate_comparison_charts(self):
        """Generate comparison chart examples."""
        print("\n3. Model Comparison")
        print("-" * 60)

        self._add_example(
            category="Model Comparison",
            name="Coefficient Comparison",
            chart_type="coefficient_comparison",
            description="Compare coefficients across models",
            code="""from panelbox.visualization import create_comparison_charts

charts = create_comparison_charts(
    [ols_results, fe_results, re_results],
    model_names=['OLS', 'Fixed Effects', 'Random Effects'],
    theme='professional'
)

charts['coefficients'].show()""",
            data_generator=lambda: {}  # Placeholder
        )

    # ========================================================================
    # DISTRIBUTION CHARTS
    # ========================================================================

    def _generate_distribution_charts(self):
        """Generate distribution chart examples."""
        print("\n6. Distribution Charts")
        print("-" * 60)

        # Histogram
        self._add_example(
            category="Distribution",
            name="Histogram",
            chart_type="histogram",
            description="Visualize data distribution",
            code="""from panelbox.visualization import ChartFactory

data = {'values': np.random.randn(500)}

chart = ChartFactory.create(
    'histogram',
    data=data,
    theme='professional'
)

chart.show()""",
            data_generator=lambda: {'values': np.random.randn(500)}
        )

    # ========================================================================
    # CORRELATION CHARTS
    # ========================================================================

    def _generate_correlation_charts(self):
        """Generate correlation chart examples."""
        print("\n7. Correlation Charts")
        print("-" * 60)

        # Correlation Heatmap
        self._add_example(
            category="Correlation",
            name="Correlation Heatmap",
            chart_type="correlation_heatmap",
            description="Visualize correlation matrix",
            code="""from panelbox.visualization import ChartFactory

# Generate correlation matrix
df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
corr_matrix = df.corr()

data = {
    'correlation_matrix': corr_matrix.values.tolist(),
    'variables': corr_matrix.columns.tolist()
}

chart = ChartFactory.create(
    'correlation_heatmap',
    data=data,
    theme='academic'
)

chart.show()""",
            data_generator=lambda: self._generate_correlation_data(6)
        )

    # ========================================================================
    # TIME SERIES CHARTS
    # ========================================================================

    def _generate_timeseries_charts(self):
        """Generate time series chart examples."""
        print("\n8. Time Series Charts")
        print("-" * 60)

        # Panel Time Series
        self._add_example(
            category="Time Series",
            name="Panel Time Series",
            chart_type="panel_timeseries",
            description="Visualize time series across entities",
            code="""from panelbox.visualization import ChartFactory

# Generate panel time series
dates = pd.date_range('2010-01-01', periods=100, freq='M')
entities = ['Entity A', 'Entity B', 'Entity C']

data = {
    'time': dates.tolist() * 3,
    'entity': np.repeat(entities, 100).tolist(),
    'value': np.random.randn(300).cumsum()
}

chart = ChartFactory.create(
    'panel_timeseries',
    data=pd.DataFrame(data),
    theme='professional'
)

chart.show()""",
            data_generator=lambda: {}  # Placeholder
        )

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    def _generate_ar1_process(self, n: int, rho: float) -> Dict:
        """Generate AR(1) process."""
        residuals = np.zeros(n)
        residuals[0] = np.random.randn()
        for t in range(1, n):
            residuals[t] = rho * residuals[t-1] + np.random.randn()
        return {'residuals': residuals}

    def _generate_panel_structure_data(self, n_entities: int, n_periods: int, missing_rate: float) -> Dict:
        """Generate panel structure data."""
        entities = [f'E{i}' for i in range(1, n_entities + 1)]
        periods = [f'T{i}' for i in range(1, n_periods + 1)]
        presence = np.random.choice([0, 1], size=(n_entities, n_periods), p=[missing_rate, 1-missing_rate])

        return {
            'entities': entities,
            'periods': periods,
            'presence_matrix': presence.tolist(),
            'n_obs_per_entity': presence.sum(axis=1).tolist(),
            'n_obs_per_period': presence.sum(axis=0).tolist()
        }

    def _generate_cointegration_data(self, n_vars: int) -> Dict:
        """Generate cointegration test data."""
        pvalues = np.random.uniform(0.01, 0.5, (n_vars, n_vars))
        np.fill_diagonal(pvalues, 1.0)
        pvalues = (pvalues + pvalues.T) / 2  # Make symmetric

        return {
            'variables': [f'Variable_{i+1}' for i in range(n_vars)],
            'pvalues': pvalues.tolist(),
            'test_name': 'Engle-Granger'
        }

    def _generate_correlation_data(self, n_vars: int) -> Dict:
        """Generate correlation matrix data."""
        # Generate random correlation matrix
        data = np.random.randn(100, n_vars)
        df = pd.DataFrame(data, columns=[f'Var{i+1}' for i in range(n_vars)])
        corr = df.corr()

        return {
            'correlation_matrix': corr.values.tolist(),
            'variables': corr.columns.tolist()
        }

    def export_to_markdown(self, output_file: str = 'examples/CHART_GALLERY.md'):
        """Export gallery to markdown format."""
        from pathlib import Path

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# PanelBox Chart Gallery\n\n")
            f.write("Complete reference of all available chart types with code examples.\n\n")
            f.write("---\n\n")

            current_category = None

            for example in self.examples:
                # Category header
                if example['category'] != current_category:
                    current_category = example['category']
                    f.write(f"## {current_category}\n\n")

                # Chart section
                f.write(f"### {example['name']}\n\n")
                f.write(f"**Chart Type**: `{example['chart_type']}`\n\n")
                f.write(f"{example['description']}\n\n")
                f.write("```python\n")
                f.write(example['code'])
                f.write("\n```\n\n")
                f.write("---\n\n")

        print(f"\n✅ Gallery exported to: {output_path}")


def generate_gallery():
    """Main function to generate gallery."""
    gallery = ChartGallery()
    examples = gallery.generate_all()
    gallery.export_to_markdown()
    return gallery


if __name__ == '__main__':
    generate_gallery()
