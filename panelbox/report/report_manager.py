"""
Report Manager for PanelBox.

Main orchestrator for report generation across all report types.
"""

import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .asset_manager import AssetManager
from .css_manager import CSSManager
from .template_manager import TemplateManager

# Version info
try:
    from .. import __version__ as PANELBOX_VERSION
except ImportError:
    PANELBOX_VERSION = "0.1.0-dev"


class ReportManager:
    """
    Main orchestrator for PanelBox report generation.

    Coordinates TemplateManager, AssetManager, and CSSManager to generate
    complete, self-contained HTML reports.

    Parameters
    ----------
    template_dir : str or Path, optional
        Directory containing templates
    asset_dir : str or Path, optional
        Directory containing assets
    enable_cache : bool, default=True
        Enable template and asset caching
    minify : bool, default=False
        Enable CSS/JS minification

    Attributes
    ----------
    template_manager : TemplateManager
        Template manager instance
    asset_manager : AssetManager
        Asset manager instance
    css_manager : CSSManager
        CSS manager instance

    Examples
    --------
    >>> report_mgr = ReportManager()
    >>> html = report_mgr.generate_report(
    ...     report_type='validation',
    ...     template='validation/interactive/index.html',
    ...     context={'title': 'My Report', ...}
    ... )
    >>> report_mgr.save_report(html, 'report.html')
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        asset_dir: Optional[Path] = None,
        enable_cache: bool = True,
        minify: bool = False,
    ):
        """Initialize Report Manager."""
        # Initialize managers
        self.template_manager = TemplateManager(
            template_dir=template_dir, enable_cache=enable_cache
        )

        self.asset_manager = AssetManager(asset_dir=asset_dir, minify=minify)

        self.css_manager = CSSManager(asset_manager=self.asset_manager, minify=minify)

        self.minify = minify
        self.enable_cache = enable_cache

    def generate_report(
        self,
        report_type: str,
        template: str,
        context: Dict[str, Any],
        embed_assets: bool = True,
        include_plotly: bool = True,
        custom_css: Optional[List[str]] = None,
        custom_js: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a complete HTML report.

        Parameters
        ----------
        report_type : str
            Type of report (e.g., 'validation', 'regression', 'gmm')
        template : str
            Template path relative to template directory
        context : dict
            Template context variables
        embed_assets : bool, default=True
            Embed all assets inline for self-contained HTML
        include_plotly : bool, default=True
            Include Plotly.js library
        custom_css : list of str, optional
            Additional custom CSS files to include
        custom_js : list of str, optional
            Additional custom JS files to include

        Returns
        -------
        str
            Complete HTML report

        Examples
        --------
        >>> html = report_mgr.generate_report(
        ...     report_type='validation',
        ...     template='validation/interactive/index.html',
        ...     context={'title': 'Panel Validation', 'data': {...}}
        ... )
        """
        # Prepare base context
        full_context = self._prepare_context(report_type, context)

        # Add custom CSS to manager
        if custom_css:
            for css_file in custom_css:
                self.css_manager.add_custom_css(css_file)

        # Compile CSS
        if embed_assets:
            compiled_css = self.css_manager.compile_for_report_type(report_type)
            full_context["css_inline"] = compiled_css
        else:
            full_context["css_files"] = self._get_css_files()

        # Collect JavaScript
        if embed_assets:
            js_files = ["utils.js", "tab-navigation.js"]
            if custom_js:
                js_files.extend(custom_js)

            compiled_js = self.asset_manager.collect_js(js_files)
            full_context["js_inline"] = compiled_js
        else:
            full_context["js_files"] = self._get_js_files(custom_js)

        # Plotly
        if include_plotly:
            full_context["plotly_js"] = self.asset_manager.embed_plotly(include_plotly=True)
        else:
            full_context["plotly_js"] = ""

        # Render template
        html = self.template_manager.render_template(template, full_context)

        return html

    def generate_validation_report(
        self,
        validation_data: Dict[str, Any],
        interactive: bool = True,
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
    ) -> str:
        """
        Generate a validation report.

        Convenience method for validation reports.

        Parameters
        ----------
        validation_data : dict
            Validation data (from ValidationTransformer)
        interactive : bool, default=True
            Generate interactive report with Plotly charts
        title : str, optional
            Report title
        subtitle : str, optional
            Report subtitle

        Returns
        -------
        str
            Complete HTML validation report

        Examples
        --------
        >>> html = report_mgr.generate_validation_report(
        ...     validation_data={'tests': [...], 'model_info': {...}},
        ...     title='Panel Data Validation'
        ... )
        """
        # Determine template
        if interactive:
            template = "validation/interactive/index.html"
        else:
            template = "validation/static/index.html"

        # Build context
        # Spread model_info to top level for easy access in templates
        model_info = validation_data.get("model_info", {})

        context = {
            "report_title": title or "Panel Data Validation Report",
            "report_subtitle": subtitle,
            **validation_data,
            # Spread model_info for template convenience
            **model_info,
        }

        # Generate
        return self.generate_report(
            report_type="validation", template=template, context=context, include_plotly=interactive
        )

    def generate_regression_report(
        self,
        regression_data: Dict[str, Any],
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
    ) -> str:
        """
        Generate a regression results report.

        Convenience method for regression reports.

        Parameters
        ----------
        regression_data : dict
            Regression data (from RegressionTransformer)
        title : str, optional
            Report title
        subtitle : str, optional
            Report subtitle

        Returns
        -------
        str
            Complete HTML regression report

        Examples
        --------
        >>> html = report_mgr.generate_regression_report(
        ...     regression_data={'coefficients': [...], 'diagnostics': {...}},
        ...     title='Fixed Effects Results'
        ... )
        """
        template = "regression/index.html"

        context = {
            "report_title": title or "Regression Results",
            "report_subtitle": subtitle,
            **regression_data,
        }

        return self.generate_report(
            report_type="regression", template=template, context=context, include_plotly=True
        )

    def generate_gmm_report(
        self, gmm_data: Dict[str, Any], title: Optional[str] = None, subtitle: Optional[str] = None
    ) -> str:
        """
        Generate a GMM results report.

        Convenience method for GMM reports.

        Parameters
        ----------
        gmm_data : dict
            GMM data (from GMMTransformer)
        title : str, optional
            Report title
        subtitle : str, optional
            Report subtitle

        Returns
        -------
        str
            Complete HTML GMM report

        Examples
        --------
        >>> html = report_mgr.generate_gmm_report(
        ...     gmm_data={'coefficients': [...], 'hansen_test': {...}},
        ...     title='System GMM Results'
        ... )
        """
        template = "gmm/index.html"

        context = {"report_title": title or "GMM Results", "report_subtitle": subtitle, **gmm_data}

        return self.generate_report(
            report_type="gmm", template=template, context=context, include_plotly=True
        )

    def generate_residual_report(
        self,
        residual_data: Dict[str, Any],
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        interactive: bool = True,
    ) -> str:
        """
        Generate a residual diagnostics report.

        Convenience method for residual diagnostic reports.

        Parameters
        ----------
        residual_data : dict
            Residual diagnostics data with charts from create_residual_diagnostics()
            Expected keys:
            - 'residual_charts': dict with diagnostic plot HTML
            - 'model_info': dict with model information (optional)
            - 'diagnostics_summary': dict with summary info (optional)
        title : str, optional
            Report title
        subtitle : str, optional
            Report subtitle
        interactive : bool, default=True
            Generate interactive report with Plotly charts

        Returns
        -------
        str
            Complete HTML residual diagnostics report

        Examples
        --------
        >>> from panelbox.visualization import create_residual_diagnostics
        >>> # After model estimation
        >>> results = model.fit()
        >>> diagnostics = create_residual_diagnostics(results, theme='professional')
        >>>
        >>> residual_data = {
        ...     'residual_charts': diagnostics,
        ...     'model_info': {
        ...         'estimator': 'FixedEffects',
        ...         'nobs': 1000,
        ...         'n_entities': 100
        ...     }
        ... }
        >>>
        >>> html = report_mgr.generate_residual_report(
        ...     residual_data=residual_data,
        ...     title='Residual Diagnostics',
        ...     subtitle='Model specification checks'
        ... )
        """
        # Determine template
        if interactive:
            template = "residuals/interactive/index.html"
        else:
            template = "residuals/static/index.html"  # Future: static version

        # Extract model info if provided
        model_info = residual_data.get("model_info", {})

        # Build context
        context = {
            "report_title": title or "Residual Diagnostics Report",
            "report_subtitle": subtitle,
            "model_type": model_info.get("estimator", model_info.get("model_type")),
            "nobs": model_info.get("nobs"),
            "n_entities": model_info.get("n_entities"),
            "n_periods": model_info.get("n_periods"),
            "n_residuals": model_info.get("nobs"),  # Same as nobs
            **residual_data,
            # Spread model_info for template convenience
            **model_info,
        }

        # Count number of diagnostic plots
        if "residual_charts" in residual_data:
            context["n_diagnostics"] = len(residual_data["residual_charts"])

        # Generate
        return self.generate_report(
            report_type="residuals",
            template=template,
            context=context,
            include_plotly=interactive,
        )

    def generate_comparison_report(
        self,
        comparison_data: Dict[str, Any],
        title: Optional[str] = None,
        subtitle: Optional[str] = None,
        interactive: bool = True,
    ) -> str:
        """
        Generate a model comparison report.

        Convenience method for model comparison reports.

        Parameters
        ----------
        comparison_data : dict
            Model comparison data with charts from create_comparison_charts()
            Expected keys:
            - 'comparison_charts': dict with comparison chart HTML
            - 'models_info': list of dicts with model information (optional)
            - 'comparison_summary': dict with summary info (optional)
            - 'best_model_aic': name of best model by AIC (optional)
            - 'best_model_bic': name of best model by BIC (optional)
        title : str, optional
            Report title
        subtitle : str, optional
            Report subtitle
        interactive : bool, default=True
            Generate interactive report with Plotly charts

        Returns
        -------
        str
            Complete HTML model comparison report

        Examples
        --------
        >>> from panelbox.visualization import create_comparison_charts
        >>> # After estimating multiple models
        >>> results_list = [model1.fit(), model2.fit(), model3.fit()]
        >>> model_names = ['Pooled OLS', 'Fixed Effects', 'Random Effects']
        >>>
        >>> charts = create_comparison_charts(
        ...     results_list=results_list,
        ...     model_names=model_names,
        ...     theme='professional'
        ... )
        >>>
        >>> comparison_data = {
        ...     'comparison_charts': charts,
        ...     'models_info': [
        ...         {'name': 'Pooled OLS', 'estimator': 'PooledOLS', 'nobs': 1000,
        ...          'r_squared': 0.65, 'aic': 2500, 'bic': 2550},
        ...         {'name': 'Fixed Effects', 'estimator': 'PanelOLS', 'nobs': 1000,
        ...          'r_squared': 0.78, 'aic': 2300, 'bic': 2400},
        ...         {'name': 'Random Effects', 'estimator': 'RandomEffects', 'nobs': 1000,
        ...          'r_squared': 0.72, 'aic': 2350, 'bic': 2420}
        ...     ],
        ...     'best_model_aic': 'Fixed Effects',
        ...     'best_model_bic': 'Fixed Effects'
        ... }
        >>>
        >>> html = report_mgr.generate_comparison_report(
        ...     comparison_data=comparison_data,
        ...     title='Model Comparison',
        ...     subtitle='Pooled vs Fixed vs Random Effects'
        ... )
        """
        # Determine template
        if interactive:
            template = "comparison/interactive/index.html"
        else:
            template = "comparison/static/index.html"  # Future: static version

        # Extract models info if provided
        models_info = comparison_data.get("models_info", [])

        # Build context
        context = {
            "report_title": title or "Model Comparison Report",
            "report_subtitle": subtitle,
            "n_models": len(models_info) if models_info else comparison_data.get("n_models"),
            "best_model_aic": comparison_data.get("best_model_aic"),
            "best_model_bic": comparison_data.get("best_model_bic"),
            "models_info": models_info,
            # Header compatibility - use first model's info if available
            "model_type": f"Comparison ({len(models_info)} models)" if models_info else "Model Comparison",
            "nobs": models_info[0].get("nobs") if models_info else None,
            **comparison_data,
        }

        # Count number of comparison charts
        if "comparison_charts" in comparison_data:
            context["n_charts"] = len(comparison_data["comparison_charts"])

        # Generate
        return self.generate_report(
            report_type="comparison",
            template=template,
            context=context,
            include_plotly=interactive,
        )

    def save_report(
        self, html: str, output_path: Union[str, Path], overwrite: bool = False
    ) -> Path:
        """
        Save HTML report to file.

        Parameters
        ----------
        html : str
            HTML content
        output_path : str or Path
            Output file path
        overwrite : bool, default=False
            Overwrite existing file

        Returns
        -------
        Path
            Path to saved file

        Examples
        --------
        >>> html = report_mgr.generate_report(...)
        >>> path = report_mgr.save_report(html, 'report.html')
        >>> print(f"Report saved to {path}")
        """
        output_path = Path(output_path)

        # Check if file exists
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"File already exists: {output_path}. " "Use overwrite=True to replace."
            )

        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        output_path.write_text(html, encoding="utf-8")

        return output_path

    def _prepare_context(self, report_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare base context with metadata.

        Parameters
        ----------
        report_type : str
            Report type
        context : dict
            User-provided context

        Returns
        -------
        dict
            Complete context with metadata
        """
        # Get current timestamp
        now = datetime.datetime.now()

        # Base context
        base_context = {
            # Metadata
            "panelbox_version": PANELBOX_VERSION,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "report_type": report_type,
            "generation_date": now.strftime("%Y-%m-%d %H:%M:%S"),
            "generation_timestamp": now.isoformat(),
            # Report display options
            "show_export_buttons": True,
            "show_navigation": True,
            # Defaults (can be overridden)
            "report_title": f"PanelBox {report_type.title()} Report",
            "report_subtitle": None,
        }

        # Merge with user context (user context takes precedence)
        base_context.update(context)

        return base_context

    def _get_css_files(self) -> List[str]:
        """
        Get list of CSS files (for non-embedded mode).

        Returns
        -------
        list of str
            CSS file paths
        """
        css_files = []

        for layer in sorted(self.css_manager.layers.values(), key=lambda layer: layer.priority):
            css_files.extend(layer.files)

        return css_files

    def _get_js_files(self, custom_js: Optional[List[str]] = None) -> List[str]:
        """
        Get list of JS files (for non-embedded mode).

        Parameters
        ----------
        custom_js : list of str, optional
            Additional JS files

        Returns
        -------
        list of str
            JS file paths
        """
        js_files = ["utils.js", "tab-navigation.js"]

        if custom_js:
            js_files.extend(custom_js)

        return js_files

    def clear_cache(self) -> None:
        """
        Clear all caches.

        Examples
        --------
        >>> report_mgr.clear_cache()
        """
        self.template_manager.clear_cache()
        self.asset_manager.clear_cache()
        self.css_manager.clear_cache()

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about report manager state.

        Returns
        -------
        dict
            Manager information

        Examples
        --------
        >>> info = report_mgr.get_info()
        >>> print(f"Templates cached: {info['templates_cached']}")
        """
        return {
            "panelbox_version": PANELBOX_VERSION,
            "template_dir": str(self.template_manager.template_dir),
            "asset_dir": str(self.asset_manager.asset_dir),
            "templates_cached": len(self.template_manager.template_cache),
            "assets_cached": len(self.asset_manager.asset_cache),
            "css_layers": len(self.css_manager.layers),
            "minify_enabled": self.minify,
            "cache_enabled": self.enable_cache,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ReportManager("
            f"version={PANELBOX_VERSION}, "
            f"cache={self.enable_cache}, "
            f"minify={self.minify})"
        )
