"""
Spatial Extension for PanelExperiment.

This module extends PanelExperiment with spatial econometrics capabilities,
including spatial model estimation, diagnostics, and effects decomposition.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.diagnostics.spatial_tests import LocalMoranI, MoranIPanelTest, run_lm_tests
from panelbox.effects.spatial_effects import compute_spatial_effects
from panelbox.models.spatial.gns import GeneralNestingSpatial
from panelbox.models.spatial.sar import SpatialLag
from panelbox.models.spatial.sdm import SpatialDurbin
from panelbox.models.spatial.sem import SpatialError


class SpatialPanelExperiment:
    """
    Mixin class that adds spatial econometrics capabilities to PanelExperiment.

    This class provides methods for spatial diagnostics, model estimation,
    and effects decomposition. It should be used as a mixin with PanelExperiment.
    """

    # Spatial model type aliases
    SPATIAL_MODEL_ALIASES = {
        "sar": "spatial_lag",
        "spatial_lag": "spatial_lag",
        "sem": "spatial_error",
        "spatial_error": "spatial_error",
        "sdm": "spatial_durbin",
        "spatial_durbin": "spatial_durbin",
        "gns": "general_nesting",
        "general_nesting": "general_nesting",
    }

    def add_spatial_model(
        self,
        model_name: str,
        W: Union[np.ndarray, SpatialWeights],
        model_type: str = "sar",
        effects: str = "fixed",
        **kwargs,
    ) -> Any:
        """
        Add spatial model to experiment.

        Parameters
        ----------
        model_name : str
            Name for this model (for comparison)
        W : array-like or SpatialWeights
            Spatial weight matrix (N x N) or SpatialWeights object
        model_type : str
            Type of spatial model:
            - 'sar': Spatial Autoregressive (spatial lag)
            - 'sem': Spatial Error Model
            - 'sdm': Spatial Durbin Model (lag + WX)
            - 'gns': General Nesting Spatial
        effects : str
            'fixed' or 'random' effects
        **kwargs
            Additional arguments passed to model estimator

        Returns
        -------
        results
            Fitted model results

        Examples
        --------
        >>> # Create spatial weights
        >>> W = SpatialWeights.from_contiguity(gdf, criterion='queen')
        >>>
        >>> # Add spatial models
        >>> experiment.add_spatial_model('SAR-FE', W, 'sar', effects='fixed')
        >>> experiment.add_spatial_model('SDM-FE', W, 'sdm', effects='fixed')
        >>>
        >>> # Compare with non-spatial models
        >>> comparison = experiment.compare_models()
        """
        # Validate model type
        model_type = self.SPATIAL_MODEL_ALIASES.get(model_type, model_type)

        # Create appropriate model class
        model_classes = {
            "spatial_lag": SpatialLag,
            "spatial_error": SpatialError,
            "spatial_durbin": SpatialDurbin,
            "general_nesting": GeneralNestingSpatial,
        }

        if model_type not in model_classes:
            raise ValueError(
                f"Unknown spatial model type: {model_type}. "
                f"Choose from: {list(model_classes.keys())}"
            )

        ModelClass = model_classes[model_type]

        # Create and fit model
        model = ModelClass(
            formula=self.formula,
            data=self.data,
            entity_col=self.entity_col,
            time_col=self.time_col,
            W=W,
        )

        # Fit model
        results = model.fit(effects=effects, **kwargs)

        # Store in experiment
        self._models[model_name] = results
        self._model_metadata[model_name] = {
            "type": model_type,
            "effects": effects,
            "spatial": True,
            "W_shape": W.shape if hasattr(W, "shape") else None,
        }

        return results

    def run_spatial_diagnostics(
        self,
        W: Union[np.ndarray, SpatialWeights],
        model_name: Optional[str] = None,
        tests: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run spatial diagnostic tests.

        Parameters
        ----------
        W : array-like or SpatialWeights
            Spatial weight matrix
        model_name : str, optional
            Name of model to test. If None, uses OLS or pooled model
        tests : list of str, optional
            Which tests to run:
            - 'moran': Moran's I test for spatial autocorrelation
            - 'lm': LM tests for model specification
            - 'lisa': Local Indicators of Spatial Association
            If None, runs all tests

        Returns
        -------
        dict
            Dictionary with test results:
            - 'moran': Moran's I test result
            - 'lm_tests': LM tests results
            - 'lisa': LISA results (if requested)
            - 'recommendation': Suggested model based on tests

        Examples
        --------
        >>> # Estimate baseline OLS
        >>> experiment.fit_model('pooled_ols', name='ols')
        >>>
        >>> # Run spatial diagnostics
        >>> W = SpatialWeights.from_contiguity(gdf)
        >>> diagnostics = experiment.run_spatial_diagnostics(W, 'ols')
        >>>
        >>> print(f"Moran's I: {diagnostics['moran']['statistic']:.4f}")
        >>> print(f"P-value: {diagnostics['moran']['pvalue']:.4f}")
        >>> print(f"Recommended model: {diagnostics['recommendation']}")
        """
        if tests is None:
            tests = ["moran", "lm", "lisa"]

        # Get model residuals
        if model_name is None:
            # Try to find OLS or pooled model
            for name in self.list_models():
                if "ols" in name.lower() or "pooled" in name.lower():
                    model_name = name
                    break
            if model_name is None and self.list_models():
                model_name = self.list_models()[0]

        if model_name is None:
            raise ValueError("No models fitted. Fit a model first before running diagnostics.")

        model_results = self.get_model(model_name)

        # Get residuals
        if hasattr(model_results, "resid"):
            residuals = model_results.resid
        else:
            residuals = model_results.residuals

        # Results dictionary
        results = {}

        # 1. Moran's I test
        if "moran" in tests:
            moran_test = MoranIPanelTest(
                residuals=residuals,
                W=W,
                entity_ids=(
                    self.data.index.get_level_values(0)
                    if hasattr(self.data, "index")
                    else self.data[self.entity_col]
                ),
                time_ids=(
                    self.data.index.get_level_values(1)
                    if hasattr(self.data, "index")
                    else self.data[self.time_col]
                ),
            )
            moran_result = moran_test.run()

            results["moran"] = {
                "statistic": moran_result.statistic,
                "pvalue": moran_result.pvalue,
                "expected": moran_result.expected_value,
                "variance": moran_result.variance,
                "significant": moran_result.pvalue < 0.05,
            }

        # 2. LM tests
        if "lm" in tests:
            lm_results = run_lm_tests(model_results, W)
            results["lm_tests"] = lm_results

            # Get recommendation based on LM tests
            results["recommendation"] = self._get_spatial_model_recommendation(lm_results)

        # 3. LISA (Local Indicators)
        if "lisa" in tests:
            # Get cross-sectional residuals for last period
            last_period = self.data.index.get_level_values(1).max()
            period_mask = self.data.index.get_level_values(1) == last_period
            period_residuals = residuals[period_mask]

            lisa_test = LocalMoranI(
                values=period_residuals,
                W=W,
                entity_ids=self.data.index.get_level_values(0)[period_mask],
            )
            lisa_results = lisa_test.run()

            results["lisa"] = {
                "local_i": lisa_results.local_i,
                "pvalues": lisa_results.pvalues,
                "clusters": lisa_results.get_clusters(),
                "significant_entities": np.sum(lisa_results.pvalues < 0.05),
            }

        return results

    def compare_spatial_models(
        self, model_names: Optional[List[str]] = None, include_non_spatial: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple spatial models.

        Parameters
        ----------
        model_names : list of str, optional
            Names of models to compare. If None, compares all spatial models
        include_non_spatial : bool, default True
            Whether to include non-spatial models in comparison

        Returns
        -------
        pd.DataFrame
            Comparison table with:
            - Model name and type
            - Log-likelihood
            - AIC, BIC
            - Spatial parameters (rho, lambda)
            - Number of parameters
            - R-squared (if available)

        Examples
        --------
        >>> # Fit models
        >>> experiment.fit_model('pooled_ols', name='OLS')
        >>> experiment.add_spatial_model('SAR', W, 'sar')
        >>> experiment.add_spatial_model('SEM', W, 'sem')
        >>> experiment.add_spatial_model('SDM', W, 'sdm')
        >>>
        >>> # Compare
        >>> comparison = experiment.compare_spatial_models()
        >>> print(comparison[['AIC', 'BIC', 'rho', 'lambda']])
        """
        if model_names is None:
            model_names = self.list_models()

        # Filter spatial models if requested
        if not include_non_spatial:
            model_names = [
                name
                for name in model_names
                if self._model_metadata.get(name, {}).get("spatial", False)
            ]

        comparison_data = []

        for name in model_names:
            model = self.get_model(name)
            metadata = self._model_metadata.get(name, {})

            row = {
                "Model": name,
                "Type": metadata.get("type", "unknown"),
                "Spatial": metadata.get("spatial", False),
            }

            # Common metrics
            if hasattr(model, "llf"):
                row["Log-Lik"] = model.llf
            if hasattr(model, "aic"):
                row["AIC"] = model.aic
            if hasattr(model, "bic"):
                row["BIC"] = model.bic
            if hasattr(model, "rsquared"):
                row["R²"] = model.rsquared
            if hasattr(model, "nobs"):
                row["N"] = model.nobs

            # Spatial parameters
            if metadata.get("spatial", False):
                if hasattr(model, "rho"):
                    row["ρ"] = model.rho
                    row["ρ_pval"] = model.rho_pvalue if hasattr(model, "rho_pvalue") else None
                if hasattr(model, "lambda_"):
                    row["λ"] = model.lambda_
                    row["λ_pval"] = model.lambda_pvalue if hasattr(model, "lambda_pvalue") else None

            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by AIC if available
        if "AIC" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("AIC")

        return comparison_df

    def decompose_spatial_effects(
        self, model_name: str, variables: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Decompose spatial effects for SDM or GNS models.

        Parameters
        ----------
        model_name : str
            Name of SDM or GNS model
        variables : list of str, optional
            Variables to decompose. If None, decomposes all

        Returns
        -------
        dict
            Dictionary with DataFrames for:
            - 'direct': Direct effects
            - 'indirect': Indirect (spillover) effects
            - 'total': Total effects

        Examples
        --------
        >>> experiment.add_spatial_model('SDM', W, 'sdm')
        >>> effects = experiment.decompose_spatial_effects('SDM')
        >>>
        >>> print("Direct Effects:")
        >>> print(effects['direct'])
        >>>
        >>> print("\\nIndirect Effects (Spillovers):")
        >>> print(effects['indirect'])
        >>>
        >>> print("\\nTotal Effects:")
        >>> print(effects['total'])
        """
        model = self.get_model(model_name)
        metadata = self._model_metadata.get(model_name, {})

        # Check if model supports effects decomposition
        if metadata.get("type") not in ["spatial_durbin", "general_nesting"]:
            raise ValueError(
                f"Model '{model_name}' does not support effects decomposition. "
                "Only SDM and GNS models support this feature."
            )

        # Get effects
        if hasattr(model, "effects_decomposition"):
            effects = model.effects_decomposition(variables=variables)
        else:
            effects = compute_spatial_effects(model, variables=variables)

        return effects

    def generate_spatial_report(
        self,
        filename: str = "spatial_report.html",
        include_diagnostics: bool = True,
        include_effects: bool = True,
        include_maps: bool = False,
    ) -> None:
        """
        Generate comprehensive HTML report for spatial analysis.

        Parameters
        ----------
        filename : str
            Output HTML filename
        include_diagnostics : bool, default True
            Include spatial diagnostic tests
        include_effects : bool, default True
            Include effects decomposition for SDM/GNS models
        include_maps : bool, default False
            Include LISA cluster maps (requires geopandas)

        Examples
        --------
        >>> experiment.generate_spatial_report(
        ...     'spatial_analysis.html',
        ...     include_diagnostics=True,
        ...     include_effects=True
        ... )
        """
        from panelbox.reporting.spatial_report import SpatialReportGenerator

        generator = SpatialReportGenerator(self)
        generator.generate(
            filename=filename,
            include_diagnostics=include_diagnostics,
            include_effects=include_effects,
            include_maps=include_maps,
        )

    def _get_spatial_model_recommendation(self, lm_results: Dict) -> str:
        """
        Get model recommendation based on LM tests.

        Parameters
        ----------
        lm_results : dict
            Results from run_lm_tests

        Returns
        -------
        str
            Recommended model type
        """
        # Extract p-values
        lm_lag_p = lm_results.get("lm_lag", {}).get("pvalue", 1.0)
        lm_error_p = lm_results.get("lm_error", {}).get("pvalue", 1.0)
        robust_lm_lag_p = lm_results.get("robust_lm_lag", {}).get("pvalue", 1.0)
        robust_lm_error_p = lm_results.get("robust_lm_error", {}).get("pvalue", 1.0)

        # Decision tree (Anselin & Rey 2014)
        if lm_lag_p < 0.05 and lm_error_p >= 0.05:
            return "SAR (Spatial Lag Model)"
        elif lm_lag_p >= 0.05 and lm_error_p < 0.05:
            return "SEM (Spatial Error Model)"
        elif lm_lag_p < 0.05 and lm_error_p < 0.05:
            # Both significant - check robust tests
            if robust_lm_lag_p < 0.05 and robust_lm_error_p >= 0.05:
                return "SAR (Spatial Lag Model)"
            elif robust_lm_lag_p >= 0.05 and robust_lm_error_p < 0.05:
                return "SEM (Spatial Error Model)"
            else:
                return "SDM (Spatial Durbin Model) or GNS"
        else:
            return "No spatial dependence detected"


# Extend the main PanelExperiment class
def extend_panel_experiment():
    """
    Dynamically extend PanelExperiment with spatial methods.

    This function adds spatial methods to the existing PanelExperiment class
    without modifying the original source code.
    """
    from panelbox.experiment import PanelExperiment

    # Add spatial methods to PanelExperiment
    for method_name in dir(SpatialPanelExperiment):
        if not method_name.startswith("_") or method_name == "_get_spatial_model_recommendation":
            method = getattr(SpatialPanelExperiment, method_name)
            if callable(method):
                setattr(PanelExperiment, method_name, method)

    # Add spatial model aliases to existing aliases
    if hasattr(PanelExperiment, "MODEL_ALIASES"):
        PanelExperiment.MODEL_ALIASES.update(SpatialPanelExperiment.SPATIAL_MODEL_ALIASES)


# Auto-extend when imported
extend_panel_experiment()
