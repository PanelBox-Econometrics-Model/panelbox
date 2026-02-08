"""
Data transformer for model comparison visualizations.

Converts multiple model results into formats expected by comparison charts.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class ComparisonDataTransformer:
    """
    Transform multiple model results to comparison chart format.

    Takes a list of fitted model results and extracts comparison data
    for coefficient plots, forest plots, and fit statistics.

    Examples
    --------
    >>> from panelbox import FixedEffects, RandomEffects
    >>> from panelbox.visualization.transformers import ComparisonDataTransformer
    >>>
    >>> # Fit multiple models
    >>> fe_results = FixedEffects('y ~ x1 + x2', data, ...).fit()
    >>> re_results = RandomEffects('y ~ x1 + x2', data, ...).fit()
    >>>
    >>> # Transform for comparison
    >>> transformer = ComparisonDataTransformer()
    >>> comparison_data = transformer.transform([fe_results, re_results],
    >>>                                          names=['Fixed Effects', 'Random Effects'])
    >>>
    >>> # Use with chart API
    >>> from panelbox.visualization import CoefficientComparisonChart
    >>> chart = CoefficientComparisonChart()
    >>> chart.create(comparison_data)
    """

    def transform(
        self, results_list: List[Any], names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Transform multiple model results to comparison format.

        Parameters
        ----------
        results_list : list of PanelResults
            List of fitted model results to compare
        names : list of str, optional
            Names for each model. If None, uses 'Model 1', 'Model 2', etc.

        Returns
        -------
        dict
            Structured data for comparison charts with keys:
            - 'models': list of model names
            - 'coefficients': dict mapping variables to lists of coefficient values
            - 'std_errors': dict mapping variables to lists of standard errors
            - 'pvalues': dict mapping variables to lists of p-values
            - 'fit_metrics': dict of fit statistics
            - 'ic_values': dict of information criteria

        Examples
        --------
        >>> data = transformer.transform([results1, results2],
        >>>                              names=['Model A', 'Model B'])
        >>> print(data['models'])
        ['Model A', 'Model B']
        >>> print(data['coefficients'])
        {'x1': [0.5, 0.6], 'x2': [0.3, 0.25]}
        """
        # Generate default names if not provided
        if names is None:
            names = [f"Model {i+1}" for i in range(len(results_list))]

        # Extract all data
        coefficients = self._extract_coefficients(results_list)
        std_errors = self._extract_std_errors(results_list)
        pvalues = self._extract_pvalues(results_list)
        fit_metrics = self._extract_fit_metrics(results_list)
        ic_values = self._extract_ic_values(results_list)

        return {
            "models": names,
            "coefficients": coefficients,
            "std_errors": std_errors,
            "pvalues": pvalues,
            "fit_metrics": fit_metrics,
            "ic_values": ic_values,
        }

    def _extract_coefficients(self, results_list: List[Any]) -> Dict[str, List[float]]:
        """
        Extract coefficients from all models.

        Parameters
        ----------
        results_list : list of PanelResults
            List of model results

        Returns
        -------
        dict
            Dictionary mapping variable names to lists of coefficient values
        """
        # Get all unique variables across models
        all_vars = set()
        for results in results_list:
            if hasattr(results, "params"):
                if hasattr(results.params, "index"):
                    all_vars.update(results.params.index)

        # Extract coefficients for each variable
        coefficients = {}
        for var in all_vars:
            coefficients[var] = []
            for results in results_list:
                if hasattr(results, "params"):
                    if hasattr(results.params, "get"):
                        # Pandas Series
                        value = results.params.get(var, np.nan)
                    elif hasattr(results.params, "index"):
                        # Try to access by index
                        try:
                            idx = list(results.params.index).index(var)
                            value = (
                                results.params.iloc[idx]
                                if hasattr(results.params, "iloc")
                                else results.params[idx]
                            )
                        except (ValueError, KeyError, IndexError):
                            value = np.nan
                    else:
                        value = np.nan
                else:
                    value = np.nan
                coefficients[var].append(self._safe_float(value))

        return coefficients

    def _extract_std_errors(self, results_list: List[Any]) -> Dict[str, List[float]]:
        """
        Extract standard errors from all models.

        Parameters
        ----------
        results_list : list of PanelResults
            List of model results

        Returns
        -------
        dict
            Dictionary mapping variable names to lists of standard errors
        """
        # Get all unique variables
        all_vars = set()
        for results in results_list:
            if hasattr(results, "params"):
                if hasattr(results.params, "index"):
                    all_vars.update(results.params.index)

        # Extract standard errors for each variable
        std_errors = {}
        for var in all_vars:
            std_errors[var] = []
            for results in results_list:
                if hasattr(results, "std_errors"):
                    # Access standard errors
                    if hasattr(results.std_errors, "get"):
                        value = results.std_errors.get(var, np.nan)
                    elif hasattr(results.std_errors, "index"):
                        try:
                            idx = list(results.std_errors.index).index(var)
                            value = (
                                results.std_errors.iloc[idx]
                                if hasattr(results.std_errors, "iloc")
                                else results.std_errors[idx]
                            )
                        except (ValueError, KeyError, IndexError):
                            value = np.nan
                    else:
                        value = np.nan
                elif hasattr(results, "bse"):  # Alternative attribute name
                    if hasattr(results.bse, "get"):
                        value = results.bse.get(var, np.nan)
                    else:
                        value = np.nan
                else:
                    value = np.nan
                std_errors[var].append(self._safe_float(value))

        return std_errors

    def _extract_pvalues(self, results_list: List[Any]) -> Dict[str, List[float]]:
        """
        Extract p-values from all models.

        Parameters
        ----------
        results_list : list of PanelResults
            List of model results

        Returns
        -------
        dict
            Dictionary mapping variable names to lists of p-values
        """
        # Get all unique variables
        all_vars = set()
        for results in results_list:
            if hasattr(results, "params"):
                if hasattr(results.params, "index"):
                    all_vars.update(results.params.index)

        # Extract p-values for each variable
        pvalues = {}
        for var in all_vars:
            pvalues[var] = []
            for results in results_list:
                if hasattr(results, "pvalues"):
                    if hasattr(results.pvalues, "get"):
                        value = results.pvalues.get(var, np.nan)
                    elif hasattr(results.pvalues, "index"):
                        try:
                            idx = list(results.pvalues.index).index(var)
                            value = (
                                results.pvalues.iloc[idx]
                                if hasattr(results.pvalues, "iloc")
                                else results.pvalues[idx]
                            )
                        except (ValueError, KeyError, IndexError):
                            value = np.nan
                    else:
                        value = np.nan
                else:
                    value = np.nan
                pvalues[var].append(self._safe_float(value))

        return pvalues

    def _safe_float(self, value: Any) -> float:
        """
        Safely convert value to float, returning NaN if conversion fails.

        Parameters
        ----------
        value : Any
            Value to convert

        Returns
        -------
        float
            Converted value or NaN
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def _extract_fit_metrics(self, results_list: List[Any]) -> Dict[str, List[float]]:
        """
        Extract fit metrics from all models.

        Parameters
        ----------
        results_list : list of PanelResults
            List of model results

        Returns
        -------
        dict
            Dictionary with fit metrics
        """
        metrics = {"R²": [], "Adj. R²": [], "F-statistic": [], "Log-Likelihood": []}

        for results in results_list:
            # R-squared
            if hasattr(results, "rsquared"):
                metrics["R²"].append(self._safe_float(results.rsquared))
            else:
                metrics["R²"].append(np.nan)

            # Adjusted R-squared
            if hasattr(results, "rsquared_adj"):
                metrics["Adj. R²"].append(self._safe_float(results.rsquared_adj))
            else:
                metrics["Adj. R²"].append(np.nan)

            # F-statistic
            if hasattr(results, "fvalue"):
                metrics["F-statistic"].append(self._safe_float(results.fvalue))
            elif hasattr(results, "f_statistic"):
                metrics["F-statistic"].append(self._safe_float(results.f_statistic))
            else:
                metrics["F-statistic"].append(np.nan)

            # Log-likelihood
            if hasattr(results, "loglik"):
                metrics["Log-Likelihood"].append(self._safe_float(results.loglik))
            elif hasattr(results, "llf"):
                metrics["Log-Likelihood"].append(self._safe_float(results.llf))
            else:
                metrics["Log-Likelihood"].append(np.nan)

        # Remove metrics that are all NaN
        metrics = {k: v for k, v in metrics.items() if not all(np.isnan(v))}

        return metrics

    def _extract_ic_values(self, results_list: List[Any]) -> Dict[str, List[float]]:
        """
        Extract information criteria from all models.

        Parameters
        ----------
        results_list : list of PanelResults
            List of model results

        Returns
        -------
        dict
            Dictionary with IC values
        """
        ic_dict = {"aic": [], "bic": [], "hqic": []}

        for results in results_list:
            # AIC
            if hasattr(results, "aic"):
                ic_dict["aic"].append(self._safe_float(results.aic))
            else:
                ic_dict["aic"].append(np.nan)

            # BIC
            if hasattr(results, "bic"):
                ic_dict["bic"].append(self._safe_float(results.bic))
            else:
                ic_dict["bic"].append(np.nan)

            # HQIC
            if hasattr(results, "hqic"):
                ic_dict["hqic"].append(self._safe_float(results.hqic))
            else:
                ic_dict["hqic"].append(np.nan)

        # Remove IC that are all NaN
        ic_dict = {k: v for k, v in ic_dict.items() if not all(np.isnan(v))}

        return ic_dict

    def prepare_coefficient_comparison(
        self,
        results_list: List[Any],
        names: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare data for coefficient comparison chart.

        Parameters
        ----------
        results_list : list of PanelResults
            List of model results
        names : list of str, optional
            Model names
        variables : list of str, optional
            Variables to include. If None, includes all

        Returns
        -------
        dict
            Data for CoefficientComparisonChart
        """
        data = self.transform(results_list, names)

        # Filter variables if specified
        if variables:
            data["coefficients"] = {k: v for k, v in data["coefficients"].items() if k in variables}
            data["std_errors"] = {k: v for k, v in data["std_errors"].items() if k in variables}

        return {
            "models": data["models"],
            "coefficients": data["coefficients"],
            "std_errors": data["std_errors"],
            "show_significance": True,
            "ci_level": 0.95,
        }

    def prepare_forest_plot(
        self, results: Any, variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare data for forest plot (single model).

        Parameters
        ----------
        results : PanelResults
            Model results
        variables : list of str, optional
            Variables to include

        Returns
        -------
        dict
            Data for ForestPlotChart
        """
        # Extract data from single model
        coef_dict = self._extract_coefficients([results])
        se_dict = self._extract_std_errors([results])
        pval_dict = self._extract_pvalues([results])

        # Get list of variables
        if variables is None:
            variables = list(coef_dict.keys())

        # Prepare arrays
        estimates = [coef_dict[v][0] for v in variables]
        std_errors = [se_dict[v][0] for v in variables]
        pvalues = [pval_dict[v][0] for v in variables]

        # Calculate 95% CI
        ci_lower = [e - 1.96 * se for e, se in zip(estimates, std_errors)]
        ci_upper = [e + 1.96 * se for e, se in zip(estimates, std_errors)]

        return {
            "variables": variables,
            "estimates": estimates,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "pvalues": pvalues,
            "sort_by_size": False,
        }

    def prepare_model_fit_comparison(
        self, results_list: List[Any], names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare data for model fit comparison chart.

        Parameters
        ----------
        results_list : list of PanelResults
            List of model results
        names : list of str, optional
            Model names

        Returns
        -------
        dict
            Data for ModelFitComparisonChart
        """
        data = self.transform(results_list, names)

        return {"models": data["models"], "metrics": data["fit_metrics"], "normalize": False}

    def prepare_ic_comparison(
        self, results_list: List[Any], names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Prepare data for information criteria comparison.

        Parameters
        ----------
        results_list : list of PanelResults
            List of model results
        names : list of str, optional
            Model names

        Returns
        -------
        dict
            Data for InformationCriteriaChart
        """
        data = self.transform(results_list, names)
        ic = data["ic_values"]

        return {
            "models": data["models"],
            "aic": ic.get("aic"),
            "bic": ic.get("bic"),
            "hqic": ic.get("hqic"),
            "show_delta": True,
        }
