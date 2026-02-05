"""
Results container for panel econometric models.

This module provides the PanelResults class which stores estimation results
and provides methods for inference, prediction, and reporting.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class PanelResults:
    """
    Container for panel model estimation results.

    This class stores all estimation results and provides methods for
    inference, testing, prediction, and reporting.

    Parameters
    ----------
    params : pd.Series
        Estimated coefficients
    std_errors : pd.Series
        Standard errors
    cov_params : pd.DataFrame
        Covariance matrix of parameters
    resid : np.ndarray
        Residuals
    fittedvalues : np.ndarray
        Fitted values
    model_info : dict
        Dictionary with model information
    data_info : dict
        Dictionary with data information

    Attributes
    ----------
    params : pd.Series
        Estimated coefficients with parameter names
    std_errors : pd.Series
        Standard errors
    tvalues : pd.Series
        t-statistics
    pvalues : pd.Series
        p-values for two-sided t-tests
    cov_params : pd.DataFrame
        Covariance matrix of parameters
    resid : np.ndarray
        Residuals
    fittedvalues : np.ndarray
        Fitted values
    nobs : int
        Number of observations
    n_entities : int
        Number of entities
    n_periods : int
        Number of time periods
    df_model : int
        Degrees of freedom for model
    df_resid : int
        Degrees of freedom for residuals
    rsquared : float
        R-squared
    rsquared_adj : float
        Adjusted R-squared
    rsquared_within : float
        Within R-squared (for panel models)
    rsquared_between : float
        Between R-squared (for panel models)
    rsquared_overall : float
        Overall R-squared (for panel models)
    """

    def __init__(
        self,
        params: pd.Series,
        std_errors: pd.Series,
        cov_params: pd.DataFrame,
        resid: np.ndarray,
        fittedvalues: np.ndarray,
        model_info: Dict[str, Any],
        data_info: Dict[str, Any],
        rsquared_dict: Optional[Dict[str, float]] = None,
        model: Optional[Any] = None,
    ):
        # Parameter estimates
        self.params = params
        self.std_errors = std_errors
        self.cov_params = cov_params

        # Residuals and fitted values
        self.resid = resid
        self.fittedvalues = fittedvalues

        # Model information
        self.model_type = model_info.get("model_type", "Unknown")
        self.formula = model_info.get("formula", "")
        self.cov_type = model_info.get("cov_type", "nonrobust")
        self.cov_kwds = model_info.get("cov_kwds", {})

        # Data information
        self.nobs = data_info["nobs"]
        self.n_entities = data_info["n_entities"]
        self.n_periods = data_info.get("n_periods", None)
        self.df_model = data_info["df_model"]
        self.df_resid = data_info["df_resid"]

        # Entity and time indices (for validation tests)
        self.entity_index = data_info.get("entity_index", None)
        self.time_index = data_info.get("time_index", None)

        # Store reference to model for validation tests
        self._model = model

        # Compute t-values and p-values (after df_resid is defined)
        self.tvalues = self.params / self.std_errors
        # Ensure pvalues is a pandas Series with the same index as params
        pvalues_array = 2 * (1 - stats.t.cdf(np.abs(self.tvalues.values), self.df_resid))
        self.pvalues = pd.Series(pvalues_array, index=self.params.index)

        # R-squared statistics
        if rsquared_dict is not None:
            self.rsquared = rsquared_dict.get("rsquared", np.nan)
            self.rsquared_adj = rsquared_dict.get("rsquared_adj", np.nan)
            self.rsquared_within = rsquared_dict.get("rsquared_within", np.nan)
            self.rsquared_between = rsquared_dict.get("rsquared_between", np.nan)
            self.rsquared_overall = rsquared_dict.get("rsquared_overall", np.nan)
        else:
            self.rsquared = np.nan
            self.rsquared_adj = np.nan
            self.rsquared_within = np.nan
            self.rsquared_between = np.nan
            self.rsquared_overall = np.nan

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Compute confidence intervals for parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (e.g., 0.05 for 95% CI)

        Returns
        -------
        pd.DataFrame
            Confidence intervals with columns 'lower' and 'upper'

        Examples
        --------
        >>> ci = results.conf_int(alpha=0.05)
        >>> print(ci)
        """
        t_critical = stats.t.ppf(1 - alpha / 2, self.df_resid)
        margin = t_critical * self.std_errors

        ci = pd.DataFrame(
            {"lower": self.params - margin, "upper": self.params + margin}, index=self.params.index
        )

        return ci

    def predict(self, newdata: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        newdata : pd.DataFrame, optional
            New data for prediction. If None, returns fitted values.

        Returns
        -------
        np.ndarray
            Predicted values

        Examples
        --------
        >>> predictions = results.predict()
        >>> new_predictions = results.predict(new_data)
        """
        if newdata is None:
            return self.fittedvalues
        else:
            raise NotImplementedError("Prediction on new data not yet implemented")

    def summary(self, title: Optional[str] = None) -> str:
        """
        Generate formatted summary of results.

        Parameters
        ----------
        title : str, optional
            Custom title for summary table

        Returns
        -------
        str
            Formatted summary table

        Examples
        --------
        >>> print(results.summary())
        """
        lines = []

        # Header
        lines.append("=" * 78)
        if title is None:
            title = f"{self.model_type} Estimation Results"
        lines.append(title.center(78))
        lines.append("=" * 78)

        # Model information
        lines.append(f"Formula: {self.formula}")
        lines.append(f"Model:   {self.model_type}")
        lines.append("-" * 78)

        # Sample information
        lines.append(f"No. Observations:          {self.nobs:>10,}")
        lines.append(f"No. Entities:              {self.n_entities:>10,}")
        if self.n_periods is not None:
            lines.append(f"No. Time Periods:          {self.n_periods:>10,}")
        lines.append(f"Degrees of Freedom:        {self.df_resid:>10,}")

        # R-squared
        if not np.isnan(self.rsquared):
            lines.append(f"R-squared:                 {self.rsquared:>10.4f}")
        if not np.isnan(self.rsquared_adj):
            lines.append(f"Adj. R-squared:            {self.rsquared_adj:>10.4f}")
        if not np.isnan(self.rsquared_within):
            lines.append(f"R-squared (within):        {self.rsquared_within:>10.4f}")
        if not np.isnan(self.rsquared_between):
            lines.append(f"R-squared (between):       {self.rsquared_between:>10.4f}")
        if not np.isnan(self.rsquared_overall):
            lines.append(f"R-squared (overall):       {self.rsquared_overall:>10.4f}")

        # Standard errors type
        lines.append(f"Standard Errors:           {self.cov_type:>10}")

        lines.append("=" * 78)

        # Coefficient table
        lines.append(
            f"{'Variable':<15} {'Coef.':<12} {'Std.Err.':<12} {'t':<8} {'P>|t|':<8} {'[0.025':<10} {'0.975]':<10}"
        )
        lines.append("-" * 78)

        ci = self.conf_int(alpha=0.05)

        for var in self.params.index:
            coef = self.params[var]
            se = self.std_errors[var]
            t = self.tvalues[var]
            p = self.pvalues[var]
            ci_lower = ci.loc[var, "lower"]
            ci_upper = ci.loc[var, "upper"]

            # Significance stars
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            elif p < 0.10:
                stars = "."
            else:
                stars = ""

            lines.append(
                f"{var:<15} {coef:>11.4f} {se:>11.4f} {t:>7.3f} "
                f"{p:>7.4f} {ci_lower:>9.4f} {ci_upper:>9.4f} {stars}"
            )

        lines.append("=" * 78)
        lines.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export results as dictionary.

        Returns
        -------
        dict
            Dictionary with all results including parameters, statistics,
            and model information. Arrays are converted to lists for
            JSON compatibility.

        Examples
        --------
        >>> results = fe.fit()
        >>> results_dict = results.to_dict()
        >>> print(results_dict.keys())
        """
        # Convert numpy arrays to lists for JSON compatibility
        resid_list = self.resid.tolist() if isinstance(self.resid, np.ndarray) else list(self.resid)
        fitted_list = (
            self.fittedvalues.tolist()
            if isinstance(self.fittedvalues, np.ndarray)
            else list(self.fittedvalues)
        )

        # Convert covariance matrix to nested list
        cov_params_dict = None
        if self.cov_params is not None:
            if isinstance(self.cov_params, pd.DataFrame):
                cov_params_dict = {
                    "values": self.cov_params.values.tolist(),
                    "index": self.cov_params.index.tolist(),
                    "columns": self.cov_params.columns.tolist(),
                }
            elif isinstance(self.cov_params, np.ndarray):
                cov_params_dict = {
                    "values": self.cov_params.tolist(),
                    "index": self.params.index.tolist(),
                    "columns": self.params.index.tolist(),
                }

        return {
            "params": self.params.to_dict(),
            "std_errors": self.std_errors.to_dict(),
            "tvalues": self.tvalues.to_dict(),
            "pvalues": self.pvalues.to_dict(),
            "cov_params": cov_params_dict,
            "resid": resid_list,
            "fittedvalues": fitted_list,
            "model_info": {
                "model_type": self.model_type,
                "formula": self.formula,
                "cov_type": self.cov_type,
                "cov_kwds": self.cov_kwds,
            },
            "sample_info": {
                "nobs": int(self.nobs),
                "n_entities": int(self.n_entities),
                "n_periods": int(self.n_periods) if self.n_periods is not None else None,
                "df_model": int(self.df_model),
                "df_resid": int(self.df_resid),
            },
            "rsquared": {
                "rsquared": float(self.rsquared) if not np.isnan(self.rsquared) else None,
                "rsquared_adj": (
                    float(self.rsquared_adj) if not np.isnan(self.rsquared_adj) else None
                ),
                "rsquared_within": (
                    float(self.rsquared_within) if not np.isnan(self.rsquared_within) else None
                ),
                "rsquared_between": (
                    float(self.rsquared_between) if not np.isnan(self.rsquared_between) else None
                ),
                "rsquared_overall": (
                    float(self.rsquared_overall) if not np.isnan(self.rsquared_overall) else None
                ),
            },
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """
        Export results to JSON format.

        Parameters
        ----------
        filepath : str or Path, optional
            Path to save JSON file. If None, returns JSON string without saving.
        indent : int, default=2
            Number of spaces for JSON indentation

        Returns
        -------
        str
            JSON string representation of results

        Examples
        --------
        >>> results = fe.fit()
        >>> # Save to file
        >>> results.to_json('results.json')
        >>> # Get JSON string
        >>> json_str = results.to_json()
        >>> print(json_str[:100])
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent, ensure_ascii=False)

        if filepath is not None:
            filepath = Path(filepath)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

    def save(self, filepath: Union[str, Path], format: str = "pickle") -> None:
        """
        Save results to file.

        Parameters
        ----------
        filepath : str or Path
            Path to save file
        format : str, default='pickle'
            Format to save results. Options:
            - 'pickle': Python pickle format (preserves all objects)
            - 'json': JSON format (text-based, limited precision)

        Examples
        --------
        >>> results = fe.fit()
        >>> # Save as pickle (recommended)
        >>> results.save('results.pkl')
        >>> # Save as JSON
        >>> results.save('results.json', format='json')

        Notes
        -----
        Pickle format is recommended as it preserves all Python objects
        exactly, including numpy arrays and pandas objects. JSON format
        is human-readable but may lose precision for floating-point values
        and does not preserve object types.
        """
        filepath = Path(filepath)

        if format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif format == "json":
            self.to_json(filepath)
        else:
            raise ValueError(
                f"Format '{format}' not supported. " f"Supported formats: 'pickle', 'json'"
            )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PanelResults":
        """
        Load results from pickle file.

        Parameters
        ----------
        filepath : str or Path
            Path to pickle file

        Returns
        -------
        PanelResults
            Loaded results object

        Examples
        --------
        >>> # Save results
        >>> results = fe.fit()
        >>> results.save('results.pkl')
        >>> # Load results later
        >>> loaded_results = PanelResults.load('results.pkl')
        >>> print(loaded_results.summary())

        Notes
        -----
        This method only works with pickle files created by the save()
        method. JSON files cannot be loaded directly as they do not
        preserve all Python object types.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "rb") as f:
            results = pickle.load(f)

        if not isinstance(results, cls):
            raise TypeError(
                f"Loaded object is not a PanelResults instance. " f"Got type: {type(results)}"
            )

        return results

    def validate(
        self, tests: str = "default", alpha: float = 0.05, verbose: bool = False
    ) -> "ValidationReport":
        """
        Run validation tests on model results.

        Parameters
        ----------
        tests : str or list of str, default='default'
            Which tests to run:
            - 'all': Run all available tests
            - 'default': Run recommended tests for this model type
            - 'serial': Serial correlation tests only
            - 'het': Heteroskedasticity tests only
            - 'cd': Cross-sectional dependence tests only
        alpha : float, default=0.05
            Significance level for tests
        verbose : bool, default=False
            If True, print progress during testing

        Returns
        -------
        ValidationReport
            Report containing all test results

        Examples
        --------
        >>> results = fe.fit()
        >>> validation = results.validate(tests='all', verbose=True)
        >>> print(validation)
        """
        from panelbox.validation.validation_suite import ValidationSuite

        suite = ValidationSuite(self)
        return suite.run(tests=tests, alpha=alpha, verbose=verbose)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PanelResults("
            f"model='{self.model_type}', "
            f"nobs={self.nobs}, "
            f"k_params={len(self.params)})"
        )

    def __str__(self) -> str:
        """String representation (calls summary)."""
        return self.summary()
