"""
PanelPipeline -- End-to-end pipeline for production deployment.

Usage:
    from panelbox.production import PanelPipeline
    from panelbox.gmm import DifferenceGMM

    # Train
    pipeline = PanelPipeline(
        model_class=DifferenceGMM,
        model_params={
            'dep_var': 'lgd', 'lags': 1,
            'exog_vars': ['saldo', 'pib', 'selic'],
            'id_var': 'contrato', 'time_var': 'dt_ref',
        }
    )
    pipeline.fit(df_train)
    pipeline.save('modelo_lgd_v1.pkl')

    # Production (daily)
    pipeline = PanelPipeline.load('modelo_lgd_v1.pkl')
    predictions = pipeline.predict(df_new)
"""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PanelPipeline:
    """
    End-to-end pipeline for panel model deployment.

    Encapsulates model specification, estimation, prediction,
    persistence and validation in a single object.

    Parameters
    ----------
    model_class : type
        Model class (e.g., DifferenceGMM, PooledOLS, FixedEffects)
    model_params : dict
        Parameters to pass to model_class constructor (except ``data``)
    name : str, optional
        Human-readable name for the pipeline
    description : str, optional
        Description of what this model does

    Attributes
    ----------
    results : object
        Fitted model results (after calling fit())
    fit_timestamp : str
        ISO timestamp of when the model was last fitted
    metadata : dict
        Pipeline metadata (name, description, version, etc.)
    """

    def __init__(
        self,
        model_class: type,
        model_params: dict[str, Any],
        name: str = "",
        description: str = "",
    ):
        self.model_class = model_class
        self.model_params = model_params
        self.name = name or model_class.__name__
        self.description = description

        self.results = None
        self.fit_timestamp = None
        self._fit_data_info = None
        self._version = self._get_panelbox_version()

    def fit(self, data: pd.DataFrame) -> PanelPipeline:
        """
        Estimate the model on training data.

        Parameters
        ----------
        data : pd.DataFrame
            Training data

        Returns
        -------
        self
            For method chaining
        """
        model = self.model_class(data=data, **self.model_params)
        self.results = model.fit()
        self.fit_timestamp = datetime.now().isoformat()
        self._fit_data_info = {
            "n_rows": len(data),
            "n_cols": len(data.columns),
            "columns": list(data.columns),
        }
        return self

    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions with new data.

        Parameters
        ----------
        new_data : pd.DataFrame
            New data for prediction

        Returns
        -------
        np.ndarray
            Predicted values

        Raises
        ------
        RuntimeError
            If model has not been fitted yet
        """
        if self.results is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        return self.results.predict(new_data)

    def forecast(self, **kwargs) -> pd.DataFrame:
        """
        Generate multi-step forecasts (for dynamic models like GMM).

        Parameters
        ----------
        **kwargs
            Passed to results.forecast() (last_obs, future_exog, steps)

        Returns
        -------
        pd.DataFrame
            Forecast results
        """
        if self.results is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        if not hasattr(self.results, "forecast"):
            raise AttributeError(
                f"{type(self.results).__name__} does not support forecast(). "
                "Only dynamic models (GMM) support multi-step forecasting."
            )

        return self.results.forecast(**kwargs)

    def refit(self, new_data: pd.DataFrame) -> PanelPipeline:
        """
        Re-estimate model with updated data.

        Parameters
        ----------
        new_data : pd.DataFrame
            Updated training data

        Returns
        -------
        self
        """
        old_results = self.results
        self.fit(new_data)

        # Log comparison if old results exist
        if old_results is not None:
            self._log_refit_comparison(old_results, self.results)

        return self

    def save(self, filepath: str | Path) -> None:
        """
        Save entire pipeline to disk.

        Parameters
        ----------
        filepath : str or Path
            Path to save the pipeline (.pkl)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str | Path) -> PanelPipeline:
        """
        Load pipeline from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to pickle file

        Returns
        -------
        PanelPipeline
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")

        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)  # noqa: S301 — intentional deserialization of user's own saved pipeline

        if not isinstance(pipeline, cls):
            raise TypeError(f"Loaded object is {type(pipeline).__name__}, expected PanelPipeline.")

        return pipeline

    def validate(self) -> dict[str, Any]:
        """
        Run pre-deployment validation checks.

        Returns
        -------
        dict
            Validation results with keys: 'passed', 'checks', 'warnings'
        """
        if self.results is None:
            return {"passed": False, "checks": [], "warnings": ["Model not fitted"]}

        checks = []
        warnings_list = []

        # Check 1: Model has parameters
        has_params = hasattr(self.results, "params") and self.results.params is not None
        checks.append({"name": "has_params", "passed": has_params})

        # Check 2: No NaN in parameters
        if has_params:
            no_nan = not np.any(np.isnan(self.results.params.values))
            checks.append({"name": "no_nan_params", "passed": no_nan})
            if not no_nan:
                warnings_list.append("Model has NaN parameters")

        # Check 3: GMM-specific checks
        if hasattr(self.results, "hansen_j"):
            # Hansen J p-value should be > 0.10
            hansen_ok = self.results.hansen_j.pvalue > 0.10
            checks.append({"name": "hansen_j_valid", "passed": hansen_ok})
            if not hansen_ok:
                warnings_list.append(f"Hansen J rejected (p={self.results.hansen_j.pvalue:.4f})")

            # AR(2) p-value should be > 0.10
            ar2_ok = self.results.ar2_test.pvalue > 0.10
            checks.append({"name": "ar2_valid", "passed": ar2_ok})
            if not ar2_ok:
                warnings_list.append(f"AR(2) rejected (p={self.results.ar2_test.pvalue:.4f})")

            # Instrument ratio
            if hasattr(self.results, "instrument_ratio"):
                ratio_ok = self.results.instrument_ratio <= 1.0
                checks.append({"name": "instrument_ratio_ok", "passed": ratio_ok})
                if not ratio_ok:
                    warnings_list.append(
                        f"Too many instruments (ratio={self.results.instrument_ratio:.2f})"
                    )

        passed = all(c["passed"] for c in checks)
        return {"passed": passed, "checks": checks, "warnings": warnings_list}

    def compare(self, other: PanelPipeline) -> pd.DataFrame:
        """
        Compare this pipeline with another (for drift detection).

        Parameters
        ----------
        other : PanelPipeline
            Another fitted pipeline to compare against

        Returns
        -------
        pd.DataFrame
            Comparison table with coefficient differences
        """
        if self.results is None or other.results is None:
            raise RuntimeError("Both pipelines must be fitted for comparison.")

        params_self = self.results.params
        params_other = other.results.params

        # Align indices
        common = params_self.index.intersection(params_other.index)

        comparison = pd.DataFrame(
            {
                "current": params_self[common],
                "previous": params_other[common],
                "diff": params_self[common] - params_other[common],
                "pct_change": (
                    (params_self[common] - params_other[common]) / params_other[common].abs()
                )
                * 100,
            }
        )

        return comparison

    def to_dict(self) -> dict[str, Any]:
        """
        Export pipeline configuration and results as dictionary.

        Returns
        -------
        dict
            Dictionary with model config, coefficients, and metadata
        """
        result = {
            "name": self.name,
            "description": self.description,
            "model_class": self.model_class.__name__,
            "model_params": self.model_params,
            "fit_timestamp": self.fit_timestamp,
            "panelbox_version": self._version,
        }

        if self.results is not None:
            if hasattr(self.results, "params"):
                result["params"] = self.results.params.to_dict()
            if hasattr(self.results, "std_errors"):
                result["std_errors"] = self.results.std_errors.to_dict()

        return result

    def to_json(self, filepath: str | Path | None = None) -> str:
        """Export as JSON string, optionally to file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)

        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(json_str)

        return json_str

    def summary(self) -> str:
        """Print pipeline summary."""
        lines = [
            f"PanelPipeline: {self.name}",
            f"  Model: {self.model_class.__name__}",
            f"  Fitted: {self.fit_timestamp or 'No'}",
            f"  Version: {self._version}",
        ]

        if self.results is not None:
            if hasattr(self.results, "nobs"):
                lines.append(f"  Observations: {self.results.nobs}")
            if hasattr(self.results, "params"):
                lines.append(f"  Parameters: {len(self.results.params)}")

        return "\n".join(lines)

    def _log_refit_comparison(self, old_results, new_results):
        """Log parameter changes after refit."""
        if hasattr(old_results, "params") and hasattr(new_results, "params"):
            old_p = old_results.params
            new_p = new_results.params
            common = old_p.index.intersection(new_p.index)
            max_change = (old_p[common] - new_p[common]).abs().max()
            if max_change > 0.5:
                warnings.warn(
                    f"Large parameter change after refit: max |delta| = {max_change:.4f}. "
                    "Consider investigating model stability.",
                    UserWarning,
                    stacklevel=2,
                )

    @staticmethod
    def _get_panelbox_version():
        try:
            import panelbox

            return getattr(panelbox, "__version__", "unknown")
        except Exception:
            return "unknown"

    def __repr__(self):
        fitted = "fitted" if self.results else "not fitted"
        return f"PanelPipeline(name='{self.name}', model={self.model_class.__name__}, {fitted})"
