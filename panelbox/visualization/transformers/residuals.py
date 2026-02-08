"""
Data transformer for residual diagnostics.

Converts model results into the data format expected by residual diagnostic charts.
"""

from typing import Any, Dict, Optional

import numpy as np


class ResidualDataTransformer:
    """
    Transform model results to residual chart format.

    Takes panel model results and extracts/computes all necessary
    data for residual diagnostic visualizations.

    Examples
    --------
    >>> from panelbox import FixedEffects
    >>> from panelbox.visualization.transformers import ResidualDataTransformer
    >>>
    >>> model = FixedEffects('y ~ x1 + x2', data, ...)
    >>> results = model.fit()
    >>>
    >>> transformer = ResidualDataTransformer()
    >>> residual_data = transformer.transform(results)
    >>>
    >>> # Use with chart API
    >>> charts = create_residual_diagnostics(residual_data)
    """

    def transform(self, results: Any) -> Dict[str, Any]:
        """
        Transform model results to residual diagnostic data.

        Parameters
        ----------
        results : PanelResults
            Fitted model results object

        Returns
        -------
        dict
            Structured data for residual diagnostics with keys:
            - 'residuals': raw residuals
            - 'fitted': fitted values
            - 'standardized_residuals': standardized residuals
            - 'leverage': hat values (if available)
            - 'cooks_d': Cook's distance (if available)
            - 'time_index': time index (if available)
            - 'entity_id': entity identifiers (if available)
            - 'model_info': model metadata

        Examples
        --------
        >>> data = transformer.transform(results)
        >>> print(data.keys())
        dict_keys(['residuals', 'fitted', 'standardized_residuals', ...])
        """
        # Extract basic residual data
        residuals = self._extract_residuals(results)
        fitted = self._extract_fitted(results)

        # Compute standardized residuals
        standardized_residuals = self._compute_standardized_residuals(residuals, results)

        # Extract or compute leverage and influence measures
        leverage = self._compute_leverage(results)
        cooks_d = self._compute_cooks_distance(results, standardized_residuals, leverage)

        # Extract time/panel structure
        time_index = self._extract_time_index(results)
        entity_id = self._extract_entity_id(results)

        # Extract model information
        model_info = self._extract_model_info(results)

        return {
            "residuals": residuals,
            "fitted": fitted,
            "standardized_residuals": standardized_residuals,
            "leverage": leverage,
            "cooks_d": cooks_d,
            "time_index": time_index,
            "entity_id": entity_id,
            "model_info": model_info,
        }

    def _extract_residuals(self, results: Any) -> np.ndarray:
        """
        Extract residuals from results.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        ndarray
            Residuals array
        """
        if hasattr(results, "resid"):
            return np.asarray(results.resid)
        elif hasattr(results, "residuals"):
            return np.asarray(results.residuals)
        else:
            raise AttributeError("Results object has no residuals attribute")

    def _extract_fitted(self, results: Any) -> np.ndarray:
        """
        Extract fitted values from results.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        ndarray
            Fitted values array
        """
        if hasattr(results, "fittedvalues"):
            return np.asarray(results.fittedvalues)
        elif hasattr(results, "fitted_values"):
            return np.asarray(results.fitted_values)
        elif hasattr(results, "predict"):
            return np.asarray(results.predict())
        else:
            raise AttributeError("Cannot extract fitted values from results")

    def _compute_standardized_residuals(self, residuals: np.ndarray, results: Any) -> np.ndarray:
        """
        Compute standardized residuals.

        Parameters
        ----------
        residuals : ndarray
            Raw residuals
        results : PanelResults
            Model results

        Returns
        -------
        ndarray
            Standardized residuals
        """
        # Try to get standard error from results
        if hasattr(results, "scale"):
            scale = results.scale
        elif hasattr(results, "mse_resid"):
            scale = results.mse_resid
        else:
            # Compute from residuals
            scale = np.var(residuals, ddof=results.df_resid if hasattr(results, "df_resid") else 1)

        return residuals / np.sqrt(scale)

    def _compute_leverage(self, results: Any) -> Optional[np.ndarray]:
        """
        Compute or extract leverage (hat) values.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        ndarray or None
            Leverage values if computable, else None
        """
        # Try to get from results
        if hasattr(results, "get_influence"):
            try:
                influence = results.get_influence()
                if hasattr(influence, "hat_matrix_diag"):
                    return np.asarray(influence.hat_matrix_diag)
            except Exception:
                pass

        # Try to compute from model matrix
        if hasattr(results, "model") and hasattr(results.model, "exog"):
            try:
                X = results.model.exog
                # H = X(X'X)^(-1)X'
                # Diagonal of H is leverage
                XtX_inv = np.linalg.inv(X.T @ X)
                leverage = np.sum((X @ XtX_inv) * X, axis=1)
                return leverage
            except Exception:
                pass

        return None

    def _compute_cooks_distance(
        self, results: Any, standardized_residuals: np.ndarray, leverage: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Compute Cook's distance.

        Parameters
        ----------
        results : PanelResults
            Model results
        standardized_residuals : ndarray
            Standardized residuals
        leverage : ndarray or None
            Leverage values

        Returns
        -------
        ndarray or None
            Cook's distance if computable, else None
        """
        if leverage is None or not isinstance(leverage, np.ndarray):
            return None

        # Try to get from results first
        if hasattr(results, "get_influence"):
            try:
                influence = results.get_influence()
                if hasattr(influence, "cooks_distance"):
                    cooks_d, _ = influence.cooks_distance
                    return np.asarray(cooks_d)
            except Exception:
                pass

        # Compute manually
        # Cook's D = (standardized_resid^2 / p) * (leverage / (1 - leverage)^2)
        p = results.df_model if hasattr(results, "df_model") else len(results.params)

        with np.errstate(divide="ignore", invalid="ignore"):
            cooks_d = (standardized_residuals**2 / p) * (leverage / (1 - leverage) ** 2)
            cooks_d = np.where(np.isfinite(cooks_d), cooks_d, 0)

        return cooks_d

    def _extract_time_index(self, results: Any) -> Optional[np.ndarray]:
        """
        Extract time index from results.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        ndarray or None
            Time index if available
        """
        # Try to get from model data
        if hasattr(results, "model") and hasattr(results.model, "data"):
            data = results.model.data
            if hasattr(data, "row_labels"):
                # MultiIndex case (entity, time)
                if hasattr(data.row_labels, "get_level_values"):
                    try:
                        return data.row_labels.get_level_values(1).values
                    except Exception:
                        pass

        # Try from original data
        if hasattr(results, "_data") and hasattr(results._data, "time_index"):
            return np.asarray(results._data.time_index)

        return None

    def _extract_entity_id(self, results: Any) -> Optional[np.ndarray]:
        """
        Extract entity identifiers from results.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        ndarray or None
            Entity IDs if available
        """
        # Try to get from model data
        if hasattr(results, "model") and hasattr(results.model, "data"):
            data = results.model.data
            if hasattr(data, "row_labels"):
                # MultiIndex case (entity, time)
                if hasattr(data.row_labels, "get_level_values"):
                    try:
                        return data.row_labels.get_level_values(0).values
                    except Exception:
                        pass

        # Try from original data
        if hasattr(results, "_data") and hasattr(results._data, "entity_id"):
            return np.asarray(results._data.entity_id)

        return None

    def _extract_model_info(self, results: Any) -> Dict[str, Any]:
        """
        Extract model metadata.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        dict
            Model information
        """
        info = {}

        # Extract common attributes
        attrs = ["nobs", "df_resid", "df_model", "rsquared", "rsquared_adj", "fvalue", "f_pvalue"]

        for attr in attrs:
            if hasattr(results, attr):
                info[attr] = getattr(results, attr)

        # Get model type
        if hasattr(results, "model"):
            model = results.model
            info["model_type"] = (
                model.__class__.__name__ if hasattr(model, "__class__") else "Unknown"
            )

        # Get parameter names
        if hasattr(results, "params"):
            info["param_names"] = (
                list(results.params.index) if hasattr(results.params, "index") else []
            )
            info["n_params"] = len(results.params)

        return info

    def prepare_qq_data(self, results: Any) -> Dict[str, Any]:
        """
        Prepare data specifically for Q-Q plot.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        dict
            Data for Q-Q plot
        """
        residuals = self._extract_residuals(results)

        return {
            "residuals": residuals,
            "standardized": True,
            "show_confidence": True,
            "confidence_level": 0.95,
        }

    def prepare_residual_fitted_data(self, results: Any) -> Dict[str, Any]:
        """
        Prepare data for residuals vs fitted plot.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        dict
            Data for residual vs fitted plot
        """
        residuals = self._extract_residuals(results)
        fitted = self._extract_fitted(results)

        return {"fitted": fitted, "residuals": residuals, "add_lowess": True, "add_reference": True}

    def prepare_scale_location_data(self, results: Any) -> Dict[str, Any]:
        """
        Prepare data for scale-location plot.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        dict
            Data for scale-location plot
        """
        residuals = self._extract_residuals(results)
        fitted = self._extract_fitted(results)

        return {"fitted": fitted, "residuals": residuals, "add_lowess": True}

    def prepare_leverage_data(self, results: Any) -> Dict[str, Any]:
        """
        Prepare data for residuals vs leverage plot.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        dict
            Data for leverage plot
        """
        residuals = self._extract_residuals(results)
        standardized_residuals = self._compute_standardized_residuals(residuals, results)
        leverage = self._compute_leverage(results)
        cooks_d = self._compute_cooks_distance(results, standardized_residuals, leverage)

        data = {
            "residuals": standardized_residuals,
            "leverage": leverage if leverage is not None else np.zeros_like(residuals),
            "show_contours": leverage is not None,
        }

        if cooks_d is not None:
            data["cooks_d"] = cooks_d

        if hasattr(results, "params"):
            data["params"] = results.params

        return data

    def prepare_timeseries_data(self, results: Any) -> Dict[str, Any]:
        """
        Prepare data for residual time series plot.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        dict
            Data for time series plot
        """
        residuals = self._extract_residuals(results)
        time_index = self._extract_time_index(results)
        entity_id = self._extract_entity_id(results)

        data = {"residuals": residuals, "add_bands": True}

        if time_index is not None:
            data["time_index"] = time_index

        if entity_id is not None:
            data["entity_id"] = entity_id

        return data

    def prepare_distribution_data(self, results: Any) -> Dict[str, Any]:
        """
        Prepare data for residual distribution plot.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        dict
            Data for distribution plot
        """
        residuals = self._extract_residuals(results)

        return {"residuals": residuals, "bins": "auto", "show_kde": True, "show_normal": True}
