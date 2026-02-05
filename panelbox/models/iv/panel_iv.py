"""
Panel Instrumental Variables (IV) / Two-Stage Least Squares (2SLS) estimator.

This module implements IV/2SLS estimation for panel data models with
endogenous regressors. Supports Pooled, Fixed Effects, and Random Effects
specifications with instrumental variables.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from panelbox.core.base_model import PanelModel
from panelbox.core.results import PanelResults
from panelbox.standard_errors import (
    cluster_by_entity,
    driscoll_kraay,
    robust_covariance,
    twoway_cluster,
)
from panelbox.utils.matrix_ops import compute_rsquared


class PanelIV(PanelModel):
    """
    Panel Instrumental Variables (IV) / Two-Stage Least Squares (2SLS) estimator.

    This estimator handles endogenous regressors using instrumental variables
    in a panel data context. Supports pooled, fixed effects, and random effects
    specifications.

    The formula syntax uses "|" to separate endogenous variables from instruments:
    "y ~ exog_vars + endog_vars | instruments"

    Parameters
    ----------
    formula : str
        Model formula with IV specification:
        "y ~ x1 + x2 + endog1 | z1 + z2 + z3"
        where x1, x2 are exogenous, endog1 is endogenous,
        and z1, z2, z3 are instruments
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of the column identifying entities
    time_col : str
        Name of the column identifying time periods
    model_type : str, default='pooled'
        Panel model type: 'pooled', 'fe' (fixed effects), or 're' (random effects)
    weights : np.ndarray, optional
        Observation weights

    Attributes
    ----------
    endog_formula : str
        Formula for endogenous variables
    instruments_formula : str
        Formula for instruments
    first_stage_results : Dict
        Results from first stage regressions
    weak_instruments : bool
        Flag indicating potential weak instruments

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_grunfeld()
    >>>
    >>> # Pooled IV: invest is endogenous, use lagged value as instrument
    >>> iv = pb.PanelIV(
    ...     "invest ~ capital + value | capital + lag_value",
    ...     data, "firm", "year"
    ... )
    >>> results = iv.fit(cov_type='robust')
    >>> print(results.summary())
    >>>
    >>> # Fixed Effects IV
    >>> iv_fe = pb.PanelIV(
    ...     "y ~ x1 + endog | x1 + z1 + z2",
    ...     data, "firm", "year",
    ...     model_type='fe'
    ... )
    >>> results = iv_fe.fit(cov_type='clustered')

    Notes
    -----
    The 2SLS procedure:
    1. First stage: Regress each endogenous variable on all instruments and exogenous vars
    2. Second stage: Regress y on exogenous vars and fitted values from first stage
    3. Correct standard errors for the two-stage procedure

    References
    ----------
    .. [1] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data. MIT press.
    .. [2] Baltagi, B. H. (2013). Econometric Analysis of Panel Data.
           John Wiley & Sons.
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        model_type: str = "pooled",
        weights: Optional[np.ndarray] = None,
    ):
        # Parse IV formula
        if "|" not in formula:
            raise ValueError(
                "IV formula must contain '|' separator. " "Format: 'y ~ exog + endog | instruments'"
            )

        # Split formula into main and instruments parts
        main_part, instruments_part = formula.split("|", 1)
        main_part = main_part.strip()
        instruments_part = instruments_part.strip()

        # Store original formula
        self.original_formula = formula
        self.instruments_formula = instruments_part
        self.model_type_iv = model_type.lower()

        if self.model_type_iv not in ["pooled", "fe", "re"]:
            raise ValueError(f"model_type must be 'pooled', 'fe', or 're', got '{model_type}'")

        # Initialize with main formula (for parsing)
        super().__init__(main_part, data, entity_col, time_col, weights)

        # Parse instruments
        self.instruments = self._parse_instruments()

        # Attributes to be filled during fit
        self.first_stage_results = {}
        self.weak_instruments = False
        self._fitted_endogenous = None

    def _estimate_coefficients(self) -> np.ndarray:
        """
        Implement abstract method from PanelModel.

        This is called by the base class but we override fit() completely,
        so this is just a placeholder.

        Returns
        -------
        np.ndarray
            Placeholder - not actually used in Panel IV
        """
        raise NotImplementedError("Use fit() method for Panel IV estimation")

    def _get_dataframe(self) -> pd.DataFrame:
        """Get underlying DataFrame from data (handles both DataFrame and PanelData)."""
        if hasattr(self.data, "data"):
            return self.data.data
        return self.data

    def _parse_instruments(self) -> List[str]:
        """Parse instrument variables from formula."""
        # Remove whitespace and split by +
        instruments_str = self.instruments_formula.replace(" ", "")
        instruments = [var.strip() for var in instruments_str.split("+") if var.strip()]

        # Get data columns - PanelData or DataFrame
        if hasattr(self.data, "data"):
            # PanelData object
            data_columns = self.data.data.columns
        else:
            # DataFrame
            data_columns = self.data.columns

        # Check that instruments exist in data
        for inst in instruments:
            if inst not in data_columns:
                raise ValueError(f"Instrument '{inst}' not found in data")

        return instruments

    def _identify_endogenous_and_exogenous(self) -> Tuple[List[str], List[str]]:
        """
        Identify which regressors are endogenous and which are exogenous.

        Exogenous variables appear in both the main formula and instruments.
        Endogenous variables appear only in main formula.

        Returns
        -------
        tuple
            (endogenous_vars, exogenous_vars)
        """
        # Get all regressors from main formula (excluding y)
        all_regressors = self.formula_parser.regressors.copy()

        # Exogenous vars are those that also appear as instruments
        exogenous = [var for var in all_regressors if var in self.instruments]

        # Endogenous vars are the rest
        endogenous = [var for var in all_regressors if var not in self.instruments]

        if len(endogenous) == 0:
            raise ValueError(
                "No endogenous variables identified. "
                "Endogenous variables must appear in main formula but not in instruments."
            )

        # Check identification: need at least as many instruments as endogenous vars
        n_endog = len(endogenous)
        n_instruments = len(self.instruments)
        n_exog = len(exogenous)

        # Total IVs = excluded instruments only
        n_excluded_instruments = n_instruments - n_exog

        if n_excluded_instruments < n_endog:
            raise ValueError(
                f"Model is under-identified: {n_endog} endogenous variables but "
                f"only {n_excluded_instruments} excluded instruments. "
                f"Need at least {n_endog} excluded instruments."
            )

        return endogenous, exogenous

    def _apply_within_transformation(self, X: np.ndarray) -> np.ndarray:
        """Apply within (fixed effects) transformation to X."""
        X_transformed = np.zeros_like(X)

        df = self._get_dataframe()
        entity_col = df[self.data.entity_col].values

        for entity in self.data.entities:
            entity_mask = entity_col == entity
            entity_data = X[entity_mask]

            # Demean
            entity_mean = entity_data.mean(axis=0)
            X_transformed[entity_mask] = entity_data - entity_mean

        return X_transformed

    def _first_stage(self, endogenous_vars: List[str], exogenous_vars: List[str]) -> Dict[str, Any]:
        """
        Run first stage regressions.

        Regress each endogenous variable on all instruments and exogenous variables.

        Parameters
        ----------
        endogenous_vars : list of str
            Names of endogenous variables
        exogenous_vars : list of str
            Names of exogenous variables

        Returns
        -------
        dict
            First stage results including fitted values and F-statistics
        """
        first_stage_results = {}

        # Build instrument matrix: all instruments + exogenous variables
        Z_vars = self.instruments + exogenous_vars
        Z_data = self._get_dataframe()[Z_vars].values

        # Add intercept if pooled or RE
        if self.model_type_iv in ["pooled", "re"]:
            Z = np.column_stack([np.ones(len(Z_data)), Z_data])
        else:
            Z = Z_data

        # Apply FE transformation if needed
        if self.model_type_iv == "fe":
            Z = self._apply_within_transformation(Z)

        # Run first stage for each endogenous variable
        for endog_var in endogenous_vars:
            # Get endogenous variable
            endog_data = self._get_dataframe()[endog_var].values

            # Apply FE transformation if needed
            if self.model_type_iv == "fe":
                endog_data = self._apply_within_transformation(endog_data.reshape(-1, 1)).ravel()

            # OLS: endog = Z * gamma + error
            gamma, residuals, rank, s = np.linalg.lstsq(Z, endog_data, rcond=None)

            # Fitted values
            fitted = Z @ gamma

            # Compute F-statistic for weak instruments test
            # F = R² / (1 - R²) * (n - k) / (k - 1)
            # where k is number of instruments
            n = len(endog_data)
            k = Z.shape[1]

            # R-squared
            ss_res = np.sum((endog_data - fitted) ** 2)
            ss_tot = np.sum((endog_data - endog_data.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # F-statistic (if well-defined)
            if k > 1 and r2 < 1:
                f_stat = (r2 / (1 - r2)) * ((n - k) / (k - 1))
            else:
                f_stat = np.nan

            first_stage_results[endog_var] = {
                "fitted": fitted,
                "gamma": gamma,
                "rsquared": r2,
                "f_statistic": f_stat,
                "residuals": endog_data - fitted,
            }

        return first_stage_results

    def _second_stage(
        self,
        endogenous_vars: List[str],
        exogenous_vars: List[str],
        first_stage_fitted: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run second stage regression.

        Regress y on exogenous vars and fitted endogenous vars from first stage.

        Parameters
        ----------
        endogenous_vars : list of str
            Names of endogenous variables
        exogenous_vars : list of str
            Names of exogenous variables
        first_stage_fitted : dict
            Fitted values from first stage

        Returns
        -------
        tuple
            (params, residuals, fittedvalues, X_matrix)
        """
        # Build X matrix: exogenous + fitted endogenous
        X_parts = []

        # Get DataFrame
        df = self._get_dataframe()

        # Add intercept for pooled/RE
        if self.model_type_iv in ["pooled", "re"]:
            X_parts.append(np.ones((len(df), 1)))

        # Add exogenous variables
        if exogenous_vars:
            X_exog = df[exogenous_vars].values
            X_parts.append(X_exog)

        # Add fitted endogenous variables
        for endog_var in endogenous_vars:
            X_parts.append(first_stage_fitted[endog_var].reshape(-1, 1))

        X = np.column_stack(X_parts)

        # Get y
        y = df[self.formula_parser.dependent].values

        # Apply FE transformation if needed
        if self.model_type_iv == "fe":
            X = self._apply_within_transformation(X)
            y = self._apply_within_transformation(y.reshape(-1, 1)).ravel()

        # Second stage OLS: y = X * beta + u
        params, residuals_lstsq, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Compute residuals and fitted values
        fittedvalues = X @ params
        residuals = y - fittedvalues

        return params, residuals, fittedvalues, X

    def fit(self, cov_type: str = "nonrobust", **cov_kwds) -> PanelResults:
        """
        Fit the Panel IV model using Two-Stage Least Squares.

        Parameters
        ----------
        cov_type : str, default='nonrobust'
            Type of covariance estimator:
            - 'nonrobust': Classical 2SLS standard errors
            - 'robust' or 'hc1': Heteroskedasticity-robust
            - 'clustered': Cluster-robust (by entity)
            - 'twoway': Two-way clustering (entity and time)
            - 'driscoll_kraay': Driscoll-Kraay
        **cov_kwds
            Additional arguments for covariance estimation

        Returns
        -------
        PanelResults
            Estimation results

        Notes
        -----
        Standard errors are corrected for the two-stage procedure.
        """
        # Identify endogenous and exogenous variables
        endogenous_vars, exogenous_vars = self._identify_endogenous_and_exogenous()

        # First stage
        self.first_stage_results = self._first_stage(endogenous_vars, exogenous_vars)

        # Check for weak instruments
        for endog_var, results in self.first_stage_results.items():
            f_stat = results["f_statistic"]
            if not np.isnan(f_stat) and f_stat < 10:
                self.weak_instruments = True
                import warnings

                warnings.warn(
                    f"Potential weak instruments for '{endog_var}': "
                    f"First-stage F-statistic = {f_stat:.2f} (< 10). "
                    f"Inference may be unreliable.",
                    UserWarning,
                )

        # Extract fitted values for second stage
        fitted_endog = {var: results["fitted"] for var, results in self.first_stage_results.items()}

        # Second stage
        params, residuals, fittedvalues, X = self._second_stage(
            endogenous_vars, exogenous_vars, fitted_endog
        )

        # Build parameter names
        param_names = []
        if self.model_type_iv in ["pooled", "re"]:
            param_names.append("Intercept")
        param_names.extend(exogenous_vars)
        param_names.extend(endogenous_vars)

        params_series = pd.Series(params, index=param_names)

        # Compute covariance matrix
        cov_params = self._compute_covariance(X, residuals, cov_type, **cov_kwds)

        # Standard errors
        std_errors = pd.Series(np.sqrt(np.diag(cov_params)), index=param_names)

        # Covariance as DataFrame
        cov_params_df = pd.DataFrame(cov_params, index=param_names, columns=param_names)

        # Compute R-squared
        y = self._get_dataframe()[self.formula_parser.dependent].values
        rsq = compute_rsquared(y, fittedvalues, residuals, True)

        # Adjusted R-squared
        n = len(y)
        k = len(params)
        rsq_adj = 1 - (1 - rsq) * (n - 1) / (n - k) if n > k else rsq

        rsquared_dict = {
            "rsquared": rsq,
            "rsquared_adj": rsq_adj,
            "rsquared_within": np.nan,
            "rsquared_between": np.nan,
            "rsquared_overall": rsq,
        }

        # Model info
        model_info = {
            "model_type": f"Panel IV ({self.model_type_iv.upper()})",
            "formula": self.original_formula,
            "cov_type": cov_type,
            "cov_kwds": cov_kwds,
            "endogenous_vars": endogenous_vars,
            "exogenous_vars": exogenous_vars,
            "instruments": self.instruments,
            "n_instruments": len(self.instruments),
            "n_endogenous": len(endogenous_vars),
            "weak_instruments": self.weak_instruments,
        }

        # Data info
        df = self._get_dataframe()
        entity_index = df[self.data.entity_col].values
        time_index = df[self.data.time_col].values

        data_info = {
            "nobs": len(df),
            "n_entities": self.data.n_entities,
            "n_periods": self.data.n_periods if hasattr(self.data, "n_periods") else None,
            "df_model": len(params),
            "df_resid": len(df) - len(params),
            "entity_index": entity_index,
            "time_index": time_index,
        }

        # Create results object
        results = PanelResults(
            params=params_series,
            std_errors=std_errors,
            cov_params=cov_params_df,
            resid=residuals,
            fittedvalues=fittedvalues,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
        )

        # Add first stage results to results object
        results.first_stage_results = self.first_stage_results

        return results

    def _compute_covariance(
        self, X: np.ndarray, residuals: np.ndarray, cov_type: str, **cov_kwds
    ) -> np.ndarray:
        """
        Compute covariance matrix of parameters.

        For 2SLS, this requires special handling to account for the
        two-stage procedure.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (second stage)
        residuals : np.ndarray
            Second stage residuals
        cov_type : str
            Covariance type
        **cov_kwds
            Additional covariance arguments

        Returns
        -------
        np.ndarray
            Covariance matrix
        """
        n = len(residuals)
        k = X.shape[1]

        # Bread: (X'X)^{-1}
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            XtX_inv = np.linalg.pinv(X.T @ X)

        if cov_type == "nonrobust":
            # Classical 2SLS standard errors
            sigma2 = np.sum(residuals**2) / (n - k)
            cov_params = sigma2 * XtX_inv

        elif cov_type in ["robust", "hc1", "hc0", "hc2", "hc3"]:
            # Robust covariance
            cov_params = robust_covariance(X, residuals, cov_type=cov_type)

        elif cov_type == "clustered":
            # Clustered by entity
            df = self._get_dataframe()
            entity_index = df[self.data.entity_col].values
            cluster_id = cov_kwds.get("cluster", entity_index)
            cov_params = cluster_by_entity(X, residuals, cluster_id)

        elif cov_type == "twoway":
            # Two-way clustering
            df = self._get_dataframe()
            entity_index = df[self.data.entity_col].values
            time_index = df[self.data.time_col].values
            cov_params = twoway_cluster(X, residuals, entity_index, time_index)

        elif cov_type == "driscoll_kraay":
            # Driscoll-Kraay
            df = self._get_dataframe()
            entity_index = df[self.data.entity_col].values
            time_index = df[self.data.time_col].values
            maxlags = cov_kwds.get("maxlags", None)
            cov_params = driscoll_kraay(X, residuals, entity_index, time_index, maxlags=maxlags)

        else:
            raise ValueError(f"Unknown covariance type: {cov_type}")

        return cov_params
