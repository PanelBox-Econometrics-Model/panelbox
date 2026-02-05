"""
Between estimator for panel data.

This module provides the Between estimator which regresses on group means,
capturing variation between entities rather than within entities.
"""

from typing import Optional

import numpy as np
import pandas as pd

from panelbox.core.base_model import PanelModel
from panelbox.core.results import PanelResults
from panelbox.standard_errors import (
    cluster_by_entity,
    driscoll_kraay,
    newey_west,
    pcse,
    robust_covariance,
    twoway_cluster,
)
from panelbox.utils.matrix_ops import compute_ols, compute_panel_rsquared, compute_vcov_nonrobust


class BetweenEstimator(PanelModel):
    """
    Between estimator for panel data.

    This estimator regresses on group (entity) means, capturing the variation
    between entities rather than within entities. It answers: "Do entities with
    higher average X also have higher average Y?"

    The between transformation computes group means:
        ȳ_i = β x̄_i + α + ū_i

    where bars denote averages over time for each entity i.

    This estimator is useful when:
    - T (time periods) is small relative to N (entities)
    - Focus is on cross-sectional (between-entity) variation
    - Time-invariant characteristics are of interest

    Contrast with Fixed Effects (within estimator):
    - FE uses deviations from entity means (within variation)
    - BE uses entity means themselves (between variation)

    Parameters
    ----------
    formula : str
        Model formula in R-style syntax (e.g., "y ~ x1 + x2")
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of the column identifying entities
    time_col : str
        Name of the column identifying time periods
    weights : np.ndarray, optional
        Observation weights (applied to entity means)

    Attributes
    ----------
    entity_means : pd.DataFrame, optional
        Entity-level means (after fitting)

    Examples
    --------
    >>> import panelbox as pb
    >>> import pandas as pd
    >>>
    >>> # Load data
    >>> data = pb.load_grunfeld()
    >>>
    >>> # Between estimator
    >>> be = pb.BetweenEstimator("invest ~ value + capital", data, "firm", "year")
    >>> results = be.fit(cov_type='robust')
    >>> print(results.summary())
    >>>
    >>> # Compare with Fixed Effects (within)
    >>> fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    >>> results_fe = fe.fit()
    >>>
    >>> # BE captures between variation, FE captures within variation
    >>> print(f"Between R²: {results.rsquared:.4f}")
    >>> print(f"Within R²: {results_fe.rsquared:.4f}")
    >>>
    >>> # Access entity means
    >>> entity_means = be.entity_means
    >>> print(entity_means.head())

    Notes
    -----
    The Between estimator:
    1. Computes entity-level means for all variables
    2. Runs OLS on the N entity means (not NT observations)
    3. Reports R² as the between R² (variation explained across entities)

    Degrees of freedom:
    - N observations (one per entity)
    - k parameters (slopes + intercept)
    - df_resid = N - k

    Standard errors:
    - All SE types are supported (robust, clustered, etc.)
    - Applied to the N entity-level observations
    - Clustering by time is possible if needed

    References
    ----------
    .. [1] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
       and Panel Data. MIT Press. Section 10.2.2.
    .. [2] Baltagi, B. H. (2013). Econometric Analysis of Panel Data.
       Wiley. Chapter 2.
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(formula, data, entity_col, time_col, weights)

        # Entity means (computed after fitting)
        self.entity_means: Optional[pd.DataFrame] = None

    def fit(self, cov_type: str = "nonrobust", **cov_kwds) -> PanelResults:
        """
        Fit the Between estimator.

        Parameters
        ----------
        cov_type : str, default='nonrobust'
            Type of covariance estimator:
            - 'nonrobust': Classical standard errors
            - 'robust' or 'hc1': Heteroskedasticity-robust (HC1)
            - 'hc0', 'hc2', 'hc3': Other HC variants
            - 'clustered': Cluster-robust (by entity by default, or custom)
            - 'twoway': Two-way clustered (entity and time at group level)
            - 'driscoll_kraay': Driscoll-Kraay (spatial/temporal dependence)
            - 'newey_west': Newey-West HAC
            - 'pcse': Panel-Corrected Standard Errors
        **cov_kwds
            Additional arguments for covariance estimation:
            - cluster_col: For custom clustering
            - max_lags: For Driscoll-Kraay and Newey-West
            - kernel: For HAC estimators ('bartlett', 'parzen', 'quadratic_spectral')

        Returns
        -------
        PanelResults
            Fitted model results

        Examples
        --------
        >>> # Classical standard errors
        >>> results = model.fit(cov_type='nonrobust')

        >>> # Heteroskedasticity-robust
        >>> results = model.fit(cov_type='robust')
        >>> results = model.fit(cov_type='hc3')

        >>> # Cluster-robust
        >>> results = model.fit(cov_type='clustered')

        >>> # Driscoll-Kraay
        >>> results = model.fit(cov_type='driscoll_kraay', max_lags=3)
        """
        # Build design matrices from original data
        y_orig, X_orig = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )

        # Get variable names
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Get entity and time identifiers
        entities = self.data.data[self.data.entity_col].values
        times = self.data.data[self.data.time_col].values

        # Compute entity means (between transformation)
        unique_entities = np.unique(entities)
        n_entities = len(unique_entities)
        k = X_orig.shape[1]

        # Initialize arrays for entity means
        y_between = np.zeros(n_entities)
        X_between = np.zeros((n_entities, k))

        # Compute means for each entity
        for i, entity in enumerate(unique_entities):
            mask = entities == entity
            y_between[i] = y_orig[mask].mean()
            X_between[i] = X_orig[mask].mean(axis=0)

        # Store entity means for user access
        entity_means_dict = {"entity": unique_entities}

        # Add dependent variable mean
        dep_var_name = self.formula_parser.dependent
        entity_means_dict[dep_var_name] = y_between

        # Add independent variable means (excluding intercept)
        for j, var_name in enumerate(var_names):
            if var_name != "Intercept":
                # Find the corresponding column in X_orig
                # var_names includes 'Intercept' if present, so adjust index
                if "Intercept" in var_names:
                    X_col_idx = j
                else:
                    X_col_idx = j
                entity_means_dict[var_name] = X_between[:, X_col_idx]

        self.entity_means = pd.DataFrame(entity_means_dict)

        # Estimate coefficients on entity means (OLS)
        beta, resid, fitted = compute_ols(y_between, X_between, self.weights)

        # Degrees of freedom
        n = n_entities  # Number of entity-level observations
        df_model = k - 1 if "Intercept" in var_names else k  # Slopes only
        df_resid = n - k

        # Ensure df_resid is positive
        if df_resid <= 0:
            raise ValueError(
                f"Insufficient degrees of freedom: df_resid = {df_resid}. "
                f"n_entities={n}, k={k}. Need more entities than parameters."
            )

        # Compute covariance matrix
        cov_type_lower = cov_type.lower()

        if cov_type_lower == "nonrobust":
            vcov = compute_vcov_nonrobust(X_between, resid, df_resid)

        elif cov_type_lower in ["robust", "hc0", "hc1", "hc2", "hc3"]:
            # Map 'robust' to 'hc1'
            method = "HC1" if cov_type_lower == "robust" else cov_type_lower.upper()
            result = robust_covariance(X_between, resid, method=method)
            vcov = result.cov_matrix

        elif cov_type_lower == "clustered":
            # For between estimator, clustering is less common but supported
            # Default: cluster by entity (though each entity appears once)
            # Could cluster by another grouping variable if specified
            cluster_col = cov_kwds.get("cluster_col", None)
            if cluster_col is None:
                # Each entity is its own cluster - equivalent to robust
                result = robust_covariance(X_between, resid, method="HC1")
            else:
                # Use custom clustering variable from entity_means
                if cluster_col not in self.entity_means.columns:
                    raise ValueError(f"cluster_col '{cluster_col}' not found in entity means")
                cluster_ids = self.entity_means[cluster_col].values
                result = cluster_by_entity(X_between, resid, cluster_ids, df_correction=True)
            vcov = result.cov_matrix

        elif cov_type_lower == "twoway":
            # Two-way clustering at entity level
            # This is unusual for between estimator but technically possible
            # Would need entity-level time groupings
            cluster_col1 = cov_kwds.get("cluster_col1", "entity")
            cluster_col2 = cov_kwds.get("cluster_col2", None)

            if cluster_col2 is None:
                raise ValueError("twoway clustering requires cluster_col2 in cov_kwds")

            cluster_ids1 = (
                self.entity_means[cluster_col1].values
                if cluster_col1 in self.entity_means.columns
                else unique_entities
            )
            cluster_ids2 = self.entity_means[cluster_col2].values

            result = twoway_cluster(
                X_between, resid, cluster_ids1, cluster_ids2, df_correction=True
            )
            vcov = result.cov_matrix

        elif cov_type_lower == "driscoll_kraay":
            # Driscoll-Kraay at entity level
            # Use entity index as "time" dimension
            max_lags = cov_kwds.get("max_lags", None)
            kernel = cov_kwds.get("kernel", "bartlett")
            result = driscoll_kraay(
                X_between, resid, unique_entities, max_lags=max_lags, kernel=kernel
            )
            vcov = result.cov_matrix

        elif cov_type_lower == "newey_west":
            # Newey-West HAC
            max_lags = cov_kwds.get("max_lags", None)
            kernel = cov_kwds.get("kernel", "bartlett")
            result = newey_west(X_between, resid, max_lags=max_lags, kernel=kernel)
            vcov = result.cov_matrix

        elif cov_type_lower == "pcse":
            # Panel-Corrected Standard Errors
            # For between estimator, each entity appears once
            # PCSE is less meaningful but technically computable
            result = pcse(X_between, resid, unique_entities, unique_entities)
            vcov = result.cov_matrix

        else:
            raise ValueError(
                f"cov_type must be one of: 'nonrobust', 'robust', 'hc0', 'hc1', 'hc2', 'hc3', "
                f"'clustered', 'twoway', 'driscoll_kraay', 'newey_west', 'pcse', got '{cov_type}'"
            )

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov))

        # Compute R-squared measures
        # For between estimator:
        # - rsquared = between R² (primary measure)
        # - within R² = 0 by construction (no within variation used)
        # - overall R² computed from fitted values mapped back to all observations

        # Between R² (on entity means)
        tss_between = np.sum((y_between - y_between.mean()) ** 2)
        ess_between = np.sum(resid**2)
        rsquared_between = 1 - ess_between / tss_between if tss_between > 0 else 0.0

        # Map fitted values back to original observations for overall R²
        fitted_all = np.zeros(len(y_orig))
        for i, entity in enumerate(unique_entities):
            mask = entities == entity
            fitted_all[mask] = fitted[i]

        resid_all = y_orig - fitted_all

        # Overall R² (on all NT observations)
        tss_overall = np.sum((y_orig - y_orig.mean()) ** 2)
        ess_overall = np.sum(resid_all**2)
        rsquared_overall = 1 - ess_overall / tss_overall if tss_overall > 0 else 0.0

        # Within R² is not meaningful for between estimator
        # (would require comparing within variation, which BE ignores)
        rsquared_within = 0.0

        # Adjusted R-squared (based on between R²)
        rsquared_adj = 1 - (1 - rsquared_between) * (n - 1) / df_resid

        # Create Series/DataFrame with variable names
        params = pd.Series(beta.ravel(), index=var_names)
        std_errors_series = pd.Series(std_errors, index=var_names)
        cov_params = pd.DataFrame(vcov, index=var_names, columns=var_names)

        # Model information
        model_info = {
            "model_type": "Between Estimator",
            "formula": self.formula,
            "cov_type": cov_type,
            "cov_kwds": cov_kwds,
            "entity_effects": False,
            "time_effects": False,
        }

        # Data information
        data_info = {
            "nobs": n,  # Number of entity-level observations
            "n_entities": self.data.n_entities,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": df_resid,
            "entity_index": unique_entities,
            "time_index": None,  # Not applicable for between estimator
        }

        # R-squared dictionary
        rsquared_dict = {
            "rsquared": rsquared_between,  # For BE, R² = between R²
            "rsquared_adj": rsquared_adj,
            "rsquared_within": rsquared_within,
            "rsquared_between": rsquared_between,
            "rsquared_overall": rsquared_overall,
        }

        # Create results object
        results = PanelResults(
            params=params,
            std_errors=std_errors_series,
            cov_params=cov_params,
            resid=resid_all,  # Residuals for all observations
            fittedvalues=fitted_all,  # Fitted values for all observations
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
        )

        # Store results and update state
        self._results = results
        self._fitted = True

        return results

    def _estimate_coefficients(self) -> np.ndarray:
        """
        Estimate coefficients (implementation of abstract method).

        Returns
        -------
        np.ndarray
            Estimated coefficients
        """
        # Build design matrices
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")

        # Get entity identifiers
        entities = self.data.data[self.data.entity_col].values

        # Compute entity means
        unique_entities = np.unique(entities)
        n_entities = len(unique_entities)
        k = X.shape[1]

        y_between = np.zeros(n_entities)
        X_between = np.zeros((n_entities, k))

        for i, entity in enumerate(unique_entities):
            mask = entities == entity
            y_between[i] = y[mask].mean()
            X_between[i] = X[mask].mean(axis=0)

        # OLS on entity means
        beta, _, _ = compute_ols(y_between, X_between, self.weights)
        return beta
