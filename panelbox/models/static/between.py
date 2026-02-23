"""
Between estimator for panel data.

This module provides the Between estimator which regresses on group means,
capturing variation between entities rather than within entities.
"""

from __future__ import annotations

import logging

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
from panelbox.utils.matrix_ops import compute_ols, compute_vcov_nonrobust

logger = logging.getLogger(__name__)


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
    >>> results = be.fit(cov_type="robust")
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
    **Mathematical Formulation:**

    The Between estimator computes entity-level averages:

        ȳ_i = (1/T_i) Σ_t y_it
        x̄_i = (1/T_i) Σ_t x_it

    Then runs OLS on the cross-section of N entities:

        ȳ_i = β₀ + β₁ x̄_i + ū_i

    where ū_i is the average of residuals u_it for entity i.

    **When to Use:**

    The Between estimator is appropriate when:

    - **Focus on cross-sectional variation**: Interest is in differences across
      entities, not changes within entities over time
    - **Small T, large N**: Many entities but few time periods per entity
    - **Time-invariant regressors**: Variables that don't vary over time can be
      included (unlike Fixed Effects)
    - **Exploratory analysis**: Understanding which entities differ and why

    **Comparison with Other Estimators:**

    | Estimator | Variation Used | Sample Size | Time-Invariant X |
    |-----------|---------------|-------------|------------------|
    | **Between** | Across entities | N | ✅ Allowed |
    | **Fixed Effects** | Within entities | NT | ❌ Dropped |
    | **Random Effects** | Both (weighted) | NT | ✅ Allowed |
    | **Pooled OLS** | Both (unweighted) | NT | ✅ Allowed |

    **Properties:**

    - **Efficiency**: Less efficient than FE or RE when T is large (uses less data)
    - **Consistency**: Consistent if E[ū_i | x̄_i] = 0 (between-entity exogeneity)
    - **R-squared**: Measures between-entity variation only
    - **Degrees of Freedom**: Based on N (not NT), so standard errors larger

    **Interpretation:**

    Between estimates answer: "Do firms with higher average investment also have
    higher average capital?" (cross-sectional question)

    Fixed Effects estimates answer: "When a firm increases investment, does it
    also increase capital?" (within-firm question)

    **Estimation Steps:**

    1. Compute entity-level means for all variables
    2. Run OLS on the N entity means (not NT observations)
    3. Report R² as the between R² (variation explained across entities)

    **Degrees of Freedom:**

    - N observations (one per entity)
    - k parameters (slopes + intercept)
    - df_resid = N - k

    **Standard Errors:**

    All SE types are supported (robust, clustered, etc.) and are applied to
    the N entity-level observations. Clustering by time is possible if needed.

    See Also
    --------
    FixedEffects : Within estimator (uses within-entity variation)
    RandomEffects : GLS estimator (uses both within and between variation)
    PooledOLS : Ignores panel structure entirely

    References
    ----------
    .. [1] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data (2nd ed.). MIT Press. Section 10.2.2.
    .. [2] Baltagi, B. H. (2021). Econometric Analysis of Panel Data
           (6th ed.). Springer. Chapter 2.
    .. [3] Hsiao, C. (2014). Analysis of Panel Data (3rd ed.). Cambridge
           University Press. Chapter 3.
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        weights: np.ndarray | None = None,
    ):
        super().__init__(formula, data, entity_col, time_col, weights)

        # Entity means (computed after fitting)
        self.entity_means: pd.DataFrame | None = None

    def _compute_between_vcov(
        self,
        cov_type: str,
        X_between: np.ndarray,
        resid: np.ndarray,
        df_resid: int,
        unique_entities: np.ndarray,
        **cov_kwds,
    ) -> np.ndarray:
        """Compute covariance matrix for the between estimator."""
        cov_type_lower = cov_type.lower()

        if cov_type_lower == "nonrobust":
            return compute_vcov_nonrobust(X_between, resid, df_resid)

        if cov_type_lower in ["robust", "hc0", "hc1", "hc2", "hc3"]:
            method = "HC1" if cov_type_lower == "robust" else cov_type_lower.upper()
            return robust_covariance(X_between, resid, method=method).cov_matrix

        if cov_type_lower == "clustered":
            return self._compute_clustered_vcov(X_between, resid, cov_kwds)

        if cov_type_lower == "twoway":
            return self._compute_twoway_vcov(X_between, resid, unique_entities, cov_kwds)

        if cov_type_lower == "driscoll_kraay":
            max_lags = cov_kwds.get("max_lags")
            kernel = cov_kwds.get("kernel", "bartlett")
            return driscoll_kraay(
                X_between, resid, unique_entities, max_lags=max_lags, kernel=kernel
            ).cov_matrix

        if cov_type_lower == "newey_west":
            max_lags = cov_kwds.get("max_lags")
            kernel = cov_kwds.get("kernel", "bartlett")
            return newey_west(X_between, resid, max_lags=max_lags, kernel=kernel).cov_matrix

        if cov_type_lower == "pcse":
            return pcse(X_between, resid, unique_entities, unique_entities).cov_matrix

        raise ValueError(
            f"cov_type must be one of: 'nonrobust', 'robust', 'hc0', 'hc1', 'hc2', 'hc3', "
            f"'clustered', 'twoway', 'driscoll_kraay', 'newey_west', 'pcse', got '{cov_type}'"
        )

    def _compute_clustered_vcov(
        self, X_between: np.ndarray, resid: np.ndarray, cov_kwds: dict
    ) -> np.ndarray:
        """Compute clustered covariance for the between estimator."""
        cluster_col = cov_kwds.get("cluster_col")
        if cluster_col is None:
            return robust_covariance(X_between, resid, method="HC1").cov_matrix
        if cluster_col not in self.entity_means.columns:
            raise ValueError(f"cluster_col '{cluster_col}' not found in entity means")
        cluster_ids = self.entity_means[cluster_col].values
        return cluster_by_entity(X_between, resid, cluster_ids, df_correction=True).cov_matrix

    def _compute_twoway_vcov(
        self,
        X_between: np.ndarray,
        resid: np.ndarray,
        unique_entities: np.ndarray,
        cov_kwds: dict,
    ) -> np.ndarray:
        """Compute two-way clustered covariance for the between estimator."""
        cluster_col1 = cov_kwds.get("cluster_col1", "entity")
        cluster_col2 = cov_kwds.get("cluster_col2")
        if cluster_col2 is None:
            raise ValueError("twoway clustering requires cluster_col2 in cov_kwds")
        cluster_ids1 = (
            self.entity_means[cluster_col1].values
            if cluster_col1 in self.entity_means.columns
            else unique_entities
        )
        cluster_ids2 = self.entity_means[cluster_col2].values
        return twoway_cluster(
            X_between, resid, cluster_ids1, cluster_ids2, df_correction=True
        ).cov_matrix

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
        >>> results = model.fit(cov_type="nonrobust")

        >>> # Heteroskedasticity-robust
        >>> results = model.fit(cov_type="robust")
        >>> results = model.fit(cov_type="hc3")

        >>> # Cluster-robust
        >>> results = model.fit(cov_type="clustered")

        >>> # Driscoll-Kraay
        >>> results = model.fit(cov_type="driscoll_kraay", max_lags=3)
        """
        # Build design matrices from original data
        y_orig, X_orig = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )

        # Get variable names
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Get entity identifiers
        entities = self.data.data[self.data.entity_col].values

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
                X_col_idx = j if "Intercept" in var_names else j
                entity_means_dict[var_name] = X_between[:, X_col_idx]

        self.entity_means = pd.DataFrame(entity_means_dict)

        # Estimate coefficients on entity means (OLS)
        beta, resid, fitted = compute_ols(y_between, X_between, self.weights)

        # Degrees of freedom
        n = n_entities
        df_model = k - 1 if "Intercept" in var_names else k
        df_resid = n - k

        if df_resid <= 0:
            raise ValueError(
                f"Insufficient degrees of freedom: df_resid = {df_resid}. "
                f"n_entities={n}, k={k}. Need more entities than parameters."
            )

        # Compute covariance matrix
        vcov = self._compute_between_vcov(
            cov_type, X_between, resid, df_resid, unique_entities, **cov_kwds
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

        # Map fitted values and residuals back to original observations
        fitted_all = np.zeros(len(y_orig))
        resid_all = np.zeros(len(y_orig))
        for i, entity in enumerate(unique_entities):
            mask = entities == entity
            fitted_all[mask] = fitted[i]
            # For between estimator, all observations in entity get same residual
            # (residual from entity mean regression)
            resid_all[mask] = resid[i]

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
            formula_parser=self.formula_parser,
        )

        # Store entity column name for predict(newdata)
        results._entity_col = self.data.entity_col

        # Override predict() to compute group means before prediction
        _results = results  # capture for closure

        def _between_predict(newdata=None):
            if newdata is None:
                return _results.fittedvalues

            parser = _results._formula_parser
            if parser is None:
                from panelbox.core.formula_parser import FormulaParser

                parser = FormulaParser(_results.formula).parse()

            x_vars = parser.regressors

            # Compute group means
            X_means = newdata.groupby(_results._entity_col)[x_vars].mean()

            # Add intercept column if formula has intercept
            if parser.has_intercept:
                X_means.insert(0, "Intercept", 1.0)

            return X_means.values @ _results.params.values

        results.predict = _between_predict

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
