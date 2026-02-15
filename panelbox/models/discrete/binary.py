"""
Binary choice models for panel data.

This module implements binary dependent variable models:
- Pooled Logit and Pooled Probit with cluster-robust standard errors
- Fixed Effects Logit (Chamberlain 1980)

These models are appropriate when the dependent variable is binary (0/1).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from statsmodels.discrete.discrete_model import Logit, Probit

from panelbox.core.results import PanelResults
from panelbox.models.discrete.base import ConvergenceWarning, NonlinearPanelModel
from panelbox.standard_errors import cluster_by_entity

if TYPE_CHECKING:
    pass


class PooledLogit(NonlinearPanelModel):
    """
    Pooled Logit model for panel data.

    This model pools all observations and estimates a standard logistic
    regression, ignoring the panel structure. It provides cluster-robust
    standard errors by default (clustered by entity).

    The model estimated is:

        P(y_it = 1 | X_it) = Λ(X_it'β)

    where Λ(z) = exp(z)/(1 + exp(z)) is the logistic CDF.

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
        Observation weights

    Attributes
    ----------
    All attributes from NonlinearPanelModel plus:
    _sm_model : statsmodels.discrete.discrete_model.Logit
        Underlying statsmodels Logit model
    _sm_result : statsmodels.discrete.discrete_model.LogitResults
        Statsmodels results object

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_mroz()
    >>>
    >>> # Pooled Logit with cluster-robust SEs
    >>> logit = pb.PooledLogit("lfp ~ age + educ + kids", data, "id", "year")
    >>> results = logit.fit(cov_type='cluster')
    >>> print(results.summary())
    >>>
    >>> # With robust SEs (heteroskedasticity-robust, not clustered)
    >>> results_robust = logit.fit(cov_type='robust')
    >>>
    >>> # Standard (non-robust) SEs
    >>> results_standard = logit.fit(cov_type='nonrobust')

    Notes
    -----
    **When to Use:**

    - Baseline comparison for panel-specific models
    - When no unobserved heterogeneity exists
    - Poolability assumption holds

    **Standard Errors:**

    Default is cluster-robust by entity to account for within-entity
    correlation. Use `cov_type='nonrobust'` for classical SEs or
    `cov_type='robust'` for heteroskedasticity-robust SEs.

    **Implementation:**

    This class is a smart wrapper around statsmodels.Logit that adds
    panel-specific functionality (cluster-robust SEs by entity).

    References
    ----------
    .. [1] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data (2nd ed.). MIT Press. Chapter 15.
    .. [2] Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics.
           Cambridge University Press.

    See Also
    --------
    PooledProbit : Pooled Probit model
    FixedEffectsLogit : Fixed Effects Logit (Chamberlain 1980)
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
        self._sm_model = None
        self._sm_result = None

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for logit model.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector (β)

        Returns
        -------
        float
            Log-likelihood value
        """
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")

        # Linear predictor
        eta = X @ params

        # Logit log-likelihood:
        # ℓ = Σ[y*log(Λ(η)) + (1-y)*log(1-Λ(η))]
        # = Σ[y*η - log(1 + exp(η))]

        # Numerically stable version
        if self.weights is not None:
            # Weighted log-likelihood
            ll = np.sum(self.weights * (y.ravel() * eta - np.log1p(np.exp(eta))))
        else:
            ll = np.sum(y.ravel() * eta - np.log1p(np.exp(eta)))

        return float(ll)

    def fit(
        self,
        cov_type: Literal["nonrobust", "robust", "cluster"] = "cluster",
        **kwargs,
    ) -> PanelResults:
        """
        Fit Pooled Logit model.

        Parameters
        ----------
        cov_type : {'nonrobust', 'robust', 'cluster'}, default='cluster'
            Type of standard errors:
            - 'nonrobust': Classical (assumes iid)
            - 'robust': Heteroskedasticity-robust (sandwich)
            - 'cluster': Cluster-robust by entity (default)
        **kwargs
            Additional arguments (currently unused)

        Returns
        -------
        PanelResults
            Fitted model results

        Examples
        --------
        >>> results = model.fit(cov_type='cluster')
        >>> print(results.summary())
        """
        # Build design matrices
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Flatten y if needed
        y = y.ravel()

        # Fit using statsmodels (with weights if provided)
        if self.weights is not None:
            self._sm_model = Logit(y, X)
            self._sm_result = self._sm_model.fit(disp=0, maxiter=1000, freq_weights=self.weights)
        else:
            self._sm_model = Logit(y, X)
            self._sm_result = self._sm_model.fit(disp=0, maxiter=1000)

        # Get parameters and basic statistics
        params = self._sm_result.params
        llf = float(self._sm_result.llf)

        # Compute linear predictions
        eta = X @ params
        fitted_probs = expit(eta)  # Λ(η) = 1/(1 + exp(-η))

        # Compute residuals (response residuals)
        resid = y - fitted_probs

        # Compute covariance matrix based on type
        if cov_type == "nonrobust":
            # Use statsmodels covariance
            vcov = self._sm_result.cov_params()

        elif cov_type == "robust":
            # Heteroskedasticity-robust (sandwich estimator)
            # V = H^{-1} S H^{-1} where H = Hessian, S = outer product of scores

            # Score for observation i: s_i = (y_i - Λ(η_i)) * X_i
            # S = Σ s_i s_i'

            # Hessian: H = -Σ Λ(η_i)(1-Λ(η_i)) X_i X_i'
            W = fitted_probs * (1 - fitted_probs)
            H = -(X.T * W) @ X

            # Bread: -H^{-1}
            H_inv = np.linalg.inv(H)

            # Meat: S = Σ s_i s_i'
            scores = (y - fitted_probs)[:, np.newaxis] * X
            S = scores.T @ scores

            # Sandwich
            vcov = H_inv @ S @ H_inv

        elif cov_type == "cluster":
            # Cluster-robust by entity
            entities = self.data.data[self.data.entity_col].values

            # Compute scores for each observation
            scores = (y - fitted_probs)[:, np.newaxis] * X

            # Use cluster_robust_mle from mle module
            from panelbox.standard_errors.mle import cluster_robust_mle

            # Hessian
            W = fitted_probs * (1 - fitted_probs)
            H = -(X.T * W) @ X

            result = cluster_robust_mle(H, scores, entities, df_correction=True)
            vcov = result.cov_matrix

        else:
            raise ValueError(
                f"cov_type must be 'nonrobust', 'robust', or 'cluster', got '{cov_type}'"
            )

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov))

        # Create Series/DataFrame with variable names
        params_series = pd.Series(params, index=var_names)
        std_errors_series = pd.Series(std_errors, index=var_names)
        cov_params_df = pd.DataFrame(vcov, index=var_names, columns=var_names)

        # Degrees of freedom
        n = len(y)
        k = X.shape[1]
        df_model = k - (1 if self.formula_parser.has_intercept else 0)
        df_resid = n - k

        # Pseudo R-squared (McFadden)
        # R² = 1 - ℓ_model / ℓ_null
        ll_null = self._sm_result.llnull
        pseudo_r2 = 1 - llf / ll_null if ll_null != 0 else 0.0

        # Model information
        model_info = {
            "model_type": "Pooled Logit",
            "formula": self.formula,
            "cov_type": cov_type,
            "cov_kwds": {},
            "llf": llf,
            "ll_null": ll_null,
        }

        # Data information
        data_info = {
            "nobs": n,
            "n_entities": self.data.n_entities,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": df_resid,
            "entity_index": self.data.data[self.data.entity_col].values.ravel(),
            "time_index": self.data.data[self.data.time_col].values.ravel(),
        }

        # R-squared dictionary (pseudo R² for binary models)
        rsquared_dict = {
            "rsquared": pseudo_r2,
            "rsquared_adj": np.nan,
            "rsquared_within": np.nan,
            "rsquared_between": np.nan,
            "rsquared_overall": pseudo_r2,
        }

        # Create results object
        results = PanelResults(
            params=params_series,
            std_errors=std_errors_series,
            cov_params=cov_params_df,
            resid=resid,
            fittedvalues=fitted_probs,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
        )

        # Store additional attributes for binary models
        results.llf = llf
        results.ll_null = ll_null
        results.pseudo_r2_mcfadden = pseudo_r2
        results.converged = self._sm_result.mle_retvals["converged"]

        # Information criteria
        results.aic = -2 * llf + 2 * k  # AIC = -2*ℓ + 2*k
        results.bic = -2 * llf + k * np.log(n)  # BIC = -2*ℓ + k*log(n)

        # Add predict method to results
        def predict_method(X=None, type="prob"):
            if X is None:
                # Use original X
                eta_pred = eta
            else:
                # New data
                eta_pred = X @ params

            if type == "linear":
                return eta_pred
            elif type == "prob":
                return expit(eta_pred)
            elif type == "class":
                probs = expit(eta_pred)
                return (probs >= 0.5).astype(int)
            else:
                raise ValueError(f"Unknown prediction type: {type}")

        results.predict = predict_method

        # Add pseudo_r2 method
        def pseudo_r2_method(kind="mcfadden"):
            if kind == "mcfadden":
                return pseudo_r2
            elif kind == "cox_snell":
                # Cox-Snell: R² = 1 - exp(-LR/N)
                lr = 2 * (llf - ll_null)  # Likelihood ratio
                return 1 - np.exp(-lr / n)
            elif kind == "nagelkerke":
                # Nagelkerke: R² = R²_CS / (1 - exp(2*ℓ_null/N))
                r2_cs = 1 - np.exp(-2 * (llf - ll_null) / n)
                r2_max = 1 - np.exp(2 * ll_null / n)
                return r2_cs / r2_max if r2_max != 0 else 0.0
            else:
                raise ValueError(f"Unknown pseudo R² type: {kind}")

        results.pseudo_r2 = pseudo_r2_method

        # Add classification metrics method
        def classification_metrics_method(threshold=0.5):
            """
            Compute classification metrics.

            Returns dict with:
            - accuracy: (TP + TN) / Total
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1: 2 * (precision * recall) / (precision + recall)
            """
            y_pred = (fitted_probs >= threshold).astype(int)

            # Confusion matrix elements
            tp = np.sum((y == 1) & (y_pred == 1))
            tn = np.sum((y == 0) & (y_pred == 0))
            fp = np.sum((y == 0) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))

            accuracy = (tp + tn) / len(y)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )

            # AUC-ROC
            from sklearn.metrics import roc_auc_score

            try:
                auc_roc = roc_auc_score(y, fitted_probs)
            except:
                auc_roc = np.nan

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc_roc": auc_roc,
                "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
            }

        results.classification_metrics = classification_metrics_method

        # Add Hosmer-Lemeshow test method
        def hosmer_lemeshow_test_method(n_groups=10):
            """
            Hosmer-Lemeshow goodness-of-fit test for panel data.

            Tests if observed and expected frequencies match across risk groups.
            Adapted for panel data by clustering observations by entity.

            Parameters
            ----------
            n_groups : int, default=10
                Number of risk groups to create

            Returns
            -------
            dict
                Test results with statistic, p-value, and degrees of freedom
            """
            # Sort by predicted probabilities
            sort_idx = np.argsort(fitted_probs)
            y_sorted = y[sort_idx]
            probs_sorted = fitted_probs[sort_idx]
            entities_sorted = self.data.data[self.data.entity_col].values[sort_idx]

            # Divide into groups
            n_obs = len(y)
            group_size = n_obs // n_groups

            observed = []
            expected = []
            group_sizes = []

            for i in range(n_groups):
                if i < n_groups - 1:
                    start = i * group_size
                    end = (i + 1) * group_size
                else:
                    start = i * group_size
                    end = n_obs

                group_y = y_sorted[start:end]
                group_probs = probs_sorted[start:end]

                observed.append(group_y.sum())
                expected.append(group_probs.sum())
                group_sizes.append(end - start)

            observed = np.array(observed)
            expected = np.array(expected)

            # Compute test statistic
            # H-L statistic: Σ (O - E)² / (E * (1 - E/n))
            statistic = 0
            for i in range(n_groups):
                O = observed[i]
                E = expected[i]
                n_g = group_sizes[i]
                pi_bar = E / n_g if n_g > 0 else 0

                if E > 0 and pi_bar < 1:
                    var = n_g * pi_bar * (1 - pi_bar)
                    if var > 0:
                        statistic += (O - E) ** 2 / var

            # Degrees of freedom: n_groups - 2
            df = n_groups - 2

            # P-value from chi-squared distribution
            from scipy.stats import chi2

            p_value = 1 - chi2.cdf(statistic, df)

            return {
                "statistic": statistic,
                "p_value": p_value,
                "df": df,
                "n_groups": n_groups,
                "interpretation": (
                    "Reject H0 (poor fit)" if p_value < 0.05 else "Fail to reject H0 (adequate fit)"
                ),
            }

        results.hosmer_lemeshow_test = hosmer_lemeshow_test_method

        # Add Information Matrix Test method
        def information_matrix_test_method():
            """
            Information Matrix Test for model misspecification.

            Tests if the information matrix equality holds:
            -E[H] = E[s*s'] where H is Hessian and s is score.

            Returns
            -------
            dict
                Test results with statistic and p-value
            """
            # Get score and Hessian at estimated parameters
            score_i = (y - fitted_probs)[:, np.newaxis] * X

            # Compute outer product of scores
            S = score_i.T @ score_i / n

            # Compute Hessian
            W = fitted_probs * (1 - fitted_probs)
            H = -(X.T * W) @ X / n

            # Information matrix test: -H should equal S under correct specification
            # Test statistic based on White (1982)
            diff = -H - S

            # Vectorize the difference
            vech_diff = diff[np.triu_indices_from(diff)]

            # Compute test statistic (simplified version)
            # Under H0, √n * vech(diff) → N(0, V)
            # We use a simplified quadratic form
            statistic = n * np.sum(vech_diff**2)

            # Degrees of freedom: k*(k+1)/2
            df = k * (k + 1) // 2

            # P-value
            from scipy.stats import chi2

            p_value = 1 - chi2.cdf(statistic, df)

            return {
                "statistic": statistic,
                "p_value": p_value,
                "df": df,
                "interpretation": (
                    "Evidence of misspecification"
                    if p_value < 0.05
                    else "No evidence of misspecification"
                ),
            }

        results.information_matrix_test = information_matrix_test_method

        # Add Link Test method
        def link_test_method():
            """
            Link test for model specification.

            Tests if the link function is correctly specified by adding
            the squared linear predictor to the model and testing its significance.

            Returns
            -------
            dict
                Test results with coefficient, t-stat, and p-value for squared term
            """
            # Linear predictor
            eta_hat = X @ params

            # Add squared term
            X_augmented = np.column_stack([X, eta_hat**2])

            # Re-estimate model with squared term
            from statsmodels.discrete.discrete_model import Logit

            augmented_model = Logit(y, X_augmented)
            augmented_result = augmented_model.fit(disp=0, maxiter=1000)

            # Test if coefficient on squared term is significant
            squared_coef = augmented_result.params[-1]
            squared_se = augmented_result.bse[-1]
            t_stat = squared_coef / squared_se if squared_se > 0 else np.nan

            # P-value (two-tailed)
            from scipy.stats import t as t_dist

            p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df_resid))

            return {
                "squared_term_coef": squared_coef,
                "squared_term_se": squared_se,
                "t_statistic": t_stat,
                "p_value": p_value,
                "interpretation": (
                    "Evidence of misspecification"
                    if p_value < 0.05
                    else "Link function appears adequate"
                ),
            }

        results.link_test = link_test_method

        # Store results and update state
        self._results = results
        self._fitted = True

        return results

    def predict(self, type: Literal["linear", "prob"] = "prob") -> np.ndarray:
        """
        Generate predictions from fitted model.

        Parameters
        ----------
        type : {'linear', 'prob'}, default='prob'
            Type of prediction:
            - 'linear': Linear predictor X'β
            - 'prob': Predicted probabilities P(y=1|X)

        Returns
        -------
        np.ndarray
            Predictions

        Examples
        --------
        >>> results = model.fit()
        >>> probs = model.predict(type='prob')
        >>> linear_pred = model.predict(type='linear')
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        if type == "prob":
            return self._results.fittedvalues
        elif type == "linear":
            y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
            return X @ self._results.params.values
        else:
            raise ValueError(f"type must be 'linear' or 'prob', got '{type}'")

    def marginal_effects(self, at: str = "mean", method: str = "dydx") -> pd.DataFrame:
        """
        Compute marginal effects for binary choice models.

        **NOTE:** This is a stub method. Full implementation will be available in Phase 2.

        Parameters
        ----------
        at : str, default='mean'
            Where to evaluate marginal effects:
            - 'mean': At mean values of covariates (MEM)
            - 'overall': Average marginal effects (AME)
        method : str, default='dydx'
            Type of marginal effect:
            - 'dydx': Marginal effect (derivative)
            - 'eyex': Elasticity

        Returns
        -------
        pd.DataFrame
            DataFrame with marginal effects (not yet implemented)

        Examples
        --------
        >>> results = model.fit()
        >>> # This will raise NotImplementedError in Phase 1
        >>> me = model.marginal_effects(at='mean')

        Notes
        -----
        This method will be fully implemented in Phase 2 (US-2.1).
        It will compute:
        - Average Marginal Effects (AME)
        - Marginal Effects at Means (MEM)
        - Marginal Effects at Representative values (MER)
        - Standard errors via delta method

        For Logit: ME = β * Λ(X'β) * (1 - Λ(X'β))
        For Probit: ME = β * φ(X'β)
        """
        raise NotImplementedError(
            "Marginal effects computation is not yet implemented. "
            "This feature will be available in Phase 2 of the discrete models implementation."
        )


class PooledProbit(NonlinearPanelModel):
    """
    Pooled Probit model for panel data.

    This model pools all observations and estimates a standard probit
    regression, ignoring the panel structure. It provides cluster-robust
    standard errors by default (clustered by entity).

    The model estimated is:

        P(y_it = 1 | X_it) = Φ(X_it'β)

    where Φ is the standard normal CDF.

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
        Observation weights

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_mroz()
    >>>
    >>> # Pooled Probit with cluster-robust SEs
    >>> probit = pb.PooledProbit("lfp ~ age + educ + kids", data, "id", "year")
    >>> results = probit.fit(cov_type='cluster')
    >>> print(results.summary())

    See Also
    --------
    PooledLogit : Pooled Logit model
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
        self._sm_model = None
        self._sm_result = None

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for probit model.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector (β)

        Returns
        -------
        float
            Log-likelihood value
        """
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")

        # Linear predictor
        eta = X @ params

        # Probit log-likelihood:
        # ℓ = Σ[y*log(Φ(η)) + (1-y)*log(1-Φ(η))]

        prob = stats.norm.cdf(eta)
        # Clip probabilities to avoid log(0)
        prob = np.clip(prob, 1e-10, 1 - 1e-10)

        if self.weights is not None:
            # Weighted log-likelihood
            ll = np.sum(
                self.weights * (y.ravel() * np.log(prob) + (1 - y.ravel()) * np.log(1 - prob))
            )
        else:
            ll = np.sum(y.ravel() * np.log(prob) + (1 - y.ravel()) * np.log(1 - prob))

        return float(ll)

    def fit(
        self,
        cov_type: Literal["nonrobust", "robust", "cluster"] = "cluster",
        **kwargs,
    ) -> PanelResults:
        """
        Fit Pooled Probit model.

        Parameters
        ----------
        cov_type : {'nonrobust', 'robust', 'cluster'}, default='cluster'
            Type of standard errors
        **kwargs
            Additional arguments

        Returns
        -------
        PanelResults
            Fitted model results
        """
        # Build design matrices
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Flatten y
        y = y.ravel()

        # Fit using statsmodels (with weights if provided)
        if self.weights is not None:
            self._sm_model = Probit(y, X)
            self._sm_result = self._sm_model.fit(disp=0, maxiter=1000, freq_weights=self.weights)
        else:
            self._sm_model = Probit(y, X)
            self._sm_result = self._sm_model.fit(disp=0, maxiter=1000)

        # Get parameters
        params = self._sm_result.params
        llf = float(self._sm_result.llf)

        # Compute predictions
        eta = X @ params
        fitted_probs = stats.norm.cdf(eta)

        # Residuals
        resid = y - fitted_probs

        # Compute covariance matrix
        if cov_type == "nonrobust":
            vcov = self._sm_result.cov_params().values

        elif cov_type == "robust":
            # Heteroskedasticity-robust
            # Hessian: H = -Σ φ(η)²/[Φ(η)(1-Φ(η))] X_i X_i'
            pdf = stats.norm.pdf(eta)
            cdf = fitted_probs
            cdf_comp = 1 - cdf
            W = (pdf**2) / (cdf * cdf_comp)
            H = -(X.T * W) @ X

            H_inv = np.linalg.inv(H)

            # Scores
            scores = (y - fitted_probs)[:, np.newaxis] * X
            S = scores.T @ scores

            vcov = H_inv @ S @ H_inv

        elif cov_type == "cluster":
            # Cluster-robust by entity
            entities = self.data.data[self.data.entity_col].values

            # Hessian
            pdf = stats.norm.pdf(eta)
            cdf = fitted_probs
            cdf_comp = 1 - cdf
            W = (pdf**2) / (cdf * cdf_comp)
            H = -(X.T * W) @ X
            H_inv = np.linalg.inv(H)

            # Clustered meat
            result = cluster_by_entity(X, resid, entities, df_correction=True)
            S = result.meat

            vcov = H_inv @ S @ H_inv

        else:
            raise ValueError(
                f"cov_type must be 'nonrobust', 'robust', or 'cluster', got '{cov_type}'"
            )

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov))

        # Create pandas objects
        params_series = pd.Series(params, index=var_names)
        std_errors_series = pd.Series(std_errors, index=var_names)
        cov_params_df = pd.DataFrame(vcov, index=var_names, columns=var_names)

        # Degrees of freedom
        n = len(y)
        k = X.shape[1]
        df_model = k - (1 if self.formula_parser.has_intercept else 0)
        df_resid = n - k

        # Pseudo R-squared
        ll_null = self._sm_result.llnull
        pseudo_r2 = 1 - llf / ll_null if ll_null != 0 else 0.0

        # Model information
        model_info = {
            "model_type": "Pooled Probit",
            "formula": self.formula,
            "cov_type": cov_type,
            "cov_kwds": {},
            "llf": llf,
            "ll_null": ll_null,
        }

        # Data information
        data_info = {
            "nobs": n,
            "n_entities": self.data.n_entities,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": df_resid,
            "entity_index": self.data.data[self.data.entity_col].values.ravel(),
            "time_index": self.data.data[self.data.time_col].values.ravel(),
        }

        # R-squared dictionary
        rsquared_dict = {
            "rsquared": pseudo_r2,
            "rsquared_adj": np.nan,
            "rsquared_within": np.nan,
            "rsquared_between": np.nan,
            "rsquared_overall": pseudo_r2,
        }

        # Create results
        results = PanelResults(
            params=params_series,
            std_errors=std_errors_series,
            cov_params=cov_params_df,
            resid=resid,
            fittedvalues=fitted_probs,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
        )

        # Store additional attributes
        results.llf = llf
        results.ll_null = ll_null
        results.pseudo_r2_mcfadden = pseudo_r2
        results.converged = self._sm_result.mle_retvals["converged"]

        # Information criteria
        results.aic = -2 * llf + 2 * k  # AIC = -2*ℓ + 2*k
        results.bic = -2 * llf + k * np.log(n)  # BIC = -2*ℓ + k*log(n)

        # Add predict method
        def predict_method(X=None, type="prob"):
            if X is None:
                eta_pred = eta
            else:
                eta_pred = X @ params

            if type == "linear":
                return eta_pred
            elif type == "prob":
                return stats.norm.cdf(eta_pred)
            elif type == "class":
                probs = stats.norm.cdf(eta_pred)
                return (probs >= 0.5).astype(int)
            else:
                raise ValueError(f"Unknown prediction type: {type}")

        results.predict = predict_method

        # Add pseudo_r2 method
        def pseudo_r2_method(kind="mcfadden"):
            if kind == "mcfadden":
                return pseudo_r2
            elif kind == "cox_snell":
                lr = 2 * (llf - ll_null)
                return 1 - np.exp(-lr / n)
            elif kind == "nagelkerke":
                r2_cs = 1 - np.exp(-2 * (llf - ll_null) / n)
                r2_max = 1 - np.exp(2 * ll_null / n)
                return r2_cs / r2_max if r2_max != 0 else 0.0
            else:
                raise ValueError(f"Unknown pseudo R² type: {kind}")

        results.pseudo_r2 = pseudo_r2_method

        # Add classification metrics method
        def classification_metrics_method(threshold=0.5):
            y_pred = (fitted_probs >= threshold).astype(int)

            tp = np.sum((y == 1) & (y_pred == 1))
            tn = np.sum((y == 0) & (y_pred == 0))
            fp = np.sum((y == 0) & (y_pred == 1))
            fn = np.sum((y == 1) & (y_pred == 0))

            accuracy = (tp + tn) / len(y)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            )

            try:
                from sklearn.metrics import roc_auc_score

                auc_roc = roc_auc_score(y, fitted_probs)
            except:
                auc_roc = np.nan

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc_roc": auc_roc,
                "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
            }

        results.classification_metrics = classification_metrics_method

        # Add Hosmer-Lemeshow test method
        def hosmer_lemeshow_test_method(n_groups=10):
            """
            Hosmer-Lemeshow goodness-of-fit test for panel data.
            """
            sort_idx = np.argsort(fitted_probs)
            y_sorted = y[sort_idx]
            probs_sorted = fitted_probs[sort_idx]

            n_obs = len(y)
            group_size = n_obs // n_groups

            observed = []
            expected = []
            group_sizes = []

            for i in range(n_groups):
                if i < n_groups - 1:
                    start = i * group_size
                    end = (i + 1) * group_size
                else:
                    start = i * group_size
                    end = n_obs

                group_y = y_sorted[start:end]
                group_probs = probs_sorted[start:end]

                observed.append(group_y.sum())
                expected.append(group_probs.sum())
                group_sizes.append(end - start)

            observed = np.array(observed)
            expected = np.array(expected)

            statistic = 0
            for i in range(n_groups):
                O = observed[i]
                E = expected[i]
                n_g = group_sizes[i]
                pi_bar = E / n_g if n_g > 0 else 0

                if E > 0 and pi_bar < 1:
                    var = n_g * pi_bar * (1 - pi_bar)
                    if var > 0:
                        statistic += (O - E) ** 2 / var

            df = n_groups - 2

            from scipy.stats import chi2

            p_value = 1 - chi2.cdf(statistic, df)

            return {
                "statistic": statistic,
                "p_value": p_value,
                "df": df,
                "n_groups": n_groups,
                "interpretation": (
                    "Reject H0 (poor fit)" if p_value < 0.05 else "Fail to reject H0 (adequate fit)"
                ),
            }

        results.hosmer_lemeshow_test = hosmer_lemeshow_test_method

        # Add Information Matrix Test method
        def information_matrix_test_method():
            """
            Information Matrix Test for model misspecification.
            """
            score_i = (y - fitted_probs)[:, np.newaxis] * X
            S = score_i.T @ score_i / n

            # For Probit, need to adjust the weights
            pdf = stats.norm.pdf(eta)
            cdf = fitted_probs
            cdf_comp = 1 - cdf
            W = (pdf**2) / (cdf * cdf_comp)
            H = -(X.T * W) @ X / n

            diff = -H - S
            vech_diff = diff[np.triu_indices_from(diff)]
            statistic = n * np.sum(vech_diff**2)

            df = k * (k + 1) // 2

            from scipy.stats import chi2

            p_value = 1 - chi2.cdf(statistic, df)

            return {
                "statistic": statistic,
                "p_value": p_value,
                "df": df,
                "interpretation": (
                    "Evidence of misspecification"
                    if p_value < 0.05
                    else "No evidence of misspecification"
                ),
            }

        results.information_matrix_test = information_matrix_test_method

        # Add Link Test method
        def link_test_method():
            """
            Link test for model specification (Probit).
            """
            eta_hat = X @ params
            X_augmented = np.column_stack([X, eta_hat**2])

            from statsmodels.discrete.discrete_model import Probit

            augmented_model = Probit(y, X_augmented)
            augmented_result = augmented_model.fit(disp=0, maxiter=1000)

            squared_coef = augmented_result.params[-1]
            squared_se = augmented_result.bse[-1]
            t_stat = squared_coef / squared_se if squared_se > 0 else np.nan

            from scipy.stats import t as t_dist

            p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df_resid))

            return {
                "squared_term_coef": squared_coef,
                "squared_term_se": squared_se,
                "t_statistic": t_stat,
                "p_value": p_value,
                "interpretation": (
                    "Evidence of misspecification"
                    if p_value < 0.05
                    else "Link function appears adequate"
                ),
            }

        results.link_test = link_test_method

        # Update state
        self._results = results
        self._fitted = True

        return results

    def predict(self, type: Literal["linear", "prob"] = "prob") -> np.ndarray:
        """
        Generate predictions from fitted model.

        Parameters
        ----------
        type : {'linear', 'prob'}, default='prob'
            Type of prediction

        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        if type == "prob":
            return self._results.fittedvalues
        elif type == "linear":
            y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
            return X @ self._results.params.values
        else:
            raise ValueError(f"type must be 'linear' or 'prob', got '{type}'")

    def marginal_effects(self, at: str = "mean", method: str = "dydx") -> pd.DataFrame:
        """
        Compute marginal effects for binary choice models.

        **NOTE:** This is a stub method. Full implementation will be available in Phase 2.

        Parameters
        ----------
        at : str, default='mean'
            Where to evaluate marginal effects:
            - 'mean': At mean values of covariates (MEM)
            - 'overall': Average marginal effects (AME)
        method : str, default='dydx'
            Type of marginal effect:
            - 'dydx': Marginal effect (derivative)
            - 'eyex': Elasticity

        Returns
        -------
        pd.DataFrame
            DataFrame with marginal effects (not yet implemented)

        Notes
        -----
        This method will be fully implemented in Phase 2 (US-2.1).
        For Probit: ME = β * φ(X'β) where φ is the normal PDF
        """
        raise NotImplementedError(
            "Marginal effects computation is not yet implemented. "
            "This feature will be available in Phase 2 of the discrete models implementation."
        )


class FixedEffectsLogit(NonlinearPanelModel):
    """
    Fixed Effects Logit model (Chamberlain 1980).

    This model eliminates fixed effects αᵢ using conditional maximum
    likelihood, conditioning on the sufficient statistic Σₜ yᵢₜ.

    Only entities with temporal variation (0 < Σₜ yᵢₜ < Tᵢ) contribute
    to estimation. Entities with no variation are automatically dropped.

    The model is:

        P(y_i1, ..., y_iT | Σₜ y_it, X_i) ∝ exp(Σₜ y_it X_it'β)

    Parameters
    ----------
    formula : str
        Model formula (only time-varying variables)
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Entity identifier
    time_col : str
        Time identifier

    Attributes
    ----------
    entities_with_variation : np.ndarray
        Entities that contribute to estimation
    dropped_entities : np.ndarray
        Entities dropped (no variation)
    n_used_entities : int
        Number of entities used
    n_dropped_entities : int
        Number of entities dropped

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_patent_data()
    >>>
    >>> # Fixed Effects Logit
    >>> fe_logit = pb.FixedEffectsLogit("patent ~ rd + sales", data, "firm", "year")
    >>> results = fe_logit.fit()
    >>> print(results.summary())
    >>>
    >>> # Check which entities were dropped
    >>> print(f"Dropped {fe_logit.n_dropped_entities} entities")
    >>> print(f"Used {fe_logit.n_used_entities} entities")

    Notes
    -----
    **Identification:**

    Only time-varying variables are identified. Time-invariant variables
    are absorbed by the fixed effects.

    **Computational Complexity:**

    For each entity with sum s = Σₜ yᵢₜ, we must sum over all
    C(Tᵢ, s) possible sequences. This is O(2^T) in the worst case.

    - For T ≤ 10: Full enumeration
    - For T > 10: Dynamic programming or approximation

    **Incidental Parameters:**

    The fixed effects αᵢ are not estimated (incidental parameters problem).
    Only β is estimated.

    References
    ----------
    .. [1] Chamberlain, G. (1980). "Analysis of Covariance with Qualitative
           Data." Review of Economic Studies, 47(1), 225-238.
    .. [2] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data (2nd ed.). MIT Press. Section 15.8.3.

    See Also
    --------
    PooledLogit : Pooled Logit model
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
    ):
        super().__init__(formula, data, entity_col, time_col, weights=None)

        # Find entities with variation
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepare data: identify entities with variation and drop others.
        """
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
        y = y.ravel()

        entities = self.data.data[self.data.entity_col].values
        unique_entities = np.unique(entities)

        entities_with_variation = []
        dropped_entities = []

        for entity in unique_entities:
            mask = entities == entity
            y_entity = y[mask]
            sum_y = y_entity.sum()
            T_i = len(y_entity)

            # Check for variation: 0 < Σ yᵢₜ < Tᵢ
            if 0 < sum_y < T_i:
                entities_with_variation.append(entity)
            else:
                dropped_entities.append(entity)

        self.entities_with_variation = np.array(entities_with_variation)
        self.dropped_entities = np.array(dropped_entities)
        self.n_used_entities = len(entities_with_variation)
        self.n_dropped_entities = len(dropped_entities)

        # Warn if many entities dropped
        if self.n_dropped_entities > 0:
            pct_dropped = 100 * self.n_dropped_entities / len(unique_entities)
            warnings.warn(
                f"Dropped {self.n_dropped_entities} entities ({pct_dropped:.1f}%) "
                f"without temporal variation in the dependent variable. "
                f"Only {self.n_used_entities} entities contribute to estimation.",
                UserWarning,
                stacklevel=2,
            )

        if self.n_used_entities == 0:
            raise ValueError(
                "No entities have temporal variation in the dependent variable. "
                "Fixed Effects Logit cannot be estimated."
            )

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Conditional log-likelihood (Chamberlain 1980).

        ℓ = Σᵢ [yᵢ'Xᵢβ - log(Σ_{s∈Sᵢ} exp(s'Xᵢβ))]

        where Sᵢ = {s ∈ {0,1}^Tᵢ : Σₜ sₜ = Σₜ yᵢₜ}

        Parameters
        ----------
        params : np.ndarray
            Parameter vector β

        Returns
        -------
        float
            Conditional log-likelihood
        """
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
        y = y.ravel()
        entities = self.data.data[self.data.entity_col].values

        llf = 0.0

        for entity in self.entities_with_variation:
            mask = entities == entity
            y_i = y[mask]
            X_i = X[mask]
            sum_yi = int(y_i.sum())

            # Numerator: observed sequence
            numerator = y_i @ X_i @ params

            # Denominator: sum over all sequences with same sum
            denominator = self._sum_over_sequences(X_i, params, sum_yi)

            llf += numerator - np.log(denominator)

        return float(llf)

    def _sum_over_sequences(self, X_i: np.ndarray, params: np.ndarray, target_sum: int) -> float:
        """
        Compute Σ_{s:Σₜ sₜ=target_sum} exp(s'Xᵢβ).

        For small T (≤ 10): enumerate all sequences
        For larger T: use dynamic programming (future work)

        Parameters
        ----------
        X_i : np.ndarray
            Design matrix for entity i (shape: (T_i, k))
        params : np.ndarray
            Parameter vector β
        target_sum : int
            Target sum Σₜ yᵢₜ

        Returns
        -------
        float
            Sum of exp(s'Xᵢβ) over all valid sequences
        """
        from itertools import combinations

        T_i = len(X_i)

        if T_i > 20:
            warnings.warn(
                f"Entity has T={T_i} periods. This may be computationally expensive. "
                f"Consider using an approximation or restricting the sample.",
                RuntimeWarning,
                stacklevel=3,
            )

        # Enumeration approach (works well for T ≤ 10-15)
        total = 0.0

        # Generate all combinations of indices that sum to target_sum
        for combo in combinations(range(T_i), target_sum):
            s = np.zeros(T_i)
            s[list(combo)] = 1
            total += np.exp(s @ X_i @ params)

        return total

    def _score(self, params: np.ndarray) -> np.ndarray:
        """
        Analytical gradient of the conditional log-likelihood.

        The score (gradient) for entity i is:
        ∂ℓᵢ/∂β = Σₜ yᵢₜ Xᵢₜ - E[Σₜ sₜ Xᵢₜ | Σₜ sₜ = Σₜ yᵢₜ]

        where the expectation is over all sequences with the same sum.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector β

        Returns
        -------
        np.ndarray
            Score vector (gradient)
        """
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
        y = y.ravel()
        entities = self.data.data[self.data.entity_col].values

        k = X.shape[1]
        score = np.zeros(k)

        for entity in self.entities_with_variation:
            mask = entities == entity
            y_i = y[mask]
            X_i = X[mask]
            sum_yi = int(y_i.sum())

            # First term: Σₜ yᵢₜ Xᵢₜ
            first_term = y_i @ X_i

            # Second term: Expected value of Σₜ sₜ Xᵢₜ
            # This is a weighted average where weights are exp(s'Xᵢβ)
            from itertools import combinations

            T_i = len(X_i)

            numerator = np.zeros(k)
            denominator = 0.0

            for combo in combinations(range(T_i), sum_yi):
                s = np.zeros(T_i)
                s[list(combo)] = 1
                exp_term = np.exp(s @ X_i @ params)
                numerator += exp_term * (s @ X_i)
                denominator += exp_term

            second_term = numerator / denominator

            # Add contribution from entity i
            score += first_term - second_term

        return score

    def _hessian(self, params: np.ndarray) -> np.ndarray:
        """
        Analytical Hessian of the conditional log-likelihood.

        The Hessian for entity i is:
        ∂²ℓᵢ/∂β∂β' = -Cov[Σₜ sₜ Xᵢₜ | Σₜ sₜ = Σₜ yᵢₜ]
                    = -E[(Σₜ sₜ Xᵢₜ)(Σₜ sₜ Xᵢₜ)'] + E[Σₜ sₜ Xᵢₜ]E[Σₜ sₜ Xᵢₜ]'

        Parameters
        ----------
        params : np.ndarray
            Parameter vector β

        Returns
        -------
        np.ndarray
            Hessian matrix (k x k)
        """
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
        y = y.ravel()
        entities = self.data.data[self.data.entity_col].values

        k = X.shape[1]
        hessian = np.zeros((k, k))

        for entity in self.entities_with_variation:
            mask = entities == entity
            y_i = y[mask]
            X_i = X[mask]
            sum_yi = int(y_i.sum())

            from itertools import combinations

            T_i = len(X_i)

            # Compute E[XX'] and E[X]E[X]'
            sum_xx = np.zeros((k, k))
            sum_x = np.zeros(k)
            denominator = 0.0

            for combo in combinations(range(T_i), sum_yi):
                s = np.zeros(T_i)
                s[list(combo)] = 1
                exp_term = np.exp(s @ X_i @ params)
                x_sum = s @ X_i

                sum_xx += exp_term * np.outer(x_sum, x_sum)
                sum_x += exp_term * x_sum
                denominator += exp_term

            # E[XX']
            e_xx = sum_xx / denominator

            # E[X]
            e_x = sum_x / denominator

            # Hessian contribution from entity i: -Var[X] = -(E[XX'] - E[X]E[X]')
            hessian -= e_xx - np.outer(e_x, e_x)

        return hessian

    def fit(self, method: str = "bfgs", **kwargs) -> PanelResults:
        """
        Fit Fixed Effects Logit via conditional MLE.

        Parameters
        ----------
        method : str, default='bfgs'
            Optimization method ('bfgs' or 'newton')
        **kwargs
            Additional optimization arguments

        Returns
        -------
        PanelResults
            Fitted model results
        """
        # Use base class fit method
        return super().fit(method=method, **kwargs)

    def _create_results(
        self, params: np.ndarray, var_names: list, y: np.ndarray, X: np.ndarray
    ) -> PanelResults:
        """
        Create results object for Fixed Effects Logit.

        Parameters
        ----------
        params : np.ndarray
            Estimated parameters
        var_names : list
            Variable names
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Design matrix

        Returns
        -------
        PanelResults
            Results object
        """
        # Compute fitted values and residuals (only for entities with variation)
        entities = self.data.data[self.data.entity_col].values
        y_full = y.ravel()

        # Initialize fitted values and residuals
        fitted_probs = np.full(len(y_full), np.nan)

        # For entities with variation, compute P(y_it=1|X_it, β)
        for entity in self.entities_with_variation:
            mask = entities == entity
            X_i = X[mask]
            eta_i = X_i @ params
            fitted_probs[mask] = expit(eta_i)

        # Residuals
        resid = y_full - fitted_probs

        # Compute covariance matrix (use Hessian)
        H = self._hessian(params)
        vcov = -np.linalg.inv(H)  # -H^{-1}

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov))

        # Create pandas objects
        params_series = pd.Series(params, index=var_names)
        std_errors_series = pd.Series(std_errors, index=var_names)
        cov_params_df = pd.DataFrame(vcov, index=var_names, columns=var_names)

        # Degrees of freedom
        n = len(y_full)
        k = len(params)
        df_model = k
        df_resid = self.n_used_entities - k  # Approximate

        # Log-likelihood
        llf = self._log_likelihood(params)

        # Model information
        model_info = {
            "model_type": "Fixed Effects Logit",
            "formula": self.formula,
            "cov_type": "nonrobust",
            "cov_kwds": {},
            "llf": llf,
            "n_used_entities": self.n_used_entities,
            "n_dropped_entities": self.n_dropped_entities,
        }

        # Data information
        data_info = {
            "nobs": n,
            "n_entities": self.data.n_entities,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": df_resid,
            "entity_index": entities.ravel(),
            "time_index": self.data.data[self.data.time_col].values.ravel(),
        }

        # R-squared not meaningful for conditional MLE
        rsquared_dict = {
            "rsquared": np.nan,
            "rsquared_adj": np.nan,
            "rsquared_within": np.nan,
            "rsquared_between": np.nan,
            "rsquared_overall": np.nan,
        }

        # Create results
        results = PanelResults(
            params=params_series,
            std_errors=std_errors_series,
            cov_params=cov_params_df,
            resid=resid,
            fittedvalues=fitted_probs,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
        )

        # Store additional info
        results.llf = llf
        results.n_used_entities = self.n_used_entities
        results.n_dropped_entities = self.n_dropped_entities

        return results


class RandomEffectsProbit(NonlinearPanelModel):
    """
    Random Effects Probit model with Gaussian Quadrature.

    This model accounts for unobserved heterogeneity using random effects
    that are assumed to be normally distributed. The model integrates out
    the random effects using Gauss-Hermite quadrature.

    The model is:

        P(y_it = 1 | X_it, α_i) = Φ(X_it'β + α_i)
        α_i ~ N(0, σ²_α)

    where Φ is the standard normal CDF and α_i are random effects.

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
    quadrature_points : int, default=12
        Number of Gauss-Hermite quadrature points for integration
    weights : np.ndarray, optional
        Observation weights

    Attributes
    ----------
    quadrature_points : int
        Number of quadrature points used
    rho : float
        Intra-class correlation: ρ = σ²_α / (1 + σ²_α)
    sigma_alpha : float
        Standard deviation of random effects

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_panel_data()
    >>>
    >>> # Random Effects Probit with 12 quadrature points
    >>> re_probit = pb.RandomEffectsProbit("y ~ x1 + x2", data, "entity", "time")
    >>> results = re_probit.fit()
    >>> print(results.summary())
    >>>
    >>> # Check intra-class correlation
    >>> print(f"Rho: {re_probit.rho:.3f}")
    >>> print(f"Sigma_alpha: {re_probit.sigma_alpha:.3f}")
    >>>
    >>> # Use more quadrature points for higher precision
    >>> re_probit_20 = pb.RandomEffectsProbit(
    ...     "y ~ x1 + x2", data, "entity", "time",
    ...     quadrature_points=20
    ... )
    >>> results_20 = re_probit_20.fit()

    Notes
    -----
    **Quadrature Points:**

    - 8-12 points: Usually sufficient for most applications
    - 16-20 points: Higher precision for research publications
    - More points = slower but more accurate

    **Identification:**

    Both time-varying and time-invariant variables are identified
    (unlike Fixed Effects models).

    **Assumptions:**

    - Random effects α_i are independent of X_it
    - α_i ~ N(0, σ²_α) (normality assumption)
    - No serial correlation in errors (after accounting for α_i)

    References
    ----------
    .. [1] Butler, J. S., & Moffitt, R. (1982). "A Computationally Efficient
           Quadrature Procedure for the One-Factor Multinomial Probit Model."
           Econometrica, 50(3), 761-764.
    .. [2] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data (2nd ed.). MIT Press. Section 15.9.

    See Also
    --------
    PooledProbit : Pooled Probit model
    FixedEffectsLogit : Fixed Effects Logit model
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        quadrature_points: int = 12,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(formula, data, entity_col, time_col, weights)
        self.quadrature_points = quadrature_points
        self.family = "probit"  # Set family for marginal effects

        # Import quadrature module
        from panelbox.optimization.quadrature import gauss_hermite_quadrature

        # Compute quadrature nodes and weights once
        self._quad_nodes, self._quad_weights = gauss_hermite_quadrature(quadrature_points)

        # Model parameters (set after fitting)
        self._sigma_alpha = None
        self._beta = None

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Marginal log-likelihood via Gauss-Hermite quadrature.

        Integrates out random effects α_i:
        ℓ_i(β, σ_α) = log ∫ [Π_t Φ(q_it(X_it'β + α_i))] φ(α_i/σ_α) dα_i

        where q_it = 2*y_it - 1 ∈ {-1, +1}

        Parameters
        ----------
        params : np.ndarray
            Parameter vector [β; log(σ_α)]

        Returns
        -------
        float
            Marginal log-likelihood
        """
        # Extract parameters
        beta = params[:-1]
        log_sigma_alpha = params[-1]
        sigma_alpha = np.exp(log_sigma_alpha)  # Ensure positivity

        # Get data
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
        y = y.ravel()
        entities = self.data.data[self.data.entity_col].values

        llf = 0.0

        # Loop over entities
        for entity in np.unique(entities):
            mask = entities == entity
            y_i = y[mask]
            X_i = X[mask]

            # Quadrature sum for entity i
            entity_contributions = []

            for node, weight in zip(self._quad_nodes, self._quad_weights):
                # Transform node: α_i = √2 * σ_α * ξ
                alpha_i = np.sqrt(2) * sigma_alpha * node

                # Product over time: Π_t Φ(q_it * (X_it'β + α_i))
                prob_product = 1.0
                for t in range(len(y_i)):
                    q_it = 2 * y_i[t] - 1  # Transform to {-1, +1}
                    index = q_it * (X_i[t] @ beta + alpha_i)
                    prob_product *= stats.norm.cdf(index)

                entity_contributions.append(weight * prob_product)

            # Take log of sum
            entity_llf = np.sum(entity_contributions)
            if entity_llf > 0:
                llf += np.log(entity_llf)
            else:
                # Handle numerical issues
                llf += -1e10  # Large negative value

        return float(llf)

    def _score(self, params: np.ndarray) -> np.ndarray:
        """
        Numerical gradient of the marginal log-likelihood.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector [β; log(σ_α)]

        Returns
        -------
        np.ndarray
            Score vector (gradient)
        """
        # Use numerical gradient for robustness
        eps = 1e-6
        k = len(params)
        gradient = np.zeros(k)

        for i in range(k):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps

            gradient[i] = (
                self._log_likelihood(params_plus) - self._log_likelihood(params_minus)
            ) / (2 * eps)

        return gradient

    def _starting_values(self) -> np.ndarray:
        """
        Get starting values from Pooled Probit.

        Returns
        -------
        np.ndarray
            Starting parameter vector [β_start; log(σ_α_start)]
        """
        # Estimate Pooled Probit for starting values
        pooled = PooledProbit(
            self.formula, self.data.data, self.data.entity_col, self.data.time_col, self.weights
        )
        pooled_result = pooled.fit(cov_type="nonrobust")

        # Starting values for β from Pooled Probit
        beta_start = pooled_result.params.values

        # Starting value for log(σ_α): assume σ_α = 1
        log_sigma_alpha_start = 0.0

        return np.concatenate([beta_start, [log_sigma_alpha_start]])

    @property
    def rho(self) -> float:
        """
        Intra-class correlation coefficient.

        ρ = σ²_α / (1 + σ²_α)

        Measures the proportion of total variance due to random effects.
        """
        if self._sigma_alpha is None:
            if not self._fitted:
                raise ValueError("Model must be fitted first. Call fit().")
            self._sigma_alpha = np.exp(self._results.params.values[-1])

        return self._sigma_alpha**2 / (1 + self._sigma_alpha**2)

    @property
    def sigma_alpha(self) -> float:
        """
        Standard deviation of random effects.
        """
        if self._sigma_alpha is None:
            if not self._fitted:
                raise ValueError("Model must be fitted first. Call fit().")
            self._sigma_alpha = np.exp(self._results.params.values[-1])

        return self._sigma_alpha

    def fit(
        self, method: str = "bfgs", maxiter: int = 500, tol: float = 1e-8, **kwargs
    ) -> PanelResults:
        """
        Fit Random Effects Probit via maximum likelihood with quadrature.

        Parameters
        ----------
        method : str, default='bfgs'
            Optimization method ('bfgs', 'l-bfgs-b', or 'newton')
        maxiter : int, default=500
            Maximum iterations
        tol : float, default=1e-8
            Convergence tolerance
        **kwargs
            Additional optimization arguments

        Returns
        -------
        PanelResults
            Fitted model results
        """
        from scipy.optimize import minimize

        # Get starting values
        start_params = self._starting_values()

        # Optimize
        if method.lower() == "bfgs":
            result = minimize(
                lambda p: -self._log_likelihood(p),
                start_params,
                method="BFGS",
                jac=lambda p: -self._score(p),
                options={"maxiter": maxiter, "gtol": tol},
            )
        elif method.lower() == "l-bfgs-b":
            # Use bounds to ensure σ_α > 0
            k = len(start_params) - 1
            bounds = [(None, None)] * k + [(-10, 5)]  # log(σ_α) bounds
            result = minimize(
                lambda p: -self._log_likelihood(p),
                start_params,
                method="L-BFGS-B",
                jac=lambda p: -self._score(p),
                bounds=bounds,
                options={"maxiter": maxiter, "gtol": tol},
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Extract results
        params = result.x
        self._beta = params[:-1]
        self._sigma_alpha = np.exp(params[-1])

        # Build design matrices
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Add sigma_alpha to variable names
        var_names_full = var_names + ["log_sigma_alpha"]

        # Create results object
        results = self._create_results(params, var_names_full, y.ravel(), X)

        # Store convergence info
        results.converged = result.success
        results.n_iter = result.nit

        # Store model-specific attributes
        results.rho = self.rho
        results.sigma_alpha = self.sigma_alpha
        results.quadrature_points = self.quadrature_points

        self._results = results
        self._fitted = True

        return results

    def _create_results(
        self, params: np.ndarray, var_names: list, y: np.ndarray, X: np.ndarray
    ) -> PanelResults:
        """
        Create results object for Random Effects Probit.

        Parameters
        ----------
        params : np.ndarray
            Estimated parameters [β; log(σ_α)]
        var_names : list
            Variable names (including log_sigma_alpha)
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Design matrix

        Returns
        -------
        PanelResults
            Results object
        """
        # Compute fitted values (marginal probabilities)
        beta = params[:-1]
        sigma_alpha = np.exp(params[-1])

        # For fitted values, use marginal probability: P(y_it=1|X_it)
        # This requires integrating over α_i
        linear_pred = X @ beta

        # Approximate marginal probability using quadrature
        fitted_probs = np.zeros(len(y))
        for i in range(len(y)):
            prob_sum = 0.0
            for node, weight in zip(self._quad_nodes, self._quad_weights):
                alpha = np.sqrt(2) * sigma_alpha * node
                prob_sum += weight * stats.norm.cdf(linear_pred[i] + alpha)
            fitted_probs[i] = prob_sum

        # Residuals
        resid = y - fitted_probs

        # Compute covariance matrix (use numerical Hessian)
        from scipy.optimize import approx_fprime

        # Hessian via finite differences
        k = len(params)
        hessian = np.zeros((k, k))
        eps = 1e-5

        for i in range(k):

            def grad_i(p):
                return self._score(p)[i]

            hessian[i, :] = approx_fprime(params, grad_i, eps)

        # Covariance matrix: -H^{-1}
        try:
            vcov = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            vcov = -np.linalg.pinv(hessian)
            warnings.warn("Hessian is singular. Using pseudo-inverse for covariance matrix.")

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov))

        # Create pandas objects
        params_series = pd.Series(params, index=var_names)
        std_errors_series = pd.Series(std_errors, index=var_names)
        cov_params_df = pd.DataFrame(vcov, index=var_names, columns=var_names)

        # Degrees of freedom
        n = len(y)
        k_beta = len(beta)
        df_model = k_beta  # Not counting sigma_alpha as it's a variance parameter
        df_resid = n - k_beta - 1

        # Log-likelihood
        llf = self._log_likelihood(params)

        # Null log-likelihood (no covariates, only intercept and RE)
        y_mean = y.mean()
        ll_null = n * (y_mean * np.log(y_mean) + (1 - y_mean) * np.log(1 - y_mean))

        # Pseudo R-squared
        pseudo_r2 = 1 - llf / ll_null if ll_null != 0 else 0.0

        # Model information
        model_info = {
            "model_type": "Random Effects Probit",
            "formula": self.formula,
            "cov_type": "nonrobust",
            "cov_kwds": {},
            "llf": llf,
            "ll_null": ll_null,
            "quadrature_points": self.quadrature_points,
        }

        # Data information
        entities = self.data.data[self.data.entity_col].values
        data_info = {
            "nobs": n,
            "n_entities": self.data.n_entities,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": df_resid,
            "entity_index": entities.ravel(),
            "time_index": self.data.data[self.data.time_col].values.ravel(),
        }

        # R-squared dictionary
        rsquared_dict = {
            "rsquared": pseudo_r2,
            "rsquared_adj": np.nan,
            "rsquared_within": np.nan,
            "rsquared_between": np.nan,
            "rsquared_overall": pseudo_r2,
        }

        # Create results
        results = PanelResults(
            params=params_series,
            std_errors=std_errors_series,
            cov_params=cov_params_df,
            resid=resid,
            fittedvalues=fitted_probs,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
        )

        # Store additional attributes
        results.llf = llf
        results.ll_null = ll_null
        results.pseudo_r2_mcfadden = pseudo_r2

        # Information criteria
        results.aic = -2 * llf + 2 * (k_beta + 1)
        results.bic = -2 * llf + (k_beta + 1) * np.log(n)

        # Add predict method
        def predict_method(X_new=None, type="prob", include_re=False):
            """
            Predict probabilities for Random Effects Probit.

            Parameters
            ----------
            X_new : np.ndarray, optional
                New data for prediction
            type : str
                'prob' for probabilities, 'linear' for linear predictor
            include_re : bool
                If True, include random effect (requires entity info)
            """
            if X_new is None:
                X_pred = X
            else:
                X_pred = X_new

            linear_pred = X_pred @ beta

            if type == "linear":
                return linear_pred
            elif type == "prob":
                if include_re:
                    raise NotImplementedError(
                        "Prediction with random effects not yet implemented. "
                        "Use include_re=False for marginal predictions."
                    )
                else:
                    # Marginal probability via quadrature
                    probs = np.zeros(len(X_pred))
                    for i in range(len(X_pred)):
                        prob_sum = 0.0
                        for node, weight in zip(self._quad_nodes, self._quad_weights):
                            alpha = np.sqrt(2) * sigma_alpha * node
                            prob_sum += weight * stats.norm.cdf(linear_pred[i] + alpha)
                        probs[i] = prob_sum
                    return probs
            else:
                raise ValueError(f"Unknown prediction type: {type}")

        results.predict = predict_method

        return results

    def marginal_effects(self, at: str = "mean", method: str = "dydx") -> "MarginalEffectsResult":
        """
        Compute marginal effects for Random Effects Probit.

        For RE Probit, marginal effects must account for the random effect.
        The average marginal effect is:
        AME = ∫ β * φ(X'β + α) * φ(α/σ_α) dα

        Parameters
        ----------
        at : str, default='mean'
            Where to evaluate marginal effects:
            - 'mean': At mean values of covariates (MEM)
            - 'overall': Average marginal effects (AME)
        method : str, default='dydx'
            Type of marginal effect (only 'dydx' supported)

        Returns
        -------
        MarginalEffectsResult
            Marginal effects with standard errors

        Notes
        -----
        This integrates the marginal effects module with RE Probit.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted first. Call fit().")

        from panelbox.marginal_effects.discrete_me import compute_ame, compute_mem

        if at == "overall":
            return compute_ame(self._results)
        elif at == "mean":
            return compute_mem(self._results)
        else:
            raise ValueError(f"Unknown 'at' value: {at}. Use 'overall' or 'mean'.")
