"""
Results classes for nonlinear panel models.

This module provides results classes for storing, computing, and displaying
results from Maximum Likelihood Estimation of nonlinear panel models.
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import OptimizeResult

from panelbox.core.results import PanelResults
from panelbox.standard_errors.mle import (
    bootstrap_mle,
    cluster_robust_mle,
    delta_method,
    sandwich_estimator,
)


class NonlinearPanelResults(PanelResults):
    """
    Results class for nonlinear panel models estimated via MLE.

    This class extends PanelResults with MLE-specific functionality:
    - Log-likelihood and information criteria
    - Pseudo-R² measures
    - Classification metrics for binary models
    - Marginal effects computation
    - Diagnostic tests

    Parameters
    ----------
    model : NonlinearPanelModel
        The fitted model instance.
    params : np.ndarray
        Parameter estimates.
    llf : float
        Log-likelihood value at optimum.
    converged : bool
        Whether optimization converged.
    n_iter : int
        Number of iterations.
    se_type : str
        Type of standard errors.
    opt_result : OptimizeResult
        Full optimization result from scipy.
    """

    def __init__(
        self,
        model,
        params: np.ndarray,
        llf: float,
        converged: bool,
        n_iter: int,
        se_type: str = "cluster",
        opt_result: Optional[OptimizeResult] = None,
    ):
        # Store MLE-specific attributes
        self.llf = llf
        self.converged = converged
        self.n_iter = n_iter
        self.opt_result = opt_result
        self._se_type = se_type

        # Compute standard errors
        cov_params = self._compute_cov_params(model, params, se_type)
        std_errors = np.sqrt(np.diag(cov_params))

        # Initialize parent class
        super().__init__(
            model=model,
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            se_type=se_type,
        )

        # Cache for expensive computations
        self._cache = {}

    def _compute_cov_params(self, model, params: np.ndarray, se_type: str) -> np.ndarray:
        """
        Compute covariance matrix of parameters.

        Parameters
        ----------
        model : NonlinearPanelModel
            Model instance.
        params : np.ndarray
            Parameter estimates.
        se_type : str
            Type of standard errors.

        Returns
        -------
        np.ndarray
            Covariance matrix.
        """
        return compute_mle_standard_errors(
            model=model, params=params, se_type=se_type, entity_id=model.entity_id
        )

    @property
    def aic(self) -> float:
        """Akaike Information Criterion."""
        return -2 * self.llf + 2 * len(self.params)

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion."""
        return -2 * self.llf + np.log(self.nobs) * len(self.params)

    @property
    def llf_null(self) -> float:
        """Log-likelihood of null model (intercept only)."""
        if "llf_null" not in self._cache:
            # Fit null model
            from copy import deepcopy

            null_model = deepcopy(self.model)
            null_model.formula = f"{null_model.depvar} ~ 1"
            null_model._prepare_matrices()

            # Compute null log-likelihood
            null_params = np.zeros(1)
            self._cache["llf_null"] = null_model._log_likelihood(null_params)

        return self._cache["llf_null"]

    def pseudo_r2(self, method: str = "mcfadden") -> float:
        """
        Compute pseudo-R² measure.

        Parameters
        ----------
        method : str, default 'mcfadden'
            Type of pseudo-R²:
            - 'mcfadden': McFadden's R² = 1 - llf/llf_null
            - 'cox_snell': Cox-Snell R² = 1 - exp(-LR/n)
            - 'nagelkerke': Nagelkerke's R² (adjusted Cox-Snell)

        Returns
        -------
        float
            Pseudo-R² value.
        """
        if method == "mcfadden":
            return 1 - self.llf / self.llf_null

        elif method == "cox_snell":
            lr_stat = 2 * (self.llf - self.llf_null)
            return 1 - np.exp(-lr_stat / self.nobs)

        elif method == "nagelkerke":
            r2_cs = self.pseudo_r2("cox_snell")
            max_r2 = 1 - np.exp(2 * self.llf_null / self.nobs)
            return r2_cs / max_r2

        else:
            raise ValueError(f"Unknown pseudo-R² method: {method}")

    def predict(
        self, exog: Optional[np.ndarray] = None, type: Literal["linear", "prob", "class"] = "prob"
    ) -> np.ndarray:
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        exog : np.ndarray, optional
            Explanatory variables for prediction. If None, uses estimation sample.
        type : str, default 'prob'
            Type of prediction:
            - 'linear': Linear predictor X'β
            - 'prob': Predicted probabilities P(y=1|X)
            - 'class': Binary predictions (0/1) using 0.5 threshold

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if exog is None:
            exog = self.model.X

        # Linear predictor
        linear_pred = exog @ self.params

        if type == "linear":
            return linear_pred

        # Transform to probabilities (model-specific)
        if hasattr(self.model, "_link_function"):
            prob = self.model._link_function(linear_pred)
        else:
            # Default to logistic for binary models
            from scipy.special import expit

            prob = expit(linear_pred)

        if type == "prob":
            return prob
        elif type == "class":
            return (prob >= 0.5).astype(int)
        else:
            raise ValueError(f"Unknown prediction type: {type}")

    def classification_table(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Create classification table (confusion matrix) for binary models.

        Parameters
        ----------
        threshold : float, default 0.5
            Classification threshold.

        Returns
        -------
        pd.DataFrame
            Confusion matrix with actual vs predicted.
        """
        y_true = self.model.y
        y_pred = (self.predict(type="prob") >= threshold).astype(int)

        # Create confusion matrix
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        return pd.DataFrame(
            cm, index=["Actual=0", "Actual=1"], columns=["Predicted=0", "Predicted=1"]
        )

    def classification_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute classification metrics for binary models.

        Parameters
        ----------
        threshold : float, default 0.5
            Classification threshold.

        Returns
        -------
        dict
            Dictionary with metrics:
            - accuracy: Overall accuracy
            - precision: Precision for class 1
            - recall: Recall for class 1
            - f1: F1-score for class 1
            - auc_roc: Area under ROC curve
        """
        y_true = self.model.y
        y_prob = self.predict(type="prob")
        y_pred = (y_prob >= threshold).astype(int)

        # Compute metrics
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_prob) if y_true.var() > 0 else np.nan,
        }

    def hosmer_lemeshow_test(self, n_groups: int = 10) -> Dict[str, float]:
        """
        Hosmer-Lemeshow goodness-of-fit test for binary models.

        Parameters
        ----------
        n_groups : int, default 10
            Number of groups for the test.

        Returns
        -------
        dict
            Test statistic and p-value.
        """
        y_true = self.model.y
        y_prob = self.predict(type="prob")

        # Sort by predicted probabilities
        sorted_idx = np.argsort(y_prob)
        y_true_sorted = y_true[sorted_idx]
        y_prob_sorted = y_prob[sorted_idx]

        # Create groups
        n_obs = len(y_true)
        group_size = n_obs // n_groups

        chi2_stat = 0.0
        for g in range(n_groups):
            if g < n_groups - 1:
                idx = slice(g * group_size, (g + 1) * group_size)
            else:
                idx = slice(g * group_size, None)

            obs_1 = y_true_sorted[idx].sum()
            exp_1 = y_prob_sorted[idx].sum()
            n_g = len(y_true_sorted[idx])
            obs_0 = n_g - obs_1
            exp_0 = n_g - exp_1

            # Add to chi-square statistic
            if exp_1 > 0:
                chi2_stat += (obs_1 - exp_1) ** 2 / exp_1
            if exp_0 > 0:
                chi2_stat += (obs_0 - exp_0) ** 2 / exp_0

        # Compute p-value
        df = n_groups - 2  # degrees of freedom
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

        return {"statistic": chi2_stat, "p_value": p_value, "df": df}

    def link_test(self) -> Dict[str, float]:
        """
        Link test for model specification.

        Tests whether the square of linear predictor is significant,
        which would indicate misspecification.

        Returns
        -------
        dict
            Coefficient and p-value for squared term.
        """
        linear_pred = self.predict(type="linear")
        linear_pred_sq = linear_pred**2

        # Create augmented design matrix
        X_aug = np.column_stack([self.model.X, linear_pred_sq])

        # Fit augmented model
        from copy import deepcopy

        aug_model = deepcopy(self.model)
        aug_model.X = X_aug
        aug_model.k_params = X_aug.shape[1]

        # Get initial values (current params + 0 for new parameter)
        x0 = np.append(self.params, 0)

        # Optimize
        from scipy.optimize import minimize

        result = minimize(fun=lambda p: -aug_model._log_likelihood(p), x0=x0, method="BFGS")

        # Test significance of squared term
        coef_sq = result.x[-1]
        se_sq = np.sqrt(np.diag(aug_model._hessian(result.x))[-1])
        z_stat = coef_sq / se_sq
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "coefficient": coef_sq,
            "std_error": se_sq,
            "z_statistic": z_stat,
            "p_value": p_value,
        }

    def marginal_effects(
        self,
        at: Literal["mean", "median", "zero"] = "mean",
        method: Literal["dydx", "eyex", "dyex", "eydx"] = "dydx",
    ) -> pd.DataFrame:
        """
        Compute marginal effects.

        Parameters
        ----------
        at : str, default 'mean'
            Where to evaluate marginal effects:
            - 'mean': At means of explanatory variables
            - 'median': At medians
            - 'zero': At zero values
        method : str, default 'dydx'
            Type of marginal effect:
            - 'dydx': ∂y/∂x (marginal effect)
            - 'eyex': ∂log(y)/∂log(x) (elasticity)
            - 'dyex': ∂y/∂log(x) (semi-elasticity)
            - 'eydx': ∂log(y)/∂x (semi-elasticity)

        Returns
        -------
        pd.DataFrame
            Marginal effects with standard errors.
        """
        # Implementation will be added in Phase 2
        raise NotImplementedError("Marginal effects will be implemented in Phase 2")

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        return self.summary().to_html()

    def summary(self) -> pd.DataFrame:
        """
        Generate summary table of results.

        Returns
        -------
        pd.DataFrame
            Summary statistics.
        """
        # Basic model info
        info = {
            "Model": self.model.__class__.__name__,
            "Observations": self.nobs,
            "Entities": self.model.n_entities,
            "Log-Likelihood": f"{self.llf:.4f}",
            "AIC": f"{self.aic:.4f}",
            "BIC": f"{self.bic:.4f}",
            "Pseudo-R² (McFadden)": f"{self.pseudo_r2('mcfadden'):.4f}",
            "Converged": "Yes" if self.converged else "No",
            "Iterations": self.n_iter,
            "Standard Errors": self._se_type,
        }

        # Add classification metrics for binary models
        if hasattr(self.model, "model_type") and self.model.model_type == "binary":
            metrics = self.classification_metrics()
            info.update(
                {
                    "Accuracy": f"{metrics['accuracy']:.4f}",
                    "AUC-ROC": (
                        f"{metrics['auc_roc']:.4f}" if not np.isnan(metrics["auc_roc"]) else "N/A"
                    ),
                }
            )

        # Create summary DataFrame
        summary_df = pd.DataFrame(list(info.items()), columns=["Statistic", "Value"])

        # Add parameter estimates
        param_df = pd.DataFrame(
            {
                "Coefficient": self.params,
                "Std. Error": self.std_errors,
                "z-statistic": self.tvalues,
                "P-value": self.pvalues,
                "95% CI Lower": self.conf_int()[:, 0],
                "95% CI Upper": self.conf_int()[:, 1],
            },
            index=self.model.exog_names,
        )

        return pd.concat(
            [
                pd.DataFrame({"": ["=" * 60]}),
                summary_df,
                pd.DataFrame({"": ["-" * 60]}),
                param_df,
                pd.DataFrame({"": ["=" * 60]}),
            ]
        )
