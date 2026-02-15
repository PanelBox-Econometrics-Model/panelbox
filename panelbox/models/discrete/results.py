"""
Results classes for nonlinear panel models.

This module provides results classes for storing, computing, and displaying
results from Maximum Likelihood Estimation of nonlinear panel models.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import OptimizeResult

from panelbox.core.results import PanelResults
from panelbox.standard_errors.mle import (
    bootstrap_mle,
    cluster_robust_mle,
    compute_mle_standard_errors,
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

    def bootstrap_se(self, n_bootstrap: int = 999, seed: int = 42):
        """
        Compute bootstrap standard errors.

        Parameters
        ----------
        n_bootstrap : int, default=999
            Number of bootstrap replications.
        seed : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        NonlinearPanelResults
            New results object with bootstrap standard errors.

        Examples
        --------
        >>> # Original estimation
        >>> model = FixedEffectsLogit.from_formula('y ~ x1 + x2', data=data)
        >>> result = model.fit()
        >>>
        >>> # Compute bootstrap SEs
        >>> boot_result = result.bootstrap_se(n_bootstrap=999)
        >>> print(boot_result.summary())
        """

        # Define estimation function for bootstrap
        def estimate_func(y, X):
            # Create a temporary model with bootstrap data
            temp_model = self.model.__class__(
                endog=y,
                exog=X,
                entity=self.model.entity,
                time=self.model.time,
            )
            temp_result = temp_model.fit(disp=False)
            return temp_result.params.values

        # Get bootstrap covariance
        boot_result = bootstrap_mle(
            estimate_func=estimate_func,
            y=self.model.endog,
            X=self.model.exog,
            n_bootstrap=n_bootstrap,
            cluster_ids=self.model.entity_id,
            seed=seed,
        )

        # Create new results object with bootstrap SEs
        return self.__class__(
            model=self.model,
            params=self.params.values,
            llf=self.llf,
            converged=self.converged,
            n_iter=self.n_iter,
            se_type="bootstrap",
            opt_result=self.opt_result,
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

    def to_html(self, filepath: Optional[Union[str, Path]] = None, **kwargs) -> str:
        """
        Generate HTML report using PanelBox report system.

        Parameters
        ----------
        filepath : str or Path, optional
            If provided, save HTML to file.
        **kwargs
            Additional options for HTML generation.

        Returns
        -------
        str
            HTML string.
        """
        from panelbox.report.report_manager import ReportManager

        report_mgr = ReportManager()

        # Prepare context for template
        context = {
            "title": f"{self.model.__class__.__name__} Results",
            "model_type": self.model.__class__.__name__,
            "observations": self.nobs,
            "entities": self.model.n_entities if hasattr(self.model, "n_entities") else None,
            "log_likelihood": self.llf,
            "aic": self.aic,
            "bic": self.bic,
            "pseudo_r2": self.pseudo_r2("mcfadden"),
            "converged": self.converged,
            "iterations": self.n_iter,
            "se_type": self._se_type,
            "parameters": self.params.to_dict(),
            "std_errors": self.std_errors.to_dict(),
            "tvalues": self.tvalues.to_dict(),
            "pvalues": self.pvalues.to_dict(),
            "conf_int_lower": self.conf_int()[:, 0],
            "conf_int_upper": self.conf_int()[:, 1],
        }

        # Add classification metrics if binary model
        if hasattr(self.model, "model_type") and self.model.model_type == "binary":
            metrics = self.classification_metrics()
            context["classification_metrics"] = metrics

        # Generate HTML
        html = report_mgr.generate_report(
            report_type="discrete", template="discrete/results.html", context=context, **kwargs
        )

        # Save if filepath provided
        if filepath:
            filepath = Path(filepath)
            filepath.write_text(html)

        return html

    def to_latex(
        self,
        filepath: Optional[Union[str, Path]] = None,
        caption: str = "Model Results",
        label: str = "tab:results",
    ) -> str:
        """
        Generate LaTeX table of results.

        Parameters
        ----------
        filepath : str or Path, optional
            If provided, save LaTeX to file.
        caption : str
            Table caption.
        label : str
            LaTeX label for referencing.

        Returns
        -------
        str
            LaTeX string.
        """
        # Build LaTeX table
        latex_lines = []
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"\centering")
        latex_lines.append(f"\\caption{{{caption}}}")
        latex_lines.append(f"\\label{{{label}}}")
        latex_lines.append(r"\begin{tabular}{lcccc}")
        latex_lines.append(r"\hline\hline")

        # Header
        latex_lines.append(r"Variable & Coefficient & Std. Error & z-stat & P-value \\")
        latex_lines.append(r"\hline")

        # Parameters
        for i, name in enumerate(self.model.exog_names):
            coef = f"{self.params[i]:.4f}"
            se = f"{self.std_errors[i]:.4f}"
            z = f"{self.tvalues[i]:.2f}"
            p = f"{self.pvalues[i]:.3f}"

            # Add significance stars
            if self.pvalues[i] < 0.01:
                coef += "***"
            elif self.pvalues[i] < 0.05:
                coef += "**"
            elif self.pvalues[i] < 0.10:
                coef += "*"

            latex_lines.append(f"{name} & {coef} & {se} & {z} & {p} \\\\")

        latex_lines.append(r"\hline")

        # Model statistics
        latex_lines.append(f"Observations & \\multicolumn{{4}}{{c}}{{{self.nobs}}} \\\\")
        if hasattr(self.model, "n_entities"):
            latex_lines.append(
                f"Entities & \\multicolumn{{4}}{{c}}{{{self.model.n_entities}}} \\\\"
            )
        latex_lines.append(f"Log-Likelihood & \\multicolumn{{4}}{{c}}{{{self.llf:.2f}}} \\\\")
        latex_lines.append(f"AIC & \\multicolumn{{4}}{{c}}{{{self.aic:.2f}}} \\\\")
        latex_lines.append(f"BIC & \\multicolumn{{4}}{{c}}{{{self.bic:.2f}}} \\\\")
        latex_lines.append(
            f"Pseudo-R$^2$ & \\multicolumn{{4}}{{c}}{{{self.pseudo_r2('mcfadden'):.4f}}} \\\\"
        )

        latex_lines.append(r"\hline\hline")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\end{table}")

        latex = "\n".join(latex_lines)

        # Save if filepath provided
        if filepath:
            filepath = Path(filepath)
            filepath.write_text(latex)

        return latex

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
