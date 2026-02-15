# Count Models Implementation Guide for PanelBox

## Quick Reference: Key Files and Patterns

This guide shows the exact patterns to follow when implementing count models (Poisson, Negative Binomial, etc.) in PanelBox.

## 1. BASIC CLASS STRUCTURE

```python
# File: panelbox/models/discrete/count.py

from panelbox.models.discrete.base import NonlinearPanelModel
from panelbox.core.results import PanelResults
import numpy as np
import pandas as pd
from scipy import stats

class PooledPoisson(NonlinearPanelModel):
    """Pooled Poisson regression for panel count data."""

    def __init__(self, formula, data, entity_col, time_col, weights=None):
        super().__init__(formula, data, entity_col, time_col, weights)
        self.family = 'poisson'  # For marginal effects

    def _log_likelihood(self, params: np.ndarray) -> float:
        """Poisson log-likelihood: ℓ = Σ[y*log(λ) - λ - log(y!)]"""
        y, X = self.formula_parser.build_design_matrices(
            self.data.data,
            return_type="array"
        )
        y = y.ravel()

        # Linear predictor on log scale
        eta = X @ params
        lam = np.exp(eta)  # Poisson mean

        # Log-likelihood (dropping log(y!) since it's constant)
        if self.weights is not None:
            ll = np.sum(self.weights * (y * eta - lam))
        else:
            ll = np.sum(y * eta - lam)

        return float(ll)

    def fit(self, cov_type='cluster', **kwargs):
        """
        Fit Poisson model.

        Parameters
        ----------
        cov_type : {'nonrobust', 'robust', 'cluster'}, default='cluster'
            Type of standard errors
        **kwargs
            Additional arguments passed to parent fit()
        """
        # Get starting values from Pooled OLS on log scale
        y, X = self.formula_parser.build_design_matrices(
            self.data.data,
            return_type="array"
        )
        y = y.ravel()
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Use parent class fit (calls _log_likelihood)
        results = super().fit(**kwargs)

        # Extract parameters
        params = results.params.values

        # Compute fitted values (expected counts)
        eta = X @ params
        fitted_counts = np.exp(eta)

        # Compute residuals
        resid = y - fitted_counts

        # Compute covariance matrix
        if cov_type == 'nonrobust':
            # H = -Σ λᵢ Xᵢ Xᵢ'
            H = -(X.T * fitted_counts) @ X
            vcov = np.linalg.inv(-H)

        elif cov_type == 'robust':
            # Sandwich: H^{-1} S H^{-1}
            # H = -Σ λᵢ Xᵢ Xᵢ'
            H = -(X.T * fitted_counts) @ X
            H_inv = np.linalg.inv(-H)

            # S = Σ (yᵢ - λᵢ)² Xᵢ Xᵢ'
            scores = (y - fitted_counts)[:, np.newaxis] * X
            S = scores.T @ scores

            vcov = H_inv @ S @ H_inv

        elif cov_type == 'cluster':
            # Cluster-robust by entity
            entities = self.data.data[self.data.entity_col].values

            # Use cluster_robust_mle
            from panelbox.standard_errors.mle import cluster_robust_mle

            H = -(X.T * fitted_counts) @ X
            scores = (y - fitted_counts)[:, np.newaxis] * X

            result = cluster_robust_mle(H, scores, entities, df_correction=True)
            vcov = result.cov_matrix

        else:
            raise ValueError(f"Invalid cov_type: {cov_type}")

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

        # Log-likelihood
        llf = self._log_likelihood(params)

        # Null log-likelihood (intercept only)
        y_mean = y.mean()
        ll_null = np.sum(y * np.log(y_mean) - y_mean)

        # Pseudo R-squared (McFadden)
        pseudo_r2 = 1 - llf / ll_null if ll_null != 0 else 0.0

        # Model information
        model_info = {
            "model_type": "Pooled Poisson",
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

        # R-squared dictionary (use pseudo R² for count models)
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
            fittedvalues=fitted_counts,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
        )

        # Add model-specific attributes
        results.llf = llf
        results.ll_null = ll_null
        results.pseudo_r2_mcfadden = pseudo_r2

        # Information criteria
        results.aic = -2 * llf + 2 * k
        results.bic = -2 * llf + k * np.log(n)

        # Add predict method
        def predict_method(X_new=None, type="response"):
            if X_new is None:
                eta_pred = eta
            else:
                eta_pred = X_new @ params

            if type == "linear":
                return eta_pred
            elif type == "response":
                return np.exp(eta_pred)
            else:
                raise ValueError(f"Unknown type: {type}")

        results.predict = predict_method

        # Store and return
        self._results = results
        self._fitted = True

        return results

    def predict(self, type='response'):
        """Make predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted first.")

        y, X = self.formula_parser.build_design_matrices(
            self.data.data,
            return_type="array"
        )
        return self._results.predict(X, type=type)
```

## 2. KEY PATTERNS FROM EXISTING MODELS

### Pattern: Log-Likelihood for Logit (Reference)
```python
def _log_likelihood(self, params: np.ndarray) -> float:
    y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")

    # Linear predictor
    eta = X @ params

    # Logit log-likelihood (numerically stable)
    if self.weights is not None:
        ll = np.sum(self.weights * (y.ravel() * eta - np.log1p(np.exp(eta))))
    else:
        ll = np.sum(y.ravel() * eta - np.log1p(np.exp(eta)))

    return float(ll)  # IMPORTANT: return scalar float
```

### Pattern: Covariance Matrix Computation
```python
# Step 1: Compute Hessian (or use numerical)
H = -(X.T * weights_vector) @ X  # Example for Logit

# Step 2: For cluster-robust, use cluster function
from panelbox.standard_errors.mle import cluster_robust_mle
scores = residuals[:, np.newaxis] * X
result = cluster_robust_mle(H, scores, entities, df_correction=True)
vcov = result.cov_matrix

# Step 3: Extract standard errors
std_errors = np.sqrt(np.diag(vcov))
```

## 3. STANDARD ERRORS IMPLEMENTATION

### For Poisson/NB (Key Difference from Logit)

For count models with heteroskedasticity-robust SEs:

```python
# Poisson case: E[Var(y|X)] = λ = E[y|X]
# So the variance function is: Var(yᵢ|Xᵢ) = λᵢ

# Cluster-robust meat matrix:
# S_i = Σ_{t in cluster i} s_it * s_it'
# where s_it = (y_it - λ_it) * X_it

# This is equivalent to what's in cluster_robust_mle
scores = (y - fitted_values)[:, np.newaxis] * X
```

## 4. MARGINAL EFFECTS FOR COUNT MODELS

```python
# In discrete_me.py, add support for count models

def compute_ame_count(result, varlist=None):
    """
    AME for count models.

    For Poisson: ME = β_k * E[λ]
    For NB: ME = β_k * E[λ]
    """
    model = result.model
    X = model.exog  # or parse from formula
    params = result.params.values

    # Linear predictions
    eta = X @ params
    lam = np.exp(eta)

    # Marginal effects
    ame = {}
    for var in varlist:
        var_idx = exog_names.index(var)
        # ME = β_k * λ (same for Poisson and NB)
        ame[var] = (params[var_idx] * lam).mean()

    return MarginalEffectsResult(ame, std_errors, result, me_type='ame')
```

## 5. TESTING PATTERN (test_count_poisson.py)

```python
import pytest
import numpy as np
import pandas as pd
from scipy import stats

from panelbox.models.discrete.count import PooledPoisson

class TestPooledPoisson:
    """Test Pooled Poisson model."""

    @pytest.fixture
    def poisson_data(self):
        """Generate synthetic Poisson data."""
        np.random.seed(42)

        n = 500
        t = 4

        entity_ids = np.repeat(np.arange(n), t)
        time_ids = np.tile(np.arange(t), n)

        x = np.random.randn(n * t)

        # True parameters
        beta = np.array([0.5, 0.8])

        # Generate Poisson data
        eta = 0.5 + 0.8 * x  # Linear predictor
        lam = np.exp(eta)    # Expected count
        y = np.random.poisson(lam)

        data = pd.DataFrame({
            'entity': entity_ids,
            'time': time_ids,
            'y': y,
            'x': x
        })

        return data, beta

    def test_poisson_fit(self, poisson_data):
        """Test basic fitting."""
        data, beta_true = poisson_data

        model = PooledPoisson("y ~ x", data, 'entity', 'time')
        results = model.fit()

        # Check results structure
        assert hasattr(results, 'params')
        assert hasattr(results, 'std_errors')
        assert hasattr(results, 'llf')

        # Check parameter estimates are reasonable
        assert np.sign(results.params['x']) == np.sign(beta_true[1])

    def test_poisson_predict(self, poisson_data):
        """Test predictions."""
        data, _ = poisson_data

        model = PooledPoisson("y ~ x", data, 'entity', 'time')
        results = model.fit()

        # Predict expected counts
        counts = results.predict(type='response')

        # Check predictions are positive
        assert np.all(counts > 0)
        assert len(counts) == len(data)
```

## 6. INTEGRATION: Update __init__.py

```python
# panelbox/models/discrete/__init__.py

from panelbox.models.discrete.base import NonlinearPanelModel
from panelbox.models.discrete.binary import (
    FixedEffectsLogit,
    PooledLogit,
    PooledProbit,
    RandomEffectsProbit
)
from panelbox.models.discrete.count import (
    PooledPoisson,
    PooledNegativeBinomial,
    # Add others as implemented
)

__all__ = [
    "NonlinearPanelModel",
    "PooledLogit",
    "PooledProbit",
    "FixedEffectsLogit",
    "RandomEffectsProbit",
    "PooledPoisson",
    "PooledNegativeBinomial",
]
```

## 7. CRITICAL CHECKLIST

When implementing a count model, ensure:

- [ ] `_log_likelihood()` returns scalar float
- [ ] Handles weights if `self.weights` is not None
- [ ] Fit method computes covariance matrix correctly
- [ ] Results object includes model_info and data_info dicts
- [ ] Predictions have both 'linear' (log scale) and 'response' (count scale)
- [ ] Standard errors support 'nonrobust', 'robust', and 'cluster'
- [ ] Numerically stable computations (avoid overflow/underflow)
- [ ] Tests for basic functionality
- [ ] Tests for edge cases (zero counts, large counts, etc.)
- [ ] Docstrings with parameter descriptions
- [ ] Mathematical formulation in README

## 8. NUMERICAL STABILITY TIPS

```python
# For Poisson/NB with large linear predictors:
# DON'T: lam = np.exp(eta)  # Can overflow if eta > 700
# DO: Handle large eta values

# Option 1: Clip extreme values
eta_clipped = np.clip(eta, -100, 100)
lam = np.exp(eta_clipped)

# Option 2: Use log-space computations where possible
# Avoid computing lam directly if not needed

# For log-likelihood with large counts:
# Use log-factorial approximations for Poisson
```

## 9. REFERENCE: Existing Model Structure

See `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` for complete patterns:
- Lines 125-156: Log-likelihood implementation (PooledLogit)
- Lines 157-608: Full fit() method with all SEs
- Lines 643-687: Marginal effects stub
- Lines 689-1208: PooledProbit (similar pattern for different link)

All count models should follow this same structure!
