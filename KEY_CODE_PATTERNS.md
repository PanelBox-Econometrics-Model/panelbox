# PanelBox - Key Code Patterns Reference

## 1. Logit Log-Likelihood (from PooledLogit)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` (lines 125-155)

```python
def _log_likelihood(self, params: np.ndarray) -> float:
    """
    Logit log-likelihood using numerically stable form
    """
    y, X = self.formula_parser.build_design_matrices(
        self.data.data, return_type="array"
    )

    # Linear predictor
    eta = X @ params

    # Numerically stable log-likelihood:
    # ℓ = Σ[y*η - log(1 + exp(η))]
    # This avoids overflow from exp(η) when η is large

    if self.weights is not None:
        llf = np.sum(self.weights * (y.ravel() * eta - np.log1p(np.exp(eta))))
    else:
        llf = np.sum(y.ravel() * eta - np.log1p(np.exp(eta)))

    return float(llf)
```

**Key Points**:
- Use `np.log1p(np.exp(eta))` instead of `np.log(1 + np.exp(eta))`
- Always return `float(llf)`, never array
- Handle weights if provided

## 2. Sandwich Covariance (from PooledLogit)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` (lines 216-235)

```python
# Heteroskedasticity-robust (sandwich estimator)
W = fitted_probs * (1 - fitted_probs)
H = -(X.T * W) @ X  # Hessian = -Σ w_i * X_i X_i'

# Bread: -H^{-1}
H_inv = np.linalg.inv(H)

# Meat: S = Σ s_i s_i' where s_i = (y_i - p_i) X_i
scores = (y - fitted_probs)[:, np.newaxis] * X
S = scores.T @ scores

# Sandwich: V = H^{-1} * S * H^{-1}
vcov = H_inv @ S @ H_inv
```

**Cluster-Robust Version**:
```python
# From panelbox.standard_errors.mle
result = cluster_robust_mle(H, scores, entities, df_correction=True)
vcov = result.cov_matrix
```

## 3. Probit Log-Likelihood (from PooledProbit)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` (lines 743-777)

```python
def _log_likelihood(self, params: np.ndarray) -> float:
    """
    Probit log-likelihood using normal CDF
    """
    y, X = self.formula_parser.build_design_matrices(
        self.data.data, return_type="array"
    )

    # Linear predictor
    eta = X @ params

    # Cumulative normal distribution
    prob = stats.norm.cdf(eta)

    # Clip to avoid log(0)
    prob = np.clip(prob, 1e-10, 1 - 1e-10)

    # Log-likelihood: ℓ = Σ[y*log(Φ(η)) + (1-y)*log(1-Φ(η))]
    if self.weights is not None:
        llf = np.sum(
            self.weights * (y.ravel() * np.log(prob) +
                          (1 - y.ravel()) * np.log(1 - prob))
        )
    else:
        llf = np.sum(y.ravel() * np.log(prob) +
                     (1 - y.ravel()) * np.log(1 - prob))

    return float(llf)
```

**Key Points**:
- Always clip probabilities to [1e-10, 1-1e-10]
- Use `stats.norm.cdf()` for normal CDF
- Similar structure to logit but uses normal CDF

## 4. Fixed Effects Logit - Conditional Likelihood (from FixedEffectsLogit)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` (lines 1351-1389)

```python
def _log_likelihood(self, params: np.ndarray) -> float:
    """
    Chamberlain (1980) conditional log-likelihood

    ℓ = Σᵢ [yᵢ'Xᵢβ - log(Σ_{s:Σsₜ=Σyᵢₜ} exp(s'Xᵢβ))]

    Only entities with temporal variation contribute.
    """
    y, X = self.formula_parser.build_design_matrices(
        self.data.data, return_type="array"
    )
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

def _sum_over_sequences(self, X_i, params, target_sum):
    """
    Sum exp(s'X_i β) over all binary sequences s with Σ s_t = target_sum
    Uses combinatorial enumeration for T ≤ 15
    """
    from itertools import combinations

    T_i = len(X_i)
    total = 0.0

    # Generate all combinations of indices that sum to target_sum
    for combo in combinations(range(T_i), target_sum):
        s = np.zeros(T_i)
        s[list(combo)] = 1
        total += np.exp(s @ X_i @ params)

    return total
```

## 5. Random Effects Probit - Quadrature Integration (from RandomEffectsProbit)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` (lines 1801-1862)

```python
def _log_likelihood(self, params: np.ndarray) -> float:
    """
    Marginal log-likelihood via Gauss-Hermite quadrature

    Integrates out random effects:
    ℓ_i(β, σ_α) = log ∫ [Π_t Φ(q_it(X_it'β + α_i))] φ(α_i/σ_α) dα_i
    """
    # Extract parameters
    beta = params[:-1]
    log_sigma_alpha = params[-1]
    sigma_alpha = np.exp(log_sigma_alpha)  # Ensure positivity

    # Get data
    y, X = self.formula_parser.build_design_matrices(
        self.data.data, return_type="array"
    )
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
            llf += -1e10  # Handle numerical issues

    return float(llf)
```

**Key Quadrature Pattern**:
- Initialize once: `self._quad_nodes, self._quad_weights = gauss_hermite_quadrature(n_points)`
- Transform: `alpha = √2 * sigma * node`
- Accumulate: `sum(weights * contribution(node))`
- Final: `llf += log(entity_sum)`

## 6. Ordered Logit - Cutpoint Handling (from OrderedLogit)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/ordered.py`

```python
def _transform_cutpoints(self, cutpoint_params: np.ndarray) -> np.ndarray:
    """
    Ensures ordered cutpoints: κ_0 < κ_1 < ... < κ_{J-2}

    Uses parameterization:
    κ_0 = γ_0
    κ_j = κ_{j-1} + exp(γ_j) for j > 0
    """
    cutpoints = np.zeros(len(cutpoint_params))
    cutpoints[0] = cutpoint_params[0]

    for j in range(1, len(cutpoint_params)):
        cutpoints[j] = cutpoints[j-1] + np.exp(cutpoint_params[j])

    return cutpoints

def _log_likelihood(self, params: np.ndarray) -> float:
    """Ordered choice log-likelihood with cutpoints"""
    K = self.n_features
    beta = params[:K]
    cutpoint_params = params[K:]

    # Transform to ensure ordered cutpoints
    cutpoints = self._transform_cutpoints(cutpoint_params)

    # Add boundary cutpoints
    cutpoints_extended = np.concatenate([
        [-np.inf], cutpoints, [np.inf]
    ])

    llf = 0.0
    for i in range(self.n_obs):
        y_i = self.endog[i]

        # P(y_i = j) = F(κ_j - X'β) - F(κ_{j-1} - X'β)
        eta_i = self.exog[i] @ beta

        lower = self._cdf(cutpoints_extended[y_i] - eta_i)
        upper = self._cdf(cutpoints_extended[y_i + 1] - eta_i)

        prob = upper - lower
        prob = np.clip(prob, 1e-10, 1.0)

        llf += np.log(prob)

    return float(llf)
```

## 7. Poisson Log-Likelihood (from PooledPoisson)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/count/poisson.py`

```python
def _log_likelihood(self, params: np.ndarray) -> float:
    """
    Poisson log-likelihood

    ℓ = Σᵢₜ [yᵢₜ log(λᵢₜ) - λᵢₜ - log(yᵢₜ!)]
    """
    linear_pred = self.exog @ params
    lambda_it = np.exp(linear_pred)

    # Use gammaln for numerical stability with large factorials
    log_factorial_y = gammaln(self.endog + 1)

    llf = np.sum(
        self.endog * linear_pred - lambda_it - log_factorial_y
    )

    return float(llf)

def _score(self, params: np.ndarray) -> np.ndarray:
    """
    Poisson score (gradient)

    ∂ℓ/∂β = Σᵢₜ (yᵢₜ - λᵢₜ) Xᵢₜ
    """
    linear_pred = self.exog @ params
    lambda_it = np.exp(linear_pred)
    residual = self.endog - lambda_it

    score = self.exog.T @ residual

    return score

def _hessian(self, params: np.ndarray) -> np.ndarray:
    """
    Poisson Hessian

    ∂²ℓ/∂β∂β' = -Σᵢₜ λᵢₜ Xᵢₜ Xᵢₜ'
    """
    linear_pred = self.exog @ params
    lambda_it = np.exp(linear_pred)

    # Weight matrix
    W = np.diag(lambda_it.flatten())

    # Hessian
    H = -self.exog.T @ W @ self.exog

    return H
```

## 8. Classification Metrics (from PooledLogit)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` (lines 371-411)

```python
def classification_metrics(threshold=0.5) -> dict:
    """Compute classification performance metrics"""
    y_pred = (fitted_probs >= threshold).astype(int)

    # Confusion matrix elements
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))

    # Metrics
    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * (precision * recall) / (precision + recall)
          if (precision + recall) > 0 else 0.0)

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
        "confusion_matrix": {"tp": int(tp), "tn": int(tn),
                           "fp": int(fp), "fn": int(fn)},
    }
```

## 9. Hosmer-Lemeshow Test (from PooledLogit)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` (lines 416-496)

```python
def hosmer_lemeshow_test(n_groups=10) -> dict:
    """Goodness-of-fit test for binary models"""
    # Sort by predicted probabilities
    sort_idx = np.argsort(fitted_probs)
    y_sorted = y[sort_idx]
    probs_sorted = fitted_probs[sort_idx]

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
    p_value = 1 - chi2.cdf(statistic, df)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "df": df,
        "n_groups": n_groups,
        "interpretation": (
            "Reject H0 (poor fit)" if p_value < 0.05
            else "Fail to reject H0 (adequate fit)"
        ),
    }
```

## 10. Results Object Creation (from PooledLogit)

**Location**: `/home/guhaase/projetos/panelbox/panelbox/models/discrete/binary.py` (lines 157-608)

```python
def fit(self, cov_type='cluster', **kwargs) -> PanelResults:
    """Standard fit pattern for all models"""

    # Build design matrices
    y, X = self.formula_parser.build_design_matrices(
        self.data.data, return_type="array"
    )
    var_names = self.formula_parser.get_variable_names(self.data.data)
    y = y.ravel()

    # ... [estimation code] ...

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

    # R-squared dictionary
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

    # Add additional attributes
    results.llf = llf
    results.ll_null = ll_null
    results.pseudo_r2_mcfadden = pseudo_r2
    results.converged = ...
    results.aic = -2 * llf + 2 * k
    results.bic = -2 * llf + k * np.log(n)

    # Add methods (predict, pseudo_r2, etc.)
    results.predict = predict_method
    results.pseudo_r2 = pseudo_r2_method
    # ... etc ...

    # Store and return
    self._results = results
    self._fitted = True

    return results
```

---

These patterns are the foundation for implementing new models. Copy, adapt, and extend as needed.
