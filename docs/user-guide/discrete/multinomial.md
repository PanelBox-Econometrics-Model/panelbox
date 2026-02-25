---
title: "Multinomial and Conditional Logit"
description: "Multinomial Logit and Conditional Logit models for unordered categorical panel data in PanelBox"
---

# Multinomial and Conditional Logit

!!! info "Quick Reference"
    **Classes:** `MultinomialLogit`, `ConditionalLogit`
    **Import:** `from panelbox.models.discrete.multinomial import MultinomialLogit, ConditionalLogit`
    **Stata equivalent:** `mlogit`, `clogit`
    **R equivalent:** `nnet::multinom()`, `survival::clogit()`, `mlogit::mlogit()`

## Overview

When the dependent variable takes on multiple **unordered** categories -- such as transportation mode (car, bus, train), occupation (blue collar, white collar, professional), or brand choice -- standard binary or ordered models are inappropriate. PanelBox provides two models for this setting:

**Multinomial Logit (MNL)** models the probability of choosing alternative $j$ based on **individual-level** characteristics:

$$P(y_i = j \mid X_i) = \frac{\exp(X_i'\beta_j)}{\sum_{k=0}^{J-1} \exp(X_i'\beta_k)}$$

with the normalization $\beta_0 = 0$ for the base alternative. Each non-base alternative has its own coefficient vector $\beta_j$, yielding a $(J-1) \times K$ parameter matrix.

**Conditional Logit (McFadden 1974)** models choice based on **alternative-level** attributes that vary across options:

$$P(y_i = j \mid Z_{ij}) = \frac{\exp(Z_{ij}'\gamma)}{\sum_{k=0}^{J-1} \exp(Z_{ik}'\gamma)}$$

where $Z_{ij}$ contains attributes of alternative $j$ (price, quality, distance). A single coefficient vector $\gamma$ applies to all alternatives.

## Quick Example

=== "Multinomial Logit"

    ```python
    import numpy as np
    from panelbox.models.discrete.multinomial import MultinomialLogit

    # y: occupational choice (0=blue collar, 1=white collar, 2=professional)
    model = MultinomialLogit(
        endog=y, exog=X,
        n_alternatives=3, base_alternative=0
    )
    result = model.fit(method="BFGS")

    # Coefficient matrix: (J-1, K)
    print(result.params_matrix)

    # Predicted probabilities: (N, J)
    probs = result.predict_proba()
    print(result.summary())
    ```

=== "Conditional Logit"

    ```python
    import pandas as pd
    from panelbox.models.discrete.multinomial import ConditionalLogit

    # Long-format: one row per (choice occasion, alternative)
    model = ConditionalLogit(
        data=df,
        choice_col="choice_id",
        alt_col="alternative",
        chosen_col="chosen",
        alt_varying_vars=["price", "quality"],
        case_varying_vars=["income"]
    )
    result = model.fit()
    print(result.summary())
    ```

## When to Use

- **Multinomial Logit**: individual-level characteristics drive the choice (education, income, demographics determine occupation)
- **Conditional Logit**: alternative-level attributes drive the choice (price, quality, travel time determine transportation mode)
- **Both models assume IIA** -- the Independence of Irrelevant Alternatives (see below)

!!! warning "Key Assumptions"
    - **IIA (Independence of Irrelevant Alternatives)**: The relative odds between any two alternatives are independent of other alternatives. Violated when alternatives are close substitutes (e.g., the red bus/blue bus problem).
    - **No unobserved heterogeneity** in the basic pooled specification. Use `method="fixed_effects"` or `method="random_effects"` in `MultinomialLogit` for panel heterogeneity.
    - **Conditional Logit**: homogeneous preferences across individuals (same $\gamma$ for all).

## Detailed Guide

### Multinomial Logit

#### Estimation

The MNL model estimates $J-1$ coefficient vectors (one per non-base alternative):

```python
from panelbox.models.discrete.multinomial import MultinomialLogit

model = MultinomialLogit(
    endog=y,
    exog=X,
    n_alternatives=3,       # J = 3 categories
    base_alternative=0,     # Base category (normalized to zero)
    method="pooled"         # "pooled", "fixed_effects", or "random_effects"
)
result = model.fit(method="BFGS", maxiter=1000)
```

#### Panel Data Extensions

The `MultinomialLogit` supports three estimation methods for panel data:

```python
# Pooled MNL (ignores panel structure)
model = MultinomialLogit(endog=y, exog=X, n_alternatives=3, method="pooled")

# Fixed Effects MNL (Chamberlain-style conditional MLE)
model = MultinomialLogit(
    endog=y, exog=X, n_alternatives=3,
    method="fixed_effects",
    entity_col="id", time_col="year"
)

# Random Effects MNL (Gauss-Hermite quadrature)
model = MultinomialLogit(
    endog=y, exog=X, n_alternatives=3,
    method="random_effects",
    entity_col="id", time_col="year"
)
```

!!! note "Computational Considerations"
    Fixed effects estimation with many alternatives ($J > 4$) or long panels ($T > 10$) can be computationally intensive. For large problems, consider the pooled specification with cluster-robust standard errors, or the random effects approach.

#### Interpreting Coefficients

Coefficients represent log-odds ratios relative to the base alternative:

$$\log \frac{P(y = j)}{P(y = 0)} = X'\beta_j$$

A coefficient $\beta_{jk}$ means: a one-unit increase in $x_k$ changes the log-odds of choosing alternative $j$ (vs. the base) by $\beta_{jk}$.

```python
# Coefficient matrix: (J-1) x K
print("Parameters for alternative 1 vs base:")
print(result.params_matrix[0, :])

print("Parameters for alternative 2 vs base:")
print(result.params_matrix[1, :])
```

!!! tip "Marginal Effects Over Coefficients"
    Direct coefficient interpretation is difficult because probabilities depend on all $\beta_j$ vectors simultaneously. Always compute marginal effects for meaningful interpretation.

#### Predictions and Classification

```python
# Predicted probabilities for each alternative: (N, J)
probs = result.predict_proba()

# Most likely alternative for each observation
predicted = result.predict()

# Classification quality
print(f"Accuracy: {result.accuracy:.3f}")
print(f"Confusion matrix:\n{result.confusion_matrix}")
```

#### Marginal Effects

Marginal effects in MNL have a special structure. For alternative $j$ and variable $k$:

$$\frac{\partial P(y = j)}{\partial x_k} = P(y = j) \left[ \beta_{jk} - \sum_{m=0}^{J-1} P(y = m) \beta_{mk} \right]$$

The marginal effects sum to zero across alternatives (probabilities must sum to 1).

```python
# Marginal effects at the mean: (J, K) matrix
me = result.marginal_effects(at="mean")

# Marginal effects at the median
me_median = result.marginal_effects(at="median")

# Average marginal effects
me_overall = result.marginal_effects(at="overall")

# For a specific variable
me_var = result.marginal_effects(at="mean", variable=0)
```

### Conditional Logit (McFadden 1974)

The Conditional Logit is designed for choice data where alternatives have distinct, observable attributes.

#### Data Format

Conditional Logit requires **long-format data**: one row per choice occasion per alternative.

```python
import pandas as pd

# Example: transportation mode choice
#   Each row: one (traveler-trip, mode) combination
#   Columns: trip ID, mode, whether chosen, mode attributes, traveler attributes
df = pd.DataFrame({
    "trip_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    "mode": ["car", "bus", "train", "car", "bus", "train", "car", "bus", "train"],
    "chosen": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    "travel_time": [30, 45, 20, 15, 60, 25, 50, 40, 15],
    "cost": [10, 3, 8, 5, 2, 7, 12, 4, 9],
    "income": [50, 50, 50, 30, 30, 30, 80, 80, 80],
})
```

#### Estimation

```python
from panelbox.models.discrete.multinomial import ConditionalLogit

model = ConditionalLogit(
    data=df,
    choice_col="trip_id",           # Choice occasion identifier
    alt_col="mode",                 # Alternative identifier
    chosen_col="chosen",            # Binary: 1 if chosen, 0 otherwise
    alt_varying_vars=["travel_time", "cost"],  # Attributes that vary by alternative
    case_varying_vars=["income"],   # Attributes that vary by individual (optional)
)
result = model.fit(method="BFGS")
print(result.summary())
```

#### Variable Types

| Variable Type | Description | Example | Coefficient |
|--------------|-------------|---------|-------------|
| Alternative-varying | Different value for each alternative | Travel time, price | Single $\gamma$ (generic) |
| Case-varying | Same value across alternatives | Income, age | $(J-1)$ coefficients (alternative-specific) |

For **case-varying** variables, the model estimates alternative-specific coefficients (relative to the base), similar to MNL. For **alternative-varying** variables, a single coefficient applies across all alternatives.

### The IIA Assumption

The Independence of Irrelevant Alternatives states that the ratio of choice probabilities between any two alternatives is independent of other alternatives:

$$\frac{P(y = j)}{P(y = k)} = \exp\left[(X'\beta_j - X'\beta_k)\right]$$

**The Red Bus / Blue Bus Problem**: If a city has car and red bus as transport options (50/50 split), adding a blue bus (identical to red) should split the bus share, yielding 50% car, 25% red bus, 25% blue bus. But IIA predicts 33/33/33 because it treats blue bus as equally distinct from car as red bus is.

!!! warning "When IIA Fails"
    If your alternatives include close substitutes, consider:

    - **Nested Logit**: groups similar alternatives
    - **Mixed Logit**: allows random taste variation
    - **Hausman-McFadden test**: formal test of IIA (drop one alternative and check if remaining estimates change)

## Configuration Options

### MultinomialLogit

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endog` | `ndarray` | required | Categorical outcome (0 to $J-1$) |
| `exog` | `ndarray` | required | Regressors |
| `n_alternatives` | `int` | `None` | Number of alternatives (inferred if `None`) |
| `base_alternative` | `int` | `0` | Reference alternative |
| `method` | `str` | `"pooled"` | `"pooled"`, `"fixed_effects"`, `"random_effects"` |
| `entity_col` | `str` | `None` | Entity identifier (required for FE/RE) |
| `time_col` | `str` | `None` | Time identifier |

**fit() parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | `ndarray` | `None` | Starting values |
| `method` | `str` | `"BFGS"` | Optimization method |
| `maxiter` | `int` | `1000` | Maximum iterations |

### ConditionalLogit

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `DataFrame` | required | Long-format choice data |
| `choice_col` | `str` | required | Choice occasion identifier |
| `alt_col` | `str` | required | Alternative identifier |
| `chosen_col` | `str` | required | Binary chosen indicator |
| `alt_varying_vars` | `list` | required | Alternative-varying attribute names |
| `case_varying_vars` | `list` | `None` | Case-varying attribute names |

**fit() parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_params` | `ndarray` | `None` | Starting values |
| `method` | `str` | `"BFGS"` | Optimization method |
| `maxiter` | `int` | `1000` | Maximum iterations |

## Result Attributes

### MultinomialLogitResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `ndarray` | Flat parameter vector (length $(J-1) \times K$) |
| `params_matrix` | `ndarray` | Parameters reshaped to $(J-1, K)$ |
| `bse` | `ndarray` | Standard errors |
| `bse_matrix` | `ndarray` | SEs reshaped to $(J-1, K)$ |
| `cov_params` | `ndarray` | Variance-covariance matrix |
| `predicted_probs` | `ndarray` | Predicted probabilities $(N, J)$ |
| `llf` | `float` | Log-likelihood |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `pseudo_r2` | `float` | McFadden pseudo $R^2$ |
| `accuracy` | `float` | Classification accuracy |
| `confusion_matrix` | `ndarray` | Confusion matrix $(J, J)$ |
| `converged` | `bool` | Convergence flag |
| `iterations` | `int` | Number of iterations |

### ConditionalLogitResult

| Attribute | Type | Description |
|-----------|------|-------------|
| `params` | `ndarray` | Estimated coefficients |
| `bse` | `ndarray` | Standard errors |
| `vcov` | `ndarray` | Variance-covariance matrix |
| `llf` | `float` | Log-likelihood |
| `aic` | `float` | Akaike Information Criterion |
| `bic` | `float` | Bayesian Information Criterion |
| `pseudo_r2` | `float` | McFadden pseudo $R^2$ |
| `accuracy` | `float` | Classification accuracy |
| `converged` | `bool` | Convergence flag |

## Diagnostics

### Model Fit

```python
# Multinomial Logit
print(f"Log-likelihood: {result.llf:.3f}")
print(f"McFadden R²:    {result.pseudo_r2:.3f}")
print(f"AIC:            {result.aic:.3f}")
print(f"BIC:            {result.bic:.3f}")
print(f"Accuracy:       {result.accuracy:.3f}")
print(f"Confusion matrix:\n{result.confusion_matrix}")
```

### Comparing Specifications

```python
# Compare pooled vs. RE
model_pooled = MultinomialLogit(endog=y, exog=X, n_alternatives=3, method="pooled")
res_pooled = model_pooled.fit()

model_re = MultinomialLogit(
    endog=y, exog=X, n_alternatives=3,
    method="random_effects", entity_col="id"
)
res_re = model_re.fit()

print(f"Pooled BIC: {res_pooled.bic:.1f}")
print(f"RE BIC:     {res_re.bic:.1f}")
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Multinomial Choice | Occupational choice and brand selection examples | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/discrete/notebooks/06_multinomial_logit.ipynb) |

## See Also

- [Binary Choice Models](binary.md) -- Logit and Probit for binary outcomes
- [Ordered Choice Models](ordered.md) -- Models for ordinal outcomes with natural ordering
- [Dynamic Binary Panel](dynamic.md) -- State dependence in binary choice
- [Marginal Effects](marginal-effects.md) -- Essential for MNL coefficient interpretation

## References

- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior." In *Frontiers in Econometrics*, ed. P. Zarembka. Academic Press.
- McFadden, D. (1981). "Econometric Models of Probabilistic Choice." In *Structural Analysis of Discrete Data*, ed. C. Manski and D. McFadden. MIT Press.
- Train, K. (2009). *Discrete Choice Methods with Simulation*. 2nd ed. Cambridge University Press.
- Hausman, J. and McFadden, D. (1984). "Specification Tests for the Multinomial Logit Model." *Econometrica*, 52(5), 1219-1240.
- Cameron, A. C. and Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 15.
