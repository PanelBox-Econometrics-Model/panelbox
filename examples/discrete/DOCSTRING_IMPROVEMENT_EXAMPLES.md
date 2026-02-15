# Docstring Improvement Examples

This document provides specific examples of what needs to be added to improve docstring coverage.

---

## 1. HonoreResults - Complete Missing Docstring

### Location
`panelbox/models/censored/honore.py` (lines 23-31)

### Current Code
```python
@dataclass
class HonoreResults:
    """Results container for Honoré estimator."""
    params: np.ndarray
    converged: bool
    n_iter: int
    n_obs: int
    n_entities: int
    n_trimmed: int
```

### What's Missing
- Field-level documentation for each attribute
- Type information details
- Description of what each field represents

### Recommended Addition
```python
@dataclass
class HonoreResults:
    """
    Results container for Honoré trimmed LAD estimator.

    This dataclass stores the estimation results from the
    Honoré (1992) semiparametric estimator for fixed effects
    Tobit models.

    Attributes
    ----------
    params : np.ndarray
        Estimated coefficients (K,) array
    converged : bool
        Whether optimization converged
    n_iter : int
        Number of optimization iterations
    n_obs : int
        Total number of observations
    n_entities : int
        Number of cross-sectional units
    n_trimmed : int
        Number of pairwise comparisons trimmed due to censoring
    """
```

---

## 2. OrderedLogit - Missing Method Documentation

### Location
`panelbox/models/discrete/ordered.py` (lines 462-490)

### Current Code
```python
class OrderedLogit(OrderedChoiceModel):
    """
    Ordered Logit model for ordinal panel data.

    The model uses the logistic distribution for the error term.

    Parameters
    ----------
    endog : array-like
        The dependent variable with ordinal categories (0, 1, 2, ...)
    exog : array-like
        The independent variables
    groups : array-like
        Group identifiers for panel structure
    time : array-like, optional
        Time identifiers for panel structure
    n_categories : int, optional
        Number of categories (inferred from data if not provided)
    """

    def _cdf(self, z: np.ndarray) -> np.ndarray:
        """Logistic CDF."""
        return expit(z)

    def _pdf(self, z: np.ndarray) -> np.ndarray:
        """Logistic PDF."""
        # PDF = F(z) * (1 - F(z))
        F_z = expit(z)
        return F_z * (1 - F_z)
```

### What's Missing
- Formal Google-style class docstring sections
- Method parameter documentation
- Method examples
- References for the ordered logit model
- Attributes section

### Recommended Addition
```python
class OrderedLogit(OrderedChoiceModel):
    """
    Ordered Logit model for ordinal panel data.

    The ordered logit model is used when the dependent variable
    takes on ordinal values (0, 1, 2, ..., J) representing ordered
    categories without assuming equal spacing between them.

    The latent variable model is:

        y*_it = X_it'β + ε_it
        y_it = j if κ_{j-1} < y*_it ≤ κ_j

    where κ_0 < κ_1 < ... < κ_{J-1} are cutpoints and
    ε_it follows a logistic distribution.

    Parameters
    ----------
    endog : array-like
        The dependent variable with ordinal categories (0, 1, 2, ...)
    exog : array-like
        The independent variables
    groups : array-like
        Group identifiers for panel structure
    time : array-like, optional
        Time identifiers for panel structure
    n_categories : int, optional
        Number of categories (inferred from data if not provided)

    Attributes
    ----------
    n_categories : int
        Number of ordinal categories
    n_cutpoints : int
        Number of cutpoints (n_categories - 1)
    params : np.ndarray
        Estimated parameters after fitting
    cutpoints : np.ndarray
        Estimated cutpoints after fitting

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.models.discrete.ordered import OrderedLogit
    >>>
    >>> # Simulate ordinal panel data
    >>> np.random.seed(42)
    >>> n_obs = 100
    >>> X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, 2)])
    >>> groups = np.repeat(range(20), 5)
    >>> beta_true = np.array([0, 0.5, -0.3])
    >>> linear_pred = X @ beta_true
    >>> y = (linear_pred > 0).astype(int) + (linear_pred > 1).astype(int)
    >>>
    >>> # Fit model
    >>> model = OrderedLogit(y, X, groups)
    >>> result = model.fit()
    >>> print(result.summary())

    References
    ----------
    .. [1] Long, J. S. (1997). Regression Models for Categorical and
           Limited Dependent Variables. SAGE Publications.
    .. [2] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data (2nd ed.). MIT Press. Section 15.10.

    See Also
    --------
    OrderedProbit : Ordered Probit model using normal distribution
    RandomEffectsOrderedLogit : RE ordered logit with random effects
    """

    def _cdf(self, z: np.ndarray) -> np.ndarray:
        """
        Logistic cumulative distribution function.

        Parameters
        ----------
        z : np.ndarray
            Evaluation points

        Returns
        -------
        np.ndarray
            CDF values F(z) = exp(z) / (1 + exp(z))
        """
        return expit(z)

    def _pdf(self, z: np.ndarray) -> np.ndarray:
        """
        Logistic probability density function.

        Parameters
        ----------
        z : np.ndarray
            Evaluation points

        Returns
        -------
        np.ndarray
            PDF values f(z) = F(z) * (1 - F(z))

        Notes
        -----
        The logistic PDF is symmetric around zero and has
        maximum density of 0.25 at z=0.
        """
        # PDF = F(z) * (1 - F(z))
        F_z = expit(z)
        return F_z * (1 - F_z)
```

---

## 3. PooledPoisson - Formal Google-Style Parameters

### Location
`panelbox/models/count/poisson.py` (lines 35-76)

### Current Code
```python
class PooledPoisson(NonlinearPanelModel):
    """
    Pooled Poisson regression for panel data.

    Standard Poisson MLE with optional cluster-robust standard errors.

    Parameters
    ----------
    endog : array-like
        Dependent variable (count data)
    exog : array-like
        Independent variables
    entity_id : array-like, optional
        Entity identifiers for clustering
    time_id : array-like, optional
        Time identifiers

    Attributes
    ----------
    overdispersion : float
        Overdispersion index (Var(y)/E(y))
        Values > 1 indicate overdispersion

    Methods
    -------
    fit(se_type='cluster', **kwargs)
        Estimate model parameters
    predict(X=None, type='response')
        Generate predictions
    check_overdispersion()
        Test for overdispersion

    Examples
    --------
    >>> model = PooledPoisson(y, X, entity_id=firms)
    >>> result = model.fit(se_type='cluster')
    >>> print(result.summary())

    References
    ----------
    Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data.
    """
```

### Issues
- Parameters not formally documented with types
- Attributes description is too brief
- Examples are incomplete (missing numpy import, etc.)
- Methods section uses old format instead of linking to actual methods

### Recommended Improvement
```python
class PooledPoisson(NonlinearPanelModel):
    """
    Pooled Poisson regression for panel data.

    Standard Poisson maximum likelihood estimation with optional
    cluster-robust standard errors. The model assumes the dependent
    variable follows a Poisson distribution conditional on covariates.

    The model is:
        y_it | X_it ~ Poisson(λ_it)
        log(λ_it) = X_it'β

    Parameters
    ----------
    endog : np.ndarray or pd.Series
        Dependent variable with count data (non-negative integers).
        Shape (N*T,) for panel data with N entities and T time periods.
    exog : np.ndarray or pd.DataFrame
        Independent variables. Shape (N*T, K) where K is number of
        covariates including the constant if needed.
    entity_id : np.ndarray or pd.Series, optional
        Entity identifiers for panel structure. Used for clustering
        standard errors and identifying panel structure.
        Shape (N*T,). If None, assumes cross-sectional data.
    time_id : np.ndarray or pd.Series, optional
        Time period identifiers. Shape (N*T,).

    Attributes
    ----------
    overdispersion : float
        Empirical overdispersion index = Var(y) / E(y).
        If > 1, indicates overdispersion suggesting need for
        Negative Binomial model. If < 1, indicates underdispersion.
    n_obs : int
        Total number of observations (N*T)
    n_entities : int
        Number of cross-sectional units (N)
    n_periods : int
        Number of time periods (T)

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.models.count.poisson import PooledPoisson
    >>>
    >>> # Simulate panel count data
    >>> np.random.seed(42)
    >>> n_obs = 100
    >>> X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, 2)])
    >>> beta_true = np.array([1.0, 0.5, -0.3])
    >>> lambda_it = np.exp(X @ beta_true)
    >>> y = np.random.poisson(lambda_it)
    >>>
    >>> # Setup panel structure
    >>> entity_id = np.repeat(range(20), 5)
    >>> time_id = np.tile(range(5), 20)
    >>>
    >>> # Fit model with cluster-robust SEs
    >>> model = PooledPoisson(y, X, entity_id=entity_id)
    >>> result = model.fit(se_type='cluster')
    >>> print(result.summary())
    >>>
    >>> # Check for overdispersion
    >>> print(f"Overdispersion index: {model.overdispersion:.3f}")

    Notes
    -----
    **Model Assumptions:**
    - Conditional mean equals conditional variance (Var(y|X) = E(y|X))
    - This assumption often violated in practice (see overdispersion check)

    **Standard Errors:**
    - Default: cluster-robust by entity (recommended for panel data)
    - Use se_type='robust' for heteroskedasticity-robust SEs
    - Use se_type='nonrobust' for standard MLE SEs (assumes homoskedasticity)

    References
    ----------
    .. [1] Cameron, A. C., & Trivedi, P. K. (2013). Regression Analysis
           of Count Data (2nd ed.). Cambridge University Press.
    .. [2] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data (2nd ed.). MIT Press. Chapter 19.

    See Also
    --------
    NegativeBinomial : For count data with overdispersion
    PoissonFixedEffects : For fixed effects Poisson
    RandomEffectsPoisson : For random effects Poisson
    """
```

---

## 4. RandomEffectsTobit - Missing Examples

### Location
`panelbox/models/censored/tobit.py` (lines 37-71)

### Current Code
```python
class RandomEffectsTobit(NonlinearPanelModel):
    """
    Random Effects Tobit model for censored panel data.

    The model is:
        y*_it = X_it'β + α_i + ε_it
        y_it = max(c, y*_it) for left censoring
        y_it = min(c, y*_it) for right censoring

    where:
        α_i ~ N(0, σ²_α) is the individual random effect
        ε_it ~ N(0, σ²_ε) is the idiosyncratic error
        c is the censoring point

    Parameters
    ----------
    endog : array-like
        The dependent variable (N*T, 1)
    exog : array-like
        The independent variables (N*T, K)
    groups : array-like
        Group identifiers for panel structure
    time : array-like, optional
        Time identifiers for panel structure
    censoring_point : float, default=0
        The censoring threshold
    censoring_type : str, default='left'
        Type of censoring: 'left', 'right', or 'both'
    lower_limit : float, optional
        Lower censoring point for 'both' type censoring
    upper_limit : float, optional
        Upper censoring point for 'both' type censoring
    quadrature_points : int, default=12
        Number of Gauss-Hermite quadrature points for integration
    """
```

### What's Missing
- Examples section
- Attributes section
- Notes section with assumptions
- References section
- Extended description of methodology

### Recommended Addition (after Parameters)
```python
    Attributes
    ----------
    sigma_eps : float
        Standard deviation of idiosyncratic error σ_ε (after fitting)
    sigma_alpha : float
        Standard deviation of random effect σ_α (after fitting)
    rho : float
        Intra-class correlation ρ = σ²_α / (σ²_α + σ²_ε)
    censored_obs : int
        Number of censored observations

    Examples
    --------
    >>> import numpy as np
    >>> from panelbox.models.censored.tobit import RandomEffectsTobit
    >>>
    >>> # Simulate censored panel data
    >>> np.random.seed(42)
    >>> n_entities = 10
    >>> n_periods = 5
    >>> n_obs = n_entities * n_periods
    >>>
    >>> # Design matrix
    >>> X = np.column_stack([
    ...     np.ones(n_obs),
    ...     np.random.randn(n_obs),
    ...     np.random.randn(n_obs)
    ... ])
    >>>
    >>> # True parameters
    >>> beta = np.array([1.0, 0.5, -0.3])
    >>> sigma_eps = 1.0
    >>> sigma_alpha = 0.5
    >>>
    >>> # Generate data
    >>> alpha_i = np.repeat(
    ...     np.random.normal(0, sigma_alpha, n_entities),
    ...     n_periods
    ... )
    >>> eps = np.random.normal(0, sigma_eps, n_obs)
    >>> y_star = X @ beta + alpha_i + eps
    >>> y = np.maximum(0, y_star)  # Left censoring at 0
    >>>
    >>> groups = np.repeat(range(n_entities), n_periods)
    >>> time = np.tile(range(n_periods), n_entities)
    >>>
    >>> # Fit model
    >>> model = RandomEffectsTobit(
    ...     y, X, groups, time,
    ...     censoring_point=0,
    ...     censoring_type='left',
    ...     quadrature_points=15
    ... )
    >>> result = model.fit()
    >>> print(result.summary())

    Notes
    -----
    **Model Specification:**

    The Random Effects Tobit uses quadrature integration to handle
    the random effects. The likelihood contribution for entity i is:

    ℓᵢ = ∫ [∏ₜ L_it(α_i)] φ(α_i/σ_α) dα_i

    where L_it is the likelihood for observation (i,t) conditional
    on the random effect α_i.

    **Quadrature:**

    - 8-12 points: Usually sufficient
    - 16-20 points: Higher accuracy for publication
    - More points = slower convergence

    **Assumptions:**

    - Random effects α_i are independent of X_it
    - Normality of errors and random effects
    - No serial correlation beyond the random effect

    References
    ----------
    .. [1] Wooldridge, J. M. (2005). "Simple Solutions to the Initial
           Conditions Problem in Dynamic, Nonlinear Panel Data Models
           with Fixed Effects." Journal of Applied Econometrics, 20(1), 39-54.
    .. [2] Honoré, B. E., & Hu, L. (2004). "Estimation of Cross-Sectional
           and Panel Data Models with Sample Selection." Handbook of
           Econometrics, 6, 4597-4637.
```

---

## 5. NegativeBinomial - Missing __init__ Docstring

### Location
`panelbox/models/count/negbin.py` (lines 48-61)

### Current Code
```python
    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Union[np.ndarray, pd.DataFrame],
        entity_id: Optional[Union[np.ndarray, pd.Series]] = None,
        time_id: Optional[Union[np.ndarray, pd.Series]] = None,
        weights: Optional[Union[np.ndarray, pd.Series]] = None
    ):
        """Initialize Negative Binomial model."""
        super().__init__(endog, exog, entity_id, time_id, weights)
        # Model-specific attributes
        self.alpha = None  # Overdispersion parameter
        self.link = 'log'  # Log link function
```

### What's Missing
- Parameter documentation
- Detailed initialization process explanation
- Raises section for validation errors

### Recommended Improvement
```python
    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Union[np.ndarray, pd.DataFrame],
        entity_id: Optional[Union[np.ndarray, pd.Series]] = None,
        time_id: Optional[Union[np.ndarray, pd.Series]] = None,
        weights: Optional[Union[np.ndarray, pd.Series]] = None
    ):
        """
        Initialize Negative Binomial model.

        Parameters
        ----------
        endog : np.ndarray, pd.Series, or pd.DataFrame
            Dependent variable with count data. Must contain
            non-negative integers. Shape (N*T,).
        exog : np.ndarray or pd.DataFrame
            Independent variables. Shape (N*T, K) where K is the
            number of covariates.
        entity_id : np.ndarray, pd.Series, optional
            Entity identifiers for panel structure. Shape (N*T,).
        time_id : np.ndarray, pd.Series, optional
            Time period identifiers. Shape (N*T,).
        weights : np.ndarray, pd.Series, optional
            Observation weights. Shape (N*T,).

        Raises
        ------
        ValueError
            If endog contains negative values or non-integer values.
        ValueError
            If dimensions of exog, endog, entity_id, time_id do not match.
        """
        super().__init__(endog, exog, entity_id, time_id, weights)
        # Model-specific attributes
        self.alpha = None  # Overdispersion parameter
        self.link = 'log'  # Log link function
```

---

## Summary of Changes Needed

| Issue | Priority | Files | Classes | Count |
|-------|----------|-------|---------|-------|
| Missing class docstring | CRITICAL | censored/honore.py, count/poisson.py | HonoreResults, PoissonFixedEffectsResults | 2 |
| Missing __init__ docstrings | HIGH | count/negbin.py, censored/tobit.py | NegativeBinomial, RandomEffectsTobit | 2 |
| Missing Examples section | HIGH | All ordered, all censored | OrderedLogit, OrderedProbit, RandomEffectsOrderedLogit, RandomEffectsTobit, HonoreTrimmedEstimator | 5 |
| Missing References section | HIGH | discrete/ordered.py, count/negbin.py | OrderedLogit, OrderedProbit, RandomEffectsOrderedLogit, NegativeBinomial | 4 |
| Missing method docstrings | MEDIUM | discrete/ordered.py | _cdf, _pdf, summary, _transform_cutpoints, etc. | 7 |
| Formal Google-style | MEDIUM | count/, discrete/ordered.py | Multiple | 10+ |
