# Theory Guide: Multinomial Logit for Panel Data

## Introduction

Multinomial logit models **unordered categorical outcomes** with $J > 2$ alternatives. Unlike binary choice (logit/probit), multinomial logit handles multiple discrete outcomes where no natural ordering exists.

**Examples**:
- Occupational choice: unemployed, blue collar, white collar
- Transportation mode: car, bus, train, bike
- Brand choice: Brand A, B, C
- Product category: electronics, clothing, food

## Model Specification

### Utility Maximization Framework

Individual $i$ at time $t$ chooses alternative $j$ that maximizes utility:

$$U_{ijt} = X_{it}'\beta_j + \varepsilon_{ijt}$$

where:
- $X_{it}$ = individual characteristics (covariates)
- $\beta_j$ = alternative-specific coefficients
- $\varepsilon_{ijt}$ = random utility component

**Choice rule**: Choose $j$ if $U_{ijt} > U_{ikt}$ for all $k \neq j$.

### Probability Model

Assuming $\varepsilon_{ijt}$ are **i.i.d. Type I Extreme Value** (Gumbel) distributed:

$$P(y_{it} = j | X_{it}) = \frac{\exp(X_{it}'\beta_j)}{\sum_{k=1}^J \exp(X_{it}'\beta_k)}$$

This is the **multinomial logit** probability.

### Identification

**Normalization required**: Set one category as baseline with $\beta_1 = 0$.

Typically:
- Baseline = most frequent category, or
- Baseline = "none of the above" option, or
- Baseline = reference category by convention

**Result**: Estimate $(J-1) \times K$ parameters where:
- $J$ = number of alternatives
- $K$ = number of covariates

## Interpretation

### Log-Odds Ratios

Coefficients represent **log-odds** of choosing alternative $j$ vs. baseline:

$$\log\left(\frac{P(y=j)}{P(y=\text{baseline})}\right) = X'\beta_j$$

**Example**: If $\beta_{2,\text{education}} = 0.5$ (white collar vs. unemployed):
- One more year of education increases log-odds of white collar vs. unemployed by 0.5
- Odds ratio: $\exp(0.5) \approx 1.65$ (65% increase in odds)

### Marginal Effects

More intuitive: **effect on probabilities**.

For continuous variable $x_k$:
$$\frac{\partial P(y=j)}{\partial x_k} = P(y=j)\left[\beta_{jk} - \sum_{m=1}^J P(y=m)\beta_{mk}\right]$$

**Key properties**:
1. **Sum to zero**: $\sum_{j=1}^J \frac{\partial P(y=j)}{\partial x_k} = 0$ (probabilities sum to 1)
2. **Sign ambiguity**: ME can differ in sign from coefficient!
3. **Depends on probabilities**: Nonlinear effect

### Average Marginal Effects (AME)

Average across individuals:
$$\text{AME}_j(x_k) = \frac{1}{N}\sum_{i=1}^N \frac{\partial P(y_i=j|X_i)}{\partial x_k}$$

**Interpretation**: "On average, one-unit increase in $x_k$ changes probability of outcome $j$ by AME percentage points."

### Example Interpretation

Suppose AME for education on white collar = 0.08:
- 1 additional year of education → 8 percentage point increase in probability of white collar
- This must be offset by decreases in other categories (unemployed, blue collar)

## IIA Assumption

### Independence of Irrelevant Alternatives

**Critical assumption** of multinomial logit:

$$\frac{P(y=j)}{P(y=k)} = \exp(X'(\beta_j - \beta_k))$$

This ratio **does not depend on other alternatives**.

### Implications

Adding or removing an alternative:
- Affects all probabilities proportionally
- Doesn't change relative odds between existing alternatives

### Classic Example: Red Bus / Blue Bus

**Scenario**:
- 3 alternatives: Car, Red Bus, Blue Bus
- Initially: P(Car) = 0.5, P(Red Bus) = 0.5
- Add Blue Bus (identical to Red Bus)
- IIA predicts: P(Car) = 1/3, P(Red Bus) = 1/3, P(Blue Bus) = 1/3

**Problem**: Adding blue bus takes probability from car! Should only affect red bus.

### When IIA Fails

IIA violated when:
- Alternatives are **substitutes** or **complements**
- Unobserved similarities between alternatives
- **Nests** of similar alternatives

**Solutions when IIA violated**:
- **Nested logit**: Group similar alternatives
- **Mixed logit** (random parameters): Allow taste heterogeneity
- **Conditional logit**: Use alternative-specific attributes

## Estimation Methods

### Pooled Multinomial Logit

Standard MLE ignoring panel structure.

**Log-likelihood**:
$$\ell(\beta) = \sum_{i=1}^N \sum_{t=1}^T \sum_{j=1}^J d_{ijt} \log P(y_{it}=j|X_{it})$$

where $d_{ijt} = 1$ if individual $i$ chose $j$ at time $t$.

**Estimation**: Maximize via Newton-Raphson or BFGS.

**Advantages**:
- ✓ Fast
- ✓ Easy to implement
- ✓ Works for large $J$

**Disadvantages**:
- ✗ Ignores panel structure
- ✗ Doesn't account for individual heterogeneity

### Fixed Effects Multinomial Logit

**Chamberlain (1980)** conditional MLE approach.

**Model**:
$$P(y_{it} = j | X_{it}, \alpha_i) = \frac{\exp(X_{it}'\beta_j + \alpha_{ij})}{\sum_{k=1}^J \exp(X_{it}'\beta_k + \alpha_{ik})}$$

where $\alpha_i = (\alpha_{i1}, ..., \alpha_{iJ})$ are individual-specific effects.

**Conditional likelihood**:
Condition on sufficient statistic to eliminate $\alpha_i$.

**Result**: Only individuals with **variation in choices** contribute to likelihood.

**Computation**:
- Enumerate all possible choice sequences
- Weight by probability conditional on observed sequence
- **Very expensive** for large $J$ or $T$

**Practical limit**: $J \leq 4$ and $T \leq 10$

**Advantages**:
- ✓ Controls for individual heterogeneity
- ✓ Consistent even if $\alpha_i$ correlated with $X_{it}$

**Disadvantages**:
- ✗ Computationally intensive
- ✗ Drops individuals without variation
- ✗ Can't estimate effects of time-invariant variables

### Random Effects Multinomial Logit

**Model**:
$$P(y_{it} = j | X_{it}, \alpha_i) = \frac{\exp(X_{it}'\beta_j + \alpha_i)}{\sum_{k=1}^J \exp(X_{it}'\beta_k + \alpha_i)}$$

Assume $\alpha_i \sim N(0, \sigma_\alpha^2)$ independent of $X_{it}$.

**Estimation**: Integrate out random effects:
$$P(y_{it}=j|X_{it}) = \int P(y_{it}=j|X_{it}, \alpha_i) \phi(\alpha_i; \sigma_\alpha^2) d\alpha_i$$

Use **Gauss-Hermite quadrature** for integration.

**Advantages**:
- ✓ Accounts for individual heterogeneity
- ✓ More efficient than FE (uses all observations)
- ✓ Faster than FE

**Disadvantages**:
- ✗ Assumes $\alpha_i \perp X_{it}$ (may be violated)
- ✗ Quadrature becomes slow for large $J$
- ✗ Sensitive to distributional assumptions

## Panel Data Issues

### Time-Varying Covariates

With panel data, covariates change over time:
$$X_{it} = (X_{it,1}, ..., X_{it,K})$$

**Identification**: Time variation identifies coefficients even with FE.

### Time-Invariant Covariates

Variables that don't change: education (after schooling), gender, race.

**Pooled/RE**: Can estimate effects
**FE**: Cannot identify (absorbed by fixed effects)

### State Dependence

**Question**: Does past choice affect current choice?

**Dynamic multinomial logit**:
$$P(y_{it}=j) = f(X_{it}, y_{it-1}, \alpha_i)$$

**Challenges**:
- Initial conditions problem
- Distinguish true state dependence from heterogeneity

## Prediction

### Predicted Probabilities

For individual $i$ with characteristics $X_i$:
$$\hat{P}(y_i = j) = \frac{\exp(X_i'\hat{\beta}_j)}{\sum_{k=1}^J \exp(X_i'\hat{\beta}_k)}$$

### Predicted Choice

$$\hat{y}_i = \arg\max_j \hat{P}(y_i = j)$$

Choose alternative with highest predicted probability.

### Classification Accuracy

$$\text{Accuracy} = \frac{1}{N}\sum_{i=1}^N \mathbb{1}(\hat{y}_i = y_i)$$

Fraction of observations correctly classified.

## Model Diagnostics

### Goodness of Fit

**McFadden's Pseudo R²**:
$$R^2 = 1 - \frac{\ell(\hat{\beta})}{\ell_0}$$

where $\ell_0$ = log-likelihood of null model (equal probabilities).

**Typical values**: 0.2 - 0.4 considered good fit.

**Warning**: Cannot compare across datasets.

### Information Criteria

**AIC**: $-2\ell(\hat{\beta}) + 2k$
**BIC**: $-2\ell(\hat{\beta}) + k\log(N)$

where $k = (J-1) \times K$ = number of parameters.

**Use**: Compare non-nested models (lower is better).

### IIA Tests

**Hausman-McFadden test**:
1. Estimate full model (all $J$ alternatives)
2. Estimate restricted model (drop alternative $j$)
3. Test if coefficients change significantly

**Null**: IIA holds (coefficients unchanged)
**Alternative**: IIA violated

## Computational Aspects

### Convergence

Multinomial logit generally converges well, but:

**Potential issues**:
- **Separation**: Perfect prediction of some alternatives
- **Collinearity**: Similar alternatives cause identification issues
- **Starting values**: Poor initialization

**Solutions**:
- Scale covariates (standardize)
- Use small random perturbations for starting values
- Check Hessian for singularity

### Speed

**Pooled**: Very fast, even for large $N$ and $J$

**FE**: Extremely slow for large $J$ or $T$:
- $J=3, T=5$: Manageable
- $J=4, T=10$: Very slow
- $J=5, T=10$: Impractical

**RE**: Moderate speed, depends on quadrature points.

## Extensions

### Nested Logit

Groups alternatives into **nests**:
```
Transportation
├── Private (Car, Motorcycle)
└── Public (Bus, Train, Subway)
```

**Advantage**: Relaxes IIA within nests.

### Mixed Logit (Random Parameters)

Allow coefficients to vary across individuals:
$$\beta_i \sim f(\beta, \Sigma)$$

**Advantage**:
- Flexible substitution patterns
- Relaxes IIA completely

**Disadvantage**: Computationally intensive (simulation-based).

### Conditional Logit

For **alternative-specific attributes**:
$$U_{ij} = Z_j'\gamma + X_i'\beta_j + \varepsilon_{ij}$$

where $Z_j$ varies across alternatives (e.g., price, quality).

**Example**: Product choice with price varying by product.

## Practical Recommendations

### Model Selection

1. **Start with pooled**: Quick baseline
2. **Test for heterogeneity**: Hausman test (FE vs. RE)
3. **Check IIA**: Hausman-McFadden test
4. **If IIA violated**: Consider nested or mixed logit

### Sample Size

**Minimum** (rule of thumb):
- At least 10-20 observations per parameter
- With $J=3, K=5$: $(J-1) \times K = 10$ parameters → need $N \geq 100-200$

**Practical**: More is better, especially for:
- Rare alternatives
- Many parameters
- Fixed effects

### Reporting Results

**Essential**:
1. Coefficients with cluster-robust SEs
2. Marginal effects (average)
3. Goodness of fit (pseudo R², accuracy)
4. IIA test results

**Useful**:
- Predicted probability plots
- Sensitivity analysis
- Comparison with nested logit

## References

### Key Papers

1. **McFadden, D. (1974)**. "Conditional Logit Analysis of Qualitative Choice Behavior." In *Frontiers in Econometrics*, ed. P. Zarembka.
   - Foundation of discrete choice theory

2. **Chamberlain, G. (1980)**. "Analysis of Covariance with Qualitative Data." *Review of Economic Studies*, 47(1), 225-238.
   - Fixed effects multinomial logit

3. **Train, K. (2009)**. *Discrete Choice Methods with Simulation*, 2nd ed. Cambridge University Press.
   - Comprehensive textbook

### Textbooks

- **Cameron, A.C., & Trivedi, P.K. (2005)**. *Microeconometrics: Methods and Applications*. Chapter 15.
- **Greene, W.H. (2018)**. *Econometric Analysis*, 8th ed. Chapter 18.
- **Wooldridge, J.M. (2010)**. *Econometric Analysis of Cross Section and Panel Data*. Chapter 15.

## See Also

- [Multinomial Logit Tutorial](../tutorials/multinomial_tutorial.ipynb)
- [Multinomial Logit API](../api/multinomial_logit.md)
- [Discrete Choice Models Guide](discrete_choice.md)
