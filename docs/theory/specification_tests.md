# Theory Guide: Model Specification Tests

## Introduction

Model specification tests help researchers choose between competing econometric models. This guide covers tests for **non-nested models** where neither model is a special case of the other.

## Davidson-MacKinnon J-Test

### Motivation

When comparing two models that are not nested:
- **Model 1**: $y = X_1'\beta_1 + \varepsilon_1$
- **Model 2**: $y = X_2'\beta_2 + \varepsilon_2$

Standard tests (F-test, LR test) don't apply because models are not nested.

### Test Procedure

The J-test uses an **artificial nesting** approach:

**Step 1:** Estimate both models separately
- Estimate Model 1 → obtain fitted values $\hat{y}_1$
- Estimate Model 2 → obtain fitted values $\hat{y}_2$

**Step 2: Forward Test** (Model 1 vs. Model 2)

Estimate the augmented regression:
$$y = X_1'\beta_1 + \alpha \hat{y}_2 + u$$

Test hypothesis: $H_0: \alpha = 0$ (Model 1 is correct)

- If **reject** $H_0$ → Model 2 has additional explanatory power
- If **don't reject** → Model 1 is adequate

**Step 3: Reverse Test** (Model 2 vs. Model 1)

Estimate:
$$y = X_2'\beta_2 + \gamma \hat{y}_1 + u$$

Test: $H_0: \gamma = 0$ (Model 2 is correct)

### Interpretation Matrix

| Forward (α=0) | Reverse (γ=0) | Conclusion |
|---------------|---------------|------------|
| Reject        | Don't reject | Prefer Model 2 |
| Don't reject  | Reject        | Prefer Model 1 |
| Reject        | Reject        | Neither model adequate |
| Don't reject  | Don't reject  | Both models acceptable |

### Key Insights

1. **Both reject**: Neither model is well-specified
   - Consider alternative specifications
   - May indicate omitted variables or functional form issues

2. **Both don't reject**: Cannot discriminate
   - Use other criteria: economic theory, simplicity, AIC/BIC
   - May indicate models are observationally equivalent

3. **One rejects**: Clear preference
   - Choose the model that is not rejected

### Mathematical Foundation

Under $H_0$ (Model 1 correct):
- $E[\hat{y}_2 | X_1]$ should not add information
- $\alpha = 0$ asymptotically

Under $H_A$ (Model 2 correct):
- $\hat{y}_2$ captures true data generating process
- $\alpha \neq 0$

### Asymptotic Properties

- **Consistency**: Test is consistent against fixed alternatives
- **Distribution**: Under $H_0$, $t$-statistic ~ $N(0,1)$ asymptotically
- **Power**: Depends on sample size and degree of misspecification

## Encompassing Tests

### Cox Test

Similar to J-test but uses likelihood-based approach.

**Null hypothesis**: Model 1 encompasses Model 2
- Model 1 can explain predictions from Model 2

**Test statistic**:
$$T = \frac{\text{LLF}_1 - \text{LLF}_2}{\text{SE}}$$

where LLF = log-likelihood function.

### Wald Encompassing Test

Tests whether restricted model is encompassed by unrestricted model.

**Statistic**:
$$W = (R\hat{\beta} - r)'[R\hat{V}(\hat{\beta})R']^{-1}(R\hat{\beta} - r)$$

Under $H_0$: $W \sim \chi^2(q)$ where $q$ = number of restrictions.

## Panel Data Considerations

### Cluster-Robust Inference

For panel data, **cluster-robust standard errors** are essential:

1. **Within-cluster correlation**: Observations for same entity correlated
2. **Heteroskedasticity**: Variance differs across entities
3. **Serial correlation**: Observations over time correlated

**Implementation**: Cluster by entity when computing test statistics.

### Fixed Effects

When comparing models with fixed effects:
- Include fixed effects in both models consistently
- J-test uses transformed (within) variables
- Interpretation unchanged

## Examples

### Example 1: Cobb-Douglas vs. Translog

**Model 1 (Cobb-Douglas)**:
$$\log Y = \beta_0 + \beta_1 \log K + \beta_2 \log L + \varepsilon$$

**Model 2 (Translog)**:
$$\log Y = \beta_0 + \beta_1 \log K + \beta_2 \log L + \beta_3 (\log K)^2 + \beta_4 (\log L)^2 + \beta_5 \log K \log L + \varepsilon$$

Are these nested? **No** — we can't obtain Translog from Cobb-Douglas by setting parameters to zero (squared terms are different variables).

**J-test procedure**:
1. Estimate both models
2. Forward test: Does Translog fitted value improve Cobb-Douglas?
3. Reverse test: Does Cobb-Douglas fitted value improve Translog?

### Example 2: Linear vs. Different Variable Set

**Model 1**: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$

**Model 2**: $y = \gamma_0 + \gamma_1 z_1 + \gamma_2 z_2 + \varepsilon$

If $x_i \neq z_i$, models are non-nested.

## Practical Guidelines

### When to Use J-Test

✓ **Use when**:
- Models are non-nested
- Same dependent variable
- Economic theory suggests multiple specifications
- Want data-driven model selection

✗ **Don't use when**:
- Models are nested (use F-test or LR test)
- Different dependent variables
- Sample sizes very small (low power)

### Complementary Approaches

Combine J-test with:

1. **Information criteria**: AIC, BIC
2. **Cross-validation**: Out-of-sample prediction
3. **Economic theory**: Plausibility of mechanisms
4. **Robustness checks**: Subsamples, alternative specifications

### Common Pitfalls

1. **Data mining**: Testing many specifications sequentially
   - **Solution**: Pre-specify based on theory

2. **Ignoring clustering**: Using non-robust SEs in panel data
   - **Solution**: Always use cluster-robust inference

3. **Over-interpreting**: "Both acceptable" doesn't mean both correct
   - **Solution**: Use economic judgment

4. **Low power**: Small samples may not reject either model
   - **Solution**: Bootstrap or simulation-based tests

## Computational Details

### Test Statistic

For forward test:
$$t_\alpha = \frac{\hat{\alpha}}{\text{SE}(\hat{\alpha})}$$

where SE is cluster-robust standard error.

**Critical value**: $|t_\alpha| > 1.96$ for 5% level (two-sided).

### Degrees of Freedom

- Large samples: use normal approximation
- Small samples: use $t$-distribution with $N - K$ df

### Clustered Data

With $G$ clusters:
$$\text{Var}(\hat{\alpha}) = (X'X)^{-1}\left(\sum_{g=1}^G u_g u_g' X_g'X_g\right)(X'X)^{-1}$$

where $u_g$ are residuals for cluster $g$.

## References

### Seminal Papers

1. **Davidson, R., & MacKinnon, J.G. (1981)**. "Several Tests for Model Specification in the Presence of Alternative Hypotheses." *Econometrica*, 49(3), 781-793.
   - Original J-test paper
   - Shows consistency and asymptotic distribution

2. **Cox, D.R. (1961)**. "Tests of Separate Families of Hypotheses." *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 105-123.
   - Likelihood-based encompassing test

3. **Mizon, G.E., & Richard, J.F. (1986)**. "The Encompassing Principle and Its Application to Testing Non-Nested Hypotheses." *Econometrica*, 54(3), 657-678.
   - General encompassing framework

### Textbooks

- **Cameron, A.C., & Trivedi, P.K. (2005)**. *Microeconometrics: Methods and Applications*. Chapter 8.
- **Greene, W.H. (2018)**. *Econometric Analysis*, 8th ed. Chapter 5.4.
- **Wooldridge, J.M. (2010)**. *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. Chapter 18.

## See Also

- [J-Test Tutorial](../tutorials/jtest_tutorial.ipynb)
- [Model Selection Guide](model_selection.md)
- [Specification Tests API](../api/specification_tests.md)
