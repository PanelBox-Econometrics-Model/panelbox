# Financial Risk Analysis with Quantile Regression

Value at Risk (VaR) and Conditional VaR analysis using quantile regression methods.

## Overview

This example demonstrates how to use quantile regression for financial risk management:

1. **Value at Risk (VaR)** - Estimating maximum expected loss at given confidence levels
2. **Conditional VaR (CVaR/ES)** - Expected Shortfall beyond VaR
3. **Backtesting** - Validating VaR models using Kupiec test
4. **Stress Testing** - Analyzing extreme tail events

## Quick Start

```python
from var_analysis import ValueAtRiskAnalysis

# Initialize with tickers (uses real data from Yahoo Finance if available)
tickers = ['SPY', 'QQQ', 'IWM']
analysis = ValueAtRiskAnalysis(tickers)

# Estimate VaR
var_results = analysis.estimate_var(confidence_levels=[0.95, 0.99])

# Estimate CVaR
cvar_results = analysis.estimate_cvar()

# Stress testing
analysis.stress_testing()

# Generate report
analysis.generate_report()
```

## Key Concepts

### Value at Risk (VaR)

VaR answers the question: **"What is the maximum loss I can expect with X% confidence?"**

For example, 95% VaR = -2% means:
- 95% of the time, losses will be less than 2%
- 5% of the time, losses may exceed 2%

**Formula:**
$$\text{VaR}_\alpha = Q_{\text{Returns}}(1-\alpha)$$

Where $Q$ is the quantile function and $\alpha$ is the confidence level.

### Conditional VaR (Expected Shortfall)

CVaR answers: **"If I do exceed VaR, how bad will it be on average?"**

CVaR is always worse (more negative) than VaR because it measures the average loss in the tail.

**Formula:**
$$\text{CVaR}_\alpha = E[R | R \leq \text{VaR}_\alpha]$$

**Advantages over VaR:**
- Coherent risk measure
- Captures tail risk
- Sub-additive (portfolio CVaR ≤ sum of individual CVaRs)

### Backtesting with Kupiec Test

The Kupiec test checks if violations occur at the expected rate:

**Null Hypothesis:** Violation rate = $1 - \alpha$

**Test Statistic:**
$$LR = -2[\ln L(\alpha) - \ln L(\hat{p})]$$

Where $\hat{p} = \frac{N_{\text{violations}}}{N_{\text{total}}}$

If LR > 3.84 (chi-square critical value at 5%), reject the model.

## Example Results

### Typical VaR Estimates

| Asset | 95% VaR | 99% VaR | CVaR (95%) |
|-------|---------|---------|------------|
| SPY   | -1.5%   | -2.5%   | -2.2%      |
| QQQ   | -2.0%   | -3.2%   | -2.9%      |
| IWM   | -2.2%   | -3.5%   | -3.1%      |

### Interpretation

- **Small-cap stocks (IWM)** have higher risk than large-cap (SPY)
- **Tech-heavy Nasdaq (QQQ)** has moderate-high risk
- **CVaR is always worse** than VaR, showing additional tail risk

## Use Cases

### 1. Risk Management

Set position limits based on VaR:
```python
portfolio_value = 1_000_000
var_95 = -0.015  # -1.5%
max_loss = portfolio_value * abs(var_95)  # $15,000

print(f"Maximum expected daily loss (95% confidence): ${max_loss:,.0f}")
```

### 2. Regulatory Compliance

Basel Accords require banks to hold capital against VaR:
```python
var_99 = -0.025  # -2.5%
portfolio_value = 100_000_000
capital_requirement = portfolio_value * abs(var_99) * 3  # Multiplier

print(f"Regulatory capital requirement: ${capital_requirement:,.0f}")
```

### 3. Performance Attribution

Compare risk-adjusted returns:
```python
annual_return = 0.10  # 10%
var_95 = -0.015

# Return-to-VaR ratio (similar to Sharpe ratio)
risk_adjusted_return = annual_return / abs(var_95)
print(f"Return-to-VaR ratio: {risk_adjusted_return:.2f}")
```

## Advantages of Quantile Regression for VaR

### Traditional Method: Historical Simulation
- Simply takes the 5th percentile of historical returns
- Assumes past distribution = future distribution
- No conditional modeling

### Quantile Regression Method
- **Conditional VaR**: VaR can depend on market conditions
- **Dynamic estimation**: Adapts to changing volatility
- **Covariate effects**: Incorporate predictors (volatility, volume, etc.)

Example with covariates:
```python
# VaR depends on recent volatility
Q_returns(0.05 | vol_high) = -3%
Q_returns(0.05 | vol_low) = -1%
```

## Outputs

The analysis generates:

- `var_analysis.png` - Time series plot of returns with VaR bands and violations
- `risk_analysis_report.html` - Complete HTML report with metrics and methodology

## Advanced Topics

### Time-Varying VaR

For more sophisticated analysis, model VaR as a function of state variables:

$$\text{VaR}_t(\tau) = \beta_0(\tau) + \beta_1(\tau) \sigma_{t-1} + \beta_2(\tau) r_{t-1}$$

Where:
- $\sigma_{t-1}$ is lagged volatility
- $r_{t-1}$ is lagged return
- $\tau = 0.05$ for 95% VaR

### CAViaR Models

Conditional Autoregressive VaR (Engle & Manganelli, 2004):

$$\text{VaR}_t = \beta_0 + \beta_1 \text{VaR}_{t-1} + \beta_2 |r_{t-1}|$$

This ensures VaR is always a function of recent market conditions.

## References

- Engle, R. F., & Manganelli, S. (2004). CAViaR: Conditional autoregressive value at risk by regression quantiles. *Journal of Business & Economic Statistics*, 22(4), 367-381.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *The Journal of Derivatives*, 3(2), 73-84.
- Acerbi, C., & Tasche, D. (2002). On the coherence of expected shortfall. *Journal of Banking & Finance*, 26(7), 1487-1503.

## Disclaimer

This example is for educational purposes only. VaR models have limitations:

- **Model risk**: All models are wrong; some are useful
- **Tail risk**: VaR may underestimate extreme events (black swans)
- **Market crises**: Historical data may not reflect future crises

Always use multiple risk measures and stress tests. Consult with financial professionals before making investment decisions.
