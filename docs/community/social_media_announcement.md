# Community Engagement Materials

## Twitter/X Announcement Thread

### Thread 1: Main Launch

🎉 Introducing PanelBox Quantile Regression!

The first comprehensive Python implementation of panel QR methods.

🔥 Features:
✅ Fixed Effects QR (Koenker 2004)
✅ Canay Two-Step
✅ Location-Scale (MSS 2019)
✅ Bootstrap inference
✅ Publication-ready plots

🧵👇

---

Why Quantile Regression?

QR lets you see the FULL picture of how variables affect outcomes across the distribution.

Perfect for:
📊 Wage inequality
📈 Financial risk
🌍 Climate extremes
💊 Treatment effects

---

Example: Education Returns

With OLS: "Education increases wages by 8%"

With QR:
- Bottom 10%: +5%
- Median: +8%
- Top 10%: +12%

→ Returns are higher for high earners!

---

Code example:
```python
from panelbox.models.quantile import PooledQuantile

model = PooledQuantile(data, 'wage ~ education',
                       tau=[0.1, 0.5, 0.9])
result = model.fit()
result.plot_coefficients()
```

---

Validated against R ✓
- Coefficients match within 10^-5
- Bootstrap SEs validated
- 95% test coverage

---

Get started:
📦 pip install panelbox[quantile]
📚 Docs: panelbox.readthedocs.io
💻 GitHub: github.com/panelbox/panelbox

## LinkedIn Post

**Excited to announce: PanelBox Quantile Regression is now available! 🎉**

We've built the first comprehensive Python implementation of panel quantile regression methods. This fills a critical gap in the Python econometrics ecosystem.

**Key Features:**
• State-of-the-art estimators (Koenker 2004, Canay 2011, MSS 2019)
• Robust bootstrap inference
• Advanced diagnostics and specification tests
• Publication-ready visualizations
• Validated against R packages with < 10^-5 numerical accuracy

**Why it matters:**
Quantile regression reveals heterogeneous effects across distributions - crucial for understanding inequality, risk, and treatment effects. While R and Stata have had these tools for years, Python users have been limited. Not anymore!

**Real-world applications:**
📊 Labor economics: Analyze wage inequality across the distribution
📈 Finance: Dynamic Value-at-Risk estimation
🌍 Environment: Model climate extremes
💊 Healthcare: Heterogeneous treatment effects

The library is open-source (MIT license) and ready for production use.

Get started: pip install panelbox[quantile]
Documentation: panelbox.readthedocs.io
GitHub: github.com/panelbox/panelbox

#Python #Econometrics #DataScience #QuantileRegression #OpenSource #MachineLearning #Statistics

## Reddit Post (r/Python, r/datascience, r/economics)

### Title: [P] PanelBox: First comprehensive Python implementation of panel quantile regression

Hey everyone! We've just released PanelBox's quantile regression module - the first comprehensive Python implementation of panel QR methods.

**The Problem:**
If you've ever needed to do quantile regression with panel data in Python, you know the pain. While R has `quantreg` and `rqpd`, and Stata has built-in commands, Python users have been stuck with basic cross-sectional methods.

**What We Built:**
- Fixed Effects QR (Koenker 2004 penalized estimator)
- Canay (2011) two-step estimator
- Location-Scale models (Machado-Santos Silva 2019)
- Multiple bootstrap methods (pairs, cluster, wild)
- Advanced diagnostics and specification tests
- Publication-ready visualizations

**Validation:**
We've extensively validated against R implementations:
- Coefficients match within 10^-5
- Bootstrap standard errors validated
- 95% test coverage
- Performance benchmarks show 2-3x speed improvement for multiple quantiles

**Example Use Case:**
```python
from panelbox.models.quantile import CanayTwoStep

# Analyze heterogeneous returns to education
model = CanayTwoStep(panel_data,
                     'log_wage ~ education + experience',
                     tau=[0.1, 0.25, 0.5, 0.75, 0.9])
result = model.fit()

# Plot coefficients across quantiles
result.plot_coefficients('education')
```

**Links:**
- GitHub: https://github.com/panelbox/panelbox
- Docs: https://panelbox.readthedocs.io
- PyPI: pip install panelbox[quantile]

Would love feedback from the community! PRs welcome.

## Conference Presentation Abstract

### Title: Panel Quantile Regression in Python: The PanelBox Implementation

**Abstract:**
We present PanelBox, the first comprehensive Python library for panel quantile regression. While quantile regression has become essential for analyzing heterogeneous effects across distributions, Python implementations have lagged behind R and Stata, particularly for panel data methods. PanelBox addresses this gap by implementing state-of-the-art estimators including Koenker (2004) penalized fixed effects, Canay (2011) two-step estimator, and Machado-Santos Silva (2019) location-scale models. The library provides robust inference via multidimensional bootstrap, advanced diagnostics, and publication-ready visualizations. We demonstrate numerical validation against R packages showing accuracy within 10^-5 and performance improvements of 2-3x for multiple quantile estimation. Three empirical applications illustrate the library's capabilities: wage inequality analysis revealing heterogeneous returns to education, dynamic Value-at-Risk estimation for financial risk management, and climate extremes modeling. PanelBox is open-source (MIT license) and designed for both research and production use, filling a critical gap in the Python econometrics ecosystem.

**Keywords:** Quantile Regression, Panel Data, Python, Econometrics, Open Source

## Blog Post Outline

### Title: Introducing PanelBox Quantile Regression: Bringing Advanced Panel QR to Python

1. **Introduction**
   - The quantile regression revolution
   - Why Python has lagged behind
   - What PanelBox brings to the table

2. **The Power of Quantile Regression**
   - Beyond average effects
   - Real-world examples
   - Visual demonstrations

3. **Technical Implementation**
   - Architecture overview
   - Key algorithms
   - Performance optimizations

4. **Validation and Testing**
   - Comparison with R
   - Numerical accuracy
   - Test coverage

5. **Getting Started**
   - Installation
   - Basic example
   - Advanced features

6. **Use Cases**
   - Wage inequality analysis
   - Financial risk (VaR)
   - Environmental extremes

7. **Contributing**
   - Open issues
   - Development roadmap
   - How to contribute

8. **Conclusion**
   - Future plans
   - Community involvement
   - Acknowledgments
