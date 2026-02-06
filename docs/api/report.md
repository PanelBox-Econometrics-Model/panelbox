# Report and Export API

API documentation for reporting and export utilities.

---

## LaTeX Export

### to_latex Method

Export estimation results to LaTeX table format.

**Usage:**

```python
results = model.fit()

# Basic export
results.to_latex("table1.tex")

# With options
results.to_latex(
    "table1.tex",
    caption="Investment Regression Results",
    label="tab:investment",
    stars=True,              # Add significance stars
    se_below=True,           # Standard errors below coefficients
    digits=3                 # Decimal places
)
```

**Available via:**

::: panelbox.core.results.PanelResults.to_latex
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## Summary Tables

### summary Method

Generate formatted summary table of estimation results.

**Usage:**

```python
results = model.fit()

# Print summary
print(results.summary())

# Get as string
summary_str = str(results.summary())
```

**Available via:**

::: panelbox.core.results.PanelResults.summary
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

---

## Model Comparison

### Compare Multiple Models

Compare results from multiple models side-by-side.

**Usage:**

```python
import pandas as pd

# Estimate models
pooled = pb.PooledOLS(...).fit()
fe = pb.FixedEffects(...).fit()
re = pb.RandomEffects(...).fit()

# Compare coefficients
comparison = pd.DataFrame({
    'Pooled OLS': pooled.params,
    'Fixed Effects': fe.params,
    'Random Effects': re.params
})

print(comparison)

# Compare standard errors
se_comparison = pd.DataFrame({
    'Pooled OLS': pooled.std_errors,
    'Fixed Effects': fe.std_errors,
    'Random Effects': re.std_errors
})

print(se_comparison)
```

---

## Export Examples

### Example 1: Basic LaTeX Table

```python
results = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year").fit()

results.to_latex(
    "results.tex",
    caption="Fixed Effects Regression",
    label="tab:fe"
)
```

**Output:**
```latex
\begin{table}[htbp]
\centering
\caption{Fixed Effects Regression}
\label{tab:fe}
\begin{tabular}{lcccc}
\toprule
            & Coef. & Std. Err. & t & P>|t| \\
\midrule
x1          & 0.156*** & 0.023 & 6.78 & 0.000 \\
x2          & -0.042* & 0.021 & -2.00 & 0.046 \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item *** p<0.01, ** p<0.05, * p<0.1
\item N=500, Entities=50, R²=0.456
\end{tablenotes}
\end{table}
```

### Example 2: GMM Results with Diagnostics

```python
results = pb.SystemGMM(...).fit()

results.to_latex(
    "gmm_results.tex",
    caption="System GMM Investment Regression",
    include_diagnostics=True  # Include Hansen J, AR tests
)
```

### Example 3: Side-by-Side Comparison

```python
# Estimate multiple models
models = {
    'Difference GMM': pb.DifferenceGMM(...).fit(),
    'System GMM': pb.SystemGMM(...).fit()
}

# Create comparison table
comparison = pd.DataFrame({
    name: res.params for name, res in models.items()
})

# Add standard errors row
for name, res in models.items():
    comparison[f"{name} (SE)"] = res.std_errors

# Export to LaTeX manually
with open("comparison.tex", "w") as f:
    f.write(comparison.to_latex(float_format="%.3f"))
```

---

## Customization

### LaTeX Preamble

If using exported tables, include in your LaTeX document:

```latex
\usepackage{booktabs}  % For \toprule, \midrule, \bottomrule
\usepackage{threeparttable}  % For table notes
```

### Custom Formatting

```python
# Custom number of digits
results.to_latex("table.tex", digits=4)

# No significance stars
results.to_latex("table.tex", stars=False)

# Standard errors in parentheses (inline)
results.to_latex("table.tex", se_below=False)
```
