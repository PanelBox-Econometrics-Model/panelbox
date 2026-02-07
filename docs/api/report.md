# Report and Export API

API documentation for reporting and export utilities.

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

### Example: Side-by-Side Model Comparison

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

