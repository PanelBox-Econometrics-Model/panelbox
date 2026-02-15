# Understanding Spatial Autocorrelation

## Introduction

Spatial autocorrelation is a fundamental concept in spatial econometrics that describes the correlation of a variable with itself across space. Just as time series data can exhibit temporal autocorrelation, spatial data often shows spatial dependence where nearby observations are more similar than distant ones.

## What is Spatial Autocorrelation?

### Tobler's First Law of Geography

> "Everything is related to everything else, but near things are more related than distant things." - Waldo Tobler (1970)

This principle underlies most spatial analysis. In economic terms, it means:
- Housing prices in neighboring areas tend to be similar
- Unemployment rates cluster geographically
- Technology adoption spreads through geographic proximity
- Environmental conditions affect nearby regions similarly

### Types of Spatial Autocorrelation

**Positive Spatial Autocorrelation**
- Similar values cluster together
- High values near high values, low near low
- Most common in economic data
- Examples: Income levels, crime rates, housing prices

**Negative Spatial Autocorrelation**
- Dissimilar values are neighbors
- High values near low values
- Less common but important
- Examples: Competing businesses, checkboard patterns

**No Spatial Autocorrelation**
- Random spatial pattern
- Values independent of location
- Rare in real-world economic data

## Why Does Spatial Autocorrelation Matter?

### Statistical Implications

When spatial autocorrelation is present but ignored:

1. **Biased Estimates**: OLS estimates may be biased and inconsistent
2. **Invalid Inference**: Standard errors are incorrect, leading to wrong hypothesis tests
3. **Inefficient Estimates**: Even when unbiased, estimates are not efficient
4. **Prediction Errors**: Forecasts ignore valuable spatial information

### Economic Implications

Spatial autocorrelation often reflects important economic mechanisms:

1. **Spillover Effects**: Economic shocks spread geographically
2. **Agglomeration**: Economic activity clusters for efficiency
3. **Diffusion Processes**: Innovation and technology spread through networks
4. **Common Factors**: Regions share unobserved characteristics

## Measuring Spatial Autocorrelation

### Global Moran's I

The most common measure of global spatial autocorrelation:

$$I = \frac{N}{\sum_{i}\sum_{j}w_{ij}} \frac{\sum_{i}\sum_{j}w_{ij}(y_i - \bar{y})(y_j - \bar{y})}{\sum_{i}(y_i - \bar{y})^2}$$

Where:
- $N$ is the number of spatial units
- $y_i$ is the value at location $i$
- $\bar{y}$ is the mean of $y$
- $w_{ij}$ is the spatial weight between units $i$ and $j$

**Interpretation**:
- $I > 0$: Positive spatial autocorrelation
- $I < 0$: Negative spatial autocorrelation
- $I ≈ 0$: No spatial autocorrelation
- Expected value under null: $E[I] = -1/(N-1)$

### Local Indicators (LISA)

Local Indicators of Spatial Association identify clusters and outliers:

$$I_i = \frac{(y_i - \bar{y})}{\sigma^2} \sum_{j}w_{ij}(y_j - \bar{y})$$

**LISA Categories**:
- **High-High (HH)**: Hot spots - high values surrounded by high values
- **Low-Low (LL)**: Cold spots - low values surrounded by low values
- **High-Low (HL)**: Spatial outliers - high values surrounded by low values
- **Low-High (LH)**: Spatial outliers - low values surrounded by high values

## Sources of Spatial Autocorrelation

### True Spatial Dependence

Direct interaction between spatial units:
- **Spillovers**: One region's outcome directly affects neighbors
- **Diffusion**: Processes that spread geographically
- **Strategic Interaction**: Regions respond to neighbors' policies

Mathematical representation:
$$y = \rho Wy + X\beta + \varepsilon$$

Where $\rho Wy$ captures the spatial dependence.

### Spatial Heterogeneity

Apparent spatial autocorrelation due to:
- **Omitted Variables**: Missing spatially correlated variables
- **Structural Differences**: Different parameters across regions
- **Measurement Issues**: Spatial aggregation effects

### Common Factors

Shared unobserved factors affecting neighboring regions:
- Climate and geography
- Cultural factors
- Historical events
- Infrastructure networks

## Testing for Spatial Autocorrelation

### Moran's I Test

**Null Hypothesis**: No spatial autocorrelation

**Test Statistic**: Standardized Moran's I
$$Z_I = \frac{I - E[I]}{\sqrt{Var[I]}}$$

Under the null, $Z_I \sim N(0,1)$ asymptotically.

**Implementation in PanelBox**:
```python
from panelbox.validation.spatial import MoranIPanelTest

test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
result = test.run()
print(f"Moran's I: {result.statistic:.4f}")
print(f"p-value: {result.pvalue:.4f}")
```

### Lagrange Multiplier Tests

More specific tests for different types of spatial dependence:

**LM-Lag Test**: Tests for spatial lag dependence
$$LM_{\rho} = \frac{(e'Wy/\sigma^2)^2}{T}$$

**LM-Error Test**: Tests for spatial error dependence
$$LM_{\lambda} = \frac{(e'We/\sigma^2)^2}{tr(W'W + W^2)}$$

Where $e$ are OLS residuals and $T = tr[(W'W + W^2)]$.

### Decision Rule for Model Selection

Based on LM test results:

1. If only LM-Lag significant → Use SAR model
2. If only LM-Error significant → Use SEM model
3. If both significant → Check robust versions:
   - Robust LM-Lag significant → SAR model
   - Robust LM-Error significant → SEM model
   - Both robust significant → SDM or GNS model

## Spatial Autocorrelation in Panel Data

### Additional Complexity

Panel data adds temporal dimension:
- Spatial autocorrelation may vary over time
- Need to account for both spatial and temporal dependence
- Fixed effects control for time-invariant spatial heterogeneity

### Testing Strategies

**Period-by-Period Testing**:
```python
# Test each time period separately
for t in time_periods:
    data_t = data[data.time == t]
    moran_t = MoranIPanelTest(data_t.y, W)
    print(f"Period {t}: I = {moran_t.statistic:.3f}")
```

**Pooled Testing**:
```python
# Test on pooled residuals
moran_pooled = MoranIPanelTest(
    residuals, W, entity_ids, time_ids,
    method='pooled'
)
```

**Joint Testing**:
```python
# Joint test accounting for panel structure
moran_joint = MoranIPanelTest(
    residuals, W, entity_ids, time_ids,
    method='joint'
)
```

## Implications for Modeling

### When to Use Spatial Models

Use spatial models when:
- Moran's I test rejects null (p < 0.05)
- Economic theory suggests spatial interaction
- Prediction across space is important
- Policy spillovers need quantification

### Model Choice Based on Autocorrelation Type

**Global Spillovers** → Spatial Lag Model (SAR)
- Outcome in one region affects all others
- Multiplier effects present
- Example: Regional GDP spillovers

**Local Spillovers** → Spatial Error Model (SEM)
- Shocks correlated across neighbors only
- No multiplier effects
- Example: Weather shocks

**Complex Spillovers** → Spatial Durbin Model (SDM)
- Both outcome and covariate spillovers
- Rich interaction structure
- Example: Technology diffusion

## Visualizing Spatial Autocorrelation

### Moran Scatterplot

Shows relationship between a variable and its spatial lag:

```python
import matplotlib.pyplot as plt
from panelbox.visualization import moran_scatterplot

fig, ax = plt.subplots(figsize=(8, 6))
moran_scatterplot(y, W, ax=ax)
plt.show()
```

Quadrants indicate:
- I (HH): High-High clusters
- II (LH): Low values near high values
- III (LL): Low-Low clusters
- IV (HL): High values near low values

### LISA Cluster Map

Identifies local clusters and outliers:

```python
from panelbox.visualization import lisa_map

lisa = LocalMoranI(y, W)
clusters = lisa.get_clusters()
lisa_map(gdf, clusters)
```

Color coding:
- Red: High-High clusters (hot spots)
- Blue: Low-Low clusters (cold spots)
- Light red: High-Low outliers
- Light blue: Low-High outliers
- White: Not significant

## Best Practices

### 1. Always Test First

Never assume spatial autocorrelation:
```python
# Standard workflow
1. Estimate OLS/Panel model
2. Test residuals for spatial autocorrelation
3. Choose spatial model if needed
4. Verify autocorrelation is removed
```

### 2. Consider Multiple Weight Matrices

Different W matrices capture different relationships:
```python
W_contiguity = SpatialWeights.from_contiguity(gdf)
W_distance = SpatialWeights.from_distance(coords, threshold=100)
W_knn = SpatialWeights.from_knn(coords, k=5)

# Test with each
for W, name in [(W_contiguity, 'Contiguity'), ...]:
    moran = MoranIPanelTest(residuals, W)
    print(f"{name}: I = {moran.statistic:.3f}")
```

### 3. Check Robustness

Results should not be overly sensitive to W specification:
```python
# Estimate with different W matrices
results = {}
for W_type in ['contiguity', 'knn5', 'knn10', 'distance50']:
    W = create_W(W_type)
    model = SpatialLag(data, W, ...)
    results[W_type] = model.fit()

# Compare key parameters
compare_parameters(results)
```

### 4. Interpret Carefully

Remember that spatial autocorrelation can arise from:
- True spatial interaction (spillovers)
- Omitted spatially correlated variables
- Model misspecification
- Measurement error

## References

1. Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers.
2. Anselin, L. (1995). Local indicators of spatial association—LISA. *Geographical Analysis*, 27(2), 93-115.
3. Moran, P.A.P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37, 17-23.
4. Tobler, W.R. (1970). A computer movie simulating urban growth in the Detroit region. *Economic Geography*, 46, 234-240.
