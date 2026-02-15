# Temperature Extremes Analysis using Quantile Regression

## Overview

This example demonstrates the use of quantile regression to analyze climate extremes and heterogeneous effects of climate change across the temperature distribution.

## Key Features

1. **Asymmetric Warming Trends**: Analyze how warming differs across cold vs. hot temperature extremes
2. **Regional Heterogeneity**: Compare temperature trends across different geographic regions
3. **Extreme Event Probability**: Track changes in the probability of extreme weather events over time
4. **Climate Change Impacts**: Examine distributional effects beyond mean temperature changes

## Scientific Context

Traditional climate analysis focuses on mean temperature changes. However, extreme events (heat waves, cold snaps) have disproportionate impacts on:
- Public health (heat-related mortality)
- Agriculture (crop failures)
- Infrastructure (energy demand)
- Ecosystems (species migration)

Quantile regression allows us to:
- Examine trends at different points of the distribution
- Test for asymmetric warming (different rates at cold vs. hot extremes)
- Identify regions with increasing temperature variance
- Predict future extreme event probabilities

## Dataset

The example uses simulated daily temperature panel data with realistic features:
- **Panel structure**: 50 weather stations × 30 years
- **Seasonal patterns**: Annual temperature cycles
- **Spatial variation**: Latitude and elevation effects
- **Climate change**: Long-term warming trend with increasing variance
- **Extreme events**: Fat-tailed distribution (Student's t)

## Key Results

### 1. Asymmetric Warming

The analysis reveals:
- Hot extremes (95th percentile) warming faster than cold extremes (5th percentile)
- Median temperature increasing at moderate rate
- Increasing temperature variance over time

**Economic Interpretation**: Air conditioning demand increases more than heating demand decreases, creating net energy burden.

### 2. Regional Differences

Different warming patterns across regions:
- Northern regions: Larger warming at cold extremes (winter warming)
- Southern regions: Larger warming at hot extremes (summer intensification)
- Coastal vs. inland: Different variance trends

### 3. Extreme Event Trends

Probability of extreme events changing over time:
- Hot extremes: Increasing probability (climate change signal)
- Cold extremes: Decreasing probability (winter warming)
- Regional variation in extreme event evolution

## Usage

### Basic Analysis

```python
from temperature_extremes import TemperatureExtremesAnalysis

# Initialize with simulated data
analysis = TemperatureExtremesAnalysis()

# Or load your own data
analysis = TemperatureExtremesAnalysis(data_path='temperature_data.csv')

# Analyze extreme trends
result, trends = analysis.analyze_extreme_trends()

# Regional comparison
regional_results = analysis.analyze_regional_differences()

# Extreme event probabilities
prob_results = analysis.estimate_extreme_probabilities()
```

### Required Data Format

If using your own data, ensure it has these columns:
- `station_id`: Weather station identifier
- `date`: Date of observation
- `temperature`: Temperature value
- `latitude`: Station latitude (for spatial analysis)
- `elevation`: Station elevation (optional)
- `region`: Geographic region (optional)

### Output

The analysis produces:

1. **Trend Estimates**: Warming rates at each quantile
2. **Statistical Tests**: Heterogeneity tests across distribution
3. **Visualizations**:
   - Warming trends across quantiles
   - Regional comparisons
   - Extreme event probability evolution
4. **Publication-ready figures** (300 DPI PNG)

## Methodological Notes

### Model Specification

```
temperature ~ year + sin(2π × day_of_year/365) + cos(2π × day_of_year/365) + latitude + elevation
```

- `year`: Linear time trend (captures climate change)
- Seasonal controls: Fourier terms for annual cycle
- Spatial controls: Latitude and elevation effects

### Quantile Regression Advantages

1. **Robustness**: Not affected by extreme outliers in ways OLS is
2. **Distributional analysis**: Complete picture beyond mean
3. **Heterogeneity**: Different trends at different quantiles
4. **Policy relevance**: Focus on extremes that matter most

### Inference

- Clustered standard errors at station level (accounts for within-station correlation)
- Bootstrap confidence intervals available
- Heterogeneity tests compare coefficients across quantiles

## Extensions

### Additional Analyses

1. **Precipitation extremes**: Apply same methods to rainfall
2. **Compound events**: Joint analysis of temperature and precipitation
3. **Attribution**: Compare observed vs. counterfactual (no climate change) distributions
4. **Projection**: Forecast future extreme event probabilities

### Advanced Methods

1. **Location-scale model**: Simultaneously model mean and variance
2. **Fixed effects**: Control for time-invariant station characteristics
3. **Spatial correlation**: Account for geographic correlation
4. **Non-stationary trends**: Allow for changing warming rates

## References

### Climate Science
- IPCC (2021). Climate Change 2021: The Physical Science Basis
- Katz, R. W., & Brown, B. G. (1992). Extreme events in a changing climate: variability is more important than averages. *Climatic Change*, 21(3), 289-302.

### Quantile Regression Methods
- Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. *Econometrica*, 33-50.
- Koenker, R. (2004). Quantile regression for longitudinal data. *Journal of Multivariate Analysis*, 91(1), 74-89.

### Applications
- Barbosa, S. M. (2008). Quantile trends in Baltic sea level. *Geophysical Research Letters*, 35(22).
- Piao, S., et al. (2010). The impacts of climate change on water resources and agriculture in China. *Nature*, 467(7311), 43-51.

## Citation

If you use this example in your research, please cite:

```
PanelBox Development Team (2024). Temperature Extremes Analysis using Quantile Regression.
PanelBox Examples. https://github.com/panelbox/panelbox
```

## Support

For questions or issues:
- GitHub Issues: https://github.com/panelbox/panelbox/issues
- Documentation: https://panelbox.readthedocs.io
- Email: contact@panelbox.org
