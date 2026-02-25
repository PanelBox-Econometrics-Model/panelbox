---
title: "Panel Plots"
description: "Panel structure visualization including entity effects, time effects, and variance decomposition with PanelBox"
---

# Panel Plots

Panel structure plots help you understand the characteristics of your panel data: entity-specific effects, time trends, variance decomposition, and data balance. PanelBox provides 4 interactive charts tailored for panel data analysis.

## Quick Start

```python
from panelbox.visualization import create_panel_charts

# Create all 4 panel charts from model results
charts = create_panel_charts(results, theme="professional", include_html=False)

# Access individual charts
charts["entity_effects"].to_html()
charts["structure"].save_image("balance.png")
```

You can also select specific chart types:

```python
charts = create_panel_charts(
    results,
    chart_types=["entity_effects", "between_within"],
    theme="academic",
    include_html=False,
)
```

## Entity Effects Plot

**Registry name**: `panel_entity_effects`

Horizontal bar chart of entity-specific fixed or random effects with optional confidence intervals. Helps identify which entities deviate most from the overall mean.

```python
from panelbox.visualization import create_entity_effects_plot

chart = create_entity_effects_plot(
    panel_results,
    theme="professional",
    sort_by="magnitude",      # 'magnitude', 'alphabetical', 'significance'
    max_entities=20,          # Limit display for large panels
    show_confidence=True,
)
chart.figure.show()
```

Using the factory directly:

```python
from panelbox.visualization import ChartFactory

chart = ChartFactory.create(
    "panel_entity_effects",
    data={
        "entity_id": ["Firm A", "Firm B", "Firm C", ...],
        "effect": [0.23, -0.15, 0.08, ...],
        "std_error": [0.05, 0.04, 0.06, ...],   # Optional
    },
    theme="academic",
)
```

**Interpretation**:

- Large positive effects: entities with higher-than-average outcomes
- Large negative effects: entities with lower-than-average outcomes
- Wide confidence intervals: imprecise estimates, possibly few observations for that entity
- Clustering of effects: groups of similar entities

## Time Effects Plot

**Registry name**: `panel_time_effects`

Line chart of time fixed effects with confidence bands and significance highlighting. Shows how the outcome variable trends over time after controlling for entity effects and covariates.

```python
from panelbox.visualization import create_time_effects_plot

chart = create_time_effects_plot(
    panel_results,
    theme="professional",
    show_confidence=True,
    highlight_significant=True,
)
```

Using the factory directly:

```python
chart = ChartFactory.create(
    "panel_time_effects",
    data={
        "time": [2000, 2001, 2002, 2003, ...],
        "effect": [0.0, 0.12, 0.25, 0.18, ...],
        "std_error": [0.03, 0.03, 0.04, 0.03, ...],   # Optional
    },
    theme="academic",
)
```

**Interpretation**:

- Upward trend: positive time effects, possibly reflecting macro trends
- Sudden jumps: structural breaks or policy changes
- Significant periods (starred): time effects statistically different from zero
- Confidence band width: precision of time effect estimates

## Between-Within Plot

**Registry name**: `panel_between_within`

Variance decomposition showing how much of each variable's variation is between entities versus within entities over time.

```python
from panelbox.visualization import create_between_within_plot

chart = create_between_within_plot(
    panel_data,
    variables=["wage", "education", "experience"],
    theme="academic",
    style="stacked",          # 'stacked', 'side_by_side', or 'scatter'
    show_percentages=True,
)
```

Three visualization styles are available:

=== "Stacked Bars"

    ```python
    chart = create_between_within_plot(data, style="stacked")
    ```

    Stacked bar chart showing between and within variance as proportions of total variance.

=== "Side-by-Side Bars"

    ```python
    chart = create_between_within_plot(data, style="side_by_side")
    ```

    Grouped bar chart for direct comparison of absolute variance magnitudes.

=== "Scatter Plot"

    ```python
    chart = create_between_within_plot(data, style="scatter")
    ```

    Scatter plot of between vs within variance, with a 45-degree reference line.

Using the factory directly:

```python
chart = ChartFactory.create(
    "panel_between_within",
    data={
        "variables": ["wage", "education", "experience"],
        "between_var": [2.5, 1.8, 3.2],
        "within_var": [1.2, 0.4, 2.1],
    },
    theme="professional",
    config={"chart_type": "stacked"},
)
```

**Interpretation**:

- High between-variation: variable differs mainly across entities (cross-sectional variation dominates)
- High within-variation: variable changes mainly over time within entities (temporal variation dominates)
- Implications for estimation:
    - Fixed Effects exploits within-variation only
    - Random Effects uses both, but requires exogeneity
    - Between Estimator uses between-variation only

## Panel Structure Plot

**Registry name**: `panel_structure`

Heatmap showing data presence/absence across entities and time periods. Essential for diagnosing unbalanced panels and understanding missing data patterns.

```python
from panelbox.visualization import create_panel_structure_plot

chart = create_panel_structure_plot(
    panel_data,
    theme="presentation",
    show_statistics=True,
    highlight_complete=True,
)
```

Using the factory directly:

```python
chart = ChartFactory.create(
    "panel_structure",
    data={
        "entities": ["Firm A", "Firm B", "Firm C"],
        "time_periods": [2000, 2001, 2002, 2003],
        "presence_matrix": [
            [1, 1, 1, 1],   # Firm A: complete
            [1, 1, 0, 1],   # Firm B: missing 2002
            [0, 1, 1, 1],   # Firm C: missing 2000
        ],
    },
    theme="professional",
)
```

**Interpretation**:

- All green: balanced panel (all entities observed in all periods)
- Red cells: missing observations, investigate the pattern
- Systematic gaps: attrition, late entry, or structural missing data
- Random gaps: may be ignorable under MAR assumption

## Data Transformers

The `PanelDataTransformer` extracts panel-specific data from model results:

```python
from panelbox.visualization.transformers.panel import PanelDataTransformer

# Extract entity effects from model results
entity_data = PanelDataTransformer.extract_entity_effects(panel_results)

# Extract time effects
time_data = PanelDataTransformer.extract_time_effects(panel_results)

# Calculate between-within variance
bw_data = PanelDataTransformer.calculate_between_within(
    panel_data, variables=["wage", "education"]
)

# Analyze panel structure
structure = PanelDataTransformer.analyze_panel_structure(panel_data)
```

## Complete Example

Full panel structure analysis workflow:

```python
import panelbox as pb
from panelbox.visualization import (
    create_panel_charts,
    create_between_within_plot,
    export_charts,
)

# Load data and estimate model
data = pb.load_dataset("wagepan")
model = pb.FixedEffects(data=data, formula="lwage ~ hours + exper + EntityEffects")
results = model.fit()

# Generate all panel charts
charts = create_panel_charts(results, theme="professional", include_html=False)

# Create additional variance decomposition
bw_chart = create_between_within_plot(
    data,
    variables=["lwage", "hours", "exper"],
    theme="academic",
    style="stacked",
    show_percentages=True,
)

# Export everything
export_charts(charts, output_dir="./panel_analysis", format="png", width=900, height=600)
bw_chart.save_image("panel_analysis/variance_decomposition.png")
```

## Comparison with Other Software

| Chart | PanelBox | Stata | R |
|-------|----------|-------|---|
| Entity effects | `panel_entity_effects` | Manual after `xtreg` | Custom ggplot |
| Time effects | `panel_time_effects` | Manual after `xtreg` | Custom ggplot |
| Variance decomposition | `panel_between_within` | `xtsum` (table only) | Custom calculation |
| Panel balance | `panel_structure` | `xtset`, `xtdescribe` | `plm::pdim()`, `plm::pbalanced()` |

## See Also

- [Diagnostics Plots](model-diagnostics.md) -- Residual diagnostics for panel models
- [Comparison Plots](comparison.md) -- Compare FE, RE, and Pooled OLS
- [Themes & Customization](themes.md) -- Style your panel charts
- [Test Plots](test-plots.md) -- Hausman test and other panel diagnostics
