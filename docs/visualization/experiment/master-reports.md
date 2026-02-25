---
title: "Master Reports"
description: "Combining validation, comparison, and residual analysis into a single navigable HTML report"
---

# Master Reports

## Overview

A **master report** is a single HTML file that combines the experiment overview, all fitted models, and links to validation, comparison, and residual sub-reports. It serves as the entry point for reviewing a complete econometric analysis -- a self-contained, shareable research artifact.

## Generating a Master Report

### Basic Usage

```python
import panelbox as pb

data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# Fit models
exp.fit_model('pooled_ols', name='ols')
exp.fit_model('fixed_effects', name='fe')
exp.fit_model('random_effects', name='re')

# Generate master report
path = exp.save_master_report("master_report.html")
```

### API Reference

```python
path = exp.save_master_report(
    file_path="master_report.html",   # Output file path
    theme="professional",              # Visual theme
    title=None,                        # Custom title (default: 'PanelBox Master Report')
    reports=None                       # List of sub-report references
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | `str` | (required) | Path for the output HTML file |
| `theme` | `str` | `'professional'` | Visual theme: `'professional'`, `'academic'`, or `'presentation'` |
| `title` | `str` or `None` | `None` | Custom report title. Defaults to `'PanelBox Master Report'` |
| `reports` | `list[dict]` or `None` | `None` | Sub-report references (see below) |

**Returns:** `Path` to the saved HTML file.

!!! warning "Prerequisite"
    At least one model must be fitted before generating a master report. If no models exist, a `ValueError` is raised.

## What is Included

The master report contains:

- **Experiment overview** -- formula, number of observations, entities, and time periods
- **Model summary table** -- all fitted models with R$^2$, AIC, and BIC
- **Navigation to sub-reports** -- clickable cards linking to validation, comparison, and residual reports

## Linking Sub-Reports

The `reports` parameter lets you reference sub-reports generated separately. Each entry is a dictionary with the following keys:

```python
exp.save_master_report(
    "master.html",
    title="Panel Data Analysis - Complete Report",
    reports=[
        {
            "type": "validation",
            "title": "Fixed Effects Validation",
            "description": "Specification tests for the FE model",
            "file_path": "validation_report.html"
        },
        {
            "type": "comparison",
            "title": "Model Comparison",
            "description": "Pooled OLS vs FE vs RE",
            "file_path": "comparison_report.html"
        },
        {
            "type": "residuals",
            "title": "Residual Diagnostics",
            "description": "Diagnostic plots and tests for FE",
            "file_path": "residuals_report.html"
        }
    ]
)
```

| Key | Type | Description |
|-----|------|-------------|
| `type` | `str` | Report type: `'validation'`, `'comparison'`, or `'residuals'` |
| `title` | `str` | Display title for the sub-report card |
| `description` | `str` | Brief description shown on the card |
| `file_path` | `str` | Relative path to the sub-report HTML file |

## Themes

PanelBox provides three visual themes for all reports:

=== "Professional"

    ```python
    exp.save_master_report("master.html", theme="professional")
    ```

    - **Color accent**: Blue (#2563eb)
    - **Best for**: Corporate reports, general analysis, client deliverables
    - **Style**: Clean, modern, business-appropriate

=== "Academic"

    ```python
    exp.save_master_report("master.html", theme="academic")
    ```

    - **Color accent**: Gray (#4b5563)
    - **Best for**: Research papers, working papers, dissertations
    - **Style**: Conservative, publication-ready, minimal

=== "Presentation"

    ```python
    exp.save_master_report("master.html", theme="presentation")
    ```

    - **Color accent**: Purple (#7c3aed)
    - **Best for**: Slide decks, demos, classroom presentations
    - **Style**: Bold, eye-catching, high contrast

## Complete Workflow

The recommended workflow generates all sub-reports first, then creates the master report linking to them:

```python
import panelbox as pb

# 1. Set up experiment
data = pb.load_grunfeld()
exp = pb.PanelExperiment(
    data=data,
    formula="invest ~ value + capital",
    entity_col="firm",
    time_col="year"
)

# 2. Fit models
exp.fit_model('pooled_ols', name='ols')
exp.fit_model('fixed_effects', name='fe')
exp.fit_model('random_effects', name='re')

# 3. Generate validation report
val = exp.validate_model('fe', tests='default')
val.save_html("validation.html", test_type="validation", theme="professional")

# 4. Generate comparison report
comp = exp.compare_models()
comp.save_html("comparison.html", test_type="comparison", theme="professional")

# 5. Generate residual report
resid = exp.analyze_residuals('fe')
resid.save_html("residuals.html", test_type="residuals", theme="professional")

# 6. Generate master report
exp.save_master_report(
    "master.html",
    theme="professional",
    title="Grunfeld Investment Analysis 1935-1954",
    reports=[
        {
            "type": "validation",
            "title": "FE Validation",
            "description": "Specification tests for Fixed Effects",
            "file_path": "validation.html"
        },
        {
            "type": "comparison",
            "title": "Model Comparison",
            "description": "OLS vs FE vs RE comparison",
            "file_path": "comparison.html"
        },
        {
            "type": "residuals",
            "title": "Residual Diagnostics",
            "description": "Diagnostic plots for FE model",
            "file_path": "residuals.html"
        }
    ]
)
```

## Master Report as Research Artifact

A well-structured master report serves as a reproducible record of your analysis:

- **Self-contained**: all HTML, CSS, and data are embedded -- no external dependencies
- **Shareable**: send a single file to colleagues, reviewers, or students
- **Archivable**: pair with JSON exports (`save_json()`) for long-term reproducibility
- **Navigable**: the table of contents and sub-report cards provide structure for complex analyses

!!! tip "Archiving Best Practice"
    For full reproducibility, save both the HTML report and JSON results:

    ```python
    val.save_json("validation.json")
    comp.save_json("comparison.json")
    resid.save_json("residuals.json")
    exp.save_master_report("master.html", reports=[...])
    ```

## Comparison with Other Software

| Task | PanelBox | Stata | R |
|------|----------|-------|---|
| Master report | `exp.save_master_report(...)` | `.do` file + `log using` | `rmarkdown::render()` |
| Theme selection | `theme='academic'` | N/A | Custom CSS in YAML |
| Sub-report linking | `reports=[...]` | Manual hyperlinks | Manual `child` documents |
| Self-contained HTML | Built-in | N/A | `self_contained: true` |

## See Also

- [Experiment Overview](index.md) -- Pattern overview and quick start
- [Workflow](fitting.md) -- Fitting and managing models
- [Validation Reports](validation.md) -- Diagnostic testing
- [Comparison Reports](comparison.md) -- Side-by-side model comparison
- [Residual Analysis](residuals.md) -- Residual diagnostics
- [Tutorials: HTML Reports](../../tutorials/visualization.md) -- Step-by-step tutorial
