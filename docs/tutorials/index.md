---
title: "Tutorials & Notebooks"
description: "193 interactive Jupyter notebooks for learning panel data econometrics with PanelBox"
---

# Tutorials & Notebooks

PanelBox provides **193 interactive Jupyter notebooks** organized across 15 tutorial categories, covering every aspect of panel data econometrics. Each notebook runs directly in Google Colab with zero setup required.

!!! tip "Quick Start"
    New to PanelBox? Start with [Fundamentals](fundamentals.md) to learn the basics,
    then explore the specific technique you need.

## Learning Paths

Choose a path based on your goals and experience level:

| Path | Level | Duration | Topics | Notebooks |
|------|-------|----------|--------|-----------|
| **Essentials** | Beginner | 4--6 hours | Fundamentals, Static Models, Standard Errors | ~18 |
| **Applied Researcher** | Intermediate | 12--16 hours | + GMM, Discrete Choice, Diagnostics, Visualization | ~50 |
| **Econometrician** | Advanced | 30--40 hours | + Spatial, Quantile, VAR, Frontier, all models | 193 |

### :material-school: Essentials Path (4--6 hours)

For researchers new to panel data or PanelBox. Covers the foundations you need for any applied work.

1. [Fundamentals](fundamentals.md) -- Panel data concepts, within/between variation (4 notebooks)
2. [Static Models](static-models.md) -- Pooled OLS, Fixed Effects, Random Effects (7 notebooks)
3. [Standard Errors](standard-errors.md) -- Robust inference basics (notebooks 01--02)

### :material-flask: Applied Researcher Path (12--16 hours)

For researchers ready to apply PanelBox to real-world problems with proper diagnostics.

1. Complete the **Essentials** path first
2. [GMM](gmm.md) -- Dynamic panel models (notebooks 01--04)
3. [Discrete Choice](discrete.md) -- Binary, ordered, multinomial (notebooks 01--04)
4. [Validation & Diagnostics](validation.md) -- Testing assumptions (4 notebooks)
5. [Visualization & Reports](visualization.md) -- Charts and HTML reports (4 notebooks)

### :material-trophy: Econometrician Path (30--40 hours)

The complete PanelBox curriculum. Master every model family and technique.

1. Complete the **Applied Researcher** path first
2. [Spatial Econometrics](spatial.md) -- SAR, SEM, SDM (8 notebooks)
3. [Quantile Regression](quantile.md) -- Beyond-the-mean analysis (10 notebooks)
4. [Panel VAR](var.md) -- Vector autoregressions (7 notebooks)
5. [Stochastic Frontier](frontier.md) -- Efficiency analysis (6 notebooks)
6. [Count Models](count.md) -- Poisson, PPML, zero-inflated (7 notebooks)
7. [Censored & Selection](censored.md) -- Tobit, Heckman (8 notebooks)
8. [Marginal Effects](marginal-effects.md) -- Interpretation of nonlinear models (6 notebooks)

## Tutorial Categories

<div class="grid cards" markdown>

-   :material-school: **Fundamentals**

    ---

    Panel data basics, formulas, estimation & interpretation

    **4 notebooks** | Beginner -- Intermediate

    [:octicons-arrow-right-24: Fundamentals](fundamentals.md)

-   :material-chart-line: **Static Models**

    ---

    Pooled OLS, Fixed Effects, Random Effects, IV estimation

    **7 notebooks** | Beginner -- Advanced

    [:octicons-arrow-right-24: Static Models](static-models.md)

-   :material-arrow-decision: **Dynamic GMM**

    ---

    Arellano-Bond, Blundell-Bond, CUE, bias correction

    **6 notebooks** | Intermediate -- Advanced

    [:octicons-arrow-right-24: GMM](gmm.md)

-   :material-map-marker-radius: **Spatial Econometrics**

    ---

    SAR, SEM, SDM, dynamic spatial panels, marginal effects

    **8 notebooks** | Intermediate -- Advanced

    [:octicons-arrow-right-24: Spatial](spatial.md)

-   :material-factory: **Stochastic Frontier**

    ---

    SFA, four-component model, TFP decomposition

    **6 notebooks** | Intermediate -- Advanced

    [:octicons-arrow-right-24: Frontier](frontier.md)

-   :material-chart-scatter-plot: **Quantile Regression**

    ---

    Panel quantile methods, Canay, location-scale, QTE

    **10 notebooks** | Intermediate -- Advanced

    [:octicons-arrow-right-24: Quantile](quantile.md)

-   :material-vector-polyline: **Panel VAR**

    ---

    VAR, VECM, IRF, FEVD, Granger causality

    **7 notebooks** | Intermediate -- Advanced

    [:octicons-arrow-right-24: VAR](var.md)

-   :material-toggle-switch: **Discrete Choice**

    ---

    Logit, probit, ordered, multinomial, dynamic models

    **9 notebooks** | Beginner -- Advanced

    [:octicons-arrow-right-24: Discrete](discrete.md)

-   :material-counter: **Count Data**

    ---

    Poisson, negative binomial, PPML, zero-inflated models

    **7 notebooks** | Beginner -- Advanced

    [:octicons-arrow-right-24: Count](count.md)

-   :material-content-cut: **Censored & Selection**

    ---

    Tobit, Honore estimator, Heckman selection models

    **8 notebooks** | Beginner -- Advanced

    [:octicons-arrow-right-24: Censored](censored.md)

-   :material-shield-check: **Standard Errors**

    ---

    Robust, clustered, HAC, Driscoll-Kraay, bootstrap

    **7 notebooks** | Beginner -- Advanced

    [:octicons-arrow-right-24: Standard Errors](standard-errors.md)

-   :material-delta: **Marginal Effects**

    ---

    AME, MEM for discrete, count, and censored models

    **6 notebooks** | Beginner -- Advanced

    [:octicons-arrow-right-24: Marginal Effects](marginal-effects.md)

-   :material-check-decagram: **Validation & Diagnostics**

    ---

    Assumption tests, unit roots, cointegration, specification

    **8 notebooks** | Intermediate -- Advanced

    [:octicons-arrow-right-24: Validation](validation.md)

-   :material-palette: **Visualization**

    ---

    Interactive charts, visual diagnostics, automated reports

    **4 notebooks** | Beginner -- Intermediate

    [:octicons-arrow-right-24: Visualization](visualization.md)

-   :material-rocket-launch: **Production**

    ---

    Prediction, model persistence, pipelines, versioning

    **6 notebooks** | Intermediate -- Advanced

    [:octicons-arrow-right-24: Production](production.md)

</div>

## How to Use These Tutorials

=== "Google Colab (Recommended)"

    Click the **Open in Colab** badge on any tutorial to launch it directly in Google Colab.
    PanelBox will be installed automatically in the first cell.

    ```python
    !pip install panelbox
    ```

    No local setup required -- just a Google account and a web browser.

=== "Local Installation"

    Clone the repository and run notebooks locally:

    ```bash
    git clone https://github.com/PanelBox-Econometrics-Model/panelbox.git
    cd panelbox
    pip install -e .
    jupyter lab examples/
    ```

=== "Download Individual Notebooks"

    Browse notebooks on GitHub and download what you need:

    ```text
    https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/examples/
    ```

## Difficulty Levels

| Level | Description | Prerequisites |
|-------|-------------|---------------|
| **Beginner** | No prior panel data experience required. Covers fundamentals and basic models. | Python, pandas, basic statistics |
| **Intermediate** | Assumes basic panel data knowledge. Introduces advanced techniques and diagnostics. | Fundamentals tutorials completed |
| **Advanced** | For experienced users. Complex models, custom workflows, and production use cases. | Multiple tutorial categories completed |

## Interactive Notebook Tutorials

In addition to the example notebooks, PanelBox includes self-contained tutorial notebooks covering specific topics:

| Tutorial | Topic | Level |
|----------|-------|-------|
| [Panel Quantile Regression](intro_panel_quantile_regression.ipynb) | Introduction to panel quantile methods | Intermediate |
| [Panel Cointegration](panel_cointegration.ipynb) | Cointegration testing for panel data | Advanced |
| [Multinomial Logit](multinomial_tutorial.ipynb) | Multinomial choice modeling | Intermediate |
| [Panel Unit Root](panel_unit_root.ipynb) | Unit root testing for panels | Intermediate |
| [Stochastic Frontier](sfa_tutorial.ipynb) | SFA fundamentals and estimation | Intermediate |
| [J-Test Specification](jtest_tutorial.ipynb) | J-test for non-nested models | Advanced |
| [PPML Gravity](ppml_gravity.ipynb) | Gravity models with PPML | Intermediate |
| [Spatial Econometrics](spatial_econometrics_complete.ipynb) | Complete spatial analysis | Advanced |

## Solutions & Answer Keys

Most tutorial categories include complete solution notebooks. These provide:

- Full code implementations for all exercises
- Detailed interpretation of results
- Best practices and common pitfalls

Look for the **Solutions** section on each category page.

## What's Next?

Start with [Fundamentals](fundamentals.md) if you are new to panel data, or jump directly to the category that matches your research needs. Each tutorial page includes recommended learning paths and prerequisites.

## See Also

- [Getting Started](../getting-started/index.md) -- Installation and first steps
- [User Guide](../user-guide/index.md) -- Comprehensive reference for all model families
- [Visualization](../visualization/charts/index.md) -- Chart gallery and customization
- [FAQ](../faq/general.md) -- Common questions and troubleshooting
