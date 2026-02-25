---
title: "Contributing Guide"
description: "How to contribute to the PanelBox panel data econometrics library — setup, code standards, templates, and PR process."
---

# Contributing to PanelBox

Thank you for your interest in contributing to PanelBox! Whether you are reporting a bug, proposing a feature, improving documentation, or submitting code, your help is welcome and appreciated.

## Types of Contributions

| Type | Where | Description |
|------|-------|-------------|
| Bug reports | [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues) | Reproducible problem with expected vs. actual behavior |
| Feature requests | [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues) | Proposals with `[Feature]` label |
| Code (PR) | [Pull Requests](https://github.com/PanelBox-Econometrics-Model/panelbox/pulls) | New estimators, tests, bug fixes |
| Documentation | `docs/` directory | Tutorials, API docs, examples |
| Test additions | `tests/` directory | Unit, integration, and validation tests |

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/PanelBox-Econometrics-Model/panelbox.git
cd panelbox
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
```

### 3. Install in Development Mode

```bash
pip install -e ".[dev,all]"
```

### 4. Install Pre-Commit Hooks

```bash
pre-commit install
```

### 5. Verify Setup

```bash
pytest tests/ -v --timeout=60
```

### Development Dependencies

| Tool | Version | Purpose |
|------|---------|---------|
| pytest | >= 8.0 | Testing (with xdist, randomly, timeout) |
| pytest-cov | >= 5.0 | Coverage measurement |
| ruff | >= 0.15.0 | Linting and formatting |
| pyright | >= 1.1.400 | Static type checking |
| interrogate | >= 1.7.0 | Docstring coverage |
| vulture | >= 2.14 | Dead code detection |

Pre-commit hooks run **black**, **isort**, and **end-of-file-fixer** automatically on every commit.

## Code Standards

### Style

- **Formatter**: black (line length 88)
- **Import sorting**: isort (black-compatible profile)
- **Type hints**: Required for all public API functions and methods
- **Docstrings**: NumPy-style for all public classes and methods
- **Python version**: 3.9+ compatibility

```bash
# Format and lint
ruff format panelbox/
ruff check panelbox/ --fix

# Type check
pyright panelbox/

# Run all hooks manually
pre-commit run --all-files
```

### Branch Naming

Use descriptive branch names:

- `feature/add-spatial-durbin` — New features
- `fix/gmm-weight-matrix` — Bug fixes
- `docs/update-tutorial` — Documentation changes
- `test/add-poisson-tests` — Test additions

### Commit Messages

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `style`, `chore`

**Example**:

```text
feat(quantile): Add Canay two-step estimator

Implements the Canay (2011) two-step estimator for panel
quantile regression with fixed effects.

Closes #45
```

## Adding a New Estimator

Place your code in the appropriate model family directory. Every estimator must:

1. Inherit from `PanelModel` (or a family-specific base class)
2. Implement `fit()` returning a results object
3. Have comprehensive tests
4. Be exported from the package `__init__.py`
5. Have a documentation page

### Estimator Template

```python
"""My estimator module.

Implements the Author (Year) estimator for panel data.
"""

import numpy as np
import pandas as pd

from panelbox.models.base import PanelModel
from panelbox.core.results import PanelResults


class MyEstimator(PanelModel):
    """Short description of the estimator.

    Longer description explaining the model, its assumptions,
    and when to use it.

    Parameters
    ----------
    endog : array_like
        Dependent variable.
    exog : array_like or DataFrame
        Independent variables.
    entity_id : array_like
        Entity identifiers.
    time_id : array_like
        Time identifiers.

    References
    ----------
    .. [1] Author, A. (Year). Title. *Journal*, vol(issue), pages.
    """

    def __init__(self, endog, exog, entity_id, time_id, **kwargs):
        super().__init__(endog, exog, entity_id, time_id)
        # Estimator-specific initialization

    def fit(self, cov_type: str = "nonrobust", **kwargs) -> PanelResults:
        """Estimate the model.

        Parameters
        ----------
        cov_type : str, default='nonrobust'
            Covariance estimator type.

        Returns
        -------
        PanelResults
            Fitted model results with params, std_errors, pvalues,
            conf_int(), summary().
        """
        # 1. Compute estimates
        # 2. Compute covariance matrix
        # 3. Return results
        return PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=resid,
            fittedvalues=fitted,
            model_info={
                "model_type": "My Estimator",
                "formula": "...",
                "cov_type": cov_type,
            },
            data_info={
                "nobs": len(endog),
                "n_entities": n_entities,
                "n_periods": n_periods,
                "df_model": k,
                "df_resid": n - k,
            },
        )

    def predict(self, params=None, exog=None) -> np.ndarray:
        """Generate predictions."""
        if params is None:
            params = self.results.params
        if exog is None:
            exog = self.exog
        return exog @ params
```

### Where to Place the Code

| Model Family | Directory | Base Class |
|---|---|---|
| Static (OLS, FE, RE) | `panelbox/models/static/` | `PanelModel` |
| GMM | `panelbox/gmm/` | `GMMEstimator` |
| Spatial | `panelbox/models/spatial/` | `SpatialPanelModel` |
| Discrete choice | `panelbox/models/discrete/` | `NonlinearPanelModel` |
| Count data | `panelbox/models/count/` | `NonlinearPanelModel` |
| Censored / Selection | `panelbox/models/censored/` or `selection/` | `PanelModel` |
| Quantile | `panelbox/models/quantile/` | `QuantileModel` |
| Frontier (SFA) | `panelbox/frontier/` | — |
| Panel VAR | `panelbox/var/` | — |

## Adding a New Diagnostic Test

Diagnostic tests inherit from `ValidationTest` and return a `ValidationTestResult`.

### Diagnostic Test Template

```python
"""My diagnostic test.

Implements the Author (Year) test for panel data.
"""

from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult
from panelbox.core.results import PanelResults


class MyDiagnosticTest(ValidationTest):
    """Test for some property in panel data.

    Parameters
    ----------
    results : PanelResults
        Fitted model results.

    References
    ----------
    .. [1] Author, A. (Year). Title. *Journal*, vol(issue), pages.
    """

    def __init__(self, results: PanelResults):
        super().__init__(results)

    def run(self, alpha: float = 0.05, **kwargs) -> ValidationTestResult:
        """Run the diagnostic test.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        ValidationTestResult
            Result with test_name, statistic, pvalue,
            null_hypothesis, reject_null, conclusion.
        """
        # 1. Compute test statistic from self.resid, self.fittedvalues, etc.
        test_stat = ...
        pvalue = 1 - stats.chi2.cdf(test_stat, df=k)

        return ValidationTestResult(
            test_name="My Diagnostic Test",
            statistic=test_stat,
            pvalue=pvalue,
            null_hypothesis="No effect present",
            alternative_hypothesis="Effect is present",
            alpha=alpha,
            df=k,
            metadata={"n_entities": self.n_entities, "detail": "..."},
        )
```

**Key attributes available via `self`** (inherited from `ValidationTest`):

| Attribute | Type | Description |
|---|---|---|
| `self.results` | `PanelResults` | Full fitted results object |
| `self.resid` | `np.ndarray` | Residuals |
| `self.fittedvalues` | `np.ndarray` | Fitted values |
| `self.params` | `pd.Series` | Estimated coefficients |
| `self.nobs` | `int` | Number of observations |
| `self.n_entities` | `int` | Number of entities |
| `self.n_periods` | `int` | Number of time periods |
| `self.model_type` | `str` | Model type string |

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/gmm/ -v
pytest tests/models/static/ -v

# Specific test
pytest tests/gmm/test_diagnostics.py::test_hansen_j -v

# With coverage
pytest tests/ --cov=panelbox --cov-report=html --cov-branch

# In parallel
pytest tests/ -n auto
```

### Writing Tests

```python
import pytest
import numpy as np
from panelbox.models.static import FixedEffects
from panelbox.datasets import load_grunfeld


class TestMyEstimator:
    """Tests for MyEstimator."""

    @pytest.fixture
    def panel_data(self):
        """Load standard test dataset."""
        return load_grunfeld()

    def test_basic_estimation(self, panel_data):
        """Test that estimation runs and returns results."""
        model = FixedEffects(
            data=panel_data, formula="invest ~ value + capital"
        )
        result = model.fit()
        assert result.params is not None
        assert len(result.params) == 2

    def test_coefficients_reasonable(self, panel_data):
        """Test coefficient signs and magnitudes."""
        model = FixedEffects(
            data=panel_data, formula="invest ~ value + capital"
        )
        result = model.fit()
        assert result.params["value"] > 0

    def test_robust_standard_errors(self, panel_data):
        """Test robust SE computation."""
        model = FixedEffects(
            data=panel_data, formula="invest ~ value + capital"
        )
        result = model.fit(cov_type="robust")
        assert all(result.std_errors > 0)
```

### Validation Against R or Stata

For estimators with R or Stata equivalents, add validation tests comparing results:

```python
def test_against_r_reference(self):
    """Validate against R plm package."""
    # R reference values (from validated R script)
    r_coefficients = {"value": 0.1101, "capital": 0.3101}

    model = FixedEffects(data=data, formula="invest ~ value + capital")
    result = model.fit()

    for var, r_val in r_coefficients.items():
        np.testing.assert_allclose(
            result.params[var], r_val, rtol=1e-3,
            err_msg=f"{var} doesn't match R"
        )
```

## Building Documentation

PanelBox uses MkDocs with Material theme:

```bash
# Local preview with auto-reload
mkdocs serve

# Build static site
mkdocs build
```

Documentation source lives in `docs/`. API reference, tutorials, and guides are written in Markdown with MkDocs Material admonitions and tabs.

## Pull Request Process

### Step-by-Step

1. **Create a feature branch**:
    ```bash
    git checkout -b feature/my-new-feature
    ```

2. **Make changes**: code, tests, documentation.

3. **Run checks locally**:
    ```bash
    pytest tests/ -v
    pre-commit run --all-files
    ```

4. **Commit with a clear message**:
    ```bash
    git commit -m "feat(models): Add MyEstimator for panel data

    - Implements Author (Year) estimator
    - Adds validation against R
    - Includes 15 unit tests

    Closes #123"
    ```

5. **Push and open a PR**:
    ```bash
    git push origin feature/my-new-feature
    ```

6. **Fill out the PR template and address review comments.**

### PR Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] New code has tests
- [ ] Public API has docstrings (NumPy style)
- [ ] Documentation updated (if applicable)
- [ ] Exports added to `__init__.py` (if applicable)

## Reporting Issues

File issues on [GitHub](https://github.com/PanelBox-Econometrics-Model/panelbox/issues) with:

1. A clear title describing the problem
2. **Minimal reproducible example** (MRE)
3. Expected vs. actual behavior
4. PanelBox version: `pip show panelbox`
5. Python version: `python --version`

## Recognition

Contributors are recognized in:

- The [Changelog](changelog.md)
- Release notes
- The AUTHORS file

Significant contributions may result in co-authorship on methodological papers.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/PanelBox-Econometrics-Model/panelbox/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues)
- **Feature requests**: [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues) with `[Feature]` label
- **Security issues**: Email security@panelbox.org

## See Also

- [Code of Conduct](code-of-conduct.md) — Community standards
- [Changelog](changelog.md) — Version history
- [Roadmap](roadmap.md) — Planned features
- [API Reference](../api/index.md) — Full API documentation
