# Contributing to PanelBox

Thank you for your interest in contributing to PanelBox! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Be respectful, inclusive, and professional in all interactions.

## Getting Started

### Development Environment

1. **Fork and clone the repository:**
```bash
git clone https://github.com/yourusername/panelbox.git
cd panelbox
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode:**
```bash
pip install -e ".[dev,test,quantile]"
```

4. **Install pre-commit hooks:**
```bash
pre-commit install
```

### Development Dependencies

```bash
pip install -r requirements-dev.txt
```

Required for development:
- pytest >= 7.0
- pytest-cov >= 3.0
- black >= 22.0 (code formatting)
- flake8 >= 4.0 (linting)
- mypy >= 0.950 (type checking)
- sphinx >= 4.0 (documentation)

## Contributing to Quantile Regression

### Understanding the Module Structure

```
panelbox/models/quantile/
├── __init__.py
├── pooled.py           # Pooled QR
├── fixed_effects.py    # FE QR (Koenker 2004)
├── canay.py           # Canay two-step
├── location_scale.py  # Location-scale models
├── treatment.py       # QTE estimation
└── utils/
    ├── optimization.py
    └── inference.py
```

### Adding a New Estimator

1. **Create a new file** in `panelbox/models/quantile/`
2. **Inherit from base class:**

```python
from panelbox.models.quantile.base import QuantileModel

class MyNewEstimator(QuantileModel):
    """
    Brief description.

    Parameters
    ----------
    data : PanelData
        Panel dataset
    formula : str
        Model formula (Wilkinson notation)
    tau : float or list
        Quantile(s) to estimate

    References
    ----------
    .. [1] Author (Year). Title. Journal.
    """

    def __init__(self, data, formula, tau, **kwargs):
        super().__init__(data, formula, tau)
        # Your initialization

    def fit(self, **kwargs):
        """Estimate the model."""
        # Your implementation
        return QuantileResults(...)
```

3. **Write comprehensive docstrings** (NumPy style)
4. **Add unit tests** in `tests/quantile/test_mynew.py`
5. **Validate against R** if comparable implementation exists
6. **Update documentation** in `docs/source/quantile/`

### Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Quotes**: Double quotes for strings
- **Imports**: Organized with isort
- **Type hints**: Use for all public APIs

**Format your code:**
```bash
black panelbox/
isort panelbox/
```

**Check linting:**
```bash
flake8 panelbox/
mypy panelbox/
```

### Testing

#### Writing Tests

All new code requires tests. Place tests in `tests/quantile/`:

```python
import pytest
import numpy as np
from panelbox.models.quantile import MyNewEstimator

class TestMyNewEstimator:
    """Tests for MyNewEstimator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        # Return test data

    def test_basic_functionality(self, sample_data):
        """Test basic estimation works."""
        model = MyNewEstimator(sample_data, 'y ~ x1 + x2', tau=0.5)
        result = model.fit()

        assert result.params is not None
        assert len(result.params) == 3

    def test_coefficients_reasonable(self, sample_data):
        """Test coefficient values are reasonable."""
        # Test against known DGP

    def test_standard_errors(self, sample_data):
        """Test standard error computation."""
        # Test SE calculation
```

#### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/quantile/test_mynew.py

# Run with coverage
pytest --cov=panelbox --cov-report=html

# Run only quantile tests
pytest tests/quantile/
```

#### Validation Against R

For estimators with R equivalents, add validation tests:

```python
def test_against_r(self):
    """Validate against R quantreg package."""
    # Load R reference output
    with open('tests/validation/quantile/reference_outputs/mynew.json') as f:
        r_result = json.load(f)

    # Run PanelBox
    model = MyNewEstimator(...)
    result = model.fit()

    # Compare
    np.testing.assert_allclose(
        result.params,
        r_result['coefficients'],
        rtol=1e-4,
        atol=1e-5,
        err_msg="Coefficients don't match R"
    )
```

### Documentation

#### Docstring Format

Use NumPy docstring style:

```python
def my_function(x, y, method='default'):
    """
    Brief one-line description.

    More detailed description with multiple paragraphs if needed.
    Explain what the function does, its purpose, and any important
    algorithmic details.

    Parameters
    ----------
    x : array_like
        Description of x
    y : array_like
        Description of y
    method : {'default', 'alternative'}, optional
        Description of method, by default 'default'

    Returns
    -------
    result : ndarray
        Description of result

    Raises
    ------
    ValueError
        If x and y have different lengths

    See Also
    --------
    related_function : Related functionality

    Notes
    -----
    Any important notes about usage, limitations, or algorithmic details.

    Mathematical notation can be included using LaTeX:

    .. math:: Q_Y(\tau|X) = X'\beta(\tau)

    References
    ----------
    .. [1] Koenker, R. (2004). Quantile regression for longitudinal data.
           Journal of Multivariate Analysis, 91(1), 74-89.

    Examples
    --------
    >>> from panelbox import PanelData
    >>> data = PanelData(...)
    >>> result = my_function(data.y, data.X)
    >>> print(result)
    """
    # Implementation
```

#### Building Documentation

```bash
cd docs/
make html
# Open build/html/index.html
```

### Pull Request Process

1. **Create a feature branch:**
```bash
git checkout -b feature/my-new-feature
```

2. **Make your changes:**
- Write code
- Add tests
- Update documentation
- Run tests locally

3. **Commit with clear messages:**
```bash
git add .
git commit -m "Add MyNewEstimator for panel QR

- Implements estimator from Author (Year)
- Adds validation against R
- Includes comprehensive tests
- Updates documentation

Closes #123"
```

4. **Push and create PR:**
```bash
git push origin feature/my-new-feature
```

5. **Fill out PR template completely**

6. **Address review comments**

### Commit Message Guidelines

Format:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `style`: Code style changes (formatting)
- `chore`: Maintenance tasks

Example:
```
feat: Add Canay two-step estimator

Implements the Canay (2011) two-step estimator for panel
quantile regression with fixed effects. The estimator:
1. Estimates fixed effects via within-transformation OLS
2. Removes fixed effects from dependent variable
3. Runs pooled QR on transformed data

Performance: O(NT) for Step 1, O(NT) for Step 2

Closes #45
```

## Specific Contribution Areas

### 1. New Estimators

We welcome implementations of new panel QR estimators from recent literature.

**Priority methods:**
- Instrumental variables QR (IVQR)
- Quantile regression with censored data
- Composite quantile regression
- Smoothed QR methods

**Requirements:**
- Published in peer-reviewed journal
- Validation against existing implementation (if available)
- Comprehensive tests
- Clear documentation with references

### 2. Performance Improvements

Optimizations are always welcome:

- Algorithmic improvements
- Parallelization
- Cython/Numba acceleration
- Memory optimization

**Requirements:**
- Benchmark showing improvement
- No accuracy loss
- Tests pass

### 3. Bug Fixes

Found a bug? Great!

1. Search existing issues
2. Create issue with reproducible example
3. Fix and submit PR
4. Link PR to issue

### 4. Documentation

Documentation improvements highly valued:

- Fix typos
- Clarify explanations
- Add examples
- Improve tutorials

### 5. Examples and Tutorials

Real-world examples are valuable:

- Applied economics examples
- Finance applications
- Policy evaluation
- Climate science

**Requirements:**
- Complete, runnable code
- Real or realistic data
- Clear interpretation
- References to relevant literature

## R Package Comparison

When adding features, check if R quantreg/rqpd has equivalent:

1. **Run R implementation** on test data
2. **Save output** to `tests/validation/quantile/reference_outputs/`
3. **Add validation test** comparing PanelBox to R
4. **Document differences** if any (and justify)

R script template:
```r
library(quantreg)
library(jsonlite)

# Generate data
set.seed(42)
data <- data.frame(...)

# Run estimator
result <- rq(y ~ x1 + x2, data=data, tau=0.5)

# Save output
output <- list(
  coefficients = coef(result),
  std_errors = summary(result, se="boot")$coefficients[,2]
)

write_json(output, "reference_output.json", digits=10)
```

## Questions?

- **General questions**: Open a Discussion on GitHub
- **Bug reports**: Open an Issue
- **Feature requests**: Open an Issue with [Feature] tag
- **Security issues**: Email security@panelbox.org

## Recognition

Contributors are recognized in:
- AUTHORS file
- Release notes
- Documentation contributors page

Significant contributions may result in co-authorship on methodological papers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to PanelBox!
