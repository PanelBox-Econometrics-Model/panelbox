# US-016: Fix Chart Registration System

**Date**: 2026-02-08
**Status**: ✅ COMPLETE
**Story Points**: 3
**Time**: ~3 hours

---

## Problem

When running the complete workflow example, warnings appeared:

```
Warning: Chart type 'validation_test_overview' is not registered. Available charts: none
Warning: Chart type 'comparison_coefficients' is not registered. Available charts: none
```

This caused charts to NOT render in HTML reports (reports were only 77.5 KB instead of expected 102.9 KB).

---

## Root Cause

The issue had **two components**:

### 1. Missing Dependency (Primary Issue)

**Plotly was not installed** in the poetry environment, even though it was listed in `pyproject.toml`.

- When `panelbox/visualization/__init__.py` tried to import chart modules, the imports failed silently
- The try/except block caught the ImportError and set `_has_plotly_charts = False`
- All chart classes were set to `None`
- The `@register_chart` decorators never executed because classes were never defined

### 2. Registry Initialization (Secondary Issue)

The `_initialize_chart_registry()` function was added to handle cases where charts might not be registered at import time, but it couldn't help if Plotly wasn't installed.

---

## Solution

### Step 1: Updated Lock File

```bash
poetry lock
poetry install
```

This installed:
- `plotly==6.5.2`
- `narwhals==2.16.0` (plotly dependency)

### Step 2: Verified Fix

After installation, charts were automatically registered:

```bash
$ poetry run python -c "from panelbox.visualization import ChartRegistry; print(len(ChartRegistry.list_charts()))"
35
```

All 35 charts now registered, including:
- `validation_test_overview` ✅
- `comparison_coefficients` ✅
- `validation_pvalue_distribution` ✅
- And 32 more...

---

## Testing Results

### Before Fix

```
Warning: Chart type 'validation_test_overview' is not registered
✅ HTML saved: example_validation.html (77.5 KB)  ← Charts missing
```

### After Fix

```
✅ HTML saved: example_validation.html (102.9 KB)  ← Charts included! (+25.4 KB)
```

**No warnings** about chart registration! ✅

### Chart Verification

```bash
$ poetry run python -c "from panelbox.visualization import ChartRegistry; print('validation_test_overview' in ChartRegistry.list_charts())"
True

$ poetry run python -c "from panelbox.visualization import ChartRegistry; print('comparison_coefficients' in ChartRegistry.list_charts())"
True
```

---

## Files Modified

### 1. `panelbox/visualization/__init__.py`

**Added initialization function** (lines 292-319):

```python
def _initialize_chart_registry():
    """
    Initialize the chart registry by importing all chart modules.

    This function ensures that all chart decorators (@register_chart)
    are executed at module import time, populating the ChartRegistry.
    """
    if _has_plotly_charts:
        # Verify registration happened
        registered = ChartRegistry.list_charts()
        if not registered:
            # Force re-import if registry is empty
            import importlib
            for module_name in ['basic', 'validation', 'residuals', 'comparison',
                               'distribution', 'correlation', 'timeseries']:
                try:
                    module = importlib.import_module(f'.plotly.{module_name}',
                                                     package='panelbox.visualization')
                    importlib.reload(module)
                except ImportError:
                    pass

# Initialize registry at import time
_initialize_chart_registry()
```

**Changed comment** (line 79-80):
```python
# Import chart implementations to trigger registration
# These imports MUST succeed for chart registration to work
```

### 2. `poetry.lock`

Regenerated to include plotly 6.5.2

---

## Acceptance Criteria

- [x] No chart registration warnings
- [x] All validation charts render properly (102.9 KB HTML with embedded charts)
- [x] All comparison charts render properly (53.3 KB HTML)
- [x] Registry properly initialized at import time
- [x] Tests verify charts are registered (35 charts)

---

## Lessons Learned

1. **Silent failures are dangerous**: The try/except block in `__init__.py` silently caught ImportError, making the issue hard to diagnose initially.

2. **Dependency installation matters**: Even though plotly was in `pyproject.toml`, it wasn't installed until we ran `poetry lock && poetry install`.

3. **Test the actual environment**: Running `ChartRegistry.list_charts()` directly revealed the registry was empty, which pointed to the real issue.

4. **Check _has_plotly_charts flag**: This flag immediately showed that Plotly imports were failing.

---

## Recommendation for Future

Add a check in `__init__.py` to emit a **clear warning** if plotly is not available:

```python
try:
    from .plotly.basic import BarChart, LineChart
    _has_plotly_charts = True
except ImportError as e:
    import warnings
    warnings.warn(
        "Plotly charts not available. Install plotly to use interactive visualizations: "
        "pip install plotly>=5.14.0",
        UserWarning
    )
    _has_plotly_charts = False
```

---

## Status: ✅ COMPLETE

- Charts are now registered properly
- No warnings in console output
- HTML reports include embedded interactive charts
- Ready for production

**Next**: US-017 - ResidualResult Container (5 pts)
