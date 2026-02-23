"""
Test that tutorial notebooks execute without errors.

Uses nbconvert to execute each notebook and checks for exceptions.
These tests are slow (marked with @pytest.mark.slow) and are typically
run in CI or with `pytest -m slow`.
"""

from pathlib import Path

import pytest

BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# List of tutorial notebooks
NOTEBOOKS = [
    "01_unit_root_tests.ipynb",
    "02_cointegration_tests.ipynb",
    "03_specification_tests.ipynb",
    "04_spatial_tests.ipynb",
]


def _execute_notebook(notebook_path: Path) -> bool:
    """Execute a notebook and return True if successful."""
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        with open(notebook_path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
        return True
    except Exception as e:
        pytest.fail(f"Notebook {notebook_path.name} failed: {e}")
        return False


@pytest.mark.slow
class TestNotebooksRun:
    """Test that notebooks execute without errors."""

    @pytest.mark.parametrize("notebook", NOTEBOOKS)
    def test_notebook_exists(self, notebook):
        path = NOTEBOOKS_DIR / notebook
        assert path.exists(), f"Notebook not found: {path}"

    @pytest.mark.parametrize("notebook", NOTEBOOKS)
    def test_notebook_runs(self, notebook):
        path = NOTEBOOKS_DIR / notebook
        if not path.exists():
            pytest.skip(f"Notebook not found: {path}")
        _execute_notebook(path)
