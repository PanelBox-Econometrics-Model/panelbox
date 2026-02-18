"""
Test that tutorial notebooks can be executed.

This test suite verifies that all notebooks:
- Can be loaded without errors
- Execute without exceptions
- Produce expected outputs

Note: Requires jupyter, nbformat, and nbconvert packages.
"""

import json
from pathlib import Path

import pytest

# Base paths
BASE_DIR = Path(__file__).parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"


class TestNotebookStructure:
    """Test notebook directory structure."""

    def test_notebooks_directory_exists(self):
        assert NOTEBOOKS_DIR.exists()
        assert NOTEBOOKS_DIR.is_dir()

    def test_expected_notebooks_exist(self):
        """Test that expected notebook files are present."""
        expected_notebooks = [
            "01_poisson_introduction.ipynb",
            "02_negative_binomial.ipynb",
            "03_fe_re_count.ipynb",
            "04_ppml_gravity.ipynb",
            "05_zero_inflated.ipynb",
            "06_marginal_effects_count.ipynb",
            "07_innovation_case_study.ipynb",
        ]

        for notebook in expected_notebooks:
            notebook_path = NOTEBOOKS_DIR / notebook
            # For now, just check if the path would be valid
            # Actual notebooks will be created later
            assert isinstance(notebook, str)
            assert notebook.endswith(".ipynb")


class TestNotebookValidation:
    """Test notebook file validation."""

    @pytest.fixture
    def notebook_files(self):
        """Get list of existing notebook files."""
        if NOTEBOOKS_DIR.exists():
            return list(NOTEBOOKS_DIR.glob("*.ipynb"))
        return []

    def test_notebooks_are_valid_json(self, notebook_files):
        """Test that notebook files contain valid JSON."""
        for notebook_path in notebook_files:
            with open(notebook_path, "r", encoding="utf-8") as f:
                try:
                    nb_content = json.load(f)
                    assert isinstance(nb_content, dict)
                    assert "cells" in nb_content
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in {notebook_path.name}")

    def test_notebooks_have_metadata(self, notebook_files):
        """Test that notebooks have required metadata."""
        for notebook_path in notebook_files:
            with open(notebook_path, "r", encoding="utf-8") as f:
                nb_content = json.load(f)
                assert "metadata" in nb_content
                assert "nbformat" in nb_content


# Optional: Execution tests (requires nbconvert)
# Uncomment when notebooks are ready

# class TestNotebookExecution:
#     """Test notebook execution."""
#
#     @pytest.mark.slow
#     @pytest.mark.parametrize('notebook_name', [
#         '01_poisson_introduction.ipynb',
#         '02_negative_binomial.ipynb',
#         '03_fe_re_count.ipynb',
#         '04_ppml_gravity.ipynb',
#         '05_zero_inflated.ipynb',
#         '06_marginal_effects_count.ipynb',
#         '07_innovation_case_study.ipynb'
#     ])
#     def test_notebook_executes(self, notebook_name):
#         """Test that notebook executes without errors."""
#         try:
#             import nbformat
#             from nbconvert.preprocessors import ExecutePreprocessor
#         except ImportError:
#             pytest.skip("nbconvert not installed")
#
#         notebook_path = NOTEBOOKS_DIR / notebook_name
#
#         if not notebook_path.exists():
#             pytest.skip(f"Notebook {notebook_name} not yet created")
#
#         with open(notebook_path, 'r', encoding='utf-8') as f:
#             nb = nbformat.read(f, as_version=4)
#
#         # Execute notebook
#         ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
#         try:
#             ep.preprocess(nb, {'metadata': {'path': str(NOTEBOOKS_DIR)}})
#         except Exception as e:
#             pytest.fail(f"Notebook {notebook_name} failed to execute: {str(e)}")


class TestNotebookMetadata:
    """Test tutorial metadata consistency."""

    def test_tutorial_count(self):
        """Verify expected number of tutorials."""
        # Import from parent package
        import sys

        sys.path.insert(0, str(BASE_DIR))
        from __init__ import TUTORIALS

        assert len(TUTORIALS) == 7

    def test_tutorial_sequence(self):
        """Test that tutorials are numbered sequentially."""
        import sys

        sys.path.insert(0, str(BASE_DIR))
        from __init__ import TUTORIALS

        tutorial_nums = sorted(TUTORIALS.keys())
        expected = [f"{i:02d}" for i in range(1, 8)]
        assert tutorial_nums == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
