"""Verify notebooks exist and have valid structure."""

import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
SOLUTIONS_DIR = Path(__file__).resolve().parent.parent / "solutions"

EXPECTED_NOTEBOOKS = [
    "01_predict_fundamentals.ipynb",
    "02_save_load_models.ipynb",
    "03_production_pipeline.ipynb",
    "04_model_validation.ipynb",
    "05_model_versioning.ipynb",
    "06_case_study_bank_lgd.ipynb",
]

EXPECTED_SOLUTIONS = [
    "01_predict_fundamentals_solutions.ipynb",
    "02_save_load_solutions.ipynb",
    "03_production_pipeline_solutions.ipynb",
    "04_model_validation_solutions.ipynb",
    "05_model_versioning_solutions.ipynb",
    "06_case_study_bank_lgd_solutions.ipynb",
]


class TestNotebookStructure:
    """Verify all expected notebooks exist."""

    @pytest.mark.parametrize("notebook", EXPECTED_NOTEBOOKS)
    def test_notebook_exists(self, notebook):
        path = NOTEBOOKS_DIR / notebook
        assert path.exists(), f"Missing notebook: {notebook}"

    @pytest.mark.parametrize("solution", EXPECTED_SOLUTIONS)
    def test_solution_exists(self, solution):
        path = SOLUTIONS_DIR / solution
        assert path.exists(), f"Missing solution: {solution}"


class TestNotebookValidation:
    """Verify notebooks are valid JSON and have cells."""

    @pytest.mark.parametrize("notebook", EXPECTED_NOTEBOOKS)
    def test_notebook_valid_json(self, notebook):
        path = NOTEBOOKS_DIR / notebook
        if not path.exists():
            pytest.skip(f"{notebook} not yet created")
        with open(path) as f:
            data = json.load(f)
        assert "cells" in data
        assert "metadata" in data
        assert len(data["cells"]) > 0

    @pytest.mark.parametrize("solution", EXPECTED_SOLUTIONS)
    def test_solution_valid_json(self, solution):
        path = SOLUTIONS_DIR / solution
        if not path.exists():
            pytest.skip(f"{solution} not yet created")
        with open(path) as f:
            data = json.load(f)
        assert "cells" in data
        assert len(data["cells"]) > 0
