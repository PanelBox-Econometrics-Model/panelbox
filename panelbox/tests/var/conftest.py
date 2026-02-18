# tests/var/conftest.py

import json
from pathlib import Path

import pytest


@pytest.fixture
def r_benchmarks():
    """Carrega resultados de referência do R."""
    benchmark_path = Path(__file__).parent / "r_validation" / "r_benchmark_results.json"

    if not benchmark_path.exists():
        pytest.skip("R benchmark results not found. Run generate_r_benchmarks.R first.")

    with open(benchmark_path) as f:
        return json.load(f)


@pytest.fixture
def panel_data():
    """Gera dados de teste com mesmo seed do R."""
    from panelbox.tests.var.fixtures.var_test_data import generate_panel_var_data

    return generate_panel_var_data(n_entities=50, n_periods=20, seed=42)


@pytest.fixture
def true_params():
    """Retorna os parâmetros verdadeiros do DGP."""
    from panelbox.tests.var.fixtures.var_test_data import TRUE_PARAMS

    return TRUE_PARAMS
