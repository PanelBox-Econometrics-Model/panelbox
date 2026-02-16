"""
Testes de integração end-to-end para o módulo Stochastic Frontier Analysis.

Este módulo testa o workflow completo:
1. Carregar dados
2. Estimar múltiplos modelos SFA
3. Comparar modelos
4. Visualizar resultados
5. Gerar relatórios

Autor: PanelBox Development Team
Data: 2026-02-15
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier import (
    DistributionType,
    FrontierType,
    ModelType,
    StochasticFrontier,
    compare_nested_distributions,
    hausman_test_tfe_tre,
    inefficiency_presence_test,
    lr_test,
    skewness_test,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def panel_data():
    """
    Gerar dados de painel balanceado para testes de integração.

    Modelo verdadeiro: Pitt-Lee (time-invariant inefficiency)
    """
    np.random.seed(123)

    n_entities = 50
    n_time = 8
    n_obs = n_entities * n_time

    # IDs
    entity = np.repeat(np.arange(1, n_entities + 1), n_time)
    time = np.tile(np.arange(1, n_time + 1), n_entities)

    # Inputs (log)
    log_labor = np.random.uniform(2, 6, n_obs)
    log_capital = np.random.uniform(3, 7, n_obs)

    # Parâmetros verdadeiros
    beta_0 = 3.0
    beta_labor = 0.6
    beta_capital = 0.35
    sigma_v = 0.10
    sigma_u = 0.15

    # Fronteira de produção
    y_frontier = beta_0 + beta_labor * log_labor + beta_capital * log_capital

    # Ruído
    v = np.random.normal(0, sigma_v, n_obs)

    # Ineficiência time-invariant
    u_entity = np.abs(np.random.normal(0, sigma_u, n_entities))
    u = np.repeat(u_entity, n_time)

    # Output observado
    log_output = y_frontier + v - u

    # True efficiency
    te_true = np.exp(-u)

    df = pd.DataFrame(
        {
            "entity": entity,
            "time": time,
            "log_output": log_output,
            "log_labor": log_labor,
            "log_capital": log_capital,
            "true_u": u,
            "true_te": te_true,
        }
    )

    return df


# ============================================================================
# Test: Workflow Completo
# ============================================================================


def test_complete_workflow(panel_data):
    """
    Teste de integração end-to-end: workflow completo de análise SFA.

    Workflow:
    1. Estimar cross-section SFA
    2. Estimar painel Pitt-Lee
    3. Estimar True FE e True RE
    4. Comparar modelos (LR test, Hausman)
    5. Testar presença de ineficiência
    6. Obter eficiências
    7. Verificar que tudo funciona sem erros
    """
    print("\n" + "=" * 70)
    print("TEST: Complete Workflow - End-to-End Integration")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Step 1: Estimar Cross-section SFA (média por entidade)
    # -------------------------------------------------------------------------
    print("\nStep 1: Estimating cross-section SFA...")

    df_cs = (
        panel_data.groupby("entity")
        .agg(
            {
                "log_output": "mean",
                "log_labor": "mean",
                "log_capital": "mean",
                "true_te": "mean",
            }
        )
        .reset_index()
    )

    sf_cs = StochasticFrontier(
        data=df_cs,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="half_normal",
    )

    result_cs = sf_cs.fit(method="mle")

    assert result_cs is not None
    assert hasattr(result_cs, "params")
    assert hasattr(result_cs, "loglik")
    assert result_cs.sigma_u > 0
    assert result_cs.sigma_v > 0

    print(f"  ✓ Cross-section estimated successfully")
    print(f"    Log-likelihood: {result_cs.loglik:.2f}")
    print(f"    σᵤ: {result_cs.sigma_u:.4f}, σᵥ: {result_cs.sigma_v:.4f}")

    # -------------------------------------------------------------------------
    # Step 2: Estimar Painel Pitt-Lee
    # -------------------------------------------------------------------------
    print("\nStep 2: Estimating panel SFA (Pitt-Lee)...")

    sf_pl = StochasticFrontier(
        data=panel_data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        entity_id="entity",
        time_id="time",
        frontier="production",
        dist="half_normal",
        model="pitt_lee",
    )

    result_pl = sf_pl.fit(method="mle")

    assert result_pl is not None
    assert result_pl.loglik > result_cs.loglik  # Panel deve ter melhor ajuste
    print(f"  ✓ Pitt-Lee estimated successfully")
    print(f"    Log-likelihood: {result_pl.loglik:.2f}")

    # -------------------------------------------------------------------------
    # Step 3: Estimar True Fixed Effects e True Random Effects
    # -------------------------------------------------------------------------
    print("\nStep 3: Estimating True FE and True RE...")

    # True FE
    sf_tfe = StochasticFrontier(
        data=panel_data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        entity_id="entity",
        time_id="time",
        frontier="production",
        dist="half_normal",
        model="tfe",
    )

    result_tfe = sf_tfe.fit(method="mle")

    assert result_tfe is not None
    print(f"  ✓ True FE estimated successfully")
    print(f"    Log-likelihood: {result_tfe.loglik:.2f}")

    # True RE
    sf_tre = StochasticFrontier(
        data=panel_data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        entity_id="entity",
        time_id="time",
        frontier="production",
        dist="half_normal",
        model="tre",
    )

    result_tre = sf_tre.fit(method="mle")

    assert result_tre is not None
    print(f"  ✓ True RE estimated successfully")
    print(f"    Log-likelihood: {result_tre.loglik:.2f}")

    # -------------------------------------------------------------------------
    # Step 4: Comparar Modelos
    # -------------------------------------------------------------------------
    print("\nStep 4: Comparing models...")

    # LR test: Pitt-Lee vs Cross-section (nested)
    # Note: In practice, these are not strictly nested, but we test the function
    lr_result = lr_test(result_pl.loglik, result_cs.loglik, df=2)
    print(
        f"  ✓ LR test executed: statistic={lr_result['lr_statistic']:.2f}, p={lr_result['p_value']:.4f}"
    )

    # Hausman test: TFE vs TRE
    hausman_result = hausman_test_tfe_tre(result_tfe, result_tre)
    print(f"  ✓ Hausman test executed: statistic={hausman_result['statistic']:.2f}")

    # -------------------------------------------------------------------------
    # Step 5: Testar Presença de Ineficiência
    # -------------------------------------------------------------------------
    print("\nStep 5: Testing inefficiency presence...")

    ineff_test = inefficiency_presence_test(result_pl)

    assert "lr_statistic" in ineff_test
    assert "p_value" in ineff_test
    print(
        f"  ✓ Inefficiency test: LR={ineff_test['lr_statistic']:.2f}, p={ineff_test['p_value']:.4f}"
    )

    if ineff_test["p_value"] < 0.05:
        print(f"    → Inefficiency is statistically significant")
    else:
        print(f"    → Inefficiency not detected (warning)")

    # -------------------------------------------------------------------------
    # Step 6: Obter Eficiências
    # -------------------------------------------------------------------------
    print("\nStep 6: Computing efficiencies...")

    # BC corrected efficiencies
    eff_pl = result_pl.efficiency(estimator="bc")

    assert isinstance(eff_pl, pd.DataFrame)
    assert "te" in eff_pl.columns
    assert len(eff_pl) == panel_data["entity"].nunique()
    assert (eff_pl["te"] > 0).all()
    assert (eff_pl["te"] <= 1).all()

    mean_te = eff_pl["te"].mean()
    true_mean_te = panel_data.groupby("entity")["true_te"].mean().mean()

    print(f"  ✓ Efficiencies computed successfully")
    print(f"    Mean TE (estimated): {mean_te:.4f}")
    print(f"    Mean TE (true):      {true_mean_te:.4f}")
    print(f"    Estimation error:    {abs(mean_te - true_mean_te):.4f}")

    # Correlation between estimated and true TE
    eff_comparison = panel_data.groupby("entity").agg({"true_te": "mean"}).reset_index()
    eff_comparison["te_estimated"] = eff_pl["te"].values

    correlation = eff_comparison[["true_te", "te_estimated"]].corr().iloc[0, 1]
    print(f"    Correlation (true vs estimated): {correlation:.4f}")

    assert correlation > 0.5, "Correlation between true and estimated TE should be > 0.5"

    # -------------------------------------------------------------------------
    # Step 7: Verificar API de Relatórios
    # -------------------------------------------------------------------------
    print("\nStep 7: Testing reporting API...")

    # Summary
    summary = result_pl.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    print(f"  ✓ Summary generated ({len(summary)} characters)")

    # Check that summary contains key information
    assert "Log-likelihood" in summary
    assert "sigma_u" in summary or "σ_u" in summary

    print("\n" + "=" * 70)
    print("INTEGRATION TEST: PASSED ✓")
    print("=" * 70)


# ============================================================================
# Test: Import from panelbox
# ============================================================================


def test_import_from_panelbox():
    """
    Teste que os imports do módulo frontier funcionam via panelbox.
    """
    print("\n" + "=" * 70)
    print("TEST: Import from panelbox")
    print("=" * 70)

    # Verificar que pode importar diretamente de panelbox
    try:
        from panelbox import DistributionType, FrontierType, ModelType, SFResult, StochasticFrontier

        print("  ✓ All imports successful from panelbox")

        # Verificar enums
        assert FrontierType.PRODUCTION == "production"
        assert FrontierType.COST == "cost"
        assert DistributionType.HALF_NORMAL == "half_normal"
        assert DistributionType.EXPONENTIAL == "exponential"
        assert ModelType.PITT_LEE == "pitt_lee"

        print("  ✓ Enums working correctly")

    except ImportError as e:
        pytest.fail(f"Import from panelbox failed: {e}")

    print("  PASSED ✓\n")


# ============================================================================
# Test: Distribuições Alternativas
# ============================================================================


def test_alternative_distributions(panel_data):
    """
    Testar que múltiplas distribuições funcionam e podem ser comparadas.
    """
    print("\n" + "=" * 70)
    print("TEST: Alternative Distributions")
    print("=" * 70)

    # Agregar para cross-section
    df_cs = (
        panel_data.groupby("entity")
        .agg({"log_output": "mean", "log_labor": "mean", "log_capital": "mean"})
        .reset_index()
    )

    distributions = ["half_normal", "exponential", "truncated_normal"]
    results = {}

    for dist in distributions:
        print(f"\n  Estimating with {dist} distribution...")

        sf = StochasticFrontier(
            data=df_cs,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist=dist,
        )

        result = sf.fit(method="mle")
        results[dist] = result

        print(f"    Log-likelihood: {result.loglik:.2f}")
        print(f"    σᵤ: {result.sigma_u:.4f}, σᵥ: {result.sigma_v:.4f}")

    # Comparar half-normal vs truncated normal (nested)
    print("\n  Comparing half-normal vs truncated normal...")

    comparison = compare_nested_distributions(results["half_normal"], results["truncated_normal"])

    assert "lr_statistic" in comparison
    assert "p_value" in comparison
    assert "preferred_model" in comparison

    print(f"    LR statistic: {comparison['lr_statistic']:.2f}")
    print(f"    p-value: {comparison['p_value']:.4f}")
    print(f"    Preferred: {comparison['preferred_model']}")

    print("\n  PASSED ✓\n")


# ============================================================================
# Test: Skewness Test
# ============================================================================


def test_skewness_test(panel_data):
    """
    Testar que o teste de skewness detecta corretamente o sinal esperado.
    """
    print("\n" + "=" * 70)
    print("TEST: Skewness Test")
    print("=" * 70)

    df_cs = (
        panel_data.groupby("entity")
        .agg({"log_output": "mean", "log_labor": "mean", "log_capital": "mean"})
        .reset_index()
    )

    y = df_cs["log_output"].values
    X = df_cs[["log_labor", "log_capital"]].values

    # Production frontier: skewness should be negative
    result = skewness_test(y, X, frontier_type="production")

    print(f"\n  Frontier type: production")
    print(f"  Skewness: {result['skewness']:.4f}")
    print(f"  Expected: negative")
    print(f"  Test statistic: {result['test_statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")

    # Skewness should be negative for production frontier
    assert result["skewness"] < 0, "Skewness should be negative for production frontier"

    if result["p_value"] < 0.05:
        print(f"  → Skewness is statistically significant")
    else:
        print(f"  → Warning: Skewness not significant")

    print("\n  PASSED ✓\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
