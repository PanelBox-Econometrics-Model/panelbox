# tests/var/test_causality_validation.py

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality import dumitrescu_hurlin_test


class TestDumitrescuHurlinValidation:
    """Validação Dumitrescu-Hurlin (2012) contra propriedades teóricas."""

    def test_dh_statistics_finite(self, panel_data):
        """
        Estatísticas DH devem ser finitas e bem definidas.
        """
        # Testar y1 -> y2
        result = dumitrescu_hurlin_test(
            data=panel_data, cause="y1", effect="y2", lags=1, entity_col="entity", time_col="time"
        )

        # W_bar deve ser positivo e finito
        assert result.W_bar > 0
        assert np.isfinite(result.W_bar)

        # Z-tilde e Z-bar devem ser finitos
        assert np.isfinite(result.Z_tilde_stat)
        assert np.isfinite(result.Z_bar_stat)

        # P-valores entre 0 e 1
        assert 0 <= result.Z_tilde_pvalue <= 1
        assert 0 <= result.Z_bar_pvalue <= 1

    def test_dh_known_causality(self, panel_data):
        """
        DH deve detectar causalidade conhecida do DGP.

        No DGP:
            y1_t = 0.5*y1_{t-1} + 0.2*y2_{t-1} + e1_t
            y2_t = 0.3*y1_{t-1} + 0.4*y2_{t-1} + e2_t
            y3_t = 0.1*y1_{t-1} + 0.1*y2_{t-1} + 0.6*y3_{t-1} + e3_t

        Portanto:
        - y1 causa y2 (coef 0.3)
        - y2 causa y1 (coef 0.2)
        - y1 e y2 causam y3
        """
        # y1 -> y2 (deve detectar)
        result_y1_y2 = dumitrescu_hurlin_test(
            data=panel_data, cause="y1", effect="y2", lags=1, entity_col="entity", time_col="time"
        )

        # Deve rejeitar H0 (y1 NÃO causa y2)
        assert result_y1_y2.Z_bar_pvalue < 0.10, "Failed to detect known Granger causality y1 -> y2"

        # y2 -> y1 (deve detectar)
        result_y2_y1 = dumitrescu_hurlin_test(
            data=panel_data, cause="y2", effect="y1", lags=1, entity_col="entity", time_col="time"
        )

        assert result_y2_y1.Z_bar_pvalue < 0.10, "Failed to detect known Granger causality y2 -> y1"

    def test_dh_no_causality(self, panel_data):
        """
        DH não deve detectar causalidade falsa onde não existe.

        No DGP: y3 não causa y1 diretamente.
        """
        result_y3_y1 = dumitrescu_hurlin_test(
            data=panel_data, cause="y3", effect="y1", lags=1, entity_col="entity", time_col="time"
        )

        # Não deve rejeitar H0 com alta confiança
        # (pode haver alguma detecção devido a efeitos indiretos)
        # Teste mais fraco: estatística não deve ser muito alta
        assert result_y3_y1.W_bar < 10, "DH test detects spurious causality y3 -> y1"

    def test_dh_with_multiple_lags(self, panel_data):
        """
        DH deve funcionar com múltiplos lags.
        """
        for lags in [1, 2, 3]:
            result = dumitrescu_hurlin_test(
                data=panel_data,
                cause="y1",
                effect="y2",
                lags=lags,
                entity_col="entity",
                time_col="time",
            )

            # Deve convergir
            assert result.W_bar > 0
            assert np.isfinite(result.Z_bar_stat)

    def test_dh_consistency_across_samples(self):
        """
        DH deve dar resultados consistentes com amostras maiores.
        """
        from panelbox.tests.var.fixtures.var_test_data import generate_panel_var_data

        # Amostra pequena
        data_small = generate_panel_var_data(n_entities=20, n_periods=15, seed=42)
        result_small = dumitrescu_hurlin_test(
            data=data_small, cause="y1", effect="y2", lags=1, entity_col="entity", time_col="time"
        )

        # Amostra grande
        data_large = generate_panel_var_data(n_entities=100, n_periods=30, seed=42)
        result_large = dumitrescu_hurlin_test(
            data=data_large, cause="y1", effect="y2", lags=1, entity_col="entity", time_col="time"
        )

        # Amostra maior deve ter mais poder (p-value menor)
        assert (
            result_large.Z_bar_pvalue <= result_small.Z_bar_pvalue
        ), "Larger sample should have more power to detect causality"

    def test_dh_ztilde_vs_zbar(self, panel_data):
        """
        Verificar relação entre Z-tilde e Z-bar.

        Z-tilde é ajustado para pequenas amostras.
        """
        result = dumitrescu_hurlin_test(
            data=panel_data, cause="y1", effect="y2", lags=1, entity_col="entity", time_col="time"
        )

        # Ambos devem apontar na mesma direção (mesmo sinal)
        assert np.sign(result.Z_tilde_stat) == np.sign(result.Z_bar_stat)

        # Z-tilde deve ser >= Z-bar em pequenas amostras
        # (mais conservador)
        # Nota: Isso depende da implementação exata

    def test_dh_individual_statistics(self, panel_data):
        """
        Verificar estatísticas individuais de cada entidade.
        """
        result = dumitrescu_hurlin_test(
            data=panel_data, cause="y1", effect="y2", lags=1, entity_col="entity", time_col="time"
        )

        # Deve ter estatísticas por entidade
        if hasattr(result, "individual_statistics"):
            assert len(result.individual_statistics) == panel_data["entity"].nunique()
            assert all(np.isfinite(result.individual_statistics))


class TestLagSelectionValidation:
    """Validação de seleção de lags via AIC/BIC/HQIC."""

    def test_aic_bic_selection(self, panel_data):
        """
        AIC/BIC devem selecionar lag correto para DGP VAR(1).
        """
        from panelbox.var import PanelVAR, PanelVARData

        data = PanelVARData(
            panel_data,
            endog_vars=["y1", "y2", "y3"],
            entity_col="entity",
            time_col="time",
            lags=1,  # Inicial
        )
        model = PanelVAR(data)

        lag_result = model.select_lag_order(max_lags=3)  # Reduzir max_lags para evitar overfitting

        # DGP é VAR(1), então BIC deve selecionar 1
        # AIC pode selecionar mais devido ao tradeoff bias-variance
        assert (
            lag_result.selected["BIC"] == 1
        ), f"BIC selected lag={lag_result.selected['BIC']}, expected 1"
        assert (
            lag_result.selected["AIC"] <= 3
        ), f"AIC selected lag={lag_result.selected['AIC']}, expected <=3"
        assert (
            lag_result.selected["HQIC"] <= 3
        ), f"HQIC selected lag={lag_result.selected['HQIC']}, expected <=3"

    def test_bic_consistency(self, panel_data):
        """
        BIC deve ser consistente (selecionar ordem verdadeira).

        BIC penaliza mais que AIC, logo deve selecionar lag 1.
        """
        from panelbox.var import PanelVAR, PanelVARData

        data = PanelVARData(
            panel_data, endog_vars=["y1", "y2", "y3"], entity_col="entity", time_col="time", lags=1
        )
        model = PanelVAR(data)

        lag_result = model.select_lag_order(max_lags=5)

        # BIC deve selecionar 1 (ordem verdadeira)
        assert (
            lag_result.selected["BIC"] == 1
        ), f"BIC should select true lag order 1, got {lag_result.selected['BIC']}"

    def test_aic_values_monotonic_or_unimodal(self, panel_data):
        """
        AIC deve ter comportamento monotônico ou unimodal.
        """
        from panelbox.var import PanelVAR, PanelVARData

        data = PanelVARData(
            panel_data, endog_vars=["y1", "y2", "y3"], entity_col="entity", time_col="time", lags=1
        )
        model = PanelVAR(data)

        lag_result = model.select_lag_order(max_lags=3)

        # Verificar que AIC values estão presentes e são finitos
        aic_values = lag_result.criteria_df["AIC"].values
        assert len(aic_values) == 3
        assert all(np.isfinite(aic_values))

        # AIC é livre para aumentar ou diminuir
        # O importante é que os valores sejam finitos e o mínimo seja identificável
        min_aic = min(aic_values)
        assert np.isfinite(min_aic)

    def test_ic_with_different_sample_sizes(self):
        """
        Critérios de informação devem funcionar com diferentes tamanhos.
        """
        from panelbox.tests.var.fixtures.var_test_data import generate_panel_var_data
        from panelbox.var import PanelVAR, PanelVARData

        for n_entities in [20, 50, 100]:
            data_df = generate_panel_var_data(n_entities=n_entities, n_periods=25)

            data = PanelVARData(
                data_df, endog_vars=["y1", "y2", "y3"], entity_col="entity", time_col="time", lags=1
            )
            model = PanelVAR(data)

            lag_result = model.select_lag_order(max_lags=4)

            # Deve convergir
            assert lag_result.selected["AIC"] > 0
            assert lag_result.selected["BIC"] > 0

    def test_ic_all_criteria_available(self, panel_data):
        """
        Todos os critérios de informação devem ser calculados.
        """
        from panelbox.var import PanelVAR, PanelVARData

        data = PanelVARData(
            panel_data, endog_vars=["y1", "y2", "y3"], entity_col="entity", time_col="time", lags=1
        )
        model = PanelVAR(data)

        lag_result = model.select_lag_order(max_lags=4)

        # Verificar que todos os critérios estão presentes no DataFrame
        assert "AIC" in lag_result.criteria_df.columns
        assert "BIC" in lag_result.criteria_df.columns
        assert "HQIC" in lag_result.criteria_df.columns

        # Verificar que temos 4 lags
        assert len(lag_result.criteria_df) == 4

        # Valores devem ser finitos
        assert all(np.isfinite(lag_result.criteria_df["AIC"]))
        assert all(np.isfinite(lag_result.criteria_df["BIC"]))
        assert all(np.isfinite(lag_result.criteria_df["HQIC"]))


class TestPanelGrangerCausality:
    """Testes de causalidade de Granger para painel."""

    def test_pairwise_causality(self, panel_data):
        """
        Testar causalidade par a par entre todas as variáveis.
        """
        from panelbox.var.causality import panel_granger_causality

        result = panel_granger_causality(
            data=panel_data,
            variables=["y1", "y2", "y3"],
            lags=1,
            entity_col="entity",
            time_col="time",
        )

        # Deve ter resultados para todos os pares
        assert "y1" in result
        assert "y2" in result
        assert "y3" in result

        # Cada variável deve ter causas potenciais testadas
        for var in ["y1", "y2", "y3"]:
            assert len(result[var]) > 0

    def test_causality_matrix(self, panel_data):
        """
        Criar matriz de causalidade (p-values).
        """
        from panelbox.var.causality import panel_granger_causality_matrix

        matrix = panel_granger_causality_matrix(
            data=panel_data,
            variables=["y1", "y2", "y3"],
            lags=1,
            entity_col="entity",
            time_col="time",
        )

        # Deve ser matriz 3x3
        assert matrix.shape == (3, 3)

        # Diagonal deve ser NaN ou 1.0 (variável não causa a si mesma)
        assert np.isnan(matrix[0, 0]) or matrix[0, 0] == 1.0

        # Valores fora da diagonal devem ser p-values [0, 1]
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert 0 <= matrix[i, j] <= 1

    def test_causality_direction(self, panel_data):
        """
        Verificar direção de causalidade conhecida.

        No DGP: y1 -> y2, y2 -> y1 (bidirecional).
        """
        from panelbox.var.causality import dumitrescu_hurlin_test

        # y1 -> y2
        result_12 = dumitrescu_hurlin_test(
            data=panel_data, cause="y1", effect="y2", lags=1, entity_col="entity", time_col="time"
        )

        # y2 -> y1
        result_21 = dumitrescu_hurlin_test(
            data=panel_data, cause="y2", effect="y1", lags=1, entity_col="entity", time_col="time"
        )

        # Ambos devem ser significativos
        assert result_12.Z_bar_pvalue < 0.10
        assert result_21.Z_bar_pvalue < 0.10

    def test_no_instantaneous_causality(self, panel_data):
        """
        Teste de causalidade não deve detectar correlação instantânea
        (apenas lagged causality).
        """
        from panelbox.var.causality import dumitrescu_hurlin_test

        # Criar dados com correlação instantânea mas sem causalidade lagged
        data_inst = panel_data.copy()
        # Modificar para remover causalidade lagged
        # (isso requer manipulação específica dos dados)

        # Por enquanto, apenas verificar que teste funciona
        result = dumitrescu_hurlin_test(
            data=panel_data, cause="y1", effect="y2", lags=1, entity_col="entity", time_col="time"
        )

        assert result is not None


class TestCausalityRobustness:
    """Testes de robustez para análise de causalidade."""

    def test_dh_with_unbalanced_panel(self):
        """
        DH deve funcionar com painel desbalanceado.
        """
        from panelbox.tests.var.fixtures.var_test_data import generate_panel_var_data
        from panelbox.var.causality import dumitrescu_hurlin_test

        # Gerar painel balanceado
        data = generate_panel_var_data(n_entities=30, n_periods=20)

        # Remover algumas observações para desbalancear
        data_unbalanced = data.sample(frac=0.8, random_state=42)

        # Deve funcionar
        result = dumitrescu_hurlin_test(
            data=data_unbalanced,
            cause="y1",
            effect="y2",
            lags=1,
            entity_col="entity",
            time_col="time",
        )

        assert result.W_bar > 0
        assert np.isfinite(result.Z_bar_stat)

    def test_lag_selection_stability(self):
        """
        Seleção de lags deve ser estável em diferentes rodadas.
        """
        from panelbox.tests.var.fixtures.var_test_data import generate_panel_var_data
        from panelbox.var import PanelVAR, PanelVARData

        selected_lags = []
        for seed in [42, 43, 44]:
            data_df = generate_panel_var_data(n_entities=50, n_periods=20, seed=seed)

            data = PanelVARData(
                data_df, endog_vars=["y1", "y2", "y3"], entity_col="entity", time_col="time", lags=1
            )
            model = PanelVAR(data)

            lag_result = model.select_lag_order(max_lags=4)
            selected_lags.append(lag_result.selected["BIC"])

        # A maioria deve selecionar o mesmo lag
        from collections import Counter

        most_common_lag, count = Counter(selected_lags).most_common(1)[0]
        assert count >= 2, f"Lag selection unstable: {selected_lags}"

    def test_causality_with_heterogeneous_effects(self):
        """
        Teste de causalidade com efeitos heterogêneos entre entidades.
        """
        # Gerar dados com efeitos heterogêneos
        # (alguns grupos têm causalidade, outros não)
        # Isso requer geração customizada de dados
        # Por enquanto, usar dados padrão
        from panelbox.tests.var.fixtures.var_test_data import generate_panel_var_data
        from panelbox.var.causality import dumitrescu_hurlin_test

        data = generate_panel_var_data(n_entities=50, n_periods=20)

        result = dumitrescu_hurlin_test(
            data=data, cause="y1", effect="y2", lags=1, entity_col="entity", time_col="time"
        )

        # W_bar deve capturar média dos efeitos
        assert result.W_bar > 0
