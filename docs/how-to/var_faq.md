# Panel VAR - FAQ & Troubleshooting

Este documento responde às perguntas mais frequentes sobre o uso do módulo Panel VAR da PanelBox e fornece soluções para problemas comuns.

---

## Perguntas Frequentes (FAQ)

### 1. Qual é a diferença entre Panel VAR e VAR tradicional?

**Panel VAR** estende o VAR tradicional (séries temporais) para dados em painel:

- **VAR tradicional:** N=1 entidade, T períodos
- **Panel VAR:** N entidades, T períodos

**Vantagens do Panel VAR:**
- Maior poder estatístico (mais observações)
- Captura heterogeneidade cross-section
- Permite análise de efeitos médios entre entidades
- Ideal para dados macro cross-country ou firmas

**Quando usar cada um:**
- Use **VAR tradicional** para análise de uma única série temporal (ex: macroeconômica de um país)
- Use **Panel VAR** quando você tem múltiplas entidades com dinâmicas similares

### 2. Quando devo usar Panel VAR vs Panel GMM (Arellano-Bond)?

**Panel VAR:**
- Sistema de equações (múltiplas variáveis endógenas)
- Interesse em dinâmicas conjuntas, causalidade de Granger, IRFs
- Exemplo: Analisar interações entre GDP, inflation, interest rate

**Panel GMM (Arellano-Bond):**
- Equação única com lagged dependent variable
- Foco em estimar efeito de covariates controlando dinâmica
- Exemplo: Estimar efeito de investimento em crescimento, controlando por crescimento passado

**Resumo:** Se você tem múltiplas variáveis endógenas e quer entender suas interações dinâmicas, use Panel VAR. Se você tem uma única equação dinâmica, use Panel GMM.

### 3. Quando devo usar Panel VAR vs Panel VECM?

**Panel VAR:**
- Variáveis são **estacionárias** (I(0))
- Modelo em níveis ou primeiras diferenças

**Panel VECM:**
- Variáveis são **não-estacionárias** (I(1)) e **cointegradas**
- Captura relações de longo prazo (cointegração) + dinâmicas de curto prazo

**Workflow recomendado:**

1. Testar raiz unitária:
```python
from panelbox.tests.unit_root import panel_unit_root_test

for var in variables:
    result = panel_unit_root_test(data[var], test='llc')
    print(f"{var}: {'I(0)' if result.reject else 'I(1)'}")
```

2. Se todas I(0) → **Panel VAR**

3. Se todas I(1) → testar cointegração:
```python
from panelbox.tests.cointegration import pedroni_test

coint_result = pedroni_test(data, endog_vars=variables)
if coint_result.reject:
    # Use Panel VECM
else:
    # Use Panel VAR em primeiras diferenças
```

### 4. OLS ou GMM para Panel VAR?

**OLS (Least Squares):**
- **Vantagens:** Simples, rápido, bom para N pequeno
- **Desvantagens:** Viés de Nickell quando T pequeno, assume exogeneidade estrita
- **Use quando:** T >> N (muitos períodos), variáveis exógenas

**GMM (Generalized Method of Moments):**
- **Vantagens:** Consistente mesmo com T pequeno, robusto a endogeneidade
- **Desvantagens:** Mais complexo, requer N grande, sensível a instrumentos
- **Use quando:** T pequeno (~10-20), endogeneidade potencial, N razoável (≥50)

**Recomendação:**
- Começar com OLS para entender os dados
- Usar GMM se diagnósticos indicarem problemas (endogeneidade, T pequeno)
- Comparar resultados entre métodos

### 5. Quantos lags devo usar no Panel VAR?

**Métodos de seleção:**

```python
# Usar critérios de informação
result = pvar.select_lag_order(max_lags=5, criterion='bic')
print(f"Optimal lags: {result.optimal_lag}")
```

**Critérios disponíveis:**
- **BIC (Bayesian):** Penaliza mais parâmetros, escolhe lags menores
- **AIC (Akaike):** Menos conservador, pode escolher mais lags
- **HQIC (Hannan-Quinn):** Intermediário

**Recomendações práticas:**
- **Dados anuais:** p = 1 ou 2
- **Dados trimestrais:** p = 1 a 4
- **Dados mensais:** p = 1 a 12

**Atenção:** Mais lags ≠ melhor modelo
- Muitos lags: overfitting, perda de graus de liberdade
- Poucos lags: especificação incorreta, autocorrelação residual

### 6. Como interpretar os Impulse Response Functions (IRFs)?

**IRF mede:** Resposta de uma variável a um choque em outra variável, ao longo do tempo.

**Exemplo:**
```python
irf_result = result.irf(periods=10, method='cholesky')
irf_result.plot(impulse='interest_rate', response='gdp')
```

**Interpretação:**
- **Eixo X:** Horizonte temporal (períodos após o choque)
- **Eixo Y:** Magnitude da resposta
- **Banda sombreada:** Intervalo de confiança

**Exemplo de interpretação:**
"Um aumento de 1% na taxa de juros leva a uma redução de 0.3% no GDP após 2 anos, com efeito persistindo por aproximadamente 5 anos."

**Atenção:**
- IRFs Cholesky dependem da **ordem das variáveis** (identificação recursiva)
- Use **Generalized IRFs** se quiser evitar dependência da ordem
- Justifique teoricamente a ordenação escolhida

### 7. O que é FEVD e quando devo usá-lo?

**FEVD (Forecast Error Variance Decomposition):** Quantifica a importância relativa de cada choque para explicar a variância do erro de previsão.

**Interpretação:**
```python
fevd_result = result.fevd(periods=10)
fevd_result.plot(variable='gdp')
```

**Resultado:**
"Após 10 períodos, 60% da variância do GDP é explicada por choques no próprio GDP, 30% por choques em interest_rate, e 10% por choques em inflation."

**Quando usar:**
- Identificar quais variáveis são mais importantes para explicar flutuações
- Complementar análise de IRFs
- Análise de transmissão de choques

### 8. Como verificar se meu Panel VAR está bem especificado?

**Checklist de diagnósticos:**

1. **Estabilidade:**
```python
if result.is_stable():
    print("✓ VAR é estável")
else:
    print("✗ VAR é instável - IRFs podem divergir")
```

2. **Autocorrelação residual:**
```python
# Teste de Portmanteau
lm_test = result.test_serial_correlation(lags=4)
if lm_test.pvalue > 0.05:
    print("✓ Sem autocorrelação")
```

3. **Normalidade dos resíduos:**
```python
jb_test = result.test_normality()
# Menos crítico assintoticamente
```

4. **Heterocedasticidade:**
```python
# Visual: plots residuais
result.plot_residuals()
```

5. **Hansen J test (somente GMM):**
```python
if result.method == 'gmm':
    if result.hansen_j_pvalue > 0.05:
        print("✓ Instrumentos válidos")
    else:
        print("✗ Overidentifying restrictions rejeitadas")
```

### 9. Meu Hansen J test falhou. O que devo fazer?

**Hansen J test rejeita:** Indica que os instrumentos podem não ser válidos ou o modelo está mal especificado.

**Possíveis soluções:**

1. **Reduzir número de instrumentos:**
```python
result = pvar.fit(method='gmm', instruments='collapsed', max_lag_instruments=2)
```

2. **Tentar transformação diferente:**
```python
# FOD ao invés de FD
result = pvar.fit(method='gmm', transform='fod')
```

3. **Verificar especificação do modelo:**
- Adicionar/remover lags
- Incluir exógenas omitidas
- Verificar presença de quebras estruturais

4. **Usar AR tests como diagnóstico adicional:**
```python
print(f"AR(1): {result.ar1_pvalue}")  # Esperado < 0.05
print(f"AR(2): {result.ar2_pvalue}")  # Esperado > 0.05
```

### 10. Posso incluir variáveis exógenas no Panel VAR?

**Sim!** Panel VAR permite variáveis exógenas determinísticas:

```python
pvar = PanelVAR(
    data,
    endog_vars=['y1', 'y2', 'y3'],
    exog_vars=['time_trend', 'oil_price'],
    entity_col='entity',
    time_col='time'
)
result = pvar.fit(lags=2)
```

**Exemplos de exógenas comuns:**
- **Time trend:** Captura tendências determinísticas
- **Dummies sazonais:** Controla sazonalidade
- **Dummies estruturais:** Captura quebras conhecidas (ex: crises)
- **Variáveis globais:** Ex: preço do petróleo em painel de países

**Atenção:** Exógenas devem ser **verdadeiramente exógenas** (não correlacionadas com erros futuros).

---

## Troubleshooting

### Problema 1: "LinAlgError: Singular matrix"

**Causa:** Matriz de dados ou instrumentos é singular (não inversível).

**Possíveis razões:**
- Variáveis perfeitamente colineares
- Muito poucos dados
- Muitos instrumentos em GMM (overidentification extremo)

**Soluções:**

1. **Verificar colinearidade:**
```python
import numpy as np
corr_matrix = data[endog_vars].corr()
print(corr_matrix)
# Se alguma correlação ≈ ±1, remover variável redundante
```

2. **Reduzir instrumentos (GMM):**
```python
result = pvar.fit(method='gmm', instruments='collapsed')
```

3. **Verificar dados faltantes:**
```python
print(data.isnull().sum())
```

4. **Aumentar N ou T** se possível

### Problema 2: "Warning: VAR is unstable"

**Causa:** Raízes do polinômio característico estão fora do círculo unitário.

**Implicações:**
- IRFs podem divergir (explodir)
- Previsões não confiáveis
- Modelo pode estar mal especificado

**Soluções:**

1. **Verificar se variáveis são estacionárias:**
```python
from panelbox.tests.unit_root import panel_unit_root_test

for var in endog_vars:
    test = panel_unit_root_test(data[var])
    if not test.reject:
        print(f"{var} parece não-estacionária - considere diferenciar")
```

2. **Usar VECM se variáveis são I(1) e cointegradas**

3. **Reduzir número de lags:**
```python
result = pvar.fit(lags=1)  # Tentar com menos lags
```

4. **Verificar outliers ou quebras estruturais**

### Problema 3: Intervalos de confiança muito amplos nos IRFs

**Causa:** Alta incerteza nas estimativas.

**Possíveis razões:**
- N ou T pequeno
- Modelo instável
- Alta correlação residual

**Soluções:**

1. **Usar mais dados** se disponível

2. **Aumentar número de bootstrap replicações:**
```python
irf_result = result.irf(periods=10, ci_method='bootstrap', n_boot=1000)
```

3. **Verificar estabilidade:**
```python
if not result.is_stable():
    print("Modelo instável - isto explica ICs amplos")
```

4. **Considerar acumular dados** (ex: trimestral → anual) se T muito pequeno

### Problema 4: "Insufficient instruments" em GMM

**Causa:** Não há instrumentos suficientes para identificar os parâmetros.

**Regra:** Número de instrumentos ≥ número de parâmetros

**Soluções:**

1. **Aumentar max_lag_instruments:**
```python
result = pvar.fit(method='gmm', max_lag_instruments=3)
```

2. **Verificar se T é suficiente:**
   - Mínimo: T ≥ p + 2 (para criar instrumentos válidos)
   - Recomendado: T ≥ 10 para GMM

3. **Usar OLS se T muito pequeno**

### Problema 5: Granger causality tests muito fracos (todos p > 0.05)

**Causa:** Variáveis podem não estar causalmente relacionadas, ou teste com baixo poder.

**Possíveis razões:**
- Variáveis genuinamente não relacionadas
- N ou T pequeno (baixo poder)
- Lag order incorreto
- Variáveis omitidas

**Soluções:**

1. **Verificar lag order:**
```python
# Testar diferentes lags
for p in range(1, 6):
    result = pvar.fit(lags=p)
    gc = result.granger_causality('x1', 'x2')
    print(f"p={p}: pvalue={gc.pvalue}")
```

2. **Usar teste Dumitrescu-Hurlin (mais poder em painel):**
```python
from panelbox.var.causality import dumitrescu_hurlin_test

dh_result = dumitrescu_hurlin_test(data, cause='x1', effect='x2', lags=2)
print(f"DH test: pvalue={dh_result.pvalue}")
```

3. **Verificar se relação existe teoricamente** - pode ser que variáveis realmente não sejam relacionadas!

### Problema 6: Resultados OLS e GMM muito diferentes

**Causa:** Pode indicar endogeneidade ou viés de Nickell.

**Interpretação:**
- **GMM >> OLS:** Provável endogeneidade, confiar em GMM
- **GMM ≈ OLS:** Boa notícia - robustez
- **GMM << OLS:** Verificar especificação GMM (instrumentos fracos?)

**Ações:**

1. **Verificar Hansen J e AR tests (GMM):**
```python
print(f"Hansen J p-value: {result.hansen_j_pvalue}")
print(f"AR(2) p-value: {result.ar2_pvalue}")
```

2. **Verificar T:**
   - Se T grande (>30), OLS pode ser adequado
   - Se T pequeno (<20), preferir GMM

3. **Comparar com outros métodos** para robustez

### Problema 7: MemoryError com bootstrap

**Causa:** Bootstrap requer armazenar muitas replicações.

**Soluções:**

1. **Reduzir n_boot:**
```python
irf_result = result.irf(ci_method='bootstrap', n_boot=500)  # ao invés de 1000
```

2. **Reduzir periods:**
```python
irf_result = result.irf(periods=10)  # ao invés de 20
```

3. **Usar IC analíticos se disponível:**
```python
irf_result = result.irf(ci_method='analytical')
```

4. **Processar por variável:**
```python
# Ao invés de todas as combinações de uma vez
for impulse_var in endog_vars:
    irf = result.irf(impulse=impulse_var, ci_method='bootstrap')
    irf.plot()
```

### Problema 8: "ValueError: Data must be balanced panel"

**Causa:** Seu painel é desbalanceado (diferentes T para diferentes entidades).

**Soluções:**

1. **Verificar estrutura:**
```python
counts = data.groupby('entity').size()
print(counts.value_counts())  # Mostra distribuição de T
```

2. **Panel VAR aceita desbalanceados:**
```python
pvar = PanelVAR(data, endog_vars=vars, allow_unbalanced=True)
```

3. **Ou balancear manualmente:**
```python
# Opção 1: Remover entidades com poucos períodos
min_periods = 15
entity_counts = data.groupby('entity').size()
valid_entities = entity_counts[entity_counts >= min_periods].index
data_balanced = data[data['entity'].isin(valid_entities)]

# Opção 2: Interpolar valores faltantes (cuidado!)
data_balanced = data.groupby('entity').apply(lambda x: x.interpolate())
```

### Problema 9: Forecast diverge ou fica constante

**Causa 1 - Diverge:** Modelo instável

**Solução:**
```python
if not result.is_stable():
    print("Modelo instável - forecast não confiável")
    # Revisar especificação
```

**Causa 2 - Fica constante:** Variáveis têm pouca persistência

**Solução:**
- Isto pode ser correto! Se os eigenvalues são pequenos, forecast converge rapidamente para média
- Verificar se faz sentido teoricamente

### Problema 10: "ImportError: cannot import PanelVAR"

**Causa:** Instalação incompleta ou path incorreto.

**Soluções:**

1. **Verificar instalação:**
```python
import panelbox
print(panelbox.__version__)
```

2. **Import correto:**
```python
# Correto:
from panelbox.var import PanelVAR

# Incorreto:
from panelbox import PanelVAR  # Não funciona
```

3. **Reinstalar se necessário:**
```bash
pip install --upgrade panelbox
```

---

## Recursos Adicionais

### Documentação
- [Tutorial Completo de Panel VAR](../tutorials/panel_var_complete_guide.md)
- [API Reference](../api/var_reference.md)
- [Panel VAR Theory Guide](../theory/panel_var_theory.md)

### Exemplos
- `examples/var/simple_panel_var.py` - Exemplo básico
- `examples/var/macro_analysis.py` - Análise macro cross-country
- `examples/var/corporate_finance.py` - Finance panel
- `examples/var/gmm_advanced.py` - GMM avançado

### Suporte
- GitHub Issues: https://github.com/panelbox/panelbox/issues
- Discussões: https://github.com/panelbox/panelbox/discussions

---

## Referências Técnicas

### Papers Fundamentais
1. **Love & Zicchino (2006)** - "Financial development and dynamic investment behavior"
   - Primeiro paper a usar Panel VAR em finanças corporativas

2. **Abrigo & Love (2016)** - "Estimation of Panel Vector Autoregression in Stata"
   - Implementação de referência em Stata

3. **Holtz-Eakin et al. (1988)** - "Estimating Vector Autoregressions with Panel Data"
   - Fundamentos teóricos de GMM para Panel VAR

### Comparações R vs Python
- R `pvar`: https://cran.r-project.org/package=pvar
- R `panelvar`: Panel VAR para R
- PanelBox replicação: Ver `tests/validation/VALIDATION_NOTES.md`

---

**Última atualização:** 2026-02-13

**Contribuições:** Este documento é vivo! Encontrou um problema não listado? Abra uma issue no GitHub.
