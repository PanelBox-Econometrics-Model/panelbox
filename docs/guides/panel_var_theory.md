# Panel VAR: Guia Teórico Completo

Este guia apresenta a fundamentação teórica do Panel Vector Autoregression (Panel VAR), um dos métodos mais poderosos para analisar dinâmicas multivariadas em dados em painel.

---

## Sumário

1. [Introdução e Motivação](#introdução-e-motivação)
2. [Modelo Panel VAR](#modelo-panel-var)
3. [Estimação](#estimação)
4. [Identificação e Impulse Response Functions](#identificação-e-impulse-response-functions)
5. [Forecast Error Variance Decomposition](#forecast-error-variance-decomposition)
6. [Causalidade de Granger em Painel](#causalidade-de-granger-em-painel)
7. [Panel VECM](#panel-vecm)
8. [Comparações com Outros Métodos](#comparações-com-outros-métodos)
9. [Limitações e Cuidados](#limitações-e-cuidados)
10. [Referências](#referências)

---

## Introdução e Motivação

### O que é Panel VAR?

**Panel Vector Autoregression (Panel VAR)** estende a metodologia VAR clássica para dados em painel, combinando:

- **VAR tradicional:** Sistema de equações para múltiplas variáveis endógenas interdependentes ao longo do tempo
- **Dados em painel:** Múltiplas entidades (cross-sections) observadas ao longo do tempo

### Por que usar Panel VAR?

**Vantagens:**

1. **Maior poder estatístico:** Exploita dimensões cross-section (N) e temporal (T)
2. **Heterogeneidade:** Captura diferenças entre entidades via fixed effects
3. **Dinâmicas complexas:** Modela feedback loops e interações multivariadas
4. **Inferência causal:** Testes de causalidade de Granger em painel
5. **Análise de choques:** Impulse response functions (IRFs) e variance decomposition (FEVD)

**Aplicações típicas:**

- **Macroeconomia:** Interações entre PIB, inflação, taxa de juros em painéis de países
- **Finanças corporativas:** Relações entre vendas, investimento, dívida em painéis de firmas
- **Economia regional:** Dinâmicas de emprego, renda, migração entre regiões
- **Comércio internacional:** Efeitos de políticas comerciais em painéis de parceiros comerciais

---

## Modelo Panel VAR

### Especificação Básica

O modelo Panel VAR(p) é dado por:

```
y_it = α_i + Σ(l=1 to p) A_l y_i,t-l + X_it γ + ε_it
```

Onde:

- **y_it:** Vetor K×1 de variáveis endógenas para entidade *i* no tempo *t*
- **α_i:** Vetor K×1 de fixed effects da entidade *i* (captura heterogeneidade)
- **A_l:** Matriz K×K de coeficientes para lag *l* (comum a todas as entidades)
- **X_it:** Vetor de variáveis exógenas/determinísticas (opcional)
- **γ:** Vetor de coeficientes das exógenas
- **ε_it:** Vetor K×1 de termos de erro
- **p:** Ordem de lags

### Pressupostos

1. **Estacionariedade:** y_it é estacionário (ou usar Panel VECM se I(1) cointegrado)
2. **Homogeneidade de slopes:** Matrizes A_l são iguais para todas as entidades
3. **Independência cross-section:** E[ε_it ε_jt'] = 0 para i ≠ j (pode ser relaxado)
4. **Exogeneidade sequencial:** E[ε_it | y_i,t-1, y_i,t-2, ..., α_i] = 0

### Forma Matricial

Para toda a amostra, empilhando observações:

```
Y = α ⊗ ι_T + Y_lag A + X Γ + ε
```

Onde:
- **Y:** Matriz NT×K de todas as observações
- **ι_T:** Vetor de 1s de tamanho T
- **⊗:** Produto de Kronecker
- **A:** Matriz de todos os coeficientes lag empilhados

---

## Estimação

### Problema da Estimação

**Desafio:** Presença de fixed effects α_i + lagged dependent variables cria **viés de Nickell**.

Quando estimamos:
```
y_it = α_i + A y_i,t-1 + ε_it
```

Por OLS com dummies, o estimador de A é **viesado** quando T é pequeno porque y_i,t-1 é correlacionado com os resíduos dentro do grupo após demeaning.

### Métodos de Estimação

#### 1. OLS com Fixed Effects

**Procedimento:**
1. Transformar dados para remover fixed effects (within transformation ou first-differences)
2. Estimar por OLS

**Prós:**
- Simples e rápido
- Consistente quando T → ∞

**Contras:**
- Viesado quando T pequeno (viés de Nickell)
- Requer T >> N idealmente

**Quando usar:**
- T grande (≥ 30)
- Análise exploratória inicial

#### 2. GMM (Generalized Method of Moments)

**Procedimento:**
1. Transformar para remover fixed effects (FOD ou FD)
2. Usar lags mais profundos como instrumentos
3. Estimar via GMM

**Transformações:**

**a) First Differences (FD):**
```
Δy_it = y_it - y_i,t-1
```

Remove α_i por diferenciação.

**b) First-Orthogonal Deviations (FOD):**
```
y*_it = √(T_i - t)/(T_i - t + 1) * (y_it - ȳ_i,t+1:T)
```

Onde ȳ_i,t+1:T é a média de todas as observações futuras.

**Vantagens de FOD:**
- Preserva mais observações em painéis desbalanceados
- Menor correlação serial dos erros transformados
- Mais eficiente que FD

**Instrumentos:**

Para equação em t, usar:
- y_i,t-2, y_i,t-3, ..., y_i,1 como instrumentos

**Tipos de GMM:**

1. **One-step GMM:**
   - Weight matrix: identidade ou baseada em estrutura conhecida
   - Mais rápido, menos eficiente

2. **Two-step GMM:**
   - Weight matrix: baseada em resíduos do one-step
   - Assintoticamente eficiente
   - Pode ter propriedades de amostra finita ruins

**Critério de momento:**
```
E[Z_i' ε*_i] = 0
```

Onde Z_i são instrumentos e ε*_i são resíduos transformados.

**Estimador GMM:**
```
θ̂_GMM = argmin (1/N Σ Z_i' ε*_i)' W (1/N Σ Z_i' ε*_i)
```

Onde W é a weight matrix.

**Prós:**
- Consistente mesmo com T pequeno
- Robusto a endogeneidade
- Permite heterogeneidade nas variâncias

**Contras:**
- Sensível a proliferação de instrumentos (quando T moderado)
- Requer N razoável (≥ 50 recomendado)
- Pode ter viés em amostras pequenas

**Diagnósticos GMM:**

1. **Hansen J Test (Overidentification):**
   ```
   J = (1/N Σ Z_i' ε̂*_i)' Ŵ^-1 (1/N Σ Z_i' ε̂*_i) ~ χ²(q - k)
   ```
   - H0: instrumentos são válidos
   - Rejeição indica problemas de especificação ou instrumentos inválidos

2. **AR(1) e AR(2) Tests (Arellano-Bond):**
   - AR(1): E[Δε_it Δε_i,t-1] → esperado **significativo**
   - AR(2): E[Δε_it Δε_i,t-2] → esperado **não significativo**
   - AR(2) significativo indica especificação inadequada

3. **Instrument Proliferation:**
   - Regra: número de instrumentos < número de entidades (N)
   - Solução: "collapsed instruments" (Roodman 2009)

---

## Identificação e Impulse Response Functions

### O Problema de Identificação

O Panel VAR na forma reduzida:
```
y_it = Σ A_l y_i,t-l + ε_it
```

Tem resíduos ε_it com matriz de covariância Σ = E[ε_it ε_it'].

Para interpretar **choques estruturais**, precisamos identificar:
```
y_it = Σ A_l y_i,t-l + B u_it
```

Onde:
- **u_it:** Choques estruturais ortogonais (E[u_it u_it'] = I)
- **B:** Matriz de impacto contemporâneo (K×K)
- **Relação:** ε_it = B u_it, então Σ = B B'

**Desafio:** Temos K(K+1)/2 elementos únicos em Σ, mas K² elementos em B.
→ Precisamos impor K(K-1)/2 restrições.

### Estratégias de Identificação

#### 1. Decomposição de Cholesky (Identificação Recursiva)

**Ideia:** Ordenar variáveis de modo que B seja triangular inferior.

**Exemplo (K=3):**
```
[ b11  0   0  ]
[ b21 b22  0  ]
[ b31 b32 b33 ]
```

**Interpretação:**
- Variável 1: afetada apenas pelo próprio choque
- Variável 2: afetada por choques 1 e 2 contemporaneamente
- Variável 3: afetada por todos os choques contemporaneamente

**Ordem importa!** Resultados dependem da ordenação.

**Exemplo econômico:**
Para [PIB, Inflação, Taxa de Juros]:
- Ordenação "recursiva": PIB afeta inflação e juros contemporaneamente, mas juros não afetam PIB contemporaneamente
- Justificativa: política monetária age com lag, mas PIB responde instantaneamente a outras forças

**Implementação:**
```python
irf_chol = result.irf(periods=10, method='cholesky')
```

#### 2. Generalized IRFs (Pesaran & Shin 1998)

**Ideia:** Não impor ordenação, mas integrar sobre históricos possíveis.

**Definição:**
```
GIRF(h) = E[y_t+h | u_j,t = σ_j, Ω_t-1] - E[y_t+h | Ω_t-1]
```

Onde:
- σ_j = desvio padrão do j-ésimo choque
- Ω_t-1 = informação no tempo t-1

**Vantagens:**
- Invariante à ordenação
- Fácil interpretação

**Desvantagens:**
- Choques não são ortogonais (captura efeitos diretos + correlações)
- Não é "estrutural" no sentido causal estrito

**Implementação:**
```python
irf_gen = result.irf(periods=10, method='generalized')
```

#### 3. Outras Identificações Estruturais

- **Sign restrictions:** Impor sinais esperados nas respostas (ex: choque monetário contracionário → inflação cai)
- **Long-run restrictions:** Certos choques não têm efeitos permanentes (Blanchard-Quah)
- **External instruments:** Usar variáveis instrumentais externas para identificar choques

### Impulse Response Functions (IRFs)

**Definição:**
IRF(h, j, k) = resposta da variável k no horizonte h a um choque de 1 desvio padrão na variável j no tempo 0.

**Interpretação:**
- **Magnitude:** Tamanho do efeito
- **Sinal:** Direção (positivo/negativo)
- **Persistência:** Quantos períodos até convergir a zero
- **Pico:** Horizonte de efeito máximo

**Forma fechada:**

Representação VMA(∞):
```
y_it = Σ(h=0 to ∞) Φ_h ε_i,t-h
```

Onde Φ_h são matrizes de coeficientes VMA.

**IRF Cholesky:**
```
IRF_chol(h) = Φ_h B
```

Onde B vem da decomposição de Cholesky de Σ.

**Intervalos de Confiança:**

Métodos:
1. **Analíticos (Delta method):** Rápido, mas assume normalidade
2. **Bootstrap:**
   - Residual bootstrap: Reamostrar resíduos
   - Pairs bootstrap: Reamostrar entidades inteiras
   - Mais robusto, mas computacionalmente intensivo

**Implementação:**
```python
irf_result = result.irf(
    periods=20,
    method='cholesky',
    ci_method='bootstrap',
    n_boot=1000,
    ci_level=0.95
)
```

---

## Forecast Error Variance Decomposition

**FEVD** responde: *Quão importante é cada choque para explicar flutuações em cada variável?*

### Definição

Para horizonte h, a fração da variância do erro de previsão da variável k explicada por choques na variável j é:

```
FEVD_k,j(h) = Σ(s=0 to h-1) (e_k' Φ_s B e_j)² / Σ(s=0 to h-1) (e_k' Φ_s Σ Φ_s' e_k)
```

Onde:
- **e_k:** Vetor de seleção para variável k
- **Φ_s:** Matriz de coeficientes VMA no lag s
- **B:** Matriz de impacto (da identificação)

### Propriedades

1. **Soma a 1:** Para cada variável k, Σ_j FEVD_k,j(h) = 1
2. **Dependente da ordenação (Cholesky):** Diferentes ordenações → diferentes FEVDs
3. **Invariante à ordenação (Generalized):** FEVD generalizado não soma exatamente a 1, mas é robusto

### Interpretação

**Exemplo:**
```
FEVD de PIB no horizonte 10:
- Choques próprios (PIB): 60%
- Choques de Inflação: 25%
- Choques de Juros: 15%
```

**Interpretação:**
Após 10 períodos, 60% da incerteza sobre PIB é devido a seus próprios choques, 25% a choques de inflação, e 15% a choques de juros.

**Usos:**
- Identificar variáveis mais "importantes" (exógenas vs endógenas)
- Avaliar dominância de choques (ex: choques de política vs tecnologia)
- Complementar análise de IRFs

**Implementação:**
```python
fevd_result = result.fevd(periods=20, method='cholesky')
print(fevd_result.fevd_matrix[9, :, :])  # Horizonte 10
```

---

## Causalidade de Granger em Painel

### Conceito

**Definição (Granger 1969):**
*x causa y no sentido de Granger se valores passados de x ajudam a prever y além do que valores passados de y já fazem.*

**No contexto Panel VAR:**
```
H0: A_l[k, j] = 0 para todo l = 1, ..., p
```

Onde A_l[k, j] é o coeficiente que liga y_j,t-l a y_k,t.

### Testes de Causalidade em Painel

#### 1. Teste de Wald (Pooled)

**Ideia:** Testar restrições conjuntas nos coeficientes do painel inteiro.

**Estatística:**
```
W = (R θ̂)' [R Var(θ̂) R']^-1 (R θ̂) ~ χ²(q)
```

Onde:
- R: matriz de restrições (q × número de parâmetros)
- θ̂: estimador GMM dos coeficientes
- q: número de restrições

**Hipóteses:**
- H0: x não causa y (todos os coefs de x nos lags de y são zero)
- H1: x causa y (pelo menos um coef ≠ 0)

**Implementação:**
```python
gc_result = result.granger_causality(cause='x', effect='y')
print(f"p-value: {gc_result.pvalue}")
```

#### 2. Teste Dumitrescu-Hurlin (2012)

**Ideia:** Testar causalidade permitindo heterogeneidade entre entidades.

**Modelo por entidade:**
```
y_i,t = α_i + Σ A_i,l y_i,t-l + Σ B_i,l x_i,t-l + ε_i,t
```

**Hipóteses:**
- H0: B_i,l = 0 para todo i (não causalidade em nenhuma entidade)
- H1: B_i,l ≠ 0 para alguns i (causalidade em algumas entidades)

**Estatística:**

Média das estatísticas Wald individuais:
```
W_HNC = (1/N) Σ W_i
```

Padronizar:
```
Z_HNC = √N (W_HNC - K) / √(2K) ~ N(0, 1)
```

**Vantagens:**
- Permite heterogeneidade
- Robusto a cross-section dependence (versão bootstrap)
- Maior poder em painéis com N grande

**Implementação:**
```python
from panelbox.var.causality import dumitrescu_hurlin_test

dh_result = dumitrescu_hurlin_test(
    data, cause='x', effect='y', lags=2, bootstrap=True, n_boot=500
)
print(f"DH test p-value: {dh_result.pvalue}")
```

### Cuidados com Causalidade de Granger

1. **Não é causalidade "verdadeira":** É causalidade *preditiva*, não *estrutural*
2. **Pode falhar com variáveis omitidas:** Se z causa x e y, pode parecer que x causa y
3. **Sensível à especificação:** Ordem de lags importa
4. **Bidirecionalidade é comum:** x → y e y → x simultaneamente (feedback)

---

## Panel VECM

### Motivação

Se variáveis são **I(1)** (não-estacionárias) mas **cointegradas**, usar Panel VAR em níveis é espúrio, e em diferenças perde informação de longo prazo.

**Solução:** Panel Vector Error Correction Model (VECM).

### Modelo Panel VECM

```
Δy_it = α_i + Π y_i,t-1 + Σ(l=1 to p-1) Γ_l Δy_i,t-l + ε_it
```

Onde:
- **Π = α β':** Matriz de cointegração (rank r < K)
  - **β:** Vetores de cointegração (relações de longo prazo)
  - **α:** Velocidades de ajustamento (loading matrix)
- **Γ_l:** Matrizes de dinâmicas de curto prazo

**Interpretação:**

- **β' y_i,t-1:** Desvios do equilíbrio de longo prazo
- **α:** Quão rápido se corrige o desequilíbrio
- **Γ_l Δy_i,t-l:** Dinâmicas de curto prazo

### Estimação

1. **Determinar rank de cointegração (r):**
   - Teste de Johansen em painel (Westerlund 2007)
   - Teste de Pedroni (1999, 2004)

2. **Estimar VECM:**
   - OLS por equação após impor rank
   - GMM se endogeneidade presente

**Implementação:**
```python
from panelbox.var import PanelVECM
from panelbox.tests.cointegration import pedroni_test

# Testar cointegração
coint_result = pedroni_test(data, endog_vars=['y1', 'y2', 'y3'])

if coint_result.reject:
    # Estimar VECM
    vecm = PanelVECM(data, endog_vars=['y1', 'y2', 'y3'],
                     entity_col='entity', time_col='time')

    # Estimar com rank = 1
    result = vecm.fit(rank=1, lags=2)

    # Acessar cointegrating vectors
    beta = result.beta  # Relações de longo prazo
    alpha = result.alpha  # Velocidades de ajustamento
```

### IRFs no VECM

IRFs em VECM separam efeitos:
- **Curto prazo:** Via Γ_l
- **Longo prazo:** Via α β'

**Tipos de IRF:**
1. **IRF acumulado:** Efeito sobre níveis
2. **IRF em diferenças:** Efeito sobre crescimento

---

## Comparações com Outros Métodos

### Panel VAR vs VAR Tradicional

| Aspecto | VAR Tradicional | Panel VAR |
|---------|----------------|-----------|
| **Dados** | N=1, T grande | N entidades, T moderado |
| **Heterogeneidade** | Não captura | Fixed effects |
| **Poder estatístico** | Limitado por T | N × T observações |
| **Uso típico** | Macro single-country | Cross-country, firmas |

### Panel VAR vs Arellano-Bond GMM

| Aspecto | Arellano-Bond | Panel VAR |
|---------|---------------|-----------|
| **Sistema** | Equação única | Sistema de equações |
| **Foco** | Estimar β em y = αy₋₁ + Xβ | Dinâmicas multivariadas |
| **Análise** | Efeitos de X em y | IRFs, FEVD, Granger |
| **Quando usar** | Modelo dinâmico univariado | Múltiplas endógenas |

### Panel VAR vs SVAR

| Aspecto | SVAR (Structural VAR) | Panel VAR |
|---------|----------------------|-----------|
| **Identificação** | Restrições estruturais complexas | Cholesky ou Generalized |
| **Foco** | Choques estruturais específicos | Dinâmicas gerais |
| **Dados** | Tipicamente time series (N=1) | Painel (N > 1) |

---

## Limitações e Cuidados

### 1. Homogeneidade de Slopes

**Pressuposto:** Matrizes A_l são iguais para todas as entidades.

**Problema:** Na prática, dinâmicas podem ser heterogêneas.

**Soluções:**
- Testar homogeneidade (ex: Pesaran & Smith 1995)
- Estimar subgrupos se heterogeneidade detectada
- Usar Mean Group estimators (permitir slopes heterogêneos e calcular média)

### 2. Cross-Section Dependence

**Problema:** ε_it pode ser correlacionado entre entidades (ex: choques globais).

**Consequências:**
- Erros padrão incorretos
- Viés nos coeficientes (casos severos)

**Soluções:**
- Adicionar time dummies (captura choques comuns)
- Usar estimadores robustos a cross-section dependence (Driscoll-Kraay)
- Common Correlated Effects (CCE) estimators

### 3. Proliferação de Instrumentos (GMM)

**Problema:** Com T moderado, número de instrumentos cresce rapidamente.

**Consequências:**
- Overfitting
- Hansen J perde poder
- Viés toward OLS

**Solução:**
- Usar "collapsed instruments" (Roodman 2009)
- Limitar max_lag_instruments
- Regra: # instrumentos < N

### 4. Estacionariedade

**Problema:** Panel VAR requer estacionariedade.

**Testes:**
- Levin-Lin-Chu (LLC)
- Im-Pesaran-Shin (IPS)
- Fisher-type tests

**Soluções se não-estacionário:**
- Diferenciar variáveis (perder info de longo prazo)
- Usar Panel VECM se cointegrado

### 5. Lag Order Selection

**Trade-off:**
- Poucos lags: especificação incorreta, autocorrelação residual
- Muitos lags: overfitting, perda de graus de liberdade

**Soluções:**
- Critérios de informação (AIC, BIC, HQIC)
- Testes de autocorrelação residual
- Considerar teoria econômica (ex: anual → p=1 ou 2)

### 6. Painéis Desbalanceados

**Problema:** Diferentes T_i para diferentes entidades.

**Consequências:**
- Perda de eficiência
- Complicações na construção de instrumentos (GMM)

**Soluções:**
- Panel VAR aceita desbalanceamento
- Usar First-Orthogonal Deviations (mais eficiente que FD)
- Remover entidades com T_i muito pequeno se necessário

---

## Workflow Recomendado

### Passo 1: Análise Exploratória

1. Plotar séries temporais por entidade
2. Verificar outliers e quebras estruturais
3. Calcular estatísticas descritivas

### Passo 2: Testes Preliminares

1. **Estacionariedade:**
   ```python
   from panelbox.tests.unit_root import panel_unit_root_test
   for var in variables:
       result = panel_unit_root_test(data[var], test='llc')
       print(f"{var}: {'I(0)' if result.reject else 'I(1)'}")
   ```

2. **Cointegração (se I(1)):**
   ```python
   from panelbox.tests.cointegration import pedroni_test
   coint_result = pedroni_test(data, endog_vars=variables)
   ```

### Passo 3: Seleção de Lag Order

```python
lag_selection = pvar.select_lag_order(max_lags=5, criterion='bic')
optimal_p = lag_selection.optimal_lag
```

### Passo 4: Estimação

```python
# OLS (baseline)
result_ols = pvar.fit(method='ols', lags=optimal_p)

# GMM (preferível se T pequeno)
result_gmm = pvar.fit(method='gmm', lags=optimal_p, transform='fod')
```

### Passo 5: Diagnósticos

```python
# Estabilidade
assert result_gmm.is_stable(), "VAR instável!"

# Hansen J
assert result_gmm.hansen_j_pvalue > 0.05, "Instrumentos inválidos!"

# AR tests
assert result_gmm.ar2_pvalue > 0.05, "AR(2) detectado - adicionar lags?"
```

### Passo 6: Análise

```python
# IRFs
irf = result_gmm.irf(periods=10, method='generalized', ci_method='bootstrap')
irf.plot()

# FEVD
fevd = result_gmm.fevd(periods=10)
fevd.plot()

# Granger causality
for cause in variables:
    for effect in variables:
        if cause != effect:
            gc = result_gmm.granger_causality(cause, effect)
            if gc.pvalue < 0.05:
                print(f"{cause} → {effect} (p={gc.pvalue:.4f})")
```

### Passo 7: Robustez

```python
# Comparar especificações
results_comparison = {
    'OLS': pvar.fit(method='ols', lags=2),
    'GMM-FOD': pvar.fit(method='gmm', transform='fod', lags=2),
    'GMM-FD': pvar.fit(method='gmm', transform='fd', lags=2),
    'lag=1': pvar.fit(method='gmm', lags=1),
    'lag=3': pvar.fit(method='gmm', lags=3),
}

# Comparar IRFs chave
for name, result in results_comparison.items():
    irf = result.irf(periods=10)
    # Plot e compare
```

---

## Referências

### Papers Fundamentais

1. **Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988).** Estimating vector autoregressions with panel data. *Econometrica*, 1371-1395.
   - Primeiro paper sobre Panel VAR e GMM

2. **Love, I., & Zicchino, L. (2006).** Financial development and dynamic investment behavior: Evidence from panel VAR. *The Quarterly Review of Economics and Finance*, 46(2), 190-210.
   - Aplicação seminal em finanças corporativas

3. **Abrigo, M. R., & Love, I. (2016).** Estimation of panel vector autoregression in Stata. *The Stata Journal*, 16(3), 778-804.
   - Implementação de referência e guia prático

### Econometria de Painel

4. **Arellano, M., & Bond, S. (1991).** Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. *The Review of Economic Studies*, 58(2), 277-297.
   - GMM para painéis dinâmicos

5. **Arellano, M., & Bover, O. (1995).** Another look at the instrumental variable estimation of error-components models. *Journal of Econometrics*, 68(1), 29-51.
   - System GMM e first-orthogonal deviations

6. **Roodman, D. (2009).** How to do xtabond2: An introduction to difference and system GMM in Stata. *The Stata Journal*, 9(1), 86-136.
   - Instrumentos colapsados e diagnósticos GMM

### VAR e Identificação

7. **Sims, C. A. (1980).** Macroeconomics and reality. *Econometrica*, 1-48.
   - VAR clássico

8. **Pesaran, H. H., & Shin, Y. (1998).** Generalized impulse response analysis in linear multivariate models. *Economics Letters*, 58(1), 17-29.
   - Generalized IRFs

9. **Lütkepohl, H. (2005).** *New introduction to multiple time series analysis.* Springer.
   - Livro-texto de referência sobre VAR

### Cointegração em Painel

10. **Pedroni, P. (1999).** Critical values for cointegration tests in heterogeneous panels with multiple regressors. *Oxford Bulletin of Economics and Statistics*, 61(S1), 653-670.

11. **Westerlund, J. (2007).** Testing for error correction in panel data. *Oxford Bulletin of Economics and Statistics*, 69(6), 709-748.

### Causalidade de Granger em Painel

12. **Dumitrescu, E. I., & Hurlin, C. (2012).** Testing for Granger non-causality in heterogeneous panels. *Economic Modelling*, 29(4), 1450-1460.

### Aplicações

13. **Canova, F., & Ciccarelli, M. (2013).** Panel vector autoregressive models: A survey. In *VAR Models in Macroeconomics–New Developments and Applications: Essays in Honor of Christopher A. Sims* (pp. 205-246). Emerald Group Publishing Limited.
   - Survey abrangente de Panel VAR

---

## Recursos Adicionais

### Tutoriais PanelBox

- [Tutorial Completo de Panel VAR](../tutorials/panel_var_complete_guide.md)
- [FAQ Panel VAR](../how-to/var_faq.md)
- [Troubleshooting](../how-to/troubleshooting.md)

### Software

- **PanelBox (Python):** Este pacote
- **Stata:** `pvar` package (Abrigo & Love)
- **R:** `panelvar`, `plm`, `vars`

### Datasets

- Penn World Tables: https://www.rug.nl/ggdc/productivity/pwt/
- OECD Data: https://data.oecd.org/
- World Bank: https://data.worldbank.org/

---

**Última atualização:** 2026-02-13

**Autores:** PanelBox Development Team

**Licença:** MIT
