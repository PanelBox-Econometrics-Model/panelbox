# Sign Convention em Stochastic Frontier Analysis

## Visão Geral

A **sign convention** (convenção de sinal) é um aspecto crucial — mas frequentemente confuso — na especificação de modelos de fronteira estocástica. A convenção define como o termo de ineficiência $u$ entra no modelo, afetando diretamente a interpretação dos resultados e o cálculo das eficiências.

Este documento esclarece as convenções usadas no PanelBox SFA e fornece orientações sobre qual usar em diferentes contextos.

---

## Resumo Executivo

| Tipo de Fronteira | Modelo | Eficiência | Intervalo | Interpretação |
|-------------------|--------|------------|-----------|---------------|
| **Produção** | $y = X\beta + v - u$ | $TE = e^{-u}$ | $(0, 1]$ | 1 = eficiente, <1 = ineficiente |
| **Custo** | $y = X\beta + v + u$ | $CE = e^{-u}$ | $(0, 1]$ | 1 = eficiente, <1 = ineficiente |

**Nota importante:** Em ambos os casos, $u \geq 0$ representa **ineficiência**, mas com sinais opostos no modelo.

---

## 1. Fronteira de Produção (Production Frontier)

### 1.1 Modelo

$$
y_i = X_i \beta + v_i - u_i
$$

onde:
- $y_i = \log(\text{output}_i)$: logaritmo do produto
- $X_i \beta$: fronteira de produção (máximo teórico)
- $v_i \sim N(0, \sigma_v^2)$: ruído idiossincrático (simétrico)
- $u_i \geq 0$: ineficiência técnica (one-sided, assimétrico)

### 1.2 Interpretação

- **Fronteira**: $X_i \beta$ representa o output **máximo** alcançável com os insumos $X_i$
- **Ineficiência**: $u_i > 0$ **reduz** o output abaixo da fronteira
- **Sinal negativo** de $u$: firmas ineficientes produzem **menos** que a fronteira

### 1.3 Eficiência Técnica (TE)

$$
TE_i = \frac{y_i^{\text{observado}}}{y_i^{\text{fronteira}}} = e^{-u_i}
$$

**Propriedades:**
- $TE_i \in (0, 1]$
- $TE_i = 1$: firma **eficiente** (na fronteira, $u_i = 0$)
- $TE_i < 1$: firma **ineficiente** (abaixo da fronteira, $u_i > 0$)

**Exemplo:**
- $TE_i = 0.85$ significa que a firma produz 85% do output máximo possível
- A firma poderia **aumentar** a produção em $(1 - 0.85)/0.85 = 17.6\%$ sem mudar insumos

### 1.4 Uso no PanelBox

```python
from panelbox.frontier import StochasticFrontier

sf = StochasticFrontier(
    data=df,
    depvar='log_output',
    exog=['log_labor', 'log_capital'],
    frontier='production',  # Fronteira de PRODUÇÃO
    dist='half_normal'
)

result = sf.fit()

# Eficiências técnicas (0, 1]
eff = result.efficiency(estimator='bc')
print(f"Mean TE: {eff['te'].mean():.3f}")  # ex: 0.850
```

---

## 2. Fronteira de Custo (Cost Frontier)

### 2.1 Modelo

$$
y_i = X_i \beta + v_i + u_i
$$

onde:
- $y_i = \log(\text{cost}_i)$: logaritmo do custo
- $X_i \beta$: fronteira de custo (mínimo teórico)
- $v_i \sim N(0, \sigma_v^2)$: ruído idiossincrático
- $u_i \geq 0$: ineficiência de custo (one-sided)

### 2.2 Interpretação

- **Fronteira**: $X_i \beta$ representa o custo **mínimo** para produzir $q$ com preços $p$
- **Ineficiência**: $u_i > 0$ **aumenta** o custo acima da fronteira
- **Sinal positivo** de $u$: firmas ineficientes gastam **mais** que a fronteira

### 2.3 Eficiência de Custo (CE)

Há **duas** convenções possíveis para CE:

#### Opção 1: $CE = e^{-u_i}$ (Recomendada — PanelBox Default)

$$
CE_i = e^{-u_i} \in (0, 1]
$$

**Vantagens:**
- Mesma escala que TE (comparável diretamente)
- $CE = 1$: eficiente
- $CE < 1$: ineficiente

**Interpretação:**
- $CE_i = 0.80$ significa que a firma poderia **reduzir** custos em $(1 - 0.80) = 20\%$

#### Opção 2: $CE = e^{u_i}$ (Alternativa)

$$
CE_i = \frac{\text{cost}^{\text{observado}}}{\text{cost}^{\text{mínimo}}} = e^{u_i} \in [1, \infty)
$$

**Vantagens:**
- Razão direta: custo observado / custo mínimo
- $CE = 1$: eficiente
- $CE > 1$: ineficiente

**Interpretação:**
- $CE_i = 1.25$ significa que a firma gasta 25% **a mais** que o mínimo

**Conversão:**
- $CE_{\text{opção 1}} = 1 / CE_{\text{opção 2}}$
- Exemplo: $CE = 0.80 \Leftrightarrow CE = 1.25$

### 2.4 Uso no PanelBox

```python
sf = StochasticFrontier(
    data=df,
    depvar='log_cost',
    exog=['log_output', 'log_price'],
    frontier='cost',  # Fronteira de CUSTO
    dist='half_normal'
)

result = sf.fit()

# Eficiências de custo (0, 1] — Opção 1 (default)
eff = result.efficiency(estimator='bc')
print(f"Mean CE: {eff['te'].mean():.3f}")  # ex: 0.750

# Converter para Opção 2 se desejado (razão)
eff['ce_ratio'] = 1 / eff['te']
print(f"Mean CE (ratio): {eff['ce_ratio'].mean():.3f}")  # ex: 1.333
```

---

## 3. Comparação: Produção vs Custo

| Aspecto | Produção | Custo |
|---------|----------|-------|
| **Modelo** | $y = X\beta + v - u$ | $y = X\beta + v + u$ |
| **Fronteira** | Máximo (output) | Mínimo (custo) |
| **Sinal de $u$** | Negativo | Positivo |
| **Ineficiência** | Reduz output | Aumenta custo |
| **Eficiência** | $TE = e^{-u}$ | $CE = e^{-u}$ ou $e^u$ |
| **Skewness OLS** | Negativa | Positiva |
| **Plot** | Fronteira ACIMA | Fronteira ABAIXO |

### 3.1 Skewness dos Resíduos

A **skewness** dos resíduos OLS é um teste informal para verificar se a fronteira está corretamente especificada:

- **Produção**: Resíduos devem ter skewness **negativa** (cauda à esquerda)
  - Ineficiência ($-u < 0$) desloca observações para baixo

- **Custo**: Resíduos devem ter skewness **positiva** (cauda à direita)
  - Ineficiência ($+u > 0$) desloca observações para cima

**Se a skewness tiver o sinal errado:**
1. Fronteira especificada incorretamente (produção vs custo)
2. Outliers distorcendo a distribuição
3. Especificação funcional incorreta (ex: Cobb-Douglas vs translog)
4. Não há ineficiência (modelo inadequado)

### 3.2 Visualização

```
Fronteira de Produção:                 Fronteira de Custo:

    Fronteira ────────────              ─────────────── Fronteira
          │                                       │
       ↓  │                                       │ ↑
          │  Ineficiência                         │  Ineficiência
          │  (u > 0)                              │  (u > 0)
          ●                                       ●
          ●  Pontos                               ●  Pontos
          ●  observados                           ●  observados
          ●                                       ●
```

---

## 4. Armadilhas Comuns (Common Pitfalls)

### 4.1 Wrong Skewness

**Problema:** Resíduos OLS têm skewness com sinal errado.

**Causas:**
- Fronteira especificada incorretamente (produção vs custo)
- Outliers extremos
- Distribuição incorreta (ex: half-normal vs exponential)

**Solução:**
1. Verificar tipo de fronteira (`frontier='production'` ou `'cost'`)
2. Remover outliers (robustez)
3. Testar distribuições alternativas
4. Verificar especificação funcional (linearidade em logs)

**Exemplo:**
```python
from panelbox.frontier import skewness_test

# Testar skewness
result = skewness_test(
    y=df['log_output'].values,
    X=df[['log_labor', 'log_capital']].values,
    frontier_type='production'
)

print(f"Skewness: {result['skewness']:.4f}")
print(f"Expected: negative for production")

if result['p_value'] > 0.05:
    print("WARNING: Skewness not significant — may indicate no inefficiency")
```

### 4.2 Confundir CE Scales

**Problema:** Interpretar $CE = e^{-u}$ como $CE = e^u$ (ou vice-versa).

**Solução:** Sempre verificar documentação do software:
- **PanelBox**: usa $CE = e^{-u} \in (0, 1]$ (default)
- **R frontier**: usa $CE = e^{-u}$ (mesma escala)
- **Stata sfpanel**: depende da opção

**Verificação rápida:**
```python
eff = result.efficiency()

# Se CE está em (0, 1]: escala exp(-u)
print(f"Min CE: {eff['te'].min():.3f}")  # ex: 0.432
print(f"Max CE: {eff['te'].max():.3f}")  # ex: 0.987

# Se algum CE > 1: escala exp(u)
if (eff['te'] > 1).any():
    print("WARNING: CE > 1 detected — check scale")
```

### 4.3 Fronteira de Custo com Sinal Errado

**Problema:** Estimar fronteira de custo com `frontier='production'`.

**Sintomas:**
- Eficiências muito altas (> 0.95)
- Skewness positiva quando se esperava negativa
- Resultados não fazem sentido economicamente

**Solução:**
```python
# ERRADO:
sf = StochasticFrontier(
    data=df,
    depvar='log_cost',  # Custo
    exog=['log_output', 'log_price'],
    frontier='production',  # ❌ ERRO!
    dist='half_normal'
)

# CORRETO:
sf = StochasticFrontier(
    data=df,
    depvar='log_cost',  # Custo
    exog=['log_output', 'log_price'],
    frontier='cost',  # ✅ Correto
    dist='half_normal'
)
```

---

## 5. Validação: Como Saber se Está Correto?

### 5.1 Checklist

- [ ] **Skewness tem o sinal correto?**
  - Produção: negativa
  - Custo: positiva

- [ ] **Eficiências estão no intervalo esperado?**
  - $(0, 1]$ se usando $e^{-u}$
  - $[1, \infty)$ se usando $e^u$

- [ ] **Média de eficiência é razoável?**
  - Muito alta (> 0.98): pode indicar pouca ineficiência
  - Muito baixa (< 0.50): verificar outliers ou especificação

- [ ] **Visualização faz sentido?**
  - Fronteira de produção: ACIMA dos pontos
  - Fronteira de custo: ABAIXO dos pontos

### 5.2 Exemplo de Validação

```python
import numpy as np
from scipy import stats

# 1. Estimar modelo
result = sf.fit()

# 2. Verificar skewness
residuals_ols = result.residuals_ols
skew = stats.skew(residuals_ols)

if sf.frontier_type == 'production':
    expected_sign = "negative"
    assert skew < 0, f"Skewness should be negative, got {skew:.4f}"
elif sf.frontier_type == 'cost':
    expected_sign = "positive"
    assert skew > 0, f"Skewness should be positive, got {skew:.4f}"

print(f"✓ Skewness is {expected_sign}: {skew:.4f}")

# 3. Verificar eficiências
eff = result.efficiency()
mean_eff = eff['te'].mean()
min_eff = eff['te'].min()
max_eff = eff['te'].max()

assert np.all(eff['te'] > 0), "All efficiencies must be > 0"
assert np.all(eff['te'] <= 1), "All efficiencies must be ≤ 1 (exp(-u) scale)"

print(f"✓ Efficiencies in (0, 1]: mean={mean_eff:.3f}, "
      f"min={min_eff:.3f}, max={max_eff:.3f}")

# 4. Teste formal de presença de ineficiência
from panelbox.frontier import inefficiency_presence_test

ineff_test = inefficiency_presence_test(result)

if ineff_test['p_value'] < 0.05:
    print(f"✓ Inefficiency is present (p-value={ineff_test['p_value']:.4f})")
else:
    print(f"⚠ WARNING: No significant inefficiency detected "
          f"(p-value={ineff_test['p_value']:.4f})")
```

---

## 6. Referências

### Papers Seminais

- **Aigner, D., Lovell, C. K., & Schmidt, P. (1977).** Formulation and estimation of stochastic frontier production function models. *Journal of Econometrics*, 6(1), 21-37.
  - Define sign convention para fronteira de produção ($-u$)

- **Christensen, L. R., & Greene, W. H. (1976).** Economies of scale in U.S. electric power generation. *Journal of Political Economy*, 84(4), 655-676.
  - Aplica SFA para fronteira de custo ($+u$)

### Livros

- **Kumbhakar, S. C., & Lovell, C. K. (2000).** *Stochastic Frontier Analysis.* Cambridge University Press.
  - Cap. 2: Detailed discussion of sign conventions

- **Greene, W. H. (2008).** Econometric Analysis (6th ed.). Pearson.
  - Seção sobre SFA: esclarece convenções

### Software Documentation

- **R frontier**: https://cran.r-project.org/web/packages/frontier/
  - Usa $TE = e^{-u}$ para ambos (produção e custo)

- **Stata sfpanel**: https://www.stata.com/features/overview/stochastic-frontier-models/
  - Opções para sign convention

- **PanelBox SFA**: Esta documentação
  - Consistente com `frontier` (R)

---

## 7. FAQ

**Q: Por que a sign convention é importante?**

**A:** O sinal de $u$ afeta diretamente:
1. Interpretação de eficiências
2. Skewness esperada dos resíduos
3. Posição da fronteira (acima vs abaixo dos pontos)
4. Cálculo de eficiências

Usar o sinal errado produz resultados incorretos e interpretações econômicas absurdas.

---

**Q: O que fazer se a skewness tem o sinal errado?**

**A:**
1. Verificar se `frontier='production'` ou `'cost'` está correto
2. Remover outliers extremos
3. Testar especificação funcional alternativa (ex: translog)
4. Considerar que pode não haver ineficiência (usar teste formal)

---

**Q: Como converter entre as duas escalas de CE?**

**A:**
- De $CE = e^{-u}$ para $CE = e^u$: use `1 / CE`
- De $CE = e^u$ para $CE = e^{-u}$: use `1 / CE`

Exemplo:
```python
# PanelBox (exp(-u))
eff = result.efficiency()
ce_exp_minus_u = eff['te']  # 0.80

# Converter para exp(u)
ce_exp_u = 1 / ce_exp_minus_u  # 1.25
```

---

**Q: Qual escala de CE é melhor?**

**A:** Ambas são válidas, mas $CE = e^{-u} \in (0, 1]$ é recomendada porque:
- Mesma escala que TE (comparável)
- Consistente com literatura moderna
- Interpretação mais intuitiva (1 = melhor)

PanelBox usa $e^{-u}$ como default para consistência com `frontier` (R).

---

**Última atualização:** 2026-02-15
**Autor:** PanelBox Development Team
