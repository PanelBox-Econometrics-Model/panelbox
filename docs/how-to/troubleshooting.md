# Panel VAR - Guia de Troubleshooting Avançado

Este guia fornece soluções detalhadas para problemas técnicos avançados ao trabalhar com Panel VAR.

---

## Diagnósticos GMM

### Problema: Hansen J Test Rejeita Consistentemente

**Sintomas:**
```python
result = pvar.fit(method='gmm', lags=2)
print(result.hansen_j_pvalue)  # < 0.05 (rejeita)
```

**Diagnóstico Detalhado:**

1. **Verificar overidentification:**
```python
n_instruments = result.n_instruments
n_params = result.n_params
print(f"Grau de overidentification: {n_instruments - n_params}")
# Se > 50, pode ser problema de "too many instruments"
```

2. **Testar instrumentos colapsados:**
```python
result_collapsed = pvar.fit(
    method='gmm',
    instruments='collapsed',
    max_lag_instruments=2
)
print(result_collapsed.hansen_j_pvalue)
```

3. **Comparar transformações:**
```python
# FOD (First-Orthogonal Deviations)
result_fod = pvar.fit(method='gmm', transform='fod')

# FD (First Differences)
result_fd = pvar.fit(method='gmm', transform='fd')

print(f"FOD Hansen J: {result_fod.hansen_j_pvalue}")
print(f"FD Hansen J: {result_fd.hansen_j_pvalue}")
```

**Soluções Avançadas:**

1. **Instrumentos adaptativos:**
```python
# Começar conservador
for max_lag in range(2, 6):
    result = pvar.fit(
        method='gmm',
        instruments='collapsed',
        max_lag_instruments=max_lag
    )
    print(f"Max lag {max_lag}: Hansen J p-value = {result.hansen_j_pvalue}")
    if result.hansen_j_pvalue > 0.05:
        print(f"✓ Aceito com max_lag={max_lag}")
        break
```

2. **Two-step GMM com weight matrix robusta:**
```python
result = pvar.fit(
    method='gmm',
    gmm_type='twostep',
    robust_weights=True
)
```

---

## Problemas de Estabilidade

### Problema: VAR Instável com Explosão de IRFs

**Sintomas:**
```python
result.is_stable()  # False
irf = result.irf(periods=20)
irf.plot()  # IRFs explodem exponencialmente
```

**Diagnóstico:**

1. **Examinar eigenvalues:**
```python
eigenvalues = result.eigenvalues()
moduli = np.abs(eigenvalues)
print("Eigenvalues moduli:")
for i, mod in enumerate(moduli):
    status = "✓" if mod < 1 else "✗"
    print(f"  {status} λ_{i}: {mod:.4f}")

# Identificar problemas
max_modulus = moduli.max()
if max_modulus > 1:
    print(f"Maior autovalor: {max_modulus:.4f} > 1 → INSTÁVEL")
```

2. **Testar estacionariedade:**
```python
from panelbox.tests.unit_root import panel_unit_root_test

for var in endog_vars:
    # LLC test
    llc_result = panel_unit_root_test(data[var], test='llc')

    # IPS test
    ips_result = panel_unit_root_test(data[var], test='ips')

    print(f"{var}:")
    print(f"  LLC: {'I(0)' if llc_result.reject else 'I(1)'} (p={llc_result.pvalue:.4f})")
    print(f"  IPS: {'I(0)' if ips_result.reject else 'I(1)'} (p={ips_result.pvalue:.4f})")
```

**Soluções:**

1. **Diferenciar variáveis I(1):**
```python
# Criar primeiras diferenças
data_diff = data.copy()
for var in endog_vars:
    data_diff[f'd_{var}'] = data.groupby('entity')[var].diff()

# Estimar VAR em diferenças
pvar_diff = PanelVAR(
    data_diff.dropna(),
    endog_vars=[f'd_{var}' for var in endog_vars],
    entity_col='entity',
    time_col='time'
)
result_diff = pvar_diff.fit(lags=2)
```

2. **Usar VECM se cointegrado:**
```python
from panelbox.tests.cointegration import pedroni_test
from panelbox.var import PanelVECM

# Testar cointegração
coint_result = pedroni_test(data, endog_vars=endog_vars)

if coint_result.reject:
    # Variáveis cointegradas → usar VECM
    vecm = PanelVECM(data, endog_vars=endog_vars, entity_col='entity', time_col='time')
    vecm_result = vecm.fit(rank=1, lags=1)  # rank = número de relações de cointegração
```

3. **Reduzir lags:**
```python
# Testar lags menores
for p in range(1, 4):
    result = pvar.fit(lags=p)
    if result.is_stable():
        print(f"✓ Estável com p={p}")
        break
```

---

## Problemas de Convergência

### Problema: GMM Não Converge

**Sintomas:**
```
RuntimeWarning: GMM estimation did not converge within max_iter iterations
```

**Diagnóstico:**

1. **Verificar condicionamento da matriz:**
```python
import numpy as np

# Matriz de momentos
X = result._X_matrix  # Dados internos
cond_number = np.linalg.cond(X.T @ X)
print(f"Condition number: {cond_number:.2e}")

if cond_number > 1e10:
    print("✗ Matriz mal condicionada!")
```

**Soluções:**

1. **Aumentar max_iter e ajustar tolerância:**
```python
result = pvar.fit(
    method='gmm',
    max_iter=500,
    tol=1e-6  # menos estrito que padrão 1e-8
)
```

2. **Escalar variáveis:**
```python
from sklearn.preprocessing import StandardScaler

# Padronizar variáveis
scaler = StandardScaler()
data_scaled = data.copy()
data_scaled[endog_vars] = scaler.fit_transform(data[endog_vars])

# Estimar com dados escalados
pvar_scaled = PanelVAR(data_scaled, endog_vars=endog_vars, ...)
result = pvar_scaled.fit(method='gmm')

# Converter coeficientes de volta à escala original se necessário
```

3. **Usar starting values do OLS:**
```python
# Primeiro estimar OLS
result_ols = pvar.fit(method='ols', lags=2)

# Usar como ponto de partida para GMM
result_gmm = pvar.fit(
    method='gmm',
    lags=2,
    start_params=result_ols.params
)
```

---

## Problemas com Intervalos de Confiança

### Problema: ICs Bootstrap Assimétricos ou Estranhos

**Sintomas:**
```python
irf_result = result.irf(periods=10, ci_method='bootstrap', n_boot=500)
irf_result.plot()  # ICs cruzam zero de forma estranha, ou são muito assimétricos
```

**Diagnóstico:**

1. **Verificar distribuição bootstrap:**
```python
# Acessar distribuição bootstrap (se disponível)
boot_irfs = irf_result._bootstrap_distribution  # (n_boot, periods, K, K)

# Examinar distribuição para impulse específico
import matplotlib.pyplot as plt

impulse_idx = 0
response_idx = 1
period = 5

boot_samples = boot_irfs[:, period, response_idx, impulse_idx]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(boot_samples, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(np.median(boot_samples), color='red', label='Median')
plt.axvline(np.mean(boot_samples), color='blue', label='Mean')
plt.legend()
plt.title(f'Bootstrap Distribution (period {period})')

plt.subplot(1, 2, 2)
from scipy import stats
stats.probplot(boot_samples, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.tight_layout()
plt.show()
```

2. **Testar diferentes métodos de IC:**
```python
# Percentil
irf_pct = result.irf(ci_method='bootstrap', ci_type='percentile')

# BC (bias-corrected)
irf_bc = result.irf(ci_method='bootstrap', ci_type='bc')

# BCa (bias-corrected and accelerated)
irf_bca = result.irf(ci_method='bootstrap', ci_type='bca')
```

**Soluções:**

1. **Aumentar n_boot para mais estabilidade:**
```python
irf_result = result.irf(
    periods=10,
    ci_method='bootstrap',
    n_boot=2000,  # ao invés de 500
    random_state=42  # para reprodutibilidade
)
```

2. **Usar bootstrap residual ao invés de pairs:**
```python
irf_result = result.irf(
    ci_method='bootstrap',
    bootstrap_type='residual'  # ao invés de 'pairs'
)
```

3. **Verificar estabilidade do modelo:**
```python
if not result.is_stable():
    print("⚠ Modelo instável - isto pode causar ICs estranhos")
    print("Considere re-especificar o modelo")
```

---

## Problemas de Dados

### Problema: Painel Altamente Desbalanceado

**Sintomas:**
```python
entity_counts = data.groupby('entity').size()
print(entity_counts.describe())
# std muito grande, min << max
```

**Diagnóstico:**

```python
import matplotlib.pyplot as plt

# Visualizar distribuição
entity_counts = data.groupby('entity').size()

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
entity_counts.hist(bins=30, edgecolor='black')
plt.xlabel('Number of periods (T)')
plt.ylabel('Number of entities')
plt.title('Distribution of Panel Lengths')

plt.subplot(1, 2, 2)
entity_counts.sort_values().plot()
plt.xlabel('Entity (sorted)')
plt.ylabel('Number of periods')
plt.title('Sorted Panel Lengths')

plt.tight_layout()
plt.show()

# Estatísticas
print(f"Min T: {entity_counts.min()}")
print(f"Max T: {entity_counts.max()}")
print(f"Mean T: {entity_counts.mean():.1f}")
print(f"Median T: {entity_counts.median():.1f}")
```

**Soluções:**

1. **Filtrar entidades com T muito pequeno:**
```python
min_periods = 10  # ou outro threshold razoável

entity_counts = data.groupby('entity').size()
valid_entities = entity_counts[entity_counts >= min_periods].index

data_filtered = data[data['entity'].isin(valid_entities)]

print(f"Entidades removidas: {len(entity_counts) - len(valid_entities)}")
print(f"Entidades mantidas: {len(valid_entities)}")
```

2. **Estratificação por tamanho:**
```python
# Dividir em grupos por tamanho
entity_counts = data.groupby('entity').size()

small = entity_counts[entity_counts < 15].index
medium = entity_counts[(entity_counts >= 15) & (entity_counts < 25)].index
large = entity_counts[entity_counts >= 25].index

# Estimar separadamente e comparar
for name, entities in [('Small', small), ('Medium', medium), ('Large', large)]:
    if len(entities) > 20:  # Mínimo de entidades
        data_subset = data[data['entity'].isin(entities)]
        pvar_subset = PanelVAR(data_subset, ...)
        result = pvar_subset.fit()
        print(f"{name}: N={len(entities)}, coefs={result.params[:3]}")
```

3. **Weighted estimation (se disponível):**
```python
# Dar mais peso a entidades com mais observações
weights = data.groupby('entity').size().to_dict()
data['weight'] = data['entity'].map(weights)

# (Se implementado)
result = pvar.fit(weights='weight')
```

### Problema: Outliers Extremos

**Sintomas:**
```python
data[endog_vars].describe()
# max >> mean, ou min << mean
```

**Diagnóstico:**

```python
import numpy as np
import matplotlib.pyplot as plt

def detect_outliers(series, method='iqr', threshold=3):
    """Detecta outliers usando IQR ou z-score"""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outliers = (series < lower) | (series > upper)
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = z_scores > threshold

    return outliers

# Para cada variável
for var in endog_vars:
    outliers = detect_outliers(data[var], method='iqr', threshold=3)
    n_outliers = outliers.sum()
    pct_outliers = 100 * n_outliers / len(data)

    print(f"{var}: {n_outliers} outliers ({pct_outliers:.1f}%)")

    if n_outliers > 0:
        # Visualizar
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        data[var].hist(bins=50, edgecolor='black')
        plt.title(f'{var} - Histogram')

        plt.subplot(1, 2, 2)
        plt.boxplot(data[var].dropna())
        plt.title(f'{var} - Boxplot')

        plt.tight_layout()
        plt.show()
```

**Soluções:**

1. **Winsorização:**
```python
from scipy.stats import mstats

# Winsorizar a 1% e 99%
data_winsor = data.copy()
for var in endog_vars:
    data_winsor[var] = mstats.winsorize(data[var], limits=[0.01, 0.01])

# Estimar com dados winsorized
pvar_winsor = PanelVAR(data_winsor, endog_vars=endog_vars, ...)
result = pvar_winsor.fit()
```

2. **Remoção de outliers:**
```python
# Remover observações com outliers extremos
data_clean = data.copy()
for var in endog_vars:
    outliers = detect_outliers(data[var], threshold=5)  # threshold alto
    data_clean = data_clean[~outliers]

print(f"Observações removidas: {len(data) - len(data_clean)}")
```

3. **Transformação logarítmica:**
```python
# Se variáveis são positivas e skewed
data_log = data.copy()
for var in endog_vars:
    if (data[var] > 0).all():
        data_log[f'log_{var}'] = np.log(data[var])

# Estimar com logs
pvar_log = PanelVAR(
    data_log,
    endog_vars=[f'log_{var}' for var in endog_vars],
    ...
)
```

---

## Problemas de Performance

### Problema: Bootstrap Muito Lento

**Sintomas:**
```python
# Demora > 10 minutos
irf_result = result.irf(periods=20, ci_method='bootstrap', n_boot=1000)
```

**Soluções:**

1. **Paralelização:**
```python
# Se implementado
irf_result = result.irf(
    periods=20,
    ci_method='bootstrap',
    n_boot=1000,
    n_jobs=-1  # usar todos os cores
)
```

2. **Reduzir parâmetros:**
```python
# Menos replicações
irf_result = result.irf(periods=10, ci_method='bootstrap', n_boot=500)
```

3. **Bootstrap estratificado:**
```python
# Bootstrap menos períodos primeiro para verificar
irf_quick = result.irf(periods=5, ci_method='bootstrap', n_boot=200)
irf_quick.plot()

# Se resultados razoáveis, então fazer completo
if input("Continuar com bootstrap completo? (y/n): ").lower() == 'y':
    irf_full = result.irf(periods=20, ci_method='bootstrap', n_boot=1000)
```

---

## Problemas de Interpretação

### Problema: IRFs com Sinais Inesperados

**Sintomas:**
```python
# Esperava-se resposta negativa de GDP a choque de juros, mas é positiva
irf = result.irf(periods=10)
irf.plot(impulse='interest_rate', response='gdp')
# Mostra efeito positivo!
```

**Diagnóstico:**

1. **Verificar ordenação Cholesky:**
```python
# A ordem importa!
print(f"Ordem das variáveis: {result.endog_names}")

# Tentar ordem diferente
pvar_reorder = PanelVAR(
    data,
    endog_vars=['interest_rate', 'inflation', 'gdp'],  # ordem alterada
    ...
)
result_reorder = pvar_reorder.fit()
irf_reorder = result_reorder.irf(periods=10, method='cholesky')
irf_reorder.plot(impulse='interest_rate', response='gdp')
```

2. **Usar Generalized IRFs (ordem-invariante):**
```python
irf_gen = result.irf(periods=10, method='generalized')
irf_gen.plot(impulse='interest_rate', response='gdp')
```

3. **Verificar dados:**
```python
# Correlações surpreendentes?
print(data[['interest_rate', 'gdp']].corr())

# Scatter plot
import matplotlib.pyplot as plt
plt.scatter(data['interest_rate'], data['gdp'], alpha=0.5)
plt.xlabel('Interest Rate')
plt.ylabel('GDP')
plt.show()
```

**Soluções:**

1. **Justificar ordenação teoricamente** e usar Cholesky

2. **Usar Generalized IRFs** se ordenação não clara

3. **Verificar especificação do modelo:**
   - Variáveis omitidas?
   - Quebras estruturais?
   - Variáveis não-estacionárias?

---

## Checklist de Debugging Completo

Quando tudo mais falhar, siga este checklist sistemático:

### 1. Dados
- [ ] Painel balanceado ou desbalanceamento documentado?
- [ ] Sem missing values nas variáveis chave?
- [ ] Outliers identificados e tratados?
- [ ] Variáveis nas unidades corretas?
- [ ] Entity e time corretamente especificados?

### 2. Especificação
- [ ] Variáveis estacionárias (ou VECM se cointegradas)?
- [ ] Lag order justificado (critérios ou teoria)?
- [ ] Exógenas verdadeiramente exógenas?
- [ ] Transformação (FOD/FD) apropriada?

### 3. Estimação
- [ ] Método (OLS/GMM) apropriado dado N e T?
- [ ] Se GMM: Hansen J e AR tests ok?
- [ ] Convergência alcançada?
- [ ] Coeficientes em magnitude razoável?

### 4. Diagnósticos
- [ ] Modelo estável (eigenvalues < 1)?
- [ ] Resíduos sem autocorrelação?
- [ ] Heterocedasticidade razoável?

### 5. Inferência
- [ ] ICs incluem zero quando esperado?
- [ ] IRFs fazem sentido economicamente?
- [ ] Causalidade consistente com teoria?
- [ ] Resultados robustos a especificações alternativas?

---

## Quando Pedir Ajuda

Se após seguir este guia o problema persistir:

1. **Prepare exemplo reprodutível:**
```python
import pandas as pd
import numpy as np
from panelbox.var import PanelVAR

# Dados mínimos que reproduzem o problema
np.random.seed(42)
data = ...  # seu código aqui

# Código que causa erro
pvar = PanelVAR(...)
result = pvar.fit(...)
```

2. **Documente ambiente:**
```python
import panelbox
import pandas as pd
import numpy as np

print(f"PanelBox version: {panelbox.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
```

3. **Abra issue no GitHub** com:
   - Descrição do problema
   - Exemplo reprodutível
   - Mensagem de erro completa
   - O que já tentou

---

**Última atualização:** 2026-02-13
