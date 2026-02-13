# FASE 4 — IMPULSE RESPONSE FUNCTIONS E FEVD

## Visão Geral

Esta fase implementa Impulse Response Functions (IRFs) e Forecast Error Variance Decomposition (FEVD), que são as ferramentas centrais para análise dinâmica em modelos VAR. Inclui IRFs ortogonalizadas (Cholesky), generalizadas (Pesaran-Shin), intervalos de confiança bootstrap, FEVD, e visualizações publicáveis de alta qualidade.

**Duração:** 5 semanas (Semanas 13-17)
**Story Points:** 42
**Horas estimadas:** 105h
**Prioridade:** CRÍTICA - Core da análise dinâmica VAR
**Dependências:** FASE 1 completa (FASE 2 recomendada para GMM)

---

## Escopo Funcional

### Funcionalidades Entregues

1. **Impulse Response Functions Ortogonalizadas (Cholesky)**
   - Decomposição de Cholesky da matriz de covariância
   - IRFs para choques de 1 desvio padrão ou unitários
   - IRFs acumuladas para efeitos de longo prazo
   - Controle de ordenação das variáveis

2. **Impulse Response Functions Generalizadas (Pesaran-Shin)**
   - GIRFs invariantes à ordenação das variáveis
   - Mais robustas quando não há estrutura teórica clara
   - IRFs acumuladas generalizadas

3. **Intervalos de Confiança Bootstrap**
   - Bootstrap tipo 1 (Efron): reamostrar resíduos
   - Bootstrap tipo 2 (Hall): bias-corrected percentile
   - IC analíticos (delta method) como alternativa rápida
   - Paralelização para performance

4. **Forecast Error Variance Decomposition (FEVD)**
   - FEVD baseada em Cholesky
   - FEVD Generalizada (independente da ordem)
   - Normalização para somar 100%
   - IC bootstrap para FEVD

5. **Visualizações Publicáveis**
   - Grid K×K de IRFs com IC
   - Gráficos de FEVD (stacked area, stacked bar)
   - Temas: academic, professional, presentation
   - Export: PNG 300 DPI, SVG, PDF, HTML

---

## Lista de Tarefas Técnicas (Backlog)

### US-4.1 — IRFs Ortogonalizadas (Cholesky) (25h)

**Prioridade:** P0 - CRÍTICA

**Objetivo:** Implementar IRFs com ortogonalização via Cholesky.

#### Conceito Teórico

**Problema:** VAR estima correlações, mas IRFs precisam de choques "puros" (ortogonais)

**Solução:** Decomposição de Cholesky
```
Σ̂ = P·P'  (P é triangular inferior)

Choque ortogonal = P · [1, 0, 0, ...] = 1 desvio padrão na primeira variável
```

**Cálculo de IRF:**
```
Φₕ = coeficientes MA de ordem h
   = efeito no período h de um choque no período 0

Via companion matrix:
Φₕ = J · Cʰ · J' · P

onde:
- C = companion matrix (Kp × Kp)
- J = [I_K, 0, ..., 0] selector matrix (K × Kp)
- P = Cholesky de Σ̂
```

**Cálculo recursivo (alternativa mais eficiente):**
```
Φ₀ = P
Φₕ = Σₗ₌₁ʰ Aₗ·Φₕ₋ₗ  para h > 0
```

#### Implementação

- [x] Criar `panelbox/var/irf.py`
- [x] Criar classe `IRFResult` que armazena IRFs
- [x] Método `result.irf(periods=20, method='cholesky')`
- [x] Implementar decomposição de Cholesky: `P = np.linalg.cholesky(Sigma)`
- [x] Implementar cálculo recursivo de Φₕ
  ```python
  def compute_irf_cholesky(A_matrices, P, periods):
      K = P.shape[0]
      p = len(A_matrices)

      # Inicializar
      Phi = np.zeros((periods+1, K, K))
      Phi[0] = P  # Φ₀ = P

      # Recursão: Φₕ = Σₗ Aₗ·Φₕ₋ₗ
      for h in range(1, periods+1):
          for lag in range(1, min(h+1, p+1)):
              Phi[h] += A_matrices[lag-1] @ Phi[h-lag]

      return Phi  # shape: (periods+1, K, K)
  ```
- [x] Implementar via companion matrix (alternativa, para validação)
  ```python
  def compute_irf_companion(companion, P, periods, K):
      J = np.hstack([np.eye(K), np.zeros((K, K*(p-1)))])
      Phi = np.zeros((periods+1, K, K))
      C_power = np.eye(K*p)

      for h in range(periods+1):
          Phi[h] = J @ C_power @ J.T @ P
          C_power = C_power @ companion

      return Phi
  ```
- [x] IRFs acumuladas: `result.irf(periods=20, cumulative=True)`
  ```python
  Psi_h = Σₛ₌₀ʰ Φₛ  (soma acumulada)
  ```
- [x] Opção de ordenação: `result.irf(order=['x1', 'x3', 'x2'])`
  - Reordenar variáveis antes da Cholesky
  - Warning sobre dependência da ordem
- [x] Opção de tamanho do choque:
  - `shock_size='one_std'` (default): choque de 1 desvio padrão
  - `shock_size=1.0`: choque unitário
- [x] Classe `IRFResult`:
  ```python
  class IRFResult:
      def __init__(self, irf_matrix, var_names, periods, method):
          self.irf_matrix = irf_matrix  # (periods+1, K, K)
          self.var_names = var_names
          self.periods = periods
          self.method = method  # 'cholesky', 'generalized'

      def __getitem__(self, key):
          # irf_result['gdp']['inflation'] retorna array (periods+1,)
          response_idx = self.var_names.index(key[0])
          impulse_idx = self.var_names.index(key[1])
          return self.irf_matrix[:, response_idx, impulse_idx]

      def to_dataframe(self, impulse=None, response=None):
          # Retorna DataFrame com IRFs
          pass
  ```

#### Testes

- [x] VAR(1) K=2 — calcular IRFs manualmente e comparar
  ```python
  # Manual: Φ₁ = A₁·P, Φ₂ = A₁·Φ₁ = A₁²·P, etc.
  ```
- [x] IRF de variável i sobre ela mesma no h=0 deve ser √σᵢᵢ (diagonal de P)
- [x] IRFs convergem para zero se VAR é estável
  ```python
  assert np.allclose(irf_matrix[periods], 0, atol=1e-4)
  ```
- [x] IRFs acumuladas convergem para efeito de longo prazo finito
- [x] **VALIDAÇÃO R:** Comparar com `vars::irf()` — valores ± 1e-2
  - **NOTA:** R está disponível neste ambiente virtual (`/home/guhaase/projetos/panelbox/desenvolvimento/VAR`). Pacote `vars` do R instalado e validação completa.
- [x] Método recursivo ≡ método companion (validação interna)
- [x] Reordenação de variáveis muda IRFs (verificar que não são invariantes)

#### Pontos de Atenção

- Cholesky requer Σ̂ positiva definida. Se não for, usar regularização ou generalizada.
- Ordenação importa! Variável primeira é considerada mais "exógena" (responde apenas a choques próprios no h=0)
- Convergência: se max_eigenvalue_modulus ≥ 1, IRFs não convergem (divergem)

---

### US-4.2 — IRFs Generalizadas (Pesaran-Shin) (18h)

**Prioridade:** P1 - ALTA

**Objetivo:** Implementar GIRFs que não dependem da ordenação.

#### Conceito Teórico

**Problema da Cholesky:** Ordenação arbitrária das variáveis afeta resultados

**Solução de Pesaran & Shin (1998):**
- Não impor estrutura causal via ordenação
- Choque na variável j: deixar correlação contemporânea conforme Σ̂ observada

**Fórmula GIRF:**
```
GIRFⱼ(h) = (1/√σⱼⱼ) · Φₕ · Σ̂ · eⱼ

onde:
- Φₕ = coeficientes MA não-ortogonalizados
- eⱼ = vetor seletor (0,...,1,...,0) com 1 na posição j
- σⱼⱼ = elemento diagonal j de Σ̂
```

**Propriedades:**
- Invariante à ordenação das variáveis
- Choques não são ortogonais (refletem correlação real)
- Útil quando não há teoria clara sobre ordenação

#### Implementação

- [x] Método `result.irf(periods=20, method='generalized')`
- [x] Calcular Φₕ não-ortogonalizado (sem P, apenas A_l)
  ```python
  def compute_phi_non_orth(A_matrices, periods):
      # Similar ao Cholesky, mas Φ₀ = I (não P)
      K = A_matrices[0].shape[0]
      Phi = np.zeros((periods+1, K, K))
      Phi[0] = np.eye(K)

      for h in range(1, periods+1):
          for lag in range(1, min(h+1, len(A_matrices)+1)):
              Phi[h] += A_matrices[lag-1] @ Phi[h-lag]

      return Phi
  ```
- [x] Aplicar fórmula GIRF
  ```python
  def compute_girf(Phi, Sigma, periods):
      K = Sigma.shape[0]
      GIRF = np.zeros((periods+1, K, K))
      sigma_diag = np.sqrt(np.diag(Sigma))

      for h in range(periods+1):
          for j in range(K):
              e_j = np.zeros(K)
              e_j[j] = 1
              GIRF[h, :, j] = (Phi[h] @ Sigma @ e_j) / sigma_diag[j]

      return GIRF
  ```
- [x] IRFs acumuladas generalizadas
  ```python
  cumulative_GIRF = np.cumsum(GIRF, axis=0)
  ```
- [x] Reutilizar mesma classe `IRFResult`
- [x] Documentar diferença Cholesky vs Generalized claramente

#### Testes

- [x] Para VAR diagonal (Σ̂ diagonal), GIRF = Cholesky IRF (independente da ordem)
- [x] Para VAR não-diagonal, GIRF ≠ Cholesky
- [x] **Teste de invariância:** Permutar ordem das variáveis
  ```python
  girf_order1 = result.irf(method='generalized', order=['x1', 'x2', 'x3'])
  girf_order2 = result.irf(method='generalized', order=['x3', 'x1', 'x2'])
  assert np.allclose(girf_order1, girf_order2_reordered)
  ```
- [x] **VALIDAÇÃO R:** Comparar com `vars::irf(ortho=FALSE)` ajustado por Σ
  - **NOTA:** Pacote `vars` do R instalado e validação completa
- [x] Cholesky muda com ordem, GIRF não muda

---

### US-4.3 — Intervalos de Confiança Bootstrap para IRFs (20h)

**Prioridade:** P0 - CRÍTICA

**Objetivo:** Quantificar incerteza das IRFs via bootstrap.

#### Conceito

**Problema:** IRFs são funções não-lineares dos parâmetros → distribuição assintótica complexa

**Solução:** Bootstrap
1. Gerar B amostras bootstrap
2. Re-estimar VAR em cada uma
3. Calcular IRFs bootstrap
4. IC = percentis da distribuição bootstrap

#### Implementação

- [x] Método `result.irf(periods=20, ci_method='bootstrap', n_bootstrap=500)`
- [x] Parâmetro `ci_level=0.95` (default)
- [x] **Bootstrap tipo 1 (Efron):** Standard residual bootstrap
  ```python
  def bootstrap_irf_iteration(data, var_model, seed):
      np.random.seed(seed)

      # 1. Reamostrar resíduos
      n_obs = len(residuals)
      indices = np.random.choice(n_obs, size=n_obs, replace=True)
      resampled_residuals = residuals[indices]

      # 2. Reconstruir dados
      y_bootstrap = reconstruct_data(var_model.coefs, resampled_residuals)

      # 3. Re-estimar VAR
      var_boot = PanelVAR(y_bootstrap).fit()

      # 4. Calcular IRFs
      irf_boot = var_boot.irf(periods=periods, method=method)

      return irf_boot.irf_matrix
  ```
- [x] Reamostrar preservando estrutura de painel
  - Opção 1: reamostrar entidades inteiras (preserva dependência temporal)
  - Opção 2: reamostrar resíduos por entidade (mais comum)
  ```python
  for entity in entities:
      resid_entity = residuals[entity]
      indices = np.random.choice(len(resid_entity), replace=True)
      resampled[entity] = resid_entity[indices]
  ```
- [x] **Bootstrap tipo 2 (Hall):** Bias-corrected percentile bootstrap
  ```python
  # Calcular viés: bias = mean(irf_bootstrap) - irf_original
  # Ajustar percentis pelo viés
  z0 = norm.ppf(np.mean(irf_boot_dist < irf_original))
  alpha = ci_level
  lower_p = norm.cdf(2*z0 + norm.ppf(alpha/2))
  upper_p = norm.cdf(2*z0 + norm.ppf(1 - alpha/2))
  ci_lower = np.percentile(irf_boot_dist, 100*lower_p)
  ci_upper = np.percentile(irf_boot_dist, 100*upper_p)
  ```
- [x] **IC analíticos (delta method):** Como fallback rápido
  ```python
  # Baseado em Lütkepohl (2005) Section 3.7
  # Usar derivadas numéricas + matriz de covariância dos parâmetros
  ```
- [x] Paralelizar loop de bootstrap
  ```python
  from joblib import Parallel, delayed

  bootstrap_irfs = Parallel(n_jobs=-1)(
      delayed(bootstrap_irf_iteration)(data, model, seed=i)
      for i in range(n_bootstrap)
  )
  bootstrap_irfs = np.array(bootstrap_irfs)  # (n_boot, periods+1, K, K)
  ```
- [x] Progress bar
  ```python
  from tqdm import tqdm
  for i in tqdm(range(n_bootstrap), desc="Bootstrap IRF"):
      ...
  ```
- [x] Retornar em `IRFResult`:
  ```python
  self.ci_lower = np.percentile(bootstrap_dist, 100*(1-ci_level)/2, axis=0)
  self.ci_upper = np.percentile(bootstrap_dist, 100*(1-(1-ci_level)/2), axis=0)
  self.bootstrap_dist = bootstrap_dist  # (n_boot, periods+1, K, K)
  ```
- [x] Seed controlável para reprodutibilidade

#### Testes

- [x] IC bootstrap contém IRF verdadeira em ~95% das simulações Monte Carlo
  ```python
  # DGP conhecido, simular 100 vezes, contar cobertura
  coverage = np.mean([
      ci_lower <= true_irf <= ci_upper
      for _ in range(100)
  ])
  assert coverage > 0.90  # com tolerância
  ```
- [x] IC bootstrap ≥ IC analítico (tipicamente mais largos)
- [x] Bias-corrected IC ≥ percentile IC (ajuste aumenta intervalo)
- [x] Reprodutibilidade: mesmo seed = mesmo IC
- [x] **Performance:** 500 bootstrap para VAR(1) K=2 N=50 T=20 em < 120s

---

### US-4.4 — Forecast Error Variance Decomposition (FEVD) (16h)

**Prioridade:** P1 - ALTA

**Objetivo:** Decompor variância do erro de previsão por fonte de choque.

#### Conceito Teórico

**Pergunta:** Quanto da variância de x₁ em h períodos é devido a choques em x₂?

**FEVD baseada em Cholesky:**
```
ωᵢⱼ(h) = [Σₛ₌₀ʰ (eᵢ'·Φₛ·P·eⱼ)²] / [Σₛ₌₀ʰ (eᵢ'·Φₛ·Σ̂·Φₛ'·eᵢ)]

onde:
- Numerador: variância acumulada devida a choques em j
- Denominador: variância total acumulada de i
```

**FEVD Generalizada (Pesaran-Shin):**
```
ωᵢⱼ(h) = σⱼⱼ⁻¹ [Σₛ₌₀ʰ (eᵢ'·Φₛ·Σ̂·eⱼ)²] / [Σₛ₌₀ʰ (eᵢ'·Φₛ·Σ̂·Φₛ'·eᵢ)]

Normalizar: dividir por Σⱼ ωᵢⱼ(h) para somar 100%
```

**Propriedades:**
- Cholesky FEVD: soma exata 100% (by construction)
- Generalized FEVD: precisa normalizar para somar 100%

#### Implementação

- [x] Criar `panelbox/var/fevd.py`
- [x] Classe `FEVDResult`
- [x] Método `result.fevd(periods=20, method='cholesky')`
- [x] Implementar FEVD Cholesky
  ```python
  def compute_fevd_cholesky(Phi, P, periods):
      K = P.shape[0]
      FEVD = np.zeros((periods+1, K, K))

      for h in range(periods+1):
          for i in range(K):
              # Variância total acumulada de variável i
              total_var = 0
              for s in range(h+1):
                  total_var += (Phi[s][i, :] @ Sigma @ Phi[s][i, :].T)

              # Contribuição de cada choque j
              for j in range(K):
                  contrib = 0
                  for s in range(h+1):
                      # (eᵢ'·Φₛ·P·eⱼ)²
                      impulse_response = Phi[s][i, :] @ P[:, j]
                      contrib += impulse_response**2

                  FEVD[h, i, j] = contrib / total_var

      return FEVD  # shape: (periods+1, K, K), FEVD[h, i, j] = % var de i devido a j no horizonte h
  ```
- [x] Implementar FEVD Generalizada
  ```python
  def compute_fevd_generalized(Phi, Sigma, periods):
      K = Sigma.shape[0]
      FEVD_raw = np.zeros((periods+1, K, K))
      sigma_diag = np.diag(Sigma)

      for h in range(periods+1):
          for i in range(K):
              # Variância total
              total_var = 0
              for s in range(h+1):
                  total_var += (Phi[s][i, :] @ Sigma @ Phi[s][i, :].T)

              # Contribuição de cada choque j
              for j in range(K):
                  contrib = 0
                  e_j = np.zeros(K)
                  e_j[j] = 1
                  for s in range(h+1):
                      impulse = Phi[s][i, :] @ Sigma @ e_j
                      contrib += impulse**2

                  FEVD_raw[h, i, j] = contrib / sigma_diag[j] / total_var

          # Normalizar para somar 100% (GFEVD não soma automaticamente)
          for i in range(K):
              row_sum = FEVD_raw[h, i, :].sum()
              FEVD_raw[h, i, :] /= row_sum

      return FEVD_raw
  ```
- [x] Verificar que FEVD soma 100%
  ```python
  assert np.allclose(FEVD.sum(axis=2), 1.0)  # soma sobre choques = 1
  ```
- [x] Classe `FEVDResult`:
  ```python
  class FEVDResult:
      def __init__(self, decomposition, var_names, periods, method):
          self.decomposition = decomposition  # (periods+1, K, K)
          self.var_names = var_names
          self.periods = periods
          self.method = method

      def to_dataframe(self, variable, horizons=None):
          # Retorna DataFrame com FEVD de `variable` nos `horizons`
          # Colunas: fonte do choque, Linhas: horizonte
          pass

      def summary(self, horizons=[1, 5, 10, 20]):
          # Imprime tabela por variável e horizonte
          pass
  ```
- [x] IC bootstrap para FEVD: reutilizar distribuição bootstrap de IRFs (pending US-4.3)
  ```python
  # Calcular FEVD para cada IRF bootstrap
  fevd_bootstrap = [compute_fevd(irf_boot) for irf_boot in bootstrap_irfs]
  ci_lower = np.percentile(fevd_bootstrap, 2.5, axis=0)
  ci_upper = np.percentile(fevd_bootstrap, 97.5, axis=0)
  ```

#### Testes

- [x] FEVD soma 100% em todos os horizontes e variáveis
  ```python
  for h in range(periods+1):
      for i in range(K):
          assert np.isclose(FEVD[h, i, :].sum(), 1.0)
  ```
- [x] No horizonte 0 (Cholesky), primeira variável explica ~100% dela mesma
  ```python
  # Se ordem=['x1', 'x2'], então FEVD[0, 0, 0] ≈ 1.0
  ```
- [x] **VALIDAÇÃO R:** Comparar com `vars::fevd()` — valores ± 1e-3
  - **NOTA:** Pacote `vars` do R instalado e validação completa
- [x] GFEVD é invariante à ordenação das variáveis
- [x] Cholesky FEVD muda com ordenação

---

### US-4.5 — Visualizações de IRFs (14h)

**Prioridade:** P1 - ALTA

**Objetivo:** Gráficos profissionais e publicáveis de IRFs.

#### Especificação

**Layout:** Grid K×K de subplots
- Subplot (i, j): resposta de variável i a choque em variável j
- Total: K² subplots

**Elementos de cada subplot:**
- Linha sólida: IRF pontual
- Área sombreada ou linhas tracejadas: IC bootstrap
- Linha horizontal em y=0 (referência)
- Título: "Response of {y_i} to {y_j}"

#### Implementação

- [x] Método `irf_result.plot()` em classe `IRFResult`
- [x] Delegação para `panelbox/visualization/var_plots.py`
  ```python
  def plot_irf_grid(irf_result, ci=True, cumulative=False, **kwargs):
      K = len(irf_result.var_names)
      fig, axes = plt.subplots(K, K, figsize=(4*K, 3*K))

      for i in range(K):
          for j in range(K):
              ax = axes[i, j]

              # IRF pontual
              irf = irf_result.irf_matrix[:, i, j]
              ax.plot(irf, color='black', linewidth=2, label='IRF')

              # IC
              if ci and hasattr(irf_result, 'ci_lower'):
                  ci_low = irf_result.ci_lower[:, i, j]
                  ci_up = irf_result.ci_upper[:, i, j]
                  ax.fill_between(range(len(irf)), ci_low, ci_up,
                                   alpha=0.3, color='blue', label='95% CI')

              # Linha zero
              ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

              # Labels
              ax.set_title(f"{irf_result.var_names[i]} ← {irf_result.var_names[j]}")
              ax.set_xlabel('Horizon')
              ax.set_ylabel('Response')

              if i == 0 and j == 0:
                  ax.legend()

      plt.tight_layout()
      return fig
  ```
- [x] Opções de filtragem:
  - `impulse='x1'`: plotar apenas respostas ao choque em x1 (K subplots em coluna)
  - `response='x2'`: plotar apenas como x2 responde (K subplots em linha)
  - `variables=['x1', 'x3']`: grid 2×2 apenas para x1 e x3
- [x] Opção `cumulative=True` para plotar IRFs acumuladas
- [x] Suporte a backends:
  - Matplotlib (default)
  - Plotly (interativo)
- [x] Temas: `theme='academic'`, `'professional'`, `'presentation'`
  ```python
  themes = {
      'academic': {'font': 'serif', 'grid': True, 'color': 'black'},
      'professional': {'font': 'sans-serif', 'grid': False, 'color': 'blue'},
      'presentation': {'font': 'sans-serif', 'fontsize': 14, 'linewidth': 3}
  }
  ```
- [x] Export: `irf_result.plot().savefig('irfs.png', dpi=300)`
  - PNG 300 DPI
  - SVG (vetorial)
  - PDF
  - HTML (se Plotly)

#### Testes

- [x] VAR(1) K=2 — gera grid 2×2 correto
- [x] VAR(2) K=4 — gera grid 4×4 legível
- [x] Filtro `impulse='x1'` gera apenas K subplots (coluna)
- [x] Filtro `response='x2'` gera apenas K subplots (linha)
- [x] Export PNG e SVG sem erros
- [x] **Validação visual:** Comparar layout com `vars::plot(irf())` do R

---

### US-4.6 — Visualizações de FEVD (12h)

**Prioridade:** P1 - ALTA

**Objetivo:** Gráficos de decomposição da variância.

#### Especificação

**Layout:** K subplots (um por variável)
- Cada subplot: stacked area chart ou stacked bar
- Eixo X: horizonte (0, 1, ..., H)
- Eixo Y: percentual (0% a 100%)
- Cores distintas para cada fonte de choque

#### Implementação

- [x] Método `fevd_result.plot(kind='area')` em classe `FEVDResult`
- [x] Stacked area chart (default)
  ```python
  def plot_fevd_stacked_area(fevd_result):
      K = len(fevd_result.var_names)
      fig, axes = plt.subplots(K, 1, figsize=(10, 3*K))

      for i in range(K):
          ax = axes[i] if K > 1 else axes

          # Dados: FEVD[h, i, j] para todos j (fontes) ao longo de h
          data = fevd_result.decomposition[:, i, :]  # (periods+1, K)

          # Stacked area
          ax.stackplot(range(fevd_result.periods+1),
                       *[data[:, j] for j in range(K)],
                       labels=fevd_result.var_names,
                       alpha=0.8)

          ax.set_title(f"FEVD of {fevd_result.var_names[i]}")
          ax.set_xlabel('Horizon')
          ax.set_ylabel('Variance Share (%)')
          ax.set_ylim([0, 1])
          ax.legend(loc='upper left')

      plt.tight_layout()
      return fig
  ```
- [x] Stacked bar chart para horizontes selecionados
  ```python
  def plot_fevd_bar(fevd_result, horizons=[1, 5, 10, 20]):
      # Bar chart agrupado por horizonte
      # Cada grupo: K barras empilhadas (uma por variável)
      pass
  ```
  - Opção: `fevd_result.plot(kind='bar', horizons=[1, 5, 10, 20])`
- [x] Opção `variables=['x1', 'x2']` para filtrar
- [x] Cores consistentes (mesmo choque = mesma cor em todos os subplots)
- [x] Suporte a temas e backends
- [x] Export PNG, SVG, PDF, HTML

#### Testes

- [x] Soma visual = 100% (verificar que áreas empilhadas chegam a 1.0)
- [x] K=2 variáveis — layout correto
- [x] K=4 variáveis — layout legível com legenda
- [x] Bar chart renderiza para horizontes selecionados
- [x] Export funciona sem erros

---

## Ordem de Execução Sugerida

### Semana 13
1. US-4.1 parte 1: IRFs Cholesky — cálculo recursivo (3 dias)
2. US-4.1 parte 2: IRFs Cholesky — testes e validação (2 dias)

### Semana 14
3. US-4.2: IRFs Generalizadas (4 dias)
4. US-4.5 parte 1: Visualizações IRFs — básico (1 dia)

### Semana 15
5. US-4.3 parte 1: Bootstrap IRFs — implementação básica (3 dias)
6. US-4.3 parte 2: Paralelização e performance (2 dias)

### Semana 16
7. US-4.4: FEVD Cholesky e Generalizada (4 dias)
8. US-4.6: Visualizações FEVD (1 dia)

### Semana 17
9. US-4.5 parte 2: Visualizações IRFs — temas e export (2 dias)
10. US-4.3 parte 3: IC analíticos e bias-corrected (2 dias)
11. Integração, validação final e documentação (1 dia)

---

## Riscos Técnicos e Pontos de Atenção

### Riscos Críticos

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| **Bootstrap muito lento** | ALTA | MÉDIO | Paralelização agressiva, opção de IC analítico |
| **IRFs não convergem (VAR instável)** | BAIXA | MÉDIO | Warning claro, verificar estabilidade antes |
| **FEVD não soma 100% (bug numérico)** | BAIXA | ALTO | Testes rigorosos, normalização explícita |
| **Visualizações ilegíveis para K grande** | MÉDIA | MÉDIO | Opções de filtragem, layout adaptativo |
| **Divergência com R/Stata** | MÉDIA | ALTO | Validação incremental, múltiplos datasets |

### Pontos de Atenção Especiais

1. **Performance do Bootstrap**
   - 500 bootstrap × K equações × N entidades = muito tempo
   - Paralelizar com `joblib` (n_jobs=-1)
   - Considerar bootstrap adaptativo (parar quando convergir)
   - Opção de reduzir n_bootstrap para testes rápidos

2. **Convergência de IRFs**
   - Se VAR instável (eigenvalue ≥ 1), IRFs divergem
   - Verificar estabilidade antes de calcular IRFs
   - Warning claro: "VAR is unstable, IRFs do not converge"

3. **FEVD: Normalização**
   - Cholesky: soma 100% by construction
   - Generalized: **deve normalizar** manualmente
   - Verificar sempre após cálculo

4. **Visualizações para K grande**
   - Grid K×K fica ilegível para K > 5
   - Sugerir filtros: `impulse='x1'` ou `response='x2'`
   - Layout alternativo: subplot apenas para pares significativos

5. **Ordenação Cholesky**
   - Documentar claramente que ordenação importa
   - Sugerir ordenação baseada em teoria econômica
   - Exemplo: variáveis mais exógenas primeiro (policy vars no fim)

---

## Critérios de Aceitação da Fase

A Fase 4 está **COMPLETA** quando:

- [x] Todas as 6 user stories implementadas e testadas
- [x] IRFs Cholesky funcionais com cálculo recursivo e companion
- [x] IRFs Generalizadas funcionais e invariantes à ordenação
- [x] Bootstrap IRFs funcional e paralelizado
- [x] Performance: 500 bootstrap < 120s para VAR(1) K=2 N=50 T=20
- [x] IC bootstrap contêm valor verdadeiro ~95% em Monte Carlo
- [x] IC analíticos implementados como fallback
- [x] Bias-corrected bootstrap implementado
- [x] FEVD Cholesky e Generalizada funcionais
- [x] FEVD soma 100% (validado em testes)
- [x] IC bootstrap para FEVD funcionais
- [x] Visualizações IRFs: grid K×K renderiza
- [x] Visualizações FEVD: stacked area e bar funcionam
- [x] Export PNG 300 DPI, SVG, PDF funciona
- [x] Temas (academic, professional) aplicam corretamente
- [x] **VALIDAÇÃO R:** IRFs ± 1e-2 vs `vars::irf()`
- [x] **VALIDAÇÃO R:** FEVD ± 1e-3 vs `vars::fevd()`
- [x] Cobertura de testes ≥ 85%
- [x] Todos os testes passando
- [x] Documentação completa
- [x] Exemplo de workflow IRF/FEVD funcional

---

## Entregáveis

### Código
- `panelbox/var/irf.py` — IRFResult, cálculo de IRFs
- `panelbox/var/fevd.py` — FEVDResult, cálculo de FEVD
- Extensão de `panelbox/var/bootstrap.py` — Bootstrap IRF/FEVD
- Extensão de `panelbox/visualization/var_plots.py` — Plots IRF e FEVD

### Testes
- `tests/var/test_irf_cholesky.py` — Testes IRF Cholesky
- `tests/var/test_irf_generalized.py` — Testes IRF Generalized
- `tests/var/test_irf_bootstrap.py` — Testes bootstrap
- `tests/var/test_fevd.py` — Testes FEVD
- `tests/var/test_irf_plots.py` — Testes visualizações
- `tests/validation/test_irf_vs_r.py` — Validação R

### Documentação
- `docs/api/var_irf.md` — API de IRFs
- `docs/api/var_fevd.md` — API de FEVD
- `docs/theory/irf_fevd_theory.md` — Teoria IRF/FEVD
- `examples/var/impulse_response_analysis.py` — Exemplo completo IRF
- `examples/var/variance_decomposition.py` — Exemplo FEVD

---

## Referências Técnicas

### Papers Fundamentais

**IRFs:**
- Sims (1980) — "Macroeconomics and Reality" (introdução de VAR e IRF)
- Lütkepohl (2005) — "New Introduction to Multiple Time Series Analysis" (referência técnica)

**IRFs Generalizadas:**
- Pesaran & Shin (1998) — "Generalized Impulse Response Analysis in Linear Multivariate Models"
- Koop, Pesaran & Potter (1996) — "Impulse Response Analysis in Nonlinear Multivariate Models"

**Bootstrap:**
- Kilian (1998) — "Small-Sample Confidence Intervals for Impulse Response Functions"
- Brüggemann et al. (2016) — "Inference in VARs with Conditional Heteroskedasticity"

**FEVD:**
- Lütkepohl (2005) — Chapter 2 (FEVD theory)
- Pesaran & Shin (1998) — (Generalized FEVD)

### Software de Referência
- R: `vars::irf()`, `vars::fevd()`
- EViews: VAR impulse response tools

---

## Glossário Técnico

- **IRF (Impulse Response Function):** Resposta dinâmica de uma variável a um choque em outra
- **Cholesky decomposition:** Fatoração Σ = P·P' para ortogonalizar choques
- **GIRF (Generalized IRF):** IRF que não depende da ordenação das variáveis
- **FEVD (Forecast Error Variance Decomposition):** Decomposição da variância do erro de previsão
- **Bootstrap:** Reamostragem para inferência não-paramétrica
- **Bias-corrected bootstrap:** Ajuste do bootstrap para viés de pequenas amostras
- **Delta method:** Método analítico para calcular variância de funções não-lineares
- **Companion matrix:** Representação VAR(p) como VAR(1) de dimensão maior

---

**Próxima Fase:** FASE 5 — Panel VECM (Vector Error Correction Model)
