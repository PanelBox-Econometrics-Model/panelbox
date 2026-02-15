# Planejamento de Jupyter Notebooks de Validação - PanelBox

**Data:** 2026-02-15
**Objetivo:** Criar uma suíte completa de Jupyter notebooks para validação, demonstração e documentação da biblioteca PanelBox.

---

## 📊 Resumo Executivo

A biblioteca PanelBox implementa **50+ modelos econométricos** em 9 categorias principais. Este planejamento organiza **60+ notebooks** de validação que cobrem:

- ✅ Validação contra R/Stata
- ✅ Exemplos com dados reais
- ✅ Testes de diagnóstico
- ✅ Visualizações interativas
- ✅ Casos de uso aplicados
- ✅ Tutoriais introdutórios
- ✅ Guias avançados

---

## 🎯 Estrutura de Pastas

```
examples/
├── 01_intro/                    # Notebooks introdutórios
├── 02_discrete/                 # Modelos de escolha discreta
├── 03_count/                    # Modelos de contagem
├── 04_quantile/                 # Regressão quantílica
├── 05_spatial/                  # Econometria espacial
├── 06_dynamic/                  # Modelos dinâmicos (GMM, VAR, VECM)
├── 07_censored_selection/       # Modelos censurados e seleção
├── 08_diagnostics/              # Testes de diagnóstico
├── 09_visualization/            # Visualizações avançadas
├── 10_advanced/                 # Tópicos avançados
├── 11_validation_r/             # Validação contra R
├── 12_validation_stata/         # Validação contra Stata
└── datasets/                    # Datasets compartilhados
```

---

## 📚 Notebooks por Categoria

### 🌟 PRIORIDADE 1: Notebooks Introdutórios (01_intro/)

#### NB01.1: Introdução ao PanelBox
**Arquivo:** `intro_panelbox_overview.ipynb`
**Objetivo:** Visão geral da biblioteca, instalação, conceitos básicos
**Conteúdo:**
- Instalação e setup
- Filosofia da biblioteca
- Comparação com outras bibliotecas (linearmodels, statsmodels)
- Estrutura de dados esperada (MultiIndex)
- Workflow básico: fit → diagnóstico → visualização
- Hello World: Modelo de efeitos fixos simples

**Dataset:** Grunfeld capital investment data
**Validação:** N/A (introdutório)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB01.2: Preparação de Dados para Painéis
**Arquivo:** `data_preparation_panel_format.ipynb`
**Objetivo:** Como preparar dados para uso com PanelBox
**Conteúdo:**
- Criar MultiIndex (entity, time)
- Balancear/desbalancear painéis
- Tratamento de missing values
- Lags e diferenças
- Merge de dados de painel
- Exploração descritiva de painéis

**Dataset:** PSID, Compustat (exemplos múltiplos)
**Validação:** N/A (tutorial)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB01.3: Modelos Estáticos Básicos de Painel
**Arquivo:** `intro_static_panel_models.ipynb`
**Objetivo:** Comparação dos 5 estimadores estáticos básicos
**Conteúdo:**
- PooledOLS vs FixedEffects vs RandomEffects
- Teste de Hausman
- Between estimator
- First Difference
- Quando usar cada um?
- Interpretação de coeficientes

**Dataset:** Wage panel (Baltagi)
**Validação:** plm (R), xtreg (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

### 🎲 PRIORIDADE 2: Modelos de Escolha Discreta (02_discrete/)

#### NB02.1: Logit e Probit Básicos para Painéis
**Arquivo:** `discrete_logit_probit_basic.ipynb`
**Objetivo:** Introdução aos modelos binários de painel
**Conteúdo:**
- PooledLogit vs ConditionalLogit
- PooledProbit vs RandomEffectsProbit
- Efeitos marginais (AME, MEM)
- Interpretação de odds ratios
- Teste de razão de verossimilhança

**Dataset:** Union membership (NLS), Health insurance (PSID)
**Validação:** glm (R), xtlogit/xtprobit (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB02.2: Modelos Logit/Probit com Efeitos Fixos
**Arquivo:** `discrete_fixed_effects.ipynb`
**Objetivo:** Estimadores que lidam com efeitos fixos não observados
**Conteúdo:**
- ConditionalLogit (Chamberlain)
- Probit correlated random effects
- Problema do parâmetro incidental
- Mundlak device
- Comparação de abordagens

**Dataset:** Patent applications (Blundell-Griffith-Van Reenen)
**Validação:** clogit (Stata), survival::clogit (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB02.3: Modelos Logit/Probit com Efeitos Aleatórios
**Arquivo:** `discrete_random_effects.ipynb`
**Objetivo:** Estimação com efeitos aleatórios via GHQ
**Conteúdo:**
- RandomEffectsLogit (integração via Gauss-Hermite)
- RandomEffectsProbit
- Escolha do número de pontos de quadratura
- Comparação com aproximação de Laplace
- Correlação intra-classe

**Dataset:** Doctor visits (SOEP)
**Validação:** melogit/meprobit (Stata), glmer (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB02.4: Modelos Logit/Probit Dinâmicos
**Arquivo:** `discrete_dynamic_models.ipynb`
**Objetivo:** Dependência de estado em modelos binários
**Conteúdo:**
- DynamicLogit / DynamicProbit
- Problema do estado inicial (Heckman, Wooldridge)
- Estimação via GMM
- Efeitos de persistência vs heterogeneidade
- APE (Average Partial Effects)

**Dataset:** Female labor force participation (PSID)
**Validação:** xtdpdml (Stata), dplyr (R)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB02.5: Modelos Ordered Logit/Probit
**Arquivo:** `discrete_ordered_models.ipynb`
**Objetivo:** Variáveis dependentes ordinais
**Conteúdo:**
- OrderedLogit pooled vs RE
- OrderedProbit pooled vs RE
- Teste de odds proporcionais
- Thresholds e interpretação
- Efeitos marginais para cada categoria

**Dataset:** Job satisfaction (GSOEP), Self-rated health (NLSY)
**Validação:** MASS::polr (R), ologit/oprobit (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB02.6: Modelos Multinomial Logit
**Arquivo:** `discrete_multinomial_models.ipynb`
**Objetivo:** Escolhas entre 3+ alternativas
**Conteúdo:**
- MultinomialLogit pooled
- ConditionalLogit (McFadden)
- Mixed logit (random parameters)
- IIA (Independence of Irrelevant Alternatives)
- Teste de Hausman-McFadden
- Elasticidades de escolha

**Dataset:** Mode choice (Greene), Occupation choice (Keane-Wolpin)
**Validação:** mlogit (R), asclogit (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB02.7: Validação Completa - Discrete Choice
**Arquivo:** `discrete_validation_complete.ipynb`
**Objetivo:** Validação cruzada com R e Stata para todos os modelos discretos
**Conteúdo:**
- Comparação de coeficientes
- Comparação de erros padrão
- Comparação de efeitos marginais
- Tolerâncias numéricas
- Benchmarks de performance

**Dataset:** Multiple (Union, Patent, Doctor visits)
**Validação:** R (glm, glmer, survival, mlogit) + Stata (xtlogit, xtprobit, clogit)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

### 📊 PRIORIDADE 2: Modelos de Contagem (03_count/)

#### NB03.1: Poisson Básico para Painéis
**Arquivo:** `count_poisson_basic.ipynb`
**Objetivo:** Introdução aos modelos de contagem
**Conteúdo:**
- PooledPoisson
- FixedEffectsPoisson (Hausman-Hall-Griliches)
- RandomEffectsPoisson
- Teste de sobredispersão (Cameron-Trivedi)
- Efeitos marginais (IRR - Incidence Rate Ratios)

**Dataset:** Doctor visits (SOEP), Patents (HGR)
**Validação:** MASS::glm.nb (R), xtpoisson (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB03.2: Negative Binomial para Painéis
**Arquivo:** `count_negative_binomial.ipynb`
**Objetivo:** Lidar com sobredispersão
**Conteúdo:**
- NegativeBinomial (NB2)
- Comparação com Poisson
- Estimação do parâmetro de dispersão
- Teste de sobredispersão
- RE vs FE

**Dataset:** Recreational trips (Cameron-Trivedi), Citations (HGR)
**Validação:** glm.nb (R), xtnbreg (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB03.3: Zero-Inflated Models
**Arquivo:** `count_zero_inflated.ipynb`
**Objetivo:** Excesso de zeros
**Conteúdo:**
- ZeroInflatedPoisson (ZIP)
- ZeroInflatedNegativeBinomial (ZINB)
- Teste de Vuong
- Interpretação de dois processos
- Efeitos marginais compostos

**Dataset:** Biochemical oxygen demand (Greene), Fishing (Zeileis)
**Validação:** pscl::zeroinfl (R), zip/zinb (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB03.4: Hurdle Models
**Arquivo:** `count_hurdle_models.ipynb`
**Objetivo:** Modelar participação vs intensidade
**Conteúde:**
- HurdlePoisson
- HurdleNegativeBinomial
- Comparação com ZIP/ZINB
- Interpretação de dois estágios
- Elasticidades

**Dataset:** Health care utilization (Mullahy)
**Validação:** pscl::hurdle (R), churdle (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB03.5: Validação Completa - Count Models
**Arquivo:** `count_validation_complete.ipynb`
**Objetivo:** Validação cruzada para modelos de contagem
**Conteúdo:**
- Comparação PanelBox vs R vs Stata
- Todos os modelos: Poisson, NB, ZIP, ZINB, Hurdle
- Coeficientes, SEs, efeitos marginais
- Performance benchmarks

**Dataset:** Multiple (Doctor visits, Patents, Recreation)
**Validação:** R (MASS, pscl) + Stata (xtpoisson, xtnbreg, zip)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

### 📈 PRIORIDADE 1: Regressão Quantílica (04_quantile/)

#### NB04.1: Introdução à Regressão Quantílica em Painéis
**Arquivo:** `quantile_intro_panel_qr.ipynb`
**Objetivo:** Conceitos fundamentais de QR para painéis
**Conteúdo:**
- Motivação: por que QR em painéis?
- Quantile regression pooled (Koenker-Bassett)
- Interpretação de coeficientes quantílicos
- Comparação com OLS
- Visualização de coeficientes por quantil

**Dataset:** Wage data (Buchinsky), Income inequality (PSID)
**Validação:** quantreg::rq (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB04.2: Quantile Regression com Efeitos Fixos (Canay)
**Arquivo:** `quantile_canay_fixed_effects.ipynb`
**Objetivo:** Estimador de Canay (2011)
**Conteúdo:**
- CanayQuantileRegression
- Two-step procedure
- Identificação de efeitos fixos
- Propriedades assintóticas
- Simulações Monte Carlo

**Dataset:** Earnings dynamics (NLSY)
**Validação:** qregpd (Stata), quantreg (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB04.3: Location-Scale Models
**Arquivo:** `quantile_location_scale.ipynb`
**Objetivo:** Modelar heteroscedasticidade condicional
**Conteúdo:**
- LocationScaleQuantile
- Interpretação de efeitos em location e scale
- Testes de heteroscedasticidade
- Comparação com modelos GARCH

**Dataset:** Returns data, Wage volatility (CPS)
**Validação:** quantreg::rqss (R)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB04.4: Dynamic Quantile Regression
**Arquivo:** `quantile_dynamic_models.ipynb`
**Objetivo:** Modelos quantílicos com dependência temporal
**Conteúdo:**
- DynamicQuantileRegression
- Persistência quantílica
- Efeitos dinâmicos heterogêneos
- Value-at-Risk dinâmico

**Dataset:** Asset returns, Income dynamics (PSID)
**Validação:** N/A (método recente)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB04.5: Quantile Treatment Effects
**Arquivo:** `quantile_treatment_effects.ipynb`
**Objetivo:** Efeitos heterogêneos de tratamento
**Conteúdo:**
- QuantileTreatmentEffects
- QTE vs ATE
- Distribuição contrafactual
- Decomposições de desigualdade
- Robustez a outliers

**Dataset:** Job training (LaLonde), Education returns (Card)
**Validação:** quantreg::rq + manual computation
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB04.6: Monotonicity e Non-Crossing Constraints
**Arquivo:** `quantile_non_crossing.ipynb`
**Objetivo:** Garantir que quantis não se cruzem
**Conteúdo:**
- MonotonicQuantileRegression
- Métodos de imposição de restrições
- Visualização de crossing violations
- Soluções: rearranjo, penalização, restrições

**Dataset:** Wage data (multiple)
**Validação:** quantreg::rearrangement (R)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB04.7: Comparação de Métodos de QR em Painéis
**Arquivo:** `quantile_methods_comparison.ipynb`
**Objetivo:** Comparar diferentes estimadores de QR
**Conteúdo:**
- QuantileComparison framework
- Pooled vs Canay vs Location-Scale
- Simulações Monte Carlo
- Viés e eficiência
- Recomendações práticas

**Dataset:** Simulado + Wage data
**Validação:** Multiple packages
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB04.8: Validação Completa - Quantile Regression
**Arquivo:** `quantile_validation_complete.ipynb`
**Objetivo:** Validação contra R para todos os modelos quantílicos
**Conteúdo:**
- Comparação com quantreg (R)
- Comparação com qregpd (Stata)
- Coeficientes, SEs, ICs
- Performance em grandes amostras

**Dataset:** Multiple
**Validação:** R (quantreg) + Stata (qregpd, xtqreg)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

### 🗺️ PRIORIDADE 1: Econometria Espacial (05_spatial/)

#### NB05.1: Introdução à Econometria Espacial em Painéis
**Arquivo:** `spatial_intro_panel_spatial.ipynb`
**Objetivo:** Conceitos fundamentais de econometria espacial
**Conteúdo:**
- Matrizes de pesos espaciais (W)
- Tipos de dependência: lag, erro, Durbin
- Autocorrelação espacial (Moran's I)
- Visualização de padrões espaciais

**Dataset:** US counties crime, European regions convergence
**Validação:** spdep (R), spmat (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB05.2: Spatial Autoregressive Model (SAR/Spatial Lag)
**Arquivo:** `spatial_sar_model.ipynb`
**Objetivo:** Modelo de lag espacial
**Conteúdo:**
- SpatialAutoregressive
- Interpretação de ρ (rho)
- Efeitos diretos vs indiretos (spillovers)
- ML vs GMM vs 2SLS
- Decomposição de efeitos totais

**Dataset:** Cigarette sales (US states), House prices (Boston)
**Validação:** spatialreg::lagsarlm (R), xsmle (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB05.3: Spatial Error Model (SEM)
**Arquivo:** `spatial_error_model.ipynb`
**Objetivo:** Modelo de erro espacial
**Conteúdo:**
- SpatialErrorModel
- Interpretação de λ (lambda)
- Diferença entre SAR e SEM
- Testes de especificação (LM tests)
- Quando usar SEM vs SAR?

**Dataset:** Crime data, Agricultural productivity
**Validação:** spatialreg::errorsarlm (R), xsmle (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB05.4: Spatial Durbin Model (SDM)
**Arquivo:** `spatial_durbin_model.ipynb`
**Objetivo:** Modelo com lag espacial em Y e X
**Conteúdo:**
- SpatialDurbin
- WX (exogenous spatial lags)
- Efeitos diretos, indiretos, totais
- Teste de restrições comuns (SAR, SEM)
- LeSage-Pace decomposition

**Dataset:** Regional growth (EU NUTS), Pollution spillovers
**Validação:** spatialreg::lagsarlm (R), xsmle (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB05.5: General Nesting Spatial (GNS) Model
**Arquivo:** `spatial_general_nesting.ipynb`
**Objetivo:** Modelo espacial geral
**Conteúdo:**
- GeneralNestingSpatial
- Combina SAR + SEM + SDM
- Estratégia de specific-to-general
- Testes LR para simplificação
- Seleção de modelo

**Dataset:** Multi-country panel
**Validação:** sphet (R), xsmle (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB05.6: Dynamic Spatial Panel Models
**Arquivo:** `spatial_dynamic_models.ipynb`
**Objetivo:** Modelos espaciais com dinâmica temporal
**Conteúdo:**
- DynamicSpatial
- Lag temporal + lag espacial
- Estimação via GMM (Kukenova-Monteiro)
- Efeitos de curto vs longo prazo
- Multiplicadores espaciais dinâmicos

**Dataset:** Investment spillovers, FDI flows
**Validação:** spgm (R), xtdpdml + spatial (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB05.7: Matrizes de Pesos Espaciais
**Arquivo:** `spatial_weight_matrices.ipynb`
**Objetivo:** Criação e manipulação de matrizes W
**Conteúdo:**
- SpatialWeights class
- Contiguidade (rook, queen)
- Distância (k-nearest, threshold)
- Econômica (trade, migration)
- Normalização (row, spectral)
- Visualização de conexões

**Dataset:** US states, EU regions, Cities
**Validação:** spdep::nb2mat (R), spmat (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB05.8: Diagnósticos Espaciais
**Arquivo:** `spatial_diagnostics.ipynb`
**Objetivo:** Testes de autocorrelação e especificação
**Conteúdo:**
- Moran's I, Geary's C
- LM tests (LMlag, LMerr, Robust LM)
- Teste de Hausman espacial
- Identificação de outliers espaciais
- LISA (Local Indicators of Spatial Association)

**Dataset:** Crime, Unemployment
**Validação:** spdep (R), spatdiag (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB05.9: Spatial HAC Standard Errors
**Arquivo:** `spatial_hac_standard_errors.ipynb`
**Objetivo:** Erros padrão robustos a correlação espacial
**Conteúdo:**
- Conley (1999) spatial HAC
- Escolha de bandwidth (cutoff distance)
- Kernel functions (uniform, triangle, Bartlett)
- Comparação com clustering

**Dataset:** Agricultural data, Environmental data
**Validação:** spatialreg (R), acreg (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB05.10: Validação Completa - Spatial Models
**Arquivo:** `spatial_validation_complete.ipynb`
**Objetivo:** Validação contra R/Stata para modelos espaciais
**Conteúdo:**
- SAR, SEM, SDM, GNS comparisons
- Coeficientes, efeitos diretos/indiretos
- Standard errors
- Performance benchmarks

**Dataset:** Multiple spatial datasets
**Validação:** R (spatialreg, spdep) + Stata (xsmle, spmat)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

### ⚡ PRIORIDADE 2: Modelos Dinâmicos (06_dynamic/)

#### NB06.1: Introdução a Painéis Dinâmicos
**Arquivo:** `dynamic_intro_panel_dynamics.ipynb`
**Objetivo:** Conceitos fundamentais de painéis dinâmicos
**Conteúdo:**
- Por que GMM?
- Viés de Nickell
- Instrumentos válidos (Anderson-Hsiao)
- Condições de momento
- Endogeneidade do lag

**Dataset:** Employment dynamics (Arellano-Bond)
**Validação:** plm::pgmm (R), xtabond (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB06.2: Difference GMM (Arellano-Bond)
**Arquivo:** `dynamic_difference_gmm.ipynb`
**Objetivo:** Estimador de diferenças GMM
**Conteúdo:**
- DifferenceGMM
- Instrumentos em diferenças
- One-step vs two-step
- Teste de Sargan/Hansen
- AR(1) e AR(2) tests
- Problema de instrumentos fracos

**Dataset:** Firm investment (Blundell-Bond)
**Validação:** plm::pgmm (R), xtabond2 (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB06.3: System GMM (Blundell-Bond)
**Arquivo:** `dynamic_system_gmm.ipynb`
**Objetivo:** Estimador de sistema GMM
**Conteúdo:**
- SystemGMM
- Condições de momento em níveis + diferenças
- Ganho de eficiência sobre difference GMM
- Forward orthogonal deviations
- Collapsed instruments (Roodman)
- Teste de instrumentos válidos

**Dataset:** Firm growth (multiple sectors)
**Validação:** plm::pgmm (R), xtabond2 (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB06.4: Diagnósticos para Modelos Dinâmicos
**Arquivo:** `dynamic_diagnostics.ipynb`
**Objetivo:** Testes de especificação para GMM
**Conteúdo:**
- Teste de Sargan/Hansen (overid)
- Teste de autocorrelação (AR1, AR2)
- Difference-in-Hansen test
- Teste de instrumentos fracos
- Bond bounds (OLS vs FE)
- Contagem de instrumentos

**Dataset:** Multiple
**Validação:** R/Stata procedures
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB06.5: Panel VAR
**Arquivo:** `dynamic_panel_var.ipynb`
**Objetivo:** Modelos VAR para painéis
**Conteúdo:**
- PanelVAR
- Seleção de lags (AIC, BIC, HQIC)
- Causalidade de Granger (Dumitrescu-Hurlin)
- Funções de resposta a impulso (IRF)
- Decomposição de variância (FEVD)
- Identificação (Cholesky, estrutural)

**Dataset:** Macro panels (GDP, inflation, interest)
**Validação:** panelvar (Stata), plm (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB06.6: Panel VECM (Cointegração)
**Arquivo:** `dynamic_panel_vecm.ipynb`
**Objetivo:** Modelos de correção de erros
**Conteúdo:**
- PanelVECM
- Testes de cointegração (Westerlund, Pedroni)
- Rank de cointegração
- Vetores cointegrantes
- IRF de longo prazo
- Speed of adjustment (alpha)

**Dataset:** PPP data, Money demand
**Validação:** urca (R), xtcointtest (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB06.7: Validação Completa - Dynamic Models
**Arquivo:** `dynamic_validation_complete.ipynb`
**Objetivo:** Validação GMM, VAR, VECM
**Conteúdo:**
- Difference GMM vs Stata xtabond2
- System GMM validation
- Panel VAR comparisons
- IRF/FEVD accuracy
- Performance benchmarks

**Dataset:** Multiple (Investment, Macro)
**Validação:** R (plm, panelvar, urca) + Stata (xtabond2, pvar, xtcointtest)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

### 🚫 PRIORIDADE 3: Modelos Censurados e Seleção (07_censored_selection/)

#### NB07.1: Tobit para Painéis
**Arquivo:** `censored_tobit_models.ipynb`
**Objetivo:** Variáveis dependentes censuradas
**Conteúdo:**
- PanelTobit (random effects)
- Censura à esquerda, direita, dupla
- Efeitos marginais condicionais vs incondicionais
- Heterocedasticidade em Tobit
- Comparação com OLS truncado

**Dataset:** Labor supply (hours worked), Consumption (durables)
**Validação:** censReg (R), xttobit (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB07.2: Honoré Trimmed LAD
**Arquivo:** `censored_honore_lad.ipynb`
**Objetivo:** Tobit com efeitos fixos
**Conteúdo:**
- HonoreTrimmedLAD
- Trimming para consistência
- Semiparametric approach
- Robustez a distribuição de erros
- Comparação com Tobit RE

**Dataset:** Charitable giving, R&D expenditure
**Validação:** N/A (implementation rare)
**Prioridade:** ⭐ COMPLEMENTAR

---

#### NB07.3: Heckman Selection para Painéis
**Arquivo:** `selection_heckman_panel.ipynb`
**Objetivo:** Correção de viés de seleção
**Conteúdo:**
- PanelHeckman (two-step, ML)
- Equação de seleção + outcome
- Inverse Mills ratio (lambda)
- Identificação via exclusion restrictions
- Comparação com Wooldridge CRE

**Dataset:** Female wages (PSID), Export decisions (firms)
**Validação:** sampleSelection (R), heckman (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB07.4: Validação - Censored and Selection
**Arquivo:** `censored_selection_validation.ipynb`
**Objetivo:** Validação de modelos censurados
**Conteúdo:**
- Tobit comparisons
- Heckman two-step validation
- Marginal effects accuracy

**Dataset:** Multiple
**Validação:** R (censReg, sampleSelection) + Stata (xttobit, heckman)
**Prioridade:** ⭐⭐ IMPORTANTE

---

### 🔍 PRIORIDADE 2: Diagnósticos (08_diagnostics/)

#### NB08.1: Testes de Especificação
**Arquivo:** `diagnostics_specification_tests.ipynb`
**Objetivo:** Testes de modelo correto
**Conteúdo:**
- Teste de Hausman (FE vs RE)
- Teste F (pooled vs FE)
- Teste LM (pooled vs RE)
- RESET test
- LinkTest
- Teste de forma funcional

**Dataset:** Wage panel
**Validação:** plm (R), xtreg postestimation (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB08.2: Testes de Autocorrelação
**Arquivo:** `diagnostics_serial_correlation.ipynb`
**Objetivo:** Detectar correlação serial
**Conteúdo:**
- Wooldridge test (AR(1) em FE)
- Durbin-Watson para painéis
- Baltagi-Wu LBI
- AR(p) tests
- Drukker test

**Dataset:** Macro panels
**Validação:** plm (R), xtserial (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB08.3: Testes de Heteroscedasticidade
**Arquivo:** `diagnostics_heteroskedasticity.ipynb`
**Objetivo:** Testar variância não constante
**Conteúdo:**
- Modified Wald test (groupwise hetero)
- Breusch-Pagan LM
- White test
- Likelihood ratio test
- Visualização de resíduos

**Dataset:** Firm-level data
**Validação:** plm (R), xttest3 (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB08.4: Testes de Correlação Contemporânea
**Arquivo:** `diagnostics_cross_sectional_dependence.ipynb`
**Objetivo:** Dependência entre unidades
**Conteúdo:**
- Breusch-Pagan LM
- Pesaran CD test
- Frees test
- Correlação entre resíduos
- Quando usar Driscoll-Kraay SEs

**Dataset:** Country panels
**Validação:** plm (R), xtcsd (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB08.5: Testes de Raiz Unitária em Painéis
**Arquivo:** `diagnostics_panel_unit_root.ipynb`
**Objetivo:** Estacionariedade em painéis
**Conteúdo:**
- Levin-Lin-Chu (LLC)
- Im-Pesaran-Shin (IPS)
- ADF-Fisher, PP-Fisher
- Breitung test
- Hadri LM test (null: stationarity)

**Dataset:** Macro time series panels
**Validação:** urca (R), xtunitroot (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB08.6: Testes de Cointegração em Painéis
**Arquivo:** `diagnostics_panel_cointegration.ipynb`
**Objetivo:** Relações de longo prazo
**Conteúdo:**
- Pedroni tests (7 statistics)
- Kao test
- Westerlund ECM tests
- Fisher-type Johansen
- Interpretação e aplicações

**Dataset:** PPP, Interest rate parity
**Validação:** urca (R), xtcointtest (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB08.7: Diagnósticos de Outliers e Influência
**Arquivo:** `diagnostics_outliers_influence.ipynb`
**Objetivo:** Detectar observações influentes
**Conteúdo:**
- DFBETA, DFFITS para painéis
- Cook's distance adaptado
- Leverage plots
- Studentized residuals
- Jackknife diagnostics

**Dataset:** Multiple
**Validação:** influence.ME (R), manual calculation
**Prioridade:** ⭐ COMPLEMENTAR

---

### 📊 PRIORIDADE 3: Visualizações (09_visualization/)

#### NB09.1: Visualizações para Modelos Discretos
**Arquivo:** `visualization_discrete_models.ipynb`
**Objetivo:** Plots para logit/probit
**Conteúdo:**
- Probability plots
- Marginal effects plots (AME by covariate)
- ROC curves, AUC
- Confusion matrices
- Predicted vs observed
- Separation plots

**Dataset:** Multiple discrete
**Validação:** N/A (visualization)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB09.2: Visualizações para Regressão Quantílica
**Arquivo:** `visualization_quantile_regression.ipynb`
**Objetivo:** Plots para QR
**Conteúdo:**
- Quantile coefficient plots
- Confidence bands across quantiles
- Comparison plots (OLS vs QR)
- Conditional quantile functions
- Treatment effect heterogeneity

**Dataset:** Wage, Income
**Validação:** N/A (visualization)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB09.3: Visualizações para Modelos Espaciais
**Arquivo:** `visualization_spatial_models.ipynb`
**Objetivo:** Mapas e plots espaciais
**Conteúdo:**
- Choropleth maps
- Moran scatterplots
- LISA cluster maps
- Network connectivity graphs
- Direct/indirect effects plots
- Spatial residuals

**Dataset:** Spatial panels
**Validação:** N/A (visualization)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB09.4: Visualizações para Dinâmicos (IRF, FEVD)
**Arquivo:** `visualization_dynamic_models.ipynb`
**Objetivo:** Plots para VAR/VECM
**Conteúdo:**
- Impulse response functions (bands)
- Forecast error variance decomposition
- Historical decomposition
- Granger causality networks
- Eigenvalue stability plots

**Dataset:** Macro VAR
**Validação:** N/A (visualization)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB09.5: Dashboards Interativos
**Arquivo:** `visualization_interactive_dashboards.ipynb`
**Objetivo:** Dashboards com Plotly/Panel
**Conteúdo:**
- Model comparison dashboard
- Diagnostic dashboard
- Marginal effects explorer
- Quantile explorer
- Spatial map explorer

**Dataset:** Multiple
**Validação:** N/A (interactive)
**Prioridade:** ⭐ COMPLEMENTAR

---

### 🎓 PRIORIDADE 3: Tópicos Avançados (10_advanced/)

#### NB10.1: Instrumental Variables (2SLS) para Painéis
**Arquivo:** `advanced_panel_iv.ipynb`
**Objetivo:** Variáveis instrumentais em painéis
**Conteúdo:**
- PanelIV (2SLS, LIML)
- Teste de instrumentos fracos (Cragg-Donald, Stock-Yogo)
- Teste de sobreidentificação (Sargan, Hansen)
- Hausman endogeneity test
- Comparação com GMM

**Dataset:** Education returns (Card), Trade (Frankel-Romer)
**Validação:** plm::plm(model="within", inst) (R), xtivreg (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB10.2: Bootstrap para Painéis
**Arquivo:** `advanced_panel_bootstrap.ipynb`
**Objetivo:** Inferência via bootstrap
**Conteúdo:**
- Panel bootstrap (block, wild, pairs)
- Bootstrap standard errors
- Bootstrap confidence intervals
- Bootstrap p-values
- Comparação com analytical SEs

**Dataset:** Multiple
**Validação:** boot (R), bootstrap (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB10.3: Clustered Standard Errors
**Arquivo:** `advanced_clustered_standard_errors.ipynb`
**Objetivo:** SEs robustos a clustering
**Conteúdo:**
- One-way clustering (entity)
- Two-way clustering (entity + time)
- Multi-way clustering
- Cameron-Gelbach-Miller
- Quando usar cada tipo

**Dataset:** Firm-level, State-level
**Validação:** sandwich (R), cluster (Stata)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB10.4: Driscoll-Kraay Standard Errors
**Arquivo:** `advanced_driscoll_kraay.ipynb`
**Objetivo:** SEs robustos a dependência cross-sectional e serial
**Conteúdo:**
- Driscoll-Kraay covariance
- Escolha de lags (Newey-West type)
- Comparação com clustering
- Aplicações em macro panels

**Dataset:** Country panels
**Validação:** plm (R), xtscc (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB10.5: Penalized Regression (Lasso, Ridge) para Painéis
**Arquivo:** `advanced_penalized_panel.ipynb`
**Objetivo:** Seleção de variáveis em painéis
**Conteúdo:**
- Panel Lasso
- Panel Ridge
- Elastic Net
- Cross-validation para λ
- Comparação com stepwise

**Dataset:** High-dimensional panels
**Validação:** glmnet (R), lassopack (Stata)
**Prioridade:** ⭐ COMPLEMENTAR

---

#### NB10.6: Missing Data em Painéis
**Arquivo:** `advanced_missing_data.ipynb`
**Objetivo:** Tratamento de dados faltantes
**Conteúdo:**
- Padrões de missingness em painéis
- Multiple imputation
- Maximum likelihood com missing
- Inverse probability weighting
- Sensitivity analysis

**Dataset:** PSID, NLSY (with missingness)
**Validação:** mice (R), mi (Stata)
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB10.7: Painéis Desbalanceados
**Arquivo:** `advanced_unbalanced_panels.ipynb`
**Objetivo:** Lidar com painéis não balanceados
**Conteúdo:**
- Implicações de desbalanceamento
- Entrada e saída dinâmica (attrition)
- Seleção amostral (Heckman)
- Inverse probability weighting
- Comparação balanced vs unbalanced

**Dataset:** PSID (unbalanced)
**Validação:** Multiple methods
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB10.8: Testes de Robustez Sistemáticos
**Arquivo:** `advanced_robustness_checks.ipynb`
**Objetivo:** Frameworks para robustness
**Conteúdo:**
- Specification curve analysis
- Multiverse analysis
- Sensitivity to outliers (winsorization)
- Sensitivity to clustering
- Sensitivity to sample period
- Automated robustness reporting

**Dataset:** Multiple
**Validação:** N/A (framework)
**Prioridade:** ⭐⭐ IMPORTANTE

---

### ✅ PRIORIDADE 1: Validação contra R (11_validation_r/)

#### NB11.1: Validação contra plm (R)
**Arquivo:** `validation_r_plm_package.ipynb`
**Objetivo:** Comparação completa com plm
**Conteúdo:**
- Todos os modelos estáticos (pooled, FE, RE, FD, between)
- Standard errors (robust, clustered)
- Tests (Hausman, F, LM)
- Identical results verification

**Dataset:** Grunfeld, Produc, Wages
**Validação:** plm (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB11.2: Validação contra glm/glmer (R)
**Arquivo:** `validation_r_discrete_choice.ipynb`
**Objetivo:** Validação de modelos discretos
**Conteúdo:**
- glm: logit, probit pooled
- glmer: random effects
- survival::clogit: conditional logit
- MASS::polr: ordered models
- mlogit: multinomial logit

**Dataset:** Multiple discrete
**Validação:** glm, lme4, survival, MASS, mlogit (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB11.3: Validação contra MASS/pscl (R)
**Arquivo:** `validation_r_count_models.ipynb`
**Objetivo:** Validação de modelos de contagem
**Conteúdo:**
- MASS::glm.nb: negative binomial
- pscl::zeroinfl: ZIP, ZINB
- pscl::hurdle: hurdle models

**Dataset:** Doctor visits, Recreation
**Validação:** MASS, pscl (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB11.4: Validação contra quantreg (R)
**Arquivo:** `validation_r_quantile_regression.ipynb`
**Objetivo:** Validação de QR
**Conteúdo:**
- quantreg::rq: pooled QR
- quantreg::rqpd: panel QR (Canay)
- Coefficients and standard errors

**Dataset:** Wage data
**Validação:** quantreg (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB11.5: Validação contra spatialreg/spdep (R)
**Arquivo:** `validation_r_spatial_models.ipynb`
**Objetivo:** Validação de modelos espaciais
**Conteúdo:**
- spatialreg::lagsarlm: SAR
- spatialreg::errorsarlm: SEM
- spdep: spatial tests (Moran's I, LM tests)
- Conley HAC

**Dataset:** Crime, Regional
**Validação:** spatialreg, spdep (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB11.6: Validação contra plm::pgmm (R)
**Arquivo:** `validation_r_dynamic_gmm.ipynb`
**Objetivo:** Validação de GMM dinâmico
**Conteúdo:**
- plm::pgmm: difference GMM, system GMM
- Coefficients, Hansen test, AR tests

**Dataset:** Employment (Arellano-Bond)
**Validação:** plm::pgmm (R)
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

### ✅ PRIORIDADE 2: Validação contra Stata (12_validation_stata/)

#### NB12.1: Validação contra xtreg (Stata)
**Arquivo:** `validation_stata_xtreg.ipynb`
**Objetivo:** Modelos lineares estáticos
**Conteúdo:**
- xtreg, fe / re / be / fd
- Standard errors (robust, cluster)
- Tests (Hausman, F, LM)

**Dataset:** Grunfeld, Wages
**Validação:** Stata xtreg
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB12.2: Validação contra xtlogit/xtprobit (Stata)
**Arquivo:** `validation_stata_discrete.ipynb`
**Objetivo:** Modelos discretos
**Conteúdo:**
- xtlogit: pooled, fe, re
- xtprobit: re
- clogit: conditional logit
- ologit/oprobit: ordered

**Dataset:** Union, Patent
**Validação:** Stata xtlogit, xtprobit, clogit
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB12.3: Validação contra xtpoisson/xtnbreg (Stata)
**Arquivo:** `validation_stata_count.ipynb`
**Objetivo:** Modelos de contagem
**Conteúdo:**
- xtpoisson: fe, re
- xtnbreg: re
- zip, zinb: zero-inflated

**Dataset:** Doctor visits, Patents
**Validação:** Stata xtpoisson, xtnbreg, zip
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB12.4: Validação contra qregpd/xtqreg (Stata)
**Arquivo:** `validation_stata_quantile.ipynb`
**Objetivo:** Regressão quantílica
**Conteúdo:**
- qregpd: panel QR (various methods)
- xtqreg: (if available)

**Dataset:** Wage data
**Validação:** Stata qregpd
**Prioridade:** ⭐⭐ IMPORTANTE

---

#### NB12.5: Validação contra xsmle (Stata)
**Arquivo:** `validation_stata_spatial.ipynb`
**Objetivo:** Modelos espaciais
**Conteúdo:**
- xsmle: SAR, SEM, SDM, SDEM
- spmat: weight matrices
- Coefficients, direct/indirect effects

**Dataset:** Crime, Regional
**Validação:** Stata xsmle, spmat
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

#### NB12.6: Validação contra xtabond2 (Stata)
**Arquivo:** `validation_stata_dynamic_gmm.ipynb`
**Objetivo:** GMM dinâmico
**Conteúdo:**
- xtabond2: difference GMM, system GMM
- One-step vs two-step
- Orthogonal deviations
- Collapsed instruments
- Hansen, AR(1), AR(2) tests

**Dataset:** Firm investment
**Validação:** Stata xtabond2
**Prioridade:** ⭐⭐⭐ CRÍTICO

---

## 📦 Datasets Compartilhados (datasets/)

### Datasets a serem incluídos:

1. **Grunfeld** - Investment data (classic panel)
2. **Produc** - US states productivity
3. **Wages** - PSID wage panel
4. **UnionMembership** - NLS union data
5. **PatentApplications** - Blundell-Griffith-Van Reenen
6. **DoctorVisits** - SOEP health data
7. **RecreationalTrips** - Cameron-Trivedi
8. **IncomeData** - PSID income inequality
9. **CrimeData** - US counties crime
10. **RegionalGrowth** - EU NUTS regions
11. **MacroPanel** - GDP, inflation, interest rates (countries)
12. **FirmInvestment** - Blundell-Bond firm data
13. **TradeData** - Bilateral trade flows
14. **RealEstateData** - House prices (spatial)
15. **EnvironmentalData** - Pollution (spatial-temporal)

**Script:** `datasets/download_and_prepare_datasets.py`

---

## 📋 Prioridades de Implementação

### FASE 1 (Crítico - 4 semanas)
**Total:** 25 notebooks

1. ✅ **Introdutórios (3):** NB01.1, NB01.2, NB01.3
2. ✅ **Discrete Choice (3):** NB02.1, NB02.2, NB02.3, NB02.7
3. ✅ **Count Models (2):** NB03.1, NB03.2, NB03.5
4. ✅ **Quantile Regression (4):** NB04.1, NB04.2, NB04.5, NB04.7, NB04.8
5. ✅ **Spatial (5):** NB05.1, NB05.2, NB05.3, NB05.4, NB05.7, NB05.10
6. ✅ **Dynamic (3):** NB06.1, NB06.2, NB06.3, NB06.5, NB06.7
7. ✅ **Diagnostics (1):** NB08.1
8. ✅ **Validation R (6):** NB11.1 - NB11.6
9. ✅ **Validation Stata (6):** NB12.1 - NB12.6

### FASE 2 (Importante - 3 semanas)
**Total:** 20 notebooks

1. Discrete Choice: NB02.4, NB02.5, NB02.6
2. Count Models: NB03.3, NB03.4
3. Quantile Regression: NB04.3, NB04.4, NB04.6
4. Spatial: NB05.5, NB05.6, NB05.8, NB05.9
5. Dynamic: NB06.4, NB06.6
6. Censored/Selection: NB07.1, NB07.3, NB07.4
7. Diagnostics: NB08.2, NB08.3, NB08.4, NB08.5, NB08.6
8. Visualization: NB09.1, NB09.2, NB09.3, NB09.4
9. Advanced: NB10.1, NB10.3, NB10.4, NB10.6, NB10.7, NB10.8

### FASE 3 (Complementar - 2 semanas)
**Total:** 15 notebooks

1. Censored/Selection: NB07.2
2. Diagnostics: NB08.7
3. Visualization: NB09.5
4. Advanced: NB10.2, NB10.5

---

## 🎯 Template de Notebook

Cada notebook deve seguir esta estrutura:

```python
# [TÍTULO DO NOTEBOOK]

## 1. Setup e Imports
- Instalação de dependências
- Imports necessários
- Configurações de visualização

## 2. Introdução e Motivação
- Contexto econômico
- Quando usar este modelo?
- Perguntas de pesquisa típicas

## 3. Carregamento de Dados
- Dataset description
- Exploração inicial (describe, plot)
- Panel structure verification

## 4. Implementação PanelBox
- Model specification
- Estimation
- Results summary

## 5. Diagnósticos
- Specification tests
- Residual analysis
- Assumption checks

## 6. Visualizações
- Coefficient plots
- Predicted vs actual
- Model-specific viz

## 7. Validação (quando aplicável)
- Comparison with R/Stata
- Numerical accuracy check
- Tolerance analysis

## 8. Interpretação
- Economic interpretation
- Policy implications
- Caveats and limitations

## 9. Extensões e Próximos Passos
- Related models
- Advanced topics
- Further reading

## 10. Referências
- Key papers
- Documentation links
- Related notebooks
```

---

## 📊 Métricas de Sucesso

Para cada notebook:

✅ **Completude**
- [ ] Código executa sem erros
- [ ] Todos os modelos especificados estão implementados
- [ ] Visualizações renderizam corretamente

✅ **Qualidade**
- [ ] Explicações claras e didáticas
- [ ] Interpretações econômicas corretas
- [ ] Código bem documentado

✅ **Validação** (quando aplicável)
- [ ] Coeficientes coincidem (tolerância < 1e-4)
- [ ] Standard errors coincidem (tolerância < 1e-3)
- [ ] Testes estatísticos coincidem

✅ **Reprodutibilidade**
- [ ] Seeds fixadas para aleatoriedade
- [ ] Datasets incluídos ou com download automático
- [ ] Versões de pacotes especificadas

---

## 🔧 Ferramentas e Infraestrutura

### Ambiente de Desenvolvimento
```bash
# Create conda environment
conda create -n panelbox-examples python=3.11
conda activate panelbox-examples

# Install PanelBox
pip install -e /home/guhaase/projetos/panelbox

# Install validation packages
pip install rpy2  # R integration
conda install -c conda-forge r-plm r-lme4 r-quantreg r-spatialreg r-spdep

# Install Stata integration (if available)
pip install pystata

# Install notebook tools
pip install jupyter jupyterlab nbconvert
pip install matplotlib seaborn plotly
```

### Automation Scripts

**`scripts/run_all_notebooks.py`**
- Execute all notebooks sequentially
- Capture outputs and errors
- Generate summary report

**`scripts/validate_notebooks.py`**
- Check all notebooks execute without errors
- Verify numerical accuracy of validations
- Generate validation report

**`scripts/generate_index.py`**
- Create HTML index of all notebooks
- Organize by category
- Add search functionality

---

## 📝 Convenções de Nomenclatura

### Arquivos
- Notebooks: `{category}_{topic}_{variant}.ipynb`
- Datasets: `{source}_{name}.csv` ou `.parquet`
- Scripts: `{action}_{target}.py`

### Variáveis no código
- DataFrames: `df_`, `data_`
- Models: `model_`, `fit_`
- Results: `results_`, `res_`
- Plots: `fig_`, `ax_`

### Commits
- `feat: Add NB04.2 - Canay quantile regression`
- `fix: Correct standard errors in NB03.1`
- `docs: Improve interpretation section in NB05.2`
- `validate: Add Stata comparison for NB12.3`

---

## 🚀 Próximos Passos

1. **Review deste planejamento** ✅
2. **Setup do ambiente** (scripts de instalação)
3. **Preparar datasets** (script de download)
4. **Template de notebook** (criar .ipynb template)
5. **FASE 1 - Notebooks críticos** (25 notebooks em 4 semanas)
6. **FASE 2 - Notebooks importantes** (20 notebooks em 3 semanas)
7. **FASE 3 - Notebooks complementares** (15 notebooks em 2 semanas)
8. **Automação e CI/CD** (GitHub Actions para validação)
9. **Documentação final** (índice, guia de uso)
10. **Publicação** (ReadTheDocs, GitHub Pages)

---

## 📚 Referências Bibliográficas Principais

1. **Wooldridge (2010)** - Econometric Analysis of Cross Section and Panel Data
2. **Baltagi (2021)** - Econometric Analysis of Panel Data
3. **Arellano (2003)** - Panel Data Econometrics
4. **Cameron & Trivedi (2005)** - Microeconometrics
5. **Koenker (2005)** - Quantile Regression
6. **LeSage & Pace (2009)** - Introduction to Spatial Econometrics
7. **Roodman (2009)** - How to do xtabond2
8. **Elhorst (2014)** - Spatial Econometrics

---

## 📞 Contato e Contribuição

- **Repositório:** `/home/guhaase/projetos/panelbox`
- **Issues:** Para reportar problemas com notebooks
- **Pull Requests:** Contribuições são bem-vindas
- **Discussões:** Para sugestões de novos exemplos

---

**Status:** 📋 PLANEJAMENTO COMPLETO
**Última Atualização:** 2026-02-15
**Versão:** 1.0
**Total de Notebooks Planejados:** 60+

---

## Apêndice A: Checklist de Implementação

```markdown
### FASE 1 - Crítico (25 notebooks)

#### Introdutórios
- [ ] NB01.1: Introdução ao PanelBox
- [ ] NB01.2: Preparação de Dados
- [ ] NB01.3: Modelos Estáticos Básicos

#### Discrete Choice
- [ ] NB02.1: Logit/Probit Básicos
- [ ] NB02.2: Fixed Effects
- [ ] NB02.3: Random Effects
- [ ] NB02.7: Validação Completa

#### Count Models
- [ ] NB03.1: Poisson Básico
- [ ] NB03.2: Negative Binomial
- [ ] NB03.5: Validação Completa

#### Quantile Regression
- [ ] NB04.1: Introdução QR
- [ ] NB04.2: Canay Fixed Effects
- [ ] NB04.5: Treatment Effects
- [ ] NB04.7: Comparação de Métodos
- [ ] NB04.8: Validação Completa

#### Spatial
- [ ] NB05.1: Introdução Espacial
- [ ] NB05.2: SAR Model
- [ ] NB05.3: SEM Model
- [ ] NB05.4: SDM Model
- [ ] NB05.7: Weight Matrices
- [ ] NB05.10: Validação Completa

#### Dynamic
- [ ] NB06.1: Introdução Dinâmicos
- [ ] NB06.2: Difference GMM
- [ ] NB06.3: System GMM
- [ ] NB06.5: Panel VAR
- [ ] NB06.7: Validação Completa

#### Diagnostics
- [ ] NB08.1: Specification Tests

#### Validation R
- [ ] NB11.1: plm validation
- [ ] NB11.2: glm/glmer validation
- [ ] NB11.3: MASS/pscl validation
- [ ] NB11.4: quantreg validation
- [ ] NB11.5: spatialreg validation
- [ ] NB11.6: pgmm validation

#### Validation Stata
- [ ] NB12.1: xtreg validation
- [ ] NB12.2: xtlogit/xtprobit validation
- [ ] NB12.3: xtpoisson/xtnbreg validation
- [ ] NB12.4: qregpd validation
- [ ] NB12.5: xsmle validation
- [ ] NB12.6: xtabond2 validation

### FASE 2 - Importante (20 notebooks)
[... continua ...]

### FASE 3 - Complementar (15 notebooks)
[... continua ...]
```

---

**FIM DO PLANEJAMENTO**
