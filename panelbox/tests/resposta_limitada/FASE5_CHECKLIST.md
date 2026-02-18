# FASE 5: Validação contra R - Checklist de Execução

Este documento acompanha o progresso da validação dos modelos de resposta limitada contra implementações em R.

---

## Status Geral

**Data de Início**: ___________
**Data de Conclusão**: ___________
**Responsável**: ___________

**Progresso Geral**: [ ] 0% - [ ] 25% - [ ] 50% - [ ] 75% - [ ] 100%

---

## 1. Preparação do Ambiente

### 1.1 Verificação do R
- [ ] Verificar que R está instalado: `R --version`
  - Versão do R: ___________
  - [ ] R >= 4.0.0

### 1.2 Instalação de Pacotes R

#### Pacotes Core
- [ ] Instalar `plm`: `install.packages("plm")`
- [ ] Instalar `jsonlite`: `install.packages("jsonlite")`
- [ ] Instalar `MASS`: `install.packages("MASS")`

#### Pacotes para Modelos Específicos
- [ ] Instalar `survival`: `install.packages("survival")` (FE Logit)
- [ ] Instalar `censReg`: `install.packages("censReg")` (Tobit)

#### Pacotes para Efeitos Marginais
- [ ] Instalar `mfx`: `install.packages("mfx")`
- [ ] Instalar `margins`: `install.packages("margins")`

#### Pacotes para Erros Robustos
- [ ] Instalar `sandwich`: `install.packages("sandwich")`
- [ ] Instalar `lmtest`: `install.packages("lmtest")`

### 1.3 Verificação de Instalação
- [ ] Executar script de verificação:
```r
packages <- c("plm", "jsonlite", "MASS", "survival", "censReg",
              "mfx", "margins", "sandwich", "lmtest")
installed <- sapply(packages, requireNamespace, quietly = TRUE)
print(data.frame(Package = packages, Installed = installed))
```

### 1.4 Estrutura de Diretórios
- [ ] Verificar existência de `tests/resposta_limitada/`
- [ ] Verificar existência de `tests/resposta_limitada/r/`
- [ ] Verificar existência de `tests/resposta_limitada/r/results/`
- [ ] Verificar existência de `tests/resposta_limitada/data/`

---

## 2. Geração de Dados de Teste

### 2.1 Executar Script de Geração
- [ ] Navegar para diretório: `cd tests/resposta_limitada/`
- [ ] Executar: `python generate_test_data.py`
- [ ] Verificar output sem erros

### 2.2 Verificar Arquivos Criados
- [ ] Verificar `data/binary_panel_test.csv` existe
  - [ ] Verificar tamanho do arquivo > 0
  - [ ] Abrir e inspecionar primeiras linhas
  - [ ] Verificar colunas: entity, time, x1, x2, y

- [ ] Verificar `data/censored_panel_test.csv` existe
  - [ ] Verificar tamanho do arquivo > 0
  - [ ] Abrir e inspecionar primeiras linhas
  - [ ] Verificar que há observações censuradas (y=0)

- [ ] Verificar `data/count_panel_test.csv` existe
  - [ ] Verificar tamanho do arquivo > 0
  - [ ] Abrir e inspecionar primeiras linhas
  - [ ] Verificar que y contém inteiros não-negativos

### 2.3 Estatísticas Descritivas
- [ ] Dataset binário:
  - N entidades: ___________
  - T períodos: ___________
  - Total observações: ___________
  - Média de y: ___________
  - % entidades com variação em y: ___________%

- [ ] Dataset censurado:
  - Total observações: ___________
  - % observações censuradas: ___________%
  - Média de y (não-censurado): ___________

- [ ] Dataset de contagem:
  - Total observações: ___________
  - Média de y: ___________
  - Variância de y: ___________
  - Razão Variância/Média: ___________ (>1 indica sobredispersão)

---

## 3. Execução de Benchmarks R

### 3.1 Modelos Binários (benchmark_discrete.R)

#### Preparação
- [ ] Navegar para: `cd tests/resposta_limitada/r/`
- [ ] Verificar que arquivo `benchmark_discrete.R` existe

#### Execução
- [ ] Executar: `Rscript benchmark_discrete.R`
- [ ] Verificar execução sem erros críticos
- [ ] Tempo de execução: ___________ segundos

#### Resultados - Pooled Logit
- [ ] Arquivo `results/pooled_logit_results.json` criado
- [ ] Abrir arquivo e verificar estrutura:
  - [ ] Contém `coef` (coeficientes)
  - [ ] Contém `se` (standard errors)
  - [ ] Contém `loglik` (log-likelihood)
  - [ ] Contém `ame` (marginal effects)
- [ ] Valores parecem razoáveis (sem NaN, Inf)

#### Resultados - Pooled Probit
- [ ] Arquivo `results/pooled_probit_results.json` criado
- [ ] Verificar estrutura similar ao Logit
- [ ] Valores razoáveis

#### Resultados - FE Logit
- [ ] Arquivo `results/fe_logit_results.json` criado
- [ ] Verificar estrutura:
  - [ ] Contém `n_dropped` (entidades removidas)
- [ ] Número de entidades removidas: ___________

### 3.2 Modelos Censurados (benchmark_tobit.R)

#### Preparação
- [ ] Verificar que arquivo `benchmark_tobit.R` existe

#### Execução
- [ ] Executar: `Rscript benchmark_tobit.R`
- [ ] Verificar execução sem erros
- [ ] Tempo de execução: ___________ segundos

#### Resultados - Pooled Tobit
- [ ] Arquivo `results/pooled_tobit_results.json` criado
- [ ] Verificar estrutura:
  - [ ] Contém `coef`
  - [ ] Contém `se`
  - [ ] Contém `sigma` (parâmetro de escala)
  - [ ] Contém `loglik`
  - [ ] Contém `me_unconditional` (efeitos marginais incondicionais)
- [ ] Valores razoáveis
- [ ] Valor de sigma: ___________

### 3.3 Modelos de Contagem (benchmark_count.R)

#### Preparação
- [ ] Verificar que arquivo `benchmark_count.R` existe

#### Execução
- [ ] Executar: `Rscript benchmark_count.R`
- [ ] Verificar execução sem erros
- [ ] Tempo de execução: ___________ segundos

#### Resultados - Pooled Poisson
- [ ] Arquivo `results/pooled_poisson_results.json` criado
- [ ] Verificar estrutura completa
- [ ] Contém `ame` com valores para x1 e x2
- [ ] Valores razoáveis

#### Resultados - FE Poisson
- [ ] Arquivo `results/fe_poisson_results.json` criado
- [ ] Verificar estrutura
- [ ] Valores razoáveis

#### Resultados - Negative Binomial
- [ ] Arquivo `results/negbin_results.json` criado
- [ ] Verificar estrutura:
  - [ ] Contém `theta` (parâmetro de dispersão)
- [ ] Valores razoáveis
- [ ] Valor de theta: ___________

### 3.4 Verificação Geral dos Resultados R
- [ ] Total de arquivos JSON criados: _____ / 7
- [ ] Todos os arquivos são válidos JSON (podem ser abertos)
- [ ] Nenhum arquivo contém apenas `{}` (vazio)
- [ ] Listar tamanhos dos arquivos: `ls -lh results/`

---

## 4. Testes de Validação Python

### 4.1 Preparação
- [ ] Verificar que arquivo `test_r_validation.py` existe
- [ ] Verificar imports:
  ```python
  from panelbox.models.discrete.binary import PooledLogit, PooledProbit, FixedEffectsLogit
  from panelbox.models.censored import PooledTobit
  from panelbox.models.count import PooledPoisson, PoissonFixedEffects, NegativeBinomial
  ```
- [ ] Todos os imports funcionam sem erro

### 4.2 Execução dos Testes

#### Executar Todos os Testes
- [ ] Navegar para raiz do projeto
- [ ] Executar: `pytest tests/resposta_limitada/test_r_validation.py -v`
- [ ] Tempo de execução: ___________ segundos

#### Resultados por Classe de Teste

**TestPooledLogitVsR**
- [ ] `test_coefficients`: _____ (PASSED/FAILED)
- [ ] `test_standard_errors`: _____ (PASSED/FAILED)
- [ ] `test_loglikelihood`: _____ (PASSED/FAILED)
- [ ] `test_marginal_effects`: _____ (PASSED/FAILED)
- **Status**: _____ / 4 testes passaram

**TestPooledProbitVsR**
- [ ] `test_coefficients`: _____ (PASSED/FAILED)
- [ ] `test_standard_errors`: _____ (PASSED/FAILED)
- [ ] `test_loglikelihood`: _____ (PASSED/FAILED)
- [ ] `test_marginal_effects`: _____ (PASSED/FAILED)
- **Status**: _____ / 4 testes passaram

**TestFELogitVsR**
- [ ] `test_coefficients`: _____ (PASSED/FAILED)
- [ ] `test_standard_errors`: _____ (PASSED/FAILED)
- [ ] `test_loglikelihood`: _____ (PASSED/FAILED)
- **Status**: _____ / 3 testes passaram

**TestPooledTobitVsR**
- [ ] `test_coefficients`: _____ (PASSED/FAILED)
- [ ] `test_standard_errors`: _____ (PASSED/FAILED)
- [ ] `test_sigma`: _____ (PASSED/FAILED)
- [ ] `test_loglikelihood`: _____ (PASSED/FAILED)
- **Status**: _____ / 4 testes passaram

**TestPooledPoissonVsR**
- [ ] `test_coefficients`: _____ (PASSED/FAILED)
- [ ] `test_standard_errors`: _____ (PASSED/FAILED)
- [ ] `test_loglikelihood`: _____ (PASSED/FAILED)
- [ ] `test_marginal_effects`: _____ (PASSED/FAILED)
- **Status**: _____ / 4 testes passaram

**TestFEPoissonVsR**
- [ ] `test_coefficients`: _____ (PASSED/FAILED)
- [ ] `test_standard_errors`: _____ (PASSED/FAILED)
- [ ] `test_loglikelihood`: _____ (PASSED/FAILED)
- **Status**: _____ / 3 testes passaram

**TestNegativeBinomialVsR**
- [ ] `test_coefficients`: _____ (PASSED/FAILED)
- [ ] `test_standard_errors`: _____ (PASSED/FAILED)
- [ ] `test_theta`: _____ (PASSED/FAILED)
- [ ] `test_loglikelihood`: _____ (PASSED/FAILED)
- **Status**: _____ / 4 testes passaram

### 4.3 Resumo Geral dos Testes
- **Total de testes executados**: ___________
- **Testes passados**: ___________
- **Testes falhos**: ___________
- **Taxa de sucesso**: ___________%

---

## 5. Análise de Discrepâncias

### 5.1 Testes que Falharam

Para cada teste que falhou, documentar:

#### Falha #1
- **Teste**: ___________
- **Modelo**: ___________
- **Variável**: ___________
- **Valor R**: ___________
- **Valor Python**: ___________
- **Diferença Absoluta**: ___________
- **Diferença Relativa (%)**: ___________%
- **Tolerância esperada**: ___________%
- **Hipótese da causa**: ___________
- **Ação necessária**: [ ] Corrigir código [ ] Ajustar tolerância [ ] Documentar [ ] Investigar

#### Falha #2
[Repetir estrutura acima]

### 5.2 Discrepâncias Dentro da Tolerância

Para testes que passaram mas com diferenças próximas ao limite:

#### Discrepância #1
- **Teste**: ___________
- **Diferença Relativa**: ___________%
- **Comentário**: ___________

### 5.3 Causas Identificadas

- [ ] Diferenças em algoritmo de otimização (scipy vs R optim)
  - Magnitude típica: ___________%
  - Modelos afetados: ___________

- [ ] Diferenças em cálculo numérico da Hessiana
  - Magnitude típica: ___________%
  - Modelos afetados: ___________

- [ ] Diferenças em tipo de erro padrão (padrão vs robusto)
  - Magnitude típica: ___________%
  - Modelos afetados: ___________

- [ ] Diferenças em métodos de integração (para RE)
  - Magnitude típica: ___________%
  - Modelos afetados: ___________

- [ ] Outras causas: ___________

---

## 6. Ações Corretivas

### 6.1 Correções de Código Necessárias

#### Correção #1
- **Arquivo**: ___________
- **Linha(s)**: ___________
- **Problema**: ___________
- **Solução proposta**: ___________
- **Status**: [ ] A fazer [ ] Em progresso [ ] Testado [ ] Concluído

#### Correção #2
[Repetir estrutura]

### 6.2 Ajustes em Tolerâncias

- [ ] Revisar tolerâncias em `test_r_validation.py`
  - COEF_RTOL atual: 0.05 → Proposto: ___________
  - SE_RTOL atual: 0.10 → Proposto: ___________
  - ME_RTOL atual: 0.10 → Proposto: ___________

- [ ] Justificar cada ajuste no relatório de validação

### 6.3 Documentação de Limitações

- [ ] Adicionar seção em `VALIDATION_REPORT.md` sobre discrepâncias aceitáveis
- [ ] Atualizar docstrings de classes afetadas
- [ ] Adicionar warnings onde apropriado

---

## 7. Relatório de Validação

### 7.1 Preencher VALIDATION_REPORT.md

- [ ] Seção: Executive Summary
  - [ ] Overall Status
  - [ ] Success Rate

- [ ] Seção: Test Environment
  - [ ] Software Versions
  - [ ] R Packages
  - [ ] Test Data Characteristics

- [ ] Seção: Validation Results by Model (para cada modelo)
  - [ ] Pooled Logit
  - [ ] Pooled Probit
  - [ ] Fixed Effects Logit
  - [ ] Pooled Tobit
  - [ ] Pooled Poisson
  - [ ] Fixed Effects Poisson
  - [ ] Negative Binomial

- [ ] Seção: Summary Statistics
  - [ ] Overall Test Results
  - [ ] Discrepancy Distribution
  - [ ] Average Relative Differences

- [ ] Seção: Known Discrepancies
  - [ ] Documentar cada tipo de discrepância
  - [ ] Status de cada uma

- [ ] Seção: Issues Requiring Investigation
  - [ ] Listar issues críticos
  - [ ] Listar issues não-críticos

- [ ] Seção: Recommendations
  - [ ] Para usuários
  - [ ] Para desenvolvedores

- [ ] Seção: Conclusion
  - [ ] Recomendação final

### 7.2 Anexos

- [ ] Anexo A: Test Execution Details
- [ ] Anexo B: R Session Info
  - [ ] Executar `sessionInfo()` em R e copiar output
- [ ] Anexo C: Python Environment
  - [ ] Executar `pip list` e copiar output
- [ ] Anexo D: Raw Test Output
  - [ ] Salvar output completo do pytest

---

## 8. Revisão e Aprovação

### 8.1 Revisão Técnica
- [ ] Código revisado por: ___________
- [ ] Data da revisão: ___________
- [ ] Comentários: ___________
- [ ] Status: [ ] Aprovado [ ] Aprovado com ressalvas [ ] Requer mudanças

### 8.2 Revisão de Documentação
- [ ] Documentação revisada por: ___________
- [ ] Data da revisão: ___________
- [ ] Comentários: ___________
- [ ] Status: [ ] Aprovado [ ] Requer mudanças

### 8.3 Aprovação Final
- [ ] Aprovado por: ___________
- [ ] Data: ___________
- [ ] Recomendação: [ ] Aprovar para release [ ] Aprovação condicional [ ] Requer mais trabalho

---

## 9. Próximos Passos

### 9.1 Integração Contínua
- [ ] Adicionar testes R ao CI/CD pipeline
- [ ] Configurar execução automática em PRs
- [ ] Configurar notificações de falhas

### 9.2 Validação Adicional
- [ ] Comparar com Stata (se disponível)
- [ ] Comparar com outros pacotes Python (statsmodels)
- [ ] Adicionar testes com dados reais

### 9.3 Documentação para Usuários
- [ ] Adicionar seção de validação ao README principal
- [ ] Criar badge de validação
- [ ] Adicionar exemplos comparativos R vs Python na documentação

---

## 10. Registro de Mudanças

| Data | Responsável | Mudança | Motivo |
|------|-------------|---------|--------|
| | | | |
| | | | |
| | | | |

---

## Assinaturas

**Desenvolvedor**: ___________________________ Data: ___________

**Revisor**: ___________________________ Data: ___________

**Aprovador**: ___________________________ Data: ___________

---

**Fim do Checklist - FASE 5**
