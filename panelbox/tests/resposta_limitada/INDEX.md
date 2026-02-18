# FASE 5: Validação contra R - Índice de Documentos

## 📋 Navegação Rápida

### Para Começar
- 👉 **[FASE5_SUMMARY.md](FASE5_SUMMARY.md)** - Leia PRIMEIRO para visão geral
- 📖 **[README_R_VALIDATION.md](README_R_VALIDATION.md)** - Guia passo-a-passo detalhado

### Durante a Execução
- ✅ **[FASE5_CHECKLIST.md](FASE5_CHECKLIST.md)** - Use para rastrear progresso

### Após Execução
- 📊 **[VALIDATION_REPORT.md](VALIDATION_REPORT.md)** - Preencha com resultados

---

## 📁 Estrutura de Arquivos

```
tests/resposta_limitada/
│
├── 📚 DOCUMENTAÇÃO
│   ├── INDEX.md                      # Este arquivo
│   ├── FASE5_SUMMARY.md              # Visão geral executiva
│   ├── README_R_VALIDATION.md        # Guia completo passo-a-passo
│   ├── FASE5_CHECKLIST.md            # Checklist de execução
│   └── VALIDATION_REPORT.md          # Template de relatório
│
├── 🔧 SCRIPTS
│   ├── generate_test_data.py         # Gera dados sintéticos
│   └── test_r_validation.py          # Suite de testes pytest
│
├── 📊 BENCHMARKS R
│   └── r/
│       ├── benchmark_discrete.R      # Logit, Probit, FE Logit
│       ├── benchmark_tobit.R         # Tobit censurado
│       ├── benchmark_count.R         # Poisson, FE Poisson, NegBin
│       └── results/                  # JSONs gerados pelo R
│           ├── pooled_logit_results.json
│           ├── pooled_probit_results.json
│           ├── fe_logit_results.json
│           ├── pooled_tobit_results.json
│           ├── pooled_poisson_results.json
│           ├── fe_poisson_results.json
│           └── negbin_results.json
│
└── 📦 DADOS DE TESTE
    └── data/
        ├── binary_panel_test.csv     # Para Logit/Probit
        ├── censored_panel_test.csv   # Para Tobit
        └── count_panel_test.csv      # Para Poisson/NegBin
```

---

## 🚀 Guia Rápido de Uso

### 1️⃣ Primeira Vez (Setup Inicial)

```bash
# Instalar R e pacotes
R
> install.packages(c("plm", "jsonlite", "MASS", "survival",
                     "censReg", "mfx", "margins", "sandwich", "lmtest"))

# Gerar dados de teste
cd tests/resposta_limitada/
python generate_test_data.py
```

### 2️⃣ Executar Benchmarks R

```bash
cd r/
Rscript benchmark_discrete.R
Rscript benchmark_tobit.R
Rscript benchmark_count.R
cd ..
```

### 3️⃣ Executar Validação Python

```bash
# Da raiz do projeto
pytest tests/resposta_limitada/test_r_validation.py -v
```

### 4️⃣ Documentar Resultados

```bash
# Editar com seus resultados
nano VALIDATION_REPORT.md
```

---

## 📖 Descrição dos Documentos

### FASE5_SUMMARY.md
**Público**: Todos
**Conteúdo**:
- Visão geral dos objetivos
- Modelos validados
- Workflow de validação
- Critérios de sucesso
- Quick start

**Quando ler**: Antes de começar

---

### README_R_VALIDATION.md
**Público**: Executores da validação
**Conteúdo**:
- Pré-requisitos detalhados
- Instruções passo-a-passo
- Troubleshooting completo
- Interpretação de resultados
- Timeline estimado

**Quando ler**: Durante execução

---

### FASE5_CHECKLIST.md
**Público**: Executores da validação
**Conteúdo**:
- Checklist item-por-item
- Espaços para preencher resultados
- Tracking de falhas
- Análise de discrepâncias
- Registro de decisões

**Quando usar**: Durante toda a execução

---

### VALIDATION_REPORT.md
**Público**: Stakeholders, revisores
**Conteúdo**:
- Template de relatório formal
- Tabelas de comparação
- Análise de discrepâncias
- Conclusões e recomendações
- Aprovação

**Quando preencher**: Após completar todos os testes

---

## 🎯 Objetivos da Validação

### Primários
1. ✅ Confirmar correção estatística dos modelos PanelBox
2. ✅ Estabelecer benchmarks de referência
3. ✅ Documentar limitações conhecidas

### Secundários
4. ✅ Criar suite de testes de regressão
5. ✅ Facilitar debugging futuro
6. ✅ Aumentar confiança dos usuários

---

## 🔍 Modelos Validados

| Categoria | Modelo | R Package | PanelBox Class |
|-----------|--------|-----------|----------------|
| **Binários** | Pooled Logit | `glm` | `PooledLogit` |
| | Pooled Probit | `glm` | `PooledProbit` |
| | FE Logit | `survival::clogit` | `FixedEffectsLogit` |
| **Censurados** | Pooled Tobit | `censReg::censReg` | `PooledTobit` |
| **Contagem** | Pooled Poisson | `glm` | `PooledPoisson` |
| | FE Poisson | `plm::pglm` | `PoissonFixedEffects` |
| | Negative Binomial | `MASS::glm.nb` | `NegativeBinomial` |

**Total**: 7 modelos

---

## ⚙️ Métricas Validadas

Para cada modelo, validamos:

| Métrica | Tolerância | Testes |
|---------|------------|--------|
| **Coeficientes** (β) | 5% | 14-21 coefs |
| **Erros Padrão** (SE) | 10% | 14-21 SEs |
| **Log-likelihood** | 0.1% | 7 valores |
| **Efeitos Marginais** | 10% | 10-14 MEs |
| **Parâmetros Auxiliares** | 5-15% | σ, θ, α |

**Total**: ~60-70 comparações numéricas

---

## 📊 Critérios de Aprovação

| Status | Taxa de Sucesso | Ação |
|--------|-----------------|------|
| ✅ **APROVADO** | ≥ 95% | Release sem ressalvas |
| ⚠️ **CONDICIONAL** | 85-95% | Release com documentação |
| ❌ **REPROVADO** | < 85% | Requer correções |

---

## 🛠️ Ferramentas Necessárias

### Software
- [x] Python ≥ 3.8
- [x] R ≥ 4.0
- [x] pytest
- [x] Pacotes Python: numpy, pandas, scipy, panelbox
- [x] Pacotes R: plm, censReg, MASS, survival, etc.

### Hardware
- CPU: Qualquer (testes são leves)
- RAM: 2GB+ recomendado
- Disco: ~100MB para dados e resultados

### Tempo
- Setup inicial: 30 min
- Geração de dados: 1 min
- Execução R: 5-10 min
- Testes Python: 2-5 min
- Análise: 30-60 min
- **Total**: ~1-2 horas

---

## 📞 Suporte

### Problemas com Instalação R
👉 Ver README_R_VALIDATION.md seção "Troubleshooting"

### Problemas com Scripts Python
👉 Verificar imports e paths em test_r_validation.py

### Testes Falhando
👉 Ver FASE5_CHECKLIST.md seção "Análise de Discrepâncias"

### Interpretação de Resultados
👉 Ver VALIDATION_REPORT.md seção "Known Discrepancies"

### Bugs no PanelBox
👉 Abrir issue no GitHub com:
- Output do teste que falhou
- Valores esperados vs obtidos
- Dados de teste (se possível)

---

## 📈 Status do Projeto

| Item | Status |
|------|--------|
| Estrutura de diretórios | ✅ Criada |
| Scripts R | ✅ Implementados |
| Script de dados | ✅ Implementado |
| Testes Python | ✅ Implementados |
| Documentação | ✅ Completa |
| Execução | ⏳ Pendente |
| Relatório | ⏳ Pendente |

**Próxima Ação**: Executar validação conforme README_R_VALIDATION.md

---

## 🔄 Manutenção Contínua

### Quando Executar Novamente

- ✅ Após modificações em modelos de resposta limitada
- ✅ Antes de cada release major
- ✅ Se usuários reportarem discrepâncias vs R
- ✅ Ao adicionar novos modelos

### Como Manter Atualizado

1. Re-executar benchmarks R periodicamente
2. Verificar se pacotes R foram atualizados
3. Ajustar tolerâncias se necessário
4. Adicionar novos casos de teste

---

## 📚 Referências

### Econometria
- Greene (2003) - *Econometric Analysis*
- Wooldridge (2010) - *Panel Data*
- Cameron & Trivedi (2005) - *Microeconometrics*

### R Packages
- [plm documentation](https://cran.r-project.org/package=plm)
- [censReg documentation](https://cran.r-project.org/package=censReg)
- [MASS documentation](https://cran.r-project.org/package=MASS)

### Papers
- Croissant & Millo (2008) - "Panel Data in R: The plm Package"

---

## ✅ Checklist Rápido

Antes de começar, verifique:

- [ ] R instalado e funcionando
- [ ] Todos os pacotes R instalados
- [ ] Python environment ativo
- [ ] PanelBox instalado
- [ ] pytest disponível

Durante execução:

- [ ] Dados gerados sem erros
- [ ] Scripts R executaram sem erros
- [ ] JSONs criados em r/results/
- [ ] Testes Python executaram
- [ ] Checklist preenchido

Após conclusão:

- [ ] Relatório preenchido
- [ ] Resultados revisados
- [ ] Discrepâncias documentadas
- [ ] Aprovação obtida

---

**Última Atualização**: 2025-XX-XX
**Versão**: 1.0
**Mantido por**: PanelBox Development Team

---

*Happy Validating! 🎉*
