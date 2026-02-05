# PanelBox Benchmarks

Este diretório contém benchmarks comparando PanelBox com implementações de referência em Stata e R.

---

## 📊 Estrutura

```
benchmarks/
├── stata_comparison/       # Comparação com Stata
│   ├── *.do               # Scripts Stata de referência
│   ├── test_*.py          # Testes Python correspondentes
│   └── results/           # Resultados das comparações
│
├── r_comparison/          # Comparação com R (plm)
│   ├── *.R                # Scripts R de referência
│   ├── test_*.py          # Testes Python correspondentes
│   └── results/           # Resultados das comparações
│
└── results/               # Resultados consolidados
    ├── benchmark_results.json
    └── BENCHMARK_REPORT.md
```

---

## 🎯 Objetivos

1. **Validação Numérica**: Garantir que PanelBox produz resultados idênticos ao Stata/R
2. **Identificação de Diferenças**: Documentar qualquer divergência metodológica
3. **Performance**: Comparar tempo de execução (secundário)

---

## 📝 Benchmarks Implementados

### Stata Comparison

| Modelo | Script Stata | Teste Python | Status |
|--------|--------------|--------------|--------|
| Pooled OLS | `pooled_ols.do` | `test_pooled_vs_stata.py` | ✅ Implementado |
| Fixed Effects | `fixed_effects.do` | `test_fe_vs_stata.py` | 🔄 Em progresso |
| Random Effects | `random_effects.do` | `test_re_vs_stata.py` | 🔄 Em progresso |
| Difference GMM | `diff_gmm.do` | `test_diff_gmm_vs_stata.py` | 🔄 Em progresso |
| System GMM | `sys_gmm.do` | `test_sys_gmm_vs_stata.py` | 🔄 Em progresso |

### R Comparison (Planejado)

| Modelo | Script R | Teste Python | Status |
|--------|----------|--------------|--------|
| Pooled OLS | `pooling.R` | `test_pooled_vs_plm.py` | ⏳ Planejado |
| Fixed Effects | `within.R` | `test_fe_vs_plm.py` | ⏳ Planejado |
| Random Effects | `random.R` | `test_re_vs_plm.py` | ⏳ Planejado |
| GMM | `pgmm.R` | `test_gmm_vs_plm.py` | ⏳ Planejado |

---

## 🚀 Como Executar

### Pré-requisitos

**Para Stata comparisons**:
- Stata 15+ instalado
- Pacote `xtabond2` (para GMM): `ssc install xtabond2`

**Para R comparisons**:
- R 4.0+ instalado
- Pacote `plm`: `install.packages("plm")`

**Para Python**:
```bash
pip install -e .  # Instalar PanelBox em modo desenvolvimento
```

### Executar Benchmarks Stata

#### Passo 1: Gerar Resultados de Referência no Stata

```bash
cd tests/benchmarks/stata_comparison

# Executar script Stata
stata -b do pooled_ols.do
stata -b do fixed_effects.do
stata -b do random_effects.do
stata -b do diff_gmm.do
stata -b do sys_gmm.do
```

Os resultados serão salvos em arquivos `.log`.

#### Passo 2: Atualizar Valores de Referência

Abra os arquivos `.log` gerados e copie os valores para os scripts Python correspondentes:
- Coeficientes
- Erros padrão
- Estatísticas de teste
- R-squared

Exemplo para `test_pooled_vs_stata.py`:
```python
stata_results = {
    'coef': {
        'value': 0.XXXXXXX,    # Copiar do .log
        'capital': 0.XXXXXXX,   # Copiar do .log
        'const': -XX.XXXXX      # Copiar do .log
    },
    # ... outros valores
}
```

#### Passo 3: Executar Testes Python

```bash
python3 test_pooled_vs_stata.py
python3 test_fe_vs_stata.py
python3 test_re_vs_stata.py
python3 test_diff_gmm_vs_stata.py
python3 test_sys_gmm_vs_stata.py
```

### Executar Benchmarks R

Similar ao Stata, mas usando scripts `.R` e pacote `plm`.

---

## 📏 Tolerâncias

Os benchmarks usam as seguintes tolerâncias para comparação:

| Métrica | Tolerância | Justificativa |
|---------|------------|---------------|
| **Coeficientes** | < 1e-6 (0.0001%) | Precisão numérica |
| **Erros Padrão** | < 1e-6 (0.0001%) | Precisão numérica |
| **Estatísticas de Teste** | < 1e-4 (0.01%) | Pequenas diferenças de arredondamento |
| **R-squared** | < 1e-6 | Precisão numérica |

Se as diferenças excederem essas tolerâncias, o benchmark **FALHA** e as diferenças devem ser investigadas.

---

## 🔍 Interpretando Resultados

### ✅ Benchmark PASSOU

```
✓ BENCHMARK PASSED: PanelBox matches Stata within tolerance (< 1e-6)
```

**Significado**: PanelBox produz resultados numericamente idênticos ao Stata/R.

### ✗ Benchmark FALHOU

```
✗ BENCHMARK FAILED: Differences exceed tolerance
```

**Possíveis causas**:
1. **Bug no PanelBox**: Implementação incorreta
2. **Diferença metodológica**: Escolhas algorítmicas diferentes (documentar)
3. **Versão diferente**: Stata/R/pacote de versão diferente
4. **Dados diferentes**: Dataset usado não é exatamente o mesmo
5. **Opções diferentes**: Configuração do modelo não é comparável

**Ação**: Investigar e documentar em `/results/differences/`.

---

## 📊 Relatório de Benchmarks

Após executar todos os benchmarks, gerar relatório consolidado:

```bash
python3 generate_benchmark_report.py
```

Isso criará:
- `results/benchmark_results.json`: Resultados em formato JSON
- `results/BENCHMARK_REPORT.md`: Relatório em markdown

---

## 📚 Datasets Usados

### Grunfeld Investment Data

- **Fonte**: Stata built-in (`grunfeld.dta`)
- **Descrição**: Investimento de 10 firmas US, 1935-1954
- **Variáveis**:
  - `company`: Firma ID (1-10)
  - `year`: Ano (1935-1954)
  - `invest`: Investimento bruto
  - `value`: Valor de mercado da firma
  - `capital`: Stock de capital
- **N**: 200 observações (10 × 20)
- **Balanced**: Sim

### Arellano-Bond Employment Data

- **Fonte**: Stata built-in (`abdata.dta`) ou R `plm::EmplUK`
- **Descrição**: Painel de firmas UK, 1976-1984
- **Uso**: Validação de GMM

---

## 🐛 Problemas Conhecidos

### Stata xtabond2

- **Versão**: Resultados podem variar ligeiramente entre versões
- **Solução**: Documentar versão usada (`which xtabond2`)
- **Orthogonal deviations**: Implementação pode diferir

### R plm

- **pgmm**: Pode usar convenções diferentes para instrumentos
- **Solução**: Verificar documentação e ajustar especificações

---

## 📖 Referências

### Stata
- **xtabond2**: Roodman, D. (2009). "How to Do xtabond2". *Stata Journal*, 9(1), 86-136.
- **Documentação**: https://www.stata.com/manuals/xt.pdf

### R (plm)
- **plm**: Croissant, Y., & Millo, G. (2008). "Panel Data Econometrics in R". *Journal of Statistical Software*, 27(2).
- **Documentação**: https://cran.r-project.org/web/packages/plm/

### Datasets
- **Grunfeld**: Grunfeld, Y. (1958). *The Determinants of Corporate Investment*.
- **Arellano-Bond**: Arellano, M., & Bond, S. (1991). "Some Tests of Specification for Panel Data". *Review of Economic Studies*, 58(2), 277-297.

---

## ✅ Checklist de Validação

Para cada modelo, garantir:

- [ ] Script Stata/R executa sem erros
- [ ] Resultados copiados corretamente para teste Python
- [ ] Teste Python executa sem erros
- [ ] Diferenças < tolerância
- [ ] Qualquer diferença > tolerância documentada
- [ ] Relatório atualizado

---

## 🤝 Contribuindo

Se você encontrar diferenças ou bugs nos benchmarks:

1. Documente a diferença detalhadamente
2. Verifique versões de software
3. Crie issue no GitHub com:
   - Modelo afetado
   - Valores esperados vs obtidos
   - Versões de Stata/R/Python
   - Script para reproduzir

---

**Última atualização**: 2026-02-05
**Status**: 🔄 Em desenvolvimento
**Fase**: 8.1 (Benchmarks Comparativos)
