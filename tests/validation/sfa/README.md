# Validação SFA - Resultados de Referência do R

Este diretório contém scripts para gerar resultados de referência do R para validação da implementação PanelBox SFA.

## Estrutura

```
tests/validation/sfa/
├── README.md                           # Este arquivo
├── generate_r_frontier_results.R       # Script para pacote `frontier`
├── generate_r_sfaR_results.R          # Script para pacote `sfaR` (TRE/BC95)
├── test_r_frontier_validation.py      # Testes de validação Python
├── test_r_sfaR_validation.py          # Testes de validação TRE/BC95
├── data/                              # Datasets de referência
│   └── riceProdPhil.csv              # Rice production data
├── r_results/                         # Resultados R (gerados)
│   ├── r_frontier_*.csv
│   ├── r_sfaR_*.csv
│   └── r_session_info.txt
└── stata_logs/                        # Logs Stata (para US-6.2)
```

## Pré-requisitos

### R e Pacotes

1. **Instalar R** (versão ≥ 4.0.0):
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install r-base r-base-dev

   # macOS
   brew install r
   ```

2. **Instalar pacotes R necessários**:
   ```r
   install.packages(c("frontier", "sfaR", "readr", "plm"))
   ```

### Python e Dependências

```bash
pip install pandas numpy scipy pytest panelbox
```

## Uso

### 1. Gerar Resultados de Referência do R

Execute os scripts R para gerar os resultados de referência:

```bash
cd tests/validation/sfa

# Pacote frontier (cross-section e painel clássico)
Rscript generate_r_frontier_results.R

# Pacote sfaR (TRE e BC95)
Rscript generate_r_sfaR_results.R
```

Os resultados serão salvos em `r_results/`:
- `r_frontier_*.csv` - Parâmetros, eficiências, log-likelihood
- `r_sfaR_*.csv` - TRE e BC95 results
- `r_session_info.txt` - Informações da sessão R (versões)

### 2. Executar Testes de Validação Python

Com os resultados R gerados, execute os testes de validação:

```bash
# Validar contra frontier package
pytest test_r_frontier_validation.py -v

# Validar contra sfaR package
pytest test_r_sfaR_validation.py -v

# Executar todos os testes de validação
pytest . -v
```

### 3. Interpretar Resultados

Os testes comparam:
- **Coeficientes β**: Tolerância ± 1e-4
- **Componentes de variância** (σ²_v, σ²_u, γ): ± 1e-3
- **Log-likelihood**: ± 1e-2
- **Eficiências**: ± 1e-3

**Critérios de aceitação:**
- ✅ **PASS**: Diferença dentro da tolerância
- ⚠️ **WARNING**: Diferença ligeiramente acima (< 2× tolerância)
- ❌ **FAIL**: Diferença significativa (> 2× tolerância)

Diferenças pequenas podem ocorrer devido a:
- Algoritmos de otimização diferentes (R usa `nlm`, Python usa `scipy.optimize`)
- Starting values diferentes
- Critérios de convergência diferentes
- Precisão numérica (float64 vs R numeric)

## Modelos Validados

### Cross-section SFA
- ✅ Half-normal distribution
- ✅ Exponential distribution (via Python DGP)
- ✅ Truncated normal distribution

### Panel SFA - Clássicos
- ✅ Pitt & Lee (1981) - Time-invariant inefficiency
- ✅ Battese & Coelli (1992) - Time-varying (decay)
- ⚠️ Battese & Coelli (1995) - Determinants (sfaR)

### Panel SFA - True Models
- ⚠️ True Fixed Effects (TFE) - Greene (2005)
- ⚠️ True Random Effects (TRE) - Greene (2005)

**Legenda:**
- ✅ Validado contra R (tolerância ≤ 1e-4)
- ⚠️ Validado parcialmente (diferenças conhecidas)
- ❌ Não validado ainda

## Datasets de Referência

### 1. Rice Production Philippines
- **Fonte**: Battese & Coelli (1992), pacote `frontier`
- **Descrição**: Produção de arroz em fazendas filipinas
- **Dimensões**: 43 firmas × 8 anos = 344 observações
- **Variáveis**:
  - `PROD`: Produção de arroz (kg)
  - `AREA`: Área plantada (hectares)
  - `LABOR`: Trabalho (homem-dias)
  - `NPK`: Fertilizante NPK (kg)
  - `OTHER`: Outros insumos (pesos filipinos)
- **Uso**: Cross-section (agregado) e painel (completo)

### 2. (Adicionar mais datasets aqui)

## Troubleshooting

### Erro: "Package 'frontier' not found"
```r
install.packages("frontier")
```

### Erro: "Package 'sfaR' not found"
```r
install.packages("sfaR")
```

### Scripts R não executam
Certifique-se de que:
1. R está instalado: `R --version`
2. Rscript está no PATH: `which Rscript`
3. Pacotes estão instalados: `Rscript -e "library(frontier); library(sfaR)"`

### Testes Python falham
1. Verifique se os resultados R foram gerados: `ls r_results/`
2. Verifique instalação PanelBox: `python -c "import panelbox.frontier"`
3. Execute com verbose: `pytest -vv --tb=short`

## Referências

### Pacotes R
- **frontier**: Coelli, T. J., & Henningsen, A. (2020). frontier: Stochastic Frontier Analysis. R package version 1.1-8.
- **sfaR**: Dakpo, K. H., Desjeux, Y., & Henningsen, A. (2021). sfaR: Stochastic Frontier Analysis Routines. R package version 0.1.1.

### Papers
- Battese, G. E., & Coelli, T. J. (1992). Frontier production functions, technical efficiency and panel data: with application to paddy farmers in India. *Journal of Productivity Analysis*, 3(1-2), 153-169.
- Battese, G. E., & Coelli, T. J. (1995). A model for technical inefficiency effects in a stochastic frontier production function for panel data. *Empirical Economics*, 20(2), 325-332.
- Greene, W. H. (2005). Reconsidering heterogeneity in panel data estimators of the stochastic frontier model. *Journal of Econometrics*, 126(2), 269-303.
- Pitt, M. M., & Lee, L. F. (1981). The measurement and sources of technical inefficiency in the Indonesian weaving industry. *Journal of Development Economics*, 9(1), 43-64.

## Contribuindo

Para adicionar novos datasets ou modelos:

1. Adicionar script R em `generate_r_<nome>.R`
2. Salvar dataset em `data/<nome>.csv`
3. Salvar resultados em `r_results/r_<nome>_*.csv`
4. Criar teste Python em `test_r_<nome>_validation.py`
5. Documentar neste README

## Contato

Para questões sobre validação SFA, abrir issue no repositório PanelBox.
