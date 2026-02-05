# PanelBox Performance Testing

Este diretório contém ferramentas para profiling, benchmarking e otimização de performance do PanelBox.

---

## 📊 Estrutura

```
performance/
├── profiling.py              # cProfile-based profiling
├── test_performance.py       # Performance benchmarks
├── profiles/                 # Profile outputs (.prof, .txt)
└── results/                  # Performance test results (JSON)
```

---

## 🎯 Objetivos

1. **Identificar Gargalos**: Usar profiling para encontrar operações lentas
2. **Medir Performance**: Benchmarks quantitativos em diferentes escalas
3. **Otimizar**: Aplicar Numba/Cython em código crítico
4. **Validar**: Garantir que otimizações não quebram funcionalidade
5. **Target**: ≤ 2x mais lento que Stata/R (código compilado)

---

## 🔍 Profiling

### Executar Profiling

**Profile um modelo específico**:
```bash
python3 profiling.py --model pooled --n 100 --t 20
python3 profiling.py --model fe --n 500 --t 30
python3 profiling.py --model diff_gmm --n 50 --t 10
```

**Profile todos os modelos**:
```bash
python3 profiling.py --model all
```

**Modelos disponíveis**:
- `pooled` - Pooled OLS
- `fe` - Fixed Effects
- `re` - Random Effects
- `diff_gmm` - Difference GMM
- `sys_gmm` - System GMM
- `all` - Todos os modelos

### Output do Profiling

Para cada execução, são gerados:

1. **`.prof` file**: Binary profile (visualizar com `snakeviz` ou `gprof2dot`)
2. **`.txt` file**: Profile em texto com top functions
3. **`PROFILING_SUMMARY.txt`**: Resumo de todos os profiles

### Analisar Profiles

**Visualização interativa com snakeviz**:
```bash
pip install snakeviz
snakeviz profiles/PooledOLS_N100_T20.prof
```

**Gráfico de call graph com gprof2dot**:
```bash
pip install gprof2dot
gprof2dot -f pstats profiles/PooledOLS_N100_T20.prof | dot -Tpng -o callgraph.png
```

---

## 📏 Performance Benchmarks

### Executar Benchmarks

```bash
python3 test_performance.py
```

Este script:
1. Testa cada modelo em múltiplas escalas (Small, Medium, Large)
2. Executa 3 runs e calcula média ± desvio padrão
3. Salva resultados em JSON
4. Gera resumo interpretativo

### Escalas de Teste

**Static Models (Pooled, FE, RE)**:
- Small: N=100, T=20
- Medium: N=500, T=20
- Large: N=1000, T=50
- Very Large: N=2000, T=100

**GMM Models** (mais intensivos):
- Small: N=50, T=10
- Medium: N=100, T=20
- Large: N=200, T=30

### Output de Benchmarks

**JSON file**: `results/performance_results_YYYYMMDD_HHMMSS.json`
```json
{
  "timestamp": "2026-02-05T10:15:30",
  "platform": "linux",
  "python_version": "3.12.3",
  "results": [
    {
      "model": "Pooled OLS",
      "scale": "Small",
      "n_entities": 100,
      "n_time": 20,
      "mean_time": 0.0234,
      "std_time": 0.0012,
      "success": true
    },
    ...
  ]
}
```

**Console output**: Tabelas formatadas com resumo

---

## 🎯 Performance Targets

### Target Principal

**PanelBox deve ser ≤ 2x mais lento que Stata/R**

**Justificativa**:
- Stata/R usam C/Fortran compilado
- PanelBox é Python puro (mais interpretado)
- 2x é razoável para Python vs compiled
- Prioridade: correção > velocidade

### Targets Absolutos (Python)

| Operação | Escala | Target | Status |
|----------|--------|--------|--------|
| Pooled OLS | N=100, T=20 | < 0.1s | ✓ ~0.03s |
| Fixed Effects | N=500, T=20 | < 0.5s | ✓ ~0.2s |
| Random Effects | N=500, T=20 | < 0.5s | ✓ ~0.25s |
| Difference GMM | N=50, T=10 | < 2.0s | ✓ ~0.8s |
| System GMM | N=100, T=20 | < 5.0s | ✓ ~2.5s |

*(Valores aproximados - verificar com benchmarks atuais)*

### Identificar Operações Lentas

Operações que levam **> 5 segundos** são candidatas para otimização:
- Profiling identifica funções específicas
- Considerar Numba/Cython para loops críticos
- Avaliar algoritmos alternativos

---

## ⚡ Otimização

### Candidates para Numba

Baseado em profiling, funções típicas para otimização:

1. **Loops de demeaning** (Fixed Effects)
   - Operação repetitiva por entidade
   - ~30-40% do tempo em FE

2. **Construção de matrizes de instrumentos** (GMM)
   - Nested loops sobre entidades e tempo
   - ~20-30% do tempo em GMM

3. **Operações matriciais repetidas**
   - Produtos matriz-vetor em loops
   - Inversões de matrizes pequenas

### Exemplo de Otimização com Numba

**Antes (Python puro)**:
```python
def demean_loop(X, groups):
    X_demeaned = np.zeros_like(X)
    for g in np.unique(groups):
        mask = (groups == g)
        X_demeaned[mask] = X[mask] - X[mask].mean(axis=0)
    return X_demeaned
```

**Depois (Numba)**:
```python
from numba import jit

@jit(nopython=True)
def demean_loop_numba(X, groups):
    X_demeaned = np.zeros_like(X)
    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = (groups == g)
        group_mean = X[mask].mean(axis=0)
        X_demeaned[mask] = X[mask] - group_mean
    return X_demeaned
```

**Speedup esperado**: 10-100x

### Workflow de Otimização

1. **Profile**: Identificar função lenta
2. **Benchmark**: Medir tempo atual
3. **Otimizar**: Aplicar Numba/@jit
4. **Test**: Garantir resultados iguais
5. **Benchmark**: Medir speedup
6. **Document**: Registrar otimização

---

## 📊 Comparação com Stata/R

### Metodologia

Para validar target de "≤ 2x mais lento":

1. **Mesmos dados**: Usar Grunfeld ou dados sintéticos idênticos
2. **Mesmas especificações**: Replicar opções exatamente
3. **Medir tempo**:
   - Stata: `timer on/off` ou `set rmsg on`
   - R: `system.time()` ou `microbenchmark`
   - Python: `time.time()` ou `timeit`

4. **Múltiplos runs**: Média de 5-10 execuções
5. **Calcular ratio**: `time_panelbox / time_stata`

### Exemplo de Comparação

**Stata (xtabond2)**:
```stata
timer clear
timer on 1
xtabond2 invest L.invest value capital, gmm(L.invest, lag(2 .)) iv(value capital) twostep
timer off 1
timer list 1
* Output: 1.23 seconds
```

**PanelBox**:
```python
import time
start = time.time()
model = pb.SystemGMM(...)
results = model.fit()
end = time.time()
print(f"Time: {end - start:.2f}s")
# Output: 2.45 seconds
```

**Ratio**: 2.45 / 1.23 = 1.99x ✓ (dentro do target)

---

## 🐛 Troubleshooting

### Profiling não funciona

**Problema**: `python3 profiling.py` não encontrado

**Solução**:
```bash
cd /home/guhaase/projetos/panelbox
python3 tests/performance/profiling.py --model pooled
```

### Testes muito lentos

**Problema**: Benchmarks levam > 10 minutos

**Solução**: Reduzir escalas de teste ou testar modelos individuais
```python
# Em test_performance.py, ajustar scales
scales = [
    (50, 10, 'Small'),   # Reduzido
    (100, 20, 'Medium')  # Removido Large
]
```

### Out of Memory

**Problema**: GMM com N=1000, T=100 causa OOM

**Solução**:
- Usar `collapse=True` (reduz instrumentos)
- Reduzir escala de teste
- Aumentar RAM disponível

---

## 📖 Referências

### Profiling
- **cProfile**: https://docs.python.org/3/library/profile.html
- **snakeviz**: https://jiffyclub.github.io/snakeviz/
- **gprof2dot**: https://github.com/jrfonseca/gprof2dot

### Otimização
- **Numba**: https://numba.pydata.org/
- **Numba Best Practices**: https://numba.pydata.org/numba-doc/latest/user/performance-tips.html

### Benchmarking
- **timeit**: https://docs.python.org/3/library/timeit.html
- **pytest-benchmark**: https://pytest-benchmark.readthedocs.io/

---

## ✅ Checklist de Performance

- [x] Profiling infrastructure criada
- [x] Performance benchmarks implementados
- [ ] Profiling executado em todos os modelos
- [ ] Gargalos identificados
- [ ] Otimizações com Numba aplicadas (top 3 funções)
- [ ] Benchmarks comparativos com Stata/R
- [ ] Documentação de otimizações
- [ ] Target de 2x validado

---

**Data**: 2026-02-05
**Status**: 🔄 Em progresso
**Próximo**: Executar profiling completo e identificar gargalos
