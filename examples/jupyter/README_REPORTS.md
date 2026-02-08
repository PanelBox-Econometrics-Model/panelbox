# Notebook de Relatórios de Visualização

**Arquivo:** `06_visualization_reports.ipynb`

## Visão Geral

Este notebook demonstra o **sistema completo de relatórios de visualização** do PanelBox v0.5.0, incluindo:

- ✅ Relatórios de Validação
- ✅ Relatórios de Diagnósticos Residuais  
- ✅ Relatórios de Comparação de Modelos
- ✅ Temas Profissionais (Professional, Academic, Presentation)
- ✅ Exportação de Gráficos (PNG, SVG, PDF)
- ✅ Workflows Automatizados

## Estrutura do Notebook

### 1. Introdução e Setup
- Carregamento de bibliotecas
- Preparação de dados
- Estimação de modelos

### 2. Relatórios de Validação
- Criar gráficos de validação (5 tipos)
- Visualizar testes estatísticos
- Gerar relatório HTML interativo

### 3. Relatórios de Diagnósticos Residuais
- Criar 7 gráficos de diagnóstico
- Q-Q plot, residual vs fitted, etc.
- Gerar relatório HTML de diagnósticos

### 4. Relatórios de Comparação de Modelos
- Comparar múltiplos modelos visualmente
- Forest plots, coeficientes, critérios de informação
- Gerar relatório HTML de comparação

### 5. Temas e Personalização
- Demonstração dos 3 temas (Professional, Academic, Presentation)
- Comparação visual lado a lado
- Escolha do tema apropriado para cada caso

### 6. Exportação de Gráficos
- Exportação individual (PNG, SVG, PDF)
- Exportação em lote
- Exportação multi-formato
- Configurações para publicação

### 7. Workflows Automatizados
- Pipeline completo automatizado
- Comparação em lote de múltiplos modelos
- Funções reutilizáveis

## Como Usar

### Abrir no Jupyter

```bash
# Ativar ambiente virtual
source publish_env/bin/activate

# Iniciar Jupyter
jupyter notebook examples/jupyter/06_visualization_reports.ipynb
```

### Executar Células

O notebook está organizado em seções. Execute as células em ordem para:

1. Carregar dados e estimar modelos
2. Criar visualizações
3. Gerar relatórios HTML
4. Exportar gráficos

### Principais Comandos Demonstrados

```python
# Criar gráficos de validação
charts = create_validation_charts(
    validation_data=validation_results,
    theme='professional'
)

# Gerar relatório HTML de validação
report_mgr = ReportManager()
report_mgr.generate_validation_report(
    validation_data=validation_results,
    output_file='validation_report.html'
)

# Exportar gráficos
export_charts_multiple_formats(
    charts=charts,
    output_dir='output/figures',
    formats=['png', 'svg', 'pdf']
)
```

## Outputs Gerados

Ao executar o notebook completo, você terá:

### Relatórios HTML
- `output/reports/validation_report.html`
- `output/reports/residual_diagnostics.html`
- `output/reports/model_comparison.html`
- `output/complete_analysis/*.html`

### Gráficos Exportados
- `output/exports/validation/*.png`
- `output/exports/multi_format/*.{png,svg,pdf}`
- `output/complete_analysis/figures/*.{png,svg,pdf}`

### Total
- ~3 relatórios HTML interativos
- ~48+ arquivos de imagem em múltiplos formatos

## Formatos de Exportação

| Formato | Uso Recomendado | Características |
|---------|----------------|----------------|
| PNG | Web, apresentações | Boa qualidade, tamanho médio |
| SVG | Impressão, publicações | Vetorial, escalável |
| PDF | Artigos científicos | Alta qualidade, portável |
| JPEG | Web (comprimido) | Menor tamanho |
| WEBP | Web moderna | Melhor compressão |

## Temas Disponíveis

| Tema | Uso | Características |
|------|-----|----------------|
| **Professional** | Relatórios corporativos | Cores sóbrias, design limpo |
| **Academic** | Artigos científicos | Alta legibilidade, P&B friendly |
| **Presentation** | Apresentações | Cores vibrantes, alto contraste |

## Exemplos de Workflows

### Workflow Básico

```python
# 1. Estimar modelo
fe = pb.FixedEffects(formula, data, entity_col, time_col)
results = fe.fit()

# 2. Criar gráficos
charts = create_residual_diagnostics(results, theme='professional')

# 3. Exportar
export_charts(charts, 'output/', format='png')
```

### Workflow Completo Automatizado

```python
# Função que gera TUDO automaticamente
report_paths = generate_complete_analysis_report(
    data=data,
    formula="invest ~ value + capital",
    entity_col='firm',
    time_col='year',
    output_dir='output/complete_analysis',
    theme='professional'
)

# Resultado: 3 relatórios HTML + ~48 gráficos exportados
```

## Customização

### Alterar Tema

```python
# Trocar tema
charts = create_validation_charts(
    validation_data,
    theme='academic'  # ou 'presentation'
)
```

### Customizar Dimensões

```python
# Exportar com tamanho customizado
export_chart(
    chart,
    'output.png',
    width=2400,   # largura
    height=1600,  # altura
    scale=2.0     # resolução (2x = retina)
)
```

### Escolher Gráficos Específicos

```python
# Criar apenas alguns gráficos de diagnóstico
charts = create_residual_diagnostics(
    results,
    charts=['qq_plot', 'residual_vs_fitted']  # apenas estes
)
```

## Requisitos

- PanelBox >= 0.5.0
- plotly >= 6.0.0
- kaleido >= 1.2.0
- pandas >= 1.3.0
- jupyter ou jupyterlab

## Dicas

💡 **Melhor Qualidade:** Use `scale=2.0` para displays retina

💡 **Publicações:** Use SVG ou PDF (vetoriais, infinitamente escaláveis)

💡 **Web:** Use PNG ou WEBP (bom balanço qualidade/tamanho)

💡 **Apresentações:** Use PNG 16:9 (1920x1080)

💡 **Automação:** Use as funções de workflow para processar múltiplos modelos

## Recursos Adicionais

- **Documentação:** `desenvolvimento/REPORT/EXPORT_FUNCTIONALITY_GUIDE.md`
- **Exemplo Python:** `examples/export_charts_example.py`
- **Outros Notebooks:** 
  - `03_validation_complete.ipynb` - Testes de validação
  - `04_robust_inference.ipynb` - Inferência robusta
  - `05_report_generation.ipynb` - Relatórios com pandas

## Troubleshooting

### Erro: "kaleido not found"

```bash
pip install kaleido
```

### Erro: "Chart type not registered"

Certifique-se de ter instalado o PanelBox em modo development:

```bash
pip install -e . --no-deps
pip install plotly kaleido
```

### Gráficos não aparecem

No Jupyter, certifique-se de ter executado:

```python
chart.show()  # mostra o gráfico inline
```

---

**Versão:** PanelBox 0.5.0  
**Status:** Produção  
**Última Atualização:** 2026-02-07
