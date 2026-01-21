# Guia Rápido de Início - PanelBox

Este guia vai ajudá-lo a configurar o ambiente de desenvolvimento do PanelBox e começar a implementar a biblioteca.

## 1. Configuração Inicial

### 1.1. Criar Estrutura de Diretórios

```bash
cd /home/guhaase/projetos/panelbox

# Executar script de setup
./setup_structure.sh
```

### 1.2. Criar Ambiente Virtual

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Linux/Mac)
source venv/bin/activate

# Ativar (Windows)
# venv\Scripts\activate
```

### 1.3. Instalar Dependências

```bash
# Instalar em modo de desenvolvimento
pip install -e ".[dev]"

# Ou instalar manualmente
pip install -r requirements-dev.txt
```

## 2. Estrutura do Projeto

```
panelbox/
├── PLANEJAMENTO_PANELBOX.md    # 📋 Planejamento completo (LEIA PRIMEIRO!)
├── README.md                    # 📖 Documentação do usuário
├── CONTRIBUTING.md              # 🤝 Guia de contribuição
├── CODE_OF_CONDUCT.md           # ⚖️  Código de conduta
├── setup_structure.sh           # 🏗️  Script de setup
├── panelbox/                    # 💻 Código-fonte
├── docs/                        # 📚 Documentação
├── papers/                      # 📄 Artigos científicos
├── examples/                    # 💡 Exemplos de uso
├── tests/                       # 🧪 Testes
└── planejamento/                # 📝 Documentos de planejamento
```

## 3. Roadmap de Implementação

### Fase 1: Core (MVP) - 2 meses

#### Semanas 1-2: PanelData e Formula Parser
- [ ] Implementar `panelbox/core/panel_data.py`
  - Classe PanelData
  - Validação de estrutura
  - Transformações básicas (demeaning, first difference, lag)

- [ ] Implementar `panelbox/core/formula_parser.py`
  - Parser de fórmulas estilo R
  - Suporte a transformações
  - Suporte a lags (para GMM)

**Arquivos a criar:**
- `panelbox/core/panel_data.py`
- `panelbox/core/formula_parser.py`
- `tests/core/test_panel_data.py`
- `tests/core/test_formula_parser.py`

#### Semanas 3-4: Pooled OLS
- [ ] Implementar `panelbox/models/base.py` (classe base)
- [ ] Implementar `panelbox/models/static/pooled_ols.py`
- [ ] Implementar `panelbox/core/results.py` (PanelResults)
- [ ] Testes unitários

**Arquivos a criar:**
- `panelbox/models/base.py`
- `panelbox/models/static/pooled_ols.py`
- `panelbox/core/results.py`
- `tests/models/test_pooled_ols.py`

#### Semanas 5-6: Fixed Effects
- [ ] Implementar `panelbox/models/static/fixed_effects.py`
- [ ] Demeaning (entity e time)
- [ ] Cálculo de R² (within, between, overall)
- [ ] Testes vs Stata/R

**Arquivos a criar:**
- `panelbox/models/static/fixed_effects.py`
- `tests/models/test_fixed_effects.py`
- `tests/benchmarks/stata_comparison/test_fe_vs_stata.py`

#### Semanas 7-8: Random Effects
- [ ] Implementar `panelbox/models/static/random_effects.py`
- [ ] Estimador Swamy-Arora
- [ ] GLS transformation
- [ ] Testes vs Stata/R

**Arquivos a criar:**
- `panelbox/models/static/random_effects.py`
- `tests/models/test_random_effects.py`

### Fase 2: Validação Básica - 1 mês

#### Semanas 9-10: Testes de Especificação
- [ ] Implementar `panelbox/validation/base.py`
- [ ] Implementar `panelbox/validation/specification/hausman.py`
- [ ] Implementar `panelbox/validation/specification/mundlak.py`
- [ ] Testes unitários

**Arquivos a criar:**
- `panelbox/validation/base.py`
- `panelbox/validation/validation_suite.py`
- `panelbox/validation/specification/hausman.py`
- `panelbox/validation/specification/mundlak.py`
- `tests/validation/test_hausman.py`

#### Semanas 11-12: Testes Diagnósticos
- [ ] Implementar testes de autocorrelação
- [ ] Implementar testes de heterocedasticidade
- [ ] Implementar ValidationReport

**Arquivos a criar:**
- `panelbox/validation/serial_correlation/wooldridge_ar.py`
- `panelbox/validation/heteroskedasticity/breusch_pagan.py`
- `panelbox/validation/heteroskedasticity/modified_wald.py`
- `panelbox/validation/validation_report.py`

### Fase 3: Sistema de Reports - 1.5 meses

#### Semanas 13-14: Arquitetura Base
- [ ] CSS de 3 camadas
- [ ] Templates base
- [ ] ReportManager
- [ ] TemplateManager, AssetManager, CSSManager

**Arquivos a criar:**
- `panelbox/templates/base_styles.css`
- `panelbox/templates/report_components.css`
- `panelbox/templates/common/header.html`
- `panelbox/templates/common/footer.html`
- `panelbox/report/report_manager.py`
- `panelbox/report/template_manager.py`
- `panelbox/report/asset_manager.py`
- `panelbox/report/css_manager.py`

#### Semanas 15-16: Renderers
- [ ] ValidationRenderer (interactive)
- [ ] ModelRenderer (interactive)
- [ ] Exportadores

**Arquivos a criar:**
- `panelbox/report/renderers/validation_renderer.py`
- `panelbox/report/renderers/model_renderer.py`
- `panelbox/report/exporters/html_exporter.py`
- `panelbox/report/exporters/latex_exporter.py`

#### Semanas 17-18: Templates e Polimento
- [ ] Templates de validation
- [ ] Templates de model
- [ ] Gráficos (Plotly e Matplotlib)
- [ ] Testes de reports

## 4. Comandos Úteis

### Desenvolvimento

```bash
# Formatar código
black panelbox/ tests/

# Ordenar imports
isort panelbox/ tests/

# Linting
flake8 panelbox/ tests/

# Type checking
mypy panelbox/
```

### Testes

```bash
# Todos os testes
pytest

# Com cobertura
pytest --cov=panelbox tests/

# Testes específicos
pytest tests/core/
pytest tests/models/test_fixed_effects.py

# Testes marcados
pytest -m unit
pytest -m "not slow"
```

### Documentação

```bash
# Servir documentação localmente
cd docs/
mkdocs serve

# Build documentação
mkdocs build
```

## 5. Fluxo de Trabalho Recomendado

### 5.1. Implementar Nova Funcionalidade

1. **Criar branch**
   ```bash
   git checkout -b feature/nome-da-feature
   ```

2. **Implementar código**
   - Seguir estilo do projeto
   - Adicionar type hints
   - Escrever docstrings

3. **Escrever testes**
   - Testes unitários
   - Testes de integração (se aplicável)
   - Testes de benchmark (se aplicável)

4. **Executar testes e linters**
   ```bash
   black panelbox/ tests/
   isort panelbox/ tests/
   flake8 panelbox/ tests/
   mypy panelbox/
   pytest --cov=panelbox tests/
   ```

5. **Commit e push**
   ```bash
   git add .
   git commit -m "feat: descrição da feature"
   git push origin feature/nome-da-feature
   ```

6. **Criar Pull Request**

### 5.2. Exemplo: Implementar Pooled OLS

```python
# panelbox/models/static/pooled_ols.py

import numpy as np
import pandas as pd
from typing import Optional
from panelbox.models.base import PanelModel
from panelbox.core.results import PanelResults

class PooledOLS(PanelModel):
    """
    Pooled OLS - ignora estrutura de painel.

    Args:
        formula: Fórmula no estilo R (e.g., "y ~ x1 + x2")
        data: DataFrame com dados em formato long
        entity_col: Nome da coluna de entidade
        time_col: Nome da coluna de tempo
        weights: Pesos opcionais

    Examples:
        >>> from panelbox.datasets import load_grunfeld
        >>> data = load_grunfeld()
        >>> model = PooledOLS("invest ~ value + capital", data, "firm", "year")
        >>> results = model.fit()
        >>> print(results.summary())
    """

    def fit(
        self,
        cov_type: str = 'nonrobust',
        cluster_col: Optional[str] = None
    ) -> PanelResults:
        """
        Estima o modelo Pooled OLS.

        Args:
            cov_type: Tipo de matriz de covariância
                - 'nonrobust': Padrão OLS
                - 'robust': White (HC0)
                - 'clustered': Cluster-robust
            cluster_col: Coluna de cluster (se cov_type='clustered')

        Returns:
            Objeto PanelResults com resultados da estimação
        """
        # TODO: Implementar
        pass

    def _estimate_coefficients(self) -> np.ndarray:
        """Implementação da estimação OLS."""
        # TODO: Implementar
        # y = X @ beta + e
        # beta_hat = (X'X)^(-1) X'y
        pass
```

```python
# tests/models/test_pooled_ols.py

import pytest
import pandas as pd
import numpy as np
from panelbox.models import PooledOLS
from panelbox.datasets import load_grunfeld

class TestPooledOLS:
    """Testes para o modelo Pooled OLS."""

    @pytest.fixture
    def data(self):
        """Fixture com dados Grunfeld."""
        return load_grunfeld()

    def test_basic_estimation(self, data):
        """Testa estimação básica."""
        model = PooledOLS("invest ~ value", data, "firm", "year")
        results = model.fit()

        assert results is not None
        assert results.params is not None
        assert len(results.params) == 2  # intercept + value

    def test_multiple_regressors(self, data):
        """Testa com múltiplos regressores."""
        model = PooledOLS("invest ~ value + capital", data, "firm", "year")
        results = model.fit()

        assert len(results.params) == 3  # intercept + value + capital

    def test_standard_errors_robust(self, data):
        """Testa erros padrão robustos."""
        model = PooledOLS("invest ~ value", data, "firm", "year")
        results = model.fit(cov_type='robust')

        assert results.std_errors is not None

    @pytest.mark.benchmark
    def test_matches_stata(self, data):
        """Verifica se resultados coincidem com Stata."""
        # Valores obtidos de: reg invest value capital
        # em Stata com dados Grunfeld
        pass
```

## 6. Recursos Importantes

### Documentos de Referência
- `PLANEJAMENTO_PANELBOX.md`: Planejamento técnico completo
- `README.md`: Visão geral para usuários
- `CONTRIBUTING.md`: Como contribuir

### Referências Técnicas
- Wooldridge (2010): Econometric Analysis of Cross Section and Panel Data
- Baltagi (2021): Econometric Analysis of Panel Data
- Stata xtabond2: Referência para GMM
- R plm: Referência para modelos estáticos

### Benchmarks
- Comparar contra Stata: `xtreg`, `xtabond2`
- Comparar contra R: pacote `plm`

## 7. Próximos Passos

1. ✅ Revisar `PLANEJAMENTO_PANELBOX.md`
2. ✅ Executar `./setup_structure.sh`
3. ✅ Configurar ambiente virtual
4. ⬜ Começar Fase 1: Core
   - Implementar PanelData
   - Implementar Formula Parser
   - Implementar Pooled OLS
   - Implementar Fixed Effects
   - Implementar Random Effects
5. ⬜ Começar Fase 2: Validação
6. ⬜ Começar Fase 3: Reports

## 8. Dúvidas?

- Abra uma issue no GitHub
- Envie email: gustavo.haase@gmail.com
- Consulte `CONTRIBUTING.md`

---

**Boa sorte com a implementação! 🚀**
