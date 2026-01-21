# Guia de Contribuição

Obrigado por considerar contribuir para o PanelBox! Este documento fornece diretrizes para contribuir com o projeto.

## Código de Conduta

Este projeto adere ao [Código de Conduta](CODE_OF_CONDUCT.md). Ao participar, você deve seguir este código.

## Como Contribuir

### Reportando Bugs

Se você encontrar um bug, por favor abra uma issue com:

- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs atual
- Versão do Python e do PanelBox
- Sistema operacional
- Código mínimo para reproduzir o erro

**Template de Bug Report:**

```markdown
**Descrição do Bug**
Descrição clara e concisa do bug.

**Para Reproduzir**
Passos para reproduzir:
1. Importar panelbox
2. Executar código X
3. Observar erro Y

**Comportamento Esperado**
O que deveria acontecer.

**Código para Reproduzir**
```python
import panelbox as pb
# código mínimo aqui
```

**Ambiente**
- Python version: 3.10
- PanelBox version: 0.1.0
- OS: Ubuntu 22.04

**Informações Adicionais**
Qualquer contexto adicional.
```

### Sugerindo Melhorias

Para sugerir melhorias ou novas funcionalidades:

1. Verifique se já não existe uma issue similar
2. Abra uma issue descrevendo:
   - Motivação para a melhoria
   - Descrição detalhada
   - Exemplos de uso propostos
   - Alternativas consideradas

### Pull Requests

#### Processo

1. **Fork** o repositório
2. **Clone** seu fork: `git clone https://github.com/seu-usuario/panelbox.git`
3. **Crie uma branch** para sua feature: `git checkout -b feature/MinhaFeature`
4. **Configure o ambiente de desenvolvimento**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate  # Windows
   pip install -e ".[dev]"
   ```
5. **Faça suas alterações** seguindo as diretrizes de código
6. **Adicione testes** para novas funcionalidades
7. **Execute os testes**:
   ```bash
   pytest
   pytest --cov=panelbox tests/
   ```
8. **Execute os linters**:
   ```bash
   black panelbox/ tests/
   isort panelbox/ tests/
   flake8 panelbox/ tests/
   mypy panelbox/
   ```
9. **Commit suas mudanças**:
   ```bash
   git commit -m "feat: Adiciona funcionalidade X"
   ```
10. **Push para sua branch**: `git push origin feature/MinhaFeature`
11. **Abra um Pull Request** no repositório principal

#### Diretrizes de Código

**Estilo de Código**

- Siga [PEP 8](https://peps.python.org/pep-0008/)
- Use [Black](https://black.readthedocs.io/) para formatação (line-length=100)
- Use [isort](https://pycqa.github.io/isort/) para ordenar imports
- Use type hints em todas as funções públicas
- Docstrings no estilo Google

**Exemplo de Docstring:**

```python
def estimate_model(
    data: pd.DataFrame,
    formula: str,
    entity_col: str,
    time_col: str
) -> PanelResults:
    """
    Estima um modelo de painel.

    Args:
        data: DataFrame com dados em formato long
        formula: Fórmula no estilo R (e.g., "y ~ x1 + x2")
        entity_col: Nome da coluna de entidade
        time_col: Nome da coluna de tempo

    Returns:
        Objeto PanelResults com resultados da estimação

    Raises:
        ValueError: Se a fórmula for inválida
        KeyError: Se colunas não existirem no DataFrame

    Examples:
        >>> data = load_grunfeld()
        >>> results = estimate_model(data, "invest ~ value", "firm", "year")
        >>> print(results.summary())
    """
    pass
```

**Commits**

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` Nova funcionalidade
- `fix:` Correção de bug
- `docs:` Mudanças na documentação
- `style:` Formatação (sem mudança de código)
- `refactor:` Refatoração de código
- `test:` Adição ou correção de testes
- `chore:` Manutenção (build, CI, etc.)

Exemplos:
```
feat: adiciona suporte para System GMM
fix: corrige cálculo de estatística de Hansen
docs: atualiza tutorial de modelos dinâmicos
test: adiciona testes para validação de fórmulas
```

**Testes**

- Toda nova funcionalidade deve ter testes
- Manter cobertura de testes ≥ 90%
- Usar pytest como framework
- Organizar testes em `tests/` espelhando `panelbox/`
- Testes unitários para funções individuais
- Testes de integração para workflows completos
- Testes de benchmark contra Stata/R quando aplicável

**Exemplo de Teste:**

```python
import pytest
import pandas as pd
import panelbox as pb
from panelbox.datasets import load_grunfeld

class TestFixedEffects:
    """Testes para o modelo Fixed Effects."""

    @pytest.fixture
    def data(self):
        """Fixture com dados de exemplo."""
        return load_grunfeld()

    def test_basic_estimation(self, data):
        """Testa estimação básica de FE."""
        model = pb.FixedEffects("invest ~ value", data, "firm", "year")
        results = model.fit()

        assert results.params is not None
        assert len(results.params) > 0
        assert results.nobs == len(data)

    def test_coefficients_match_stata(self, data):
        """Testa se coeficientes coincidem com Stata."""
        model = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
        results = model.fit()

        # Valores obtidos de Stata xtreg, fe
        expected_value = 0.1101
        expected_capital = 0.3100

        assert pytest.approx(results.params['value'], rel=1e-3) == expected_value
        assert pytest.approx(results.params['capital'], rel=1e-3) == expected_capital
```

#### Documentação

- Toda função/classe pública deve ter docstring
- Atualizar README.md se necessário
- Adicionar exemplos em `examples/` para funcionalidades complexas
- Atualizar documentação técnica em `docs/`
- Adicionar entrada no CHANGELOG.md

#### Code Review

Seu PR será revisado considerando:

- Qualidade do código
- Cobertura de testes
- Documentação
- Compatibilidade com versões Python suportadas
- Performance (se aplicável)
- Consistência com arquitetura do projeto

## Áreas para Contribuição

### Prioridade Alta
- [ ] Implementação de modelos core (Pooled OLS, FE, RE)
- [ ] Parser de fórmulas
- [ ] Testes de validação básicos
- [ ] Sistema de reports

### Prioridade Média
- [ ] Modelos dinâmicos (GMM)
- [ ] Testes de validação avançados
- [ ] Erros padrão robustos
- [ ] CLI

### Prioridade Baixa
- [ ] Testes de raiz unitária
- [ ] Testes de cointegração
- [ ] Otimizações de performance
- [ ] Integrações com outras bibliotecas

### Documentação
- [ ] Tutoriais em português e inglês
- [ ] Exemplos de uso
- [ ] Comparações com Stata/R
- [ ] Papers técnicos

## Comunicação

- **Issues**: Para bugs, melhorias e discussões
- **Pull Requests**: Para contribuições de código
- **Email**: gustavo.haase@gmail.com para questões privadas

## Reconhecimento

Todos os contribuidores serão reconhecidos no README.md e na documentação.

## Dúvidas?

Se tiver dúvidas sobre como contribuir, abra uma issue com a tag `question` ou entre em contato.

---

Obrigado por contribuir para o PanelBox! 🎉
