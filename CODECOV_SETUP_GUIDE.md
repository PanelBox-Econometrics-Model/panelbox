# Guia de Configuração do Codecov

## Status Atual
O badge do Codecov está aparecendo como "unknown" em https://app.codecov.io/gh/PanelBox-Econometrics-Model/panelbox/new

## Problemas Identificados

### 1. Token do Codecov (CRÍTICO)
O workflow requer `CODECOV_TOKEN` mas pode não estar configurado.

**Solução:**

1. Vá para https://codecov.io/ e faça login com sua conta GitHub
2. Adicione o repositório `PanelBox-Econometrics-Model/panelbox`
3. Copie o token fornecido pelo Codecov
4. No GitHub, vá para: `Settings` > `Secrets and variables` > `Actions` > `New repository secret`
5. Nome: `CODECOV_TOKEN`
6. Valor: Cole o token do Codecov
7. Clique em "Add secret"

### 2. Workflow de Upload Limitado
Atualmente, o upload só acontece para:
- OS: `ubuntu-latest`
- Python: `3.9`

Se os testes falharem nessa combinação específica, o coverage não será enviado.

### 3. Verificar se coverage.xml está sendo gerado
O comando atual gera `coverage.xml`, mas precisamos confirmar.

## Passos para Testar Localmente

```bash
# 1. Gerar coverage.xml localmente
poetry run pytest --cov=panelbox --cov-report=xml --cov-report=term-missing

# 2. Verificar se coverage.xml foi criado
ls -la coverage.xml

# 3. Ver o conteúdo (primeiras linhas)
head -20 coverage.xml
```

## Melhorias Sugeridas no Workflow

### Opção 1: Usar tokenless upload (recomendado para repos públicos)

Se o repositório for público, você pode remover o token e usar o upload tokenless do Codecov v4:

```yaml
- name: Upload coverage to Codecov
  if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.9'
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    flags: unittests
    name: codecov-umbrella
    fail_ci_if_error: false
    # Token não necessário para repos públicos com codecov-action@v4
```

### Opção 2: Adicionar debug e verbose

```yaml
- name: Upload coverage to Codecov
  if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.9'
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    flags: unittests
    name: codecov-umbrella
    token: ${{ secrets.CODECOV_TOKEN }}
    fail_ci_if_error: false
    verbose: true  # Adicionar para debug
```

### Opção 3: Upload para múltiplas combinações

Se quiser garantir que pelo menos uma combinação envie:

```yaml
- name: Upload coverage to Codecov
  if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.12'
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    flags: unittests-${{ matrix.python-version }}
    name: codecov-${{ matrix.os }}-${{ matrix.python-version }}
    token: ${{ secrets.CODECOV_TOKEN }}
    fail_ci_if_error: false
```

## Verificar Status do GitHub Actions

1. Vá para: https://github.com/PanelBox-Econometrics-Model/panelbox/actions
2. Clique no workflow "Tests" mais recente
3. Expanda o step "Upload coverage to Codecov"
4. Veja se há erros (token inválido, arquivo não encontrado, etc.)

## Badge do Codecov

O badge atual está correto:
```markdown
[![codecov](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox/branch/main/graph/badge.svg)](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox)
```

Mas você pode usar um badge mais moderno:
```markdown
[![codecov](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox)
```

## Checklist de Diagnóstico

- [ ] Token CODECOV_TOKEN configurado no GitHub Secrets
- [ ] Repositório adicionado no Codecov (https://codecov.io/)
- [ ] Workflow "Tests" executado com sucesso no GitHub Actions
- [ ] Step "Upload coverage to Codecov" executado (não pulado)
- [ ] Arquivo coverage.xml gerado nos testes
- [ ] Sem erros no log do upload do Codecov
- [ ] Badge atualizado após primeiro upload bem-sucedido

## Comandos Úteis

```bash
# Testar localmente a geração de coverage
poetry run pytest --cov=panelbox --cov-report=xml -v

# Ver estrutura do coverage.xml
cat coverage.xml | head -50

# Verificar coverage total
poetry run pytest --cov=panelbox --cov-report=term | grep TOTAL
```

## Próximos Passos

1. **PRIMEIRO**: Configure o token CODECOV_TOKEN no GitHub Secrets
2. **SEGUNDO**: Faça um push para a branch main para triggerar o workflow
3. **TERCEIRO**: Monitore o workflow no GitHub Actions
4. **QUARTO**: Verifique se o upload foi bem-sucedido nos logs
5. **QUINTO**: Aguarde alguns minutos e recarregue a página do Codecov

## Recursos

- Documentação Codecov: https://docs.codecov.com/docs
- codecov-action: https://github.com/codecov/codecov-action
- Troubleshooting: https://docs.codecov.com/docs/common-issues
