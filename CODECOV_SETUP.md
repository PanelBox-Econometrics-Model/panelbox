# Codecov Setup Guide - Automatic Coverage Badges

**Date**: 2026-02-08
**Status**: ⏳ **PENDING ACTIVATION**

---

## 🎯 O Que Foi Feito

### 1. Badges Atualizados no README

**Antes** (badges estáticos):
```markdown
[![Tests](https://img.shields.io/badge/tests-1257%20passed-success.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-30%25-yellow.svg)]()
```

**Depois** (badges dinâmicos):
```markdown
[![Tests](https://github.com/PanelBox-Econometrics-Model/panelbox/workflows/Tests/badge.svg)](...)
[![codecov](https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox/branch/main/graph/badge.svg)](...)
```

### 2. Arquivo codecov.yml Criado

Configuração do Codecov para:
- Precisão de 2 casas decimais
- Range de 60-100% (cores)
- Ignorar arquivos de teste e docs
- Comentários automáticos em PRs

### 3. GitHub Actions Já Configurado

O workflow `.github/workflows/tests.yml` **já tem integração com Codecov**:
```yaml
- name: Upload coverage to Codecov
  if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.9'
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    flags: unittests
    name: codecov-umbrella
    fail_ci_if_error: false
```

---

## 🚀 Como Ativar o Codecov

### Passo 1: Acessar Codecov

1. Vá para: https://codecov.io
2. Clique em **"Sign up with GitHub"**
3. Faça login com sua conta GitHub

### Passo 2: Adicionar Repositório

1. No dashboard do Codecov, clique em **"Add Repository"**
2. Procure por: `PanelBox-Econometrics-Model/panelbox`
3. Clique em **"Setup repo"**

### Passo 3: Configurar Token (Opcional)

**Nota**: Para repositórios públicos, o token NÃO é necessário!

Se quiser adicionar token (opcional):
1. No Codecov, vá em Settings do repositório
2. Copie o **Upload Token**
3. No GitHub, vá em Settings → Secrets → New repository secret
4. Nome: `CODECOV_TOKEN`
5. Valor: Cole o token copiado

### Passo 4: Executar CI

Faça um commit qualquer para disparar o CI:
```bash
git commit --allow-empty -m "chore: trigger CI for Codecov"
git push origin main
```

### Passo 5: Verificar

1. Aguarde o CI completar (GitHub Actions)
2. Visite: https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox
3. Verifique se o coverage apareceu
4. O badge no README deve atualizar automaticamente

---

## 📊 O Que os Badges Fazem

### Badge de Tests (GitHub Actions)

**URL**: `https://github.com/PanelBox-Econometrics-Model/panelbox/workflows/Tests/badge.svg`

**Atualiza automaticamente quando**:
- ✅ Push para main ou develop
- ✅ Pull requests
- ✅ Workflow de testes completa

**Estados possíveis**:
- 🟢 **Passing**: Todos os testes passaram
- 🔴 **Failing**: Algum teste falhou
- 🟡 **Pending**: Testes em execução
- ⚫ **No status**: CI não executou ainda

**Clicando no badge**: Leva para a página de Actions do GitHub

### Badge de Coverage (Codecov)

**URL**: `https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox/branch/main/graph/badge.svg`

**Atualiza automaticamente quando**:
- ✅ CI completa e envia coverage.xml
- ✅ Codecov processa o relatório

**Mostra**:
- Percentual de cobertura atual
- Cor baseada no percentual:
  - 🔴 < 60%: Vermelho
  - 🟡 60-80%: Amarelo
  - 🟢 > 80%: Verde

**Clicando no badge**: Leva para o dashboard do Codecov com:
- Gráficos de cobertura ao longo do tempo
- Cobertura por arquivo
- Linhas não cobertas
- Diff de coverage entre commits

---

## 🎨 Vantagens dos Badges Automáticos

### 1. Sempre Atualizados
- ❌ Antes: Precisava atualizar manualmente
- ✅ Agora: Atualiza automaticamente a cada push

### 2. Dados Reais
- ❌ Antes: Podia ficar desatualizado
- ✅ Agora: Sempre reflete o estado atual

### 3. CI/CD Integration
- ✅ Mostra se testes estão passando
- ✅ Mostra tendência de coverage
- ✅ Alertas quando coverage diminui

### 4. Transparência
- ✅ Usuários veem status real do projeto
- ✅ Contribuidores sabem se CI está OK
- ✅ Profissionalismo e confiança

---

## 📈 Dashboard do Codecov

Após ativar, você terá acesso a:

### Gráficos
- **Coverage over time**: Evolução da cobertura
- **Sunburst chart**: Cobertura por módulo
- **File browser**: Cobertura arquivo por arquivo

### Relatórios
- **Commit comparison**: Diff de coverage entre commits
- **Pull request comments**: Comentários automáticos em PRs
- **Coverage reports**: Relatórios detalhados

### Métricas
- **Project coverage**: Cobertura geral
- **Patch coverage**: Cobertura do código novo
- **Complexity**: Complexidade ciclomática
- **Files changed**: Arquivos com mudanças

---

## 🔧 Configuração Avançada

### codecov.yml Explicado

```yaml
coverage:
  precision: 2              # 30.25% (2 casas decimais)
  round: down              # Arredonda para baixo
  range: "60...100"        # Verde >80%, Amarelo 60-80%, Vermelho <60%

  status:
    project:
      target: auto         # Meta de coverage (auto = manter atual)
      threshold: 1%        # Tolerância de 1% de queda

ignore:
  - "tests/**/*"          # Não contar testes na cobertura
  - "examples/**/*"       # Não contar exemplos
  - "docs/**/*"           # Não contar docs
```

### Comentários em Pull Requests

Codecov pode adicionar comentários automáticos em PRs mostrando:
- Mudança de coverage
- Arquivos com cobertura reduzida
- Linhas não cobertas no diff

Para ativar:
```yaml
comment:
  layout: "reach,diff,flags,tree"
  behavior: default
  require_changes: false
```

---

## 🎯 Cobertura Atual do Projeto

### Cobertura Total: ~30%

**Módulos com Alta Cobertura (>75%)**:
- panelbox/experiment/panel_experiment.py: 79%
- panelbox/experiment/results/base.py: 83%
- panelbox/experiment/results/residual_result.py: 86%
- panelbox/experiment/tests/validation_test.py: 79%
- panelbox/experiment/tests/comparison_test.py: 79%
- panelbox/report/validation_transformer.py: 83%
- panelbox/validation/cross_sectional_dependence/: 76-94%
- panelbox/validation/serial_correlation/: 81-91%

**Módulos com Baixa Cobertura (<30%)**:
- panelbox/gmm/: 9-28%
- panelbox/models/static/: 13-63%
- panelbox/visualization/: 10-77%
- panelbox/report/exporters/: 6-21%

**Oportunidades de Melhoria**:
1. Adicionar testes para GMM (prioridade alta)
2. Melhorar testes de Static Models
3. Testar exporters (LaTeX, Markdown)

---

## ✅ Checklist de Ativação

### Setup no Codecov
- [ ] Acessar https://codecov.io
- [ ] Login com GitHub
- [ ] Adicionar repositório PanelBox-Econometrics-Model/panelbox
- [ ] (Opcional) Configurar token como secret no GitHub

### Verificação
- [ ] Fazer commit para disparar CI
- [ ] Aguardar GitHub Actions completar
- [ ] Verificar upload no Codecov
- [ ] Checar badge no README atualizado
- [ ] Explorar dashboard do Codecov

### Manutenção
- [ ] Monitorar coverage nos PRs
- [ ] Revisar relatórios semanalmente
- [ ] Meta: Aumentar para 50%+ em 3 meses

---

## 🔗 Links Úteis

### Codecov
- **Dashboard**: https://codecov.io/gh/PanelBox-Econometrics-Model/panelbox
- **Docs**: https://docs.codecov.com/docs
- **Badge Guide**: https://docs.codecov.com/docs/status-badges

### GitHub Actions
- **Workflow**: https://github.com/PanelBox-Econometrics-Model/panelbox/actions/workflows/tests.yml
- **Runs**: https://github.com/PanelBox-Econometrics-Model/panelbox/actions

### Badges
- **Tests Badge**: Atualiza automaticamente
- **Coverage Badge**: Atualiza após CI + Codecov

---

## 💡 Dicas

### Para Aumentar Coverage

1. **Priorize módulos críticos**:
   - GMM (core functionality)
   - Static Models (widely used)

2. **Escreva testes unitários**:
   ```python
   def test_difference_gmm():
       model = DifferenceGMM(...)
       result = model.fit()
       assert result.hansen_j.pvalue > 0.10
   ```

3. **Use pytest-cov**:
   ```bash
   pytest --cov=panelbox --cov-report=html
   # Abra htmlcov/index.html para ver detalhes
   ```

4. **Foque em linhas não cobertas**:
   - Codecov mostra exatamente quais linhas não têm testes
   - Priorize paths críticos

### Para Manter CI Verde

1. **Rode testes localmente**:
   ```bash
   pytest tests/
   ```

2. **Verifique formatação**:
   ```bash
   black panelbox/
   isort panelbox/
   ```

3. **Use pre-commit hooks** (já configurado):
   ```bash
   pre-commit run --all-files
   ```

---

## 🎉 Resultado Final

Após ativar o Codecov, você terá:

### Badges Dinâmicos
```
[Tests: Passing ✅] [Coverage: 30% 🟡]
```

### Dashboard Rico
- Gráficos de tendência
- Cobertura por arquivo
- Histórico de commits
- Comparação de PRs

### Automação
- Badges atualizam sozinhos
- Comentários em PRs
- Alertas de queda de coverage
- Relatórios detalhados

---

**Status**: ⏳ **Pronto para ativar no Codecov**

Siga os passos acima para ativar e os badges começarão a atualizar automaticamente!

**Made with ❤️ using PanelBox v0.8.0**
