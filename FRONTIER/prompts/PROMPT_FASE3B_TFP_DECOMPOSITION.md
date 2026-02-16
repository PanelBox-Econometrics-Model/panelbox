# 🤖 Prompt para IA - FASE 3B: TFP Decomposition + Efeitos Marginais

**Projeto:** PanelBox - Módulo de Fronteira Estocástica (SFA)
**Fase:** 3B - Análise de Produtividade Total dos Fatores
**Duração Estimada:** 2 semanas
**Complexidade:** Média

---

## 📋 Contexto do Projeto

Você está implementando ferramentas de **análise de produtividade** para complementar os modelos SFA:

1. **TFP Decomposition:** Separar crescimento da produtividade em componentes
2. **Marginal Effects:** Completar efeitos marginais para todos os modelos

Estas são ferramentas essenciais para **aplicações empíricas** e **policy analysis**.

---

## 📚 PARTE 1: TFP Decomposition

### Teoria

**Produtividade Total dos Fatores (TFP):**
```
TFP_it = y_it / f(x_it)

Em logs:
ln(TFP_it) = ln(y_it) - ln(f(x_it))
```

**Decomposição do crescimento:**
```
Δ ln(TFP) = ΔTC + ΔTE + ΔSE

onde:
ΔTC = Mudança Técnica (Technical Change)
      → Deslocamento da fronteira ao longo do tempo

ΔTE = Mudança em Eficiência Técnica (Technical Efficiency Change)
      → Aproximação ou afastamento da fronteira

ΔSE = Mudança em Eficiência de Escala (Scale Efficiency Change)
      → Movimento ao longo da fronteira
```

### Implementação

#### Classe TFPDecomposition

```python
"""
Total Factor Productivity (TFP) decomposition for panel SFA models.

This module decomposes TFP growth into:
1. Technical change (frontier shift)
2. Technical efficiency change (catch-up)
3. Scale efficiency change (movement along frontier)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class TFPDecomposition:
    """Decompose TFP growth for panel SFA models.

    Decomposes productivity growth into:
        Δ ln(TFP) = ΔTC + ΔTE + ΔSE

    Works with any panel SFA model that provides:
        - Frontier estimates (β)
        - Efficiency estimates (TE_it)

    Parameters:
        result: SFResult from panel model
        periods: List of two periods to compare (default: first and last)

    Example:
        >>> result = model.fit()  # Panel model
        >>> tfp = TFPDecomposition(result)
        >>> decomp = tfp.decompose()
        >>> tfp.plot_decomposition()

    References:
        Kumbhakar, S. C., & Lovell, C. A. K. (2000).
            Stochastic Frontier Analysis. Cambridge University Press.
            Chapter 7: Productivity and its components.

        Färe, R., Grosskopf, S., Norris, M., & Zhang, Z. (1994).
            Productivity growth, technical progress, and efficiency change.
            American Economic Review, 84(1), 66-83.
    """

    def __init__(
        self,
        result,
        periods: Optional[Tuple[int, int]] = None,
    ):
        self.result = result
        self.model = result.model

        # Check that model is panel
        if not hasattr(self.model, 'entity') or self.model.entity is None:
            raise ValueError("TFP decomposition requires panel data model")

        # Get time periods
        unique_times = np.unique(self.model.data[self.model.time])
        if periods is None:
            # Compare first and last
            self.t1 = unique_times[0]
            self.t2 = unique_times[-1]
        else:
            self.t1, self.t2 = periods

        # Store data
        self.data = self.model.data
        self.depvar = self.model.depvar
        self.exog = self.model.exog

    def decompose(self) -> pd.DataFrame:
        """Compute TFP decomposition for all firms.

        Returns:
            DataFrame with columns:
                - entity: Firm identifier
                - delta_tfp: Total TFP change
                - delta_tc: Technical change
                - delta_te: Technical efficiency change
                - delta_se: Scale efficiency change
                - verification: delta_tfp - (delta_tc + delta_te + delta_se)
        """
        # Get data for both periods
        data_t1 = self.data[self.data[self.model.time] == self.t1].copy()
        data_t2 = self.data[self.data[self.model.time] == self.t2].copy()

        # Merge on entity
        df = data_t1.merge(
            data_t2,
            on=self.model.entity,
            suffixes=('_t1', '_t2'),
        )

        # Get efficiencies
        eff = self.result.efficiency(estimator='bc')
        eff_t1 = eff[eff.time == self.t1].set_index('entity')['efficiency']
        eff_t2 = eff[eff.time == self.t2].set_index('entity')['efficiency']

        results = []

        for _, row in df.iterrows():
            entity = row[self.model.entity]

            # Output change
            y_t1 = row[f'{self.depvar}_t1']
            y_t2 = row[f'{self.depvar}_t2']
            delta_y = y_t2 - y_t1  # Log difference

            # Input change (aggregate)
            x_t1 = np.array([row[f'{var}_t1'] for var in self.exog])
            x_t2 = np.array([row[f'{var}_t2'] for var in self.exog])

            # Frontier parameters (assumed constant or take average)
            beta = self.result.params[:len(self.exog)]

            # Input contribution (weighted by elasticities)
            delta_inputs = beta @ (x_t2 - x_t1)

            # TFP growth (Solow residual)
            delta_tfp = delta_y - delta_inputs

            # Component 1: Technical Efficiency Change
            te_t1 = eff_t1.loc[entity]
            te_t2 = eff_t2.loc[entity]
            delta_te = np.log(te_t2) - np.log(te_t1)

            # Component 2: Technical Change (frontier shift)
            # Approximate as unexplained growth minus TE change
            # (requires time-varying frontier for exact calculation)
            # For now, compute residual

            # Component 3: Scale Efficiency Change
            # Requires computing returns to scale
            rts = self._compute_returns_to_scale(beta, x_t1, x_t2)
            delta_se = self._compute_scale_efficiency_change(rts, delta_inputs)

            # Technical change (residual)
            delta_tc = delta_tfp - delta_te - delta_se

            results.append({
                'entity': entity,
                'delta_tfp': delta_tfp,
                'delta_tc': delta_tc,
                'delta_te': delta_te,
                'delta_se': delta_se,
                'verification': delta_tfp - (delta_tc + delta_te + delta_se),
            })

        return pd.DataFrame(results)

    def _compute_returns_to_scale(
        self,
        beta: np.ndarray,
        x_t1: np.ndarray,
        x_t2: np.ndarray,
    ) -> float:
        """Compute returns to scale.

        RTS = Σ β_j (output elasticities)

        If Cobb-Douglas: RTS = Σ β_j
        If Translog: RTS depends on output level
        """
        # For Cobb-Douglas (log-linear)
        rts = beta.sum()
        return rts

    def _compute_scale_efficiency_change(
        self,
        rts: float,
        delta_inputs: float,
    ) -> float:
        """Compute scale efficiency change.

        SE change depends on distance from optimal scale.

        If RTS = 1: CRS, no scale effect
        If RTS > 1: IRS, expansion increases SE
        If RTS < 1: DRS, expansion decreases SE
        """
        # Simplified calculation
        # Full calculation requires distance to optimal scale
        scale_effect = (rts - 1.0) * delta_inputs
        return scale_effect

    def aggregate_decomposition(self) -> Dict[str, float]:
        """Compute aggregate (mean) decomposition across all firms."""
        decomp = self.decompose()

        return {
            'mean_delta_tfp': decomp['delta_tfp'].mean(),
            'mean_delta_tc': decomp['delta_tc'].mean(),
            'mean_delta_te': decomp['delta_te'].mean(),
            'mean_delta_se': decomp['delta_se'].mean(),
            'pct_from_tc': 100 * decomp['delta_tc'].mean() / decomp['delta_tfp'].mean(),
            'pct_from_te': 100 * decomp['delta_te'].mean() / decomp['delta_tfp'].mean(),
            'pct_from_se': 100 * decomp['delta_se'].mean() / decomp['delta_tfp'].mean(),
        }

    def plot_decomposition(
        self,
        kind: str = 'bar',
        top_n: int = 20,
    ):
        """Plot TFP decomposition.

        Parameters:
            kind: 'bar' (stacked bars) or 'scatter'
            top_n: Number of firms to show (if > n_entities, show all)
        """
        import matplotlib.pyplot as plt

        decomp = self.decompose()

        if kind == 'bar':
            # Stacked bar chart
            decomp_sorted = decomp.nlargest(top_n, 'delta_tfp')

            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(decomp_sorted))
            width = 0.8

            # Plot components
            ax.bar(x, decomp_sorted['delta_tc'], width, label='Technical Change')
            ax.bar(
                x,
                decomp_sorted['delta_te'],
                width,
                bottom=decomp_sorted['delta_tc'],
                label='Efficiency Change',
            )
            ax.bar(
                x,
                decomp_sorted['delta_se'],
                width,
                bottom=decomp_sorted['delta_tc'] + decomp_sorted['delta_te'],
                label='Scale Effect',
            )

            # Total TFP as line
            ax.plot(x, decomp_sorted['delta_tfp'], 'ko-', linewidth=2, label='Total TFP Growth')

            ax.set_xlabel('Firm')
            ax.set_ylabel('Growth Rate')
            ax.set_title(f'TFP Decomposition (Period {self.t1} → {self.t2})')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            return fig

        elif kind == 'scatter':
            # Scatter plot: TE change vs TC
            fig, ax = plt.subplots(figsize=(8, 8))

            ax.scatter(decomp['delta_tc'], decomp['delta_te'], alpha=0.6)
            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax.axvline(0, color='k', linestyle='--', alpha=0.3)

            ax.set_xlabel('Technical Change (Frontier Shift)')
            ax.set_ylabel('Efficiency Change (Catch-up)')
            ax.set_title('Decomposition of TFP Growth')
            ax.grid(alpha=0.3)

            # Add quadrant labels
            ax.text(0.02, 0.98, 'Innovation\n+ Catch-up', transform=ax.transAxes, va='top')
            ax.text(0.02, 0.02, 'Catch-up\nonly', transform=ax.transAxes)
            ax.text(0.98, 0.98, 'Innovation\nonly', transform=ax.transAxes, ha='right', va='top')
            ax.text(0.98, 0.02, 'Decline', transform=ax.transAxes, ha='right')

            plt.tight_layout()
            return fig
```

---

## 📚 PARTE 2: Efeitos Marginais (Completar)

Já implementamos para Wang (2002) na FASE 2B. Agora vamos **generalizar** para todos os modelos.

### Expandir `utils/marginal_effects.py`

```python
def marginal_effects(
    result,
    method: str = "mean",
    var: Optional[str] = None,
    at_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute marginal effects on inefficiency or efficiency.

    Works for models with inefficiency determinants:
    - Wang (2002): μ_i = z_i'δ, ln(σ²_u,i) = w_i'γ
    - Battese & Coelli (1995): u_it ~ N⁺(z_it'δ, σ²_u)

    Parameters:
        result: SFResult from model with determinants
        method: Type of marginal effect
            'mean' - Effect on E[u_i]
            'efficiency' - Effect on E[TE_i]
            'variance' - Effect on Var[u_i] (Wang only)
        var: Specific variable (if None, compute for all)
        at_values: Values at which to evaluate (default: sample means)

    Returns:
        DataFrame with marginal effects and standard errors

    Example:
        >>> # BC95 model
        >>> model = StochasticFrontier(
        ...     ...,
        ...     inefficiency_vars=['age', 'education'],
        ... )
        >>> result = model.fit()
        >>> me = marginal_effects(result, method='mean')
        >>> print(me)
               variable  marginal_effect  std_error  z_stat  p_value
        0      age            0.023        0.005     4.6     0.000
        1      education     -0.015        0.007    -2.1     0.035

    References:
        Wang, H. J., & Schmidt, P. (2002).
            One-step and two-step estimation of the effects of exogenous
            variables on technical efficiency levels.
            Journal of Productivity Analysis, 18, 129-144.
    """
    model = result.model

    # Detect model type
    if hasattr(model, 'hetero_vars') and model.hetero_vars:
        # Wang (2002)
        from .marginal_effects_wang import marginal_effects_wang_2002
        return marginal_effects_wang_2002(result, method=method)

    elif hasattr(model, 'inefficiency_vars') and model.inefficiency_vars:
        # BC95 or similar
        return marginal_effects_bc95(result, method=method, at_values=at_values)

    else:
        raise ValueError(
            "Marginal effects require model with inefficiency determinants. "
            "Use inefficiency_vars parameter in model specification."
        )


def marginal_effects_bc95(
    result,
    method: str = "mean",
    at_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Marginal effects for BC95 model.

    Model: u_it ~ N⁺(z_it'δ, σ²_u)

    Effect on E[u_i]:
        ∂E[u_i] / ∂z_k = δ_k · Φ(δ_k) + φ(z_i'δ/σ_u)·(∂z_i'δ/∂z_k)/σ_u

    For location only (no scale):
        ∂E[u_i] / ∂z_k ≈ δ_k

    Parameters:
        result: SFResult from BC95 model
        method: 'mean' or 'efficiency'
        at_values: Specific values for evaluation

    Returns:
        DataFrame with marginal effects
    """
    model = result.model
    params = result.params

    # Extract δ parameters
    k = model.n_exog
    m = len(model.ineff_var_names)

    delta = params[k + 2 : k + 2 + m]  # After β and σ²_v, σ²_u

    if method == "mean":
        # For BC95 with location only:
        # ME is approximately delta
        me = delta

        # Standard errors from variance-covariance matrix
        vcov = result.vcov
        se = np.sqrt(np.diag(vcov[k + 2 : k + 2 + m, k + 2 : k + 2 + m]))

        # Z-statistics and p-values
        z_stat = me / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        df = pd.DataFrame({
            'variable': model.ineff_var_names,
            'marginal_effect': me,
            'std_error': se,
            'z_stat': z_stat,
            'p_value': p_value,
        })

        return df

    elif method == "efficiency":
        # Effect on E[TE_i]
        # This requires numerical differentiation
        raise NotImplementedError(
            "Marginal effects on efficiency for BC95 not yet implemented. "
            "Use method='mean' for effects on E[u]."
        )

    else:
        raise ValueError(f"Unknown method: {method}")
```

---

## ✅ CHECKLIST COMPLETO - FASE 3B

### TFP Decomposition

- [x] **1. Criar módulo**
  - [x] Criar `utils/decomposition.py`
  - [x] Classe `TFPDecomposition`
  - [x] Commit: "feat(frontier): Create TFP decomposition module"

- [x] **2. Implementar decomposição**
  - [x] Método `decompose()` para firmas individuais
  - [x] Cálculo de ΔTC, ΔTE, ΔSE
  - [x] Método `aggregate_decomposition()` para médias
  - [x] Commit: "feat(frontier): Implement TFP decomposition"

- [x] **3. Retornos de escala**
  - [x] Método `_compute_returns_to_scale`
  - [x] Suporte a Cobb-Douglas
  - [x] Suporte a Translog (ponto-específico)
  - [x] Commit: "feat(frontier): Add returns to scale computation"

- [x] **4. Visualizações**
  - [x] Método `plot_decomposition()` com stacked bars
  - [x] Scatter plot TC vs TE
  - [x] Decomposição ao longo do tempo
  - [x] Commit: "feat(frontier): Add TFP visualization methods"

- [x] **5. Integrar ao Result**
  - [x] Adicionar método `result.tfp_decomposition()`
  - [x] Disponível apenas para modelos de painel
  - [x] Commit: "feat(frontier): Integrate TFP into SFResult"

- [x] **6. Testes**
  - [x] Criar `tests/frontier/test_tfp_decomposition.py`
  - [x] Teste: soma dos componentes = TFP total
  - [x] Teste: valores razoáveis
  - [x] Teste com dados sintéticos
  - [x] Commit: "test(frontier): Add TFP decomposition tests"

### Efeitos Marginais (Completar)

- [x] **7. Generalizar função**
  - [x] Função `marginal_effects()` em `utils/marginal_effects.py`
  - [x] Detecção automática de modelo (Wang vs BC95)
  - [x] Dispatch apropriado
  - [x] Commit: "feat(frontier): Generalize marginal effects function"

- [x] **8. BC95 marginal effects**
  - [x] Implementar `marginal_effects_bc95`
  - [x] Efeito sobre E[u]
  - [x] Standard errors via delta method
  - [x] Z-stats e p-values
  - [x] Commit: "feat(frontier): Add BC95 marginal effects"

- [x] **9. Integração completa**
  - [x] Verificar que `result.marginal_effects()` funciona para:
    - [x] Wang (2002)
    - [x] BC95
    - [x] True FE/RE com BC95 (Four-Component model)
  - [x] Commit: "feat(frontier): Complete marginal effects integration"

- [x] **10. Testes**
  - [x] Atualizar `tests/frontier/test_marginal_effects.py`
  - [x] Teste para BC95
  - [x] Teste para Wang
  - [x] Comparar com diferenças finitas
  - [x] Commit: "test(frontier): Expand marginal effects tests"

### Documentação

- [x] **11. Docstrings**
  - [x] Docstring completo em `TFPDecomposition`
  - [x] Docstring em `marginal_effects`
  - [x] Exemplos práticos
  - [x] Commit: "docs(frontier): Add TFP and ME docstrings"

- [x] **12. Tutoriais**
  - [x] `examples/notebooks/tfp_decomposition.ipynb`
  - [x] Aplicação: crescimento de produtividade em manufatura
  - [x] Interpretação dos componentes
  - [x] Policy implications
  - [x] Commit: "docs(frontier): Add TFP tutorial"

- [x] **13. Exemplo integrado**
  - [x] `examples/productivity_analysis.py`
  - [x] Combinar TFP decomposition + marginal effects
  - [x] Análise completa de produtividade
  - [x] Commit: "docs(frontier): Add productivity analysis example"

---

## 🧪 Critérios de Validação

### TFP:
1. ✅ Σ componentes = Δ TFP (diff < 1e-6)
2. ✅ Componentes têm magnitude razoável
3. ✅ Plot gerado sem erros
4. ✅ Funciona para N > T e N < T

### Marginal Effects:
5. ✅ SEs positivos
6. ✅ P-values em [0, 1]
7. ✅ MEs comparáveis com diferenças finitas
8. ✅ Funciona para Wang e BC95

---

## 📦 Arquivos Criados/Modificados

```
panelbox/frontier/utils/
├── __init__.py                    # ✏️ Exports
├── decomposition.py               # 🆕 TFPDecomposition
└── marginal_effects.py            # ✏️ Expandir (já existe)

panelbox/frontier/
└── result.py                      # ✏️ +tfp_decomposition()

tests/frontier/
├── test_tfp_decomposition.py      # 🆕 Novo
└── test_marginal_effects.py       # ✏️ Expandir

examples/
├── productivity_analysis.py       # 🆕 Novo
└── notebooks/
    └── tfp_decomposition.ipynb    # 🆕 Novo
```

---

## 📚 Referências

### TFP Decomposition:
1. **Kumbhakar & Lovell (2000).** *Stochastic Frontier Analysis.* Cambridge University Press. Chapter 7.
2. **Färe et al. (1994).** "Productivity growth, technical progress, and efficiency change." *AER*, 84(1), 66-83.

### Marginal Effects:
3. **Wang & Schmidt (2002).** "One-step and two-step estimation." *JPA*, 18, 129-144.

---

## 💡 Dicas

1. **TFP:** Verifique sempre que componentes somam ao total!
2. **Scale effect:** Requer atenção ao tipo de função (CD vs Translog)
3. **ME para eficiência:** Mais complexo que ME para ineficiência
4. **Interpretação:** ΔTE positivo = catching up, ΔTC positivo = innovation

---

**Progresso:** 90% → 95%+ do roadmap total! 🎉

**Quase completo! Esta é a última implementação importante! 🏁**
