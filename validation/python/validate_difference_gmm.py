"""
Validation: Arellano-Bond (1991) Employment Equation
====================================================

Replicate the classic employment equation from Arellano & Bond (1991)
using PanelBox and compare with published Stata xtabond2 results.

Model:
    n_it = γ n_{i,t-1} + β_w w_it + β_wL w_{i,t-1} +
           β_k k_it + β_kL k_{i,t-1} +
           β_ys ys_it + β_ysL ys_{i,t-1} +
           time_dummies + η_i + ε_it

Dataset: abdata (140 UK firms, 1976-1984)
Reference: Review of Economic Studies, 58(2), 277-297

Expected Results (from literature):
-----------------------------------
Arellano-Bond (1991), Table 4, Column a1:
- One-step n(-1): ~0.686
- Two-step n(-1): ~0.629

Range check: Should be between 0.733 (LSDV) and 1.045 (OLS)
