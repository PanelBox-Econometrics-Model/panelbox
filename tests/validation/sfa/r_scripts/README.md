# R Scripts for SFA Validation

This directory contains R scripts to generate reference results for validating the PanelBox SFA implementation.

## Requirements

Install required R packages:

```R
install.packages("frontier")
install.packages("sfaR")
install.packages("readr")
```

## Generating Reference Results

### 1. Generate frontier package results

```bash
cd /home/guhaase/projetos/panelbox/tests/validation/sfa/r_scripts
Rscript generate_r_frontier_results.R
```

This generates:
- `data/riceProdPhil.csv` - Raw rice production data
- `data/r_frontier_halfnormal_*.csv` - Half-normal cross-section results
- `data/r_frontier_truncnormal_*.csv` - Truncated normal results
- `data/r_frontier_cost_*.csv` - Cost frontier results
- `r_session_info.txt` - R session information

### 2. Generate sfaR package results

```bash
Rscript generate_r_sfaR_results.R
```

This generates:
- `data/r_sfaR_pittlee_*.csv` - Pitt-Lee (1981) panel results
- `data/r_sfaR_bc92_*.csv` - Battese-Coelli (1992) results
- `data/r_sfaR_tre_*.csv` - True Random Effects results
- `data/r_sfaR_tfe_*.csv` - True Fixed Effects results
- `r_sfaR_session_info.txt` - R session information

## Output Format

Each model generates three CSV files:

1. `*_params.csv` - Parameter estimates with standard errors
   - Columns: parameter, estimate, se, tvalue, pvalue

2. `*_efficiencies.csv` - Efficiency estimates
   - Columns: firm_id, efficiency, te_jlms (when available), ci_lower, ci_upper

3. `*_loglik.csv` - Model fit statistics
   - Columns: loglik, aic, bic, nobs

## Validation Tolerances

When comparing PanelBox results with R:

- **Coefficients (β):** ± 1e-4 (relative tolerance)
- **Variance components (σ²_v, σ²_u):** ± 1e-3
- **Efficiencies:** ± 1e-3
- **Log-likelihood:** ± 1e-2

Small differences are expected due to:
- Different optimization algorithms (Python scipy vs R optim)
- Different starting values
- Numerical precision differences

## Session Info

Always check the session info files to ensure reproducibility:
- R version
- Package versions
- Platform information

## Troubleshooting

If scripts fail:

1. **Missing packages:** Run `install.packages()` for missing packages
2. **Data loading errors:** Ensure `frontier` package is loaded before accessing data
3. **Convergence issues:** Some models may need adjusted starting values

## References

- Coelli, T. (1996). frontier: A Computer Program for Stochastic Frontier Production and Cost Function Estimation.
- Belotti, F., et al. (2013). Stochastic frontier analysis using Stata. The Stata Journal.
- Greene, W. (2005). Reconsidering heterogeneity in panel data estimators of the stochastic frontier model. Journal of Econometrics.
