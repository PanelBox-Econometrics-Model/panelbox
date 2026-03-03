* =============================================================================
* Validation Script 03: Specification Tests
* PanelBox vs Stata
*
* Tests: Hausman, Breusch-Pagan LM, F-test for FE
* Datasets: nlswork.csv, firm_productivity.csv, grunfeld.csv
* =============================================================================

clear all
set more off

* =============================================================================
* PART A: NLS Work - Wage Equation
* =============================================================================

display "=== PART A: NLS Work - Wage Equation ==="

local nls_path "/home/guhaase/projetos/panelbox/examples/diagnostics/data/specification/nlswork.csv"
import delimited using "`nls_path'", clear

describe
summarize ln_wage experience tenure education union married

* Set panel structure
xtset idcode year
xtdescribe

* --- 1. Fixed Effects ---
display ""
display "=== Fixed Effects: ln_wage ~ experience tenure union married ==="
xtreg ln_wage experience tenure union married, fe
estimates store fe_nls

* --- 2. Random Effects ---
display ""
display "=== Random Effects: ln_wage ~ experience tenure union married ==="
xtreg ln_wage experience tenure union married, re
estimates store re_nls

* --- 3. Hausman Test ---
display ""
display "=== Hausman Test: FE vs RE ==="
display "H0: RE is consistent (no correlation between alpha_i and X)"
display "H1: RE is inconsistent, use FE"
hausman fe_nls re_nls

* --- 4. Breusch-Pagan LM Test ---
display ""
display "=== Breusch-Pagan LM Test: RE vs Pooled OLS ==="
display "H0: Var(alpha_i) = 0 (no individual effects)"
display "H1: Var(alpha_i) > 0 (individual effects present)"

* Need to re-estimate RE first
quietly xtreg ln_wage experience tenure union married, re
xttest0

* --- 5. F-test for Fixed Effects ---
display ""
display "=== F-test for Fixed Effects ==="
display "H0: All alpha_i = 0 (Pooled OLS)"
display "H1: At least one alpha_i != 0"
* The F-test is reported automatically in xtreg, fe output
quietly xtreg ln_wage experience tenure union married, fe
display "F-test: F(" e(df_a) "," e(df_r) ") = " e(F_f)
display "p-value = " Ftail(e(df_a), e(df_r), e(F_f))

* =============================================================================
* PART B: Grunfeld - Classic Panel Data
* =============================================================================

display ""
display "=== PART B: Grunfeld ==="

local grunfeld_path "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv"
import delimited using "`grunfeld_path'", clear

xtset firm year

* --- FE and RE ---
xtreg invest value capital, fe
estimates store fe_grun

xtreg invest value capital, re
estimates store re_grun

* --- Hausman Test ---
display ""
display "=== Hausman Test: Grunfeld ==="
hausman fe_grun re_grun

* --- Breusch-Pagan LM ---
display ""
display "=== Breusch-Pagan LM: Grunfeld ==="
quietly xtreg invest value capital, re
xttest0

* --- F-test ---
display ""
display "=== F-test for FE: Grunfeld ==="
quietly xtreg invest value capital, fe
display "F-test for individual effects:"
display "F(" e(df_a) "," e(df_r) ") = " e(F_f)

* =============================================================================
* PART C: Firm Productivity - Production Function
* =============================================================================

display ""
display "=== PART C: Firm Productivity ==="

local firm_path "/home/guhaase/projetos/panelbox/examples/diagnostics/data/specification/firm_productivity.csv"
import delimited using "`firm_path'", clear

xtset firm_id year

* --- FE and RE ---
display ""
display "=== FE: log_output ~ log_capital log_labor log_materials ==="
xtreg log_output log_capital log_labor log_materials, fe
estimates store fe_firm

display ""
display "=== RE: log_output ~ log_capital log_labor log_materials ==="
xtreg log_output log_capital log_labor log_materials, re
estimates store re_firm

* --- Hausman Test ---
display ""
display "=== Hausman Test: Firm Productivity ==="
hausman fe_firm re_firm

* --- Breusch-Pagan LM ---
display ""
display "=== Breusch-Pagan LM: Firm Productivity ==="
quietly xtreg log_output log_capital log_labor log_materials, re
xttest0

* =============================================================================
* PART D: Additional Diagnostic Tests
* =============================================================================

display ""
display "=== PART D: Additional Tests on Grunfeld ==="

* Reload Grunfeld
import delimited using "`grunfeld_path'", clear
xtset firm year

* --- Pesaran CD test for cross-sectional dependence ---
display ""
display "=== Pesaran CD Test for Cross-sectional Dependence ==="
quietly xtreg invest value capital, fe
xtcsd, pesaran abs

* --- Wooldridge test for serial correlation ---
display ""
display "=== Wooldridge Test for Serial Correlation ==="
display "H0: No first-order autocorrelation"
xtserial invest value capital

display ""
display "=== Specification Tests Complete ==="
