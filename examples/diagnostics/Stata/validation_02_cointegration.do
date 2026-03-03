* =============================================================================
* Validation Script 02: Panel Cointegration Tests
* PanelBox vs Stata
*
* Tests: Pedroni, Kao
* Datasets: OECD Macro (consumption-income), PPP Data
* =============================================================================

clear all
set more off

* =============================================================================
* PART A: OECD Macro - Consumption-Income Cointegration
* =============================================================================

display "=== PART A: OECD Macro Cointegration ==="

local data_path "/home/guhaase/projetos/panelbox/examples/diagnostics/data/cointegration/oecd_macro.csv"
import delimited using "`data_path'", clear

describe
summarize log_c log_y

* Encode country for panel structure
encode country, gen(country_id)
xtset country_id year
xtdescribe

* --- 1. Verify I(1) behavior ---
display ""
display "--- Preliminary: Unit root tests on levels ---"

xtunitroot ips log_c, trend lags(aic 4)
xtunitroot ips log_y, trend lags(aic 4)

display ""
display "--- Unit root tests on first differences (should reject) ---"
xtunitroot ips D.log_c, lags(aic 4)
xtunitroot ips D.log_y, lags(aic 4)

* --- 2. Pedroni Cointegration Test ---
display ""
display "=== Pedroni Cointegration Test: log_C ~ log_Y ==="
display "H0: No cointegration"
display "H1: Cointegration exists"

* Pedroni test
xtcointtest pedroni log_c log_y, trend

* --- 3. Kao Cointegration Test ---
display ""
display "=== Kao Cointegration Test: log_C ~ log_Y ==="
display "H0: No cointegration"
display "H1: Cointegration exists (homogeneous beta)"

xtcointtest kao log_c log_y

* --- 4. Cointegrating regression (FMOLS/DOLS style) ---
display ""
display "=== Cointegrating Regression (FE) ==="
xtreg log_c log_y, fe
estimates store fe_oecd

* =============================================================================
* PART B: PPP Data - Exchange Rate Cointegration
* =============================================================================

display ""
display "=== PART B: PPP Cointegration ==="

local ppp_path "/home/guhaase/projetos/panelbox/examples/diagnostics/data/cointegration/ppp_data.csv"
import delimited using "`ppp_path'", clear

encode country, gen(country_id)
xtset country_id year

* --- Unit root tests ---
display "--- Unit root tests on PPP variables ---"
xtunitroot ips log_s, trend lags(aic 4)
xtunitroot ips log_p_ratio, trend lags(aic 4)

* --- Pedroni test ---
display ""
display "=== Pedroni Test: log_S ~ log_P_ratio ==="
xtcointtest pedroni log_s log_p_ratio, trend

* --- Kao test ---
display ""
display "=== Kao Test: log_S ~ log_P_ratio ==="
xtcointtest kao log_s log_p_ratio

* --- Cointegrating regression ---
display ""
display "=== Cointegrating Regression (FE) ==="
xtreg log_s log_p_ratio, fe

display ""
display "=== Cointegration Tests Complete ==="
