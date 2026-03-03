* =============================================================================
* Validation Script 02: FE/RE Poisson and PPML Gravity Models
* =============================================================================
* Compares Stata results with PanelBox for panel count models.
*
* Models estimated:
*   1. Pooled Poisson (city_crime): crime_count ~ unemployment_rate +
*      police_per_capita + median_income + temperature
*   2. FE Poisson (xtpoisson, fe)
*   3. RE Poisson (xtpoisson, re)
*   4. Hausman test (FE vs RE)
*   5. PPML Gravity (bilateral_trade): pooled, pair FE, year FE, pair+year FE
*
* Note: This script is for syntax reference only (no Stata license available).
* Requires: ppmlhdfe (install via: ssc install ppmlhdfe)
* =============================================================================

clear all
set more off

* --- Define paths ---
local data_dir "/home/guhaase/projetos/panelbox/examples/count/data"
local output_dir "/home/guhaase/projetos/panelbox/examples/count/Stata"

* =============================================================================
* PART 1: Panel Poisson Models on city_crime
* =============================================================================

display _newline(2) "============================================="
display "PART 1: Panel Poisson Models (city_crime)"
display "============================================="

* Load city_crime dataset
import delimited using "`data_dir'/city_crime.csv", clear

* Inspect data
describe
summarize crime_count unemployment_rate police_per_capita median_income temperature

* Declare panel structure
xtset city_id year

* Check panel balance
xtdescribe

* --- 1a. Pooled Poisson with cluster-robust SE ---
display _newline "--- 1a. Pooled Poisson (cluster-robust SE) ---"
poisson crime_count unemployment_rate police_per_capita median_income temperature, ///
    vce(cluster city_id)
estimates store pooled

* --- 1b. Fixed Effects Poisson (Conditional MLE) ---
* Uses Hausman, Hall, Griliches (1984) conditional MLE
display _newline "--- 1b. FE Poisson (xtpoisson, fe) ---"
xtpoisson crime_count unemployment_rate police_per_capita median_income temperature, ///
    fe vce(robust)
estimates store fe_poisson

* Display IRR
display _newline "--- FE Poisson (IRR) ---"
xtpoisson crime_count unemployment_rate police_per_capita median_income temperature, ///
    fe vce(robust) irr

* --- 1c. Random Effects Poisson ---
display _newline "--- 1c. RE Poisson (xtpoisson, re) ---"
xtpoisson crime_count unemployment_rate police_per_capita median_income temperature, ///
    re vce(robust)
estimates store re_poisson

* --- 1d. Hausman Test: FE vs RE ---
display _newline "--- 1d. Hausman Test ---"
* Note: Need to re-estimate without vce(robust) for Hausman test
quietly xtpoisson crime_count unemployment_rate police_per_capita ///
    median_income temperature, fe
estimates store fe_haus

quietly xtpoisson crime_count unemployment_rate police_per_capita ///
    median_income temperature, re
estimates store re_haus

hausman fe_haus re_haus
* H0: RE is consistent (difference not systematic)
* If p < 0.05, reject H0 -> use FE

* --- 1e. FE Poisson with Year Fixed Effects ---
display _newline "--- 1e. FE Poisson + Year FE ---"

* Create year dummies
tabulate year, generate(yr_)

* FE Poisson with year dummies
xtpoisson crime_count unemployment_rate police_per_capita median_income ///
    temperature yr_2-yr_10, fe vce(robust)
estimates store fe_year

* =============================================================================
* PART 2: PPML Gravity Model on bilateral_trade
* =============================================================================

display _newline(2) "============================================="
display "PART 2: PPML Gravity Model (bilateral_trade)"
display "============================================="

* Load bilateral_trade dataset
import delimited using "`data_dir'/bilateral_trade.csv", clear

* Inspect data
describe
summarize trade_value distance gdp_exporter gdp_importer

* Check zeros
count if trade_value == 0
display "Fraction of zero trade: " r(N) / _N

* Create log variables
generate log_gdp_exporter = ln(gdp_exporter)
generate log_gdp_importer = ln(gdp_importer)
generate log_distance = ln(distance)

* Create pair identifier (numeric)
encode exporter, generate(exp_id)
encode importer, generate(imp_id)
egen pair_id = group(exp_id imp_id)

* --- 2a. Pooled PPML (no fixed effects) ---
* PPML = Poisson regression on levels (Santos Silva & Tenreyro 2006)
display _newline "--- 2a. Pooled PPML ---"
poisson trade_value log_gdp_exporter log_gdp_importer log_distance ///
    contiguous common_language trade_agreement, vce(cluster pair_id)
estimates store ppml_pooled

* Display IRR
poisson trade_value log_gdp_exporter log_gdp_importer log_distance ///
    contiguous common_language trade_agreement, vce(cluster pair_id) irr

* --- 2b. PPML with Pair Fixed Effects ---
* Using ppmlhdfe for high-dimensional FE (recommended)
* Install: ssc install ppmlhdfe
* Alternative: use reghdfe + ppml approach
display _newline "--- 2b. PPML with Pair FE (ppmlhdfe) ---"
ppmlhdfe trade_value trade_agreement, absorb(pair_id) cluster(pair_id)
estimates store ppml_pair_fe

display "FTA coefficient: " _b[trade_agreement]
display "IRR: " exp(_b[trade_agreement])
display "FTA increases trade by " (exp(_b[trade_agreement]) - 1) * 100 "%"

* --- 2c. PPML with Year FE ---
display _newline "--- 2c. PPML with Year FE ---"
ppmlhdfe trade_value log_gdp_exporter log_gdp_importer log_distance ///
    contiguous common_language trade_agreement, absorb(year) cluster(pair_id)
estimates store ppml_year_fe

* --- 2d. PPML with Pair + Year FE ---
display _newline "--- 2d. PPML with Pair + Year FE ---"
ppmlhdfe trade_value trade_agreement, absorb(pair_id year) cluster(pair_id)
estimates store ppml_full

display "FTA coefficient (pair+year FE): " _b[trade_agreement]
display "FTA increases trade by " (exp(_b[trade_agreement]) - 1) * 100 "% (pair+year FE)"

* --- 2e. Model comparison ---
display _newline "--- Model Comparison ---"
estimates table ppml_pooled ppml_pair_fe ppml_year_fe ppml_full, ///
    stats(ll N) keep(trade_agreement log_gdp_exporter log_gdp_importer ///
    log_distance contiguous common_language)

* =============================================================================
* SAVE RESULTS
* =============================================================================

display _newline(2) "============================================="
display "Saving results..."
display "============================================="

* Example with esttab (requires estout package):
* esttab pooled fe_poisson re_poisson ppml_pooled ppml_pair_fe ppml_full ///
*     using "`output_dir'/results_02_fe_re_ppml_stata.csv", ///
*     cells(b se t p) stats(ll N) csv replace

display "Validation 02 complete."
display "============================================="
