* =============================================================================
* Validation Script 03: Spatial Direct, Indirect, and Total Effects
* =============================================================================
* Dataset: Columbus - 49 neighborhoods in Columbus, Ohio
* Model: CRIME ~ INC + HOVAL
* Weight matrix: Queen contiguity (row-standardized)
* Post-estimation: Direct, indirect, and total effects decomposition
* =============================================================================

clear all
set more off

* ---- Load Columbus dataset ----
* Assume Columbus dataset is loaded and spatial weights W are defined.
* See validation_01_moran_sar.do for data loading and weight matrix setup.

* If loading from shapefile:
* spshape2dta columbus, replace
* use columbus, clear
* spset _ID
* spmatrix create contiguity W, replace normalize(row)

* ---- 1. SAR Model (Spatial Lag) via ML ----
display "=== SAR Model for Effects Decomposition ==="

* Estimate SAR model via ML
spregress CRIME INC HOVAL, ml dvarlag(W)
estimates store sar_ml

* ---- 2. SAR Direct and Indirect Effects ----
display "=== SAR Direct, Indirect, and Total Effects ==="

* In Stata, after spregress, use estat impact to decompose effects
* The spatial multiplier S(rho) = (I - rho*W)^{-1} creates:
*   Direct effect: (1/N) * tr(S * beta)
*   Total effect: (1/N) * sum(S * beta)
*   Indirect effect: Total - Direct

estat impact

* This displays:
*   - Direct effects for each variable
*   - Indirect effects (spillover to neighbors)
*   - Total effects (direct + indirect)
* With standard errors computed via Delta method

* Store the results
matrix sar_direct = r(b_direct)
matrix sar_indirect = r(b_indirect)
matrix sar_total = r(b_total)

display "SAR Direct Effects:"
matrix list sar_direct
display "SAR Indirect Effects:"
matrix list sar_indirect
display "SAR Total Effects:"
matrix list sar_total

* ---- 3. SDM Model (Spatial Durbin) via ML ----
display "=== SDM Model for Effects Decomposition ==="

* Estimate SDM model via ML
spregress CRIME INC HOVAL, ml dvarlag(W) ivarlag(W: INC HOVAL)
estimates store sdm_ml

* ---- 4. SDM Direct and Indirect Effects ----
display "=== SDM Direct, Indirect, and Total Effects ==="

* For SDM, effects decomposition accounts for both rho*W*y and W*X*theta
* Direct: (1/N) * tr(S * beta_k) where S = (I - rho*W)^{-1}
* Indirect: includes both feedback through W*y and exogenous spillovers W*X
estat impact

matrix sdm_direct = r(b_direct)
matrix sdm_indirect = r(b_indirect)
matrix sdm_total = r(b_total)

display "SDM Direct Effects:"
matrix list sdm_direct
display "SDM Indirect Effects:"
matrix list sdm_indirect
display "SDM Total Effects:"
matrix list sdm_total

* ---- 5. Comparison of Effects ----
display "=== Comparison: SAR vs SDM Effects ==="

* Display side-by-side comparison
display "Variable     | SAR Direct | SAR Indirect | SAR Total | SDM Direct | SDM Indirect | SDM Total"
display "-------------|------------|--------------|-----------|------------|--------------|----------"

* For INC
display "INC          | " %9.4f sar_direct[1,1] " | " %9.4f sar_indirect[1,1] " | " %9.4f sar_total[1,1] ///
    " | " %9.4f sdm_direct[1,1] " | " %9.4f sdm_indirect[1,1] " | " %9.4f sdm_total[1,1]

* For HOVAL
display "HOVAL        | " %9.4f sar_direct[1,2] " | " %9.4f sar_indirect[1,2] " | " %9.4f sar_total[1,2] ///
    " | " %9.4f sdm_direct[1,2] " | " %9.4f sdm_indirect[1,2] " | " %9.4f sdm_total[1,2]

* ---- 6. Spillover Share ----
display ""
display "=== Spillover Share (Indirect/Total) ==="

* The spillover share indicates what fraction of the total effect
* comes from spatial spillovers (indirect effects)
display "SAR:"
display "  INC spillover share: " %6.3f sar_indirect[1,1]/sar_total[1,1]
display "  HOVAL spillover share: " %6.3f sar_indirect[1,2]/sar_total[1,2]

display "SDM:"
display "  INC spillover share: " %6.3f sdm_indirect[1,1]/sdm_total[1,1]
display "  HOVAL spillover share: " %6.3f sdm_indirect[1,2]/sdm_total[1,2]

* ---- 7. Export results ----
display "=== Export Results ==="

* Create a dataset with effects for export
clear
input str10 model str10 variable double(direct_effect indirect_effect total_effect)
"sar_ml" "INC" . . .
"sar_ml" "HOVAL" . . .
"sdm_ml" "INC" . . .
"sdm_ml" "HOVAL" . . .
end

* Fill in values from matrices
* replace direct_effect = sar_direct[1,1] if model == "sar_ml" & variable == "INC"
* (etc. for all combinations)

* Export to CSV
* export delimited using "results_03_spatial_effects.csv", replace

display "Done."
