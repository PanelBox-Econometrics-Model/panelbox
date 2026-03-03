###############################################################################
# Validation 03 - HAC (Newey-West) Standard Errors
#
# Estimates Pooled OLS on Grunfeld data and computes Newey-West HAC standard
# errors with different lag specifications (1, 2, 3, 4, auto).
# Uses sandwich::NeweyWest() + lmtest::coeftest().
#
# Dataset: Grunfeld (10 firms, 20 years)
# Model:   invest ~ value + capital
###############################################################################

library(plm)
library(sandwich)
library(lmtest)

# --- Data Loading -----------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv"
grunfeld <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("Rows:", nrow(grunfeld), "\n")
cat("Firms:", length(unique(grunfeld$firm)), "\n")
cat("Years:", paste(sort(unique(grunfeld$year)), collapse = ", "), "\n\n")

# --- OLS Estimation ---------------------------------------------------------
ols_model <- lm(invest ~ value + capital, data = grunfeld)

cat("=== OLS Coefficients ===\n")
print(summary(ols_model)$coefficients)
cat("\n")

# --- Non-robust SE (baseline) -----------------------------------------------
ct_classical <- coeftest(ols_model)
cat("=== Classical (Non-robust) SE ===\n")
print(ct_classical)
cat("\n")

# --- Newey-West HAC with lag=1 (Bartlett kernel) ----------------------------
vcov_hac1 <- NeweyWest(ols_model, lag = 1, prewhite = FALSE,
                        adjust = TRUE)
ct_hac1 <- coeftest(ols_model, vcov. = vcov_hac1)
cat("=== HAC Newey-West (lag=1) ===\n")
print(ct_hac1)
cat("\n")

# --- Newey-West HAC with lag=2 ----------------------------------------------
vcov_hac2 <- NeweyWest(ols_model, lag = 2, prewhite = FALSE,
                        adjust = TRUE)
ct_hac2 <- coeftest(ols_model, vcov. = vcov_hac2)
cat("=== HAC Newey-West (lag=2) ===\n")
print(ct_hac2)
cat("\n")

# --- Newey-West HAC with lag=3 ----------------------------------------------
vcov_hac3 <- NeweyWest(ols_model, lag = 3, prewhite = FALSE,
                        adjust = TRUE)
ct_hac3 <- coeftest(ols_model, vcov. = vcov_hac3)
cat("=== HAC Newey-West (lag=3) ===\n")
print(ct_hac3)
cat("\n")

# --- Newey-West HAC with lag=4 ----------------------------------------------
vcov_hac4 <- NeweyWest(ols_model, lag = 4, prewhite = FALSE,
                        adjust = TRUE)
ct_hac4 <- coeftest(ols_model, vcov. = vcov_hac4)
cat("=== HAC Newey-West (lag=4) ===\n")
print(ct_hac4)
cat("\n")

# --- Newey-West HAC with automatic lag selection ----------------------------
# Automatic bandwidth selection (Andrews 1991)
vcov_hac_auto <- NeweyWest(ols_model, prewhite = FALSE)
ct_hac_auto <- coeftest(ols_model, vcov. = vcov_hac_auto)
cat("=== HAC Newey-West (automatic lag) ===\n")
print(ct_hac_auto)
cat("\n")

# --- Also compute panel HAC using plm (Newey-West for panel) ----------------
pgrunfeld <- pdata.frame(grunfeld, index = c("firm", "year"))
pooled_plm <- plm(invest ~ value + capital, data = pgrunfeld, model = "pooling")

# Panel Newey-West (entity-level NW within each cluster)
vcov_plm_nw <- vcovNW(pooled_plm, maxlag = 2)
ct_plm_nw <- coeftest(pooled_plm, vcov. = vcov_plm_nw)
cat("=== Panel HAC (plm::vcovNW, maxlag=2) ===\n")
print(ct_plm_nw)
cat("\n")

# Driscoll-Kraay (cross-sectional dependence-robust HAC)
vcov_plm_scc <- vcovSCC(pooled_plm, maxlag = 2)
ct_plm_scc <- coeftest(pooled_plm, vcov. = vcov_plm_scc)
cat("=== Driscoll-Kraay (plm::vcovSCC, maxlag=2) ===\n")
print(ct_plm_scc)
cat("\n")

# --- Build results data.frame -----------------------------------------------
build_rows <- function(ct, se_type, model_name = "pooled_ols") {
  vars <- rownames(ct)
  data.frame(
    model_name  = model_name,
    se_type     = se_type,
    variable    = vars,
    coefficient = ct[, "Estimate"],
    std_error   = ct[, "Std. Error"],
    t_statistic = ct[, "t value"],
    p_value     = ct[, "Pr(>|t|)"],
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

results <- rbind(
  build_rows(ct_classical, "classical"),
  build_rows(ct_hac1, "HAC_lag1"),
  build_rows(ct_hac2, "HAC_lag2"),
  build_rows(ct_hac3, "HAC_lag3"),
  build_rows(ct_hac4, "HAC_lag4"),
  build_rows(ct_hac_auto, "HAC_auto"),
  build_rows(ct_plm_nw, "panel_NW_lag2", "pooled_ols_plm"),
  build_rows(ct_plm_scc, "driscoll_kraay_lag2", "pooled_ols_plm")
)

# --- Sanity check: coefficients must be identical (within lm model) ---------
cat("=== Sanity Check: Coefficient Consistency ===\n")
lm_results <- results[results$model_name == "pooled_ols", ]
coef_by_type <- tapply(lm_results$coefficient, lm_results$se_type, function(x) x)
ref <- coef_by_type[[1]]
all_equal <- all(sapply(coef_by_type, function(x) isTRUE(all.equal(x, ref))))
cat("All OLS coefficients identical across HAC types:", all_equal, "\n")

plm_results <- results[results$model_name == "pooled_ols_plm", ]
coef_by_type2 <- tapply(plm_results$coefficient, plm_results$se_type, function(x) x)
ref2 <- coef_by_type2[[1]]
all_equal2 <- all(sapply(coef_by_type2, function(x) isTRUE(all.equal(x, ref2))))
cat("All PLM coefficients identical across HAC types:", all_equal2, "\n\n")

# --- Save to CSV -------------------------------------------------------------
out_dir <- "/home/guhaase/projetos/panelbox/examples/standard_errors/R"
out_file <- file.path(out_dir, "results_hac.csv")
write.csv(results, out_file, row.names = FALSE)
cat("Results saved to:", out_file, "\n")

# --- Summary table -----------------------------------------------------------
cat("\n=== SE Comparison Summary ===\n")
se_wide <- reshape(results[, c("se_type", "variable", "std_error")],
                   idvar = "variable", timevar = "se_type",
                   direction = "wide")
print(se_wide)
cat("\nDone.\n")
