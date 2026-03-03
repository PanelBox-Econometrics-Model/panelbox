###############################################################################
# Validation 01 - Robust Standard Errors (HC0, HC1, HC2, HC3)
#
# Estimates Pooled OLS on Grunfeld data and computes White heteroskedasticity-
# consistent standard errors using sandwich::vcovHC() + lmtest::coeftest().
# Coefficients are identical across all SE types; only SEs differ.
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
cat("Years:", paste(sort(unique(grunfeld$year)), collapse = ", "), "\n")
cat("Columns:", paste(names(grunfeld), collapse = ", "), "\n\n")

# --- OLS Estimation ---------------------------------------------------------
ols_model <- lm(invest ~ value + capital, data = grunfeld)

cat("=== OLS Coefficients ===\n")
print(summary(ols_model)$coefficients)
cat("\n")

# --- Classical (Non-robust) SE ----------------------------------------------
ct_classical <- coeftest(ols_model)
cat("=== Classical (Non-robust) SE ===\n")
print(ct_classical)
cat("\n")

# --- HC0 (White 1980) ------------------------------------------------------
vcov_hc0 <- vcovHC(ols_model, type = "HC0")
ct_hc0 <- coeftest(ols_model, vcov. = vcov_hc0)
cat("=== HC0 Standard Errors ===\n")
print(ct_hc0)
cat("\n")

# --- HC1 (Stata default 'robust') ------------------------------------------
vcov_hc1 <- vcovHC(ols_model, type = "HC1")
ct_hc1 <- coeftest(ols_model, vcov. = vcov_hc1)
cat("=== HC1 Standard Errors ===\n")
print(ct_hc1)
cat("\n")

# --- HC2 (Leverage-adjusted) -----------------------------------------------
vcov_hc2 <- vcovHC(ols_model, type = "HC2")
ct_hc2 <- coeftest(ols_model, vcov. = vcov_hc2)
cat("=== HC2 Standard Errors ===\n")
print(ct_hc2)
cat("\n")

# --- HC3 (Davidson-MacKinnon, most conservative) ---------------------------
vcov_hc3 <- vcovHC(ols_model, type = "HC3")
ct_hc3 <- coeftest(ols_model, vcov. = vcov_hc3)
cat("=== HC3 Standard Errors ===\n")
print(ct_hc3)
cat("\n")

# --- Build results data.frame -----------------------------------------------
build_rows <- function(ct, se_type) {
  vars <- rownames(ct)
  data.frame(
    model_name = "pooled_ols",
    se_type    = se_type,
    variable   = vars,
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
  build_rows(ct_hc0, "HC0"),
  build_rows(ct_hc1, "HC1"),
  build_rows(ct_hc2, "HC2"),
  build_rows(ct_hc3, "HC3")
)

# --- Sanity check: coefficients must be identical ---------------------------
cat("=== Sanity Check: Coefficient Consistency ===\n")
coef_by_type <- tapply(results$coefficient, results$se_type, function(x) x)
ref <- coef_by_type[[1]]
all_equal <- all(sapply(coef_by_type, function(x) all.equal(x, ref)))
cat("All coefficients identical across SE types:", all_equal, "\n\n")

# --- Save to CSV -------------------------------------------------------------
out_dir <- "/home/guhaase/projetos/panelbox/examples/standard_errors/R"
out_file <- file.path(out_dir, "results_robust_hc.csv")
write.csv(results, out_file, row.names = FALSE)
cat("Results saved to:", out_file, "\n")

# --- Summary table -----------------------------------------------------------
cat("\n=== SE Comparison Summary ===\n")
se_wide <- reshape(results[, c("se_type", "variable", "std_error")],
                   idvar = "variable", timevar = "se_type",
                   direction = "wide")
print(se_wide)
cat("\nDone.\n")
