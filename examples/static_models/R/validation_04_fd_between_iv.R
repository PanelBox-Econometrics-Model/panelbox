# =============================================================================
# Validation Script 04: First Difference, Between, and IV Estimators
# PanelBox vs R (plm) comparison
# Dataset: Grunfeld (N=10, T=20, 200 obs)
# Model: invest ~ value + capital
# =============================================================================

library(plm)
library(lmtest)
library(sandwich)
library(AER)

# --- Load data ---------------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv"
grunfeld <- read.csv(data_path)
pdata <- pdata.frame(grunfeld, index = c("firm", "year"))

n_obs <- nrow(grunfeld)
n_groups <- length(unique(grunfeld$firm))

# --- Model 1: First Difference -----------------------------------------------
cat("=== Model 1: First Difference ===\n")
fd <- plm(invest ~ value + capital, data = pdata, model = "fd")
s_fd <- summary(fd)
print(s_fd)
cat("\nR-squared:", s_fd$r.squared[1], "\n\n")

# FD with robust SE
cat("=== First Difference with Robust SE ===\n")
coef_fd_robust <- coeftest(fd, vcov = vcovHC(fd, type = "HC1"))
print(coef_fd_robust)
cat("\n")

# --- Model 2: Between Estimator ----------------------------------------------
cat("=== Model 2: Between Estimator ===\n")
be <- plm(invest ~ value + capital, data = pdata, model = "between")
s_be <- summary(be)
print(s_be)
cat("\nR-squared:", s_be$r.squared[1], "\n")
cat("Note: Between uses N =", n_groups, "group means (not NT =", n_obs, ")\n\n")

# --- Model 3: IV-Pooled (2SLS) using lagged value as instrument --------------
# Create lagged instrument within panel structure
cat("=== Model 3: Panel IV (2SLS) ===\n")
cat("Using lag(value) as instrument for value\n")

# IV-Pooled: invest ~ capital + value | capital + lag(value)
# In plm, use formula with | for instruments
iv_pooled <- plm(invest ~ capital + value | capital + lag(value),
                 data = pdata, model = "pooling")
s_iv_pooled <- summary(iv_pooled)
print(s_iv_pooled)
cat("\n")

# --- Model 4: IV-FE (2SLS + Fixed Effects) -----------------------------------
cat("=== Model 4: IV-FE (2SLS + Fixed Effects) ===\n")
iv_fe <- plm(invest ~ capital + value | capital + lag(value),
             data = pdata, model = "within")
s_iv_fe <- summary(iv_fe)
print(s_iv_fe)
cat("\n")

# --- Model 5: IV-RE (2SLS + Random Effects) -----------------------------------
cat("=== Model 5: IV-RE (2SLS + Random Effects) ===\n")
iv_re <- plm(invest ~ capital + value | capital + lag(value),
             data = pdata, model = "random")
s_iv_re <- summary(iv_re)
print(s_iv_re)
cat("\n")

# --- Comparison: All estimators -----------------------------------------------
cat("=== Comparison of All Estimators ===\n")
pooled <- plm(invest ~ value + capital, data = pdata, model = "pooling")
fe <- plm(invest ~ value + capital, data = pdata, model = "within")
re <- plm(invest ~ value + capital, data = pdata, model = "random")

cat("\nCoefficients Comparison:\n")
comp <- data.frame(
  Pooled = coef(pooled)[c("value", "capital")],
  FE = coef(fe)[c("value", "capital")],
  RE = coef(re)[c("value", "capital")],
  FD = coef(fd)[c("value", "capital")],
  Between = coef(be)[c("value", "capital")]
)
print(comp)
cat("\n")

# --- Save results to CSV -----------------------------------------------------
extract_results <- function(coef_table, model_name, r2, n_obs, n_groups) {
  vars <- rownames(coef_table)
  data.frame(
    model_name = model_name,
    variable = vars,
    coefficient = coef_table[, 1],
    std_error = coef_table[, 2],
    t_statistic = coef_table[, 3],
    p_value = coef_table[, 4],
    r_squared = r2,
    n_obs = n_obs,
    n_groups = n_groups,
    stringsAsFactors = FALSE
  )
}

coef_fd <- coeftest(fd)
coef_be <- coeftest(be)
coef_iv_pooled <- coeftest(iv_pooled)
coef_iv_fe <- coeftest(iv_fe)
coef_iv_re <- coeftest(iv_re)

results <- rbind(
  extract_results(coef_fd, "first_difference", s_fd$r.squared[1], n_obs, n_groups),
  extract_results(coef_fd_robust, "first_difference_robust", s_fd$r.squared[1], n_obs, n_groups),
  extract_results(coef_be, "between", s_be$r.squared[1], n_obs, n_groups),
  extract_results(coef_iv_pooled, "iv_pooled", NA, n_obs, n_groups),
  extract_results(coef_iv_fe, "iv_fe", NA, n_obs, n_groups),
  extract_results(coef_iv_re, "iv_re", NA, n_obs, n_groups)
)

# Write CSV
output_path <- "/home/guhaase/projetos/panelbox/examples/static_models/R/results_04_fd_between_iv.csv"
write.csv(results, output_path, row.names = FALSE)
cat("Results saved to:", output_path, "\n")

# --- Merge all results into one master CSV ------------------------------------
cat("\n=== Merging all results into master CSV ===\n")

r1 <- read.csv("/home/guhaase/projetos/panelbox/examples/static_models/R/results_01_pooled_ols.csv",
               stringsAsFactors = FALSE)
r2 <- read.csv("/home/guhaase/projetos/panelbox/examples/static_models/R/results_02_fixed_effects.csv",
               stringsAsFactors = FALSE)
r3 <- read.csv("/home/guhaase/projetos/panelbox/examples/static_models/R/results_03_random_effects_hausman.csv",
               stringsAsFactors = FALSE)
r4 <- read.csv("/home/guhaase/projetos/panelbox/examples/static_models/R/results_04_fd_between_iv.csv",
               stringsAsFactors = FALSE)

all_results <- rbind(r1, r2, r3, r4)
master_path <- "/home/guhaase/projetos/panelbox/examples/static_models/R/results_static_models.csv"
write.csv(all_results, master_path, row.names = FALSE)
cat("Master results saved to:", master_path, "\n")
cat("Total rows:", nrow(all_results), "\n")

# Print summary
cat("\n=== Summary Table ===\n")
print(results, row.names = FALSE)
cat("\nDone.\n")
