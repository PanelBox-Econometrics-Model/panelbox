# =============================================================================
# Validation Script 01: Pooled OLS
# PanelBox vs R (plm) comparison
# Dataset: Grunfeld (N=10, T=20, 200 obs)
# Model: invest ~ value + capital
# =============================================================================

library(plm)
library(lmtest)
library(sandwich)

# --- Load data ---------------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv"
grunfeld <- read.csv(data_path)

cat("=== Dataset Info ===\n")
cat("Observations:", nrow(grunfeld), "\n")
cat("Firms:", length(unique(grunfeld$firm)), "\n")
cat("Years:", length(unique(grunfeld$year)), "\n\n")

# Convert to panel data frame
pdata <- pdata.frame(grunfeld, index = c("firm", "year"))

# --- Model 1: Pooled OLS (non-robust SE) ------------------------------------
cat("=== Model 1: Pooled OLS (non-robust SE) ===\n")
pooled <- plm(invest ~ value + capital, data = pdata, model = "pooling")
s_pooled <- summary(pooled)
print(s_pooled)

# Extract coefficients table
coef_pooled <- coeftest(pooled)
cat("\nR-squared:", s_pooled$r.squared[1], "\n")
cat("Adj R-squared:", s_pooled$r.squared[2], "\n\n")

# --- Model 2: Pooled OLS with HC1 robust SE ---------------------------------
cat("=== Model 2: Pooled OLS (HC1 robust SE) ===\n")
coef_hc1 <- coeftest(pooled, vcov = vcovHC(pooled, type = "HC1"))
print(coef_hc1)
cat("\n")

# --- Model 3: Pooled OLS with clustered SE (by firm) ------------------------
cat("=== Model 3: Pooled OLS (clustered SE by firm) ===\n")
coef_cluster <- coeftest(pooled, vcov = vcovHC(pooled, type = "HC1", cluster = "group"))
print(coef_cluster)
cat("\n")

# --- Model 4: Pooled OLS with two-way clustered SE --------------------------
cat("=== Model 4: Pooled OLS (two-way clustered SE) ===\n")
coef_twoway <- coeftest(pooled, vcov = vcovDC(pooled, type = "HC1"))
print(coef_twoway)
cat("\n")

# --- Save results to CSV -----------------------------------------------------
results <- data.frame(
  model_name = character(),
  variable = character(),
  coefficient = numeric(),
  std_error = numeric(),
  t_statistic = numeric(),
  p_value = numeric(),
  r_squared = numeric(),
  n_obs = integer(),
  n_groups = integer(),
  stringsAsFactors = FALSE
)

# Helper function to extract results
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

n_obs <- nrow(grunfeld)
n_groups <- length(unique(grunfeld$firm))
r2 <- s_pooled$r.squared[1]

results <- rbind(
  extract_results(coef_pooled, "pooled_ols_nonrobust", r2, n_obs, n_groups),
  extract_results(coef_hc1, "pooled_ols_hc1", r2, n_obs, n_groups),
  extract_results(coef_cluster, "pooled_ols_clustered", r2, n_obs, n_groups),
  extract_results(coef_twoway, "pooled_ols_twoway_cluster", r2, n_obs, n_groups)
)

# Write CSV
output_path <- "/home/guhaase/projetos/panelbox/examples/static_models/R/results_01_pooled_ols.csv"
write.csv(results, output_path, row.names = FALSE)
cat("Results saved to:", output_path, "\n")

# Print summary table
cat("\n=== Summary Table ===\n")
print(results, row.names = FALSE)
cat("\nDone.\n")
