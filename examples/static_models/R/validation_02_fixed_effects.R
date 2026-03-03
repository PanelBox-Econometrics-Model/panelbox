# =============================================================================
# Validation Script 02: Fixed Effects (Within Estimator)
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
pdata <- pdata.frame(grunfeld, index = c("firm", "year"))

n_obs <- nrow(grunfeld)
n_groups <- length(unique(grunfeld$firm))

# --- Model 1: Fixed Effects (one-way, entity) --------------------------------
cat("=== Model 1: Fixed Effects (One-Way, Entity) ===\n")
fe_one <- plm(invest ~ value + capital, data = pdata, model = "within", effect = "individual")
s_fe_one <- summary(fe_one)
print(s_fe_one)

cat("\nWithin R-squared:", s_fe_one$r.squared[1], "\n")
cat("Adj Within R-squared:", s_fe_one$r.squared[2], "\n\n")

# Fixed effects (entity-specific intercepts)
cat("=== Entity Fixed Effects ===\n")
fe_values <- fixef(fe_one)
print(fe_values)
cat("\n")

# --- Model 2: Fixed Effects (two-way, entity + time) -------------------------
cat("=== Model 2: Fixed Effects (Two-Way, Entity + Time) ===\n")
fe_two <- plm(invest ~ value + capital, data = pdata, model = "within", effect = "twoways")
s_fe_two <- summary(fe_two)
print(s_fe_two)

cat("\nWithin R-squared:", s_fe_two$r.squared[1], "\n")
cat("Adj Within R-squared:", s_fe_two$r.squared[2], "\n\n")

# --- F-test for individual effects (FE vs Pooled OLS) ------------------------
cat("=== F-test: Individual Effects (FE vs Pooled OLS) ===\n")
pooled <- plm(invest ~ value + capital, data = pdata, model = "pooling")
f_test <- pFtest(fe_one, pooled)
print(f_test)
cat("\n")

# --- F-test for time effects --------------------------------------------------
cat("=== F-test: Time Effects ===\n")
f_test_time <- pFtest(fe_two, fe_one)
print(f_test_time)
cat("\n")

# --- FE with clustered SE ----------------------------------------------------
cat("=== FE with Clustered SE (by firm) ===\n")
coef_fe_cluster <- coeftest(fe_one, vcov = vcovHC(fe_one, type = "HC1", cluster = "group"))
print(coef_fe_cluster)
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

# FE one-way (non-robust)
coef_fe_one <- coeftest(fe_one)
# FE two-way (non-robust)
coef_fe_two <- coeftest(fe_two)

results <- rbind(
  extract_results(coef_fe_one, "fe_oneway", s_fe_one$r.squared[1], n_obs, n_groups),
  extract_results(coef_fe_two, "fe_twoway", s_fe_two$r.squared[1], n_obs, n_groups),
  extract_results(coef_fe_cluster, "fe_oneway_clustered", s_fe_one$r.squared[1], n_obs, n_groups)
)

# Add F-test results as separate rows
f_test_rows <- data.frame(
  model_name = c("f_test_individual", "f_test_time"),
  variable = c("F_statistic", "F_statistic"),
  coefficient = c(f_test$statistic, f_test_time$statistic),
  std_error = NA,
  t_statistic = NA,
  p_value = c(f_test$p.value, f_test_time$p.value),
  r_squared = NA,
  n_obs = n_obs,
  n_groups = n_groups,
  stringsAsFactors = FALSE
)
results <- rbind(results, f_test_rows)

# Add fixed effects values
fe_rows <- data.frame(
  model_name = "entity_fixed_effects",
  variable = names(fe_values),
  coefficient = as.numeric(fe_values),
  std_error = NA,
  t_statistic = NA,
  p_value = NA,
  r_squared = NA,
  n_obs = n_obs,
  n_groups = n_groups,
  stringsAsFactors = FALSE
)
results <- rbind(results, fe_rows)

# Write CSV
output_path <- "/home/guhaase/projetos/panelbox/examples/static_models/R/results_02_fixed_effects.csv"
write.csv(results, output_path, row.names = FALSE)
cat("Results saved to:", output_path, "\n")

# Print summary
cat("\n=== Summary Table ===\n")
print(results, row.names = FALSE)
cat("\nDone.\n")
