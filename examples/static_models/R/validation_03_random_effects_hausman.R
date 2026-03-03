# =============================================================================
# Validation Script 03: Random Effects & Hausman Test
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

# --- Model 1: Random Effects (Swamy-Arora, default) -------------------------
cat("=== Model 1: Random Effects (Swamy-Arora) ===\n")
re_sa <- plm(invest ~ value + capital, data = pdata, model = "random",
             random.method = "swar")
s_re_sa <- summary(re_sa)
print(s_re_sa)

# Variance components
cat("\nVariance Components (Swamy-Arora):\n")
cat("sigma2_u (entity):", s_re_sa$ercomp$sigma2["id"], "\n")
cat("sigma2_e (idiosyncratic):", s_re_sa$ercomp$sigma2["idios"], "\n")
cat("theta:", s_re_sa$ercomp$theta, "\n\n")

# --- Model 2: Random Effects (Wallace-Hussain) -------------------------------
cat("=== Model 2: Random Effects (Wallace-Hussain) ===\n")
re_wh <- plm(invest ~ value + capital, data = pdata, model = "random",
             random.method = "walhus")
s_re_wh <- summary(re_wh)
print(s_re_wh)
cat("\n")

# --- Model 3: Random Effects (Amemiya) ---------------------------------------
cat("=== Model 3: Random Effects (Amemiya) ===\n")
re_am <- plm(invest ~ value + capital, data = pdata, model = "random",
             random.method = "amemiya")
s_re_am <- summary(re_am)
print(s_re_am)
cat("\n")

# --- Model 4: Random Effects (Nerlove) ----------------------------------------
cat("=== Model 4: Random Effects (Nerlove) ===\n")
re_ne <- plm(invest ~ value + capital, data = pdata, model = "random",
             random.method = "nerlove")
s_re_ne <- summary(re_ne)
print(s_re_ne)
cat("\n")

# --- Fixed Effects (for Hausman test) ----------------------------------------
fe <- plm(invest ~ value + capital, data = pdata, model = "within")

# --- Hausman Test (FE vs RE) ------------------------------------------------
cat("=== Hausman Test (FE vs RE) ===\n")
hausman <- phtest(fe, re_sa)
print(hausman)
cat("\nHausman statistic:", hausman$statistic, "\n")
cat("p-value:", hausman$p.value, "\n")
cat("df:", hausman$parameter, "\n")
cat("Decision:", ifelse(hausman$p.value < 0.05, "Use FE", "Use RE"), "\n\n")

# --- RE with clustered SE ---------------------------------------------------
cat("=== RE with Clustered SE ===\n")
coef_re_cluster <- coeftest(re_sa, vcov = vcovHC(re_sa, type = "HC1", cluster = "group"))
print(coef_re_cluster)
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

coef_re_sa <- coeftest(re_sa)
coef_re_wh <- coeftest(re_wh)
coef_re_am <- coeftest(re_am)
coef_re_ne <- coeftest(re_ne)

results <- rbind(
  extract_results(coef_re_sa, "re_swamy_arora", s_re_sa$r.squared[1], n_obs, n_groups),
  extract_results(coef_re_wh, "re_wallace_hussain", s_re_wh$r.squared[1], n_obs, n_groups),
  extract_results(coef_re_am, "re_amemiya", s_re_am$r.squared[1], n_obs, n_groups),
  extract_results(coef_re_ne, "re_nerlove", s_re_ne$r.squared[1], n_obs, n_groups),
  extract_results(coef_re_cluster, "re_clustered", s_re_sa$r.squared[1], n_obs, n_groups)
)

# Add variance components
var_rows <- data.frame(
  model_name = rep("variance_components_swar", 3),
  variable = c("sigma2_u", "sigma2_e", "theta"),
  coefficient = c(s_re_sa$ercomp$sigma2["id"],
                   s_re_sa$ercomp$sigma2["idios"],
                   s_re_sa$ercomp$theta),
  std_error = NA,
  t_statistic = NA,
  p_value = NA,
  r_squared = NA,
  n_obs = n_obs,
  n_groups = n_groups,
  stringsAsFactors = FALSE
)
results <- rbind(results, var_rows)

# Add Hausman test results
hausman_row <- data.frame(
  model_name = "hausman_test",
  variable = "chi2_statistic",
  coefficient = as.numeric(hausman$statistic),
  std_error = NA,
  t_statistic = NA,
  p_value = as.numeric(hausman$p.value),
  r_squared = NA,
  n_obs = n_obs,
  n_groups = n_groups,
  stringsAsFactors = FALSE
)
hausman_df_row <- data.frame(
  model_name = "hausman_test",
  variable = "df",
  coefficient = as.numeric(hausman$parameter),
  std_error = NA,
  t_statistic = NA,
  p_value = NA,
  r_squared = NA,
  n_obs = n_obs,
  n_groups = n_groups,
  stringsAsFactors = FALSE
)
results <- rbind(results, hausman_row, hausman_df_row)

# Write CSV
output_path <- "/home/guhaase/projetos/panelbox/examples/static_models/R/results_03_random_effects_hausman.csv"
write.csv(results, output_path, row.names = FALSE)
cat("Results saved to:", output_path, "\n")

# Print summary
cat("\n=== Summary Table ===\n")
print(results, row.names = FALSE)
cat("\nDone.\n")
