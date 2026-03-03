# =============================================================================
# Validation 01: Pooled Quantile Regression
# Replicates PanelBox Notebook 01 results using R quantreg package
#
# Model: lwage ~ educ + exper + I(exper^2)
# Dataset: card_education.csv
# Quantiles: tau = 0.1, 0.25, 0.5, 0.75, 0.9
# =============================================================================

library(quantreg)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/quantile/data/card_education.csv"
df <- read.csv(data_path)

cat("Dataset loaded:\n")
cat(sprintf("  Observations: %d\n", nrow(df)))
cat(sprintf("  Individuals: %d\n", length(unique(df$id))))
cat(sprintf("  Time periods: %d\n", length(unique(df$year))))
cat("\nFirst rows:\n")
print(head(df))

# Create exper_sq variable
df$exper_sq <- df$exper^2

# ---------------------------------------------------------------------------
# 2. OLS baseline
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("OLS BASELINE\n")
cat("==============================================================================\n")

ols_model <- lm(lwage ~ educ + exper + exper_sq, data = df)
cat("\nOLS Results:\n")
print(summary(ols_model))

ols_coefs <- coef(ols_model)
ols_se <- summary(ols_model)$coefficients[, "Std. Error"]
ols_tvals <- summary(ols_model)$coefficients[, "t value"]
ols_pvals <- summary(ols_model)$coefficients[, "Pr(>|t|)"]

# ---------------------------------------------------------------------------
# 3. Quantile regression at selected quantiles
# ---------------------------------------------------------------------------
tau_list <- c(0.1, 0.25, 0.5, 0.75, 0.9)
n_obs <- nrow(df)

results_list <- list()

cat("\n==============================================================================\n")
cat("QUANTILE REGRESSION RESULTS\n")
cat("==============================================================================\n")

for (tau in tau_list) {
  cat(sprintf("\n--- Quantile tau = %.2f ---\n", tau))

  qr_model <- rq(lwage ~ educ + exper + exper_sq, data = df, tau = tau)
  qr_summ <- summary(qr_model, se = "iid")

  cat("Coefficients:\n")
  print(qr_summ$coefficients)

  coefs <- coef(qr_model)
  se_vals <- qr_summ$coefficients[, "Std. Error"]
  t_vals <- qr_summ$coefficients[, "t value"]
  p_vals <- qr_summ$coefficients[, "Pr(>|t|)"]
  var_names <- names(coefs)

  for (j in seq_along(var_names)) {
    results_list[[length(results_list) + 1]] <- data.frame(
      model_name = sprintf("pooled_qr_tau%.2f", tau),
      quantile = tau,
      variable = var_names[j],
      coefficient = coefs[j],
      std_error = se_vals[j],
      t_statistic = t_vals[j],
      p_value = p_vals[j],
      n_obs = n_obs,
      stringsAsFactors = FALSE
    )
  }
}

# Add OLS results
ols_var_names <- names(ols_coefs)
for (j in seq_along(ols_var_names)) {
  results_list[[length(results_list) + 1]] <- data.frame(
    model_name = "ols",
    quantile = NA,
    variable = ols_var_names[j],
    coefficient = ols_coefs[j],
    std_error = ols_se[j],
    t_statistic = ols_tvals[j],
    p_value = ols_pvals[j],
    n_obs = n_obs,
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# 4. Combine and save results
# ---------------------------------------------------------------------------
results_df <- do.call(rbind, results_list)
rownames(results_df) <- NULL

cat("\n==============================================================================\n")
cat("COMBINED RESULTS\n")
cat("==============================================================================\n")
print(results_df)

output_path <- "/home/guhaase/projetos/panelbox/examples/quantile/R/results_01_quantile_pooled.csv"
write.csv(results_df, output_path, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", output_path))

# ---------------------------------------------------------------------------
# 5. Summary comparison across quantiles for education coefficient
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("EDUCATION COEFFICIENT ACROSS QUANTILES\n")
cat("==============================================================================\n")

educ_rows <- results_df[results_df$variable == "educ", ]
cat(sprintf("%-12s  %12s  %12s\n", "Model", "Coefficient", "Std. Error"))
cat(paste(rep("-", 40), collapse = ""), "\n")
for (i in seq_len(nrow(educ_rows))) {
  cat(sprintf("%-12s  %12.6f  %12.6f\n",
              educ_rows$model_name[i],
              educ_rows$coefficient[i],
              educ_rows$std_error[i]))
}

cat("\nDone.\n")
