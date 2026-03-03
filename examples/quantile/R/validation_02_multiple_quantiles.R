# =============================================================================
# Validation 02: Multiple Quantiles and Quantile Process
# Replicates PanelBox Notebook 02 results using R quantreg package
#
# Model: lwage ~ female + educ + exper + I(exper^2)
# Dataset: card_education.csv
# Quantile process: tau = seq(0.05, 0.95, 0.05)
# Inter-quantile tests for the female (gender gap) coefficient
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

df$exper_sq <- df$exper^2
n_obs <- nrow(df)

# ---------------------------------------------------------------------------
# 2. OLS baseline (with female)
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("OLS BASELINE\n")
cat("==============================================================================\n")

ols_model <- lm(lwage ~ female + educ + exper + exper_sq, data = df)
cat("\nOLS Results:\n")
print(summary(ols_model))

# ---------------------------------------------------------------------------
# 3. Full quantile process: tau = seq(0.05, 0.95, 0.05)
# ---------------------------------------------------------------------------
tau_grid <- seq(0.05, 0.95, by = 0.05)

cat("\n==============================================================================\n")
cat(sprintf("QUANTILE PROCESS: %d quantiles\n", length(tau_grid)))
cat("==============================================================================\n")

qr_process <- rq(lwage ~ female + educ + exper + exper_sq,
                  data = df, tau = tau_grid)

# Extract coefficients matrix (variables x quantiles)
coef_matrix <- coef(qr_process)
cat("\nCoefficient matrix (rows=variables, cols=quantiles):\n")
print(round(coef_matrix, 6))

# Get summaries with standard errors for each quantile
results_list <- list()

for (k in seq_along(tau_grid)) {
  tau <- tau_grid[k]

  # Fit individual model to get SEs
  qr_k <- rq(lwage ~ female + educ + exper + exper_sq,
              data = df, tau = tau)
  qr_summ <- summary(qr_k, se = "iid")

  coefs <- coef(qr_k)
  se_vals <- qr_summ$coefficients[, "Std. Error"]
  t_vals <- qr_summ$coefficients[, "t value"]
  p_vals <- qr_summ$coefficients[, "Pr(>|t|)"]
  var_names <- names(coefs)

  for (j in seq_along(var_names)) {
    results_list[[length(results_list) + 1]] <- data.frame(
      model_name = sprintf("qr_process_tau%.2f", tau),
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

# Add OLS
ols_coefs <- coef(ols_model)
ols_se <- summary(ols_model)$coefficients[, "Std. Error"]
ols_tvals <- summary(ols_model)$coefficients[, "t value"]
ols_pvals <- summary(ols_model)$coefficients[, "Pr(>|t|)"]
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

results_df <- do.call(rbind, results_list)
rownames(results_df) <- NULL

# ---------------------------------------------------------------------------
# 4. Inter-quantile tests (equality of female coefficient across quantiles)
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("INTER-QUANTILE TESTS FOR GENDER WAGE GAP (female)\n")
cat("==============================================================================\n")

# Use anova.rq for formal testing
# Test H0: beta_female(tau1) = beta_female(tau2)

test_pairs <- list(
  c(0.10, 0.50),
  c(0.50, 0.90),
  c(0.10, 0.90),
  c(0.25, 0.75)
)

iq_test_list <- list()

for (pair in test_pairs) {
  tau1 <- pair[1]
  tau2 <- pair[2]

  qr1 <- rq(lwage ~ female + educ + exper + exper_sq, data = df, tau = tau1)
  qr2 <- rq(lwage ~ female + educ + exper + exper_sq, data = df, tau = tau2)
  summ1 <- summary(qr1, se = "iid")
  summ2 <- summary(qr2, se = "iid")

  beta1 <- coef(qr1)["female"]
  beta2 <- coef(qr2)["female"]
  se1 <- summ1$coefficients["female", "Std. Error"]
  se2 <- summ2$coefficients["female", "Std. Error"]

  diff <- beta2 - beta1
  se_diff <- sqrt(se1^2 + se2^2)
  t_stat <- diff / se_diff
  p_val <- 2 * (1 - pnorm(abs(t_stat)))

  cat(sprintf("\nH0: beta_female(%.2f) = beta_female(%.2f)\n", tau1, tau2))
  cat(sprintf("  beta(%.2f) = %.4f  (SE: %.4f)\n", tau1, beta1, se1))
  cat(sprintf("  beta(%.2f) = %.4f  (SE: %.4f)\n", tau2, beta2, se2))
  cat(sprintf("  Difference = %.4f\n", diff))
  cat(sprintf("  t-statistic = %.2f\n", t_stat))
  cat(sprintf("  p-value = %.6f\n", p_val))
  cat(sprintf("  Result: %s\n", ifelse(p_val < 0.05, "REJECT H0", "FAIL TO REJECT H0")))

  iq_test_list[[length(iq_test_list) + 1]] <- data.frame(
    tau1 = tau1,
    tau2 = tau2,
    beta_tau1 = beta1,
    beta_tau2 = beta2,
    diff = diff,
    se = se_diff,
    t_stat = t_stat,
    p_value = p_val,
    significant = p_val < 0.05,
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# 5. Formal joint test: anova.rq (Koenker-Bassett equality test)
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("JOINT TEST: EQUALITY OF COEFFICIENTS ACROSS QUANTILES\n")
cat("==============================================================================\n")

# Fit the simultaneous model
qr_joint <- rq(lwage ~ female + educ + exper + exper_sq,
                data = df, tau = c(0.1, 0.25, 0.5, 0.75, 0.9))

# anova.rq tests equality of specific coefficients across quantiles
# Test H0: beta_female is constant across all quantiles
cat("\nTesting H0: female coefficient is equal across tau = (0.1, 0.25, 0.5, 0.75, 0.9)\n")
anova_result <- tryCatch({
  anova(qr_joint, test = "Wald")
}, error = function(e) {
  cat(sprintf("anova.rq error: %s\n", e$message))
  NULL
})

if (!is.null(anova_result)) {
  print(anova_result)
}

# ---------------------------------------------------------------------------
# 6. Save all results
# ---------------------------------------------------------------------------
output_path <- "/home/guhaase/projetos/panelbox/examples/quantile/R/results_02_multiple_quantiles.csv"
write.csv(results_df, output_path, row.names = FALSE)
cat(sprintf("\nQuantile process results saved to: %s\n", output_path))

# Save inter-quantile tests
iq_df <- do.call(rbind, iq_test_list)
rownames(iq_df) <- NULL
iq_path <- "/home/guhaase/projetos/panelbox/examples/quantile/R/results_02_interquantile_tests.csv"
write.csv(iq_df, iq_path, row.names = FALSE)
cat(sprintf("Inter-quantile test results saved to: %s\n", iq_path))

# ---------------------------------------------------------------------------
# 7. Summary: Female coefficient across quantiles
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("GENDER WAGE GAP ACROSS QUANTILES\n")
cat("==============================================================================\n")

female_rows <- results_df[results_df$variable == "female" & !is.na(results_df$quantile), ]
female_rows <- female_rows[order(female_rows$quantile), ]

cat(sprintf("%-12s  %12s  %12s  %12s\n", "Quantile", "Coefficient", "Std. Error", "Gap (%)"))
cat(paste(rep("-", 52), collapse = ""), "\n")
for (i in seq_len(nrow(female_rows))) {
  gap_pct <- 100 * (exp(female_rows$coefficient[i]) - 1)
  cat(sprintf("tau = %.2f    %12.6f  %12.6f  %12.1f%%\n",
              female_rows$quantile[i],
              female_rows$coefficient[i],
              female_rows$std_error[i],
              gap_pct))
}

ols_female <- coef(ols_model)["female"]
ols_gap_pct <- 100 * (exp(ols_female) - 1)
cat(sprintf("OLS          %12.6f  %12.6f  %12.1f%%\n",
            ols_female,
            summary(ols_model)$coefficients["female", "Std. Error"],
            ols_gap_pct))

cat("\nDone.\n")
