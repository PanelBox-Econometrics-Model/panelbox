# =============================================================================
# Validation 03: Fixed Effects Quantile Regression - Canay (2011) Two-Step
# Replicates PanelBox Notebook 03 results using R quantreg package
#
# Dataset: firm_production.csv (panel: 500 firms x 10 years)
# Model: log_output ~ log_capital + log_labor + log_materials
# Method: Canay (2011) two-step:
#   Step 1: Estimate FE by within (OLS) estimator
#   Step 2: Subtract FE from Y, run pooled QR on demeaned Y
# =============================================================================

library(quantreg)
library(plm)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/quantile/data/firm_production.csv"
df <- read.csv(data_path)

cat("Dataset loaded:\n")
cat(sprintf("  Observations: %d\n", nrow(df)))
cat(sprintf("  Firms: %d\n", length(unique(df$firm_id))))
cat(sprintf("  Time periods: %d\n", length(unique(df$year))))
cat("\nFirst rows:\n")
print(head(df))

n_obs <- nrow(df)

# ---------------------------------------------------------------------------
# 2. Step 1: Fixed Effects OLS (within estimator)
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("STEP 1: FIXED EFFECTS OLS (WITHIN ESTIMATOR)\n")
cat("==============================================================================\n")

# Create pdata.frame for plm
pdata <- pdata.frame(df, index = c("firm_id", "year"))

fe_ols <- plm(log_output ~ log_capital + log_labor + log_materials,
              data = pdata, model = "within")

cat("\nFE-OLS Results:\n")
print(summary(fe_ols))

fe_ols_coefs <- coef(fe_ols)
cat("\nFE-OLS Coefficients:\n")
print(fe_ols_coefs)

# Returns to scale
rts <- sum(fe_ols_coefs)
cat(sprintf("\nReturns to scale (FE-OLS): %.4f\n", rts))

# ---------------------------------------------------------------------------
# 3. Recover fixed effects (alpha_i)
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("RECOVERING FIXED EFFECTS\n")
cat("==============================================================================\n")

# Get individual fixed effects from plm
alpha_hat <- fixef(fe_ols)

cat(sprintf("Number of FE: %d\n", length(alpha_hat)))
cat(sprintf("Mean FE:  %.4f\n", mean(alpha_hat)))
cat(sprintf("Std FE:   %.4f\n", sd(alpha_hat)))
cat(sprintf("Min FE:   %.4f\n", min(alpha_hat)))
cat(sprintf("Max FE:   %.4f\n", max(alpha_hat)))

# ---------------------------------------------------------------------------
# 4. Step 2: Transform Y and run pooled QR
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("STEP 2: CANAY TWO-STEP QUANTILE REGRESSION\n")
cat("==============================================================================\n")

# Create Y_tilde = Y - alpha_hat_i
df$alpha_hat <- alpha_hat[as.character(df$firm_id)]
df$y_tilde <- df$log_output - df$alpha_hat

cat(sprintf("Y_tilde summary:\n"))
print(summary(df$y_tilde))

# Estimate pooled QR on transformed data
tau_list <- c(0.1, 0.25, 0.5, 0.75, 0.9)
canay_results_list <- list()

for (tau in tau_list) {
  cat(sprintf("\n--- Canay FE-QR: tau = %.2f ---\n", tau))

  qr_model <- rq(y_tilde ~ log_capital + log_labor + log_materials,
                  data = df, tau = tau)
  qr_summ <- summary(qr_model, se = "iid")

  cat("Coefficients:\n")
  print(qr_summ$coefficients)

  coefs <- coef(qr_model)
  se_vals <- qr_summ$coefficients[, "Std. Error"]
  t_vals <- qr_summ$coefficients[, "t value"]
  p_vals <- qr_summ$coefficients[, "Pr(>|t|)"]
  var_names <- names(coefs)

  for (j in seq_along(var_names)) {
    canay_results_list[[length(canay_results_list) + 1]] <- data.frame(
      model_name = sprintf("canay_fe_qr_tau%.2f", tau),
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

# ---------------------------------------------------------------------------
# 5. Also estimate pooled QR (no FE) for comparison
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("POOLED QR (NO FIXED EFFECTS) FOR COMPARISON\n")
cat("==============================================================================\n")

pooled_results_list <- list()

for (tau in tau_list) {
  cat(sprintf("\n--- Pooled QR: tau = %.2f ---\n", tau))

  qr_model <- rq(log_output ~ log_capital + log_labor + log_materials,
                  data = df, tau = tau)
  qr_summ <- summary(qr_model, se = "iid")

  cat("Coefficients:\n")
  print(qr_summ$coefficients)

  coefs <- coef(qr_model)
  se_vals <- qr_summ$coefficients[, "Std. Error"]
  t_vals <- qr_summ$coefficients[, "t value"]
  p_vals <- qr_summ$coefficients[, "Pr(>|t|)"]
  var_names <- names(coefs)

  for (j in seq_along(var_names)) {
    pooled_results_list[[length(pooled_results_list) + 1]] <- data.frame(
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

# ---------------------------------------------------------------------------
# 6. Add FE-OLS results
# ---------------------------------------------------------------------------
fe_ols_se <- summary(fe_ols)$coefficients[, "Std. Error"]
fe_ols_tvals <- summary(fe_ols)$coefficients[, "t-value"]
fe_ols_pvals <- summary(fe_ols)$coefficients[, "Pr(>|t|)"]
fe_var_names <- names(fe_ols_coefs)

fe_ols_list <- list()
for (j in seq_along(fe_var_names)) {
  fe_ols_list[[j]] <- data.frame(
    model_name = "fe_ols",
    quantile = NA,
    variable = fe_var_names[j],
    coefficient = fe_ols_coefs[j],
    std_error = fe_ols_se[j],
    t_statistic = fe_ols_tvals[j],
    p_value = fe_ols_pvals[j],
    n_obs = n_obs,
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# 7. Combine all results and save
# ---------------------------------------------------------------------------
canay_df <- do.call(rbind, canay_results_list)
pooled_df <- do.call(rbind, pooled_results_list)
fe_ols_df <- do.call(rbind, fe_ols_list)

all_results <- rbind(canay_df, pooled_df, fe_ols_df)
rownames(all_results) <- NULL

output_path <- "/home/guhaase/projetos/panelbox/examples/quantile/R/results_03_fe_canay.csv"
write.csv(all_results, output_path, row.names = FALSE)
cat(sprintf("\nAll results saved to: %s\n", output_path))

# ---------------------------------------------------------------------------
# 8. Comparison table: Canay vs Pooled vs FE-OLS
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("COMPARISON: CANAY FE-QR vs POOLED QR vs FE-OLS\n")
cat("==============================================================================\n")

for (var in c("log_capital", "log_labor", "log_materials")) {
  cat(sprintf("\n%s:\n", var))
  cat(sprintf("%-20s", "Method"))
  for (tau in tau_list) {
    cat(sprintf("  tau=%.2f", tau))
  }
  cat("\n")
  cat(paste(rep("-", 70), collapse = ""), "\n")

  # Canay
  cat(sprintf("%-20s", "Canay FE-QR"))
  for (tau in tau_list) {
    row <- canay_df[canay_df$variable == var & canay_df$quantile == tau, ]
    cat(sprintf("  %7.4f", row$coefficient))
  }
  cat("\n")

  # Pooled
  cat(sprintf("%-20s", "Pooled QR"))
  for (tau in tau_list) {
    row <- pooled_df[pooled_df$variable == var & pooled_df$quantile == tau, ]
    cat(sprintf("  %7.4f", row$coefficient))
  }
  cat("\n")

  # Difference
  cat(sprintf("%-20s", "Difference"))
  for (tau in tau_list) {
    canay_row <- canay_df[canay_df$variable == var & canay_df$quantile == tau, ]
    pooled_row <- pooled_df[pooled_df$variable == var & pooled_df$quantile == tau, ]
    cat(sprintf("  %7.4f", canay_row$coefficient - pooled_row$coefficient))
  }
  cat("\n")

  # FE-OLS
  fe_row <- fe_ols_df[fe_ols_df$variable == var, ]
  cat(sprintf("%-20s  %7.4f (constant across quantiles)\n", "FE-OLS", fe_row$coefficient))
}

# ---------------------------------------------------------------------------
# 9. Returns to scale across quantiles (Canay)
# ---------------------------------------------------------------------------
cat("\n==============================================================================\n")
cat("RETURNS TO SCALE ACROSS QUANTILES (CANAY)\n")
cat("==============================================================================\n")

for (tau in tau_list) {
  rows <- canay_df[canay_df$quantile == tau & canay_df$variable != "(Intercept)", ]
  rts_tau <- sum(rows$coefficient)
  cat(sprintf("  tau = %.2f: RTS = %.4f\n", tau, rts_tau))
}

fe_rts <- sum(fe_ols_coefs)
cat(sprintf("  FE-OLS:    RTS = %.4f\n", fe_rts))

cat("\nDone.\n")
