# ==============================================================================
# Validation Script: Stochastic Frontier Analysis - Panel Data
# ==============================================================================
# Purpose: Reproduce PanelBox panel SFA results using R's frontier package
# Notebook: examples/frontier/notebooks/02_panel_sfa.ipynb
# Data: bank_panel.csv (100 banks x 10 years)
# Models: Pitt-Lee (time-invariant), BC92 (time-varying with eta)
# ==============================================================================

library(frontier)
library(plm)

cat("=" , rep("=", 69), "\n", sep = "")
cat("SFA Panel Data Validation (R frontier package)\n")
cat("=" , rep("=", 69), "\n", sep = "")

# ------------------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/frontier/data/bank_panel.csv"
bank_raw <- read.csv(data_path)

cat("\nDataset shape:", nrow(bank_raw), "x", ncol(bank_raw), "\n")
cat("Variables:", paste(names(bank_raw), collapse = ", "), "\n")
cat(sprintf("Number of banks: %d\n", length(unique(bank_raw$bank_id))))
cat(sprintf("Number of years: %d\n", length(unique(bank_raw$year))))
cat(sprintf("Time range: %d - %d\n", min(bank_raw$year), max(bank_raw$year)))

# Check balance
obs_per_bank <- table(bank_raw$bank_id)
cat(sprintf("Balanced panel: %s\n", ifelse(length(unique(obs_per_bank)) == 1, "YES", "NO")))

# Convert to pdata.frame for frontier to recognize panel structure
bank <- pdata.frame(bank_raw, index = c("bank_id", "year"))

cat("\nSummary Statistics:\n")
print(summary(bank[, c("log_loans", "log_labor", "log_capital", "log_deposits",
                        "public_ownership", "npl_ratio")]))

# ------------------------------------------------------------------------------
# 2. OLS Baseline (Pooled)
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("POOLED OLS BASELINE\n")
cat(rep("=", 70), "\n", sep = "")

ols_panel <- lm(log_loans ~ log_labor + log_capital + log_deposits, data = bank)
cat("\nPooled OLS:\n")
print(summary(ols_panel))

ols_loglik <- as.numeric(logLik(ols_panel))
cat(sprintf("OLS Log-Likelihood: %.4f\n", ols_loglik))

# Skewness check
resid_ols <- residuals(ols_panel)
n <- length(resid_ols)
skew <- (sum((resid_ols - mean(resid_ols))^3) / n) / (sum((resid_ols - mean(resid_ols))^2) / n)^1.5
cat(sprintf("OLS Residual Skewness: %.4f\n", skew))

# ------------------------------------------------------------------------------
# 3. Pitt-Lee Model (Time-Invariant Inefficiency)
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("PITT-LEE (1981) - TIME-INVARIANT INEFFICIENCY\n")
cat(rep("=", 70), "\n", sep = "")

# frontier::sfa with panel data:
# - timeEffect = FALSE => time-invariant (Pitt-Lee)
# - truncNorm = FALSE => half-normal distribution
sfa_pl <- sfa(log_loans ~ log_labor + log_capital + log_deposits,
              data = bank,
              ineffDecrease = TRUE,
              truncNorm = FALSE,
              timeEffect = FALSE)

cat("\nPitt-Lee SFA Results:\n")
print(summary(sfa_pl))

# Extract parameters
coef_pl <- coef(sfa_pl)
sigma_sq_pl <- coef_pl["sigmaSq"]
gamma_pl <- coef_pl["gamma"]
sigma_u_sq_pl <- sigma_sq_pl * gamma_pl
sigma_v_sq_pl <- sigma_sq_pl * (1 - gamma_pl)
sigma_u_pl <- sqrt(sigma_u_sq_pl)
sigma_v_pl <- sqrt(sigma_v_sq_pl)
lambda_pl <- sigma_u_pl / sigma_v_pl
loglik_pl <- as.numeric(logLik(sfa_pl))

cat(sprintf("\nVariance Decomposition (Pitt-Lee):\n"))
cat(sprintf("  sigma_sq:   %.6f\n", sigma_sq_pl))
cat(sprintf("  sigma_u_sq: %.6f\n", sigma_u_sq_pl))
cat(sprintf("  sigma_v_sq: %.6f\n", sigma_v_sq_pl))
cat(sprintf("  sigma_u:    %.6f\n", sigma_u_pl))
cat(sprintf("  sigma_v:    %.6f\n", sigma_v_pl))
cat(sprintf("  gamma:      %.6f\n", gamma_pl))
cat(sprintf("  lambda:     %.6f\n", lambda_pl))
cat(sprintf("  Log-Lik:    %.4f\n", loglik_pl))

# Efficiency
te_pl <- efficiencies(sfa_pl)
cat(sprintf("\nTechnical Efficiency (Pitt-Lee):\n"))
cat(sprintf("  Mean TE:   %.4f\n", mean(te_pl)))
cat(sprintf("  Std TE:    %.4f\n", sd(te_pl)))
cat(sprintf("  Min TE:    %.4f\n", min(te_pl)))
cat(sprintf("  Max TE:    %.4f\n", max(te_pl)))
cat(sprintf("  Median TE: %.4f\n", median(te_pl)))

# LR test for presence of inefficiency
lr_stat_pl <- -2 * (ols_loglik - loglik_pl)
lr_pval_pl <- 0.5 * pchisq(lr_stat_pl, df = 1, lower.tail = FALSE)
cat(sprintf("\nLR Test (OLS vs Pitt-Lee):\n"))
cat(sprintf("  LR Statistic: %.4f\n", lr_stat_pl))
cat(sprintf("  P-value (mixed chi-sq): %.6f\n", lr_pval_pl))

# ------------------------------------------------------------------------------
# 4. Battese-Coelli 1992 (Time-Varying, eta decay)
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("BATTESE-COELLI (1992) - TIME-VARYING INEFFICIENCY\n")
cat(rep("=", 70), "\n", sep = "")

# frontier::sfa with timeEffect = TRUE => BC92 (time-varying with eta)
# u_it = u_i * exp[-eta * (t - T)]
sfa_bc92 <- sfa(log_loans ~ log_labor + log_capital + log_deposits,
                data = bank,
                ineffDecrease = TRUE,
                truncNorm = FALSE,
                timeEffect = TRUE)

cat("\nBC92 SFA Results:\n")
print(summary(sfa_bc92))

# Extract parameters
coef_bc92 <- coef(sfa_bc92)
sigma_sq_bc92 <- coef_bc92["sigmaSq"]
gamma_bc92 <- coef_bc92["gamma"]
sigma_u_sq_bc92 <- sigma_sq_bc92 * gamma_bc92
sigma_v_sq_bc92 <- sigma_sq_bc92 * (1 - gamma_bc92)
sigma_u_bc92 <- sqrt(sigma_u_sq_bc92)
sigma_v_bc92 <- sqrt(sigma_v_sq_bc92)
lambda_bc92 <- sigma_u_bc92 / sigma_v_bc92
eta_bc92 <- coef_bc92["time"]
loglik_bc92 <- as.numeric(logLik(sfa_bc92))

cat(sprintf("\nVariance Decomposition (BC92):\n"))
cat(sprintf("  sigma_sq:   %.6f\n", sigma_sq_bc92))
cat(sprintf("  sigma_u_sq: %.6f\n", sigma_u_sq_bc92))
cat(sprintf("  sigma_v_sq: %.6f\n", sigma_v_sq_bc92))
cat(sprintf("  sigma_u:    %.6f\n", sigma_u_bc92))
cat(sprintf("  sigma_v:    %.6f\n", sigma_v_bc92))
cat(sprintf("  gamma:      %.6f\n", gamma_bc92))
cat(sprintf("  lambda:     %.6f\n", lambda_bc92))
cat(sprintf("  eta (time): %.6f\n", eta_bc92))
cat(sprintf("  Log-Lik:    %.4f\n", loglik_bc92))

if (eta_bc92 > 0) {
  cat("  Interpretation: eta > 0 => Efficiency IMPROVES over time (learning)\n")
} else if (eta_bc92 < 0) {
  cat("  Interpretation: eta < 0 => Efficiency WORSENS over time (degradation)\n")
} else {
  cat("  Interpretation: eta = 0 => Time-invariant (reduces to Pitt-Lee)\n")
}

# Efficiency (time-varying)
te_bc92 <- efficiencies(sfa_bc92)
cat(sprintf("\nTechnical Efficiency (BC92):\n"))
cat(sprintf("  Mean TE:   %.4f\n", mean(te_bc92)))
cat(sprintf("  Std TE:    %.4f\n", sd(te_bc92)))
cat(sprintf("  Min TE:    %.4f\n", min(te_bc92)))
cat(sprintf("  Max TE:    %.4f\n", max(te_bc92)))

# Average efficiency by year
cat("\nAverage TE by Year (BC92):\n")
# te_bc92 is a vector with one value per observation, same order as data
bank_raw$te_bc92 <- as.numeric(te_bc92)
avg_te_by_year <- aggregate(te_bc92 ~ year, data = bank_raw, FUN = mean)
print(avg_te_by_year)

# ------------------------------------------------------------------------------
# 5. LR Test: Pitt-Lee vs BC92
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("LIKELIHOOD RATIO TEST: Pitt-Lee vs BC92\n")
cat(rep("=", 70), "\n", sep = "")
cat("H0: eta = 0 (time-invariant inefficiency - Pitt-Lee)\n")
cat("H1: eta != 0 (time-varying inefficiency - BC92)\n")

lr_stat <- -2 * (loglik_pl - loglik_bc92)
lr_pval <- pchisq(lr_stat, df = 1, lower.tail = FALSE)

cat(sprintf("\nLog-likelihood (Pitt-Lee): %.4f\n", loglik_pl))
cat(sprintf("Log-likelihood (BC92):     %.4f\n", loglik_bc92))
cat(sprintf("LR Statistic: %.4f\n", lr_stat))
cat(sprintf("Degrees of Freedom: 1\n"))
cat(sprintf("P-value: %.6f\n", lr_pval))

if (lr_pval < 0.05) {
  cat("Conclusion: Reject H0 - BC92 (time-varying) significantly better\n")
} else {
  cat("Conclusion: Fail to reject H0 - Pitt-Lee (time-invariant) adequate\n")
}

# ------------------------------------------------------------------------------
# 6. BC95 Model with Inefficiency Determinants
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("BATTESE-COELLI (1995) - INEFFICIENCY DETERMINANTS\n")
cat(rep("=", 70), "\n", sep = "")

# BC95: inefficiency mean is a function of Z variables
# In frontier, use zIntercept and zu argument
sfa_bc95 <- sfa(log_loans ~ log_labor + log_capital + log_deposits |
                  log_assets + public_ownership + npl_ratio,
                data = bank,
                ineffDecrease = TRUE,
                truncNorm = TRUE,
                timeEffect = FALSE)

cat("\nBC95 SFA Results:\n")
print(summary(sfa_bc95))

coef_bc95 <- coef(sfa_bc95)
sigma_sq_bc95 <- coef_bc95["sigmaSq"]
gamma_bc95 <- coef_bc95["gamma"]
loglik_bc95 <- as.numeric(logLik(sfa_bc95))

# Efficiency
te_bc95 <- efficiencies(sfa_bc95)
cat(sprintf("\nTechnical Efficiency (BC95):\n"))
cat(sprintf("  Mean TE:   %.4f\n", mean(te_bc95)))
cat(sprintf("  Std TE:    %.4f\n", sd(te_bc95)))
cat(sprintf("  Min TE:    %.4f\n", min(te_bc95)))
cat(sprintf("  Max TE:    %.4f\n", max(te_bc95)))

# Efficiency by ownership
bank_raw$te_bc95 <- as.numeric(te_bc95)
avg_by_ownership <- aggregate(te_bc95 ~ public_ownership, data = bank_raw, FUN = mean)
cat("\nMean TE by Ownership (BC95):\n")
print(avg_by_ownership)

# Mann-Whitney test
public_te <- bank_raw$te_bc95[bank_raw$public_ownership == 1]
private_te <- bank_raw$te_bc95[bank_raw$public_ownership == 0]
wt <- wilcox.test(public_te, private_te, alternative = "two.sided")
cat(sprintf("\nMann-Whitney Test (Public vs Private):\n"))
cat(sprintf("  U Statistic: %.2f\n", wt$statistic))
cat(sprintf("  P-value: %.6f\n", wt$p.value))

# ------------------------------------------------------------------------------
# 7. Comprehensive Model Comparison
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("COMPREHENSIVE MODEL COMPARISON\n")
cat(rep("=", 70), "\n", sep = "")

n_params_pl <- length(coef_pl)
n_params_bc92 <- length(coef_bc92)
n_params_bc95 <- length(coef_bc95)
n_obs <- nrow(bank)

aic_pl <- -2 * loglik_pl + 2 * n_params_pl
aic_bc92 <- -2 * loglik_bc92 + 2 * n_params_bc92
aic_bc95 <- -2 * loglik_bc95 + 2 * n_params_bc95

bic_pl <- -2 * loglik_pl + log(n_obs) * n_params_pl
bic_bc92 <- -2 * loglik_bc92 + log(n_obs) * n_params_bc92
bic_bc95 <- -2 * loglik_bc95 + log(n_obs) * n_params_bc95

cat(sprintf("%-20s %12s %12s %12s\n", "", "Pitt-Lee", "BC92", "BC95"))
cat(rep("-", 56), "\n", sep = "")
cat(sprintf("%-20s %12.4f %12.4f %12.4f\n", "Log-Likelihood", loglik_pl, loglik_bc92, loglik_bc95))
cat(sprintf("%-20s %12.4f %12.4f %12.4f\n", "AIC", aic_pl, aic_bc92, aic_bc95))
cat(sprintf("%-20s %12.4f %12.4f %12.4f\n", "BIC", bic_pl, bic_bc92, bic_bc95))
cat(sprintf("%-20s %12d %12d %12d\n", "N Parameters", n_params_pl, n_params_bc92, n_params_bc95))
cat(sprintf("%-20s %12.4f %12.4f %12.4f\n", "Mean TE", mean(te_pl), mean(te_bc92), mean(te_bc95)))
cat(sprintf("%-20s %12.4f %12.4f %12.4f\n", "gamma", gamma_pl, gamma_bc92, gamma_bc95))

# Identify best
models <- c("Pitt-Lee", "BC92", "BC95")
aics <- c(aic_pl, aic_bc92, aic_bc95)
bics <- c(bic_pl, bic_bc92, bic_bc95)
cat(sprintf("\nBest by AIC: %s\n", models[which.min(aics)]))
cat(sprintf("Best by BIC: %s\n", models[which.min(bics)]))

# ------------------------------------------------------------------------------
# 8. Save Results to CSV
# ------------------------------------------------------------------------------
output_dir <- "/home/guhaase/projetos/panelbox/examples/frontier/R"
output_file <- file.path(output_dir, "results_sfa_panel.csv")

# Helper to extract results
extract_sfa_panel_results <- function(sfa_result, model_name, te_vec) {
  s <- summary(sfa_result)
  coef_tab <- s$mleParam
  var_names <- rownames(coef_tab)

  coefs <- coef(sfa_result)
  sigma_sq <- coefs["sigmaSq"]
  gam <- coefs["gamma"]
  sv_sq <- sigma_sq * (1 - gam)
  su_sq <- sigma_sq * gam
  sv <- sqrt(sv_sq)
  su <- sqrt(su_sq)
  lam <- su / sv
  ll <- as.numeric(logLik(sfa_result))
  np <- length(coefs)
  n <- nobs(sfa_result)
  a <- -2 * ll + 2 * np
  b <- -2 * ll + log(n) * np
  mte <- mean(te_vec)

  # Extract eta if present
  eta_val <- ifelse("time" %in% names(coefs), coefs["time"], NA)

  rows <- data.frame(
    model_name = rep(model_name, nrow(coef_tab)),
    variable = var_names,
    coefficient = coef_tab[, "Estimate"],
    std_error = coef_tab[, "Std. Error"],
    statistic = coef_tab[, "z value"],
    p_value = coef_tab[, "Pr(>|z|)"],
    sigma_v = rep(sv, nrow(coef_tab)),
    sigma_u = rep(su, nrow(coef_tab)),
    sigma_v_sq = rep(sv_sq, nrow(coef_tab)),
    sigma_u_sq = rep(su_sq, nrow(coef_tab)),
    gamma = rep(gam, nrow(coef_tab)),
    lambda_param = rep(lam, nrow(coef_tab)),
    eta = rep(eta_val, nrow(coef_tab)),
    log_likelihood = rep(ll, nrow(coef_tab)),
    aic = rep(a, nrow(coef_tab)),
    bic = rep(b, nrow(coef_tab)),
    mean_te = rep(mte, nrow(coef_tab)),
    stringsAsFactors = FALSE
  )
  rownames(rows) <- NULL
  return(rows)
}

results_pl <- extract_sfa_panel_results(sfa_pl, "pitt_lee", te_pl)
results_bc92 <- extract_sfa_panel_results(sfa_bc92, "bc92", te_bc92)
results_bc95 <- extract_sfa_panel_results(sfa_bc95, "bc95", te_bc95)

results <- rbind(results_pl, results_bc92, results_bc95)

write.csv(results, output_file, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", output_file))
cat(sprintf("Total rows: %d\n", nrow(results)))

# Save the combined results CSV as requested by the spec
combined_output <- file.path(output_dir, "results_frontier.csv")
# Combine cross-section (if exists) and panel results
results_xs_file <- file.path(output_dir, "results_sfa_cross_section.csv")
if (file.exists(results_xs_file)) {
  results_xs <- read.csv(results_xs_file)
  # Align columns
  common_cols <- intersect(names(results_xs), names(results))
  combined <- rbind(results_xs[, common_cols], results[, common_cols])
  write.csv(combined, combined_output, row.names = FALSE)
  cat(sprintf("Combined frontier results saved to: %s\n", combined_output))
} else {
  write.csv(results, combined_output, row.names = FALSE)
  cat(sprintf("Panel frontier results saved as combined to: %s\n", combined_output))
}

cat("\n", rep("=", 70), "\n", sep = "")
cat("PANEL SFA VALIDATION COMPLETE\n")
cat(rep("=", 70), "\n", sep = "")
