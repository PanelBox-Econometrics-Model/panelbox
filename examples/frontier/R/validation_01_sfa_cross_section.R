# ==============================================================================
# Validation Script: Stochastic Frontier Analysis - Cross-Section
# ==============================================================================
# Purpose: Reproduce PanelBox SFA cross-section results using R's frontier package
# Notebook: examples/frontier/notebooks/01_introduction_sfa.ipynb
# Data: hospital_data.csv (500 hospitals, cross-section)
# Models: Half-normal SFA, Truncated-normal SFA
# ==============================================================================

library(frontier)

cat("=" , rep("=", 69), "\n", sep = "")
cat("SFA Cross-Section Validation (R frontier package)\n")
cat("=" , rep("=", 69), "\n", sep = "")

# ------------------------------------------------------------------------------
# 1. Load Data
# ------------------------------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/frontier/data/hospital_data.csv"
hospital <- read.csv(data_path)

cat("\nDataset shape:", nrow(hospital), "x", ncol(hospital), "\n")
cat("Variables:", paste(names(hospital), collapse = ", "), "\n")
cat("\nSummary Statistics:\n")
print(summary(hospital[, c("log_cases", "log_doctors", "log_nurses", "log_beds",
                            "teaching", "urban")]))

# ------------------------------------------------------------------------------
# 2. OLS Baseline
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("OLS BASELINE\n")
cat(rep("=", 70), "\n", sep = "")

ols_model <- lm(log_cases ~ log_doctors + log_nurses + log_beds + teaching + urban,
                data = hospital)
cat("\nOLS Results:\n")
print(summary(ols_model))

# Check residual skewness
resid_ols <- residuals(ols_model)
n <- length(resid_ols)
skewness <- (sum((resid_ols - mean(resid_ols))^3) / n) / (sum((resid_ols - mean(resid_ols))^2) / n)^1.5
cat(sprintf("\nOLS Residual Skewness: %.4f (expect < 0 for production frontier)\n", skewness))

# Save OLS log-likelihood for LR test
ols_loglik <- logLik(ols_model)
cat(sprintf("OLS Log-Likelihood: %.4f\n", as.numeric(ols_loglik)))

# ------------------------------------------------------------------------------
# 3. SFA Model 1: Half-Normal
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("SFA MODEL 1: HALF-NORMAL\n")
cat(rep("=", 70), "\n", sep = "")

# frontier::sfa uses the Battese & Coelli (1992/1995) parameterization
# For cross-section with half-normal: truncNorm = FALSE, mu = 0
sfa_hn <- sfa(log_cases ~ log_doctors + log_nurses + log_beds + teaching + urban,
              data = hospital,
              ineffDecrease = TRUE,   # production frontier (u reduces output)
              truncNorm = FALSE)      # half-normal (not truncated normal)

cat("\nHalf-Normal SFA Results:\n")
print(summary(sfa_hn))

# Extract parameters
coef_hn <- coef(sfa_hn)
cat("\nCoefficients:\n")
print(coef_hn)

# Variance decomposition
# frontier uses parameterization: sigmaSq = sigma_v^2 + sigma_u^2, gamma = sigma_u^2 / sigmaSq
sigma_sq_hn <- coef_hn["sigmaSq"]
gamma_hn <- coef_hn["gamma"]
sigma_u_sq_hn <- sigma_sq_hn * gamma_hn
sigma_v_sq_hn <- sigma_sq_hn * (1 - gamma_hn)
sigma_u_hn <- sqrt(sigma_u_sq_hn)
sigma_v_hn <- sqrt(sigma_v_sq_hn)
lambda_hn <- sigma_u_hn / sigma_v_hn

cat(sprintf("\nVariance Decomposition (Half-Normal):\n"))
cat(sprintf("  sigma_sq (total):        %.6f\n", sigma_sq_hn))
cat(sprintf("  sigma_u_sq (inefficiency): %.6f\n", sigma_u_sq_hn))
cat(sprintf("  sigma_v_sq (noise):        %.6f\n", sigma_v_sq_hn))
cat(sprintf("  sigma_u:                   %.6f\n", sigma_u_hn))
cat(sprintf("  sigma_v:                   %.6f\n", sigma_v_hn))
cat(sprintf("  gamma (sigma_u^2/sigma^2): %.6f\n", gamma_hn))
cat(sprintf("  lambda (sigma_u/sigma_v):  %.6f\n", lambda_hn))
cat(sprintf("  Log-Likelihood:            %.4f\n", logLik(sfa_hn)))

# Technical Efficiency (BC estimator)
te_hn <- efficiencies(sfa_hn)
cat(sprintf("\nTechnical Efficiency Summary (Half-Normal, BC estimator):\n"))
cat(sprintf("  Mean TE:   %.4f\n", mean(te_hn)))
cat(sprintf("  Std TE:    %.4f\n", sd(te_hn)))
cat(sprintf("  Min TE:    %.4f\n", min(te_hn)))
cat(sprintf("  Max TE:    %.4f\n", max(te_hn)))
cat(sprintf("  Median TE: %.4f\n", median(te_hn)))

# LR test for the presence of inefficiency
# H0: gamma = 0 (no inefficiency, OLS is adequate)
sfa_loglik_hn <- as.numeric(logLik(sfa_hn))
lr_stat_hn <- -2 * (as.numeric(ols_loglik) - sfa_loglik_hn)
# Mixed chi-squared: p-value = 0.5 * P(chi2(1) > LR)
lr_pval_hn <- 0.5 * pchisq(lr_stat_hn, df = 1, lower.tail = FALSE)
cat(sprintf("\nLR Test for Inefficiency (Half-Normal):\n"))
cat(sprintf("  LR Statistic: %.4f\n", lr_stat_hn))
cat(sprintf("  P-value (mixed chi-sq): %.6f\n", lr_pval_hn))
if (lr_pval_hn < 0.05) {
  cat("  Conclusion: Reject H0 - inefficiency is significant\n")
} else {
  cat("  Conclusion: Fail to reject H0 - no evidence of inefficiency\n")
}

# ------------------------------------------------------------------------------
# 4. SFA Model 2: Truncated Normal
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("SFA MODEL 2: TRUNCATED NORMAL\n")
cat(rep("=", 70), "\n", sep = "")

sfa_tn <- sfa(log_cases ~ log_doctors + log_nurses + log_beds + teaching + urban,
              data = hospital,
              ineffDecrease = TRUE,
              truncNorm = TRUE)      # truncated normal (mu estimated)

cat("\nTruncated Normal SFA Results:\n")
print(summary(sfa_tn))

coef_tn <- coef(sfa_tn)
cat("\nCoefficients:\n")
print(coef_tn)

# Variance decomposition
sigma_sq_tn <- coef_tn["sigmaSq"]
gamma_tn <- coef_tn["gamma"]
sigma_u_sq_tn <- sigma_sq_tn * gamma_tn
sigma_v_sq_tn <- sigma_sq_tn * (1 - gamma_tn)
sigma_u_tn <- sqrt(sigma_u_sq_tn)
sigma_v_tn <- sqrt(sigma_v_sq_tn)
lambda_tn <- sigma_u_tn / sigma_v_tn
mu_tn <- coef_tn["mu"]

cat(sprintf("\nVariance Decomposition (Truncated Normal):\n"))
cat(sprintf("  sigma_sq (total):        %.6f\n", sigma_sq_tn))
cat(sprintf("  sigma_u_sq (inefficiency): %.6f\n", sigma_u_sq_tn))
cat(sprintf("  sigma_v_sq (noise):        %.6f\n", sigma_v_sq_tn))
cat(sprintf("  sigma_u:                   %.6f\n", sigma_u_tn))
cat(sprintf("  sigma_v:                   %.6f\n", sigma_v_tn))
cat(sprintf("  gamma (sigma_u^2/sigma^2): %.6f\n", gamma_tn))
cat(sprintf("  lambda (sigma_u/sigma_v):  %.6f\n", lambda_tn))
cat(sprintf("  mu (truncation point):     %.6f\n", mu_tn))
cat(sprintf("  Log-Likelihood:            %.4f\n", logLik(sfa_tn)))

# Technical Efficiency (BC estimator)
te_tn <- efficiencies(sfa_tn)
cat(sprintf("\nTechnical Efficiency Summary (Truncated Normal, BC estimator):\n"))
cat(sprintf("  Mean TE:   %.4f\n", mean(te_tn)))
cat(sprintf("  Std TE:    %.4f\n", sd(te_tn)))
cat(sprintf("  Min TE:    %.4f\n", min(te_tn)))
cat(sprintf("  Max TE:    %.4f\n", max(te_tn)))
cat(sprintf("  Median TE: %.4f\n", median(te_tn)))

# LR test for truncated normal
sfa_loglik_tn <- as.numeric(logLik(sfa_tn))
lr_stat_tn <- -2 * (as.numeric(ols_loglik) - sfa_loglik_tn)
lr_pval_tn <- 0.5 * pchisq(lr_stat_tn, df = 1, lower.tail = FALSE)
cat(sprintf("\nLR Test for Inefficiency (Truncated Normal):\n"))
cat(sprintf("  LR Statistic: %.4f\n", lr_stat_tn))
cat(sprintf("  P-value (mixed chi-sq): %.6f\n", lr_pval_tn))

# ------------------------------------------------------------------------------
# 5. Model Comparison
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("MODEL COMPARISON\n")
cat(rep("=", 70), "\n", sep = "")

n_params_hn <- length(coef_hn)
n_params_tn <- length(coef_tn)
aic_hn <- -2 * sfa_loglik_hn + 2 * n_params_hn
aic_tn <- -2 * sfa_loglik_tn + 2 * n_params_tn
bic_hn <- -2 * sfa_loglik_hn + log(nrow(hospital)) * n_params_hn
bic_tn <- -2 * sfa_loglik_tn + log(nrow(hospital)) * n_params_tn

cat(sprintf("%-20s %12s %12s\n", "", "Half-Normal", "Trunc-Normal"))
cat(sprintf("%-20s %12.4f %12.4f\n", "Log-Likelihood", sfa_loglik_hn, sfa_loglik_tn))
cat(sprintf("%-20s %12.4f %12.4f\n", "AIC", aic_hn, aic_tn))
cat(sprintf("%-20s %12.4f %12.4f\n", "BIC", bic_hn, bic_tn))
cat(sprintf("%-20s %12d %12d\n", "N Parameters", n_params_hn, n_params_tn))
cat(sprintf("%-20s %12.4f %12.4f\n", "sigma_u", sigma_u_hn, sigma_u_tn))
cat(sprintf("%-20s %12.4f %12.4f\n", "sigma_v", sigma_v_hn, sigma_v_tn))
cat(sprintf("%-20s %12.4f %12.4f\n", "gamma", gamma_hn, gamma_tn))
cat(sprintf("%-20s %12.4f %12.4f\n", "lambda", lambda_hn, lambda_tn))
cat(sprintf("%-20s %12.4f %12.4f\n", "Mean TE", mean(te_hn), mean(te_tn)))
cat(sprintf("%-20s %12.4f %12.4f\n", "Std TE", sd(te_hn), sd(te_tn)))
cat(sprintf("%-20s %12.4f %12.4f\n", "Min TE", min(te_hn), min(te_tn)))
cat(sprintf("%-20s %12.4f %12.4f\n", "Max TE", max(te_hn), max(te_tn)))

# Correlation between efficiency scores
cat(sprintf("\nCorrelation between TE (HN) and TE (TN): %.4f\n",
            cor(te_hn, te_tn)))

# ------------------------------------------------------------------------------
# 6. Also run with frontier::riceProdPhil for reproducibility
# ------------------------------------------------------------------------------
cat("\n", rep("=", 70), "\n", sep = "")
cat("ROBUSTNESS CHECK: frontier::riceProdPhil DATASET\n")
cat(rep("=", 70), "\n", sep = "")

data(riceProdPhil, package = "frontier")
cat(sprintf("riceProdPhil dataset: %d observations, %d variables\n",
            nrow(riceProdPhil), ncol(riceProdPhil)))
cat("Variables:", paste(names(riceProdPhil), collapse = ", "), "\n")

# Estimate SFA on riceProdPhil (Cobb-Douglas production function)
# Model: log(PROD) ~ log(AREA) + log(LABOR) + log(NPK)
sfa_rice <- sfa(log(PROD) ~ log(AREA) + log(LABOR) + log(NPK),
                data = riceProdPhil,
                ineffDecrease = TRUE,
                truncNorm = FALSE)

cat("\nriceProdPhil Half-Normal SFA:\n")
print(summary(sfa_rice))

te_rice <- efficiencies(sfa_rice)
cat(sprintf("\nriceProdPhil Mean TE: %.4f (SD: %.4f)\n", mean(te_rice), sd(te_rice)))

# ------------------------------------------------------------------------------
# 7. Save Results to CSV
# ------------------------------------------------------------------------------
output_dir <- "/home/guhaase/projetos/panelbox/examples/frontier/R"
output_file <- file.path(output_dir, "results_sfa_cross_section.csv")

# Build results data frame
results <- data.frame(
  model_name = character(),
  variable = character(),
  coefficient = numeric(),
  std_error = numeric(),
  statistic = numeric(),
  p_value = numeric(),
  sigma_v = numeric(),
  sigma_u = numeric(),
  sigma_v_sq = numeric(),
  sigma_u_sq = numeric(),
  gamma = numeric(),
  lambda_param = numeric(),
  log_likelihood = numeric(),
  aic = numeric(),
  bic = numeric(),
  mean_te = numeric(),
  stringsAsFactors = FALSE
)

# Helper function to extract rows from sfa summary
extract_sfa_results <- function(sfa_result, model_name, te_vec) {
  s <- summary(sfa_result)
  # Get coefficient table
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
    log_likelihood = rep(ll, nrow(coef_tab)),
    aic = rep(a, nrow(coef_tab)),
    bic = rep(b, nrow(coef_tab)),
    mean_te = rep(mte, nrow(coef_tab)),
    stringsAsFactors = FALSE
  )
  rownames(rows) <- NULL
  return(rows)
}

# Extract results for each model
results_hn <- extract_sfa_results(sfa_hn, "sfa_half_normal", te_hn)
results_tn <- extract_sfa_results(sfa_tn, "sfa_truncated_normal", te_tn)
results_rice <- extract_sfa_results(sfa_rice, "sfa_ricefarm_hn", te_rice)

# Combine
results <- rbind(results_hn, results_tn, results_rice)

# Save
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", output_file))
cat(sprintf("Total rows: %d\n", nrow(results)))

# Also save efficiency scores
eff_output <- file.path(output_dir, "efficiency_scores_cross_section.csv")
eff_df <- data.frame(
  hospital_id = hospital$hospital_id,
  te_half_normal = as.numeric(te_hn),
  te_truncated_normal = as.numeric(te_tn)
)
write.csv(eff_df, eff_output, row.names = FALSE)
cat(sprintf("Efficiency scores saved to: %s\n", eff_output))

cat("\n", rep("=", 70), "\n", sep = "")
cat("CROSS-SECTION VALIDATION COMPLETE\n")
cat(rep("=", 70), "\n", sep = "")
