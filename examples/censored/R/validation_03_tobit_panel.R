# =============================================================================
# Validation Script 03: Tobit Panel (Random Effects)
# PanelBox vs R (censReg::censReg with panel)
#
# Dataset: health_expenditure_panel.csv
# Model: expenditure ~ income + age + chronic + insurance + female + bmi
# Panel: id (individual), time (period)
# Censoring: left-censored at 0
# =============================================================================

library(censReg)
library(plm)

# --- Load data ---------------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/censored/data/health_expenditure_panel.csv"
df <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("N observations:", nrow(df), "\n")
cat("N individuals:", length(unique(df$id)), "\n")
cat("N time periods:", length(unique(df$time)), "\n")
cat("N censored (expenditure == 0):", sum(df$expenditure == 0), "\n")
cat("N uncensored (expenditure > 0):", sum(df$expenditure > 0), "\n")
cat("Censoring rate:", round(mean(df$expenditure == 0) * 100, 2), "%\n\n")

# Convert to panel data frame
pdf <- pdata.frame(df, index = c("id", "time"))

# --- Model 1: Pooled Tobit (cross-sectional, ignoring panel structure) -------
cat("=== Pooled Tobit (censReg, no panel structure) ===\n")
tobit_pooled <- censReg(expenditure ~ income + age + chronic + insurance +
                          female + bmi, left = 0, right = Inf, data = df)
cat(capture.output(summary(tobit_pooled)), sep = "\n")
cat("\n")

# --- Model 2: Random Effects Tobit (panel) -----------------------------------
cat("=== Random Effects Tobit (censReg, panel) ===\n")
cat("Estimating RE Tobit with Gauss-Hermite quadrature...\n")
tobit_re <- censReg(expenditure ~ income + age + chronic + insurance +
                      female + bmi, left = 0, right = Inf,
                    data = pdf, method = "BHHH", nGHQ = 8)
cat(capture.output(summary(tobit_re)), sep = "\n")
cat("\n")

# --- Extract results ---------------------------------------------------------

# Pooled Tobit results
pooled_coefs <- summary(tobit_pooled)$estimate
pooled_results <- data.frame(
  model_name = "tobit_pooled",
  variable = rownames(pooled_coefs),
  coefficient = pooled_coefs[, "Estimate"],
  std_error = pooled_coefs[, "Std. error"],
  statistic = pooled_coefs[, "t value"],
  p_value = pooled_coefs[, "Pr(> t)"],
  stringsAsFactors = FALSE
)

# RE Tobit results
re_coefs <- summary(tobit_re)$estimate
re_results <- data.frame(
  model_name = "tobit_re_panel",
  variable = rownames(re_coefs),
  coefficient = re_coefs[, "Estimate"],
  std_error = re_coefs[, "Std. error"],
  statistic = re_coefs[, "t value"],
  p_value = re_coefs[, "Pr(> t)"],
  stringsAsFactors = FALSE
)

all_results <- rbind(pooled_results, re_results)
row.names(all_results) <- NULL

# --- Model-level statistics --------------------------------------------------

# Pooled: sigma, log-likelihood
sigma_pooled <- exp(coef(tobit_pooled)["logSigma"])
loglik_pooled <- as.numeric(logLik(tobit_pooled))

# RE panel: sigma_u (individual), sigma_e (idiosyncratic), rho (ICC)
# censReg panel provides logSigmaMu (log of sigma_u) and logSigmaNu (log of sigma_e)
re_params <- coef(tobit_re)
sigma_u <- exp(re_params["logSigmaMu"])
sigma_e <- exp(re_params["logSigmaNu"])
loglik_re <- as.numeric(logLik(tobit_re))

# ICC: rho = sigma_u^2 / (sigma_u^2 + sigma_e^2)
rho_icc <- sigma_u^2 / (sigma_u^2 + sigma_e^2)

model_stats <- data.frame(
  model_name = c("tobit_pooled", "tobit_pooled", "tobit_pooled", "tobit_pooled",
                 "tobit_re_panel", "tobit_re_panel", "tobit_re_panel",
                 "tobit_re_panel", "tobit_re_panel", "tobit_re_panel",
                 "tobit_re_panel"),
  variable = c("_sigma", "_log_likelihood", "_n_censored", "_n_uncensored",
               "_sigma_e", "_sigma_u", "_rho_icc", "_log_likelihood",
               "_n_censored", "_n_uncensored", "_n_individuals"),
  coefficient = c(sigma_pooled, loglik_pooled, sum(df$expenditure == 0), sum(df$expenditure > 0),
                  sigma_e, sigma_u, rho_icc, loglik_re,
                  sum(df$expenditure == 0), sum(df$expenditure > 0),
                  length(unique(df$id))),
  std_error = NA,
  statistic = NA,
  p_value = NA,
  stringsAsFactors = FALSE
)

all_results <- rbind(all_results, model_stats)

# --- Save results to CSV ----------------------------------------------------
output_path <- "/home/guhaase/projetos/panelbox/examples/censored/R/results_tobit_panel.csv"
write.csv(all_results, output_path, row.names = FALSE)
cat("\nResults saved to:", output_path, "\n")

# --- Print comparison table --------------------------------------------------
cat("\n=== Coefficient Comparison: Pooled Tobit vs RE Tobit ===\n")
common_vars <- intersect(
  rownames(pooled_coefs)[!grepl("^log", rownames(pooled_coefs))],
  rownames(re_coefs)[!grepl("^log", rownames(re_coefs))]
)
comparison <- data.frame(
  Variable = common_vars,
  Pooled = round(pooled_coefs[common_vars, "Estimate"], 6),
  RE_Panel = round(re_coefs[common_vars, "Estimate"], 6)
)
print(comparison, row.names = FALSE)

cat("\n=== Variance Components (RE Tobit) ===\n")
cat("sigma_e (idiosyncratic):", round(sigma_e, 6), "\n")
cat("sigma_u (individual):", round(sigma_u, 6), "\n")
cat("rho (ICC):", round(rho_icc, 6), "\n")

cat("\n=== Log-Likelihood ===\n")
cat("Pooled:", round(loglik_pooled, 4), "\n")
cat("RE Panel:", round(loglik_re, 4), "\n")

# LR test: H0: sigma_u = 0 (pooled vs RE)
lr_stat <- 2 * (loglik_re - loglik_pooled)
lr_pvalue <- 0.5 * pchisq(lr_stat, df = 1, lower.tail = FALSE)  # boundary test
cat("\nLR test for RE (sigma_u = 0):\n")
cat("  Statistic:", round(lr_stat, 4), "\n")
cat("  p-value (boundary):", format(lr_pvalue, scientific = TRUE), "\n")

cat("\nDone.\n")
