# =============================================================================
# Validation Script 02: Spatial Error Model (SEM) and Spatial Durbin Model (SDM)
# =============================================================================
# Dataset: Columbus (spdep package) - 49 neighborhoods in Columbus, Ohio
# Model: CRIME ~ INC + HOVAL
# Weight matrix: Queen contiguity (row-standardized)
# Estimation: Maximum Likelihood (ML)
# =============================================================================

library(spdep)
library(spatialreg)

# ---- Output directory ----
output_dir <- "/home/guhaase/projetos/panelbox/examples/spatial/R"

# ---- Load Columbus dataset and spatial weights ----
data(columbus, package = "spdep")
col_nb <- col.gal.nb
col_listw <- nb2listw(col_nb, style = "W")

cat("=== Columbus Dataset ===\n")
cat("Observations:", nrow(columbus), "\n\n")

# ---- 1. Spatial Error Model (SEM) via ML ----
cat("=== Spatial Error Model (SEM) - Maximum Likelihood ===\n")
sem_model <- errorsarlm(CRIME ~ INC + HOVAL, data = columbus, listw = col_listw)
sem_sum <- summary(sem_model)
print(sem_sum)
cat("\n")

# Extract SEM parameters
sem_coefs <- coef(sem_model)
sem_se <- sqrt(diag(vcov(sem_model)))
sem_z <- sem_coefs / sem_se
sem_p <- 2 * pnorm(-abs(sem_z))

lambda <- sem_model$lambda
lambda_se <- sem_model$lambda.se
lambda_z <- lambda / lambda_se
lambda_p <- 2 * pnorm(-abs(lambda_z))

sem_loglik <- as.numeric(logLik(sem_model))
sem_aic <- AIC(sem_model)

cat("Lambda:", lambda, "\n")
cat("Lambda SE:", lambda_se, "\n")
cat("Log-Likelihood:", sem_loglik, "\n")
cat("AIC:", sem_aic, "\n\n")

# ---- 2. Spatial Durbin Model (SDM) via ML ----
cat("=== Spatial Durbin Model (SDM) - Maximum Likelihood ===\n")
sdm_model <- lagsarlm(CRIME ~ INC + HOVAL, data = columbus,
                       listw = col_listw, type = "mixed")
sdm_sum <- summary(sdm_model)
print(sdm_sum)
cat("\n")

# Extract SDM parameters
sdm_coefs <- coef(sdm_model)
sdm_se <- sqrt(diag(vcov(sdm_model)))
sdm_z <- sdm_coefs / sdm_se
sdm_p <- 2 * pnorm(-abs(sdm_z))

rho_sdm <- sdm_model$rho
rho_sdm_se <- sdm_model$rho.se
rho_sdm_z <- rho_sdm / rho_sdm_se
rho_sdm_p <- 2 * pnorm(-abs(rho_sdm_z))

sdm_loglik <- as.numeric(logLik(sdm_model))
sdm_aic <- AIC(sdm_model)

cat("Rho (SDM):", rho_sdm, "\n")
cat("Rho SE:", rho_sdm_se, "\n")
cat("Log-Likelihood:", sdm_loglik, "\n")
cat("AIC:", sdm_aic, "\n\n")

# ---- 3. Spatial Durbin Error Model (SDEM) via ML ----
cat("=== Spatial Durbin Error Model (SDEM) - Maximum Likelihood ===\n")
sdem_model <- errorsarlm(CRIME ~ INC + HOVAL, data = columbus,
                          listw = col_listw, etype = "emixed")
sdem_sum <- summary(sdem_model)
print(sdem_sum)
cat("\n")

sdem_coefs <- coef(sdem_model)
sdem_se <- sqrt(diag(vcov(sdem_model)))
sdem_z <- sdem_coefs / sdem_se
sdem_p <- 2 * pnorm(-abs(sdem_z))

lambda_sdem <- sdem_model$lambda
lambda_sdem_se <- sdem_model$lambda.se

sdem_loglik <- as.numeric(logLik(sdem_model))
sdem_aic <- AIC(sdem_model)

# ---- 4. Model comparison ----
cat("=== Model Comparison ===\n")
cat("SEM  Log-Likelihood:", sem_loglik, "  AIC:", sem_aic, "\n")
cat("SDM  Log-Likelihood:", sdm_loglik, "  AIC:", sdm_aic, "\n")
cat("SDEM Log-Likelihood:", sdem_loglik, "  AIC:", sdem_aic, "\n\n")

# LR test: SDM vs SAR (test if theta = 0, i.e., if WX terms are needed)
# First fit SAR for comparison
sar_model <- lagsarlm(CRIME ~ INC + HOVAL, data = columbus, listw = col_listw)
lr_sdm_sar <- as.numeric(2 * (logLik(sdm_model) - logLik(sar_model)))
lr_sdm_sar_p <- pchisq(lr_sdm_sar, df = 2, lower.tail = FALSE)  # 2 WX vars
cat("LR test SDM vs SAR: statistic =", lr_sdm_sar, ", p-value =", lr_sdm_sar_p, "\n")

# LR test: SDEM vs SEM (test if WX terms in error model are needed)
lr_sdem_sem <- as.numeric(2 * (logLik(sdem_model) - logLik(sem_model)))
lr_sdem_sem_p <- pchisq(lr_sdem_sem, df = 2, lower.tail = FALSE)
cat("LR test SDEM vs SEM: statistic =", lr_sdem_sem, ", p-value =", lr_sdem_sem_p, "\n\n")

# ---- 5. Build results data frame ----

# SEM results
sem_vars <- names(sem_coefs)
sem_df <- data.frame(
  model_name = "sem_ml",
  variable = sem_vars,
  coefficient = as.numeric(sem_coefs),
  std_error = as.numeric(sem_se),
  statistic = as.numeric(sem_z),
  p_value = as.numeric(sem_p),
  lambda = lambda,
  rho = NA,
  log_likelihood = sem_loglik,
  aic = sem_aic,
  stringsAsFactors = FALSE
)

# SEM lambda row
lambda_df <- data.frame(
  model_name = "sem_ml",
  variable = "lambda",
  coefficient = lambda,
  std_error = lambda_se,
  statistic = lambda_z,
  p_value = lambda_p,
  lambda = lambda,
  rho = NA,
  log_likelihood = sem_loglik,
  aic = sem_aic,
  stringsAsFactors = FALSE
)

# SDM results
sdm_vars <- names(sdm_coefs)
sdm_df <- data.frame(
  model_name = "sdm_ml",
  variable = sdm_vars,
  coefficient = as.numeric(sdm_coefs),
  std_error = as.numeric(sdm_se),
  statistic = as.numeric(sdm_z),
  p_value = as.numeric(sdm_p),
  lambda = NA,
  rho = rho_sdm,
  log_likelihood = sdm_loglik,
  aic = sdm_aic,
  stringsAsFactors = FALSE
)

# SDM rho row
rho_sdm_df <- data.frame(
  model_name = "sdm_ml",
  variable = "rho",
  coefficient = rho_sdm,
  std_error = rho_sdm_se,
  statistic = rho_sdm_z,
  p_value = rho_sdm_p,
  lambda = NA,
  rho = rho_sdm,
  log_likelihood = sdm_loglik,
  aic = sdm_aic,
  stringsAsFactors = FALSE
)

# SDEM results
sdem_vars <- names(sdem_coefs)
sdem_df <- data.frame(
  model_name = "sdem_ml",
  variable = sdem_vars,
  coefficient = as.numeric(sdem_coefs),
  std_error = as.numeric(sdem_se),
  statistic = as.numeric(sdem_z),
  p_value = as.numeric(sdem_p),
  lambda = lambda_sdem,
  rho = NA,
  log_likelihood = sdem_loglik,
  aic = sdem_aic,
  stringsAsFactors = FALSE
)

# SDEM lambda row
lambda_sdem_df <- data.frame(
  model_name = "sdem_ml",
  variable = "lambda",
  coefficient = lambda_sdem,
  std_error = lambda_sdem_se,
  statistic = lambda_sdem / lambda_sdem_se,
  p_value = 2 * pnorm(-abs(lambda_sdem / lambda_sdem_se)),
  lambda = lambda_sdem,
  rho = NA,
  log_likelihood = sdem_loglik,
  aic = sdem_aic,
  stringsAsFactors = FALSE
)

# LR test results
lr_df <- data.frame(
  model_name = c("lr_test_sdm_vs_sar", "lr_test_sdem_vs_sem"),
  variable = c("LR_SDM_vs_SAR", "LR_SDEM_vs_SEM"),
  coefficient = NA,
  std_error = NA,
  statistic = c(lr_sdm_sar, lr_sdem_sem),
  p_value = c(lr_sdm_sar_p, lr_sdem_sem_p),
  lambda = NA,
  rho = NA,
  log_likelihood = NA,
  aic = NA,
  stringsAsFactors = FALSE
)

# Combine all
results <- rbind(sem_df, lambda_df, sdm_df, rho_sdm_df, sdem_df, lambda_sdem_df, lr_df)
rownames(results) <- NULL

# ---- 6. Save to CSV ----
output_file <- file.path(output_dir, "results_02_sem_sdm.csv")
write.csv(results, output_file, row.names = FALSE)
cat("Results saved to:", output_file, "\n")

# Print summary table
cat("\n=== Summary Results Table ===\n")
print(results)
cat("\nDone.\n")
