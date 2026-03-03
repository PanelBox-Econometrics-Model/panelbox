# =============================================================================
# Validation Script 01: Moran's I and Spatial Autoregressive (SAR) Model
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
col_nb <- col.gal.nb  # queen contiguity neighbors (built-in)
col_listw <- nb2listw(col_nb, style = "W")  # row-standardized weights

cat("=== Columbus Dataset ===\n")
cat("Observations:", nrow(columbus), "\n")
cat("Variables:", names(columbus), "\n\n")

# ---- 1. Moran's I Test ----
cat("=== Moran's I Test for Spatial Autocorrelation ===\n")
cat("Variable: CRIME\n\n")

moran_result <- moran.test(columbus$CRIME, col_listw)
print(moran_result)
cat("\n")

# Extract Moran's I statistics
moran_i <- as.numeric(moran_result$estimate["Moran I statistic"])
expected_i <- as.numeric(moran_result$estimate["Expectation"])
variance_i <- as.numeric(moran_result$estimate["Variance"])
z_score <- as.numeric(moran_result$statistic)
p_value_moran <- as.numeric(moran_result$p.value)

cat("Moran I statistic:", moran_i, "\n")
cat("Expected I:", expected_i, "\n")
cat("Variance:", variance_i, "\n")
cat("Z-score:", z_score, "\n")
cat("P-value:", p_value_moran, "\n\n")

# ---- 2. OLS Baseline (for comparison) ----
cat("=== OLS Baseline Model ===\n")
ols_model <- lm(CRIME ~ INC + HOVAL, data = columbus)
ols_sum <- summary(ols_model)
print(ols_sum)
cat("\n")

# LM tests for spatial dependence on OLS residuals
cat("=== LM Tests for Spatial Dependence (on OLS residuals) ===\n")
lm_tests <- lm.RStests(ols_model, col_listw,
                        test = c("RSerr", "RSlag", "adjRSerr", "adjRSlag"))
print(lm_tests)
cat("\n")

# ---- 3. SAR Model (Spatial Lag) via ML ----
cat("=== SAR (Spatial Lag) Model - Maximum Likelihood ===\n")
sar_model <- lagsarlm(CRIME ~ INC + HOVAL, data = columbus, listw = col_listw)
sar_sum <- summary(sar_model)
print(sar_sum)
cat("\n")

# Extract SAR parameters
sar_coefs <- coef(sar_model)
sar_se <- sqrt(diag(vcov(sar_model)))
sar_z <- sar_coefs / sar_se
sar_p <- 2 * pnorm(-abs(sar_z))

rho <- sar_model$rho
rho_se <- sar_model$rho.se
rho_z <- rho / rho_se
rho_p <- 2 * pnorm(-abs(rho_z))

sar_loglik <- as.numeric(logLik(sar_model))
sar_aic <- AIC(sar_model)

cat("Rho:", rho, "\n")
cat("Rho SE:", rho_se, "\n")
cat("Log-Likelihood:", sar_loglik, "\n")
cat("AIC:", sar_aic, "\n\n")

# ---- 4. Build results data frame ----

# Moran's I results
moran_df <- data.frame(
  model_name = "moran_i_test",
  variable = "CRIME",
  coefficient = moran_i,
  std_error = sqrt(variance_i),
  statistic = z_score,
  p_value = p_value_moran,
  expected_i = expected_i,
  variance_i = variance_i,
  rho = NA,
  log_likelihood = NA,
  aic = NA,
  stringsAsFactors = FALSE
)

# OLS results
ols_coef_table <- as.data.frame(ols_sum$coefficients)
ols_df <- data.frame(
  model_name = "ols_baseline",
  variable = rownames(ols_coef_table),
  coefficient = ols_coef_table[, 1],
  std_error = ols_coef_table[, 2],
  statistic = ols_coef_table[, 3],
  p_value = ols_coef_table[, 4],
  expected_i = NA,
  variance_i = NA,
  rho = NA,
  log_likelihood = as.numeric(logLik(ols_model)),
  aic = AIC(ols_model),
  stringsAsFactors = FALSE
)

# SAR results - coefficients
sar_vars <- names(sar_coefs)
sar_coef_df <- data.frame(
  model_name = "sar_ml",
  variable = sar_vars,
  coefficient = as.numeric(sar_coefs),
  std_error = as.numeric(sar_se),
  statistic = as.numeric(sar_z),
  p_value = as.numeric(sar_p),
  expected_i = NA,
  variance_i = NA,
  rho = rho,
  log_likelihood = sar_loglik,
  aic = sar_aic,
  stringsAsFactors = FALSE
)

# SAR rho parameter row
rho_df <- data.frame(
  model_name = "sar_ml",
  variable = "rho",
  coefficient = rho,
  std_error = rho_se,
  statistic = rho_z,
  p_value = rho_p,
  expected_i = NA,
  variance_i = NA,
  rho = rho,
  log_likelihood = sar_loglik,
  aic = sar_aic,
  stringsAsFactors = FALSE
)

# LM test results
lm_err <- lm_tests$RSerr
lm_lag <- lm_tests$RSlag
rlm_err <- lm_tests$adjRSerr
rlm_lag <- lm_tests$adjRSlag

lm_df <- data.frame(
  model_name = c("lm_test_error", "lm_test_lag", "rlm_test_error", "rlm_test_lag"),
  variable = c("RS_error", "RS_lag", "adjRS_error", "adjRS_lag"),
  coefficient = NA,
  std_error = NA,
  statistic = c(as.numeric(lm_err$statistic), as.numeric(lm_lag$statistic),
                as.numeric(rlm_err$statistic), as.numeric(rlm_lag$statistic)),
  p_value = c(as.numeric(lm_err$p.value), as.numeric(lm_lag$p.value),
              as.numeric(rlm_err$p.value), as.numeric(rlm_lag$p.value)),
  expected_i = NA,
  variance_i = NA,
  rho = NA,
  log_likelihood = NA,
  aic = NA,
  stringsAsFactors = FALSE
)

# Combine all results
results <- rbind(moran_df, ols_df, sar_coef_df, rho_df, lm_df)
rownames(results) <- NULL

# ---- 5. Save to CSV ----
output_file <- file.path(output_dir, "results_01_moran_sar.csv")
write.csv(results, output_file, row.names = FALSE)
cat("Results saved to:", output_file, "\n")

# Print summary table
cat("\n=== Summary Results Table ===\n")
print(results)
cat("\nDone.\n")
