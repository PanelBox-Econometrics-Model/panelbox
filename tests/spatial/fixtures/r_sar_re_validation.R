# tests/spatial/fixtures/r_sar_re_validation.R
# SAR Random Effects validation script

library(splm)
library(spdep)
library(plm)
library(jsonlite)

# Working directory is already set by cd command
cat("Working directory:", getwd(), "\n")

# Load test data
cat("Loading test data...\n")
data <- read.csv("spatial_test_data.csv")
W <- as.matrix(read.csv("spatial_weights.csv", header=FALSE))

cat("Data dimensions:", nrow(data), "x", ncol(data), "\n")
cat("W dimensions:", nrow(W), "x", ncol(W), "\n")

# Create spatial weights list
W_list <- mat2listw(W, style="W")

# Convert to pdata.frame
pdata <- pdata.frame(data, index=c("entity", "time"))

cat("Panel structure:\n")
cat("  Entities:", length(unique(data$entity)), "\n")
cat("  Time periods:", length(unique(data$time)), "\n")

# ==========================================
# SAR Random Effects
# ==========================================
cat("\n========================================\n")
cat("Estimating SAR Random Effects...\n")
cat("========================================\n")

results_sar_re <- tryCatch({
  sar_re <- spml(
    y ~ x1 + x2 + x3,
    data = pdata,
    listw = W_list,
    model = "random",
    lag = TRUE,
    spatial.error = "none",
    effect = "individual"
  )

  # Print summary
  print(summary(sar_re))

  # Extract coefficients
  coefs <- coef(sar_re)

  # Get spatial coefficient (lambda or rho)
  # In splm, the spatial lag coefficient is in arcoef field
  rho_val <- NA
  if(!is.null(sar_re$arcoef)) {
    rho_val <- as.numeric(sar_re$arcoef)
  } else if("lambda" %in% names(coefs)) {
    rho_val <- as.numeric(coefs["lambda"])
  } else if("rho" %in% names(coefs)) {
    rho_val <- as.numeric(coefs["rho"])
  }

  # Get variance components
  # For splm random effects, errcomp contains variance info
  sigma_alpha2_val <- NA
  sigma_epsilon2_val <- NA
  theta_val <- NA

  if(!is.null(sar_re$errcomp)) {
    # Get phi - it's a scalar value
    phi <- as.numeric(sar_re$errcomp["phi"])

    # sigma2 is the overall error variance
    sigma2_total <- NA
    if(!is.null(sar_re$sigma2)) {
      sigma2_total <- as.numeric(sar_re$sigma2)
    }

    # Calculate variance components from phi
    # In splm: phi = sigma_alpha^2 / sigma_eps^2
    # sigma2 = sigma_eps^2 (the within variance)
    T_periods <- length(unique(data$time))

    # From phi and sigma2, calculate components
    if(length(sigma2_total) > 0 && length(phi) > 0) {
      if(!is.na(sigma2_total[1]) && !is.na(phi[1])) {
        # sigma2 is sigma_epsilon^2
        sigma_epsilon2_val <- sigma2_total[1]
        # phi = sigma_alpha^2 / sigma_eps^2
        sigma_alpha2_val <- phi[1] * sigma_epsilon2_val
        # Calculate theta
        theta_val <- 1 - sqrt(sigma_epsilon2_val / (sigma_epsilon2_val + T_periods * sigma_alpha2_val))
      }
    }
  }

  # Get log-likelihood
  loglik_val <- NA
  if(!is.null(sar_re$logLik)) {
    loglik_val <- as.numeric(sar_re$logLik)
  }

  # Calculate AIC and BIC
  n_params <- length(coefs) + 2  # coefficients + 2 variance parameters
  n_obs <- nrow(data)
  aic_val <- -2 * loglik_val + 2 * n_params
  bic_val <- -2 * loglik_val + log(n_obs) * n_params

  # Get convergence
  convergence_val <- NA
  if(!is.null(sar_re$optres)) {
    if(!is.null(sar_re$optres$convergence)) {
      convergence_val <- sar_re$optres$convergence
    }
  }

  cat("\nExtracted SAR RE results:\n")
  cat("  rho/lambda:", as.numeric(rho_val), "\n")
  cat("  beta x1:", as.numeric(coefs["x1"]), "\n")
  cat("  beta x2:", as.numeric(coefs["x2"]), "\n")
  cat("  beta x3:", as.numeric(coefs["x3"]), "\n")
  cat("  sigma_alpha2:", sigma_alpha2_val, "\n")
  cat("  sigma_epsilon2:", sigma_epsilon2_val, "\n")
  cat("  theta:", theta_val, "\n")
  cat("  logLik:", loglik_val, "\n")

  list(
    rho = as.numeric(rho_val),
    beta = list(
      x1 = as.numeric(coefs["x1"]),
      x2 = as.numeric(coefs["x2"]),
      x3 = as.numeric(coefs["x3"])
    ),
    sigma_alpha2 = sigma_alpha2_val,
    sigma_epsilon2 = sigma_epsilon2_val,
    theta = theta_val,
    logLik = loglik_val,
    aic = aic_val,
    bic = bic_val,
    convergence = convergence_val
  )

}, error = function(e) {
  cat("ERROR in SAR RE estimation:", conditionMessage(e), "\n")
  list(error = conditionMessage(e))
})

# ==========================================
# SAR Fixed Effects
# ==========================================
cat("\n========================================\n")
cat("Estimating SAR Fixed Effects...\n")
cat("========================================\n")

results_sar_fe <- tryCatch({
  sar_fe <- spml(
    y ~ x1 + x2 + x3,
    data = pdata,
    listw = W_list,
    model = "within",
    lag = TRUE,
    spatial.error = "none"
  )

  print(summary(sar_fe))

  coefs_fe <- coef(sar_fe)

  # Get spatial coefficient
  rho_fe <- NA
  if(!is.null(sar_fe$arcoef)) {
    rho_fe <- as.numeric(sar_fe$arcoef)
  } else if("lambda" %in% names(coefs_fe)) {
    rho_fe <- as.numeric(coefs_fe["lambda"])
  } else if("rho" %in% names(coefs_fe)) {
    rho_fe <- as.numeric(coefs_fe["rho"])
  }

  # Get log-likelihood
  loglik_fe <- NA
  if(!is.null(sar_fe$logLik)) {
    loglik_fe <- as.numeric(sar_fe$logLik)
  }

  # Calculate AIC and BIC
  n_params_fe <- length(coefs_fe)
  n_obs_fe <- nrow(data)
  aic_fe <- -2 * loglik_fe + 2 * n_params_fe
  bic_fe <- -2 * loglik_fe + log(n_obs_fe) * n_params_fe

  cat("\nExtracted SAR FE results:\n")
  cat("  rho/lambda:", as.numeric(rho_fe), "\n")
  cat("  beta x1:", as.numeric(coefs_fe["x1"]), "\n")
  cat("  beta x2:", as.numeric(coefs_fe["x2"]), "\n")
  cat("  beta x3:", as.numeric(coefs_fe["x3"]), "\n")
  cat("  logLik:", loglik_fe, "\n")

  list(
    rho = as.numeric(rho_fe),
    beta = list(
      x1 = as.numeric(coefs_fe["x1"]),
      x2 = as.numeric(coefs_fe["x2"]),
      x3 = as.numeric(coefs_fe["x3"])
    ),
    logLik = loglik_fe,
    aic = aic_fe,
    bic = bic_fe
  )

}, error = function(e) {
  cat("ERROR in SAR FE estimation:", conditionMessage(e), "\n")
  list(error = conditionMessage(e))
})

# Compile all results
results <- list(
  sar_re = results_sar_re,
  sar_fe = results_sar_fe
)

# Save results
output_file <- "r_sar_re_results.json"
write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE, na = "null")

cat("\n========================================\n")
cat("Results saved to", output_file, "\n")
cat("========================================\n")
