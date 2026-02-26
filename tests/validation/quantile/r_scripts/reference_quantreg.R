# tests/validation/quantile/r_scripts/reference_quantreg.R
library(quantreg)
library(jsonlite)

# Function to generate reference results
generate_reference <- function(data_file, output_file) {
  # Load data
  data <- read.csv(data_file)

  results <- list()

  # 1. Pooled Quantile Regression
  tau_list <- c(0.1, 0.25, 0.5, 0.75, 0.9)

  for (tau in tau_list) {
    # Basic QR
    qr_model <- rq(y ~ x1 + x2 + x3, data = data, tau = tau)

    # Extract results
    coef <- coef(qr_model)
    se <- summary(qr_model, se = "boot", R = 999)$coefficients[, 2]

    results[[paste0("pooled_tau_", tau)]] <- list(
      coefficients = as.numeric(coef),
      std_errors = as.numeric(se),
      tau = as.numeric(tau),
      method = "pooled"
    )
  }

  # 2. Bootstrap inference
  X_matrix <- cbind(1, data$x1, data$x2, data$x3)
  boot_result <- boot.rq(
    x = X_matrix,
    y = data$y,
    tau = 0.5,
    R = 999,
    method = "mcmb"  # Markov chain marginal bootstrap
  )

  results[["bootstrap"]] <- list(
    coefficients = as.numeric(boot_result$B[1,]),
    std_errors = as.numeric(apply(boot_result$B, 2, sd)),
    ci_lower = as.numeric(apply(boot_result$B, 2, quantile, 0.025)),
    ci_upper = as.numeric(apply(boot_result$B, 2, quantile, 0.975)),
    n_boot = as.numeric(999)
  )

  # 3. Process plots data
  tau_grid <- seq(0.05, 0.95, by = 0.05)
  process_coefs <- matrix(NA, length(tau_grid), 4)

  for (i in seq_along(tau_grid)) {
    qr_tau <- rq(y ~ x1 + x2 + x3, data = data, tau = tau_grid[i])
    process_coefs[i, ] <- coef(qr_tau)
  }

  results[["process"]] <- list(
    tau_grid = as.numeric(tau_grid),
    coefficients = process_coefs
  )

  # 4. Diagnostics
  # Pseudo R-squared
  qr_50 <- rq(y ~ x1 + x2 + x3, data = data, tau = 0.5)
  resid_full <- residuals(qr_50)

  # Null model (intercept only)
  qr_null <- rq(y ~ 1, data = data, tau = 0.5)
  resid_null <- residuals(qr_null)

  # Pseudo R² (Koenker-Machado)
  pseudo_r2 <- 1 - sum(abs(resid_full)) / sum(abs(resid_null))

  results[["diagnostics"]] <- list(
    pseudo_r2 = as.numeric(pseudo_r2),
    n_obs = as.numeric(nrow(data)),
    df_residual = as.numeric(nrow(data) - 4)
  )

  # Save results
  write_json(results, output_file, pretty = TRUE, digits = 10, auto_unbox = TRUE)

  return(results)
}

# Generate test cases
set.seed(42)

# Test Case 1: Simple cross-section
n <- 500
data_simple <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  x3 = rnorm(n),
  epsilon = rt(n, df = 5)  # Heavy-tailed errors
)
data_simple$y <- 1 + 2*data_simple$x1 - 1.5*data_simple$x2 +
                  0.5*data_simple$x3 + data_simple$epsilon

write.csv(data_simple, "test_data_simple.csv", row.names = FALSE)
results_simple <- generate_reference("test_data_simple.csv",
                                    "reference_simple.json")

print("Reference outputs generated successfully!")
