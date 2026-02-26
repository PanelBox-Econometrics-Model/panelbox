# tests/validation/quantile/r_scripts/reference_location_scale.R
library(jsonlite)

# Manual implementation of location-scale model
implement_location_scale <- function(data_file, output_file) {
  # Load data
  data <- read.csv(data_file)

  # Step 1: Estimate location (OLS)
  lm_location <- lm(y ~ x1 + x2 + x3, data = data)
  alpha <- coef(lm_location)
  resid <- residuals(lm_location)

  # Step 2: Estimate scale (log squared residuals)
  log_resid2 <- log(pmax(resid^2, 1e-10))
  lm_scale <- lm(log_resid2 ~ x1 + x2 + x3, data = data)
  gamma <- coef(lm_scale)

  # Step 3: Compute quantile coefficients
  tau_list <- c(0.1, 0.25, 0.5, 0.75, 0.9)
  results <- list()

  for (tau in tau_list) {
    # Normal quantile function
    q_tau <- qnorm(tau)

    # β(τ) = α + σ * q(τ)
    # σ = exp(γ/2)
    beta_tau <- alpha + exp(gamma/2) * q_tau

    results[[paste0("tau_", tau)]] <- list(
      coefficients = as.numeric(beta_tau),
      tau = tau,
      location = as.numeric(alpha),
      scale = as.numeric(exp(gamma/2))
    )
  }

  # Save results
  write_json(results, output_file, pretty = TRUE, digits = 10, auto_unbox = TRUE)

  return(results)
}

# Generate test data
set.seed(42)
n <- 500
data_simple <- data.frame(
  x1 = rnorm(n),
  x2 = rnorm(n),
  x3 = rnorm(n),
  epsilon = rt(n, df = 5)
)
data_simple$y <- 1 + 2*data_simple$x1 - 1.5*data_simple$x2 +
                  0.5*data_simple$x3 + data_simple$epsilon

write.csv(data_simple, "test_data_location_scale.csv", row.names = FALSE)
ls_results <- implement_location_scale("test_data_location_scale.csv",
                                       "reference_location_scale.json")

print("Location-scale reference outputs generated successfully!")
