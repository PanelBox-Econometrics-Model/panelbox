#!/usr/bin/env Rscript
# Validation script for SAR Fixed Effects model
# Compares PanelBox results with R splm package

# Load required libraries
library(splm)
library(spdep)
library(plm)
library(jsonlite)

# Set seed for reproducibility
set.seed(42)

# Function to generate spatial panel data
generate_spatial_data <- function(N = 50, T = 10, rho = 0.5) {
  # Create simple contiguity matrix (chain structure)
  W <- matrix(0, N, N)
  for (i in 1:N) {
    if (i > 1) W[i, i-1] <- 1
    if (i < N) W[i, i+1] <- 1
  }

  # Row-normalize
  W <- W / rowSums(W)
  W[is.nan(W)] <- 0

  # Generate panel data
  NT <- N * T
  entity <- rep(1:N, T)
  time <- rep(1:T, each = N)

  # Exogenous variables
  x1 <- rnorm(NT)
  x2 <- rnorm(NT)

  # Fixed effects
  alpha <- rnorm(N)[entity]

  # True parameters
  beta1 <- 1.0
  beta2 <- -0.5

  # Generate y with spatial lag
  y <- numeric(NT)
  for (t in 1:T) {
    idx <- ((t-1)*N + 1):(t*N)
    X_t <- cbind(x1[idx], x2[idx])
    beta <- c(beta1, beta2)

    # Linear prediction with fixed effects
    Xbeta <- X_t %*% beta + alpha[idx]
    epsilon <- rnorm(N)

    # Apply spatial lag: y = (I - rho*W)^{-1} (Xbeta + epsilon)
    I_rhoW <- diag(N) - rho * W
    y_t <- solve(I_rhoW, Xbeta + epsilon)
    y[idx] <- y_t
  }

  # Create data frame
  data <- data.frame(
    entity = entity,
    time = time,
    y = y,
    x1 = x1,
    x2 = x2
  )

  return(list(data = data, W = W))
}

# Function to estimate SAR-FE and save results
estimate_sar_fe <- function(data, W, output_file) {
  # Convert to pdata.frame
  pdata <- pdata.frame(data, index = c("entity", "time"))

  # Create spatial weights list object
  W_list <- mat2listw(W, style = "W")

  # Estimate SAR with fixed effects using ML
  model <- spml(
    formula = y ~ x1 + x2,
    data = pdata,
    listw = W_list,
    model = "within",
    effect = "individual",
    lag = TRUE,
    spatial.error = "none"
  )

  # Extract results
  results <- list(
    coefficients = coef(model),
    rho = model$rho,
    sigma2 = model$sigma2,
    logLik = logLik(model),
    vcov = vcov(model),
    method = "ML-splm",
    effects = "fixed",
    nobs = nrow(data),
    n_entities = length(unique(data$entity)),
    n_periods = length(unique(data$time))
  )

  # Save to JSON
  write_json(results, output_file, auto_unbox = TRUE, digits = 10)

  return(results)
}

# Main execution
main <- function() {
  cat("SAR Fixed Effects Validation Script\n")
  cat("===================================\n\n")

  # Test cases with different parameters
  test_cases <- list(
    list(N = 25, T = 10, rho = 0.3, name = "small"),
    list(N = 50, T = 20, rho = 0.5, name = "medium"),
    list(N = 100, T = 15, rho = 0.2, name = "large")
  )

  for (case in test_cases) {
    cat(sprintf("Test case: %s (N=%d, T=%d, rho=%.2f)\n",
                case$name, case$N, case$T, case$rho))

    # Generate data
    sim <- generate_spatial_data(case$N, case$T, case$rho)

    # Save data for Python testing
    data_file <- sprintf("../data/sar_fe_%s.csv", case$name)
    write.csv(sim$data, data_file, row.names = FALSE)

    # Save W matrix
    W_file <- sprintf("../data/W_%s.csv", case$name)
    write.csv(sim$W, W_file, row.names = FALSE)

    # Estimate model
    output_file <- sprintf("../data/sar_fe_%s_results.json", case$name)
    results <- estimate_sar_fe(sim$data, sim$W, output_file)

    # Print summary
    cat(sprintf("  Rho: %.6f\n", results$rho))
    cat(sprintf("  Beta: [%.6f, %.6f]\n",
                results$coefficients[1], results$coefficients[2]))
    cat(sprintf("  Log-likelihood: %.2f\n", results$logLik))
    cat(sprintf("  Sigma2: %.6f\n\n", results$sigma2))
  }

  cat("Validation data generated successfully!\n")
}

# Run if script is executed directly
if (!interactive()) {
  main()
}
