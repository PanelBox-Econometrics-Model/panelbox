#!/usr/bin/env Rscript
# Validation script for SEM Fixed Effects model
# Compares PanelBox results with R splm package

# Load required libraries
library(splm)
library(spdep)
library(plm)
library(jsonlite)

# Set seed for reproducibility
set.seed(42)

# Function to generate spatial error panel data
generate_sem_data <- function(N = 50, T = 10, lambda = 0.5) {
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

  # Generate y with spatial error
  y <- numeric(NT)
  for (t in 1:T) {
    idx <- ((t-1)*N + 1):(t*N)
    X_t <- cbind(x1[idx], x2[idx])
    beta <- c(beta1, beta2)

    # Linear prediction with fixed effects
    Xbeta <- X_t %*% beta + alpha[idx]

    # Generate spatially correlated errors
    # u = (I - lambda*W)^{-1} * epsilon
    epsilon <- rnorm(N)
    I_lambdaW <- diag(N) - lambda * W
    u <- solve(I_lambdaW, epsilon)

    y[idx] <- Xbeta + u
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

# Function to estimate SEM-FE and save results
estimate_sem_fe <- function(data, W, output_file) {
  # Convert to pdata.frame
  pdata <- pdata.frame(data, index = c("entity", "time"))

  # Create spatial weights list object
  W_list <- mat2listw(W, style = "W")

  # Estimate SEM with fixed effects
  model <- spml(
    formula = y ~ x1 + x2,
    data = pdata,
    listw = W_list,
    model = "within",
    effect = "individual",
    lag = FALSE,
    spatial.error = "b"  # Baltagi type spatial error
  )

  # Extract results
  results <- list(
    coefficients = coef(model),
    lambda = model$lambda,
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

# Function to estimate SEM-FE with GMM
estimate_sem_fe_gmm <- function(data, W, output_file) {
  # Convert to pdata.frame
  pdata <- pdata.frame(data, index = c("entity", "time"))

  # Create spatial weights list object
  W_list <- mat2listw(W, style = "W")

  # Estimate SEM with GMM
  model <- spgm(
    formula = y ~ x1 + x2,
    data = pdata,
    listw = W_list,
    model = "within",
    moments = "fullweights",
    spatial.error = TRUE,
    lag = FALSE
  )

  # Extract results
  results <- list(
    coefficients = coef(model),
    lambda = model$lambda,
    method = "GMM-splm",
    effects = "fixed",
    nobs = nrow(data),
    n_entities = length(unique(data$entity)),
    n_periods = length(unique(data$time))
  )

  # Save to JSON
  gmm_file <- sub(".json", "_gmm.json", output_file)
  write_json(results, gmm_file, auto_unbox = TRUE, digits = 10)

  return(results)
}

# Main execution
main <- function() {
  cat("SEM Fixed Effects Validation Script\n")
  cat("===================================\n\n")

  # Test cases with different parameters
  test_cases <- list(
    list(N = 25, T = 10, lambda = 0.4, name = "small"),
    list(N = 50, T = 20, lambda = 0.3, name = "medium"),
    list(N = 100, T = 15, lambda = 0.5, name = "large")
  )

  for (case in test_cases) {
    cat(sprintf("Test case: %s (N=%d, T=%d, lambda=%.2f)\n",
                case$name, case$N, case$T, case$lambda))

    # Generate data
    sim <- generate_sem_data(case$N, case$T, case$lambda)

    # Save data for Python testing
    data_file <- sprintf("../data/sem_fe_%s.csv", case$name)
    write.csv(sim$data, data_file, row.names = FALSE)

    # Save W matrix
    W_file <- sprintf("../data/W_sem_%s.csv", case$name)
    write.csv(sim$W, W_file, row.names = FALSE)

    # Estimate model with ML
    output_file <- sprintf("../data/sem_fe_%s_results.json", case$name)
    results_ml <- estimate_sem_fe(sim$data, sim$W, output_file)

    # Print ML summary
    cat("  ML Estimation:\n")
    cat(sprintf("    Lambda: %.6f\n", results_ml$lambda))
    cat(sprintf("    Beta: [%.6f, %.6f]\n",
                results_ml$coefficients[1], results_ml$coefficients[2]))
    cat(sprintf("    Log-likelihood: %.2f\n", results_ml$logLik))
    cat(sprintf("    Sigma2: %.6f\n", results_ml$sigma2))

    # Also estimate with GMM
    results_gmm <- estimate_sem_fe_gmm(sim$data, sim$W, output_file)

    cat("  GMM Estimation:\n")
    cat(sprintf("    Lambda: %.6f\n", results_gmm$lambda))
    cat(sprintf("    Beta: [%.6f, %.6f]\n\n",
                results_gmm$coefficients[1], results_gmm$coefficients[2]))
  }

  cat("Validation data generated successfully!\n")
}

# Run if script is executed directly
if (!interactive()) {
  main()
}
