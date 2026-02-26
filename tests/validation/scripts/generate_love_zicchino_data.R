#!/usr/bin/env Rscript
# Generate synthetic Panel VAR data based on Love & Zicchino (2006) specification
# This creates a dataset similar to their corporate finance panel

library(jsonlite)

set.seed(42)  # For reproducibility

# Parameters (based on Love & Zicchino 2006)
N <- 50   # Number of firms
T <- 15   # Time periods per firm
K <- 4    # Number of variables: sales, inventory, AR (accounts receivable), debt

# True coefficient matrices (VAR(2) specification)
# These are stylized values representing corporate finance dynamics
A1 <- matrix(c(
  0.50, 0.10, 0.05, -0.05,  # sales equation
  0.15, 0.45, 0.08, 0.02,   # inventory equation
  0.10, 0.12, 0.40, 0.03,   # AR equation
  -0.08, 0.05, 0.06, 0.35   # debt equation
), nrow=4, ncol=4, byrow=TRUE)

A2 <- matrix(c(
  0.15, 0.05, 0.02, -0.02,
  0.08, 0.20, 0.04, 0.01,
  0.05, 0.06, 0.18, 0.02,
  -0.04, 0.02, 0.03, 0.15
), nrow=4, ncol=4, byrow=TRUE)

# Residual covariance matrix (Cholesky factor)
# This creates cross-sectionally correlated shocks
Sigma_chol <- matrix(c(
  1.0, 0.0, 0.0, 0.0,
  0.3, 0.9, 0.0, 0.0,
  0.2, 0.2, 0.85, 0.0,
  -0.15, 0.1, 0.15, 0.8
), nrow=4, ncol=4, byrow=TRUE)

Sigma <- Sigma_chol %*% t(Sigma_chol)

# Generate data
data_list <- list()

for (i in 1:N) {
  # Initialize with unconditional mean (approximately zero for simplicity)
  y <- matrix(0, nrow=T+10, ncol=K)  # Extra 10 periods for burn-in

  # Generate shocks
  eps <- matrix(rnorm((T+10)*K), ncol=K) %*% t(Sigma_chol)

  # Generate VAR(2) process
  for (t in 3:(T+10)) {
    y[t, ] <- A1 %*% y[t-1, ] + A2 %*% y[t-2, ] + eps[t, ]
  }

  # Drop burn-in period
  y <- y[11:(T+10), ]

  # Create dataframe for this entity
  df_i <- data.frame(
    firm_id = i,
    year = 1:T,
    sales = y[, 1],
    inventory = y[, 2],
    ar = y[, 3],
    debt = y[, 4]
  )

  data_list[[i]] <- df_i
}

# Combine all entities
data <- do.call(rbind, data_list)

# Save as CSV
write.csv(data, "tests/validation/data/love_zicchino_2006.csv", row.names=FALSE)

# Save metadata
metadata <- list(
  description = "Synthetic panel data based on Love & Zicchino (2006) specification",
  N = N,
  T = T,
  K = K,
  variables = c("sales", "inventory", "ar", "debt"),
  true_params = list(
    A1 = A1,
    A2 = A2,
    Sigma = Sigma
  ),
  dgp_type = "Panel VAR(2)",
  seed = 42
)

write_json(metadata, "tests/validation/data/love_zicchino_2006_metadata.json", auto_unbox=TRUE, pretty=TRUE)

cat("✓ Love & Zicchino (2006) dataset created successfully\n")
cat(sprintf("  N = %d firms, T = %d periods, K = %d variables\n", N, T, K))
cat(sprintf("  Total observations: %d\n", nrow(data)))
