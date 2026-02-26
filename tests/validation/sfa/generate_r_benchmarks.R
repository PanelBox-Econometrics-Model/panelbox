#!/usr/bin/env Rscript
#
# Generate benchmark results from R frontier package
#
# This script estimates panel SFA models using R's frontier package
# and exports results for validation against PanelBox.
#
# Required packages:
#   install.packages("frontier")
#   install.packages("plm")
#
# Usage:
#   Rscript generate_r_benchmarks.R

library(frontier)
library(plm)

cat("=======================================================================\n")
cat("R FRONTIER PACKAGE - PANEL SFA BENCHMARKS\n")
cat("=======================================================================\n\n")

# Simulate panel data
set.seed(42)
N <- 100  # Number of entities
T <- 10   # Number of time periods
n <- N * T

# Create panel structure
entity_id <- rep(1:N, each = T)
time_id <- rep(1:T, times = N)

# Generate data
x1 <- runif(n, 0, 3)
x2 <- runif(n, 0, 3)

# True parameters
beta_0 <- 2.0
beta_1 <- 0.6
beta_2 <- 0.3
sigma_v <- 0.1
sigma_u <- 0.2

# Errors
v <- rnorm(n, 0, sigma_v)
u_i <- abs(rnorm(N, 0, sigma_u))
u <- rep(u_i, each = T)

# Output
y <- beta_0 + beta_1 * x1 + beta_2 * x2 + v - u

# Create data frame
panel_data <- data.frame(
  entity = entity_id,
  time = time_id,
  output = y,
  x1 = x1,
  x2 = x2
)

# Convert to pdata.frame for panel structure
panel_data_plm <- pdata.frame(panel_data, index = c("entity", "time"))

# Save data for Python
write.csv(panel_data, "r_panel_data.csv", row.names = FALSE)
cat("Panel data saved to: r_panel_data.csv\n\n")

# -----------------------------------------------------------------------
# Model 1: Pitt & Lee (1981) - Time-invariant efficiency
# -----------------------------------------------------------------------
cat("=======================================================================\n")
cat("MODEL 1: Pitt & Lee (1981) - timeEffect=FALSE\n")
cat("=======================================================================\n\n")

sfa_pl <- sfa(
  output ~ x1 + x2,
  data = panel_data_plm,
  timeEffect = FALSE
)

# Extract results
cat("\nParameter Estimates:\n")
print(summary(sfa_pl))

# Get efficiencies
eff_pl <- efficiencies(sfa_pl)

# Save results
# Extract variance components from parameterization
sigmaSq <- sfa_pl$mleParam["sigmaSq"]
gamma <- sfa_pl$mleParam["gamma"]
sigma_u_sq <- sigmaSq * gamma
sigma_v_sq <- sigmaSq * (1 - gamma)

pl_results <- list(
  beta = coef(sfa_pl)[1:3],  # Only frontier parameters
  sigma_v = sqrt(sigma_v_sq),
  sigma_u = sqrt(sigma_u_sq),
  lambda = sqrt(sigma_u_sq / sigma_v_sq),
  gamma = gamma,
  sigmaSq = sigmaSq,
  loglik = sfa_pl$mleLogl,
  mean_efficiency = mean(eff_pl),
  efficiencies = eff_pl
)

saveRDS(pl_results, "r_pitt_lee_results.rds")
write.csv(
  data.frame(entity = 1:N, efficiency = eff_pl),
  "r_pitt_lee_efficiencies.csv",
  row.names = FALSE
)

cat("\nResults saved to: r_pitt_lee_results.rds\n")
cat("Efficiencies saved to: r_pitt_lee_efficiencies.csv\n\n")

# -----------------------------------------------------------------------
# Model 2: Battese & Coelli (1992) - Time-varying efficiency
# -----------------------------------------------------------------------
cat("=======================================================================\n")
cat("MODEL 2: Battese & Coelli (1992) - timeEffect=TRUE\n")
cat("=======================================================================\n\n")

sfa_bc92 <- sfa(
  output ~ x1 + x2,
  data = panel_data_plm,
  timeEffect = TRUE
)

cat("\nParameter Estimates:\n")
print(summary(sfa_bc92))

# Get efficiencies
eff_bc92 <- efficiencies(sfa_bc92)

# Save results
# Extract variance components from parameterization
sigmaSq_bc92 <- sfa_bc92$mleParam["sigmaSq"]
gamma_bc92 <- sfa_bc92$mleParam["gamma"]
sigma_u_sq_bc92 <- sigmaSq_bc92 * gamma_bc92
sigma_v_sq_bc92 <- sigmaSq_bc92 * (1 - gamma_bc92)
eta_bc92 <- sfa_bc92$mleParam["time"]

bc92_results <- list(
  beta = coef(sfa_bc92)[1:3],  # Only frontier parameters
  sigma_v = sqrt(sigma_v_sq_bc92),
  sigma_u = sqrt(sigma_u_sq_bc92),
  eta = eta_bc92,
  gamma = gamma_bc92,
  sigmaSq = sigmaSq_bc92,
  loglik = sfa_bc92$mleLogl,
  mean_efficiency = mean(eff_bc92),
  efficiencies = eff_bc92
)

saveRDS(bc92_results, "r_bc92_results.rds")
write.csv(
  data.frame(
    entity = rep(1:N, each = T),
    time = rep(1:T, times = N),
    efficiency = eff_bc92
  ),
  "r_bc92_efficiencies.csv",
  row.names = FALSE
)

cat("\nResults saved to: r_bc92_results.rds\n")
cat("Efficiencies saved to: r_bc92_efficiencies.csv\n\n")

# -----------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------
cat("=======================================================================\n")
cat("COMPARISON OF MODELS\n")
cat("=======================================================================\n\n")

comparison <- data.frame(
  Model = c("Pitt-Lee", "BC92"),
  LogLik = c(pl_results$loglik, bc92_results$loglik),
  sigma_v = c(pl_results$sigma_v, bc92_results$sigma_v),
  sigma_u = c(pl_results$sigma_u, bc92_results$sigma_u),
  lambda = c(pl_results$lambda, bc92_results$lambda),
  Mean_Eff = c(pl_results$mean_efficiency, bc92_results$mean_efficiency),
  eta = c(NA, bc92_results$eta)
)

print(comparison)

write.csv(comparison, "r_model_comparison.csv", row.names = FALSE)
cat("\nComparison saved to: r_model_comparison.csv\n\n")

cat("=======================================================================\n")
cat("BENCHMARK GENERATION COMPLETE\n")
cat("=======================================================================\n\n")

cat("Files created:\n")
cat("  - r_panel_data.csv (panel data)\n")
cat("  - r_pitt_lee_results.rds (Pitt-Lee estimates)\n")
cat("  - r_pitt_lee_efficiencies.csv (Pitt-Lee efficiencies)\n")
cat("  - r_bc92_results.rds (BC92 estimates)\n")
cat("  - r_bc92_efficiencies.csv (BC92 efficiencies)\n")
cat("  - r_model_comparison.csv (model comparison)\n\n")

cat("Use these files to validate PanelBox estimates.\n")
