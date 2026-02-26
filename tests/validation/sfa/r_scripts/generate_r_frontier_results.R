# Script to generate reference results from R frontier package
# This generates benchmark results for validating PanelBox SFA implementation

library(frontier)
library(readr)

# Session info for reproducibility
sink("r_session_info.txt")
sessionInfo()
sink()

cat("Generating R frontier reference results...\n")

# ============================================================================
# 1. Rice Production Data (riceProdPhil)
# ============================================================================

data(riceProdPhil)

# Save raw data for Python
write_csv(riceProdPhil, "data/riceProdPhil.csv")

cat("Dataset riceProdPhil:\n")
cat("  Observations:", nrow(riceProdPhil), "\n")
cat("  Variables:", names(riceProdPhil), "\n\n")

# --- Cross-section SFA: Half-Normal ---
cat("Estimating cross-section SFA with half-normal distribution...\n")

sfa_hn <- sfa(
  log(PROD) ~ log(AREA) + log(LABOR) + log(NPK) + log(OTHER),
  data = riceProdPhil,
  ineffDecrease = TRUE,
  truncNorm = FALSE,
  timeEffect = FALSE
)

# Extract results
results_hn <- data.frame(
  parameter = names(coef(sfa_hn)),
  estimate = as.numeric(coef(sfa_hn)),
  se = sqrt(diag(vcov(sfa_hn))),
  tvalue = as.numeric(coef(sfa_hn)) / sqrt(diag(vcov(sfa_hn)))
)

# Variance components
sigma_v <- sqrt(sfa_hn$sigmaSqV)
sigma_u <- sqrt(sfa_hn$sigmaSqU)
sigma_sq <- sfa_hn$sigmaSqV + sfa_hn$sigmaSqU
gamma <- sfa_hn$sigmaSqU / sigma_sq
lambda <- sqrt(sfa_hn$sigmaSqU / sfa_hn$sigmaSqV)

variance_components <- data.frame(
  parameter = c('sigma_v', 'sigma_u', 'sigma_sq', 'gamma', 'lambda'),
  estimate = c(sigma_v, sigma_u, sigma_sq, gamma, lambda),
  se = NA,
  tvalue = NA
)

results_hn <- rbind(results_hn, variance_components)

# Efficiencies
eff_hn <- efficiencies(sfa_hn)
eff_results_hn <- data.frame(
  firm_id = 1:length(eff_hn),
  efficiency = as.numeric(eff_hn)
)

# Log-likelihood
loglik_hn <- data.frame(
  loglik = as.numeric(logLik(sfa_hn)),
  aic = AIC(sfa_hn),
  bic = BIC(sfa_hn)
)

# Save
write_csv(results_hn, "data/r_frontier_halfnormal_params.csv")
write_csv(eff_results_hn, "data/r_frontier_halfnormal_efficiencies.csv")
write_csv(loglik_hn, "data/r_frontier_halfnormal_loglik.csv")

cat("  Log-likelihood:", loglik_hn$loglik, "\n")
cat("  gamma:", gamma, "\n")
cat("  Mean efficiency:", mean(eff_hn), "\n\n")

# --- Cross-section SFA: Truncated Normal ---
cat("Estimating cross-section SFA with truncated normal distribution...\n")

sfa_tn <- sfa(
  log(PROD) ~ log(AREA) + log(LABOR) + log(NPK) + log(OTHER),
  data = riceProdPhil,
  ineffDecrease = TRUE,
  truncNorm = TRUE,
  timeEffect = FALSE
)

results_tn <- data.frame(
  parameter = names(coef(sfa_tn)),
  estimate = as.numeric(coef(sfa_tn)),
  se = sqrt(diag(vcov(sfa_tn))),
  tvalue = as.numeric(coef(sfa_tn)) / sqrt(diag(vcov(sfa_tn)))
)

# Variance components
sigma_v_tn <- sqrt(sfa_tn$sigmaSqV)
sigma_u_tn <- sqrt(sfa_tn$sigmaSqU)
sigma_sq_tn <- sfa_tn$sigmaSqV + sfa_tn$sigmaSqU
gamma_tn <- sfa_tn$sigmaSqU / sigma_sq_tn
lambda_tn <- sqrt(sfa_tn$sigmaSqU / sfa_tn$sigmaSqV)

variance_components_tn <- data.frame(
  parameter = c('sigma_v', 'sigma_u', 'sigma_sq', 'gamma', 'lambda'),
  estimate = c(sigma_v_tn, sigma_u_tn, sigma_sq_tn, gamma_tn, lambda_tn),
  se = NA,
  tvalue = NA
)

results_tn <- rbind(results_tn, variance_components_tn)

eff_tn <- efficiencies(sfa_tn)
eff_results_tn <- data.frame(
  firm_id = 1:length(eff_tn),
  efficiency = as.numeric(eff_tn)
)

loglik_tn <- data.frame(
  loglik = as.numeric(logLik(sfa_tn)),
  aic = AIC(sfa_tn),
  bic = BIC(sfa_tn)
)

write_csv(results_tn, "data/r_frontier_truncnormal_params.csv")
write_csv(eff_results_tn, "data/r_frontier_truncnormal_efficiencies.csv")
write_csv(loglik_tn, "data/r_frontier_truncnormal_loglik.csv")

cat("  Log-likelihood:", loglik_tn$loglik, "\n")
cat("  mu:", coef(sfa_tn)["mu"], "\n")
cat("  Mean efficiency:", mean(eff_tn), "\n\n")

# --- Cost Frontier: Half-Normal ---
cat("Estimating cost frontier with half-normal distribution...\n")

sfa_cost <- sfa(
  log(PROD) ~ log(AREA) + log(LABOR) + log(NPK) + log(OTHER),
  data = riceProdPhil,
  ineffDecrease = FALSE,  # Cost frontier
  truncNorm = FALSE,
  timeEffect = FALSE
)

results_cost <- data.frame(
  parameter = names(coef(sfa_cost)),
  estimate = as.numeric(coef(sfa_cost)),
  se = sqrt(diag(vcov(sfa_cost))),
  tvalue = as.numeric(coef(sfa_cost)) / sqrt(diag(vcov(sfa_cost)))
)

sigma_v_cost <- sqrt(sfa_cost$sigmaSqV)
sigma_u_cost <- sqrt(sfa_cost$sigmaSqU)
sigma_sq_cost <- sfa_cost$sigmaSqV + sfa_cost$sigmaSqU
gamma_cost <- sfa_cost$sigmaSqU / sigma_sq_cost
lambda_cost <- sqrt(sfa_cost$sigmaSqU / sfa_cost$sigmaSqV)

variance_components_cost <- data.frame(
  parameter = c('sigma_v', 'sigma_u', 'sigma_sq', 'gamma', 'lambda'),
  estimate = c(sigma_v_cost, sigma_u_cost, sigma_sq_cost, gamma_cost, lambda_cost),
  se = NA,
  tvalue = NA
)

results_cost <- rbind(results_cost, variance_components_cost)

eff_cost <- efficiencies(sfa_cost)
eff_results_cost <- data.frame(
  firm_id = 1:length(eff_cost),
  efficiency = as.numeric(eff_cost)
)

loglik_cost <- data.frame(
  loglik = as.numeric(logLik(sfa_cost)),
  aic = AIC(sfa_cost),
  bic = BIC(sfa_cost)
)

write_csv(results_cost, "data/r_frontier_cost_params.csv")
write_csv(eff_results_cost, "data/r_frontier_cost_efficiencies.csv")
write_csv(loglik_cost, "data/r_frontier_cost_loglik.csv")

cat("  Log-likelihood:", loglik_cost$loglik, "\n")
cat("  Mean efficiency:", mean(eff_cost), "\n")
cat("  Note: Cost efficiency >= 1 (invert for TE-like metric)\n\n")

cat("R frontier reference results generated successfully!\n")
cat("Files saved in data/ directory\n")
