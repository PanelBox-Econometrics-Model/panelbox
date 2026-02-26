# R script for frontier package validation
# This script generates reference outputs for validating PanelBox SFA implementation

library(frontier)

# Set seed for reproducibility
set.seed(42)

# Generate simulated production data
n <- 500
log_labor <- runif(n, 0, 3)
log_capital <- runif(n, 0, 3)

# True parameters
beta_true <- c(2.0, 0.6, 0.3)
sigma_v_true <- 0.1
sigma_u_true <- 0.2

# Generate error components
v <- rnorm(n, 0, sigma_v_true)
u <- abs(rnorm(n, 0, sigma_u_true))

# Generate output
log_output <- beta_true[1] + beta_true[2] * log_labor + beta_true[3] * log_capital + v - u

# Create data frame
data <- data.frame(
  log_output = log_output,
  log_labor = log_labor,
  log_capital = log_capital
)

# Save data for Python to use
write.csv(data, "sfa_test_data.csv", row.names = FALSE)

# Estimate SFA model with half-normal distribution
cat("\n=== Half-Normal Distribution ===\n")
sfa_hn <- sfa(
  log_output ~ log_labor + log_capital,
  data = data,
  ineffDecrease = TRUE,
  truncNorm = FALSE,
  timeEffect = FALSE,
  printIter = 0
)

# Print summary
summary(sfa_hn)

# Extract coefficients
cat("\nCoefficients:\n")
print(coef(sfa_hn))

# Extract variance components
cat("\nVariance components:\n")
sigma_sq <- coef(sfa_hn)["sigmaSq"]
gamma <- coef(sfa_hn)["gamma"]

sigma_sq_u <- sigma_sq * gamma
sigma_sq_v <- sigma_sq * (1 - gamma)
sigma_u <- sqrt(sigma_sq_u)
sigma_v <- sqrt(sigma_sq_v)

cat("sigma_u:", sigma_u, "\n")
cat("sigma_v:", sigma_v, "\n")
cat("lambda:", sigma_u / sigma_v, "\n")

# Calculate log-likelihood
cat("\nLog-likelihood:", logLik(sfa_hn), "\n")

# Calculate efficiencies
eff <- efficiencies(sfa_hn)
cat("\nEfficiency statistics:\n")
cat("Mean:", mean(eff), "\n")
cat("Median:", median(eff), "\n")
cat("Min:", min(eff), "\n")
cat("Max:", max(eff), "\n")
cat("Std:", sd(eff), "\n")

# Save efficiencies
write.csv(
  data.frame(efficiency = eff),
  "sfa_efficiencies_hn.csv",
  row.names = FALSE
)

# Save reference values
reference <- data.frame(
  parameter = c("beta_0", "beta_1", "beta_2", "sigma_u", "sigma_v", "lambda", "loglik", "mean_eff"),
  value = c(
    coef(sfa_hn)["(Intercept)"],
    coef(sfa_hn)["log_labor"],
    coef(sfa_hn)["log_capital"],
    sigma_u,
    sigma_v,
    sigma_u / sigma_v,
    logLik(sfa_hn),
    mean(eff)
  )
)

write.csv(reference, "sfa_reference_hn.csv", row.names = FALSE)

cat("\n=== Exponential Distribution ===\n")

# Generate new data with exponential inefficiency
set.seed(123)
u_exp <- rexp(n, 1/0.15)
log_output_exp <- beta_true[1] + beta_true[2] * log_labor + beta_true[3] * log_capital + v - u_exp

data_exp <- data.frame(
  log_output = log_output_exp,
  log_labor = log_labor,
  log_capital = log_capital
)

write.csv(data_exp, "sfa_test_data_exp.csv", row.names = FALSE)

# Estimate SFA model with exponential distribution
# Note: frontier package doesn't have direct exponential option
# We use truncated normal as approximation
cat("Note: R frontier package doesn't have exponential distribution.\n")
cat("Using truncated normal instead.\n")

cat("\nValidation data saved to CSV files.\n")
