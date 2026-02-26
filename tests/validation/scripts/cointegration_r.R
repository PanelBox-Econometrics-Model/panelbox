# Panel Cointegration Tests Validation Script
# Compares PanelBox results with R packages (plm, urca, etc.)

library(plm)
library(urca)

# Function to run Pedroni tests manually (plm doesn't have built-in Pedroni)
pedroni_tests <- function(y, x, data, index) {
  # Convert to pdata.frame
  pdata <- pdata.frame(data, index = index)

  # Estimate cointegrating regression for each entity
  entities <- unique(data[[index[1]]])
  N <- length(entities)

  residuals_list <- list()

  for (i in 1:N) {
    entity_data <- subset(data, data[[index[1]]] == entities[i])

    # Cointegrating regression
    model <- lm(as.formula(paste(y, "~", x)), data = entity_data)
    residuals_list[[i]] <- residuals(model)
  }

  # Combine residuals
  all_residuals <- unlist(residuals_list)

  # Panel unit root test on residuals (simplified version)
  cat("Pedroni-style residual-based test:\n")
  cat("(Note: This is a simplified version. Full Pedroni requires specialized packages)\n\n")

  return(list(residuals = all_residuals))
}

# Function to run Kao test
kao_test <- function(y, x, data, index) {
  # Convert to pdata.frame
  pdata <- pdata.frame(data, index = index)

  # Pooled cointegrating regression
  formula_str <- paste(y, "~", x)
  pooled_model <- plm(as.formula(formula_str), data = pdata, model = "pooling")

  residuals_pooled <- residuals(pooled_model)

  # ADF test on pooled residuals
  cat("Kao test (DF/ADF on pooled residuals):\n")

  # Reshape residuals to panel format
  resid_df <- data.frame(
    entity = pdata[[index[1]]],
    time = pdata[[index[2]]],
    resid = residuals_pooled
  )

  # Run panel unit root test
  pdata_resid <- pdata.frame(resid_df, index = c("entity", "time"))

  # Use purtest for panel unit root
  # For residuals from cointegrating regression, use "intercept" exo option
  test_result <- purtest(pdata_resid$resid, test = "ips", exo = "intercept", lags = "AIC", pmax = 4)

  print(test_result)

  return(list(test = test_result, residuals = residuals_pooled))
}

# Generate test data - COINTEGRATED
set.seed(42)
N <- 30
T <- 80

cat("=== GENERATING COINTEGRATED PANEL DATA ===\n")
cat("N =", N, "entities, T =", T, "periods\n\n")

data_list <- list()
for (i in 1:N) {
  u <- rnorm(T)
  x <- cumsum(u)  # I(1) process
  epsilon <- 0.5 * rnorm(T)  # I(0) error
  y <- 1.5 * x + epsilon  # Cointegrated relationship

  data_list[[i]] <- data.frame(
    entity = paste0("Entity_", i),
    time = 1:T,
    y = y,
    x = x
  )
}

data_coint <- do.call(rbind, data_list)

# Convert to pdata.frame
pdata <- pdata.frame(data_coint, index = c("entity", "time"))

cat("=== PANEL COINTEGRATION TESTS IN R ===\n\n")

# 1. Panel unit root tests (prerequisite - check if variables are I(1))
cat("--- 1. PANEL UNIT ROOT TESTS (check if y and x are I(1)) ---\n\n")

cat("Testing y for unit root (should NOT reject H0 - has unit root):\n")
test_y <- purtest(pdata$y, test = "ips", exo = "intercept", lags = "AIC", pmax = 4)
print(test_y)
cat("\n")

cat("Testing x for unit root (should NOT reject H0 - has unit root):\n")
test_x <- purtest(pdata$x, test = "ips", exo = "intercept", lags = "AIC", pmax = 4)
print(test_x)
cat("\n")

# 2. Kao test
cat("--- 2. KAO (1999) TEST ---\n\n")
kao_result <- kao_test("y", "x", data_coint, c("entity", "time"))
cat("\n")

# 3. Pedroni-style test (simplified)
cat("--- 3. PEDRONI-STYLE TEST (simplified) ---\n\n")
pedroni_result <- pedroni_tests("y", "x", data_coint, c("entity", "time"))
cat("\n")

# 4. Save results to CSV for Python validation
cat("--- 4. SAVING RESULTS FOR VALIDATION ---\n\n")

# Extract key statistics
results_df <- data.frame(
  test = c("IPS_y", "IPS_x", "Kao"),
  statistic = c(
    test_y$statistic$statistic,
    test_x$statistic$statistic,
    kao_result$test$statistic$statistic
  ),
  p_value = c(
    test_y$statistic$p.value,
    test_x$statistic$p.value,
    kao_result$test$statistic$p.value
  )
)

output_dir <- "/home/guhaase/projetos/panelbox/tests/validation/outputs"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

write.csv(results_df,
          file.path(output_dir, "cointegration_r_results.csv"),
          row.names = FALSE)

# Save data for reproducibility
write.csv(data_coint,
          file.path(output_dir, "cointegration_test_data.csv"),
          row.names = FALSE)

cat("Results saved to:", file.path(output_dir, "cointegration_r_results.csv"), "\n")
cat("Data saved to:", file.path(output_dir, "cointegration_test_data.csv"), "\n\n")

# Print summary
cat("=== SUMMARY ===\n")
print(results_df)

cat("\n=== VALIDATION SCRIPT COMPLETED ===\n")
cat("Note: Full Westerlund and Pedroni tests require additional R packages.\n")
cat("This script provides baseline validation using plm and urca packages.\n")
