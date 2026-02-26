# Panel Unit Root Test Validation in R
#
# This script validates PanelBox unit root tests against R's plm package
#
# Required packages: plm, haven

library(plm)

# Set seed for reproducibility
set.seed(42)

# ============================================================================
# 1. Generate Stationary Panel Data (AR(1) with |ρ| < 1)
# ============================================================================

generate_stationary_panel <- function(N = 10, T = 100, rho = 0.6) {
  data <- data.frame()

  for (i in 1:N) {
    y <- numeric(T)
    y[1] <- rnorm(1)

    for (t in 2:T) {
      y[t] <- rho * y[t-1] + rnorm(1)
    }

    entity_data <- data.frame(
      entity = i,
      time = 1:T,
      y = y
    )
    data <- rbind(data, entity_data)
  }

  return(data)
}

# ============================================================================
# 2. Generate Unit Root Panel Data (Random Walk)
# ============================================================================

generate_unit_root_panel <- function(N = 10, T = 100) {
  set.seed(123)
  data <- data.frame()

  for (i in 1:N) {
    y <- cumsum(rnorm(T))

    entity_data <- data.frame(
      entity = i,
      time = 1:T,
      y = y
    )
    data <- rbind(data, entity_data)
  }

  return(data)
}

# ============================================================================
# 3. Run Tests
# ============================================================================

cat("=" %R% 80, "\n")
cat("Panel Unit Root Test Validation (R)\n")
cat("=" %R% 80, "\n\n")

# --- Stationary Data ---
cat("TEST 1: Stationary Panel Data (ρ = 0.6)\n")
cat("-" %R% 80, "\n")

df_stat <- generate_stationary_panel(N = 10, T = 100, rho = 0.6)

# Hadri test (H0: stationarity)
cat("\nHadri Test (H0: Stationarity):\n")
tryCatch({
  hadri_result <- purtest(y ~ 1, data = df_stat,
                          index = c("entity", "time"),
                          test = "hadri",
                          exo = "none",
                          lags = "AIC")
  print(hadri_result)
}, error = function(e) {
  cat("Hadri test not available or error:", e$message, "\n")
})

# Breitung test is not directly available in plm
# We use IPS as alternative
cat("\nIm-Pesaran-Shin Test (H0: Unit Root):\n")
ips_result_stat <- purtest(y ~ 1, data = df_stat,
                            index = c("entity", "time"),
                            test = "ips",
                            exo = "none",
                            lags = "AIC")
print(ips_result_stat)

# Levin-Lin-Chu test
cat("\nLevin-Lin-Chu Test (H0: Unit Root):\n")
llc_result_stat <- purtest(y ~ 1, data = df_stat,
                            index = c("entity", "time"),
                            test = "levinlin",
                            exo = "none",
                            lags = "AIC")
print(llc_result_stat)

cat("\n")
cat("=" %R% 80, "\n\n")

# --- Unit Root Data ---
cat("TEST 2: Unit Root Panel Data (Random Walk)\n")
cat("-" %R% 80, "\n")

df_ur <- generate_unit_root_panel(N = 10, T = 100)

# Hadri test (H0: stationarity)
cat("\nHadri Test (H0: Stationarity):\n")
tryCatch({
  hadri_result_ur <- purtest(y ~ 1, data = df_ur,
                              index = c("entity", "time"),
                              test = "hadri",
                              exo = "none",
                              lags = "AIC")
  print(hadri_result_ur)
}, error = function(e) {
  cat("Hadri test not available or error:", e$message, "\n")
})

# IPS test
cat("\nIm-Pesaran-Shin Test (H0: Unit Root):\n")
ips_result_ur <- purtest(y ~ 1, data = df_ur,
                          index = c("entity", "time"),
                          test = "ips",
                          exo = "none",
                          lags = "AIC")
print(ips_result_ur)

# LLC test
cat("\nLevin-Lin-Chu Test (H0: Unit Root):\n")
llc_result_ur <- purtest(y ~ 1, data = df_ur,
                          index = c("entity", "time"),
                          test = "levinlin",
                          exo = "none",
                          lags = "AIC")
print(llc_result_ur)

cat("\n")
cat("=" %R% 80, "\n\n")

# ============================================================================
# 4. Export Data for Python Validation
# ============================================================================

cat("Exporting data for Python validation...\n")

# Save stationary data
write.csv(df_stat, "stationary_panel_data.csv", row.names = FALSE)

# Save unit root data
write.csv(df_ur, "unit_root_panel_data.csv", row.names = FALSE)

cat("Data exported successfully!\n\n")

# ============================================================================
# 5. Summary
# ============================================================================

cat("=" %R% 80, "\n")
cat("VALIDATION SUMMARY\n")
cat("=" %R% 80, "\n\n")

cat("Expected Results:\n")
cat("  Stationary Data:\n")
cat("    - Hadri: Should NOT reject H0 (high p-value)\n")
cat("    - IPS/LLC: Should REJECT H0 (low p-value)\n\n")
cat("  Unit Root Data:\n")
cat("    - Hadri: Should REJECT H0 (low p-value)\n")
cat("    - IPS/LLC: Should NOT reject H0 (high p-value)\n\n")

cat("Compare these results with PanelBox Python implementation.\n")
cat("Test statistics should be within ±0.01 tolerance.\n\n")

cat("=" %R% 80, "\n")
