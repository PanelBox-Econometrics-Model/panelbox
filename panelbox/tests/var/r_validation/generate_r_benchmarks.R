# tests/var/r_validation/generate_r_benchmarks.R

library(panelvar)
library(plm)
library(jsonlite)

# Carregar dados (mesmo DGP do Python)
set.seed(42)

# Função para gerar dados VAR(1)
generate_var_data <- function(n_entities = 50, n_periods = 20) {
  # Matriz de coeficientes (transposta para corresponder ao Python)
  # No Python: y[t] = A1 @ y[t-1] + epsilon
  # Onde A1[i,j] é o efeito de y_j(t-1) em y_i(t)
  A1 <- matrix(c(0.5, 0.3, 0.1,
                 0.2, 0.4, 0.1,
                 0.0, 0.0, 0.6), nrow = 3, byrow = TRUE)

  # Matriz de covariância dos erros
  Sigma <- matrix(c(1.0, 0.3, 0.1,
                    0.3, 1.0, 0.2,
                    0.1, 0.2, 1.0), nrow = 3)

  data_list <- list()
  for (i in 1:n_entities) {
    y <- matrix(0, nrow = n_periods, ncol = 3)
    y[1, ] <- MASS::mvrnorm(1, rep(0, 3), Sigma)

    for (t in 2:n_periods) {
      eps <- MASS::mvrnorm(1, rep(0, 3), Sigma)
      y[t, ] <- A1 %*% y[t-1, ] + eps
    }

    df <- data.frame(
      entity = i - 1,  # Start from 0 to match Python
      time = 0:(n_periods-1),  # Start from 0 to match Python
      y1 = y[, 1],
      y2 = y[, 2],
      y3 = y[, 3]
    )
    data_list[[i]] <- df
  }

  do.call(rbind, data_list)
}

# Gerar dados
panel_data <- generate_var_data()

# Converter para pdata.frame
pdata <- pdata.frame(panel_data, index = c("entity", "time"))

cat("Generated panel data with dimensions:", nrow(panel_data), "x", ncol(panel_data), "\n")
cat("First few rows:\n")
print(head(panel_data))

# ============================================
# 1. Panel VAR GMM (panelvar)
# ============================================
cat("\n=== Estimating Panel VAR using GMM ===\n")

tryCatch({
  pvar_gmm <- pvargmm(
    dependent_vars = c("y1", "y2", "y3"),
    lags = 1,
    transformation = "fod",
    data = panel_data,
    panel_identifier = c("entity", "time"),
    steps = c("twostep"),
    system_instruments = TRUE,
    max_instr_dependent_vars = 99,
    max_instr_predet_vars = 99,
    collapse = FALSE
  )

  cat("GMM estimation completed successfully\n")

  # Extrair coeficientes
  coef_gmm <- coef(pvar_gmm)
  se_gmm <- sqrt(diag(vcov(pvar_gmm)))

  cat("Coefficients:\n")
  print(coef_gmm)

  # ============================================
  # 2. IRFs
  # ============================================
  cat("\n=== Computing Impulse Response Functions ===\n")

  irf_result <- oirf(pvar_gmm, n.ahead = 10)
  cat("Orthogonalized IRF computed\n")

  girf_result <- girf(pvar_gmm, n.ahead = 10, ma_approx_steps = 10)
  cat("Generalized IRF computed\n")

  # Bootstrap CIs (commented out for speed in initial testing)
  # irf_boot <- oirf(pvar_gmm, n.ahead = 10, ci = 0.95, nboot = 100)

  # ============================================
  # 3. FEVD
  # ============================================
  cat("\n=== Computing Forecast Error Variance Decomposition ===\n")

  fevd_result <- fevd_orthogonal(pvar_gmm, n.ahead = 10)
  cat("FEVD computed\n")

  # ============================================
  # 4. Exportar para JSON
  # ============================================
  cat("\n=== Exporting results to JSON ===\n")

  results <- list(
    gmm = list(
      coefficients = as.list(coef_gmm),
      standard_errors = as.list(se_gmm),
      vcov = as.matrix(vcov(pvar_gmm))
    ),
    irf = list(
      orthogonalized = lapply(irf_result, function(x) as.matrix(x)),
      generalized = lapply(girf_result, function(x) as.matrix(x))
    ),
    fevd = lapply(fevd_result, function(x) as.matrix(x)),
    metadata = list(
      n_entities = 50,
      n_periods = 20,
      lags = 1,
      package_version = as.character(packageVersion("panelvar")),
      transformation = "fod",
      steps = "twostep"
    )
  )

  # Salvar
  output_file <- "r_benchmark_results.json"
  write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE, digits = 10)

  cat("Benchmark results saved to", output_file, "\n")
  cat("\n=== VALIDATION COMPLETE ===\n")

}, error = function(e) {
  cat("\n!!! ERROR during estimation !!!\n")
  cat("Error message:", e$message, "\n")
  cat("\nThis may be due to:\n")
  cat("1. Missing R packages - install with: install.packages(c('panelvar', 'plm', 'jsonlite', 'MASS'))\n")
  cat("2. Data structure issues\n")
  cat("3. Numerical issues in GMM estimation\n")
  stop(e)
})
