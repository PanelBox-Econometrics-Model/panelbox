#!/usr/bin/env Rscript

# Generate reference outputs using R for Panel VAR validation
# Uses plm and vars packages (standard approach when pvar is not available)

library(jsonlite)
library(plm)
library(vars)

# Set working directory - assume we're running from scripts directory
# (or set manually if needed)
if (!interactive()) {
  script_path <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("--file=", "", script_path[grep("--file=", script_path)])
  if (length(script_path) > 0) {
    setwd(dirname(script_path))
  }
}

# Helper function to estimate Panel VAR using plm approach
estimate_panel_var <- function(data, entity_col, time_col, y_cols, lags = 2) {

  # Create panel data structure
  pdata <- pdata.frame(data, index = c(entity_col, time_col))

  # First-difference transformation (similar to FD in GMM)
  pdata_fd <- data.frame(pdata)

  results <- list()

  # Estimate each equation separately using plm
  for (y_var in y_cols) {
    # Build formula with lags
    lag_terms <- c()
    for (var in y_cols) {
      for (l in 1:lags) {
        lag_terms <- c(lag_terms, sprintf("lag(%s, %d)", var, l))
      }
    }

    formula_str <- sprintf("%s ~ %s - 1", y_var, paste(lag_terms, collapse = " + "))
    formula_obj <- as.formula(formula_str)

    # Estimate with fixed effects
    model <- plm(formula_obj, data = pdata, model = "within", effect = "individual")

    results[[y_var]] <- list(
      coefficients = coef(model),
      std_errors = sqrt(diag(vcov(model))),
      residuals = residuals(model),
      fitted = fitted(model)
    )
  }

  return(results)
}

# Helper function to compute Granger causality tests
compute_granger_tests <- function(results, y_cols, lags) {
  granger_matrix <- matrix(NA, nrow = length(y_cols), ncol = length(y_cols))
  rownames(granger_matrix) <- y_cols
  colnames(granger_matrix) <- y_cols

  # This is a placeholder - proper implementation would require
  # Wald tests on coefficient restrictions
  # For now, we'll return NA to indicate this needs R pvar package

  return(granger_matrix)
}

# Process simple_pvar.csv dataset
cat("Processing simple_pvar.csv...\n")
data_simple <- read.csv("../data/simple_pvar.csv")

simple_results <- estimate_panel_var(
  data_simple,
  entity_col = "entity",
  time_col = "time",
  y_cols = c("y1", "y2", "y3"),
  lags = 2
)

# Extract coefficients for export
simple_output <- list(
  dataset = "simple_pvar",
  method = "plm_within",
  lags = 2,
  equations = simple_results,
  note = "Estimated using plm package with within (fixed effects) estimator"
)

# Save to JSON
write_json(simple_output, "../reference_outputs/r_simple_pvar.json",
           pretty = TRUE, auto_unbox = TRUE, digits = 10)
cat("  ✓ Saved r_simple_pvar.json\n")

# Process love_zicchino_synthetic.csv dataset
cat("Processing love_zicchino_synthetic.csv...\n")
data_lz <- read.csv("../data/love_zicchino_synthetic.csv")

lz_results <- estimate_panel_var(
  data_lz,
  entity_col = "firm_id",
  time_col = "year",
  y_cols = c("sales", "inv", "ar", "debt"),
  lags = 2
)

lz_output <- list(
  dataset = "love_zicchino_synthetic",
  method = "plm_within",
  lags = 2,
  equations = lz_results,
  note = "Estimated using plm package with within (fixed effects) estimator"
)

write_json(lz_output, "../reference_outputs/r_love_zicchino.json",
           pretty = TRUE, auto_unbox = TRUE, digits = 10)
cat("  ✓ Saved r_love_zicchino.json\n")

# Process unbalanced_panel.csv dataset
cat("Processing unbalanced_panel.csv...\n")
data_unbal <- read.csv("../data/unbalanced_panel.csv")

unbal_results <- estimate_panel_var(
  data_unbal,
  entity_col = "entity",
  time_col = "time",
  y_cols = c("y1", "y2"),
  lags = 2
)

unbal_output <- list(
  dataset = "unbalanced_panel",
  method = "plm_within",
  lags = 2,
  equations = unbal_results,
  note = "Estimated using plm package with within (fixed effects) estimator. Unbalanced panel."
)

write_json(unbal_output, "../reference_outputs/r_unbalanced.json",
           pretty = TRUE, auto_unbox = TRUE, digits = 10)
cat("  ✓ Saved r_unbalanced.json\n")

cat("\n✓ All R reference outputs generated successfully!\n")
cat("\nIMPORTANT NOTE:\n")
cat("These outputs use plm (fixed effects) as a baseline reference.\n")
cat("For full Panel VAR GMM validation, the 'pvar' package would be needed.\n")
cat("The current outputs provide a good starting point for basic validation.\n")
