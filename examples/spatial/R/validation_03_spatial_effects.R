# =============================================================================
# Validation Script 03: Spatial Direct, Indirect, and Total Effects
# =============================================================================
# Dataset: Columbus (spdep package) - 49 neighborhoods in Columbus, Ohio
# Model: CRIME ~ INC + HOVAL
# Weight matrix: Queen contiguity (row-standardized)
# Estimation: ML for SAR and SDM, then impacts() for effect decomposition
# =============================================================================

library(spdep)
library(spatialreg)

# ---- Output directory ----
output_dir <- "/home/guhaase/projetos/panelbox/examples/spatial/R"

# ---- Load Columbus dataset and spatial weights ----
data(columbus, package = "spdep")
col_nb <- col.gal.nb
col_listw <- nb2listw(col_nb, style = "W")

cat("=== Columbus Dataset ===\n")
cat("Observations:", nrow(columbus), "\n\n")

# Helper: extract impacts into a data frame
extract_impacts <- function(impacts_obj, impacts_sum, model_label, var_names) {
  # Extract point estimates from res
  direct_vals <- as.numeric(impacts_obj$res$direct)
  indirect_vals <- as.numeric(impacts_obj$res$indirect)
  total_vals <- as.numeric(impacts_obj$res$total)

  n_vars <- length(var_names)

  # Build effects data frame
  effects_df <- data.frame(
    model_name = model_label,
    variable = rep(var_names, 3),
    effect_type = rep(c("direct", "indirect", "total"), each = n_vars),
    effect_value = c(direct_vals, indirect_vals, total_vals),
    stringsAsFactors = FALSE
  )

  # Extract SE, z-stats, and p-values from summary
  # semat, zmat, pzmat all have rows=variables, columns=Direct/Indirect/Total
  se_mat <- impacts_sum$semat
  z_mat <- impacts_sum$zmat
  p_mat <- impacts_sum$pzmat

  effects_df$std_error <- c(se_mat[, "Direct"], se_mat[, "Indirect"], se_mat[, "Total"])
  effects_df$z_value <- c(z_mat[, "Direct"], z_mat[, "Indirect"], z_mat[, "Total"])
  effects_df$p_value <- c(p_mat[, "Direct"], p_mat[, "Indirect"], p_mat[, "Total"])

  return(effects_df)
}

# ---- 1. SAR Model ----
cat("=== SAR Model (for effects decomposition) ===\n")
sar_model <- lagsarlm(CRIME ~ INC + HOVAL, data = columbus, listw = col_listw)
sar_sum <- summary(sar_model)
print(sar_sum)
cat("\n")

# ---- 2. SAR Impacts (simulation-based) ----
cat("=== SAR Impacts (simulation-based, R=999) ===\n")
set.seed(12345)
sar_impacts <- impacts(sar_model, listw = col_listw, R = 999)
sar_impacts_sum <- summary(sar_impacts, zstats = TRUE, short = TRUE)
print(sar_impacts_sum)
cat("\n")

# Variable names from the model (excluding intercept)
var_names <- c("INC", "HOVAL")

sar_effects_df <- extract_impacts(sar_impacts, sar_impacts_sum, "sar_ml", var_names)

cat("SAR Effects:\n")
print(sar_effects_df)
cat("\n")

# ---- 3. SDM Model ----
cat("=== SDM Model (for effects decomposition) ===\n")
sdm_model <- lagsarlm(CRIME ~ INC + HOVAL, data = columbus,
                       listw = col_listw, type = "mixed")
sdm_sum <- summary(sdm_model)
print(sdm_sum)
cat("\n")

# ---- 4. SDM Impacts (simulation-based) ----
cat("=== SDM Impacts (simulation-based, R=999) ===\n")
set.seed(12345)
sdm_impacts <- impacts(sdm_model, listw = col_listw, R = 999)
sdm_impacts_sum <- summary(sdm_impacts, zstats = TRUE, short = TRUE)
print(sdm_impacts_sum)
cat("\n")

sdm_effects_df <- extract_impacts(sdm_impacts, sdm_impacts_sum, "sdm_ml", var_names)

cat("SDM Effects:\n")
print(sdm_effects_df)
cat("\n")

# ---- 5. Combine and save ----
results <- rbind(sar_effects_df, sdm_effects_df)
rownames(results) <- NULL

output_file <- file.path(output_dir, "results_03_spatial_effects.csv")
write.csv(results, output_file, row.names = FALSE)
cat("Results saved to:", output_file, "\n")

cat("\n=== Summary Results Table ===\n")
print(results)

# ---- 6. Also save a combined results_spatial.csv ----
results_01_file <- file.path(output_dir, "results_01_moran_sar.csv")
results_02_file <- file.path(output_dir, "results_02_sem_sdm.csv")

if (file.exists(results_01_file) && file.exists(results_02_file)) {
  r01 <- read.csv(results_01_file, stringsAsFactors = FALSE)
  r02 <- read.csv(results_02_file, stringsAsFactors = FALSE)

  # Standardize columns for combined file
  cols_shared <- c("model_name", "variable", "coefficient", "std_error", "statistic", "p_value")

  r01_std <- r01[, cols_shared]
  r02_std <- r02[, cols_shared]

  # Script 03 effects - adapt column names
  r03_std <- data.frame(
    model_name = paste0(results$model_name, "_", results$effect_type),
    variable = results$variable,
    coefficient = results$effect_value,
    std_error = results$std_error,
    statistic = results$z_value,
    p_value = results$p_value,
    stringsAsFactors = FALSE
  )

  combined <- rbind(r01_std, r02_std, r03_std)
  combined_file <- file.path(output_dir, "results_spatial.csv")
  write.csv(combined, combined_file, row.names = FALSE)
  cat("\nCombined results saved to:", combined_file, "\n")
}

cat("\nDone.\n")
