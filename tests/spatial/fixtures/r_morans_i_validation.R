# R Validation Script for Moran's I and LISA Tests
# This script generates reference results for validating Python implementation

library(spdep)
library(plm)
library(jsonlite)

# Load test data
data <- read.csv("spatial_test_data.csv")
W <- as.matrix(read.csv("spatial_weights.csv", header=FALSE))

# Create spatial weights list (row-normalized)
W_list <- mat2listw(W, style="W")

# Convert to pdata.frame
pdata <- pdata.frame(data, index=c("entity", "time"))

# Fit pooled OLS
pooled_ols <- plm(y ~ x1 + x2 + x3, data = pdata, model = "pooling")
residuals <- residuals(pooled_ols)

# Get unique entities and time periods
n_entities <- length(unique(data$entity))
n_periods <- length(unique(data$time))

cat("Number of entities:", n_entities, "\n")
cat("Number of periods:", n_periods, "\n")
cat("Total observations:", nrow(data), "\n\n")

# ============================================================================
# MORAN'S I - POOLED METHOD
# ============================================================================

# Compute time-averaged residuals by entity
resid_by_entity <- aggregate(residuals,
                              by = list(entity = data$entity),
                              FUN = mean)$x

cat("Computing Moran's I (pooled method)...\n")
moran_result <- moran.test(resid_by_entity, W_list)

cat("\nMoran's I Test (Pooled):\n")
cat("  Statistic:", moran_result$estimate["Moran I statistic"], "\n")
cat("  Expected:", moran_result$estimate["Expectation"], "\n")
cat("  Variance:", moran_result$estimate["Variance"], "\n")
cat("  Z-score:", moran_result$statistic, "\n")
cat("  P-value:", moran_result$p.value, "\n\n")

# ============================================================================
# MORAN'S I - BY PERIOD METHOD
# ============================================================================

cat("Computing Moran's I by period...\n")
moran_by_period <- list()
unique_times <- unique(data$time)

for (t in unique_times) {
    idx <- data$time == t
    resid_t <- residuals[idx]

    tryCatch({
        moran_t <- moran.test(resid_t, W_list)

        moran_by_period[[as.character(t)]] <- list(
            statistic = as.numeric(moran_t$estimate["Moran I statistic"]),
            expected = as.numeric(moran_t$estimate["Expectation"]),
            variance = as.numeric(moran_t$estimate["Variance"]),
            z_score = as.numeric(moran_t$statistic),
            pvalue = as.numeric(moran_t$p.value)
        )

        cat("  Period", t, "- I:", moran_t$estimate["Moran I statistic"],
            "p-value:", moran_t$p.value, "\n")
    }, error = function(e) {
        cat("  Period", t, "- Error:", e$message, "\n")
        moran_by_period[[as.character(t)]] <- list(
            statistic = NA,
            expected = NA,
            variance = NA,
            z_score = NA,
            pvalue = NA
        )
    })
}

cat("\n")

# ============================================================================
# LOCAL MORAN'S I (LISA)
# ============================================================================

cat("Computing Local Moran's I (LISA)...\n")
lisa_result <- localmoran(resid_by_entity, W_list)

# Standardize residuals for cluster classification
z_resid <- scale(resid_by_entity)
Wz <- lag.listw(W_list, z_resid)

# Classify cluster types
cluster_types <- rep("Not significant", n_entities)
sig_level <- 0.05

for (i in 1:n_entities) {
    if (lisa_result[i, "Pr(z != E(Ii))"] < sig_level) {
        if (z_resid[i] > 0 && Wz[i] > 0) {
            cluster_types[i] <- "HH"
        } else if (z_resid[i] < 0 && Wz[i] < 0) {
            cluster_types[i] <- "LL"
        } else if (z_resid[i] > 0 && Wz[i] < 0) {
            cluster_types[i] <- "HL"
        } else {
            cluster_types[i] <- "LH"
        }
    }
}

cluster_counts <- table(cluster_types)

cat("\nLISA Cluster Counts:\n")
print(cluster_counts)
cat("\n")

# ============================================================================
# COMPILE AND SAVE RESULTS
# ============================================================================

results <- list(
    moran_pooled = list(
        statistic = as.numeric(moran_result$estimate["Moran I statistic"]),
        expected = as.numeric(moran_result$estimate["Expectation"]),
        variance = as.numeric(moran_result$estimate["Variance"]),
        z_score = as.numeric(moran_result$statistic),
        pvalue = as.numeric(moran_result$p.value)
    ),
    moran_by_period = moran_by_period,
    lisa = list(
        local_i = as.numeric(lisa_result[, "Ii"]),
        pvalues = as.numeric(lisa_result[, "Pr(z != E(Ii))"]),
        z_scores = as.numeric(lisa_result[, "Z.Ii"]),
        expected = as.numeric(lisa_result[, "E.Ii"]),
        variance = as.numeric(lisa_result[, "Var.Ii"]),
        cluster_counts = as.list(cluster_counts),
        z_values = as.numeric(z_resid),
        Wz_values = as.numeric(Wz)
    ),
    metadata = list(
        n_entities = n_entities,
        n_periods = n_periods,
        n_observations = nrow(data)
    )
)

# Save to JSON
output_file <- "r_morans_i_results.json"
write_json(results, output_file, pretty = TRUE, auto_unbox = TRUE, digits = 10)

cat("\nResults saved to", output_file, "\n")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

cat("\n========================================\n")
cat("SUMMARY\n")
cat("========================================\n\n")

cat("Moran's I (Pooled):\n")
cat("  Statistic:", results$moran_pooled$statistic, "\n")
cat("  P-value:", results$moran_pooled$pvalue, "\n")
cat("  Conclusion:", ifelse(results$moran_pooled$pvalue < 0.05,
                            "Significant spatial autocorrelation",
                            "No significant spatial autocorrelation"), "\n\n")

cat("LISA Analysis:\n")
cat("  Total entities:", n_entities, "\n")
cat("  Significant clusters:", sum(cluster_types != "Not significant"), "\n")
cat("  Cluster breakdown:\n")
for (cluster_type in names(cluster_counts)) {
    pct <- 100 * cluster_counts[cluster_type] / n_entities
    cat("    ", cluster_type, ":", cluster_counts[cluster_type],
        "(", sprintf("%.1f%%", pct), ")\n")
}

cat("\n========================================\n")
cat("Validation script completed successfully!\n")
cat("========================================\n")
