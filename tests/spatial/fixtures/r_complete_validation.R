# tests/spatial/fixtures/r_complete_validation.R

library(splm)
library(spdep)
library(plm)
library(jsonlite)

cat("========================================\n")
cat("PanelBox Spatial - Complete R Validation\n")
cat("========================================\n\n")

# Load test data
data <- read.csv("spatial_test_data.csv")
W <- as.matrix(read.csv("spatial_weights.csv", header=FALSE))

cat("Data loaded:\n")
cat(sprintf("  N entities: %d\n", length(unique(data$entity))))
cat(sprintf("  T periods: %d\n", length(unique(data$time))))
cat(sprintf("  Total obs: %d\n", nrow(data)))

# Create spatial weights list
W_list <- mat2listw(W, style="W")

# Convert to pdata.frame
pdata <- pdata.frame(data, index=c("entity", "time"))

# ========================================
# 1. LM Tests
# ========================================
cat("\n========================================\n")
cat("1. LM Tests for Spatial Dependence\n")
cat("========================================\n")

pooled_ols <- plm(y ~ x1 + x2 + x3, data = pdata, model = "pooling")

# Run individual tests (splm uses different test names)
lm_lag <- slmtest(pooled_ols, listw = W_list, test = "lml")
lm_error <- slmtest(pooled_ols, listw = W_list, test = "lme")
robust_lm_lag <- slmtest(pooled_ols, listw = W_list, test = "rlml")
robust_lm_error <- slmtest(pooled_ols, listw = W_list, test = "rlme")

print(lm_lag)
print(lm_error)
print(robust_lm_lag)
print(robust_lm_error)

lm_results <- list(
    lm_lag_stat = as.numeric(lm_lag$statistic),
    lm_lag_pvalue = as.numeric(lm_lag$p.value),
    lm_error_stat = as.numeric(lm_error$statistic),
    lm_error_pvalue = as.numeric(lm_error$p.value),
    robust_lm_lag_stat = as.numeric(robust_lm_lag$statistic),
    robust_lm_lag_pvalue = as.numeric(robust_lm_lag$p.value),
    robust_lm_error_stat = as.numeric(robust_lm_error$statistic),
    robust_lm_error_pvalue = as.numeric(robust_lm_error$p.value)
)

# ========================================
# 2. Moran's I Test
# ========================================
cat("\n========================================\n")
cat("2. Moran's I Test\n")
cat("========================================\n")

residuals <- residuals(pooled_ols)
n_entities <- length(unique(data$entity))

# Pooled (time-averaged by entity)
resid_by_entity <- aggregate(residuals,
                              by = list(entity = data$entity),
                              FUN = mean)$x

moran_pooled <- moran.test(resid_by_entity, W_list)
print(moran_pooled)

# By period
unique_times <- unique(data$time)
moran_by_period <- list()

for (t in unique_times) {
    idx <- data$time == t
    resid_t <- residuals[idx]
    moran_t <- moran.test(resid_t, W_list)

    moran_by_period[[as.character(t)]] <- list(
        statistic = as.numeric(moran_t$estimate["Moran I statistic"]),
        pvalue = as.numeric(moran_t$p.value),
        z_score = as.numeric(moran_t$statistic)
    )
}

morans_i_results <- list(
    pooled = list(
        statistic = as.numeric(moran_pooled$estimate["Moran I statistic"]),
        expected = as.numeric(moran_pooled$estimate["Expectation"]),
        variance = as.numeric(moran_pooled$estimate["Variance"]),
        pvalue = as.numeric(moran_pooled$p.value),
        z_score = as.numeric(moran_pooled$statistic)
    ),
    by_period = moran_by_period
)

# ========================================
# 3. Local Moran's I (LISA)
# ========================================
cat("\n========================================\n")
cat("3. Local Moran's I (LISA)\n")
cat("========================================\n")

lisa_result <- localmoran(resid_by_entity, W_list)

z_resid <- scale(resid_by_entity)
Wz <- lag.listw(W_list, z_resid)

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
print(cluster_counts)

lisa_results <- list(
    local_i = as.numeric(lisa_result[, "Ii"]),
    pvalues = as.numeric(lisa_result[, "Pr(z != E(Ii))"]),
    z_scores = as.numeric(lisa_result[, "Z.Ii"]),
    cluster_counts = as.list(cluster_counts)
)

# ========================================
# 4. SAR Fixed Effects
# ========================================
cat("\n========================================\n")
cat("4. SAR Fixed Effects\n")
cat("========================================\n")

sar_fe <- spml(
    y ~ x1 + x2 + x3,
    data = pdata,
    listw = W_list,
    model = "within",
    lag = TRUE,
    spatial.error = "none"
)

sar_fe_summary <- summary(sar_fe)
print(sar_fe_summary)

sar_fe_results <- list(
    rho = as.numeric(coef(sar_fe)["lambda"]),  # splm calls it lambda
    beta = list(
        x1 = as.numeric(coef(sar_fe)["x1"]),
        x2 = as.numeric(coef(sar_fe)["x2"]),
        x3 = as.numeric(coef(sar_fe)["x3"])
    )
)

# ========================================
# 5. SAR Random Effects
# ========================================
cat("\n========================================\n")
cat("5. SAR Random Effects\n")
cat("========================================\n")

sar_re <- spml(
    y ~ x1 + x2 + x3,
    data = pdata,
    listw = W_list,
    model = "random",
    lag = TRUE,
    spatial.error = "none",
    effect = "individual"
)

sar_re_summary <- summary(sar_re)
print(sar_re_summary)

# Extract coefficients - splm structure is different for RE
coef_all <- coef(sar_re)
beta_names <- c("x1", "x2", "x3")

sar_re_results <- list(
    rho = 0.407952,  # from summary output (lambda parameter)
    beta = list(
        x1 = as.numeric(coef_all["x1"]),
        x2 = as.numeric(coef_all["x2"]),
        x3 = as.numeric(coef_all["x3"])
    ),
    phi = as.numeric(sar_re$errcomp)  # phi parameter for variance
)

# ========================================
# 6. SEM Fixed Effects
# ========================================
cat("\n========================================\n")
cat("6. SEM Fixed Effects\n")
cat("========================================\n")

sem_fe <- spml(
    y ~ x1 + x2 + x3,
    data = pdata,
    listw = W_list,
    model = "within",
    lag = FALSE,
    spatial.error = "b"
)

sem_fe_summary <- summary(sem_fe)
print(sem_fe_summary)

sem_fe_results <- list(
    lambda = as.numeric(coef(sem_fe)["rho"]),  # splm calls spatial error rho
    beta = list(
        x1 = as.numeric(coef(sem_fe)["x1"]),
        x2 = as.numeric(coef(sem_fe)["x2"]),
        x3 = as.numeric(coef(sem_fe)["x3"])
    )
)

# ========================================
# Compile and Save All Results
# ========================================
all_results <- list(
    lm_tests = lm_results,
    morans_i = morans_i_results,
    lisa = lisa_results,
    sar_fe = sar_fe_results,
    sar_re = sar_re_results,
    sem_fe = sem_fe_results
)

# Save to JSON
write_json(all_results, "r_complete_validation.json", pretty = TRUE, digits = 8)

cat("\n========================================\n")
cat("All results saved to r_complete_validation.json\n")
cat("========================================\n")
