###############################################################################
# Validation 02 - Clustered Standard Errors
#
# Estimates Fixed Effects (within) on Grunfeld data and computes clustered
# standard errors: by entity (firm), by time (year), and two-way (firm+year).
# Uses sandwich::vcovCL() and plm::vcovHC()/vcovDC().
#
# Dataset: Grunfeld (10 firms, 20 years)
# Model:   invest ~ value + capital (FE by firm)
###############################################################################

library(plm)
library(sandwich)
library(lmtest)

# --- Data Loading -----------------------------------------------------------
data_path <- "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv"
grunfeld <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("Rows:", nrow(grunfeld), "\n")
cat("Firms:", length(unique(grunfeld$firm)), "\n")
cat("Years:", paste(sort(unique(grunfeld$year)), collapse = ", "), "\n\n")

# --- Panel data frame -------------------------------------------------------
pgrunfeld <- pdata.frame(grunfeld, index = c("firm", "year"))

# --- Fixed Effects (within) estimation --------------------------------------
fe_model <- plm(invest ~ value + capital, data = pgrunfeld,
                model = "within", effect = "individual")

cat("=== FE (Within) Coefficients ===\n")
print(summary(fe_model)$coefficients)
cat("\n")

# --- 1. Non-robust SE (default) ---------------------------------------------
ct_nonrobust <- coeftest(fe_model)
cat("=== Non-robust SE ===\n")
print(ct_nonrobust)
cat("\n")

# --- 2. HC1 Robust SE -------------------------------------------------------
vcov_hc1 <- vcovHC(fe_model, type = "HC1")
ct_hc1 <- coeftest(fe_model, vcov. = vcov_hc1)
cat("=== HC1 Robust SE ===\n")
print(ct_hc1)
cat("\n")

# --- 3. Clustered by Entity (firm) ------------------------------------------
# For plm objects, vcovHC with cluster="group" gives entity-clustered SE
vcov_cl_entity <- vcovHC(fe_model, type = "HC1", cluster = "group")
ct_cl_entity <- coeftest(fe_model, vcov. = vcov_cl_entity)
cat("=== Clustered by Entity (firm) SE ===\n")
print(ct_cl_entity)
cat("\n")

# --- 4. Clustered by Time (year) --------------------------------------------
vcov_cl_time <- vcovHC(fe_model, type = "HC1", cluster = "time")
ct_cl_time <- coeftest(fe_model, vcov. = vcov_cl_time)
cat("=== Clustered by Time (year) SE ===\n")
print(ct_cl_time)
cat("\n")

# --- 5. Two-way clustering (firm + year) ------------------------------------
# plm::vcovDC() implements Cameron-Gelbach-Miller (2011) two-way clustering
vcov_twoway <- vcovDC(fe_model)
ct_twoway <- coeftest(fe_model, vcov. = vcov_twoway)
cat("=== Two-Way Clustered (firm + year) SE ===\n")
print(ct_twoway)
cat("\n")

# --- Build results data.frame -----------------------------------------------
build_rows <- function(ct, se_type) {
  vars <- rownames(ct)
  data.frame(
    model_name  = "fe_within",
    se_type     = se_type,
    variable    = vars,
    coefficient = ct[, "Estimate"],
    std_error   = ct[, "Std. Error"],
    t_statistic = ct[, "t value"],
    p_value     = ct[, "Pr(>|t|)"],
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

results <- rbind(
  build_rows(ct_nonrobust, "nonrobust"),
  build_rows(ct_hc1, "HC1"),
  build_rows(ct_cl_entity, "cluster_entity"),
  build_rows(ct_cl_time, "cluster_time"),
  build_rows(ct_twoway, "cluster_twoway")
)

# --- Sanity check: coefficients must be identical ---------------------------
cat("=== Sanity Check: Coefficient Consistency ===\n")
coef_by_type <- tapply(results$coefficient, results$se_type, function(x) x)
ref <- coef_by_type[[1]]
all_equal <- all(sapply(coef_by_type, function(x) isTRUE(all.equal(x, ref))))
cat("All coefficients identical across SE types:", all_equal, "\n\n")

# --- Save to CSV -------------------------------------------------------------
out_dir <- "/home/guhaase/projetos/panelbox/examples/standard_errors/R"
out_file <- file.path(out_dir, "results_clustered.csv")
write.csv(results, out_file, row.names = FALSE)
cat("Results saved to:", out_file, "\n")

# --- Summary table -----------------------------------------------------------
cat("\n=== SE Comparison Summary ===\n")
se_wide <- reshape(results[, c("se_type", "variable", "std_error")],
                   idvar = "variable", timevar = "se_type",
                   direction = "wide")
print(se_wide)
cat("\nDone.\n")
