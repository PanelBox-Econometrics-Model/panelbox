###############################################################################
# Validation 03 - Multinomial Logit
#
# Replicates PanelBox notebook: 06_multinomial_logit.ipynb
# Dataset: career_choice.csv
# Models:
#   1. Multinomial Logit (nnet::multinom): career ~ educ + exper + age + female
#   2. Extended Multinomial Logit: career ~ educ + exper + age + female + urban
#
# career categories: 0=Manual, 1=Technical, 2=Managerial
# Base category: 0 (Manual)
###############################################################################

# --- Setup -------------------------------------------------------------------
rm(list = ls())
suppressPackageStartupMessages({
  library(nnet)
})

data_path <- "/home/guhaase/projetos/panelbox/examples/discrete/data/career_choice.csv"
out_path  <- "/home/guhaase/projetos/panelbox/examples/discrete/R/results_03_multinomial.csv"

data <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("Observations:", nrow(data), "\n")
cat("Individuals:", length(unique(data$id)), "\n")
cat("Time periods:", length(unique(data$year)), "\n")
cat("\nCareer distribution:\n")
print(table(data$career))
cat("\n")

# Ensure career is a factor with base = 0
data$career <- factor(data$career, levels = c(0, 1, 2),
                      labels = c("Manual", "Technical", "Managerial"))

# --- Model 1: Base Multinomial Logit -----------------------------------------
cat("=== Model 1: Multinomial Logit (base specification) ===\n")
cat("Formula: career ~ educ + exper + age + female\n")
cat("Base category: Manual (0)\n\n")

mlogit1 <- multinom(career ~ educ + exper + age + female,
                    data = data, trace = FALSE)
s1 <- summary(mlogit1)

cat("Coefficients:\n")
print(s1$coefficients)
cat("\nStandard Errors:\n")
print(s1$standard.errors)

# Compute z-statistics and p-values
z1 <- s1$coefficients / s1$standard.errors
p1 <- 2 * pnorm(-abs(z1))

cat("\nZ-statistics:\n")
print(z1)
cat("\nP-values:\n")
print(p1)

cat("\nLog-likelihood:", as.numeric(logLik(mlogit1)), "\n")
cat("AIC:", AIC(mlogit1), "\n")
cat("BIC:", BIC(mlogit1), "\n")
cat("Residual deviance:", mlogit1$deviance, "\n\n")

# --- Model 2: Extended Multinomial Logit --------------------------------------
cat("=== Model 2: Extended Multinomial Logit ===\n")
cat("Formula: career ~ educ + exper + age + female + urban\n\n")

mlogit2 <- multinom(career ~ educ + exper + age + female + urban,
                    data = data, trace = FALSE)
s2 <- summary(mlogit2)

cat("Coefficients:\n")
print(s2$coefficients)
cat("\nStandard Errors:\n")
print(s2$standard.errors)

z2 <- s2$coefficients / s2$standard.errors
p2 <- 2 * pnorm(-abs(z2))

cat("\nZ-statistics:\n")
print(z2)
cat("\nP-values:\n")
print(p2)

cat("\nLog-likelihood:", as.numeric(logLik(mlogit2)), "\n")
cat("AIC:", AIC(mlogit2), "\n")
cat("BIC:", BIC(mlogit2), "\n\n")

# --- Collect results for CSV --------------------------------------------------
collect_multinom <- function(model, model_name) {
  s <- summary(model)
  categories <- rownames(s$coefficients)
  variables  <- colnames(s$coefficients)
  z_mat <- s$coefficients / s$standard.errors
  p_mat <- 2 * pnorm(-abs(z_mat))

  rows <- list()
  for (cat_idx in seq_along(categories)) {
    for (var_idx in seq_along(variables)) {
      rows[[length(rows) + 1]] <- data.frame(
        model_name     = model_name,
        category       = categories[cat_idx],
        variable       = variables[var_idx],
        coefficient    = s$coefficients[cat_idx, var_idx],
        std_error      = s$standard.errors[cat_idx, var_idx],
        z_statistic    = z_mat[cat_idx, var_idx],
        p_value        = p_mat[cat_idx, var_idx],
        log_likelihood = as.numeric(logLik(model)),
        aic            = AIC(model),
        bic            = BIC(model),
        n_obs          = nrow(model$fitted.values),
        base_category  = "Manual",
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

results <- rbind(
  collect_multinom(mlogit1, "multinomial_logit_base"),
  collect_multinom(mlogit2, "multinomial_logit_extended")
)

# --- Save CSV ----------------------------------------------------------------
write.csv(results, out_path, row.names = FALSE)
cat("Results saved to:", out_path, "\n")

cat("\n=== All Results ===\n")
print(results, digits = 6, row.names = FALSE)
