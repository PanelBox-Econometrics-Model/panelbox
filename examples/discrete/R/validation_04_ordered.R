###############################################################################
# Validation 04 - Ordered Logit and Ordered Probit
#
# Replicates PanelBox notebook: 07_ordered_models.ipynb
# Dataset: credit_rating.csv
# Models:
#   1. Ordered Logit:  rating ~ income + debt_ratio + age + size + profitability
#   2. Ordered Probit: rating ~ income + debt_ratio + age + size + profitability
#
# rating categories: 0=Poor, 1=Fair, 2=Good, 3=Excellent (ordinal)
###############################################################################

# --- Setup -------------------------------------------------------------------
rm(list = ls())
suppressPackageStartupMessages({
  library(MASS)
})

data_path <- "/home/guhaase/projetos/panelbox/examples/discrete/data/credit_rating.csv"
out_path  <- "/home/guhaase/projetos/panelbox/examples/discrete/R/results_04_ordered.csv"

data <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("Observations:", nrow(data), "\n")
cat("Firms:", length(unique(data$id)), "\n")
cat("Time periods:", length(unique(data$year)), "\n")
cat("\nRating distribution:\n")
print(table(data$rating))
cat("\n")

# Convert rating to ordered factor
data$rating <- ordered(data$rating, levels = c(0, 1, 2, 3),
                       labels = c("Poor", "Fair", "Good", "Excellent"))

# --- Model 1: Ordered Logit --------------------------------------------------
cat("=== Model 1: Ordered Logit (polr) ===\n")

ologit <- polr(rating ~ income + debt_ratio + age + size + profitability,
               data = data, method = "logistic", Hess = TRUE)
s_ologit <- summary(ologit)
print(s_ologit)

# Compute p-values (polr does not provide them by default)
ct_ologit <- coef(s_ologit)
p_ologit <- 2 * pnorm(-abs(ct_ologit[, "t value"]))

cat("\nLog-likelihood:", as.numeric(logLik(ologit)), "\n")
cat("AIC:", AIC(ologit), "\n")
cat("BIC:", BIC(ologit), "\n\n")

# --- Model 2: Ordered Probit -------------------------------------------------
cat("=== Model 2: Ordered Probit (polr) ===\n")

oprobit <- polr(rating ~ income + debt_ratio + age + size + profitability,
                data = data, method = "probit", Hess = TRUE)
s_oprobit <- summary(oprobit)
print(s_oprobit)

ct_oprobit <- coef(s_oprobit)
p_oprobit <- 2 * pnorm(-abs(ct_oprobit[, "t value"]))

cat("\nLog-likelihood:", as.numeric(logLik(oprobit)), "\n")
cat("AIC:", AIC(oprobit), "\n")
cat("BIC:", BIC(oprobit), "\n\n")

# --- Collect Results ----------------------------------------------------------
collect_ordered <- function(model, model_name, p_values) {
  s <- summary(model)
  ct <- coef(s)
  var_names <- rownames(ct)
  n_coef <- length(model$coefficients)

  rows <- list()
  for (i in seq_len(nrow(ct))) {
    is_cutpoint <- i > n_coef
    rows[[i]] <- data.frame(
      model_name     = model_name,
      variable       = var_names[i],
      coefficient    = ct[i, "Value"],
      std_error      = ct[i, "Std. Error"],
      t_statistic    = ct[i, "t value"],
      p_value        = p_values[i],
      is_cutpoint    = is_cutpoint,
      log_likelihood = as.numeric(logLik(model)),
      aic            = AIC(model),
      bic            = BIC(model),
      n_obs          = nrow(model$model),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, rows)
}

results <- rbind(
  collect_ordered(ologit,  "ordered_logit",  p_ologit),
  collect_ordered(oprobit, "ordered_probit", p_oprobit)
)

# --- Save CSV ----------------------------------------------------------------
write.csv(results, out_path, row.names = FALSE)
cat("Results saved to:", out_path, "\n")

cat("\n=== All Results ===\n")
print(results, digits = 6, row.names = FALSE)

# --- Print cutpoints separately for clarity -----------------------------------
cat("\n=== Ordered Logit Cutpoints ===\n")
print(ologit$zeta)

cat("\n=== Ordered Probit Cutpoints ===\n")
print(oprobit$zeta)
