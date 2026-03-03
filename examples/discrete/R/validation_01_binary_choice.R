###############################################################################
# Validation 01 - Binary Choice Models (Pooled Logit and Pooled Probit)
#
# Replicates PanelBox notebook: 01_binary_choice_introduction.ipynb
# Dataset: labor_participation.csv
# Models:
#   1. Pooled Logit:  lfp ~ age + educ + kids + married
#   2. Pooled Probit: lfp ~ age + educ + kids + married
#   3. Full Logit:    lfp ~ age + I(age^2) + educ + kids + married + exper
###############################################################################

# --- Setup -------------------------------------------------------------------
rm(list = ls())

data_path <- "/home/guhaase/projetos/panelbox/examples/discrete/data/labor_participation.csv"
out_path  <- "/home/guhaase/projetos/panelbox/examples/discrete/R/results_01_binary_choice.csv"

data <- read.csv(data_path)

cat("=== Dataset Summary ===\n")
cat("Observations:", nrow(data), "\n")
cat("Individuals:", length(unique(data$id)), "\n")
cat("Time periods:", length(unique(data$year)), "\n")
cat("LFP rate:", round(mean(data$lfp), 4), "\n\n")

# --- Model 1: Pooled Logit ---------------------------------------------------
cat("=== Model 1: Pooled Logit ===\n")
logit_model <- glm(lfp ~ age + educ + kids + married,
                   data = data, family = binomial(link = "logit"))
summary(logit_model)

# --- Model 2: Pooled Probit --------------------------------------------------
cat("\n=== Model 2: Pooled Probit ===\n")
probit_model <- glm(lfp ~ age + educ + kids + married,
                    data = data, family = binomial(link = "probit"))
summary(probit_model)

# --- Model 3: Full Logit (with age^2 and exper) ------------------------------
cat("\n=== Model 3: Full Logit (quadratic age + exper) ===\n")
logit_full <- glm(lfp ~ age + I(age^2) + educ + kids + married + exper,
                  data = data, family = binomial(link = "logit"))
summary(logit_full)

# --- Collect Results ----------------------------------------------------------
extract_results <- function(model, model_name) {
  s <- summary(model)
  cf <- coef(s)
  data.frame(
    model_name  = model_name,
    variable    = rownames(cf),
    coefficient = cf[, "Estimate"],
    std_error   = cf[, "Std. Error"],
    z_statistic = cf[, "z value"],
    p_value     = cf[, "Pr(>|z|)"],
    log_likelihood = as.numeric(logLik(model)),
    aic         = AIC(model),
    bic         = BIC(model),
    n_obs       = nrow(model$data),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

results <- rbind(
  extract_results(logit_model,  "pooled_logit"),
  extract_results(probit_model, "pooled_probit"),
  extract_results(logit_full,   "full_logit")
)

# --- Save CSV ----------------------------------------------------------------
write.csv(results, out_path, row.names = FALSE)
cat("\nResults saved to:", out_path, "\n")

# --- Print final table -------------------------------------------------------
cat("\n=== All Results ===\n")
print(results, digits = 6, row.names = FALSE)
