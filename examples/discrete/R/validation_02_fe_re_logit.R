###############################################################################
# Validation 02 - Fixed Effects and Random Effects Logit/Probit
#
# Replicates PanelBox notebooks:
#   02_fixed_effects_logit.ipynb (FE Logit on job_training data)
#   03_random_effects.ipynb (RE Probit on labor_participation data)
#
# Models:
#   1. FE Logit (conditional logit) on job_training: employed ~ training + age + prior_wage
#   2. RE Probit on labor_participation: lfp ~ age + educ + kids + married
#   3. RE Logit on labor_participation: lfp ~ age + educ + kids + married
#   4. Pooled Probit baseline (for comparison)
###############################################################################

# --- Setup -------------------------------------------------------------------
rm(list = ls())
suppressPackageStartupMessages({
  library(survival)
  library(bife)
  library(lme4)
})

data_jt_path <- "/home/guhaase/projetos/panelbox/examples/discrete/data/job_training.csv"
data_lp_path <- "/home/guhaase/projetos/panelbox/examples/discrete/data/labor_participation.csv"
out_path     <- "/home/guhaase/projetos/panelbox/examples/discrete/R/results_02_fe_re_logit.csv"

data_jt <- read.csv(data_jt_path)
data_lp <- read.csv(data_lp_path)

cat("=== Job Training Dataset ===\n")
cat("Observations:", nrow(data_jt), "\n")
cat("Individuals:", length(unique(data_jt$id)), "\n")
cat("Employment rate:", round(mean(data_jt$employed), 4), "\n\n")

cat("=== Labor Participation Dataset ===\n")
cat("Observations:", nrow(data_lp), "\n")
cat("Individuals:", length(unique(data_lp$id)), "\n")
cat("LFP rate:", round(mean(data_lp$lfp), 4), "\n\n")

# --- Helper to collect results -----------------------------------------------
all_results <- list()

add_result <- function(model_name, variable, coefficient, std_error,
                       z_statistic, p_value,
                       log_likelihood = NA, n_obs = NA,
                       n_switchers = NA, n_dropped = NA,
                       sigma_u = NA, rho = NA) {
  row <- data.frame(
    model_name     = model_name,
    variable       = variable,
    coefficient    = coefficient,
    std_error      = std_error,
    z_statistic    = z_statistic,
    p_value        = p_value,
    log_likelihood = log_likelihood,
    n_obs          = n_obs,
    n_switchers    = n_switchers,
    n_dropped      = n_dropped,
    sigma_u        = sigma_u,
    rho            = rho,
    stringsAsFactors = FALSE
  )
  all_results[[length(all_results) + 1]] <<- row
}

# ==============================================================================
# Model 1: FE Logit (Conditional Logit) - Job Training data
# ==============================================================================
cat("=== Model 1: FE Logit (bife) - Job Training ===\n")

# Identify switchers
switch_info <- aggregate(employed ~ id, data = data_jt, FUN = function(x) length(unique(x)))
n_switchers <- sum(switch_info$employed > 1)
n_total     <- nrow(switch_info)
cat("Switchers:", n_switchers, "out of", n_total, "individuals\n")

fe_logit <- bife(employed ~ training + age + prior_wage | id,
                 data = data_jt, model = "logit")
s_fe <- summary(fe_logit)
print(s_fe)

cf_fe <- s_fe$cm
for (i in seq_len(nrow(cf_fe))) {
  add_result(
    "fe_logit_job_training",
    rownames(cf_fe)[i],
    cf_fe[i, 1], cf_fe[i, 2], cf_fe[i, 3], cf_fe[i, 4],
    log_likelihood = as.numeric(logLik(fe_logit)),
    n_obs = nrow(data_jt),
    n_switchers = n_switchers,
    n_dropped = n_total - n_switchers
  )
}

# ==============================================================================
# Model 2: FE Logit (Conditional Logit) - Labor Participation data
# ==============================================================================
cat("\n=== Model 2: FE Logit (bife) - Labor Participation ===\n")

switch_lp <- aggregate(lfp ~ id, data = data_lp, FUN = function(x) length(unique(x)))
n_switchers_lp <- sum(switch_lp$lfp > 1)
n_total_lp     <- nrow(switch_lp)
cat("Switchers:", n_switchers_lp, "out of", n_total_lp, "individuals\n")

fe_logit_lp <- bife(lfp ~ age + educ + kids + married | id,
                    data = data_lp, model = "logit")
s_fe_lp <- summary(fe_logit_lp)
print(s_fe_lp)

cf_fe_lp <- s_fe_lp$cm
for (i in seq_len(nrow(cf_fe_lp))) {
  add_result(
    "fe_logit_labor",
    rownames(cf_fe_lp)[i],
    cf_fe_lp[i, 1], cf_fe_lp[i, 2], cf_fe_lp[i, 3], cf_fe_lp[i, 4],
    log_likelihood = as.numeric(logLik(fe_logit_lp)),
    n_obs = nrow(data_lp),
    n_switchers = n_switchers_lp,
    n_dropped = n_total_lp - n_switchers_lp
  )
}

# ==============================================================================
# Model 3: Random Effects Probit - Labor Participation
# ==============================================================================
cat("\n=== Model 3: RE Probit (glmer) - Labor Participation ===\n")

re_probit <- glmer(lfp ~ age + educ + kids + married + (1 | id),
                   data = data_lp, family = binomial(link = "probit"),
                   nAGQ = 12,
                   control = glmerControl(optimizer = "bobyqa",
                                          optCtrl = list(maxfun = 100000)))
s_re_probit <- summary(re_probit)
print(s_re_probit)

# Extract RE-specific parameters
vc <- VarCorr(re_probit)
sigma_u <- as.numeric(attr(vc$id, "stddev"))
# For probit: Var(error) = 1, rho = sigma_u^2 / (sigma_u^2 + 1)
rho_probit <- sigma_u^2 / (sigma_u^2 + 1)

cat("\nRE Probit: sigma_u =", round(sigma_u, 4), ", rho =", round(rho_probit, 4), "\n")

cf_re_p <- coef(s_re_probit)
for (i in seq_len(nrow(cf_re_p))) {
  add_result(
    "re_probit_labor",
    rownames(cf_re_p)[i],
    cf_re_p[i, "Estimate"], cf_re_p[i, "Std. Error"],
    cf_re_p[i, "z value"], cf_re_p[i, "Pr(>|z|)"],
    log_likelihood = as.numeric(logLik(re_probit)),
    n_obs = nrow(data_lp),
    sigma_u = sigma_u, rho = rho_probit
  )
}

# ==============================================================================
# Model 4: Random Effects Logit - Labor Participation
# ==============================================================================
cat("\n=== Model 4: RE Logit (glmer) - Labor Participation ===\n")

re_logit <- glmer(lfp ~ age + educ + kids + married + (1 | id),
                  data = data_lp, family = binomial(link = "logit"),
                  nAGQ = 12,
                  control = glmerControl(optimizer = "bobyqa",
                                         optCtrl = list(maxfun = 100000)))
s_re_logit <- summary(re_logit)
print(s_re_logit)

vc_l <- VarCorr(re_logit)
sigma_u_l <- as.numeric(attr(vc_l$id, "stddev"))
# For logit: Var(error) = pi^2/3, rho = sigma_u^2 / (sigma_u^2 + pi^2/3)
rho_logit <- sigma_u_l^2 / (sigma_u_l^2 + pi^2 / 3)

cat("\nRE Logit: sigma_u =", round(sigma_u_l, 4), ", rho =", round(rho_logit, 4), "\n")

cf_re_l <- coef(s_re_logit)
for (i in seq_len(nrow(cf_re_l))) {
  add_result(
    "re_logit_labor",
    rownames(cf_re_l)[i],
    cf_re_l[i, "Estimate"], cf_re_l[i, "Std. Error"],
    cf_re_l[i, "z value"], cf_re_l[i, "Pr(>|z|)"],
    log_likelihood = as.numeric(logLik(re_logit)),
    n_obs = nrow(data_lp),
    sigma_u = sigma_u_l, rho = rho_logit
  )
}

# ==============================================================================
# Model 5: Pooled Probit baseline
# ==============================================================================
cat("\n=== Model 5: Pooled Probit (baseline) ===\n")
pooled_probit <- glm(lfp ~ age + educ + kids + married,
                     data = data_lp, family = binomial(link = "probit"))
summary(pooled_probit)

cf_pp <- coef(summary(pooled_probit))
for (i in seq_len(nrow(cf_pp))) {
  add_result(
    "pooled_probit_labor",
    rownames(cf_pp)[i],
    cf_pp[i, "Estimate"], cf_pp[i, "Std. Error"],
    cf_pp[i, "z value"], cf_pp[i, "Pr(>|z|)"],
    log_likelihood = as.numeric(logLik(pooled_probit)),
    n_obs = nrow(data_lp)
  )
}

# --- Combine and Save --------------------------------------------------------
results <- do.call(rbind, all_results)

write.csv(results, out_path, row.names = FALSE)
cat("\nResults saved to:", out_path, "\n")

cat("\n=== All Results ===\n")
print(results, digits = 6, row.names = FALSE)
