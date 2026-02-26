# Script to generate reference results from R sfaR package
# This generates benchmark results for panel SFA models

# Install sfaR if needed
if (!require("sfaR")) {
  install.packages("sfaR")
}

library(sfaR)
library(readr)

# Session info for reproducibility
sink("r_sfaR_session_info.txt")
sessionInfo()
sink()

cat("Generating R sfaR reference results...\n")

# ============================================================================
# Load Rice Production Panel Data
# ============================================================================

# Use the dataset from frontier package
library(frontier)
data(riceProdPhil)

# Prepare panel structure
riceProdPhil$FMERCODE <- as.integer(riceProdPhil$FMERCODE)
riceProdPhil$YEARDUM <- as.integer(riceProdPhil$YEARDUM)

# Save for Python
write_csv(riceProdPhil, "data/riceProdPhil_panel.csv")

cat("Panel Dataset:\n")
cat("  Firms:", length(unique(riceProdPhil$FMERCODE)), "\n")
cat("  Time periods:", length(unique(riceProdPhil$YEARDUM)), "\n")
cat("  Total observations:", nrow(riceProdPhil), "\n\n")

# ============================================================================
# 1. Pitt-Lee (1981) - Time-Invariant Inefficiency
# ============================================================================

cat("Estimating Pitt-Lee (1981) model...\n")

sfa_pittlee <- sfacross(
  formula = log(PROD) ~ log(AREA) + log(LABOR) + log(NPK) + log(OTHER),
  data = riceProdPhil,
  udist = "hnormal",
  start = NULL
)

# Extract results
params_pl <- summary(sfa_pittlee, grad = FALSE)$coefficients
params_pl_df <- data.frame(
  parameter = rownames(params_pl),
  estimate = params_pl[, "Estimate"],
  se = params_pl[, "Std. Error"],
  tvalue = params_pl[, "t value"],
  pvalue = params_pl[, "Pr(>|t|)"]
)

# Efficiency estimates
eff_pl <- efficiencies(sfa_pittlee, level = 0.95)
eff_pl_df <- data.frame(
  firm_id = 1:nrow(eff_pl),
  efficiency = eff_pl$teBC,
  te_jlms = eff_pl$teJLMS,
  ci_lower = eff_pl$teLB,
  ci_upper = eff_pl$teUB
)

# Log-likelihood
loglik_pl <- data.frame(
  loglik = logLik(sfa_pittlee),
  aic = AIC(sfa_pittlee),
  bic = BIC(sfa_pittlee),
  nobs = nobs(sfa_pittlee)
)

write_csv(params_pl_df, "data/r_sfaR_pittlee_params.csv")
write_csv(eff_pl_df, "data/r_sfaR_pittlee_efficiencies.csv")
write_csv(loglik_pl, "data/r_sfaR_pittlee_loglik.csv")

cat("  Log-likelihood:", loglik_pl$loglik, "\n")
cat("  Mean efficiency (BC):", mean(eff_pl$teBC), "\n\n")

# ============================================================================
# 2. Battese-Coelli (1992) - Time-Varying Inefficiency
# ============================================================================

cat("Estimating Battese-Coelli (1992) model...\n")

sfa_bc92 <- sfapanel(
  formula = log(PROD) ~ log(AREA) + log(LABOR) + log(NPK) + log(OTHER),
  data = riceProdPhil,
  ineffDecrease = TRUE,
  timeEffect = TRUE,
  printInfo = FALSE
)

# Note: sfapanel from sfaR uses different interface
# We'll use the frontier package for BC92 if sfaR doesn't support it directly

# Alternative: use frontier package for BC92
library(frontier)

sfa_bc92_frontier <- sfa(
  log(PROD) ~ log(AREA) + log(LABOR) + log(NPK) + log(OTHER),
  data = riceProdPhil,
  ineffDecrease = TRUE,
  truncNorm = FALSE,
  timeEffect = TRUE  # BC92 time effect
)

params_bc92 <- data.frame(
  parameter = names(coef(sfa_bc92_frontier)),
  estimate = as.numeric(coef(sfa_bc92_frontier)),
  se = sqrt(diag(vcov(sfa_bc92_frontier))),
  tvalue = as.numeric(coef(sfa_bc92_frontier)) / sqrt(diag(vcov(sfa_bc92_frontier)))
)

# Variance components
sigma_v_bc92 <- sqrt(sfa_bc92_frontier$sigmaSqV)
sigma_u_bc92 <- sqrt(sfa_bc92_frontier$sigmaSqU)
gamma_bc92 <- sfa_bc92_frontier$sigmaSqU / (sfa_bc92_frontier$sigmaSqV + sfa_bc92_frontier$sigmaSqU)

variance_bc92 <- data.frame(
  parameter = c('sigma_v', 'sigma_u', 'gamma'),
  estimate = c(sigma_v_bc92, sigma_u_bc92, gamma_bc92),
  se = NA,
  tvalue = NA
)

params_bc92 <- rbind(params_bc92, variance_bc92)

eff_bc92 <- efficiencies(sfa_bc92_frontier)
eff_bc92_df <- data.frame(
  firm_id = 1:length(eff_bc92),
  efficiency = as.numeric(eff_bc92)
)

loglik_bc92 <- data.frame(
  loglik = as.numeric(logLik(sfa_bc92_frontier)),
  aic = AIC(sfa_bc92_frontier),
  bic = BIC(sfa_bc92_frontier)
)

write_csv(params_bc92, "data/r_sfaR_bc92_params.csv")
write_csv(eff_bc92_df, "data/r_sfaR_bc92_efficiencies.csv")
write_csv(loglik_bc92, "data/r_sfaR_bc92_loglik.csv")

cat("  Log-likelihood:", loglik_bc92$loglik, "\n")
cat("  Mean efficiency:", mean(eff_bc92), "\n\n")

# ============================================================================
# 3. True Random Effects (Greene 2005)
# ============================================================================

cat("Estimating True Random Effects model...\n")

# sfaR supports TRE
sfa_tre <- sfapanel(
  formula = log(PROD) ~ log(AREA) + log(LABOR) + log(NPK) + log(OTHER),
  data = riceProdPhil,
  uhet = ~ 1,  # No heterogeneity in u for simplicity
  vhet = ~ 1,  # No heterogeneity in v
  S = -1,      # Production frontier
  udist = "hnormal",
  method = "tre"
)

params_tre <- summary(sfa_tre)$coefficients
params_tre_df <- data.frame(
  parameter = rownames(params_tre),
  estimate = params_tre[, "Estimate"],
  se = params_tre[, "Std. Error"],
  tvalue = params_tre[, "t value"],
  pvalue = params_tre[, "Pr(>|t|)"]
)

eff_tre <- efficiencies(sfa_tre, level = 0.95)
eff_tre_df <- data.frame(
  firm_id = 1:nrow(eff_tre),
  efficiency = eff_tre$teBC,
  te_jlms = eff_tre$teJLMS,
  ci_lower = eff_tre$teLB,
  ci_upper = eff_tre$teUB
)

loglik_tre <- data.frame(
  loglik = logLik(sfa_tre),
  aic = AIC(sfa_tre),
  bic = BIC(sfa_tre),
  nobs = nobs(sfa_tre)
)

write_csv(params_tre_df, "data/r_sfaR_tre_params.csv")
write_csv(eff_tre_df, "data/r_sfaR_tre_efficiencies.csv")
write_csv(loglik_tre, "data/r_sfaR_tre_loglik.csv")

cat("  Log-likelihood:", loglik_tre$loglik, "\n")
cat("  Mean efficiency (BC):", mean(eff_tre$teBC), "\n\n")

# ============================================================================
# 4. True Fixed Effects (Greene 2005)
# ============================================================================

cat("Estimating True Fixed Effects model...\n")

sfa_tfe <- sfapanel(
  formula = log(PROD) ~ log(AREA) + log(LABOR) + log(NPK) + log(OTHER),
  data = riceProdPhil,
  uhet = ~ 1,
  vhet = ~ 1,
  S = -1,
  udist = "hnormal",
  method = "tfe"
)

params_tfe <- summary(sfa_tfe)$coefficients
params_tfe_df <- data.frame(
  parameter = rownames(params_tfe),
  estimate = params_tfe[, "Estimate"],
  se = params_tfe[, "Std. Error"],
  tvalue = params_tfe[, "t value"],
  pvalue = params_tfe[, "Pr(>|t|)"]
)

eff_tfe <- efficiencies(sfa_tfe, level = 0.95)
eff_tfe_df <- data.frame(
  firm_id = 1:nrow(eff_tfe),
  efficiency = eff_tfe$teBC,
  te_jlms = eff_tfe$teJLMS,
  ci_lower = eff_tfe$teLB,
  ci_upper = eff_tfe$teUB
)

loglik_tfe <- data.frame(
  loglik = logLik(sfa_tfe),
  aic = AIC(sfa_tfe),
  bic = BIC(sfa_tfe),
  nobs = nobs(sfa_tfe)
)

write_csv(params_tfe_df, "data/r_sfaR_tfe_params.csv")
write_csv(eff_tfe_df, "data/r_sfaR_tfe_efficiencies.csv")
write_csv(loglik_tfe, "data/r_sfaR_tfe_loglik.csv")

cat("  Log-likelihood:", loglik_tfe$loglik, "\n")
cat("  Mean efficiency (BC):", mean(eff_tfe$teBC), "\n\n")

cat("R sfaR reference results generated successfully!\n")
cat("Files saved in data/ directory\n")
