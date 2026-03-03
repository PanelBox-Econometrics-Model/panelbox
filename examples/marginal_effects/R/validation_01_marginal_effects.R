# ==============================================================================
# Validation Script: Marginal Effects
# ==============================================================================
# Purpose: Reproduce PanelBox marginal effects using R's margins package
# Notebooks: 01_me_fundamentals.ipynb, 02_discrete_me_complete.ipynb
# Models: OLS (Grunfeld), Logit/Probit (Mroz), Poisson (patents)
# ==============================================================================

library(margins)

cat("=" , rep("=", 69), "\n", sep = "")
cat("Marginal Effects Validation (R margins package)\n")
cat("=" , rep("=", 69), "\n", sep = "")

# ==============================================================================
# PART 1: OLS Marginal Effects (trivial - coefficients ARE the marginal effects)
# ==============================================================================
cat("\n", rep("=", 70), "\n", sep = "")
cat("PART 1: OLS MARGINAL EFFECTS (Grunfeld data)\n")
cat(rep("=", 70), "\n", sep = "")

grunfeld_path <- "/home/guhaase/projetos/panelbox/examples/datasets/panel/grunfeld.csv"
grunfeld <- read.csv(grunfeld_path)
cat(sprintf("Grunfeld: %d obs, %d vars\n", nrow(grunfeld), ncol(grunfeld)))

# Pooled OLS: invest ~ value + capital
ols_model <- lm(invest ~ value + capital, data = grunfeld)
cat("\nOLS Model:\n")
print(summary(ols_model))

# Marginal effects for OLS are the coefficients themselves
ols_ame <- margins(ols_model)
cat("\nOLS Average Marginal Effects (AME):\n")
print(summary(ols_ame))

# MEM (at means) - for OLS, same as AME
ols_mem <- margins(ols_model, at = list(
  value = mean(grunfeld$value),
  capital = mean(grunfeld$capital)
))
cat("\nOLS Marginal Effects at Means (MEM):\n")
print(summary(ols_mem))

# ==============================================================================
# PART 2: LOGIT Marginal Effects (Mroz data)
# ==============================================================================
cat("\n", rep("=", 70), "\n", sep = "")
cat("PART 2: LOGIT MARGINAL EFFECTS (Mroz labor participation)\n")
cat(rep("=", 70), "\n", sep = "")

mroz_path <- "/home/guhaase/projetos/panelbox/examples/marginal_effects/data/mroz.csv"
mroz <- read.csv(mroz_path)
cat(sprintf("Mroz: %d obs, %d vars\n", nrow(mroz), ncol(mroz)))
cat("Variables:", paste(names(mroz), collapse = ", "), "\n")

# Binary logit: inlf ~ educ + age + kidslt6 + kidsge6 + nwifeinc
logit_model <- glm(inlf ~ educ + age + kidslt6 + kidsge6 + nwifeinc,
                   data = mroz,
                   family = binomial(link = "logit"))

cat("\nLogit Model:\n")
print(summary(logit_model))

# AME for logit
logit_ame <- margins(logit_model)
cat("\nLogit Average Marginal Effects (AME):\n")
logit_ame_summary <- summary(logit_ame)
print(logit_ame_summary)

# MEM for logit
logit_mem <- margins(logit_model, at = list(
  educ = mean(mroz$educ),
  age = mean(mroz$age),
  kidslt6 = mean(mroz$kidslt6),
  kidsge6 = mean(mroz$kidsge6),
  nwifeinc = mean(mroz$nwifeinc)
))
cat("\nLogit Marginal Effects at Means (MEM):\n")
logit_mem_summary <- summary(logit_mem)
print(logit_mem_summary)

# Compare coefficients vs AME
cat("\nComparison: Logit Coefficients vs AME:\n")
cat(sprintf("  %-10s %12s %12s %12s\n", "Variable", "Coef", "AME", "Ratio(C/AME)"))
for (var in c("educ", "age", "kidslt6", "kidsge6", "nwifeinc")) {
  cf <- coef(logit_model)[var]
  am <- logit_ame_summary$AME[logit_ame_summary$factor == var]
  cat(sprintf("  %-10s %12.6f %12.6f %12.4f\n", var, cf, am, cf / am))
}
cat("\nNote: For logit, AME != coefficient. AME = mean(Lambda(Xb)*(1-Lambda(Xb)) * beta_k)\n")

# ==============================================================================
# PART 3: PROBIT Marginal Effects (Mroz data)
# ==============================================================================
cat("\n", rep("=", 70), "\n", sep = "")
cat("PART 3: PROBIT MARGINAL EFFECTS (Mroz labor participation)\n")
cat(rep("=", 70), "\n", sep = "")

probit_model <- glm(inlf ~ educ + age + kidslt6 + kidsge6 + nwifeinc,
                    data = mroz,
                    family = binomial(link = "probit"))

cat("\nProbit Model:\n")
print(summary(probit_model))

# AME for probit
probit_ame <- margins(probit_model)
cat("\nProbit Average Marginal Effects (AME):\n")
probit_ame_summary <- summary(probit_ame)
print(probit_ame_summary)

# MEM for probit
probit_mem <- margins(probit_model, at = list(
  educ = mean(mroz$educ),
  age = mean(mroz$age),
  kidslt6 = mean(mroz$kidslt6),
  kidsge6 = mean(mroz$kidsge6),
  nwifeinc = mean(mroz$nwifeinc)
))
cat("\nProbit Marginal Effects at Means (MEM):\n")
probit_mem_summary <- summary(probit_mem)
print(probit_mem_summary)

# Compare Logit vs Probit AME
cat("\nComparison: Logit AME vs Probit AME:\n")
cat(sprintf("  %-10s %12s %12s %12s\n", "Variable", "Logit AME", "Probit AME", "Difference"))
for (var in c("educ", "age", "kidslt6", "kidsge6", "nwifeinc")) {
  logit_am <- logit_ame_summary$AME[logit_ame_summary$factor == var]
  probit_am <- probit_ame_summary$AME[probit_ame_summary$factor == var]
  cat(sprintf("  %-10s %12.6f %12.6f %12.6f\n", var, logit_am, probit_am, logit_am - probit_am))
}

# ==============================================================================
# PART 4: POISSON Marginal Effects (Patents data)
# ==============================================================================
cat("\n", rep("=", 70), "\n", sep = "")
cat("PART 4: POISSON MARGINAL EFFECTS (Patents data)\n")
cat(rep("=", 70), "\n", sep = "")

patents_path <- "/home/guhaase/projetos/panelbox/examples/marginal_effects/data/patents.csv"
patents <- read.csv(patents_path)
cat(sprintf("Patents: %d obs, %d vars\n", nrow(patents), ncol(patents)))

# Poisson regression: patents ~ log_rnd + log_sales + log_capital
poisson_model <- glm(patents ~ log_rnd + log_sales + log_capital,
                     data = patents,
                     family = poisson(link = "log"))

cat("\nPoisson Model:\n")
print(summary(poisson_model))

# AME for Poisson
poisson_ame <- margins(poisson_model)
cat("\nPoisson Average Marginal Effects (AME):\n")
poisson_ame_summary <- summary(poisson_ame)
print(poisson_ame_summary)

# MEM for Poisson
poisson_mem <- margins(poisson_model, at = list(
  log_rnd = mean(patents$log_rnd),
  log_sales = mean(patents$log_sales),
  log_capital = mean(patents$log_capital)
))
cat("\nPoisson Marginal Effects at Means (MEM):\n")
poisson_mem_summary <- summary(poisson_mem)
print(poisson_mem_summary)

# Incidence Rate Ratios
cat("\nIncidence Rate Ratios (IRR = exp(beta)):\n")
irr <- exp(coef(poisson_model))
for (var in c("log_rnd", "log_sales", "log_capital")) {
  cat(sprintf("  %-12s: IRR = %.4f (a 1-unit increase multiplies expected count by %.4f)\n",
              var, irr[var], irr[var]))
}

# Compare coefficients vs AME for Poisson
cat("\nComparison: Poisson Coefficients vs AME:\n")
cat(sprintf("  %-12s %12s %12s %12s\n", "Variable", "Coef", "AME", "Ratio(C/AME)"))
for (var in c("log_rnd", "log_sales", "log_capital")) {
  cf <- coef(poisson_model)[var]
  am <- poisson_ame_summary$AME[poisson_ame_summary$factor == var]
  cat(sprintf("  %-12s %12.6f %12.6f %12.4f\n", var, cf, am, cf / am))
}
cat("\nNote: For Poisson, AME = mean(exp(Xb) * beta_k), which varies across observations\n")

# ==============================================================================
# 5. Save Results to CSV
# ==============================================================================
cat("\n", rep("=", 70), "\n", sep = "")
cat("SAVING RESULTS\n")
cat(rep("=", 70), "\n", sep = "")

output_dir <- "/home/guhaase/projetos/panelbox/examples/marginal_effects/R"

# Build results data frame with AME and MEM for all models
build_me_results <- function(model_name, ame_summary, mem_summary) {
  ame_rows <- data.frame(
    model_name = model_name,
    variable = ame_summary$factor,
    ame = ame_summary$AME,
    ame_std_error = ame_summary$SE,
    ame_z_statistic = ame_summary$z,
    ame_p_value = ame_summary$p,
    ame_lower = ame_summary$lower,
    ame_upper = ame_summary$upper,
    stringsAsFactors = FALSE
  )

  mem_rows <- data.frame(
    variable = mem_summary$factor,
    mem = mem_summary$AME,
    mem_std_error = mem_summary$SE,
    mem_z_statistic = mem_summary$z,
    mem_p_value = mem_summary$p,
    mem_lower = mem_summary$lower,
    mem_upper = mem_summary$upper,
    stringsAsFactors = FALSE
  )

  # Merge AME and MEM
  merged <- merge(ame_rows, mem_rows, by = "variable", all = TRUE)
  return(merged)
}

# OLS results
ols_ame_s <- summary(margins(ols_model))
ols_mem_s <- summary(margins(ols_model, at = list(
  value = mean(grunfeld$value),
  capital = mean(grunfeld$capital)
)))
res_ols <- build_me_results("ols_pooled", ols_ame_s, ols_mem_s)

# Logit results
res_logit <- build_me_results("logit", logit_ame_summary, logit_mem_summary)

# Probit results
res_probit <- build_me_results("probit", probit_ame_summary, probit_mem_summary)

# Poisson results
res_poisson <- build_me_results("poisson", poisson_ame_summary, poisson_mem_summary)

# Combine all
all_results <- rbind(res_ols, res_logit, res_probit, res_poisson)

output_file <- file.path(output_dir, "results_marginal_effects.csv")
write.csv(all_results, output_file, row.names = FALSE)
cat(sprintf("Results saved to: %s\n", output_file))
cat(sprintf("Total rows: %d\n", nrow(all_results)))

# Print final summary table
cat("\n", rep("=", 70), "\n", sep = "")
cat("FINAL SUMMARY: ALL MARGINAL EFFECTS\n")
cat(rep("=", 70), "\n", sep = "")
print(all_results[, c("model_name", "variable", "ame", "ame_std_error",
                       "ame_p_value", "mem", "mem_std_error")])

cat("\n", rep("=", 70), "\n", sep = "")
cat("MARGINAL EFFECTS VALIDATION COMPLETE\n")
cat(rep("=", 70), "\n", sep = "")
