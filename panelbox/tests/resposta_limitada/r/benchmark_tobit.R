# ============================================
# Benchmark: Panel Tobit Models
# ============================================

library(censReg)
library(plm)
library(jsonlite)

# Load test data
data <- read.csv("../data/censored_panel_test.csv")

# ---- Pooled Tobit ----
cat("\n=== Pooled Tobit ===\n")
pooled_tobit <- censReg(y ~ x1 + x2, left = 0, data = data)
summary(pooled_tobit)

pooled_tobit_results <- list(
  coef = coef(pooled_tobit),
  se = sqrt(diag(vcov(pooled_tobit))),
  sigma = pooled_tobit$estimate["sigma"],
  loglik = logLik(pooled_tobit)[1]
)

# Marginal effects
# Unconditional: dE[y|X]/dx
marg_uncond <- margEff(pooled_tobit)
pooled_tobit_results$me_unconditional <- summary(marg_uncond)

write_json(pooled_tobit_results, "../r/results/pooled_tobit_results.json", pretty = TRUE)

# ---- Random Effects Tobit ----
# Note: RE Tobit is more complex in R, may need custom implementation
# or use packages like plm with appropriate adjustments

cat("\nPooled Tobit results saved to r/results/\n")
