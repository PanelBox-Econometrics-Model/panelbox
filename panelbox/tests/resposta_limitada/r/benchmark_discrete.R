# ============================================
# Benchmark: Panel Discrete Choice Models
# ============================================

library(plm)
library(mfx)
library(jsonlite)

# Load test data
data <- read.csv("../data/binary_panel_test.csv")
data$entity <- as.factor(data$entity)
data$time <- as.factor(data$time)

# Convert to pdata.frame
pdata <- pdata.frame(data, index = c("entity", "time"))

# ---- Pooled Logit ----
cat("\n=== Pooled Logit ===\n")
pooled_logit <- glm(y ~ x1 + x2, data = data, family = binomial(link = "logit"))
summary(pooled_logit)

# Extract results
pooled_logit_results <- list(
  coef = coef(pooled_logit),
  se = sqrt(diag(vcov(pooled_logit))),
  loglik = logLik(pooled_logit)[1]
)

# Cluster-robust SE
library(sandwich)
library(lmtest)
pooled_logit_cluster <- coeftest(pooled_logit, vcov = vcovHC(pooled_logit, type = "HC1", cluster = "entity"))
pooled_logit_results$cluster_se <- pooled_logit_cluster[, "Std. Error"]

# Marginal effects (AME)
library(margins)
ame <- margins(pooled_logit)
pooled_logit_results$ame <- summary(ame)[, c("factor", "AME", "SE")]

# Save results
write_json(pooled_logit_results, "../r/results/pooled_logit_results.json", pretty = TRUE)

# ---- Pooled Probit ----
cat("\n=== Pooled Probit ===\n")
pooled_probit <- glm(y ~ x1 + x2, data = data, family = binomial(link = "probit"))
summary(pooled_probit)

pooled_probit_results <- list(
  coef = coef(pooled_probit),
  se = sqrt(diag(vcov(pooled_probit))),
  loglik = logLik(pooled_probit)[1]
)

# Marginal effects
ame_probit <- margins(pooled_probit)
pooled_probit_results$ame <- summary(ame_probit)[, c("factor", "AME", "SE")]

write_json(pooled_probit_results, "../r/results/pooled_probit_results.json", pretty = TRUE)

# ---- Fixed Effects Logit ----
# Note: FE Logit drops entities without variation
cat("\n=== FE Logit (Conditional Logit) ===\n")
# Use survival::clogit or plm package
library(survival)

# Prepare data for clogit
data_fe <- data[order(data$entity, data$time), ]
fe_logit <- clogit(y ~ x1 + x2 + strata(entity), data = data_fe)
summary(fe_logit)

fe_logit_results <- list(
  coef = coef(fe_logit),
  se = sqrt(diag(vcov(fe_logit))),
  loglik = fe_logit$loglik[2],
  n_dropped = sum(tapply(data$y, data$entity, function(x) length(unique(x)) == 1))
)

write_json(fe_logit_results, "../r/results/fe_logit_results.json", pretty = TRUE)

cat("\nResults saved to r/results/\n")
