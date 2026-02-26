#!/usr/bin/env Rscript
# Test script to inspect the structure of frontier::sfa results

library(frontier)

# Load data
data(riceProdPhil)

# Aggregate to cross-section
rice_cs <- aggregate(
    cbind(PROD, AREA, LABOR, NPK, OTHER) ~ FMERCODE,
    data = riceProdPhil,
    FUN = mean
)

# Transform to log
rice_cs$log_output <- log(rice_cs$PROD)
rice_cs$log_area <- log(rice_cs$AREA)
rice_cs$log_labor <- log(rice_cs$LABOR)
rice_cs$log_npk <- log(rice_cs$NPK)
rice_cs$log_other <- log(rice_cs$OTHER)

# Estimate SFA
sfa_hn <- sfa(
    log_output ~ log_area + log_labor + log_npk + log_other,
    data = rice_cs,
    ineffDecrease = TRUE
)

# Print structure
cat("=== Structure of sfa object ===\n")
print(names(sfa_hn))

cat("\n=== Coefficients ===\n")
print(coef(sfa_hn))

cat("\n=== Variance components ===\n")
cat("sigmaSq:", sfa_hn$sigmaSq, "\n")
cat("sigmaSqV:", if("sigmaSqV" %in% names(sfa_hn)) sfa_hn$sigmaSqV else "NOT FOUND", "\n")
cat("sigmaSqU:", if("sigmaSqU" %in% names(sfa_hn)) sfa_hn$sigmaSqU else "NOT FOUND", "\n")
cat("gamma:", if("gamma" %in% names(sfa_hn)) sfa_hn$gamma else "NOT FOUND", "\n")
cat("gammaParm:", if("gammaParm" %in% names(sfa_hn)) sfa_hn$gammaParm else "NOT FOUND", "\n")

cat("\n=== Log-likelihood ===\n")
cat("logLik:", logLik(sfa_hn), "\n")

cat("\n=== Efficiencies (first 5) ===\n")
eff <- efficiencies(sfa_hn)
print(head(eff, 5))
