# CUE-GMM Benchmark usando pacote gmm
# Instalar: install.packages("gmm")

library(gmm)

# Configurar seed para reprodutibilidade
set.seed(42)

# Gerar dados
n <- 500
z1 <- rnorm(n)
z2 <- rnorm(n)
z3 <- rnorm(n)
x1 <- 0.5*z1 + 0.3*z2 + rnorm(n, sd=0.5)
x2 <- 0.4*z2 + 0.3*z3 + rnorm(n, sd=0.5)
y <- 1 + 2*x1 + 1.5*x2 + rnorm(n)

# Moment conditions
g <- function(theta, x) {
    y <- x[, 1]
    X <- cbind(1, x[, 2:3])
    Z <- cbind(1, x[, 4:6])
    e <- y - X %*% theta
    return(Z * as.vector(e))
}

# Data matrix
dat <- cbind(y, x1, x2, z1, z2, z3)

# CUE estimation
cat("Estimating CUE-GMM...\n")
result_cue <- gmm(g, dat, c(0, 0, 0), type="cue",
                  vcov="HAC", kernel="Bartlett")

# Print results
cat("\n=== CUE-GMM Results ===\n")
cat("Parameters:\n")
print(coef(result_cue))
cat("\nStandard Errors:\n")
print(sqrt(diag(vcov(result_cue))))
cat("\nJ-statistic:\n")
print(result_cue$objective * n)

# Save results to CSV for Python comparison
results_df <- data.frame(
    param = c("const", "x1", "x2"),
    coef = coef(result_cue),
    se = sqrt(diag(vcov(result_cue)))
)
write.csv(results_df, "cue_gmm_results.csv", row.names=FALSE)

# Save data
data_df <- data.frame(y=y, x1=x1, x2=x2, z1=z1, z2=z2, z3=z3)
write.csv(data_df, "cue_gmm_data.csv", row.names=FALSE)

cat("\nResults saved to cue_gmm_results.csv\n")
cat("Data saved to cue_gmm_data.csv\n")
