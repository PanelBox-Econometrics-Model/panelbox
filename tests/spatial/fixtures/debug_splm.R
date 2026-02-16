library(splm)
library(spdep)
library(plm)

# Load test data
data <- read.csv("spatial_test_data.csv")
W <- as.matrix(read.csv("spatial_weights.csv", header=FALSE))
W_list <- mat2listw(W, style="W")
pdata <- pdata.frame(data, index=c("entity", "time"))

# Estimate SAR RE
sar_re <- spml(
  y ~ x1 + x2 + x3,
  data = pdata,
  listw = W_list,
  model = "random",
  lag = TRUE,
  spatial.error = "none",
  effect = "individual"
)

cat("\n=== Object Structure ===\n")
cat("Class:", class(sar_re), "\n")
cat("Names:", names(sar_re), "\n\n")

cat("=== Coefficients ===\n")
print(coef(sar_re))
cat("Names of coefs:", names(coef(sar_re)), "\n\n")

cat("=== Error components ===\n")
print(sar_re$errcomp)
cat("Class:", class(sar_re$errcomp), "\n")
cat("Names:", names(sar_re$errcomp), "\n\n")

cat("=== Other fields ===\n")
if(!is.null(sar_re$spat.coef)) {
  cat("spat.coef:", sar_re$spat.coef, "\n")
}
if(!is.null(sar_re$arcoef)) {
  cat("arcoef:", sar_re$arcoef, "\n")
}
if(!is.null(sar_re$logLik)) {
  cat("logLik:", sar_re$logLik, "\n")
}
