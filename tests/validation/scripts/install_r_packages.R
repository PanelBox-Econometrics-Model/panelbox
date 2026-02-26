# Install required R packages for Panel VAR validation

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# List of required packages
required_packages <- c(
  "jsonlite",  # For JSON export
  "plm",       # Panel data econometrics
  "vars",      # VAR modeling (for IRF, FEVD)
  "gmm"        # GMM estimation for CUE-GMM validation
)

# Install packages if not already installed
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, dependencies = TRUE)
  } else {
    cat(sprintf("%s is already installed.\n", pkg))
  }
}

# Try to install pvar if available (may not be on CRAN)
# Note: pvar package may need to be installed from source or GitHub
cat("\nNote: The 'pvar' package may need to be installed manually from GitHub.\n")
cat("If needed, run: devtools::install_github('rforge/pvar')\n")

cat("\n✓ R package installation complete!\n")
