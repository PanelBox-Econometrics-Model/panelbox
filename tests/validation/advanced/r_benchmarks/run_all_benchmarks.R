# Run all R benchmarks
# Usage: Rscript run_all_benchmarks.R

cat("==========================================\n")
cat("Running all R benchmarks for PanelBox\n")
cat("==========================================\n\n")

# Get script directory
args <- commandArgs(trailingOnly = FALSE)
script_path <- dirname(sub("--file=", "", args[grep("--file=", args)]))
if (length(script_path) == 0) script_path <- "."
setwd(script_path)

# Install packages if needed
required_packages <- c("gmm", "sampleSelection", "mlogit")
for (pkg in required_packages) {
    if (!require(pkg, character.only=TRUE, quietly=TRUE)) {
        cat(paste("Installing", pkg, "...\n"))
        install.packages(pkg, repos="https://cloud.r-project.org")
    }
}

# Run each benchmark
benchmarks <- c(
    "cue_gmm_benchmark.R",
    "heckman_benchmark.R",
    "multinomial_benchmark.R",
    "conditional_logit_benchmark.R"
)

for (benchmark in benchmarks) {
    cat("\n------------------------------------------\n")
    cat(paste("Running:", benchmark, "\n"))
    cat("------------------------------------------\n")

    if (file.exists(benchmark)) {
        tryCatch({
            source(benchmark)
            cat(paste("\n[OK]", benchmark, "completed successfully\n"))
        }, error = function(e) {
            cat(paste("\n[ERROR]", benchmark, "failed:", e$message, "\n"))
        })
    } else {
        cat(paste("[SKIP] File not found:", benchmark, "\n"))
    }
}

cat("\n==========================================\n")
cat("All benchmarks completed\n")
cat("==========================================\n")
