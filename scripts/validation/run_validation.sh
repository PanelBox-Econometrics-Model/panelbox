#!/bin/bash
#
# Master script to run complete validation pipeline
#
# This script:
# 1. Generates test data and runs PanelBox tests
# 2. Runs equivalent tests in R
# 3. Compares results and generates validation report
#

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "PANELBOX VALIDATION PIPELINE"
echo "================================================================================"
echo ""
echo "This will:"
echo "  1. Generate test data with known properties"
echo "  2. Run PanelBox validation tests"
echo "  3. Run equivalent R tests (plm package)"
echo "  4. Compare results and generate report"
echo ""
echo "================================================================================"
echo ""

# Activate virtual environment
if [ -f "../../venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ../../venv/bin/activate
fi

# Step 1: Generate data and run PanelBox tests
echo ""
echo "STEP 1: Running PanelBox tests"
echo "--------------------------------------------------------------------------------"
python generate_test_data_and_run.py

if [ $? -ne 0 ]; then
    echo "ERROR: PanelBox tests failed!"
    exit 1
fi

# Step 2: Run R tests
echo ""
echo "STEP 2: Running R tests"
echo "--------------------------------------------------------------------------------"
Rscript run_r_tests.R

if [ $? -ne 0 ]; then
    echo "ERROR: R tests failed!"
    exit 1
fi

# Step 3: Compare results
echo ""
echo "STEP 3: Comparing results"
echo "--------------------------------------------------------------------------------"
python compare_results.py

if [ $? -ne 0 ]; then
    echo "ERROR: Comparison failed!"
    exit 1
fi

echo ""
echo "================================================================================"
echo "VALIDATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved in: output/"
echo ""
echo "Key files:"
echo "  - validation_report.txt           : Human-readable validation report"
echo "  - validation_comparisons.json     : Detailed comparison data"
echo "  - data_*.csv                      : Test datasets"
echo "  - panelbox_results_*.json         : PanelBox test results"
echo "  - r_results_*.json                : R test results"
echo ""
echo "To view the report:"
echo "  cat output/validation_report.txt"
echo ""
