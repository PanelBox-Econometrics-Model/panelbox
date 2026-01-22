/*==============================================================================
Master Script: Run All Validation Scripts
==============================================================================

Executes all 5 validation scripts in sequence
Generates comparison-ready output files

Usage:
  cd validation/stata
  stata -b do run_all.do

Output:
  All results saved in validation/results/stata/

==============================================================================*/

clear all
set more off

display ""
display "=========================================="
display "PanelBox GMM Validation Suite"
display "Stata xtabond2 Reference Implementation"
display "=========================================="
display ""

* Check Stata version
display "Stata version: " c(stata_version)
display "Current directory: " c(pwd)
display ""

* Check if xtabond2 is installed
capture which xtabond2
if _rc != 0 {
    display as error "ERROR: xtabond2 not installed"
    display as error "Install with: ssc install xtabond2"
    exit 111
}

display "xtabond2 is installed ✓"
display ""

* Verify data file exists
capture confirm file "../data/abdata.dta"
if _rc != 0 {
    display as error "ERROR: Dataset not found: ../data/abdata.dta"
    exit 601
}

display "Dataset found ✓"
display ""

/*------------------------------------------------------------------------------
Run validation scripts
------------------------------------------------------------------------------*/

local scripts "01_difference_gmm_basic 02_difference_gmm_collapsed 03_system_gmm_basic 04_system_gmm_collapsed 05_with_time_dummies"

local script_num = 1
foreach script of local scripts {
    display ""
    display "=========================================="
    display "Running: `script'.do (`script_num'/5)"
    display "=========================================="
    display ""

    * Run script
    capture noisily do `script'.do

    if _rc != 0 {
        display as error "ERROR in `script'.do"
        display as error "Return code: " _rc
        exit _rc
    }

    display ""
    display "✓ `script'.do completed successfully"
    display ""

    local script_num = `script_num' + 1
}

/*------------------------------------------------------------------------------
Summary
------------------------------------------------------------------------------*/

display ""
display "=========================================="
display "Validation Suite Complete"
display "=========================================="
display ""

display "Results saved in: validation/results/stata/"
display ""

display "Files generated:"
display "  01_difference_gmm_basic.txt"
display "  02_difference_gmm_collapsed.txt"
display "  03_system_gmm_basic.txt"
display "  04_system_gmm_collapsed.txt"
display "  05_with_time_dummies.txt"
display ""

display "Next steps:"
display "  1. Run Python replication scripts"
display "  2. Compare results with validation/python/compare_results.py"
display ""

/* End of master script */
