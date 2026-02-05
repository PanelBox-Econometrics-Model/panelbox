"""
Generate Consolidated Benchmark Report

This script runs all benchmark tests and generates a consolidated report
comparing PanelBox with Stata and R implementations.

Usage:
    python3 generate_benchmark_report.py [--stata] [--r] [--all]

Options:
    --stata    Run Stata benchmarks only
    --r        Run R benchmarks only
    --all      Run all benchmarks (default)
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class BenchmarkRunner:
    """Run benchmark tests and collect results."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.stata_dir = base_dir / "stata_comparison"
        self.r_dir = base_dir / "r_comparison"
        self.results_dir = base_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "stata": {},
            "r": {},
            "summary": {},
        }

    def run_stata_benchmarks(self) -> Dict:
        """Run all Stata comparison tests."""
        print("=" * 80)
        print("RUNNING STATA BENCHMARKS")
        print("=" * 80)

        tests = [
            ("Pooled OLS", "test_pooled_vs_stata.py"),
            ("Fixed Effects", "test_fe_vs_stata.py"),
            ("Random Effects", "test_re_vs_stata.py"),
            ("Difference GMM", "test_diff_gmm_vs_stata.py"),
            ("System GMM", "test_sys_gmm_vs_stata.py"),
        ]

        results = {}

        for name, script in tests:
            print(f"\n{'=' * 80}")
            print(f"Running: {name}")
            print(f"{'=' * 80}\n")

            script_path = self.stata_dir / script

            if not script_path.exists():
                print(f"✗ SKIPPED: {script} not found")
                results[name] = {"status": "skipped", "reason": "Script not found"}
                continue

            try:
                # Run test and capture output
                result = subprocess.run(
                    ["python3", str(script_path)],
                    cwd=str(self.stata_dir),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                # Parse output
                output = result.stdout
                passed = "BENCHMARK PASSED" in output

                results[name] = {
                    "status": "passed" if passed else "failed",
                    "exit_code": result.returncode,
                    "output_length": len(output),
                    "has_placeholders": "PLACEHOLDER" in output,
                }

                if passed:
                    print(f"✓ PASSED: {name}")
                else:
                    print(f"✗ FAILED: {name}")
                    if "PLACEHOLDER" in output:
                        print(f"  Note: Contains placeholder values - need Stata run")

            except subprocess.TimeoutExpired:
                print(f"✗ TIMEOUT: {name}")
                results[name] = {"status": "timeout", "reason": "Execution timeout (60s)"}
            except Exception as e:
                print(f"✗ ERROR: {name} - {e}")
                results[name] = {"status": "error", "error": str(e)}

        return results

    def run_r_benchmarks(self) -> Dict:
        """Run all R comparison tests (placeholder)."""
        print("\n" + "=" * 80)
        print("RUNNING R BENCHMARKS")
        print("=" * 80)
        print("\nR benchmarks not yet implemented.")
        print("Planned: Pooled OLS, FE, RE, GMM comparisons with plm package")

        return {
            "status": "not_implemented",
            "message": "R benchmarks planned for future implementation",
        }

    def generate_summary(self):
        """Generate summary statistics."""
        stata_results = self.results["stata"]

        total = len(stata_results)
        passed = sum(1 for r in stata_results.values() if r.get("status") == "passed")
        failed = sum(1 for r in stata_results.values() if r.get("status") == "failed")
        skipped = sum(1 for r in stata_results.values() if r.get("status") == "skipped")
        errors = sum(1 for r in stata_results.values() if r.get("status") == "error")

        placeholders = sum(1 for r in stata_results.values() if r.get("has_placeholders", False))

        self.results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "needs_stata_run": placeholders > 0,
            "placeholder_count": placeholders,
        }

    def save_json(self):
        """Save results to JSON."""
        output_file = self.results_dir / "benchmark_results.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {output_file}")

    def generate_markdown_report(self):
        """Generate markdown report."""
        output_file = self.results_dir / "BENCHMARK_REPORT.md"

        report = self._build_markdown()

        with open(output_file, "w") as f:
            f.write(report)

        print(f"✓ Report saved to: {output_file}")

    def _build_markdown(self) -> str:
        """Build markdown report content."""
        summary = self.results["summary"]
        stata = self.results["stata"]

        report = f"""# PanelBox Benchmark Report

**Generated**: {self.results['timestamp']}
**Status**: {'🟢 Ready for validation' if not summary['needs_stata_run'] else '🟡 Awaiting Stata results'}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Tests | {summary['total_tests']} |
| Passed | {summary['passed']} ✓ |
| Failed | {summary['failed']} ✗ |
| Skipped | {summary['skipped']} ⊘ |
| Errors | {summary['errors']} ⚠ |
| **Pass Rate** | **{summary['pass_rate']:.1f}%** |

"""

        if summary["needs_stata_run"]:
            report += f"""
### ⚠️ Action Required

**{summary['placeholder_count']} tests contain placeholder values.**

To complete validation:
1. Run Stata scripts: `cd tests/benchmarks/stata_comparison && stata -b do *.do`
2. Copy values from `.log` files to Python test scripts
3. Re-run this report generator

"""

        report += """
---

## Stata Benchmarks

### Results by Model

| Model | Status | Notes |
|-------|--------|-------|
"""

        for name, result in stata.items():
            status = result.get("status", "unknown")

            if status == "passed":
                icon = "✓"
                status_text = "PASSED"
            elif status == "failed":
                icon = "✗"
                status_text = "FAILED"
            elif status == "skipped":
                icon = "⊘"
                status_text = "SKIPPED"
            else:
                icon = "⚠"
                status_text = "ERROR"

            notes = []
            if result.get("has_placeholders"):
                notes.append("Needs Stata run")
            if result.get("reason"):
                notes.append(result["reason"])

            notes_str = ", ".join(notes) if notes else "—"

            report += f"| {name} | {icon} {status_text} | {notes_str} |\n"

        report += """
---

## R Benchmarks

Status: Not yet implemented

Planned comparisons with R `plm` package:
- Pooled OLS
- Fixed Effects (within estimator)
- Random Effects
- GMM (pgmm function)

---

## Methodology

### Tolerance Levels

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| Coefficients (Static) | < 1e-6 | Numerical precision |
| Standard Errors (Static) | < 1e-6 | Numerical precision |
| Coefficients (GMM) | < 1e-3 | Algorithm differences |
| Standard Errors (GMM) | < 1e-3 | Windmeijer correction |
| Test Statistics | < 1e-3 | Chi-square/F approximations |

### Comparison Approach

1. **Exact Replication**: Use identical data (Grunfeld dataset)
2. **Same Specification**: Match model options precisely
3. **Numerical Comparison**: Compare coefficients, SE, tests
4. **Document Differences**: Any deviation > tolerance is investigated

---

## Next Steps

"""

        if summary["passed"] == summary["total_tests"]:
            report += """
✓ **All benchmarks passed!**

The PanelBox implementation matches Stata/R within tolerance.

**Recommended actions**:
1. Update main documentation with benchmark results
2. Include in academic papers as validation
3. Proceed with release
"""
        elif summary["needs_stata_run"]:
            report += """
**Immediate**:
1. Install Stata 15+ and xtabond2
2. Run all `.do` scripts in `stata_comparison/`
3. Update placeholder values in Python tests
4. Re-run: `python3 generate_benchmark_report.py`

**Then**:
1. Investigate any failed tests
2. Document methodology differences
3. Update tolerance if justified
"""
        else:
            report += """
**Investigate Failures**:
1. Check error messages in test outputs
2. Verify Stata/PanelBox versions
3. Ensure datasets match exactly
4. Consider algorithm differences (especially GMM)

**Document**:
- Any systematic differences
- Methodology variations
- Justified tolerance adjustments
"""

        report += f"""

---

## Technical Details

- **PanelBox Version**: 0.3.0
- **Python**: {sys.version.split()[0]}
- **Platform**: {sys.platform}
- **Report Generator**: `generate_benchmark_report.py`

---

**For questions or issues**: Open an issue on GitHub
"""

        return report


def main():
    """Main entry point."""
    # Determine what to run
    run_stata = "--stata" in sys.argv or "--all" in sys.argv or len(sys.argv) == 1
    run_r = "--r" in sys.argv or "--all" in sys.argv

    # Setup
    base_dir = Path(__file__).parent
    runner = BenchmarkRunner(base_dir)

    print("\n" + "=" * 80)
    print("PANELBOX BENCHMARK SUITE")
    print("=" * 80)
    print(f"\nBase directory: {base_dir}")
    print(f"Results directory: {runner.results_dir}")

    # Run benchmarks
    if run_stata:
        runner.results["stata"] = runner.run_stata_benchmarks()

    if run_r:
        runner.results["r"] = runner.run_r_benchmarks()

    # Generate summary
    runner.generate_summary()

    # Save results
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)

    runner.save_json()
    runner.generate_markdown_report()

    # Print summary
    summary = runner.results["summary"]

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal tests: {summary['total_tests']}")
    print(f"Passed:      {summary['passed']} ✓")
    print(f"Failed:      {summary['failed']} ✗")
    print(f"Pass rate:   {summary['pass_rate']:.1f}%")

    if summary["needs_stata_run"]:
        print(f"\n⚠️  Warning: {summary['placeholder_count']} tests need Stata results")
        print("   Run Stata scripts and update placeholder values")

    print("\n" + "=" * 80)
    print("Reports generated successfully!")
    print("=" * 80)

    # Exit code based on results
    if summary["errors"] > 0:
        sys.exit(2)  # Errors
    elif summary["failed"] > 0:
        sys.exit(1)  # Failures
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()
