#!/usr/bin/env python3
"""
Script to check test coverage for spatial econometrics module.

Ensures we meet the >= 85% coverage requirement.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_coverage():
    """Check test coverage for spatial modules."""

    # Modules to check coverage
    modules_to_test = [
        "panelbox/models/spatial",
        "panelbox/core/spatial_weights.py",
        "panelbox/validation/spatial",
        "panelbox/effects/spatial_effects.py",
        "panelbox/standard_errors/spatial_hac.py",
        "panelbox/optimization/spatial_optimizations.py",
        "panelbox/optimization/parallel_inference.py",
    ]

    # Test directories
    test_dirs = [
        "tests/models/spatial",
        "tests/validation/spatial",
        "tests/standard_errors",
        "tests/performance",
    ]

    print("=" * 60)
    print("SPATIAL ECONOMETRICS - TEST COVERAGE REPORT")
    print("=" * 60)

    # Check if pytest and coverage are installed
    try:
        import coverage
        import pytest
    except ImportError:
        print("\n❌ Missing dependencies. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov", "coverage"])
        print("✓ Dependencies installed")

    # Run tests with coverage
    print("\n📊 Running tests with coverage analysis...")
    print("-" * 40)

    # Prepare coverage command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=panelbox/models/spatial",
        "--cov=panelbox/core/spatial_weights",
        "--cov=panelbox/validation/spatial",
        "--cov=panelbox/effects",
        "--cov=panelbox/standard_errors/spatial_hac",
        "--cov=panelbox/optimization",
        "--cov-report=term-missing:skip-covered",
        "--cov-report=html:htmlcov_spatial",
        "--cov-report=xml:coverage_spatial.xml",
        "-v",
        "--tb=short",
    ]

    # Add test directories
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            cmd.append(test_dir)

    # Run coverage
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse coverage output
    output_lines = result.stdout.split("\n")
    coverage_data = {}
    total_coverage = 0
    in_coverage_section = False

    for line in output_lines:
        if "Name" in line and "Stmts" in line:
            in_coverage_section = True
            continue

        if in_coverage_section:
            if line.startswith("TOTAL"):
                parts = line.split()
                if len(parts) >= 4:
                    total_coverage = float(parts[-1].rstrip("%"))
                break

            if "spatial" in line or "effects" in line or "optimization" in line:
                parts = line.split()
                if len(parts) >= 4 and parts[-1].endswith("%"):
                    module = parts[0]
                    coverage_pct = float(parts[-1].rstrip("%"))
                    coverage_data[module] = coverage_pct

    # Display results
    print("\n📈 Coverage Results by Module:")
    print("-" * 40)

    for module, cov in sorted(coverage_data.items()):
        status = "✅" if cov >= 85 else "⚠️"
        print(f"{status} {module}: {cov:.1f}%")

    print("\n" + "=" * 40)
    print(f"OVERALL SPATIAL MODULE COVERAGE: {total_coverage:.1f}%")
    print("=" * 40)

    # Check if we meet the target
    if total_coverage >= 85:
        print("\n✅ SUCCESS: Coverage target (≥85%) achieved!")
    else:
        print(f"\n⚠️  WARNING: Coverage ({total_coverage:.1f}%) below target (85%)")
        print("\nModules needing more tests:")
        for module, cov in coverage_data.items():
            if cov < 85:
                print(f"  - {module}: {cov:.1f}% (needs {85-cov:.1f}% more)")

    # Generate detailed report
    print("\n📄 Detailed HTML report: htmlcov_spatial/index.html")
    print("📄 XML report for CI: coverage_spatial.xml")

    return total_coverage >= 85


def identify_missing_tests():
    """Identify what tests are missing."""

    print("\n🔍 Analyzing Missing Test Coverage...")
    print("-" * 40)

    # Check which spatial files exist
    spatial_files = list(Path("panelbox/models/spatial").glob("*.py"))
    test_files = list(Path("tests/models/spatial").glob("test_*.py"))

    # Map source to test files
    print("\nSource → Test File Mapping:")
    for src_file in spatial_files:
        if src_file.name == "__init__.py":
            continue

        test_name = f"test_{src_file.stem}.py"
        test_path = Path("tests/models/spatial") / test_name

        if test_path.exists():
            print(f"✅ {src_file.name} → {test_name}")
        else:
            print(f"❌ {src_file.name} → {test_name} (MISSING)")

    # Check test completeness
    print("\n📋 Test Completeness Checklist:")
    required_tests = [
        ("SAR estimation", "test_sar_estimation.py"),
        ("SEM estimation", "test_sem_estimation.py"),
        ("SDM estimation", "test_spatial_durbin.py"),
        ("GNS estimation", "test_gns.py"),
        ("Spatial weights", "test_spatial_weights.py"),
        ("Moran's I test", "test_morans_i.py"),
        ("LM tests", "test_lm_tests.py"),
        ("Effects decomposition", "test_spatial_effects.py"),
        ("Spatial HAC", "test_spatial_hac.py"),
        ("Performance benchmarks", "test_spatial_benchmarks.py"),
    ]

    for test_desc, test_file in required_tests:
        full_path = Path("tests") / "models" / "spatial" / test_file
        if not full_path.exists():
            full_path = Path("tests") / test_file

        if full_path.exists() or any(Path(".").glob(f"**/{ test_file}")):
            print(f"✅ {test_desc}")
        else:
            print(f"⚠️  {test_desc} - Consider adding {test_file}")


def generate_coverage_badge():
    """Generate coverage badge for README."""

    try:
        # Read XML coverage report
        import xml.etree.ElementTree as ET

        tree = ET.parse("coverage_spatial.xml")
        root = tree.getroot()

        # Get coverage percentage
        coverage_pct = float(root.attrib["line-rate"]) * 100

        # Determine badge color
        if coverage_pct >= 90:
            color = "brightgreen"
        elif coverage_pct >= 80:
            color = "green"
        elif coverage_pct >= 70:
            color = "yellowgreen"
        elif coverage_pct >= 60:
            color = "yellow"
        else:
            color = "red"

        # Badge URL
        badge_url = f"https://img.shields.io/badge/coverage-{coverage_pct:.1f}%25-{color}"

        print(f"\n🏷️  Coverage Badge: {badge_url}")
        print("\nAdd to README.md:")
        print(f"![Coverage]({badge_url})")

    except Exception as e:
        print(f"\n⚠️  Could not generate badge: {e}")


def main():
    """Main function."""

    print("\n🚀 PanelBox Spatial Econometrics - Test Coverage Check\n")

    # Check current directory
    if not os.path.exists("panelbox"):
        print("❌ Error: Must run from PanelBox root directory")
        sys.exit(1)

    # Run coverage check
    meets_target = check_coverage()

    # Analyze missing tests
    identify_missing_tests()

    # Generate badge
    generate_coverage_badge()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if meets_target:
        print("✅ Test coverage meets requirement (≥ 85%)")
        print("✅ Ready for production release")
        sys.exit(0)
    else:
        print("⚠️  Test coverage below requirement (< 85%)")
        print("📝 Add more tests to missing areas")
        sys.exit(1)


if __name__ == "__main__":
    main()
