#!/usr/bin/env python3
"""
PanelBox Tutorial Environment Setup Checker

This script verifies that your Python environment has all necessary
packages and versions to run the PanelBox tutorials.

Usage:
    python setup_environment.py
"""

import importlib.util
import sys


def check_python_version() -> tuple[bool, str]:
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"✓ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"✗ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)"


def check_package(package_name: str, min_version: str = None) -> tuple[bool, str]:
    """Check if a package is installed and meets minimum version."""
    spec = importlib.util.find_spec(package_name)

    if spec is None:
        return False, f"✗ {package_name} not installed"

    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")

        if min_version and version != "unknown":
            # Simple version comparison using tuple comparison
            from packaging import version as pkg_version

            try:
                if pkg_version.parse(version) >= pkg_version.parse(min_version):
                    return True, f"✓ {package_name} {version}"
                else:
                    return False, f"✗ {package_name} {version} (Need {min_version}+)"
            except:
                # Fallback to string comparison if packaging not available
                return True, f"✓ {package_name} {version}"
        else:
            return True, f"✓ {package_name} {version}"
    except ImportError:
        return False, f"✗ {package_name} import failed"


def main():
    """Run all environment checks."""
    print("=" * 60)
    print("PanelBox Tutorial Environment Check")
    print("=" * 60)
    print()

    # Core requirements
    requirements = [
        ("Python Version", check_python_version()),
        ("NumPy", check_package("numpy", "1.20.0")),
        ("Pandas", check_package("pandas", "1.3.0")),
        ("SciPy", check_package("scipy", "1.7.0")),
        ("Matplotlib", check_package("matplotlib", "3.3.0")),
        ("Seaborn", check_package("seaborn", "0.11.0")),
        ("Statsmodels", check_package("statsmodels", "0.13.0")),
        ("PanelBox", check_package("panelbox")),
    ]

    # Optional but recommended
    optional = [
        ("Jupyter", check_package("jupyter")),
        ("IPython", check_package("IPython")),
        ("Patsy", check_package("patsy")),
    ]

    # Check core requirements
    print("Core Requirements:")
    print("-" * 60)
    all_passed = True
    for name, (passed, message) in requirements:
        print(f"  {message}")
        if not passed:
            all_passed = False

    print()
    print("Optional (Recommended):")
    print("-" * 60)
    for name, (passed, message) in optional:
        print(f"  {message}")

    print()
    print("=" * 60)

    if all_passed:
        print("✓ All core requirements satisfied!")
        print("\nYou're ready to start the tutorials:")
        print("  cd tutorials/01_fundamentals")
        print("  jupyter notebook 01_introduction_panel_data.ipynb")
    else:
        print("✗ Some requirements are missing.")
        print("\nTo install missing packages:")
        print("  pip install panelbox jupyter seaborn")
        return 1

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
