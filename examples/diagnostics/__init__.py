"""
PanelBox Diagnostics Tutorial Series.
=====================================

Comprehensive tutorial series on diagnostic tests for panel data
econometrics using the PanelBox library.

Tutorials
---------
01. Unit Root Tests — IPS, LLC, Breitung, Hadri, unified interface
02. Cointegration Tests — Pedroni, Westerlund, Kao, ECM
03. Specification Tests — Hausman, J-test, encompassing
04. Spatial Tests — LM tests, Moran's I, LISA

Example
-------
>>> from panelbox.examples import diagnostics
>>> diagnostics.list_tutorials()
>>> diagnostics.verify_installation()
"""

__version__ = "1.0.0"
__author__ = "PanelBox Development Team"
__date__ = "2026-02-22"

from pathlib import Path

# Base directory
MODULE_DIR = Path(__file__).parent

# Tutorial metadata
TUTORIALS = {
    "01": {
        "name": "Unit Root Tests",
        "level": "Intermediate",
        "duration": "90 min",
        "file": "notebooks/01_unit_root_tests.ipynb",
        "prerequisites": ["Panel data basics", "Time series stationarity"],
        "dataset": "unit_root/penn_world_table.csv",
        "description": "Panel unit root tests: IPS, LLC, Breitung, Hadri",
    },
    "02": {
        "name": "Cointegration Tests",
        "level": "Intermediate-Advanced",
        "duration": "110 min",
        "file": "notebooks/02_cointegration_tests.ipynb",
        "prerequisites": ["Unit root tests", "Cointegration concepts"],
        "dataset": "cointegration/oecd_macro.csv",
        "description": "Panel cointegration: Pedroni, Westerlund, Kao",
    },
    "03": {
        "name": "Specification Tests",
        "level": "Intermediate",
        "duration": "110 min",
        "file": "notebooks/03_specification_tests.ipynb",
        "prerequisites": ["Fixed/Random effects models"],
        "dataset": "specification/nlswork.csv",
        "description": "Hausman test, J-test, encompassing tests",
    },
    "04": {
        "name": "Spatial Tests",
        "level": "Intermediate-Advanced",
        "duration": "120 min",
        "file": "notebooks/04_spatial_tests.ipynb",
        "prerequisites": ["OLS regression", "Spatial concepts"],
        "dataset": "spatial/us_counties.csv",
        "description": "LM tests, Moran's I, LISA clusters",
    },
}

# Learning pathways
LEARNING_PATHWAYS = {
    "sequential": {
        "name": "Complete Path",
        "sequence": ["01", "02", "03", "04"],
        "description": "Full diagnostic toolkit, recommended order",
    },
    "time_series_focus": {
        "name": "Time Series Diagnostics",
        "sequence": ["01", "02"],
        "description": "Unit root and cointegration for macro panels",
    },
    "model_selection": {
        "name": "Model Selection",
        "sequence": ["03"],
        "description": "Hausman, J-test, and encompassing tests",
    },
    "spatial_focus": {
        "name": "Spatial Diagnostics",
        "sequence": ["04"],
        "description": "Spatial dependence detection and testing",
    },
}


def list_tutorials(verbose: bool = False):
    """List all available tutorials."""
    print("Panel Data Diagnostics — Tutorial Series")
    print("=" * 50)
    for num, info in TUTORIALS.items():
        print(f"\n  {num}. {info['name']} ({info['level']}, ~{info['duration']})")
        if verbose:
            print(f"      File: {info['file']}")
            print(f"      Dataset: {info['dataset']}")
            print(f"      Prerequisites: {', '.join(info['prerequisites'])}")
            print(f"      {info['description']}")


def list_pathways():
    """Print recommended learning pathways."""
    print("Learning Pathways")
    print("=" * 40)
    for _key, path in LEARNING_PATHWAYS.items():
        seq = " -> ".join(path["sequence"])
        print(f"\n  {path['name']}: {seq}")
        print(f"    {path['description']}")


def get_tutorial_path(tutorial_num: str) -> Path:
    """Get absolute path to a tutorial notebook."""
    if tutorial_num not in TUTORIALS:
        valid = ", ".join(TUTORIALS.keys())
        raise ValueError(f"Invalid tutorial: {tutorial_num}. Valid: {valid}")
    return MODULE_DIR / TUTORIALS[tutorial_num]["file"]


def get_data_path(dataset_name: str) -> Path:
    """Get absolute path to a dataset file."""
    path = MODULE_DIR / "data" / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


def verify_installation() -> dict:
    """Verify that the tutorial installation is complete."""
    results = {"directories": {}, "notebooks": {}, "datasets": {}, "utils": {}}

    # Check directories
    for d in ["data", "notebooks", "solutions", "outputs", "utils", "tests"]:
        results["directories"][d] = (MODULE_DIR / d).is_dir()

    # Check notebooks
    for _num, info in TUTORIALS.items():
        nb_path = MODULE_DIR / info["file"]
        results["notebooks"][info["file"]] = nb_path.exists()

    # Check datasets
    for _num, info in TUTORIALS.items():
        ds_path = MODULE_DIR / "data" / info["dataset"]
        results["datasets"][info["dataset"]] = ds_path.exists()

    # Check utils
    utils_dir = MODULE_DIR / "utils"
    for f in [
        "__init__.py",
        "data_generators.py",
        "unit_root_helpers.py",
        "cointegration_helpers.py",
        "spatial_helpers.py",
        "visualization_helpers.py",
        "diagnostics_utils.py",
    ]:
        results["utils"][f] = (utils_dir / f).exists()

    # Summary
    all_ok = all(v for section in results.values() for v in section.values())
    results["all_ok"] = all_ok

    if all_ok:
        print("All checks passed!")
    else:
        print("Some checks failed:")
        for section, checks in results.items():
            if section == "all_ok":
                continue
            for name, ok in checks.items():
                if not ok:
                    print(f"  MISSING: {section}/{name}")

    return results


__all__ = [
    "LEARNING_PATHWAYS",
    "TUTORIALS",
    "get_data_path",
    "get_tutorial_path",
    "list_pathways",
    "list_tutorials",
    "verify_installation",
]
