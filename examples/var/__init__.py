"""
PanelBox Panel VAR Tutorials

This module contains comprehensive tutorials for Panel Vector Autoregression
(VAR) models including:
- VAR estimation and lag selection
- Impulse Response Functions (IRFs)
- Forecast Error Variance Decomposition (FEVD)
- Granger Causality and Dumitrescu-Hurlin tests
- VECM and Cointegration
- Dynamic GMM estimation
- Applied case study (monetary policy transmission)

Tutorials progress from beginner to advanced level.

Example:
    To view available tutorials::

        from panelbox.examples import var
        print(var.list_tutorials())
"""

__version__ = "1.0.0"
__author__ = "PanelBox Development Team"
__date__ = "2026-02-17"

from pathlib import Path

# Base directory
MODULE_DIR = Path(__file__).parent

TUTORIALS = {
    "01": {
        "name": "VAR Introduction",
        "level": "Beginner",
        "duration": "60-90 min",
        "file": "notebooks/01_var_introduction.ipynb",
        "prerequisites": ["Basic panel data knowledge"],
        "dataset": "macro_panel.csv",
        "description": "Learn VAR fundamentals: estimation, lag selection, stability",
    },
    "02": {
        "name": "IRF Analysis",
        "level": "Intermediate",
        "duration": "90-120 min",
        "file": "notebooks/02_irf_analysis.ipynb",
        "prerequisites": ["Tutorial 01"],
        "dataset": "energy_panel.csv",
        "description": "Impulse Response Functions: Cholesky, Generalized, Bootstrap CI",
    },
    "03": {
        "name": "FEVD Decomposition",
        "level": "Intermediate",
        "duration": "60-90 min",
        "file": "notebooks/03_fevd_decomposition.ipynb",
        "prerequisites": ["Tutorial 01"],
        "dataset": "finance_panel.csv",
        "description": "Forecast Error Variance Decomposition and interpretation",
    },
    "04": {
        "name": "Granger Causality",
        "level": "Intermediate-Advanced",
        "duration": "90-120 min",
        "file": "notebooks/04_granger_causality.ipynb",
        "prerequisites": ["Tutorial 01"],
        "dataset": "macro_panel.csv",
        "description": "Wald tests, Dumitrescu-Hurlin test, causality networks",
    },
    "05": {
        "name": "VECM Cointegration",
        "level": "Advanced",
        "duration": "120-150 min",
        "file": "notebooks/05_vecm_cointegration.ipynb",
        "prerequisites": ["Tutorial 01", "Tutorial 02"],
        "dataset": "Generated inline",
        "description": "I(1) variables, cointegration rank tests, VECM estimation",
    },
    "06": {
        "name": "Dynamic GMM",
        "level": "Advanced",
        "duration": "120-150 min",
        "file": "notebooks/06_dynamic_gmm.ipynb",
        "prerequisites": ["Tutorial 01"],
        "dataset": "Generated inline",
        "description": "Nickell bias, Arellano-Bond, Blundell-Bond estimation",
    },
    "07": {
        "name": "Case Study",
        "level": "Capstone",
        "duration": "180-240 min",
        "file": "notebooks/07_case_study.ipynb",
        "prerequisites": ["Tutorials 01-06"],
        "dataset": "monetary_policy.csv",
        "description": "Complete monetary policy transmission analysis",
    },
}

LEARNING_PATHWAYS = {
    "beginner": {
        "name": "Beginner Pathway",
        "sequence": ["01", "02", "03"],
        "description": "Start with VAR basics, then learn IRFs and FEVD",
    },
    "causality_focus": {
        "name": "Causality Focus",
        "sequence": ["01", "04"],
        "description": "Quick path to Granger causality analysis",
    },
    "advanced_time_series": {
        "name": "Advanced Time Series",
        "sequence": ["01", "02", "05", "06"],
        "description": "Covers cointegration and GMM methods",
    },
    "complete": {
        "name": "Complete Course",
        "sequence": ["01", "02", "03", "04", "05", "06", "07"],
        "description": "Full mastery of Panel VAR methods",
    },
}


def list_tutorials(verbose=False):
    """Print a formatted list of all available tutorials."""
    print("=" * 70)
    print("PanelBox Panel VAR Tutorials")
    print("=" * 70)
    print()

    if not verbose:
        print("Available Tutorials:")
        print()
        for num, info in TUTORIALS.items():
            print(f"  {num}. {info['name']}")
            print(f"      Level: {info['level']} | Duration: {info['duration']}")
            print(f"      {info['description']}")
            print()
    else:
        for num, info in TUTORIALS.items():
            print(f"Tutorial {num}: {info['name']}")
            print(f"  Level: {info['level']}")
            print(f"  Duration: {info['duration']}")
            print(f"  File: {info['file']}")
            print(f"  Dataset: {info['dataset']}")
            print(f"  Prerequisites: {', '.join(info['prerequisites'])}")
            print(f"  {info['description']}")
            print()


def list_pathways():
    """Print available learning pathways."""
    print("Learning Pathways:")
    print()
    for key, info in LEARNING_PATHWAYS.items():
        sequence = " -> ".join(info["sequence"])
        print(f"  {info['name']}: {sequence}")
        print(f"    {info['description']}")
        print()


def get_tutorial_path(tutorial_num):
    """Get the absolute path to a tutorial notebook."""
    if tutorial_num not in TUTORIALS:
        valid = ", ".join(TUTORIALS.keys())
        raise ValueError(f"Invalid tutorial number '{tutorial_num}'. Valid: {valid}")
    return MODULE_DIR / TUTORIALS[tutorial_num]["file"]


def get_data_path(dataset_name):
    """Get the absolute path to a tutorial dataset."""
    data_path = MODULE_DIR / "data" / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return data_path


def verify_installation():
    """Verify that the tutorial installation is complete."""
    results = {
        "all_passed": True,
        "directories": {},
        "datasets": {},
    }

    for dirname in ["notebooks", "data", "outputs", "solutions", "utils", "tests"]:
        dirpath = MODULE_DIR / dirname
        results["directories"][dirname] = dirpath.exists()
        if not dirpath.exists():
            results["all_passed"] = False

    for dataset in [
        "macro_panel.csv",
        "energy_panel.csv",
        "finance_panel.csv",
        "monetary_policy.csv",
    ]:
        datapath = MODULE_DIR / "data" / dataset
        results["datasets"][dataset] = datapath.exists()
        if not datapath.exists():
            results["all_passed"] = False

    return results


__all__ = [
    "TUTORIALS",
    "LEARNING_PATHWAYS",
    "list_tutorials",
    "list_pathways",
    "get_tutorial_path",
    "get_data_path",
    "verify_installation",
]
