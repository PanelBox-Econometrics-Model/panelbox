"""
Stochastic Frontier Analysis (SFA) Tutorial Series.
===================================================

Comprehensive tutorial series on Stochastic Frontier Analysis and
technical efficiency measurement using the PanelBox library.

Tutorials:
    01 - Introduction to SFA (cross-section, MLE, efficiency estimation)
    02 - Panel SFA (Pitt-Lee, BC92, BC95, CSS, Kumbhakar)
    03 - Four-Component & TFP (persistent/transient efficiency, TFP decomposition)
    04 - Determinants & Heterogeneity (BC95 determinants, Wang 2002, marginal effects)
    05 - Testing & Comparison (LR tests, Vuong, bootstrap, model selection)
    06 - Complete Case Study (Brazilian manufacturing analysis)
"""

from pathlib import Path

__version__ = "1.0.0"
__author__ = "PanelBox Development Team"
__date__ = "2026-02-18"

MODULE_DIR = Path(__file__).parent

TUTORIALS = {
    "01": {
        "name": "Introduction to SFA",
        "level": "Beginner",
        "duration": "90-120 min",
        "file": "notebooks/01_introduction_sfa.ipynb",
        "prerequisites": ["Basic regression knowledge", "MLE concepts"],
        "dataset": "hospital_data.csv",
        "description": "Frontier concept, MLE estimation, efficiency measurement",
    },
    "02": {
        "name": "Panel SFA",
        "level": "Intermediate",
        "duration": "120-150 min",
        "file": "notebooks/02_panel_sfa.ipynb",
        "prerequisites": ["Tutorial 01", "Panel data concepts"],
        "dataset": "bank_panel.csv",
        "description": "Pitt-Lee, BC92, BC95, CSS, Kumbhakar panel models",
    },
    "03": {
        "name": "Four-Component & TFP",
        "level": "Advanced",
        "duration": "90-120 min",
        "file": "notebooks/03_four_component_tfp.ipynb",
        "prerequisites": ["Tutorial 02"],
        "dataset": "manufacturing_panel.csv",
        "description": "Persistent/transient efficiency, TFP decomposition",
    },
    "04": {
        "name": "Determinants & Heterogeneity",
        "level": "Intermediate-Advanced",
        "duration": "90-120 min",
        "file": "notebooks/04_determinants_heterogeneity.ipynb",
        "prerequisites": ["Tutorial 02"],
        "dataset": "hospital_panel.csv",
        "description": "BC95 determinants, Wang 2002, marginal effects",
    },
    "05": {
        "name": "Testing & Comparison",
        "level": "Intermediate-Advanced",
        "duration": "90-120 min",
        "file": "notebooks/05_testing_comparison.ipynb",
        "prerequisites": ["Tutorials 01-02"],
        "dataset": "dairy_farm.csv",
        "description": "LR tests, Vuong test, bootstrap, model selection",
    },
    "06": {
        "name": "Complete Case Study",
        "level": "Capstone",
        "duration": "180-240 min",
        "file": "notebooks/06_complete_case_study.ipynb",
        "prerequisites": ["Tutorials 01-05"],
        "dataset": "brazilian_firms.csv",
        "description": "Brazilian manufacturing analysis integrating all methods",
    },
}

LEARNING_PATHWAYS = {
    "recommended": {
        "description": "Standard learning path",
        "sequence": ["01", "02", "03", "04", "05", "06"],
    },
    "quick_start": {
        "description": "Core concepts only",
        "sequence": ["01", "02", "05"],
    },
    "applied": {
        "description": "For applied researchers",
        "sequence": ["01", "02", "04", "06"],
    },
}


def list_tutorials(verbose: bool = False) -> None:
    """Print tutorial catalog."""
    print("Stochastic Frontier Analysis Tutorial Series")
    print("=" * 60)
    for num, info in TUTORIALS.items():
        print(f"  {num}. {info['name']} [{info['level']}] ({info['duration']})")
        if verbose:
            print(f"      Dataset: {info['dataset']}")
            print(f"      {info['description']}")
            print()


def get_tutorial_path(tutorial_num: str) -> Path:
    """Return path to a tutorial notebook."""
    if tutorial_num in TUTORIALS:
        return MODULE_DIR / TUTORIALS[tutorial_num]["file"]
    raise ValueError(f"Tutorial {tutorial_num} not found.")


def get_data_path(dataset_name: str) -> Path:
    """Return path to a dataset file."""
    return MODULE_DIR / "data" / dataset_name


__all__ = [
    "LEARNING_PATHWAYS",
    "TUTORIALS",
    "get_data_path",
    "get_tutorial_path",
    "list_tutorials",
]
