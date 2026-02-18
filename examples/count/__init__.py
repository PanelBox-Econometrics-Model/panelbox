"""
PanelBox Count Models Tutorials

This module contains comprehensive tutorials for count data models including:
- Poisson regression (pooled, FE, RE)
- Negative Binomial models
- PPML for gravity equations
- Zero-Inflated models (ZIP, ZINB)
- Marginal effects computation

Tutorials progress from beginner to advanced level.

Example:
    To view available tutorials::

        from panelbox.examples import count
        print(count.list_tutorials())

    To access tutorial metadata::

        tutorials = count.TUTORIALS
        print(tutorials['01']['name'])
"""

__version__ = "1.0.0"
__author__ = "PanelBox Development Team"
__date__ = "2026-02-16"

from pathlib import Path

# Base directory
MODULE_DIR = Path(__file__).parent

# Tutorial metadata
TUTORIALS = {
    "01": {
        "name": "Poisson Introduction",
        "level": "Beginner",
        "duration": "60-75 min",
        "file": "notebooks/01_poisson_introduction.ipynb",
        "prerequisites": ["Basic regression knowledge"],
        "dataset": "healthcare_visits.csv",
        "description": "Learn fundamentals of Poisson regression for count data",
    },
    "02": {
        "name": "Negative Binomial",
        "level": "Intermediate",
        "duration": "60 min",
        "file": "notebooks/02_negative_binomial.ipynb",
        "prerequisites": ["Tutorial 01"],
        "dataset": "firm_patents.csv",
        "description": "Handle overdispersed count data with Negative Binomial models",
    },
    "03": {
        "name": "Fixed and Random Effects",
        "level": "Intermediate-Advanced",
        "duration": "75 min",
        "file": "notebooks/03_fe_re_count.ipynb",
        "prerequisites": ["Tutorials 01-02", "Panel data basics"],
        "dataset": "city_crime.csv",
        "description": "Analyze count panel data with FE and RE specifications",
    },
    "04": {
        "name": "PPML and Gravity Models",
        "level": "Advanced",
        "duration": "90 min",
        "file": "notebooks/04_ppml_gravity.ipynb",
        "prerequisites": ["Tutorial 01"],
        "dataset": "bilateral_trade.csv",
        "description": "Apply PPML to gravity equations following Santos Silva & Tenreyro (2006)",
    },
    "05": {
        "name": "Zero-Inflated Models",
        "level": "Advanced",
        "duration": "70 min",
        "file": "notebooks/05_zero_inflated.ipynb",
        "prerequisites": ["Tutorials 01-02"],
        "dataset": "healthcare_zinb.csv",
        "description": "Model excess zeros with ZIP and ZINB specifications",
    },
    "06": {
        "name": "Marginal Effects",
        "level": "Intermediate-Advanced",
        "duration": "65 min",
        "file": "notebooks/06_marginal_effects_count.ipynb",
        "prerequisites": ["Tutorials 01-02"],
        "dataset": "policy_impact.csv",
        "description": "Compute and interpret marginal effects for count models",
    },
    "07": {
        "name": "Innovation Case Study",
        "level": "Advanced",
        "duration": "90-120 min",
        "file": "notebooks/07_innovation_case_study.ipynb",
        "prerequisites": ["All previous tutorials"],
        "dataset": "firm_innovation_full.csv",
        "description": "Complete workflow: from data exploration to publication-ready results",
    },
}

# Learning pathways
LEARNING_PATHWAYS = {
    "beginner": {
        "name": "Beginner Pathway",
        "sequence": ["01", "02", "06", "07"],
        "description": "Start with basics, learn overdispersion, then apply to real cases",
    },
    "panel_focus": {
        "name": "Panel Data Focus",
        "sequence": ["01", "03", "04"],
        "description": "Emphasis on panel count models and PPML",
    },
    "applied_researcher": {
        "name": "Applied Researcher",
        "sequence": ["01", "02", "05", "06", "07"],
        "description": "Comprehensive coverage for empirical work",
    },
    "complete": {
        "name": "Complete Course",
        "sequence": ["01", "02", "03", "04", "05", "06", "07"],
        "description": "Full mastery of count models",
    },
}


def list_tutorials(verbose=False):
    """
    Print a formatted list of all available tutorials.

    Parameters
    ----------
    verbose : bool, default False
        If True, include full details for each tutorial.
        If False, show compact list.

    Returns
    -------
    None
        Prints tutorial information to console.

    Examples
    --------
    >>> list_tutorials()
    Count Models Tutorials - Available Notebooks:
    01. Poisson Introduction (Beginner, 60-75 min)
    02. Negative Binomial (Intermediate, 60 min)
    ...

    >>> list_tutorials(verbose=True)
    [Full details for each tutorial]
    """
    print("=" * 70)
    print("PanelBox Count Models Tutorials")
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
            print("-" * 70)
            print(f"  Level:         {info['level']}")
            print(f"  Duration:      {info['duration']}")
            print(f"  File:          {info['file']}")
            print(f"  Dataset:       {info['dataset']}")
            print(f"  Prerequisites: {', '.join(info['prerequisites'])}")
            print(f"  Description:   {info['description']}")
            print()

    print("=" * 70)
    print(f"Total tutorials: {len(TUTORIALS)}")
    print()
    print("For detailed setup instructions, see GETTING_STARTED.md")
    print("For learning pathways, use: list_pathways()")


def list_pathways():
    """
    Print recommended learning pathways through the tutorials.

    Returns
    -------
    None
        Prints pathway information to console.

    Examples
    --------
    >>> list_pathways()
    Recommended Learning Pathways:
    1. Beginner Pathway
       Sequence: 01 → 02 → 06 → 07
       ...
    """
    print("=" * 70)
    print("Recommended Learning Pathways")
    print("=" * 70)
    print()

    for i, (key, pathway) in enumerate(LEARNING_PATHWAYS.items(), 1):
        print(f"{i}. {pathway['name']}")
        print(f"   Sequence: {' → '.join(pathway['sequence'])}")
        print(f"   Description: {pathway['description']}")
        print()

        tutorials_in_path = [TUTORIALS[num]["name"] for num in pathway["sequence"]]
        for j, (num, name) in enumerate(zip(pathway["sequence"], tutorials_in_path), 1):
            print(f"      {j}. Tutorial {num}: {name}")
        print()

    print("=" * 70)


def get_tutorial_path(tutorial_num):
    """
    Get the absolute path to a tutorial notebook.

    Parameters
    ----------
    tutorial_num : str
        Tutorial number (e.g., "01", "02", etc.)

    Returns
    -------
    Path
        Absolute path to the tutorial notebook.

    Raises
    ------
    ValueError
        If tutorial_num is not valid.

    Examples
    --------
    >>> path = get_tutorial_path("01")
    >>> print(path)
    /path/to/panelbox/examples/count/notebooks/01_poisson_introduction.ipynb
    """
    if tutorial_num not in TUTORIALS:
        valid = ", ".join(TUTORIALS.keys())
        raise ValueError(f"Invalid tutorial number '{tutorial_num}'. Valid: {valid}")

    return MODULE_DIR / TUTORIALS[tutorial_num]["file"]


def get_data_path(dataset_name):
    """
    Get the absolute path to a tutorial dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset file (e.g., "healthcare_visits.csv")

    Returns
    -------
    Path
        Absolute path to the dataset.

    Examples
    --------
    >>> path = get_data_path("healthcare_visits.csv")
    >>> import pandas as pd
    >>> df = pd.read_csv(path)
    """
    data_path = MODULE_DIR / "data" / dataset_name
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return data_path


def verify_installation():
    """
    Verify that the tutorial installation is complete and correct.

    Returns
    -------
    dict
        Dictionary with verification results.

    Examples
    --------
    >>> results = verify_installation()
    >>> if results['all_passed']:
    ...     print("Installation verified!")
    """
    results = {"all_passed": True, "directories": {}, "notebooks": {}, "datasets": {}, "utils": {}}

    # Check directories
    for dirname in ["notebooks", "data", "outputs", "solutions", "utils", "tests"]:
        dirpath = MODULE_DIR / dirname
        results["directories"][dirname] = dirpath.exists()
        if not dirpath.exists():
            results["all_passed"] = False

    # Check notebooks
    for num, info in TUTORIALS.items():
        notebook_path = MODULE_DIR / info["file"]
        results["notebooks"][num] = notebook_path.exists()
        if not notebook_path.exists():
            results["all_passed"] = False

    # Check datasets
    datasets = [
        "healthcare_visits.csv",
        "firm_patents.csv",
        "city_crime.csv",
        "bilateral_trade.csv",
        "healthcare_zinb.csv",
        "policy_impact.csv",
        "firm_innovation_full.csv",
    ]
    for dataset in datasets:
        dataset_path = MODULE_DIR / "data" / dataset
        results["datasets"][dataset] = dataset_path.exists()
        if not dataset_path.exists():
            results["all_passed"] = False

    # Check utils
    for util_file in ["data_generators.py", "visualization_helpers.py", "diagnostics_helpers.py"]:
        util_path = MODULE_DIR / "utils" / util_file
        results["utils"][util_file] = util_path.exists()
        if not util_path.exists():
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
