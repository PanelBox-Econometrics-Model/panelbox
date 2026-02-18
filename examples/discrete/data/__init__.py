"""
Discrete Choice Models - Data

Part of the PanelBox tutorial series on discrete choice econometrics.

This module contains datasets used throughout the discrete choice tutorials:

- labor_participation.csv : Labor force participation panel data
- job_training.csv         : Job training program evaluation data
- transportation_choice.csv: Multi-modal transportation choice data
- credit_rating.csv        : Ordered credit rating panel data
- career_choice.csv        : Career choice after graduation data

See README.md for complete data dictionary and variable descriptions.
"""

from pathlib import Path

__all__ = ["DATA_DIR"]

# Data directory path
DATA_DIR = Path(__file__).parent
