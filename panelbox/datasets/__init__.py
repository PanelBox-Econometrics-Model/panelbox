"""
Panel Data Datasets.

===================

This module provides 100+ bundled datasets for panel data econometrics,
organized by category (count, gmm, spatial, discrete, etc.).

Quick Start
-----------
>>> from panelbox.datasets import load_dataset, list_datasets
>>>
>>> # Load any dataset by name
>>> data = load_dataset("healthcare_visits")
>>>
>>> # List all available datasets
>>> print(list_datasets())
>>>
>>> # List datasets in a category
>>> print(list_datasets("count"))
>>>
>>> # Load classic datasets
>>> grunfeld = load_grunfeld()
>>> abdata = load_abdata()
"""

from __future__ import annotations

from .load import (
    get_dataset_info,
    list_categories,
    list_datasets,
    load_abdata,
    load_dataset,
    load_grunfeld,
)

__all__ = [
    "get_dataset_info",
    "list_categories",
    "list_datasets",
    "load_abdata",
    "load_dataset",
    "load_grunfeld",
]
