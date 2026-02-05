"""
Panel Data Datasets
===================

This module provides access to example panel datasets commonly used
in econometrics education and research.

Functions
---------
load_grunfeld : Load Grunfeld investment data
load_abdata : Load Arellano-Bond employment data
list_datasets : List all available datasets
get_dataset_info : Get information about a specific dataset

Examples
--------
>>> import panelbox as pb
>>>
>>> # Load Grunfeld data
>>> data = pb.load_grunfeld()
>>> print(data.head())
>>>
>>> # List all datasets
>>> pb.list_datasets()
"""

from .load import get_dataset_info, list_datasets, load_abdata, load_grunfeld

__all__ = ["load_grunfeld", "load_abdata", "list_datasets", "get_dataset_info"]
