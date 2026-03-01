"""
Additional coverage tests for datasets/load.py.

Targets uncovered branches related to:
- _find_dataset with category parameter
- load_dataset with category/name format
- load_dataset for unknown datasets
- list_datasets with category filter
- list_categories
- get_dataset_info for known/unknown datasets
- load_abdata with return_panel_data
- Various error paths
"""

from __future__ import annotations

import pandas as pd

from panelbox.datasets.load import (
    _find_dataset,
    get_dataset_info,
    list_categories,
    list_datasets,
    load_abdata,
    load_dataset,
    load_grunfeld,
)


class TestFindDataset:
    """Cover branches in _find_dataset."""

    def test_find_dataset_root_level(self):
        """Find a dataset at the root data directory level."""
        result = _find_dataset("grunfeld")
        assert result is not None
        assert result.endswith("grunfeld.csv")

    def test_find_dataset_with_category(self):
        """Cover category-specific search branch."""
        # Try to find a dataset in a known category
        categories = list_categories()
        if categories:
            cat = categories[0]
            datasets = list_datasets(category=cat)
            if datasets:
                ds_name = datasets[0]
                result = _find_dataset(ds_name, category=cat)
                assert result is not None

    def test_find_dataset_with_category_not_found(self):
        """Cover category search that returns None."""
        result = _find_dataset("nonexistent_dataset_xyz", category="nonexistent_cat")
        assert result is None

    def test_find_dataset_search_subdirectories(self):
        """Cover subdirectory search when no category given and not in root."""
        # Any categorized dataset should be found via subdirectory search
        all_datasets = list_datasets()
        categorized = [d for d in all_datasets if "/" in d]
        if categorized:
            # Get just the name part (after /)
            full_name = categorized[0]
            name_only = full_name.split("/")[1]
            result = _find_dataset(name_only)
            assert result is not None

    def test_find_dataset_nonexistent(self):
        """Cover case where dataset is not found anywhere."""
        result = _find_dataset("absolutely_nonexistent_dataset_12345")
        assert result is None


class TestLoadDatasetFunction:
    """Cover branches in load_dataset."""

    def test_load_grunfeld_shortcut(self):
        """Cover shortcut for grunfeld."""
        df = load_dataset("grunfeld")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_abdata_shortcut(self):
        """Cover shortcut for abdata."""
        result = load_dataset("abdata")
        # May be None if file doesn't exist, or DataFrame
        if result is not None:
            assert isinstance(result, pd.DataFrame)

    def test_load_dataset_with_slash_format(self):
        """Cover 'category/name' format parsing."""
        all_datasets = list_datasets()
        categorized = [d for d in all_datasets if "/" in d]
        if categorized:
            result = load_dataset(categorized[0])
            assert isinstance(result, pd.DataFrame)

    def test_load_dataset_with_category_param(self):
        """Cover explicit category parameter."""
        categories = list_categories()
        for cat in categories:
            datasets = list_datasets(category=cat)
            if datasets:
                result = load_dataset(datasets[0], category=cat)
                assert isinstance(result, pd.DataFrame)
                break

    def test_load_dataset_not_found(self):
        """Cover not-found path (returns None, logs warning)."""
        result = load_dataset("totally_fake_dataset_name_xyz")
        assert result is None


class TestListDatasetsFunction:
    """Cover branches in list_datasets."""

    def test_list_all_datasets(self):
        """Cover listing all datasets (no category)."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_list_datasets_specific_category(self):
        """Cover listing with specific category."""
        categories = list_categories()
        if categories:
            datasets = list_datasets(category=categories[0])
            assert isinstance(datasets, list)

    def test_list_datasets_nonexistent_category(self):
        """Cover listing with category that doesn't exist (empty list)."""
        datasets = list_datasets(category="nonexistent_category_xyz")
        assert datasets == []


class TestListCategoriesFunction:
    """Cover list_categories function."""

    def test_list_categories_returns_list(self):
        """Cover list_categories returns non-empty list."""
        cats = list_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0

    def test_list_categories_no_data_dir(self):
        """Cover list_categories when _DATA_DIR doesn't exist."""
        import panelbox.datasets.load as load_module

        original = load_module._DATA_DIR
        load_module._DATA_DIR = "/nonexistent/path"
        try:
            cats = list_categories()
            assert cats == []
        finally:
            load_module._DATA_DIR = original


class TestGetDatasetInfo:
    """Cover get_dataset_info function branches."""

    def test_get_info_grunfeld(self):
        """Cover known dataset info with entity/time cols."""
        info = get_dataset_info("grunfeld")
        assert info["name"] == "Grunfeld Investment Data"
        assert "n_obs" in info
        assert "n_entities" in info
        assert "n_periods" in info
        assert "balanced" in info

    def test_get_info_abdata(self):
        """Cover known dataset info for abdata."""
        info = get_dataset_info("abdata")
        assert "name" in info

    def test_get_info_unknown_dataset(self):
        """Cover unknown dataset (default info dict)."""
        info = get_dataset_info("totally_unknown_dataset")
        assert info["description"] == "Unknown dataset"

    def test_get_info_categorized_dataset(self):
        """Cover info for a categorized dataset (no entity_col/time_col)."""
        all_datasets = list_datasets()
        categorized = [d for d in all_datasets if "/" in d]
        if categorized:
            info = get_dataset_info(categorized[0])
            assert "n_obs" in info


class TestLoadAbdata:
    """Cover load_abdata branches."""

    def test_load_abdata_as_dataframe(self):
        """Cover default return_panel_data=False."""
        result = load_abdata()
        if result is not None:
            assert isinstance(result, pd.DataFrame)

    def test_load_abdata_as_panel_data(self):
        """Cover return_panel_data=True branch."""
        result = load_abdata(return_panel_data=True)
        if result is not None:
            from panelbox.core.panel_data import PanelData

            assert isinstance(result, PanelData)

    def test_load_abdata_file_not_exist(self):
        """Cover file not exists branch (returns None)."""
        import panelbox.datasets.load as load_module

        original_path = load_module._DATA_DIR
        load_module._DATA_DIR = "/nonexistent/path"
        try:
            result = load_abdata()
            assert result is None
        finally:
            load_module._DATA_DIR = original_path


class TestLoadGrunfeldBranches:
    """Cover load_grunfeld branches."""

    def test_load_grunfeld_as_panel_data(self):
        """Cover return_panel_data=True branch."""
        from panelbox.core.panel_data import PanelData

        result = load_grunfeld(return_panel_data=True)
        assert isinstance(result, PanelData)
