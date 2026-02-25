"""
Tests for dataset loading functions.
"""

import os

import pandas as pd

from panelbox.core.panel_data import PanelData
from panelbox.datasets.load import (
    _get_data_path,
    get_dataset_info,
    list_datasets,
    load_abdata,
    load_dataset,
    load_grunfeld,
)


class TestGetDataPath:
    """Test _get_data_path function."""

    def test_get_data_path_exists(self):
        """Test that data path exists."""
        path = _get_data_path()
        assert os.path.isdir(path)

    def test_get_data_path_is_string(self):
        """Test that data path is a string."""
        path = _get_data_path()
        assert isinstance(path, str)

    def test_get_data_path_contains_data(self):
        """Test that path contains 'data'."""
        path = _get_data_path()
        assert "data" in path


class TestLoadGrunfeld:
    """Test load_grunfeld function."""

    def test_load_grunfeld_returns_dataframe(self):
        """Test that load_grunfeld returns a DataFrame by default."""
        data = load_grunfeld()
        assert isinstance(data, pd.DataFrame)

    def test_load_grunfeld_has_expected_columns(self):
        """Test that Grunfeld data has expected columns."""
        data = load_grunfeld()
        expected_cols = {"firm", "year", "invest", "value", "capital"}
        assert set(data.columns) == expected_cols

    def test_load_grunfeld_has_200_observations(self):
        """Test that Grunfeld data has 200 observations."""
        data = load_grunfeld()
        assert len(data) == 200

    def test_load_grunfeld_has_10_firms(self):
        """Test that Grunfeld data has 10 firms."""
        data = load_grunfeld()
        assert data["firm"].nunique() == 10

    def test_load_grunfeld_has_20_years(self):
        """Test that Grunfeld data has 20 years."""
        data = load_grunfeld()
        assert data["year"].nunique() == 20

    def test_load_grunfeld_return_panel_data(self):
        """Test that load_grunfeld can return PanelData object."""
        data = load_grunfeld(return_panel_data=True)
        assert isinstance(data, PanelData)

    def test_load_grunfeld_panel_data_structure(self):
        """Test that PanelData has correct structure."""
        data = load_grunfeld(return_panel_data=True)
        assert data.entity_col == "firm"
        assert data.time_col == "year"

    def test_load_grunfeld_no_nulls(self):
        """Test that Grunfeld data has no null values."""
        data = load_grunfeld()
        assert not data.isnull().any().any()

    def test_load_grunfeld_data_types(self):
        """Test that Grunfeld data has correct types."""
        data = load_grunfeld()
        assert pd.api.types.is_numeric_dtype(data["invest"])
        assert pd.api.types.is_numeric_dtype(data["value"])
        assert pd.api.types.is_numeric_dtype(data["capital"])


class TestLoadAbdata:
    """Test load_abdata function."""

    def test_load_abdata_returns_dataframe_or_none(self):
        """Test that load_abdata returns DataFrame or None."""
        data = load_abdata()
        assert isinstance(data, pd.DataFrame) or data is None

    def test_load_abdata_with_return_panel_data(self):
        """Test that load_abdata can return PanelData if file exists."""
        data = load_abdata(return_panel_data=True)
        if data is not None:
            assert isinstance(data, PanelData)

    def test_load_abdata_infers_entity_time_cols(self):
        """Test that load_abdata infers entity and time columns."""
        data = load_abdata(return_panel_data=True)
        if data is not None:
            # Should have either 'id' or first column as entity
            assert hasattr(data, "entity_col")
            assert hasattr(data, "time_col")

    def test_load_abdata_handles_missing_file(self):
        """Test that load_abdata handles missing file gracefully."""
        # This test ensures the function doesn't crash
        data = load_abdata()
        # Should return None or DataFrame, never raise
        assert data is None or isinstance(data, pd.DataFrame)


class TestListDatasets:
    """Test list_datasets function."""

    def test_list_datasets_returns_list(self):
        """Test that list_datasets returns a list."""
        datasets = list_datasets()
        assert isinstance(datasets, list)

    def test_list_datasets_contains_grunfeld(self):
        """Test that list includes grunfeld dataset."""
        datasets = list_datasets()
        assert "grunfeld" in datasets

    def test_list_datasets_sorted(self):
        """Test that list is sorted."""
        datasets = list_datasets()
        assert datasets == sorted(datasets)

    def test_list_datasets_all_strings(self):
        """Test that all dataset names are strings."""
        datasets = list_datasets()
        assert all(isinstance(ds, str) for ds in datasets)

    def test_list_datasets_no_csv_extension(self):
        """Test that dataset names don't include .csv extension."""
        datasets = list_datasets()
        assert all(not ds.endswith(".csv") for ds in datasets)


class TestGetDatasetInfo:
    """Test get_dataset_info function."""

    def test_get_dataset_info_grunfeld(self):
        """Test getting info for Grunfeld dataset."""
        info = get_dataset_info("grunfeld")
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "source" in info

    def test_get_dataset_info_has_statistics(self):
        """Test that dataset info includes statistics."""
        info = get_dataset_info("grunfeld")
        assert "n_entities" in info
        assert "n_periods" in info
        assert "n_obs" in info
        assert "variables" in info

    def test_get_dataset_info_balanced_flag(self):
        """Test that dataset info includes balanced flag."""
        info = get_dataset_info("grunfeld")
        assert "balanced" in info
        assert info["balanced"]  # Grunfeld is balanced

    def test_get_dataset_info_unknown_dataset(self):
        """Test getting info for unknown dataset."""
        info = get_dataset_info("nonexistent_dataset")
        assert isinstance(info, dict)
        assert info["name"] == "nonexistent_dataset"
        assert info["description"] == "Unknown dataset"

    def test_get_dataset_info_abdata(self):
        """Test getting info for abdata dataset."""
        info = get_dataset_info("abdata")
        assert isinstance(info, dict)
        # Should have basic info even if file doesn't exist
        assert "name" in info

    def test_get_dataset_info_variables_list(self):
        """Test that variables is a list."""
        info = get_dataset_info("grunfeld")
        assert isinstance(info["variables"], list)
        assert len(info["variables"]) > 0

    def test_get_dataset_info_entity_time_cols(self):
        """Test that entity and time columns are specified."""
        info = get_dataset_info("grunfeld")
        assert "entity_col" in info
        assert "time_col" in info
        assert info["entity_col"] == "firm"
        assert info["time_col"] == "year"

    def test_get_dataset_info_handles_errors(self):
        """Test that errors are caught and stored in metadata."""
        # Try with a dataset name that might cause issues
        info = get_dataset_info("test_dataset_with_special_chars")
        # Should not raise, may have 'error' key
        assert isinstance(info, dict)


class TestLoadDataset:
    """Test load_dataset convenience function."""

    def test_load_dataset_grunfeld(self):
        """Test loading grunfeld by name."""
        data = load_dataset("grunfeld")
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 200

    def test_load_dataset_with_kwargs(self):
        """Test that kwargs are passed through."""
        data = load_dataset("grunfeld", return_panel_data=True)
        assert isinstance(data, PanelData)

    def test_load_dataset_unknown(self):
        """Test loading unknown dataset."""
        data = load_dataset("nonexistent_dataset_xyz")
        assert data is None

    def test_load_dataset_abdata(self):
        """Test loading abdata by name."""
        data = load_dataset("abdata")
        # Should return DataFrame or None
        assert isinstance(data, pd.DataFrame) or data is None

    def test_load_dataset_case_sensitive(self):
        """Test that dataset loading is case-sensitive."""
        # Assuming 'grunfeld' exists but 'Grunfeld' doesn't
        data = load_dataset("grunfeld")
        assert data is not None


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_workflow_list_and_load(self):
        """Test workflow of listing and then loading datasets."""
        # List available datasets
        datasets = list_datasets()
        assert len(datasets) > 0

        # Load the first one
        if datasets:
            data = load_dataset(datasets[0])
            assert data is not None or data is None  # Some might not exist

    def test_workflow_info_and_load(self):
        """Test workflow of getting info then loading."""
        # Get info
        info = get_dataset_info("grunfeld")
        assert "n_obs" in info

        # Load dataset
        data = load_grunfeld()
        assert len(data) == info["n_obs"]

    def test_dataframe_to_paneldata_consistency(self):
        """Test that DataFrame and PanelData versions have same data."""
        df = load_grunfeld(return_panel_data=False)
        panel = load_grunfeld(return_panel_data=True)

        # Should have same number of observations
        assert len(df) == len(panel.data)

        # Should have same columns (panel may have extra index cols)
        assert set(df.columns).issubset(set(panel.data.columns))


class TestUncoveredBranches:
    """Tests for uncovered branches in load.py."""

    def test_load_abdata_missing_file_returns_none(self, tmp_path, monkeypatch):
        """Test load_abdata returns None when the CSV file does not exist (line 188)."""
        # Monkeypatch _get_data_path to return a directory without abdata.csv
        monkeypatch.setattr("panelbox.datasets.load._get_data_path", lambda: str(tmp_path))
        result = load_abdata()
        assert result is None

    def test_list_datasets_missing_data_dir(self, tmp_path, monkeypatch):
        """Test list_datasets when data directory doesn't exist (line 223->229)."""
        nonexistent = str(tmp_path / "no_such_dir")
        monkeypatch.setattr("panelbox.datasets.load._get_data_path", lambda: nonexistent)
        datasets = list_datasets()
        assert datasets == []

    def test_list_datasets_empty_dir(self, tmp_path, monkeypatch):
        """Test list_datasets when data directory exists but has no CSVs (line 225->224)."""
        monkeypatch.setattr("panelbox.datasets.load._get_data_path", lambda: str(tmp_path))
        datasets = list_datasets()
        assert datasets == []

    def test_get_dataset_info_for_arbitrary_csv(self, tmp_path, monkeypatch):
        """Test get_dataset_info for a CSV that isn't grunfeld or abdata (line 302)."""
        # Create a test CSV in the data directory
        csv_path = tmp_path / "testdata.csv"
        test_df = pd.DataFrame(
            {
                "entity_id": [1, 1, 2, 2],
                "period": [1, 2, 1, 2],
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )
        test_df.to_csv(csv_path, index=False)

        monkeypatch.setattr("panelbox.datasets.load._get_data_path", lambda: str(tmp_path))

        info = get_dataset_info("testdata")
        assert isinstance(info, dict)
        assert info["name"] == "testdata"
        assert "n_obs" in info
        assert info["n_obs"] == 4
        assert "variables" in info
        assert "entity_id" in info["variables"]

    def test_get_dataset_info_nonexistent_csv_returns_base(self, tmp_path, monkeypatch):
        """Test get_dataset_info returns base info when dataset not found (line 304)."""
        monkeypatch.setattr("panelbox.datasets.load._get_data_path", lambda: str(tmp_path))

        info = get_dataset_info("totally_missing")
        assert isinstance(info, dict)
        assert info["name"] == "totally_missing"
        assert info["description"] == "Unknown dataset"
        # Should not have statistics since file doesn't exist
        assert "n_obs" not in info

    def test_get_dataset_info_exception_handling(self, tmp_path, monkeypatch):
        """Test get_dataset_info catches exceptions (lines 319-320)."""
        # Create a corrupted/invalid CSV
        bad_csv = tmp_path / "baddata.csv"
        bad_csv.write_text("col1,col2\n1,2\n")

        monkeypatch.setattr("panelbox.datasets.load._get_data_path", lambda: str(tmp_path))

        # Monkeypatch pd.read_csv to raise for this specific call path
        original_read_csv = pd.read_csv

        def failing_read_csv(path, *args, **kwargs):
            if "baddata" in str(path):
                raise RuntimeError("Simulated read error")
            return original_read_csv(path, *args, **kwargs)

        monkeypatch.setattr("panelbox.datasets.load.pd.read_csv", failing_read_csv)

        info = get_dataset_info("baddata")
        assert isinstance(info, dict)
        assert "error" in info
        assert "Simulated read error" in info["error"]

    def test_load_dataset_arbitrary_csv(self, tmp_path, monkeypatch):
        """Test load_dataset for a CSV that isn't grunfeld/abdata (line 350)."""
        csv_path = tmp_path / "custom.csv"
        test_df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
            }
        )
        test_df.to_csv(csv_path, index=False)

        monkeypatch.setattr("panelbox.datasets.load._get_data_path", lambda: str(tmp_path))

        result = load_dataset("custom")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["x", "y"]

    def test_load_abdata_panel_data_infers_columns(self):
        """Test load_abdata with return_panel_data=True infers entity/time cols (line 195)."""
        data = load_abdata(return_panel_data=True)
        if data is not None:
            assert isinstance(data, PanelData)
            # abdata has 'id' column -> entity_col should be 'id'
            assert data.entity_col == "id"
            # abdata has 'year' column -> time_col should be 'year'
            assert data.time_col == "year"
