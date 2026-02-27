"""Tests for ModelRegistry -- register, load, delete, auto-version."""

import pytest

from panelbox.production.versioning import ModelRegistry


class TestRegister:
    def test_register_auto_version(self, tmp_path, fitted_pipeline):
        """Test register auto-generates version."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        version = registry.register(fitted_pipeline)
        assert version == "v1"

    def test_register_explicit_version(self, tmp_path, fitted_pipeline):
        """Test register with explicit version name."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        version = registry.register(fitted_pipeline, version="release_1")
        assert version == "release_1"

    def test_register_with_notes(self, tmp_path, fitted_pipeline):
        """Test register with notes metadata."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        registry.register(fitted_pipeline, version="v1", notes="Initial model")
        versions = registry.list_versions()
        assert versions[0]["notes"] == "Initial model"

    def test_register_saves_model_metadata(self, tmp_path, fitted_pipeline):
        """Test register saves model class and param info."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        registry.register(fitted_pipeline, version="v1")
        versions = registry.list_versions()
        assert versions[0]["model_class"] == "PooledOLS"
        assert "n_params" in versions[0]


class TestLoadVersion:
    def test_load_version_nonexistent_raises(self, tmp_path):
        """Test load_version raises for nonexistent version."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        with pytest.raises(FileNotFoundError, match="not found"):
            registry.load_version("v_nonexistent")

    def test_load_latest_empty_registry_raises(self, tmp_path):
        """Test load_latest raises on empty registry."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        with pytest.raises(RuntimeError, match="No versions registered"):
            registry.load_latest()

    def test_load_version_roundtrip(self, tmp_path, fitted_pipeline):
        """Test save and load version preserves results."""
        import numpy as np

        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        registry.register(fitted_pipeline, version="v1")
        loaded = registry.load_version("v1")
        assert np.allclose(
            fitted_pipeline.results.params.values,
            loaded.results.params.values,
        )


class TestDeleteVersion:
    def test_delete_version(self, tmp_path, fitted_pipeline):
        """Test deleting a version removes it from index."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        registry.register(fitted_pipeline, version="v1")
        registry.register(fitted_pipeline, version="v2")
        registry.delete_version("v1")
        versions = registry.list_versions()
        version_names = [v["version"] for v in versions]
        assert "v1" not in version_names
        assert "v2" in version_names

    def test_delete_latest_updates_pointer(self, tmp_path, fitted_pipeline):
        """Test deleting latest version updates the latest pointer."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        registry.register(fitted_pipeline, version="v1")
        registry.register(fitted_pipeline, version="v2")
        registry.delete_version("v2")
        # Latest should now point to v1 -- verify by loading latest
        latest = registry.load_latest()
        assert latest is not None

    def test_delete_all_versions(self, tmp_path, fitted_pipeline):
        """Test deleting all versions leaves empty registry."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        registry.register(fitted_pipeline, version="v1")
        registry.delete_version("v1")
        assert len(registry.list_versions()) == 0
        with pytest.raises(RuntimeError, match="No versions registered"):
            registry.load_latest()


class TestListAndIndex:
    def test_load_index_new_registry(self, tmp_path):
        """Test loading index on new registry creates empty index."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        versions = registry.list_versions()
        assert len(versions) == 0

    def test_auto_version_incrementing(self, tmp_path, fitted_pipeline):
        """Test auto-version increments correctly."""
        registry = ModelRegistry(registry_dir=str(tmp_path / "registry"))
        v1 = registry.register(fitted_pipeline)
        v2 = registry.register(fitted_pipeline)
        assert v1 == "v1"
        assert v2 == "v2"
        assert len(registry.list_versions()) == 2
