"""ModelRegistry -- Simple file-based model versioning."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Simple file-based model registry for versioning.

    Stores model versions in a directory structure::

        registry_dir/
            registry.json          # Index of all versions
            v1/model.pkl           # Version 1
            v2/model.pkl           # Version 2
            ...

    Parameters
    ----------
    registry_dir : str or Path
        Directory to store model versions

    Examples
    --------
    >>> registry = ModelRegistry("./models/lgd")
    >>> registry.register(pipeline, version="v1", notes="Initial model")
    >>> registry.register(pipeline_v2, version="v2", notes="Added macro vars")
    >>> latest = registry.load_latest()
    >>> v1 = registry.load_version("v1")
    """

    def __init__(self, registry_dir: str | Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.registry_dir / "registry.json"
        self._index = self._load_index()

    def register(
        self,
        pipeline: Any,
        version: str | None = None,
        notes: str = "",
    ) -> str:
        """
        Register a new model version.

        Parameters
        ----------
        pipeline : PanelPipeline
            Fitted pipeline to register
        version : str, optional
            Version name (auto-generated if not provided)
        notes : str
            Human-readable notes about this version

        Returns
        -------
        str
            Version name
        """
        if version is None:
            version = f"v{len(self._index['versions']) + 1}"

        version_dir = self.registry_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save the pipeline
        model_path = version_dir / "model.pkl"
        pipeline.save(model_path)

        # Save metadata
        metadata = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "notes": notes,
            "model_class": pipeline.model_class.__name__,
            "name": pipeline.name,
        }

        if pipeline.results is not None:
            if hasattr(pipeline.results, "params"):
                metadata["n_params"] = len(pipeline.results.params)
            if hasattr(pipeline.results, "nobs"):
                metadata["nobs"] = pipeline.results.nobs

        meta_path = version_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))

        # Update index
        self._index["versions"].append(metadata)
        self._index["latest"] = version
        self._save_index()

        return version

    def load_version(self, version: str) -> Any:
        """Load a specific version."""
        from panelbox.production.pipeline import PanelPipeline

        model_path = self.registry_dir / version / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Version '{version}' not found at {model_path}")
        return PanelPipeline.load(model_path)

    def load_latest(self) -> Any:
        """Load the latest registered version."""
        if not self._index["latest"]:
            raise RuntimeError("No versions registered yet.")
        return self.load_version(self._index["latest"])

    def list_versions(self) -> list[dict]:
        """List all registered versions."""
        return self._index["versions"]

    def delete_version(self, version: str) -> None:
        """Delete a version from the registry."""
        version_dir = self.registry_dir / version
        if version_dir.exists():
            shutil.rmtree(version_dir)

        self._index["versions"] = [v for v in self._index["versions"] if v["version"] != version]
        if self._index["latest"] == version:
            if self._index["versions"]:
                self._index["latest"] = self._index["versions"][-1]["version"]
            else:
                self._index["latest"] = None
        self._save_index()

    def _load_index(self) -> dict:
        if self._index_file.exists():
            return json.loads(self._index_file.read_text())
        return {"versions": [], "latest": None}

    def _save_index(self):
        self._index_file.write_text(json.dumps(self._index, indent=2, default=str))
