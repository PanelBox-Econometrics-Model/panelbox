"""Tests for BaseResult class."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from panelbox.experiment.results.base import BaseResult


class ConcreteResult(BaseResult):
    """Concrete implementation of BaseResult for testing."""

    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        self.data = data or {}

    def to_dict(self):
        """Convert to dict."""
        return {"data": self.data}

    def summary(self):
        """Return summary."""
        return "Test Result Summary"


class TestBaseResult:
    """Tests for BaseResult class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        result = ConcreteResult()

        assert isinstance(result.timestamp, datetime)
        assert result.metadata == {}

    def test_init_with_timestamp(self):
        """Test initialization with custom timestamp."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        result = ConcreteResult(timestamp=ts)

        assert result.timestamp == ts

    def test_init_with_metadata(self):
        """Test initialization with metadata."""
        metadata = {"key": "value"}
        result = ConcreteResult(metadata=metadata)

        assert result.metadata == metadata

    def test_to_dict(self):
        """Test to_dict method."""
        result = ConcreteResult(data={"test": 123})
        data = result.to_dict()

        assert data == {"data": {"test": 123}}

    def test_summary(self):
        """Test summary method."""
        result = ConcreteResult()
        summary = result.summary()

        assert summary == "Test Result Summary"

    def test_save_json(self):
        """Test save_json method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ConcreteResult(data={"test": 123})
            path = Path(tmpdir) / "result.json"

            output_path = result.save_json(str(path))

            assert output_path.exists()
            assert output_path == path

    def test_save_json_with_no_indent(self):
        """Test save_json with no indentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ConcreteResult(data={"test": 123})
            path = Path(tmpdir) / "result.json"

            output_path = result.save_json(str(path), indent=None)

            assert output_path.exists()

    def test_repr(self):
        """Test string representation."""
        result = ConcreteResult()
        repr_str = repr(result)

        assert "ConcreteResult" in repr_str
        assert "timestamp" in repr_str
        assert "metadata" in repr_str
