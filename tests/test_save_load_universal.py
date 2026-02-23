"""
Tests for universal save/load across all model types.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_panel():
    """Simple panel data for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "entity": np.repeat(range(20), 10),
            "time": np.tile(range(10), 20),
            "y": np.random.randn(n),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
        }
    )


class TestPooledOLSSaveLoad:
    def test_roundtrip(self, simple_panel):
        from panelbox.models.static.pooled_ols import PooledOLS

        model = PooledOLS("y ~ x1 + x2", simple_panel, "entity", "time")
        results = model.fit()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            results.save(f.name)
            from panelbox.core.results import PanelResults

            loaded = PanelResults.load(f.name)

        assert np.allclose(results.params.values, loaded.params.values)
        assert np.allclose(results.std_errors.values, loaded.std_errors.values)


class TestGMMSaveLoad:
    def test_roundtrip(self):
        """Verify GMMResults still has save/load."""
        from panelbox.gmm.results import GMMResults

        assert hasattr(GMMResults, "save")
        assert hasattr(GMMResults, "load")


class TestPanelModelResultsSaveLoad:
    def test_has_save_load(self):
        """Verify PanelModelResults has save/load via SerializableMixin."""
        from panelbox.models.base import PanelModelResults

        assert hasattr(PanelModelResults, "save")
        assert hasattr(PanelModelResults, "load")


class TestHonoreResultsSaveLoad:
    def test_has_save_load(self):
        """Verify HonoreResults has save/load via SerializableMixin."""
        from panelbox.models.censored.honore import HonoreResults

        assert hasattr(HonoreResults, "save")
        assert hasattr(HonoreResults, "load")


class TestPanelVARResultSaveLoad:
    def test_has_save_load(self):
        """Verify PanelVARResult has save/load via SerializableMixin."""
        from panelbox.var.result import PanelVARResult

        assert hasattr(PanelVARResult, "save")
        assert hasattr(PanelVARResult, "load")


class TestSFResultSaveLoad:
    def test_has_save_load(self):
        """Verify SFResult has save/load via SerializableMixin."""
        from panelbox.frontier.result import SFResult

        assert hasattr(SFResult, "save")
        assert hasattr(SFResult, "load")

    def test_panel_sf_has_save_load(self):
        """Verify PanelSFResult inherits save/load from SFResult."""
        from panelbox.frontier.result import PanelSFResult

        assert hasattr(PanelSFResult, "save")
        assert hasattr(PanelSFResult, "load")


class TestQuantileResultsSaveLoad:
    def test_has_save_load(self):
        """Verify QuantilePanelResult has save/load via SerializableMixin."""
        from panelbox.models.quantile.base import QuantilePanelResult

        assert hasattr(QuantilePanelResult, "save")
        assert hasattr(QuantilePanelResult, "load")

    def test_pooled_quantile_has_save_load(self):
        """Verify PooledQuantileResults inherits save/load."""
        from panelbox.models.quantile.pooled import PooledQuantileResults

        assert hasattr(PooledQuantileResults, "save")
        assert hasattr(PooledQuantileResults, "load")


class TestLoadModelFunction:
    def test_load_model_convenience(self, simple_panel):
        """Test load_model() standalone function."""
        from panelbox.core.serialization import load_model
        from panelbox.models.static.pooled_ols import PooledOLS

        model = PooledOLS("y ~ x1 + x2", simple_panel, "entity", "time")
        results = model.fit()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            results.save(f.name)
            loaded = load_model(f.name)

        assert np.allclose(results.params.values, loaded.params.values)

    def test_load_nonexistent_raises(self):
        from panelbox.core.serialization import load_model

        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/path.pkl")

    def test_load_model_importable_from_panelbox(self):
        """Verify load_model is importable from top-level panelbox."""
        from panelbox import load_model

        assert callable(load_model)


class TestSaveLoadPreservesPredictCapability:
    """Verify that loaded models can still predict."""

    def test_pooled_ols_predict_after_load(self, simple_panel):
        from panelbox.core.serialization import load_model
        from panelbox.models.static.pooled_ols import PooledOLS

        model = PooledOLS("y ~ x1 + x2", simple_panel, "entity", "time")
        results = model.fit()

        newdata = pd.DataFrame({"x1": [1.0], "x2": [2.0], "y": [0]})
        preds_before = results.predict(newdata)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            results.save(f.name)
            loaded = load_model(f.name)

        preds_after = loaded.predict(newdata)
        assert np.allclose(preds_before, preds_after)


class TestVersionMetadata:
    """Verify that version metadata is saved."""

    def test_version_metadata_stored(self, simple_panel):
        from panelbox.core.serialization import load_model
        from panelbox.models.static.pooled_ols import PooledOLS

        model = PooledOLS("y ~ x1 + x2", simple_panel, "entity", "time")
        results = model.fit()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            results.save(f.name)
            loaded = load_model(f.name)

        assert hasattr(loaded, "_panelbox_version")
        assert hasattr(loaded, "_save_timestamp")
        assert loaded._panelbox_version is not None
        assert loaded._save_timestamp is not None

    def test_version_metadata_on_mixin(self, simple_panel):
        """Verify SerializableMixin save also stores version metadata."""
        from panelbox.core.serialization import load_model
        from panelbox.models.censored.honore import HonoreResults

        # Create a minimal HonoreResults instance
        result = HonoreResults(
            params=np.array([1.0, 2.0]),
            converged=True,
            n_iter=10,
            n_obs=100,
            n_entities=10,
            n_trimmed=5,
            exog_names=["x1", "x2"],
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            result.save(f.name)
            loaded = load_model(f.name)

        assert hasattr(loaded, "_panelbox_version")
        assert hasattr(loaded, "_save_timestamp")
