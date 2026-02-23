"""
Tests for PanelPipeline, ModelValidator, ModelRegistry.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from panelbox.production import ModelRegistry, ModelValidator, PanelPipeline


@pytest.fixture
def panel_data():
    np.random.seed(42)
    n_entities, n_periods = 20, 15
    entities = np.repeat(range(n_entities), n_periods)
    times = np.tile(range(n_periods), n_entities)
    x1 = np.random.randn(n_entities * n_periods)
    y = 1.0 + 2.0 * x1 + np.random.randn(n_entities * n_periods) * 0.5
    return pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": y,
            "x1": x1,
        }
    )


class TestPanelPipeline:
    def test_fit_predict(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        pipe = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        pipe.fit(panel_data)
        assert pipe.results is not None

        preds = pipe.predict(panel_data)
        assert len(preds) == len(panel_data)

    def test_save_load(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        pipe = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        pipe.fit(panel_data)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pipe.save(f.name)
            loaded = PanelPipeline.load(f.name)

        assert np.allclose(pipe.results.params.values, loaded.results.params.values)

    def test_validate(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        pipe = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        pipe.fit(panel_data)
        report = pipe.validate()
        assert report["passed"]

    def test_compare(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        pipe1 = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        pipe1.fit(panel_data)

        pipe2 = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        pipe2.fit(panel_data)

        comparison = pipe1.compare(pipe2)
        assert "diff" in comparison.columns

    def test_to_dict(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        pipe = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        pipe.fit(panel_data)
        d = pipe.to_dict()
        assert "params" in d
        assert "model_class" in d

    def test_refit(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        pipe = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        pipe.fit(panel_data)
        old_params = pipe.results.params.copy()

        # Refit with slightly different data
        new_data = panel_data.copy()
        new_data["y"] = new_data["y"] + np.random.randn(len(new_data)) * 0.1
        pipe.refit(new_data)
        # Params should be slightly different
        assert not np.allclose(pipe.results.params.values, old_params.values, atol=1e-6)

    def test_not_fitted_raises(self):
        from panelbox.models.static.pooled_ols import PooledOLS

        pipe = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            pipe.predict(pd.DataFrame({"x1": [1.0]}))


class TestModelValidator:
    def test_run_all(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        model = PooledOLS("y ~ x1", panel_data, "entity", "time")
        results = model.fit()
        validator = ModelValidator(results, training_data=panel_data)
        report = validator.run_all()
        assert report["passed"]
        assert "summary" in report


class TestModelRegistry:
    def test_register_and_load(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)

            pipe = PanelPipeline(
                model_class=PooledOLS,
                model_params={
                    "formula": "y ~ x1",
                    "entity_col": "entity",
                    "time_col": "time",
                },
            )
            pipe.fit(panel_data)

            version = registry.register(pipe, notes="First version")
            assert version == "v1"

            loaded = registry.load_version("v1")
            assert np.allclose(
                pipe.results.params.values,
                loaded.results.params.values,
            )

    def test_list_versions(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)

            pipe = PanelPipeline(
                model_class=PooledOLS,
                model_params={
                    "formula": "y ~ x1",
                    "entity_col": "entity",
                    "time_col": "time",
                },
            )
            pipe.fit(panel_data)

            registry.register(pipe, notes="v1")
            registry.register(pipe, notes="v2")

            versions = registry.list_versions()
            assert len(versions) == 2

    def test_load_latest(self, panel_data):
        from panelbox.models.static.pooled_ols import PooledOLS

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)

            pipe = PanelPipeline(
                model_class=PooledOLS,
                model_params={
                    "formula": "y ~ x1",
                    "entity_col": "entity",
                    "time_col": "time",
                },
            )
            pipe.fit(panel_data)

            registry.register(pipe)
            latest = registry.load_latest()
            assert latest.results is not None
