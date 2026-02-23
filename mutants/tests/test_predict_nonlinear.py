"""
Tests for predict(DataFrame) on non-linear models and SerializableMixin.
"""

import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def count_data():
    """Generate simple count panel data."""
    np.random.seed(42)
    n_entities, n_time = 20, 10
    n = n_entities * n_time
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    lam = np.exp(0.5 + 0.3 * x1 + 0.2 * x2)
    y = np.random.poisson(lam)
    return pd.DataFrame(
        {
            "entity": np.repeat(range(n_entities), n_time),
            "time": np.tile(range(n_time), n_entities),
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )


@pytest.fixture
def censored_data():
    """Generate censored panel data (left-censored at 0)."""
    np.random.seed(42)
    n_entities, n_time = 30, 8
    n = n_entities * n_time
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y_star = 1.0 + 0.5 * x1 + 0.3 * x2 + np.random.randn(n)
    y = np.maximum(y_star, 0.0)  # left-censored at 0
    return pd.DataFrame(
        {
            "entity": np.repeat(range(n_entities), n_time),
            "time": np.tile(range(n_time), n_entities),
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )


@pytest.fixture
def heckman_data():
    """Generate data for Heckman selection model."""
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    z1 = np.random.randn(n)  # exclusion restriction

    # Selection equation
    s_star = 0.5 + 0.3 * x1 + 0.4 * z1 + np.random.randn(n)
    selection = (s_star > 0).astype(int)

    # Outcome equation
    y = 1.0 + 0.5 * x1 + 0.3 * x2 + np.random.randn(n)

    const = np.ones(n)
    exog = np.column_stack([const, x1, x2])
    exog_selection = np.column_stack([const, x1, z1])

    return {
        "y": y,
        "exog": exog,
        "exog_selection": exog_selection,
        "selection": selection,
    }


# ---------------------------------------------------------------------------
# Count model tests
# ---------------------------------------------------------------------------


class TestCountPredict:
    """Test predict(DataFrame) for count models."""

    def test_pooled_poisson_predict_dataframe(self, count_data):
        """PooledPoisson predict() accepts DataFrame."""
        from panelbox.models.count.poisson import PooledPoisson

        X = count_data[["x1", "x2"]].values
        y = count_data["y"].values
        entity_id = count_data["entity"].values

        model = PooledPoisson(y, X, entity_id=entity_id)
        model.exog_names = ["x1", "x2"]
        result = model.fit(se_type="robust")

        # predict with DataFrame
        new_data = pd.DataFrame({"x1": [0.5, 1.0], "x2": [0.3, 0.6]})
        preds = model.predict(new_data, type="response")
        assert len(preds) == 2
        assert all(preds > 0)

        # predict via results object
        preds2 = result.predict(new_data)
        assert len(preds2) == 2
        assert all(preds2 > 0)

    def test_poisson_fe_predict_dataframe(self, count_data):
        """PoissonFixedEffects predict() accepts DataFrame."""
        from panelbox.models.count.poisson import PoissonFixedEffects

        X = count_data[["x1", "x2"]].values
        y = count_data["y"].values
        entity_id = count_data["entity"].values

        model = PoissonFixedEffects(y, X, entity_id=entity_id)
        model.exog_names = ["x1", "x2"]
        result = model.fit()

        new_data = pd.DataFrame({"x1": [0.5, 1.0], "x2": [0.3, 0.6]})

        # predict on model
        preds = model.predict(new_data, type="response")
        assert len(preds) == 2
        assert all(preds > 0)

        # predict on result
        preds2 = result.predict(new_data, type="response")
        assert len(preds2) == 2
        assert all(preds2 > 0)

    def test_negbin_predict_dataframe(self, count_data):
        """NegativeBinomial predict() accepts DataFrame."""
        from panelbox.models.count.negbin import NegativeBinomial

        X = count_data[["x1", "x2"]].values
        y = count_data["y"].values
        entity_id = count_data["entity"].values

        model = NegativeBinomial(y, X, entity_id=entity_id)
        model.exog_names = ["x1", "x2"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit()

        new_data = pd.DataFrame({"x1": [0.5, 1.0], "x2": [0.3, 0.6]})
        preds = result.predict(new_data, which="mean")
        assert len(preds) == 2
        assert all(preds > 0)

    def test_zip_predict_dataframe(self, count_data):
        """ZeroInflatedPoisson predict() accepts DataFrame."""
        from panelbox.models.count.zero_inflated import ZeroInflatedPoisson

        const = np.ones(len(count_data))
        X = np.column_stack([const, count_data[["x1", "x2"]].values])
        y = count_data["y"].values

        model = ZeroInflatedPoisson(
            y,
            X,
            exog_count_names=["const", "x1", "x2"],
            exog_inflate_names=["const", "x1", "x2"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit()

        # predict with DataFrame (count part)
        new_data = pd.DataFrame({"const": [1.0, 1.0], "x1": [0.5, 1.0], "x2": [0.3, 0.6]})
        preds = result.predict(exog_count=new_data, exog_inflate=new_data, which="mean")
        assert len(preds) == 2
        assert all(preds >= 0)

    def test_count_exog_names_stored(self, count_data):
        """Verify exog_names are stored in results."""
        from panelbox.models.count.poisson import PooledPoisson

        X = count_data[["x1", "x2"]].values
        y = count_data["y"].values

        model = PooledPoisson(y, X)
        model.exog_names = ["x1", "x2"]
        result = model.fit(se_type="robust")
        assert result.exog_names == ["x1", "x2"]

    def test_predict_missing_column_raises(self, count_data):
        """predict() with DataFrame missing a column raises ValueError."""
        from panelbox.models.count.poisson import PooledPoisson

        X = count_data[["x1", "x2"]].values
        y = count_data["y"].values

        model = PooledPoisson(y, X)
        model.exog_names = ["x1", "x2"]
        result = model.fit(se_type="robust")

        bad_data = pd.DataFrame({"x1": [0.5]})
        with pytest.raises(ValueError, match="Missing columns"):
            result.predict(bad_data)


# ---------------------------------------------------------------------------
# Censored model tests
# ---------------------------------------------------------------------------


class TestCensoredPredict:
    """Test predict(DataFrame) for censored models."""

    def test_pooled_tobit_predict_dataframe(self, censored_data):
        """PooledTobit predict() accepts DataFrame."""
        from panelbox.models.censored.tobit import PooledTobit

        const = np.ones(len(censored_data))
        X = np.column_stack([const, censored_data[["x1", "x2"]].values])
        y = censored_data["y"].values
        groups = censored_data["entity"].values

        model = PooledTobit(y, X, groups=groups, censoring_point=0.0)
        model.exog_names = ["const", "x1", "x2"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit()

        new_data = pd.DataFrame({"const": [1.0, 1.0], "x1": [0.5, 1.0], "x2": [0.3, 0.6]})
        preds = fitted_model.predict(new_data, pred_type="latent")
        assert len(preds) == 2

    def test_honore_predict_dataframe(self):
        """HonoreTrimmedEstimator predict() accepts DataFrame."""
        from panelbox.models.censored.honore import HonoreTrimmedEstimator

        np.random.seed(42)
        n_entities, n_time = 10, 5
        n = n_entities * n_time
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y_star = 0.5 * x1 + 0.3 * x2 + np.random.randn(n)
        y = np.maximum(y_star, 0.0)
        groups = np.repeat(range(n_entities), n_time)
        time = np.tile(range(n_time), n_entities)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = HonoreTrimmedEstimator(y, np.column_stack([x1, x2]), groups, time)
            model.exog_names = ["x1", "x2"]
            model.fit(verbose=False)

        new_data = pd.DataFrame({"x1": [0.5, 1.0], "x2": [0.3, 0.6]})
        preds = model.predict(new_data)
        assert len(preds) == 2

    def test_honore_results_predict_dataframe(self):
        """HonoreResults.predict() accepts DataFrame."""
        from panelbox.models.censored.honore import HonoreResults

        params = np.array([0.5, 0.3])
        results = HonoreResults(
            params=params,
            converged=True,
            n_iter=10,
            n_obs=100,
            n_entities=10,
            n_trimmed=5,
            exog_names=["x1", "x2"],
        )

        new_data = pd.DataFrame({"x1": [0.5, 1.0], "x2": [0.3, 0.6]})
        preds = results.predict(new_data)
        assert len(preds) == 2
        np.testing.assert_allclose(preds, new_data.values @ params)


# ---------------------------------------------------------------------------
# Heckman selection model tests
# ---------------------------------------------------------------------------


class TestHeckmanPredict:
    """Test predict(DataFrame) for Heckman model."""

    def test_heckman_predict_dataframe(self, heckman_data):
        """PanelHeckmanResult.predict() accepts DataFrame."""
        from panelbox.models.selection.heckman import PanelHeckman

        model = PanelHeckman(
            endog=heckman_data["y"],
            exog=heckman_data["exog"],
            selection=heckman_data["selection"],
            exog_selection=heckman_data["exog_selection"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit()

        # Predict with arrays (existing behavior)
        new_exog = np.array([[1.0, 0.5, 0.3], [1.0, 1.0, 0.6]])
        preds_arr = result.predict(exog=new_exog, type="unconditional")
        assert len(preds_arr) == 2


# ---------------------------------------------------------------------------
# SerializableMixin tests
# ---------------------------------------------------------------------------


class TestSaveLoadNonlinear:
    """Test save/load for non-PanelResults classes."""

    def test_panel_model_results_save_load(self, count_data):
        """PanelModelResults save/load roundtrip."""
        from panelbox.models.count.poisson import PooledPoisson

        X = count_data[["x1", "x2"]].values
        y = count_data["y"].values
        model = PooledPoisson(y, X)
        model.exog_names = ["x1", "x2"]
        result = model.fit(se_type="robust")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            result.save(f.name)
            loaded = type(result).load(f.name)

        np.testing.assert_allclose(loaded.params, result.params)

    def test_honore_results_save_load(self):
        """HonoreResults save/load roundtrip."""
        from panelbox.models.censored.honore import HonoreResults

        results = HonoreResults(
            params=np.array([0.5, 0.3]),
            converged=True,
            n_iter=10,
            n_obs=100,
            n_entities=10,
            n_trimmed=5,
            exog_names=["x1", "x2"],
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            results.save(f.name)
            loaded = HonoreResults.load(f.name)

        np.testing.assert_allclose(loaded.params, results.params)
        assert loaded.exog_names == ["x1", "x2"]

    def test_zip_results_save_load(self, count_data):
        """ZeroInflatedPoissonResult save/load roundtrip."""
        from panelbox.models.count.zero_inflated import ZeroInflatedPoisson

        const = np.ones(len(count_data))
        X = np.column_stack([const, count_data[["x1", "x2"]].values])
        y = count_data["y"].values

        model = ZeroInflatedPoisson(y, X, exog_count_names=["const", "x1", "x2"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            result.save(f.name)
            loaded = type(result).load(f.name)

        np.testing.assert_allclose(loaded.params, result.params)

    def test_poisson_fe_results_save_load(self, count_data):
        """PoissonFixedEffectsResults save/load roundtrip."""
        from panelbox.models.count.poisson import PoissonFixedEffects

        X = count_data[["x1", "x2"]].values
        y = count_data["y"].values
        entity_id = count_data["entity"].values

        model = PoissonFixedEffects(y, X, entity_id=entity_id)
        model.exog_names = ["x1", "x2"]
        result = model.fit()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            result.save(f.name)
            loaded = type(result).load(f.name)

        np.testing.assert_allclose(loaded.params, result.params)
        assert loaded.exog_names == ["x1", "x2"]

    def test_tobit_save_load(self, censored_data):
        """PooledTobit (model with SerializableMixin) save/load roundtrip."""
        from panelbox.models.censored.tobit import PooledTobit

        const = np.ones(len(censored_data))
        X = np.column_stack([const, censored_data[["x1", "x2"]].values])
        y = censored_data["y"].values

        model = PooledTobit(y, X, censoring_point=0.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = model.fit()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            fitted.save(f.name)
            loaded = type(fitted).load(f.name)

        np.testing.assert_allclose(loaded.params, fitted.params)
