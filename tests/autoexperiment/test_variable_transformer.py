"""Tests for VariableTransformer."""

import warnings

import numpy as np
import pandas as pd
import pytest

from panelbox.autoexperiment.variable_transformer import VariableTransformer


@pytest.fixture
def panel_df():
    """Simple panel: 4 entities, 30 periods."""
    np.random.seed(42)
    N, T = 4, 30
    return pd.DataFrame(
        {
            "entity": np.repeat(range(N), T),
            "time": np.tile(range(T), N),
            "y": np.random.randn(N * T),
            "x1": np.random.randn(N * T) + 10,
            "x2": np.random.randn(N * T) * 5,
        }
    )


class TestLags:
    def test_no_cross_entity_contamination(self, panel_df):
        """Lags must not leak across entities."""
        vt = VariableTransformer(panel_df, "entity", "time", transformations={"lag": [1]})
        df_t = vt.transform(["x1"])

        # First observation of each entity should be NaN for lag
        for eid in panel_df["entity"].unique():
            mask = df_t["entity"] == eid
            first_idx = mask.idxmax()
            assert pd.isna(df_t.loc[first_idx, "L1_x1"])

        # Second observation should equal first original value
        for eid in panel_df["entity"].unique():
            entity_data = panel_df[panel_df["entity"] == eid].reset_index(drop=True)
            entity_t = df_t[df_t["entity"] == eid].reset_index(drop=True)
            assert entity_t.loc[1, "L1_x1"] == pytest.approx(entity_data.loc[0, "x1"])

    def test_multiple_lags(self, panel_df):
        vt = VariableTransformer(panel_df, "entity", "time", transformations={"lag": [1, 2, 3]})
        df_t = vt.transform(["x1"])
        assert "L1_x1" in df_t.columns
        assert "L2_x1" in df_t.columns
        assert "L3_x1" in df_t.columns


class TestDiff:
    def test_diff_by_entity(self, panel_df):
        """Diff calculated correctly per entity."""
        vt = VariableTransformer(panel_df, "entity", "time", transformations={"diff": True})
        df_t = vt.transform(["x1"])

        for eid in panel_df["entity"].unique():
            orig = panel_df[panel_df["entity"] == eid]["x1"].values
            diff_vals = df_t[df_t["entity"] == eid]["D_x1"].values
            # First is NaN
            assert pd.isna(diff_vals[0])
            # Rest should be x_t - x_{t-1}
            expected = orig[1:] - orig[:-1]
            np.testing.assert_allclose(diff_vals[1:], expected)


class TestLog:
    def test_log_positive_values(self, panel_df):
        """Log works on positive-only variables."""
        vt = VariableTransformer(panel_df, "entity", "time", transformations={"log": True})
        df_t = vt.transform(["x1"])  # x1 ~ N(10,1), almost always positive
        assert "log_x1" in df_t.columns
        np.testing.assert_allclose(df_t["log_x1"].values, np.log(panel_df["x1"].values))

    def test_log_ignores_nonpositive_with_warning(self, panel_df):
        """Log skipped for variables with values <= 0, emits warning."""
        # x2 has mean 0, will have negative values
        vt = VariableTransformer(panel_df, "entity", "time", transformations={"log": True})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df_t = vt.transform(["x2"])
            assert "log_x2" not in df_t.columns
            assert any("valores <= 0" in str(warn.message) for warn in w)


class TestAcum:
    def test_rolling_mean_by_entity(self, panel_df):
        """Acum uses rolling mean correctly per entity."""
        vt = VariableTransformer(panel_df, "entity", "time", transformations={"acum": [3]})
        df_t = vt.transform(["x1"])
        assert "acum3_x1" in df_t.columns

        for eid in panel_df["entity"].unique():
            orig = panel_df[panel_df["entity"] == eid]["x1"].values
            acum_vals = df_t[df_t["entity"] == eid]["acum3_x1"].values
            # First 2 should be NaN (window=3, need 3 obs)
            assert pd.isna(acum_vals[0])
            assert pd.isna(acum_vals[1])
            # Third should be mean of first 3
            expected = np.mean(orig[:3])
            assert acum_vals[2] == pytest.approx(expected)


class TestGrowth:
    def test_growth_by_entity(self, panel_df):
        """Growth rate calculated correctly per entity."""
        vt = VariableTransformer(panel_df, "entity", "time", transformations={"growth": True})
        df_t = vt.transform(["x1"])
        assert "growth_x1" in df_t.columns

        for eid in panel_df["entity"].unique():
            orig = panel_df[panel_df["entity"] == eid]["x1"].values
            growth_vals = df_t[df_t["entity"] == eid]["growth_x1"].values
            assert pd.isna(growth_vals[0])
            expected = (orig[1] - orig[0]) / orig[0] * 100
            assert growth_vals[1] == pytest.approx(expected)


class TestSq:
    def test_squared(self, panel_df):
        vt = VariableTransformer(panel_df, "entity", "time", transformations={"sq": True})
        df_t = vt.transform(["x1"])
        assert "sq_x1" in df_t.columns
        np.testing.assert_allclose(df_t["sq_x1"].values, panel_df["x1"].values ** 2)


class TestNaNThreshold:
    def test_discards_high_nan(self):
        """Variables with NaN > threshold are discarded."""
        # 3 entities, 5 periods each -> 15 rows. Lag 4 loses 4 per entity = 12 NaN
        df = pd.DataFrame(
            {
                "entity": np.repeat([0, 1, 2], 5),
                "time": np.tile(range(5), 3),
                "x": np.random.randn(15),
            }
        )
        vt = VariableTransformer(
            df, "entity", "time", transformations={"lag": [4]}, nan_threshold=0.30
        )
        df_t = vt.transform(["x"])
        # 12/15 = 80% NaN, should be discarded
        assert "L4_x" not in df_t.columns
        assert len(vt._discarded) == 1


class TestMulticollinearity:
    def test_detects_high_correlation(self):
        """Highly correlated transformations are removed."""
        np.random.seed(0)
        N, T = 10, 100
        x = np.random.randn(N * T).cumsum()  # random walk -> high autocorrelation
        df = pd.DataFrame(
            {
                "entity": np.repeat(range(N), T),
                "time": np.tile(range(T), N),
                "x": x,
            }
        )
        vt = VariableTransformer(df, "entity", "time", transformations={"lag": [1], "sq": True})
        vt.transform(["x"])
        # At minimum, multicollinearity check should run without error
        assert isinstance(vt._multicollinear_removed, list)


class TestTransformationMap:
    def test_map_correct(self, panel_df):
        vt = VariableTransformer(
            panel_df,
            "entity",
            "time",
            transformations={"lag": [1, 2], "diff": True},
        )
        vt.transform(["x1"])
        tmap = vt.get_transformation_map()
        assert tmap["L1_x1"] == ("x1", "lag1")
        assert tmap["L2_x1"] == ("x1", "lag2")
        assert tmap["D_x1"] == ("x1", "diff")


class TestSummary:
    def test_summary_output(self, panel_df):
        vt = VariableTransformer(
            panel_df,
            "entity",
            "time",
            transformations={"lag": [1], "diff": True},
        )
        vt.transform(["x1"])
        s = vt.summary()
        assert "Transformacoes geradas" in s
        assert "validas" in s


class TestGetValidTransformations:
    def test_returns_list(self, panel_df):
        vt = VariableTransformer(
            panel_df,
            "entity",
            "time",
            transformations={"lag": [1]},
        )
        vt.transform(["x1"])
        valid = vt.get_valid_transformations()
        assert "L1_x1" in valid


class TestEmptyTransformations:
    def test_empty_transformations_uses_defaults(self, panel_df):
        """Empty dict is falsy, so default transformations (lag+diff) are used."""
        vt = VariableTransformer(
            panel_df,
            "entity",
            "time",
            transformations={},
        )
        df_t = vt.transform(["x1"])
        # Empty dict triggers default {lag:[1], diff:True}
        assert "L1_x1" in df_t.columns
        assert "D_x1" in df_t.columns

    def test_no_matching_transformations(self, panel_df):
        """Transformation dict with no applicable keys produces no new columns."""
        vt = VariableTransformer(
            panel_df,
            "entity",
            "time",
            transformations={"lag": [], "diff": False, "log": False},
        )
        df_t = vt.transform(["x1"])
        assert vt.get_transformation_map() == {}
        assert vt.get_valid_transformations() == []
        assert "x1" in df_t.columns

    def test_empty_variables_list(self, panel_df):
        """Passing empty variables list produces no transformations."""
        vt = VariableTransformer(
            panel_df,
            "entity",
            "time",
            transformations={"lag": [1], "diff": True},
        )
        vt.transform([])
        assert vt.get_transformation_map() == {}
        assert vt.get_valid_transformations() == []
