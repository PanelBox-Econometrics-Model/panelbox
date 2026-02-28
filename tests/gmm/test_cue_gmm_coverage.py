"""
Coverage tests for ContinuousUpdatedGMM (panelbox.gmm.cue_gmm).

Targets uncovered lines from the coverage report to improve branch and
statement coverage for the CUE-GMM estimator.
"""

from __future__ import annotations

import logging
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from panelbox.gmm.cue_gmm import ContinuousUpdatedGMM
from panelbox.gmm.results import GMMResults, TestResult


def _make_panel_data(n_entities=5, n_periods=10, seed=42, n_exog=1, n_instruments=2):
    """
    Create a small panel dataset with MultiIndex for testing.

    DGP:
        y = 1.0 + 2.0 * x1 [+ 0.5 * x2] + eps
        x1 = 0.5 * z1 + 0.4 * z2 [+ 0.3 * z3] + v
        endogeneity: eps += 0.3 * v
    """
    np.random.seed(seed)
    n = n_entities * n_periods

    # Build instruments
    z_data = {}
    z_names = []
    for j in range(n_instruments):
        name = f"z{j + 1}"
        z_names.append(name)
        z_data[name] = np.random.normal(0, 1, n)

    # Endogenous part
    v = np.random.normal(0, 1, n)
    x1 = 0.5 * z_data["z1"] + 0.4 * z_data["z2"] + v

    eps = np.random.normal(0, 0.5, n) + 0.3 * v
    y = 1.0 + 2.0 * x1 + eps

    exog_names = ["x1"]
    col_data = {"y": y, "x1": x1}

    if n_exog >= 2:
        x2 = np.random.normal(0, 1, n)
        y += 0.5 * x2
        col_data["x2"] = x2
        exog_names.append("x2")

    col_data.update(z_data)

    entities = np.repeat(np.arange(n_entities), n_periods)
    times = np.tile(np.arange(n_periods), n_entities)

    df = pd.DataFrame(col_data)
    df["entity"] = entities
    df["time"] = times
    df = df.set_index(["entity", "time"])

    return df, exog_names, z_names


class TestCUEGMMCoverage:
    """Coverage tests targeting uncovered lines in cue_gmm.py."""

    # ------------------------------------------------------------------
    # Fixture: standard panel data
    # ------------------------------------------------------------------
    @pytest.fixture
    def panel_data(self):
        """Standard small panel dataset."""
        df, exog, instr = _make_panel_data()
        return df, exog, instr

    @pytest.fixture
    def fitted_model(self, panel_data):
        """A pre-fitted CUE-GMM model for reuse."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            weighting="hac",
            bandwidth="auto",
        )
        model.fit()
        return model

    # ------------------------------------------------------------------
    # Line 215: bandwidth must be int or 'auto' (TypeError)
    # ------------------------------------------------------------------
    def test_bandwidth_bad_type_raises_type_error(self, panel_data):
        """Passing a float for bandwidth must raise TypeError."""
        df, exog, instr = panel_data
        with pytest.raises(TypeError, match="bandwidth must be int or 'auto'"):
            ContinuousUpdatedGMM(
                data=df,
                dep_var="y",
                exog_vars=exog,
                instruments=instr,
                bandwidth=3.5,
            )

    # ------------------------------------------------------------------
    # Line 219: bandwidth must be non-negative (ValueError)
    # ------------------------------------------------------------------
    def test_bandwidth_negative_raises_value_error(self, panel_data):
        """Negative integer bandwidth must raise ValueError."""
        df, exog, instr = panel_data
        with pytest.raises(ValueError, match="bandwidth must be non-negative"):
            ContinuousUpdatedGMM(
                data=df,
                dep_var="y",
                exog_vars=exog,
                instruments=instr,
                bandwidth=-1,
            )

    # ------------------------------------------------------------------
    # Line 229: bootstrap_method validation when se_type='bootstrap'
    #           AND n_bootstrap is invalid
    # ------------------------------------------------------------------
    def test_bootstrap_n_bootstrap_invalid(self, panel_data):
        """n_bootstrap must be a positive integer when se_type='bootstrap'."""
        df, exog, instr = panel_data
        with pytest.raises(ValueError, match="n_bootstrap must be a positive integer"):
            ContinuousUpdatedGMM(
                data=df,
                dep_var="y",
                exog_vars=exog,
                instruments=instr,
                se_type="bootstrap",
                n_bootstrap=-5,
            )

    def test_bootstrap_method_invalid(self, panel_data):
        """Invalid bootstrap_method must raise ValueError."""
        df, exog, instr = panel_data
        with pytest.raises(ValueError, match="bootstrap_method must be one of"):
            ContinuousUpdatedGMM(
                data=df,
                dep_var="y",
                exog_vars=exog,
                instruments=instr,
                se_type="bootstrap",
                n_bootstrap=10,
                bootstrap_method="wild",
            )

    # ------------------------------------------------------------------
    # Line 239: data must be pandas DataFrame (TypeError)
    # ------------------------------------------------------------------
    def test_data_not_dataframe_raises_type_error(self):
        """Passing a non-DataFrame for data must raise TypeError."""
        with pytest.raises(TypeError, match="data must be pandas DataFrame"):
            ContinuousUpdatedGMM(
                data={"y": [1, 2], "x": [3, 4], "z": [5, 6]},
                dep_var="y",
                exog_vars=["x"],
                instruments=["z"],
            )

    # ------------------------------------------------------------------
    # Line 245: Variables not found in data (ValueError)
    # ------------------------------------------------------------------
    def test_missing_variables_raises_value_error(self, panel_data):
        """Referencing columns not present in data must raise ValueError."""
        df, _exog, instr = panel_data
        with pytest.raises(ValueError, match="Variables not found in data"):
            ContinuousUpdatedGMM(
                data=df,
                dep_var="y",
                exog_vars=["nonexistent_var"],
                instruments=instr,
            )

    # ------------------------------------------------------------------
    # Lines 309-318: LinAlgError in _criterion (singular W)
    # ------------------------------------------------------------------
    def test_criterion_linalg_error_regularize_true(self, panel_data):
        """When W is singular in _criterion and regularize=True, falls back to regularized solve."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            regularize=True,
        )

        # Patch linalg.solve to raise LinAlgError first time, then work
        original_solve = linalg.solve
        call_count = [0]

        def patched_solve(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                raise linalg.LinAlgError("singular matrix")
            return original_solve(*args, **kwargs)

        params = np.zeros(model.k)
        with patch("panelbox.gmm.cue_gmm.linalg.solve", side_effect=patched_solve):
            q = model._criterion(params)
        assert np.isfinite(q)

    def test_criterion_linalg_error_regularize_false(self, panel_data):
        """When W is singular in _criterion and regularize=False, falls back to lstsq."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            regularize=False,
        )

        params = np.zeros(model.k)
        with patch(
            "panelbox.gmm.cue_gmm.linalg.solve",
            side_effect=linalg.LinAlgError("singular matrix"),
        ):
            q = model._criterion(params)
        assert np.isfinite(q)

    # ------------------------------------------------------------------
    # Line 351: homoskedastic weighting branch
    # ------------------------------------------------------------------
    def test_homoskedastic_weighting(self, panel_data):
        """Homoskedastic weighting must produce a valid fit."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            weighting="homoskedastic",
        )
        model.fit()
        assert model.converged_
        assert model.params_ is not None
        assert len(model.params_) == model.k

    # ------------------------------------------------------------------
    # Line 362: Unknown weighting raise
    # ------------------------------------------------------------------
    def test_unknown_weighting_in_compute(self, panel_data):
        """
        If weighting is somehow set to an invalid value after init
        (bypassing validation), _compute_weighting_matrix must raise.
        """
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        # Manually override after validation
        model.weighting = "invalid_method"
        residuals = model.y - model.X @ np.zeros((model.k, 1))
        with pytest.raises(ValueError, match="Unknown weighting"):
            model._compute_weighting_matrix(np.zeros(model.k), residuals)

    # ------------------------------------------------------------------
    # Line 392: fixed int bandwidth in _compute_hac_variance
    # ------------------------------------------------------------------
    def test_fixed_int_bandwidth(self, panel_data):
        """Using a fixed integer bandwidth must work without errors."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            weighting="hac",
            bandwidth=3,
        )
        model.fit()
        assert model.converged_
        assert model.params_ is not None

    # ------------------------------------------------------------------
    # Lines 436-439: cluster weighting without MultiIndex (warning + fallback)
    # ------------------------------------------------------------------
    def test_cluster_without_multiindex_warns(self):
        """Cluster weighting on non-MultiIndex data must warn and fall back."""
        np.random.seed(42)
        n = 50
        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)
        v = np.random.normal(0, 1, n)
        x = 0.5 * z1 + 0.4 * z2 + v
        eps = np.random.normal(0, 0.5, n) + 0.3 * v
        y = 1.0 + 2.0 * x + eps

        # RangeIndex (no MultiIndex)
        df = pd.DataFrame({"y": y, "x1": x, "z1": z1, "z2": z2})

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = ContinuousUpdatedGMM(
                data=df,
                dep_var="y",
                exog_vars=["x1"],
                instruments=["z1", "z2"],
                weighting="cluster",
            )
            model.fit()

        warning_messages = [str(ww.message) for ww in w]
        assert any("MultiIndex" in msg for msg in warning_messages)
        assert model.params_ is not None

    # ------------------------------------------------------------------
    # Lines 507, 515: Performance warnings (>50 moments, >10000 obs)
    # ------------------------------------------------------------------
    def test_warn_many_moments(self, panel_data):
        """Performance warning when more than 50 moments."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        # Artificially set m > 50 to trigger the warning
        model.m = 51
        with pytest.warns(UserWarning, match="CUE-GMM with >50 moments"):
            model.fit()

    def test_warn_large_sample(self, panel_data):
        """Performance warning when more than 10000 observations."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        # Artificially set n > 10000 to trigger the warning
        model.n = 10001
        # model.fit() may raise because n doesn't match actual data,
        # but the warning should still fire before that point.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                model.fit()
            except Exception:
                pass
        assert any("CUE-GMM with N>10,000" in str(x.message) for x in w)

    # ------------------------------------------------------------------
    # Line 535: start_params length validation
    # ------------------------------------------------------------------
    def test_start_params_wrong_length(self, panel_data):
        """start_params with wrong length must raise ValueError."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        wrong_params = np.zeros(model.k + 3)
        with pytest.raises(ValueError, match="start_params must have length"):
            model.fit(start_params=wrong_params)

    # ------------------------------------------------------------------
    # Line 555: Non-convergence warning
    # ------------------------------------------------------------------
    def test_non_convergence_warning(self, panel_data):
        """Model with max_iter=1 likely will not converge and should warn."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            max_iter=1,
            tol=1e-30,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit()

        # Check if non-convergence warning was raised
        non_conv_warnings = [ww for ww in w if "did not converge" in str(ww.message)]
        # It is possible it converges in 1 iteration for simple data.
        # If it did converge, the test is still valid -- we just verify no crash.
        if not model.converged_:
            assert len(non_conv_warnings) > 0

    # ------------------------------------------------------------------
    # Lines 561->569: Bootstrap SE path (se_type='bootstrap')
    # ------------------------------------------------------------------
    def test_bootstrap_se_residual(self, panel_data):
        """Fitting with se_type='bootstrap' and residual method must compute bootstrap SEs."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            se_type="bootstrap",
            n_bootstrap=5,
            bootstrap_method="residual",
        )
        model.fit()
        assert model.bootstrap_params_ is not None
        assert model.bootstrap_params_.shape == (5, model.k)
        assert model.bse_ is not None
        assert len(model.bse_) == model.k

    # ------------------------------------------------------------------
    # Lines 607-615: LinAlgError in _compute_variance
    # ------------------------------------------------------------------
    def test_compute_variance_linalg_error_regularize_true(self, panel_data):
        """LinAlgError in _compute_variance with regularize=True triggers fallback."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            regularize=True,
        )
        # Fit the model first so params_ is set
        model.fit()

        # Patch linalg.solve to fail only on the first call (line 605 with assume_a="pos")
        # and succeed on the second call (line 612, the regularized fallback)
        original_solve = linalg.solve
        call_count = [0]

        def selective_raise(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise linalg.LinAlgError("singular matrix")
            return original_solve(*args, **kwargs)

        with (
            warnings.catch_warnings(record=True) as w,
            patch("panelbox.gmm.cue_gmm.linalg.solve", side_effect=selective_raise),
            patch(
                "panelbox.gmm.cue_gmm.linalg.inv",
                side_effect=linalg.LinAlgError("singular"),
            ),
        ):
            warnings.simplefilter("always")
            vcov = model._compute_variance()
        assert vcov is not None
        assert vcov.shape == (model.k, model.k)
        singular_warnings = [ww for ww in w if "Singular variance matrix" in str(ww.message)]
        assert len(singular_warnings) > 0

    def test_compute_variance_linalg_error_regularize_false(self, panel_data):
        """LinAlgError in _compute_variance with regularize=False uses lstsq."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            regularize=False,
        )
        model.fit()

        def raise_linalg(*args, **kwargs):
            raise linalg.LinAlgError("singular matrix")

        with (
            warnings.catch_warnings(record=True),
            patch("panelbox.gmm.cue_gmm.linalg.solve", side_effect=raise_linalg),
            patch("panelbox.gmm.cue_gmm.linalg.inv", side_effect=raise_linalg),
        ):
            warnings.simplefilter("always")
            vcov = model._compute_variance()
        assert vcov is not None
        assert vcov.shape == (model.k, model.k)

    # ------------------------------------------------------------------
    # Line 658: verbose bootstrap progress logging
    # ------------------------------------------------------------------
    def test_bootstrap_verbose_logging(self, panel_data, caplog):
        """Verbose bootstrap must log progress at every 100th iteration."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            se_type="bootstrap",
            n_bootstrap=100,
            bootstrap_method="residual",
        )

        with caplog.at_level(logging.INFO, logger="panelbox.gmm.cue_gmm"):
            model.fit(verbose=True)

        # At iteration 100 (b=99, b+1=100), it should log
        assert any("Bootstrap iteration 100" in record.message for record in caplog.records)

    # ------------------------------------------------------------------
    # Lines 673->682: pairs bootstrap method
    # ------------------------------------------------------------------
    def test_bootstrap_pairs_method(self, panel_data):
        """Pairs bootstrap must produce valid bootstrap parameter samples."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            se_type="bootstrap",
            n_bootstrap=5,
            bootstrap_method="pairs",
        )
        model.fit()
        assert model.bootstrap_params_ is not None
        assert model.bootstrap_params_.shape == (5, model.k)

    # ------------------------------------------------------------------
    # Lines 686-690: Bootstrap iteration failure catch
    # ------------------------------------------------------------------
    def test_bootstrap_iteration_failure_fallback(self, panel_data):
        """If a bootstrap iteration fails, it should fall back to original params."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            se_type="bootstrap",
            n_bootstrap=3,
            bootstrap_method="residual",
        )
        # First fit the model normally to get params_
        model_normal = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        model_normal.fit()

        # Now manually set params_ on the bootstrap model and call _compute_bootstrap
        model.params_ = model_normal.params_.copy()

        with (
            patch.object(
                model,
                "_estimate_bootstrap_sample",
                side_effect=RuntimeError("Bootstrap sample failed"),
            ),
            warnings.catch_warnings(record=True),
        ):
            warnings.simplefilter("always")
            boot_params = model._compute_bootstrap(verbose=True)

        # All rows should fall back to original params
        for b in range(3):
            np.testing.assert_array_equal(boot_params[b], model.params_)

    # ------------------------------------------------------------------
    # Line 799: n_groups for non-MultiIndex data
    # ------------------------------------------------------------------
    def test_n_groups_non_multiindex(self):
        """For non-MultiIndex data, n_groups should equal n (total obs)."""
        np.random.seed(42)
        n = 30
        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)
        v = np.random.normal(0, 1, n)
        x = 0.5 * z1 + 0.4 * z2 + v
        eps = np.random.normal(0, 0.5, n) + 0.3 * v
        y = 1.0 + 2.0 * x + eps

        df = pd.DataFrame({"y": y, "x1": x, "z1": z1, "z2": z2})

        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=["x1"],
            instruments=["z1", "z2"],
        )
        result = model.fit()
        # Non-MultiIndex -> n_groups = n
        assert result.n_groups == n

    # ------------------------------------------------------------------
    # Line 857: j_statistic before fit
    # ------------------------------------------------------------------
    def test_j_statistic_before_fit_raises(self, panel_data):
        """Calling j_statistic() before fit() must raise RuntimeError."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        with pytest.raises(RuntimeError, match="Must call fit"):
            model.j_statistic()

    # ------------------------------------------------------------------
    # Line 872: interpret_j_test with reject=True path
    # ------------------------------------------------------------------
    def test_interpret_j_test_reject(self, fitted_model):
        """When J-test rejects, the interpretation says 'Reject'."""
        model = fitted_model
        interp = model._interpret_j_test(reject=True)
        assert "Reject" in interp
        assert "misspecified" in interp or "invalid" in interp

    def test_interpret_j_test_not_reject(self, fitted_model):
        """When J-test does not reject, the interpretation says 'Do not reject'."""
        model = fitted_model
        interp = model._interpret_j_test(reject=False)
        assert "Do not reject" in interp

    # ------------------------------------------------------------------
    # Line 912: compare_with_two_step before fit
    # ------------------------------------------------------------------
    def test_compare_with_two_step_before_fit_raises(self, panel_data):
        """Calling compare_with_two_step() before fit() must raise RuntimeError."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        # Create a dummy GMMResults-like object
        dummy_result = GMMResults(
            params=pd.Series([0.0, 0.0]),
            std_errors=pd.Series([1.0, 1.0]),
            tvalues=pd.Series([0.0, 0.0]),
            pvalues=pd.Series([1.0, 1.0]),
            nobs=50,
            n_groups=5,
            n_instruments=3,
            n_params=2,
            hansen_j=TestResult(name="Hansen J-test", statistic=0, pvalue=1, df=1),
            sargan=TestResult(name="Sargan test", statistic=0, pvalue=1, df=1),
            ar1_test=TestResult(name="AR(1) test", statistic=0, pvalue=1),
            ar2_test=TestResult(name="AR(2) test", statistic=0, pvalue=1),
            vcov=np.eye(2),
            converged=True,
            two_step=True,
            windmeijer_corrected=False,
            model_type="Difference GMM",
            transformation="fd",
        )
        with pytest.raises(RuntimeError, match="Must call fit"):
            model.compare_with_two_step(dummy_result)

    # ------------------------------------------------------------------
    # Line 912 (after fit): compare_with_two_step actually works
    # ------------------------------------------------------------------
    def test_compare_with_two_step_after_fit(self, fitted_model):
        """compare_with_two_step returns a DataFrame with expected columns."""
        model = fitted_model
        param_names = ["const", *model.exog_vars]
        dummy_result = GMMResults(
            params=pd.Series(model.params_ * 1.01, index=param_names),
            std_errors=pd.Series(model.bse_ * 1.05, index=param_names),
            tvalues=pd.Series(np.zeros(model.k), index=param_names),
            pvalues=pd.Series(np.ones(model.k), index=param_names),
            nobs=model.n,
            n_groups=5,
            n_instruments=model.n_instruments,
            n_params=model.k,
            hansen_j=TestResult(name="Hansen J-test", statistic=0, pvalue=1, df=1),
            sargan=TestResult(name="Sargan test", statistic=0, pvalue=1, df=1),
            ar1_test=TestResult(name="AR(1) test", statistic=0, pvalue=1),
            ar2_test=TestResult(name="AR(2) test", statistic=0, pvalue=1),
            vcov=np.eye(model.k),
            converged=True,
            two_step=True,
            windmeijer_corrected=False,
            model_type="Difference GMM",
            transformation="fd",
        )
        comparison = model.compare_with_two_step(dummy_result)
        assert isinstance(comparison, pd.DataFrame)
        assert "CUE Coef" in comparison.columns
        assert "TS Coef" in comparison.columns
        assert "Efficiency Ratio" in comparison.columns

    # ------------------------------------------------------------------
    # Line 977: conf_int before fit
    # ------------------------------------------------------------------
    def test_conf_int_before_fit_raises(self, panel_data):
        """Calling conf_int() before fit() must raise RuntimeError."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        with pytest.raises(RuntimeError, match="Must call fit"):
            model.conf_int()

    # ------------------------------------------------------------------
    # Lines 992, 1004: percentile and basic CI methods
    # ------------------------------------------------------------------
    def test_conf_int_percentile_without_bootstrap_raises(self, fitted_model):
        """Percentile CI without bootstrap must raise ValueError."""
        model = fitted_model
        # Ensure no bootstrap params
        model.bootstrap_params_ = None
        with pytest.raises(ValueError, match="Percentile method requires bootstrap"):
            model.conf_int(method="percentile")

    def test_conf_int_basic_without_bootstrap_raises(self, fitted_model):
        """Basic CI without bootstrap must raise ValueError."""
        model = fitted_model
        model.bootstrap_params_ = None
        with pytest.raises(ValueError, match="Basic method requires bootstrap"):
            model.conf_int(method="basic")

    def test_conf_int_percentile_with_bootstrap(self, panel_data):
        """Percentile CI with bootstrap must return valid DataFrame."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            se_type="bootstrap",
            n_bootstrap=10,
            bootstrap_method="residual",
        )
        model.fit()
        ci = model.conf_int(method="percentile")
        assert isinstance(ci, pd.DataFrame)
        assert "lower" in ci.columns
        assert "upper" in ci.columns
        # Lower should be less than upper
        for idx in ci.index:
            assert ci.loc[idx, "lower"] < ci.loc[idx, "upper"]

    def test_conf_int_basic_with_bootstrap(self, panel_data):
        """Basic bootstrap CI must return valid DataFrame."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            se_type="bootstrap",
            n_bootstrap=10,
            bootstrap_method="residual",
        )
        model.fit()
        ci = model.conf_int(method="basic")
        assert isinstance(ci, pd.DataFrame)
        assert "lower" in ci.columns
        assert "upper" in ci.columns

    # ------------------------------------------------------------------
    # Line 1016: invalid CI method
    # ------------------------------------------------------------------
    def test_conf_int_invalid_method_raises(self, fitted_model):
        """Invalid CI method must raise ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            fitted_model.conf_int(method="bayesian")

    # ------------------------------------------------------------------
    # Line 1041: bse property before fit
    # ------------------------------------------------------------------
    def test_bse_before_fit_raises(self, panel_data):
        """Accessing bse property before fit() must raise RuntimeError."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        with pytest.raises(RuntimeError, match="Must call fit"):
            _ = model.bse

    # ------------------------------------------------------------------
    # bse property after fit
    # ------------------------------------------------------------------
    def test_bse_after_fit(self, fitted_model):
        """bse property after fit returns a pd.Series with correct index."""
        bse = fitted_model.bse
        assert isinstance(bse, pd.Series)
        assert "const" in bse.index
        assert all(np.isfinite(bse.values))

    # ------------------------------------------------------------------
    # Normal conf_int after fit
    # ------------------------------------------------------------------
    def test_conf_int_normal(self, fitted_model):
        """Normal CI after fit returns correct shape."""
        ci = fitted_model.conf_int(alpha=0.05, method="normal")
        assert isinstance(ci, pd.DataFrame)
        assert ci.shape == (fitted_model.k, 2)
        for idx in ci.index:
            assert ci.loc[idx, "lower"] < ci.loc[idx, "upper"]

    # ------------------------------------------------------------------
    # j_statistic after fit
    # ------------------------------------------------------------------
    def test_j_statistic_after_fit(self, fitted_model):
        """j_statistic after fit returns a proper dict."""
        j = fitted_model.j_statistic()
        assert "statistic" in j
        assert "pvalue" in j
        assert "df" in j
        assert "reject" in j
        assert "interpretation" in j
        assert isinstance(j["reject"], (bool, np.bool_))

    # ------------------------------------------------------------------
    # Cluster weighting with MultiIndex (normal path)
    # ------------------------------------------------------------------
    def test_cluster_weighting_multiindex(self, panel_data):
        """Cluster weighting with proper MultiIndex data must fit successfully."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
            weighting="cluster",
        )
        model.fit()
        assert model.params_ is not None
        assert model.converged_

    # ------------------------------------------------------------------
    # Fit with explicit start_params
    # ------------------------------------------------------------------
    def test_fit_with_custom_start_params(self, panel_data):
        """Fitting with explicit start_params must work."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        start = np.array([0.0, 0.0])  # const + x1
        model.fit(start_params=start)
        assert model.params_ is not None

    # ------------------------------------------------------------------
    # Verbose fit (for coverage of verbose logger.info paths)
    # ------------------------------------------------------------------
    def test_fit_verbose(self, panel_data, caplog):
        """Verbose fit must log info messages."""
        df, exog, instr = panel_data
        model = ContinuousUpdatedGMM(
            data=df,
            dep_var="y",
            exog_vars=exog,
            instruments=instr,
        )
        with caplog.at_level(logging.INFO, logger="panelbox.gmm.cue_gmm"):
            model.fit(verbose=True)

        log_messages = [record.message for record in caplog.records]
        assert any("two-step GMM" in msg for msg in log_messages)
        assert any("Optimizing" in msg for msg in log_messages)
