"""
Unit tests for report transformer classes.

Tests that each transformer correctly converts raw data into
template-ready dictionaries.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from panelbox.report.transformers import (
    DiscreteTransformer,
    GMMTransformer,
    QuantileTransformer,
    RegressionTransformer,
    SFATransformer,
    VARTransformer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gmm_full_data():
    """Full GMM result dict with all fields populated."""
    return {
        "model_type": "System GMM",
        "nobs": 1000,
        "n_groups": 100,
        "n_instruments": 50,
        "two_step": True,
        "coefficients": [
            {"name": "L.y", "coef": 0.5, "se": 0.1, "tstat": 5.0, "pvalue": 0.001},
            {"name": "x1", "coef": 0.3, "se": 0.15, "tstat": 2.0, "pvalue": 0.045},
            {"name": "x2", "coef": 0.1, "se": 0.08, "tstat": 1.25, "pvalue": 0.08},
            {"name": "x3", "coef": 0.02, "se": 0.05, "tstat": 0.4, "pvalue": 0.69},
        ],
        "hansen_test": {"statistic": 45.3, "pvalue": 0.35, "df": 40},
        "ar_tests": {
            "ar1": {"statistic": -2.5, "pvalue": 0.012},
            "ar2": {"statistic": 0.8, "pvalue": 0.42},
        },
    }


@pytest.fixture
def regression_full_data():
    """Full regression result dict with all fields populated."""
    return {
        "model_type": "Fixed Effects",
        "formula": "y ~ x1 + x2 + x3",
        "nobs": 500,
        "n_entities": 50,
        "n_periods": 10,
        "se_type": "clustered",
        "coefficients": [
            {
                "name": "x1",
                "coef": 1.5,
                "se": 0.3,
                "tstat": 5.0,
                "pvalue": 0.001,
                "ci_lower": 0.9,
                "ci_upper": 2.1,
            },
            {
                "name": "x2",
                "coef": -0.8,
                "se": 0.4,
                "tstat": -2.0,
                "pvalue": 0.046,
                "ci_lower": -1.6,
                "ci_upper": 0.0,
            },
            {
                "name": "x3",
                "coef": 0.1,
                "se": 0.06,
                "tstat": 1.67,
                "pvalue": 0.095,
                "ci_lower": -0.02,
                "ci_upper": 0.22,
            },
            {
                "name": "x4",
                "coef": 0.01,
                "se": 0.05,
                "tstat": 0.2,
                "pvalue": 0.84,
                "ci_lower": -0.09,
                "ci_upper": 0.11,
            },
        ],
        "r_squared": 0.85,
        "adj_r_squared": 0.83,
        "f_statistic": 120.5,
        "f_pvalue": 0.0001,
        "aic": 1500.3,
        "bic": 1520.7,
    }


@pytest.fixture
def discrete_full_data():
    """Full discrete/MLE result dict with all fields populated."""
    return {
        "model_type": "Logit",
        "model_type_full": "Random Effects Logit",
        "distribution": "logistic",
        "nobs": 800,
        "n_entities": 80,
        "n_periods": 10,
        "converged": True,
        "n_iter": 25,
        "se_type": "robust",
        "coefficients": [
            {"name": "x1", "coef": 2.1, "se": 0.5, "tstat": 4.2, "pvalue": 0.001},
            {"name": "x2", "coef": -0.5, "se": 0.25, "zstat": -2.0, "pvalue": 0.045},
            {"name": "x3", "coef": 0.3, "se": 0.18, "tstat": 1.67, "pvalue": 0.095},
            {"name": "x4", "coef": 0.01, "se": 0.1, "tstat": 0.1, "pvalue": 0.92},
        ],
        "loglikelihood": -350.5,
        "aic": 709.0,
        "bic": 730.2,
        "pseudo_r_squared": 0.25,
        "classification_metrics": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80,
        },
    }


@pytest.fixture
def sfa_mock_result():
    """Mock SFResult object with required attributes."""
    params = pd.Series(
        {"const": 1.0, "x1": 0.5, "x2": -0.3, "sigma_v": 0.2, "sigma_u": 0.4},
    )
    se = pd.Series(
        {"const": 0.2, "x1": 0.1, "x2": 0.15, "sigma_v": 0.05, "sigma_u": 0.08},
    )
    tvalues = params / se
    pvalues = pd.Series(
        {"const": 0.001, "x1": 0.001, "x2": 0.045, "sigma_v": 0.001, "sigma_u": 0.001},
    )
    efficiency = pd.Series([0.85, 0.90, 0.75, 0.92, 0.88, 0.70, 0.95, 0.80])

    result = SimpleNamespace(
        frontier_type="production",
        distribution="half-normal",
        nobs=200,
        n_entities=20,
        n_periods=10,
        converged=True,
        nparams=5,
        params=params,
        se=se,
        tvalues=tvalues,
        pvalues=pvalues,
        sigma_v=0.2,
        sigma_u=0.4,
        sigma=0.45,
        sigma_sq=0.2025,
        lambda_param=2.0,
        gamma=0.8,
        efficiency_scores=efficiency,
        loglik=-120.5,
        aic=251.0,
        bic=268.3,
    )
    return result


@pytest.fixture
def var_mock_result():
    """Mock PanelVARResult object with required attributes."""
    return SimpleNamespace(
        K=2,
        p=1,
        N=50,
        n_obs=450,
        method="ols",
        cov_type="robust",
        endog_names=["gdp", "investment"],
        exog_names=["L1.gdp", "L1.investment", "const"],
        params_by_eq=[
            [0.8, 0.2, 0.5],  # gdp equation
            [0.1, 0.6, 0.3],  # investment equation
        ],
        std_errors_by_eq=[
            [0.05, 0.1, 0.15],
            [0.08, 0.04, 0.12],
        ],
        aic=350.2,
        bic=370.5,
        hqic=358.1,
        loglik=-168.1,
        max_eigenvalue_modulus=0.85,
        stability_margin=0.15,
    )


@pytest.fixture
def quantile_mock_report():
    """Mock DiagnosticReport object with diagnostics."""
    diag1 = SimpleNamespace(
        test_name="Heteroscedasticity",
        statistic=12.5,
        p_value=0.001,
        status="fail",
        message="Evidence of heteroscedasticity",
        recommendation="Use robust standard errors",
    )
    diag2 = SimpleNamespace(
        test_name="Normality",
        statistic=3.2,
        p_value=0.07,
        status="warning",
        message="Marginal normality",
        recommendation="Consider quantile regression",
    )
    diag3 = SimpleNamespace(
        test_name="Stationarity",
        statistic=8.5,
        p_value=0.5,
        status="pass",
        message="Data appears stationary",
        recommendation=None,
    )
    return SimpleNamespace(
        health_score=0.65,
        health_status="fair",
        diagnostics=[diag1, diag2, diag3],
    )


# ===========================================================================
# GMMTransformer Tests
# ===========================================================================


class TestGMMTransformer:
    """Tests for GMMTransformer."""

    def test_full_transform(self, gmm_full_data):
        """Test transform with all fields populated."""
        result = GMMTransformer(gmm_full_data).transform()

        assert "model_info" in result
        assert "coefficients" in result
        assert "diagnostics" in result
        assert "summary" in result

    def test_model_info(self, gmm_full_data):
        """Test model_info extraction."""
        result = GMMTransformer(gmm_full_data).transform()
        info = result["model_info"]

        assert info["estimator"] == "System GMM"
        assert info["nobs"] == 1000
        assert info["n_groups"] == 100
        assert info["n_instruments"] == 50
        assert info["two_step"] is True
        assert info["instrument_ratio"] == "50/100"

    def test_significance_stars(self, gmm_full_data):
        """Test significance stars assignment for all levels."""
        result = GMMTransformer(gmm_full_data).transform()
        coeffs = result["coefficients"]

        # p=0.001 -> ***
        assert coeffs[0]["stars"] == "***"
        # p=0.045 -> **
        assert coeffs[1]["stars"] == "**"
        # p=0.08 -> *
        assert coeffs[2]["stars"] == "*"
        # p=0.69 -> no stars
        assert coeffs[3]["stars"] == ""

    def test_hansen_pass(self, gmm_full_data):
        """Test Hansen test with p > 0.05 reports pass."""
        result = GMMTransformer(gmm_full_data).transform()
        hansen = result["diagnostics"]["hansen"]

        assert hansen["status"] == "pass"
        assert hansen["statistic"] == 45.3
        assert hansen["pvalue"] == 0.35
        assert hansen["df"] == 40

    def test_hansen_fail(self, gmm_full_data):
        """Test Hansen test with p < 0.05 reports fail."""
        gmm_full_data["hansen_test"]["pvalue"] = 0.02
        result = GMMTransformer(gmm_full_data).transform()
        hansen = result["diagnostics"]["hansen"]

        assert hansen["status"] == "fail"

    def test_ar2_pass_fail(self, gmm_full_data):
        """Test AR2 status pass/fail based on p-value."""
        # pass case (p=0.42 > 0.05)
        result = GMMTransformer(gmm_full_data).transform()
        assert result["diagnostics"]["ar2"]["status"] == "pass"

        # fail case
        gmm_full_data["ar_tests"]["ar2"]["pvalue"] = 0.03
        result = GMMTransformer(gmm_full_data).transform()
        assert result["diagnostics"]["ar2"]["status"] == "fail"

    def test_summary_good_when_both_pass(self, gmm_full_data):
        """Test summary is 'good' when hansen and ar2 both pass."""
        result = GMMTransformer(gmm_full_data).transform()
        summary = result["summary"]

        assert summary["overall_status"] == "good"
        assert summary["hansen_ok"] is True
        assert summary["ar2_ok"] is True

    def test_summary_warning_when_hansen_fails(self, gmm_full_data):
        """Test summary is 'warning' when hansen fails."""
        gmm_full_data["hansen_test"]["pvalue"] = 0.01
        result = GMMTransformer(gmm_full_data).transform()

        assert result["summary"]["overall_status"] == "warning"
        assert result["summary"]["hansen_ok"] is False

    def test_missing_fields_defaults(self):
        """Test graceful defaults when fields are missing."""
        result = GMMTransformer({}).transform()
        info = result["model_info"]

        assert info["estimator"] == "GMM"
        assert info["nobs"] == "\u2014"
        assert info["n_groups"] == "\u2014"
        assert info["n_instruments"] == "\u2014"
        assert info["two_step"] is False
        assert info["instrument_ratio"] == "\u2014"

    def test_alternative_key_names(self):
        """Test that alternative key names (estimator, pval, etc.) work."""
        data = {
            "estimator": "Difference GMM",
            "coefficients": [
                {"name": "y", "coefficient": 0.5, "std_error": 0.1, "t_stat": 5.0, "pval": 0.001},
            ],
            "hansen_j": {"statistic": 30.0, "p_value": 0.5, "df": 25},
        }
        result = GMMTransformer(data).transform()

        assert result["model_info"]["estimator"] == "Difference GMM"
        assert result["coefficients"][0]["coef"] == 0.5
        assert result["coefficients"][0]["se"] == 0.1
        assert result["coefficients"][0]["stars"] == "***"
        assert result["diagnostics"]["hansen"]["pvalue"] == 0.5


# ===========================================================================
# RegressionTransformer Tests
# ===========================================================================


class TestRegressionTransformer:
    """Tests for RegressionTransformer."""

    def test_full_transform(self, regression_full_data):
        """Test transform with all fields populated."""
        result = RegressionTransformer(regression_full_data).transform()

        assert "model_info" in result
        assert "coefficients" in result
        assert "fit_statistics" in result

    def test_model_info(self, regression_full_data):
        """Test model_info extraction."""
        result = RegressionTransformer(regression_full_data).transform()
        info = result["model_info"]

        assert info["estimator"] == "Fixed Effects"
        assert info["formula"] == "y ~ x1 + x2 + x3"
        assert info["nobs"] == 500
        assert info["n_entities"] == 50
        assert info["n_periods"] == 10
        assert info["se_type"] == "clustered"

    def test_significance_stars(self, regression_full_data):
        """Test significance stars for all levels."""
        result = RegressionTransformer(regression_full_data).transform()
        coeffs = result["coefficients"]

        assert coeffs[0]["stars"] == "***"  # p=0.001
        assert coeffs[1]["stars"] == "**"  # p=0.046
        assert coeffs[2]["stars"] == "*"  # p=0.095
        assert coeffs[3]["stars"] == ""  # p=0.84

    def test_confidence_intervals(self, regression_full_data):
        """Test confidence interval values are passed through."""
        result = RegressionTransformer(regression_full_data).transform()
        first = result["coefficients"][0]

        assert first["ci_lower"] == 0.9
        assert first["ci_upper"] == 2.1

    def test_fit_statistics(self, regression_full_data):
        """Test fit statistics extraction."""
        result = RegressionTransformer(regression_full_data).transform()
        fit = result["fit_statistics"]

        assert fit["r_squared"] == 0.85
        assert fit["adj_r_squared"] == 0.83
        assert fit["f_statistic"] == 120.5
        assert fit["f_pvalue"] == 0.0001
        assert fit["aic"] == 1500.3
        assert fit["bic"] == 1520.7

    def test_missing_fields_defaults(self):
        """Test graceful defaults for missing fields."""
        result = RegressionTransformer({}).transform()
        info = result["model_info"]

        assert info["estimator"] == "Panel OLS"
        assert info["formula"] == "\u2014"
        assert info["nobs"] == "\u2014"

    def test_alternative_key_names(self):
        """Test alternative key names work (estimator, cov_type)."""
        data = {
            "estimator": "Random Effects",
            "cov_type": "HC1",
            "coefficients": [
                {"name": "z", "coefficient": 1.0, "std_error": 0.2, "t_stat": 5.0, "pval": 0.001},
            ],
        }
        result = RegressionTransformer(data).transform()

        assert result["model_info"]["estimator"] == "Random Effects"
        assert result["model_info"]["se_type"] == "HC1"
        assert result["coefficients"][0]["coef"] == 1.0

    def test_empty_coefficients(self):
        """Test with no coefficients."""
        result = RegressionTransformer({"coefficients": []}).transform()
        assert result["coefficients"] == []


# ===========================================================================
# DiscreteTransformer Tests
# ===========================================================================


class TestDiscreteTransformer:
    """Tests for DiscreteTransformer."""

    def test_full_transform(self, discrete_full_data):
        """Test transform with all fields populated."""
        result = DiscreteTransformer(discrete_full_data).transform()

        assert "model_info" in result
        assert "coefficients" in result
        assert "fit_statistics" in result
        assert "classification" in result

    def test_model_info(self, discrete_full_data):
        """Test model_info extraction with model_type_full precedence."""
        result = DiscreteTransformer(discrete_full_data).transform()
        info = result["model_info"]

        assert info["model_type"] == "Random Effects Logit"
        assert info["distribution"] == "logistic"
        assert info["nobs"] == 800
        assert info["converged"] is True
        assert info["n_iter"] == 25
        assert info["se_type"] == "robust"

    def test_significance_stars(self, discrete_full_data):
        """Test significance stars for all p-value levels."""
        result = DiscreteTransformer(discrete_full_data).transform()
        coeffs = result["coefficients"]

        assert coeffs[0]["stars"] == "***"  # p=0.001
        assert coeffs[1]["stars"] == "**"  # p=0.045
        assert coeffs[2]["stars"] == "*"  # p=0.095
        assert coeffs[3]["stars"] == ""  # p=0.92

    def test_zstat_from_tstat(self, discrete_full_data):
        """Test that zstat is populated from tstat or zstat key."""
        result = DiscreteTransformer(discrete_full_data).transform()
        coeffs = result["coefficients"]

        # First coef uses 'tstat' key
        assert coeffs[0]["zstat"] == 4.2
        # Second coef uses 'zstat' key
        assert coeffs[1]["zstat"] == -2.0

    def test_classification_metrics_present(self, discrete_full_data):
        """Test classification metrics for binary models."""
        result = DiscreteTransformer(discrete_full_data).transform()
        cls = result["classification"]

        assert cls is not None
        assert cls["accuracy"] == 0.85
        assert cls["precision"] == 0.82
        assert cls["recall"] == 0.78
        assert cls["f1_score"] == 0.80

    def test_classification_metrics_absent(self, discrete_full_data):
        """Test classification is None for non-binary models."""
        del discrete_full_data["classification_metrics"]
        result = DiscreteTransformer(discrete_full_data).transform()

        assert result["classification"] is None

    def test_fit_statistics(self, discrete_full_data):
        """Test fit statistics extraction."""
        result = DiscreteTransformer(discrete_full_data).transform()
        fit = result["fit_statistics"]

        assert fit["loglikelihood"] == -350.5
        assert fit["aic"] == 709.0
        assert fit["bic"] == 730.2
        assert fit["pseudo_r_squared"] == 0.25

    def test_convergence_status(self, discrete_full_data):
        """Test converged flag is passed through."""
        result = DiscreteTransformer(discrete_full_data).transform()
        assert result["model_info"]["converged"] is True

        discrete_full_data["converged"] = False
        result = DiscreteTransformer(discrete_full_data).transform()
        assert result["model_info"]["converged"] is False

    def test_missing_fields_defaults(self):
        """Test graceful defaults for empty input."""
        result = DiscreteTransformer({}).transform()
        info = result["model_info"]

        assert info["model_type"] == "MLE"
        assert info["distribution"] == "\u2014"
        assert info["converged"] is False
        assert result["coefficients"] == []
        assert result["classification"] is None

    def test_alternative_loglik_key(self):
        """Test alternative key 'loglik' for loglikelihood."""
        data = {"loglik": -100.0, "pseudo_r2": 0.15}
        result = DiscreteTransformer(data).transform()
        fit = result["fit_statistics"]

        assert fit["loglikelihood"] == -100.0
        assert fit["pseudo_r_squared"] == 0.15


# ===========================================================================
# SFATransformer Tests
# ===========================================================================


class TestSFATransformer:
    """Tests for SFATransformer."""

    def test_full_transform(self, sfa_mock_result):
        """Test transform with all attributes populated."""
        result = SFATransformer(sfa_mock_result).transform()

        assert "model_info" in result
        assert "coefficients" in result
        assert "variance_components" in result
        assert "efficiency" in result
        assert "fit_statistics" in result

    def test_model_info(self, sfa_mock_result):
        """Test model_info extraction from object attributes."""
        result = SFATransformer(sfa_mock_result).transform()
        info = result["model_info"]

        assert info["frontier_type"] == "production"
        assert info["distribution"] == "half-normal"
        assert info["nobs"] == 200
        assert info["n_entities"] == 20
        assert info["n_periods"] == 10
        assert info["converged"] is True
        assert info["nparams"] == 5

    def test_coefficients_exclude_variance_params(self, sfa_mock_result):
        """Test that variance parameters (sigma_v, sigma_u) are excluded."""
        result = SFATransformer(sfa_mock_result).transform()
        coeffs = result["coefficients"]
        names = [c["name"] for c in coeffs]

        assert "const" in names
        assert "x1" in names
        assert "x2" in names
        assert "sigma_v" not in names
        assert "sigma_u" not in names

    def test_coefficients_stars(self, sfa_mock_result):
        """Test significance stars on coefficients."""
        result = SFATransformer(sfa_mock_result).transform()
        coeffs = result["coefficients"]
        stars_map = {c["name"]: c["stars"] for c in coeffs}

        assert stars_map["const"] == "***"  # p=0.001
        assert stars_map["x1"] == "***"  # p=0.001
        assert stars_map["x2"] == "**"  # p=0.045

    def test_variance_components(self, sfa_mock_result):
        """Test variance components extraction."""
        result = SFATransformer(sfa_mock_result).transform()
        vc = result["variance_components"]

        assert vc["sigma_v"] == 0.2
        assert vc["sigma_u"] == 0.4
        assert vc["sigma"] == 0.45
        assert vc["lambda_param"] == 2.0
        assert vc["gamma"] == 0.8

    def test_efficiency_scores(self, sfa_mock_result):
        """Test efficiency score summary statistics."""
        result = SFATransformer(sfa_mock_result).transform()
        eff = result["efficiency"]

        assert eff is not None
        assert eff["count"] == 8
        assert 0.0 < eff["mean"] < 1.0
        assert eff["min"] == 0.70
        assert eff["max"] == 0.95

    def test_fit_statistics(self, sfa_mock_result):
        """Test fit statistics extraction."""
        result = SFATransformer(sfa_mock_result).transform()
        fit = result["fit_statistics"]

        assert fit["loglikelihood"] == -120.5
        assert fit["aic"] == 251.0
        assert fit["bic"] == 268.3

    def test_missing_efficiency_scores(self):
        """Test graceful handling when efficiency_scores is missing."""
        result_obj = SimpleNamespace(
            params=pd.Series({"const": 1.0}),
            se=pd.Series({"const": 0.2}),
            tvalues=pd.Series({"const": 5.0}),
            pvalues=pd.Series({"const": 0.001}),
        )
        result = SFATransformer(result_obj).transform()

        assert result["efficiency"] is None

    def test_missing_params(self):
        """Test graceful handling when params is None."""
        result_obj = SimpleNamespace()
        result = SFATransformer(result_obj).transform()

        assert result["coefficients"] == []


# ===========================================================================
# VARTransformer Tests
# ===========================================================================


class TestVARTransformer:
    """Tests for VARTransformer."""

    def test_full_transform(self, var_mock_result):
        """Test transform with all attributes populated."""
        result = VARTransformer(var_mock_result).transform()

        assert "model_info" in result
        assert "equations" in result
        assert "diagnostics" in result
        assert "stability" in result

    def test_model_info(self, var_mock_result):
        """Test model_info extraction."""
        result = VARTransformer(var_mock_result).transform()
        info = result["model_info"]

        assert info["K"] == 2
        assert info["p"] == 1
        assert info["N"] == 50
        assert info["n_obs"] == 450
        assert info["method"] == "ols"
        assert info["cov_type"] == "robust"
        assert info["endog_names"] == ["gdp", "investment"]

    def test_equations_count(self, var_mock_result):
        """Test that we get one equation per endogenous variable."""
        result = VARTransformer(var_mock_result).transform()
        eqs = result["equations"]

        assert len(eqs) == 2
        assert eqs[0]["name"] == "gdp"
        assert eqs[1]["name"] == "investment"

    def test_equation_coefficients(self, var_mock_result):
        """Test coefficient extraction within equations."""
        result = VARTransformer(var_mock_result).transform()
        gdp_eq = result["equations"][0]
        coeffs = gdp_eq["coefficients"]

        assert len(coeffs) == 3
        assert coeffs[0]["name"] == "L1.gdp"
        assert coeffs[0]["coef"] == 0.8
        assert coeffs[0]["se"] == 0.05
        # t = 0.8/0.05 = 16.0 -> p ~ 0 -> ***
        assert coeffs[0]["stars"] == "***"

    def test_stability_stable(self, var_mock_result):
        """Test stability with max eigenvalue < 1."""
        result = VARTransformer(var_mock_result).transform()
        stab = result["stability"]

        assert stab["is_stable"] is True
        assert stab["max_eigenvalue_modulus"] == 0.85
        assert stab["stability_margin"] == 0.15

    def test_stability_unstable(self, var_mock_result):
        """Test stability with max eigenvalue >= 1."""
        var_mock_result.max_eigenvalue_modulus = 1.05
        result = VARTransformer(var_mock_result).transform()
        stab = result["stability"]

        assert stab["is_stable"] is False

    def test_diagnostics(self, var_mock_result):
        """Test diagnostic statistics extraction."""
        result = VARTransformer(var_mock_result).transform()
        diag = result["diagnostics"]

        assert diag["aic"] == 350.2
        assert diag["bic"] == 370.5
        assert diag["hqic"] == 358.1
        assert diag["loglik"] == -168.1

    def test_empty_equations(self):
        """Test with no equations (empty params)."""
        result_obj = SimpleNamespace(
            K=0,
            endog_names=[],
            exog_names=[],
            params_by_eq=[],
            std_errors_by_eq=[],
        )
        result = VARTransformer(result_obj).transform()
        assert result["equations"] == []

    def test_missing_attributes(self):
        """Test graceful handling of missing attributes."""
        result_obj = SimpleNamespace()
        result = VARTransformer(result_obj).transform()
        info = result["model_info"]

        assert info["K"] == "\u2014"
        assert info["method"] == "ols"
        assert result["equations"] == []


# ===========================================================================
# QuantileTransformer Tests
# ===========================================================================


class TestQuantileTransformer:
    """Tests for QuantileTransformer."""

    def test_full_transform(self, quantile_mock_report):
        """Test transform with all diagnostics populated."""
        result = QuantileTransformer(quantile_mock_report).transform()

        assert "health" in result
        assert "tests" in result
        assert "recommendations" in result

    def test_health_score(self, quantile_mock_report):
        """Test health score extraction and formatting."""
        result = QuantileTransformer(quantile_mock_report).transform()
        health = result["health"]

        assert health["score"] == 0.65
        assert health["score_pct"] == "65.0%"
        assert health["status"] == "fair"
        assert health["color"] == "#f59e0b"

    def test_health_colors(self):
        """Test health color mapping for each status level."""
        for status, expected_color in [
            ("good", "#10b981"),
            ("fair", "#f59e0b"),
            ("poor", "#ef4444"),
        ]:
            report = SimpleNamespace(
                health_score=0.5,
                health_status=status,
                diagnostics=[],
            )
            result = QuantileTransformer(report).transform()
            assert result["health"]["color"] == expected_color

    def test_unknown_health_status_color(self):
        """Test fallback color for unknown status."""
        report = SimpleNamespace(
            health_score=0.5,
            health_status="unknown",
            diagnostics=[],
        )
        result = QuantileTransformer(report).transform()
        assert result["health"]["color"] == "#6b7280"

    def test_test_status_icons(self, quantile_mock_report):
        """Test status icon mapping for pass/warning/fail."""
        result = QuantileTransformer(quantile_mock_report).transform()
        tests = result["tests"]

        assert tests[0]["status"] == "fail"
        assert tests[0]["status_icon"] == "&#10007;"
        assert tests[0]["status_class"] == "text-danger"

        assert tests[1]["status"] == "warning"
        assert tests[1]["status_icon"] == "&#9888;"
        assert tests[1]["status_class"] == "text-warning"

        assert tests[2]["status"] == "pass"
        assert tests[2]["status_icon"] == "&#10003;"
        assert tests[2]["status_class"] == "text-success"

    def test_test_details(self, quantile_mock_report):
        """Test that test details are correctly extracted."""
        result = QuantileTransformer(quantile_mock_report).transform()
        first_test = result["tests"][0]

        assert first_test["name"] == "Heteroscedasticity"
        assert first_test["statistic"] == 12.5
        assert first_test["pvalue"] == 0.001
        assert first_test["message"] == "Evidence of heteroscedasticity"

    def test_recommendations(self, quantile_mock_report):
        """Test recommendations list extraction."""
        result = QuantileTransformer(quantile_mock_report).transform()
        recs = result["recommendations"]

        # Only non-None recommendations
        assert len(recs) == 2
        assert "Use robust standard errors" in recs
        assert "Consider quantile regression" in recs

    def test_empty_diagnostics(self):
        """Test with no diagnostics."""
        report = SimpleNamespace(
            health_score=1.0,
            health_status="good",
            diagnostics=[],
        )
        result = QuantileTransformer(report).transform()

        assert result["tests"] == []
        assert result["recommendations"] == []
