"""
Tests for panelbox.var.diagnostics module coverage.

Targets uncovered lines:
- 124-128 (Hansen J singular Omega)
- 180-193 (interpret_hansen_j p > 0.99)
- 310-337 (difference_hansen_test)
- 842-885 (compare_transforms)
"""

import numpy as np
import pytest

from panelbox.var.diagnostics import GMMDiagnostics, ar_test, hansen_j_test, sargan_test


class TestDiagnosticsCoverage:
    """
    Additional tests to improve coverage for panelbox/var/diagnostics.py.

    Targets uncovered lines:
    - 124-128 (Hansen J singular Omega)
    - 180-193 (_interpret_hansen_j with p > 0.99, ideal range, rejection)
    - 310-337 (difference_hansen_test)
    - instrument_diagnostics_report with various ratios
    - ar_test function
    """

    # ---------------------------------------------------------------
    # hansen_j_test() when Omega is singular (lines 124-128)
    # ---------------------------------------------------------------
    def test_hansen_j_singular_omega_all_zero_residuals(self):
        """Test Hansen J with zero residuals producing singular Omega."""
        np.random.seed(42)
        n_obs = 50

        # All-zero residuals => Omega = Z' diag(0) Z = 0 => singular
        residuals = np.zeros((n_obs, 1))
        instruments = np.random.randn(n_obs, 8)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=20)
        result = diag.hansen_j_test()

        assert np.isnan(result["statistic"])
        assert "warnings" in result

    def test_hansen_j_singular_omega_collinear_instruments(self):
        """Test Hansen J when instruments are perfectly collinear."""
        np.random.seed(42)
        n_obs = 100

        residuals = np.random.randn(n_obs, 1)

        # All columns identical => rank-1 Z'diag(e^2)Z is singular
        base = np.random.randn(n_obs, 1)
        instruments = np.hstack([base] * 6)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=20)
        result = diag.hansen_j_test()

        assert np.isnan(result["statistic"])
        assert "singular" in result["interpretation"].lower() or "Singular" in " ".join(
            result.get("warnings", [])
        )

    def test_hansen_j_exactly_identified(self):
        """Test Hansen J when model is exactly identified (df=0)."""
        np.random.seed(42)
        residuals = np.random.randn(100, 2)
        instruments = np.random.randn(100, 4)
        n_params = 4

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities=20)
        result = diag.hansen_j_test()

        assert np.isnan(result["statistic"])
        assert "exactly identified" in result["interpretation"].lower()

    def test_hansen_j_valid_overidentified(self):
        """Test Hansen J with valid overidentified model."""
        np.random.seed(42)
        n_obs = 200
        residuals = np.random.randn(n_obs, 2)
        instruments = np.random.randn(n_obs, 10)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=40)
        result = diag.hansen_j_test()

        assert result["statistic"] >= 0
        assert 0 <= result["p_value"] <= 1
        assert result["df"] == 6  # 10 - 4

    def test_hansen_j_1d_residuals(self):
        """Test Hansen J with 1D residuals."""
        np.random.seed(42)
        n_obs = 100
        residuals = np.random.randn(n_obs)  # 1D
        instruments = np.random.randn(n_obs, 8)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=20)
        result = diag.hansen_j_test()

        assert "statistic" in result
        assert "p_value" in result

    # ---------------------------------------------------------------
    # _interpret_hansen_j() with various p-values (lines 180-193)
    # ---------------------------------------------------------------
    def test_interpret_hansen_j_p_below_005(self):
        """Test _interpret_hansen_j when p < 0.05 (rejection)."""
        np.random.seed(42)
        diag = GMMDiagnostics(
            residuals=np.random.randn(200, 1),
            instruments=np.random.randn(200, 8),
            n_params=4,
            n_entities=200,
        )

        interpretation, test_warnings = diag._interpret_hansen_j(20.0, 0.01)
        assert "Reject" in interpretation or "reject" in interpretation.lower()
        assert any("rejects" in w.lower() for w in test_warnings)

    def test_interpret_hansen_j_p_above_099(self):
        """Test _interpret_hansen_j when p > 0.99 (weak instruments warning, lines 180-185)."""
        np.random.seed(42)
        diag = GMMDiagnostics(
            residuals=np.random.randn(200, 1),
            instruments=np.random.randn(200, 8),
            n_params=4,
            n_entities=200,
        )

        interpretation, test_warnings = diag._interpret_hansen_j(0.01, 0.999)
        assert "p-value very high" in interpretation or "weak instruments" in interpretation.lower()
        assert any("p > 0.99" in w for w in test_warnings)

    def test_interpret_hansen_j_ideal_range(self):
        """Test _interpret_hansen_j when 0.10 <= p <= 0.90 (ideal range, lines 187-189)."""
        np.random.seed(42)
        diag = GMMDiagnostics(
            residuals=np.random.randn(200, 1),
            instruments=np.random.randn(200, 8),
            n_params=4,
            n_entities=200,
        )

        interpretation, _warnings = diag._interpret_hansen_j(5.0, 0.50)
        assert "ideal range" in interpretation.lower()
        assert "Do not reject" in interpretation

    def test_interpret_hansen_j_marginal_below_ideal(self):
        """Test _interpret_hansen_j when 0.05 <= p < 0.10 (valid but outside ideal, line 192-193)."""
        np.random.seed(42)
        diag = GMMDiagnostics(
            residuals=np.random.randn(200, 1),
            instruments=np.random.randn(200, 8),
            n_params=4,
            n_entities=200,
        )

        interpretation, _warnings = diag._interpret_hansen_j(5.0, 0.08)
        assert "Do not reject" in interpretation
        assert "ideal range" not in interpretation.lower()

    def test_interpret_hansen_j_marginal_above_ideal(self):
        """Test _interpret_hansen_j when 0.90 < p <= 0.99."""
        np.random.seed(42)
        diag = GMMDiagnostics(
            residuals=np.random.randn(200, 1),
            instruments=np.random.randn(200, 8),
            n_params=4,
            n_entities=200,
        )

        interpretation, _warnings = diag._interpret_hansen_j(1.0, 0.95)
        assert "Do not reject" in interpretation
        assert "ideal range" not in interpretation.lower()

    def test_interpret_hansen_j_roodman_rule_warning(self):
        """Test that Roodman rule warning fires when instruments > entities."""
        np.random.seed(42)
        diag = GMMDiagnostics(
            residuals=np.random.randn(200, 1),
            instruments=np.random.randn(200, 30),
            n_params=4,
            n_entities=10,
        )

        _interpretation, test_warnings = diag._interpret_hansen_j(5.0, 0.50)
        assert any("Rule-of-thumb violated" in w for w in test_warnings)

    # ---------------------------------------------------------------
    # difference_hansen_test() (lines 310-337)
    # ---------------------------------------------------------------
    def test_difference_hansen_basic(self):
        """Test basic difference-in-Hansen test."""
        np.random.seed(42)
        n_obs = 200
        n_params = 4

        residuals_full = np.random.randn(n_obs, 2)
        instruments_full = np.random.randn(n_obs, 12)
        instruments_restricted = instruments_full[:, :6]
        residuals_restricted = residuals_full + np.random.randn(n_obs, 2) * 0.01

        diag = GMMDiagnostics(residuals_full, instruments_full, n_params, n_entities=40)

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals_full,
            residuals_restricted=residuals_restricted,
            n_params=n_params,
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "df" in result
        assert "interpretation" in result
        assert result["df"] == 12 - 6  # = 6

    def test_difference_hansen_df_correct(self):
        """Test that difference-in-Hansen df equals instrument count difference."""
        np.random.seed(42)
        n_obs = 200
        n_params = 4

        residuals = np.random.randn(n_obs, 1)
        instruments_full = np.random.randn(n_obs, 15)
        instruments_restricted = instruments_full[:, :8]

        diag = GMMDiagnostics(residuals, instruments_full, n_params, n_entities=50)

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals,
            residuals_restricted=residuals,
            n_params=n_params,
        )

        assert result["df"] == 7  # 15 - 8

    def test_difference_hansen_not_reject_path(self):
        """Test difference-in-Hansen where additional instruments are valid (line 334-335)."""
        np.random.seed(42)
        n_obs = 300
        n_params = 4

        residuals = np.random.randn(n_obs, 1) * 0.5
        instruments_full = np.random.randn(n_obs, 10)
        instruments_restricted = instruments_full[:, :6]
        residuals_restricted = residuals + np.random.randn(n_obs, 1) * 0.001

        diag = GMMDiagnostics(residuals, instruments_full, n_params, n_entities=60)

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals,
            residuals_restricted=residuals_restricted,
            n_params=n_params,
        )

        assert 0 <= result["p_value"] <= 1
        assert isinstance(result["interpretation"], str)

    def test_difference_hansen_reject_path(self):
        """Test difference-in-Hansen with very different restricted residuals (line 329-333)."""
        np.random.seed(42)
        n_obs = 200
        n_params = 4

        residuals_full = np.random.randn(n_obs, 1)
        instruments_full = np.random.randn(n_obs, 12)
        instruments_restricted = instruments_full[:, :4]
        residuals_restricted = np.random.randn(n_obs, 1) * 5  # Very different

        diag = GMMDiagnostics(residuals_full, instruments_full, n_params, n_entities=40)

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals_full,
            residuals_restricted=residuals_restricted,
            n_params=n_params,
        )

        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)

    # ---------------------------------------------------------------
    # instrument_diagnostics_report() with various instrument/entity ratios
    # ---------------------------------------------------------------
    def test_diagnostics_report_good_ratios(self):
        """Test diagnostics report with good instrument/entity ratios."""
        np.random.seed(42)
        n_obs = 200
        residuals = np.random.randn(n_obs, 2)
        instruments = np.random.randn(n_obs, 8)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=50)
        report = diag.instrument_diagnostics_report()

        assert report["n_instruments"] == 8
        assert report["n_params"] == 4
        assert report["n_entities"] == 50
        assert report["ratio_instr_entities"] == 8 / 50
        assert report["ratio_instr_params"] == 8 / 4

    def test_diagnostics_report_bad_ratios_triggers_warnings(self):
        """Test diagnostics report with bad ratios triggers warnings."""
        np.random.seed(42)
        n_obs = 50
        residuals = np.random.randn(n_obs, 2)
        instruments = np.random.randn(n_obs, 60)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=10)
        report = diag.instrument_diagnostics_report()

        assert len(report["warnings"]) > 0
        assert report["ratio_instr_entities"] > 1
        assert report["ratio_instr_params"] > 3
        assert len(report["suggestions"]) > 0

    def test_diagnostics_report_high_ratio_instr_params(self):
        """Test diagnostics when instrument/params ratio > 3 triggers warning."""
        np.random.seed(42)
        n_obs = 100
        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, 16)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=100)
        report = diag.instrument_diagnostics_report()

        # ratio = 16/4 = 4 > 3
        assert report["ratio_instr_params"] == 4.0
        warning_text = " ".join(report["warnings"])
        assert (
            "instrument count" in warning_text.lower()
            or "instruments/params" in warning_text.lower()
        )

    def test_diagnostics_report_format_string(self):
        """Test that format_diagnostics_report returns a proper string."""
        np.random.seed(42)
        n_obs = 100
        residuals = np.random.randn(n_obs, 2)
        instruments = np.random.randn(n_obs, 8)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=20)
        formatted = diag.format_diagnostics_report()

        assert isinstance(formatted, str)
        assert "Hansen J" in formatted
        assert "DIAGNOSIS" in formatted

    # ---------------------------------------------------------------
    # ar_test() function directly
    # ---------------------------------------------------------------
    def test_ar_test_order_1(self):
        """Test ar_test with order=1."""
        np.random.seed(42)
        n_entities = 20
        n_periods = 10

        residuals_list = []
        entity_ids_list = []
        for entity_id in range(n_entities):
            resid = np.random.randn(n_periods)
            residuals_list.append(resid)
            entity_ids_list.append(np.full(n_periods, entity_id))

        residuals = np.concatenate(residuals_list)
        entity_ids = np.concatenate(entity_ids_list)

        result = ar_test(residuals, entity_ids, order=1)

        assert "statistic" in result
        assert "p_value" in result
        assert "order" in result
        assert result["order"] == 1
        assert "interpretation" in result
        assert result["n_products"] > 0

    def test_ar_test_order_2(self):
        """Test ar_test with order=2."""
        np.random.seed(42)
        n_entities = 20
        n_periods = 10

        residuals_list = []
        entity_ids_list = []
        for entity_id in range(n_entities):
            resid = np.random.randn(n_periods)
            residuals_list.append(resid)
            entity_ids_list.append(np.full(n_periods, entity_id))

        residuals = np.concatenate(residuals_list)
        entity_ids = np.concatenate(entity_ids_list)

        result = ar_test(residuals, entity_ids, order=2)

        assert result["order"] == 2
        assert result["n_products"] > 0
        assert 0 <= result["p_value"] <= 1

    def test_ar_test_order_3(self):
        """Test ar_test with higher order."""
        np.random.seed(42)
        n_entities = 20
        n_periods = 10

        residuals = np.random.randn(n_entities * n_periods)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)

        result = ar_test(residuals, entity_ids, order=3)

        assert result["order"] == 3
        assert "interpretation" in result

    def test_ar_test_multi_equation_residuals(self):
        """Test ar_test with multi-equation (2D) residuals."""
        np.random.seed(42)
        n_entities = 15
        n_periods = 8
        n_obs = n_entities * n_periods

        residuals = np.random.randn(n_obs, 2)  # 2 equations
        entity_ids = np.repeat(np.arange(n_entities), n_periods)

        result = ar_test(residuals, entity_ids, order=1)

        assert result["n_products"] > 0
        assert "statistic" in result

    def test_ar_test_no_entity_ids_via_class(self):
        """Test ar_test via GMMDiagnostics without entity_ids returns nan."""
        np.random.seed(42)
        residuals = np.random.randn(100, 2)
        instruments = np.random.randn(100, 8)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=20, entity_ids=None)
        result = diag.ar_test(order=1)

        assert np.isnan(result["statistic"])
        assert "requires entity_ids" in result["interpretation"]

    def test_ar_test_insufficient_data(self):
        """Test ar_test with too few observations per entity."""
        np.random.seed(42)

        # Each entity has only 1 observation -- not enough for even AR(1)
        residuals = np.array([1.0, 2.0, 3.0])
        entity_ids = np.array([0, 1, 2])

        result = ar_test(residuals, entity_ids, order=1)

        # No products possible with 1 obs per entity
        assert result["n_products"] == 0
        assert np.isnan(result["statistic"])

    # ---------------------------------------------------------------
    # Convenience functions hansen_j_test and sargan_test
    # ---------------------------------------------------------------
    def test_convenience_hansen_j_test(self):
        """Test the convenience hansen_j_test function."""
        np.random.seed(42)
        residuals = np.random.randn(200, 2)
        instruments = np.random.randn(200, 10)

        result = hansen_j_test(residuals, instruments, n_params=4, n_entities=40)

        assert "statistic" in result
        assert "p_value" in result
        assert result["statistic"] >= 0

    def test_convenience_sargan_test(self):
        """Test the convenience sargan_test function."""
        np.random.seed(42)
        residuals = np.random.randn(200, 2)
        instruments = np.random.randn(200, 10)

        result = sargan_test(residuals, instruments, n_params=4, n_entities=40)

        assert "statistic" in result
        assert "p_value" in result
        assert result["statistic"] >= 0

    # ---------------------------------------------------------------
    # Sargan test exactly identified
    # ---------------------------------------------------------------
    def test_sargan_exactly_identified(self):
        """Test that Sargan test handles exact identification."""
        np.random.seed(42)
        residuals = np.random.randn(100, 1)
        instruments = np.random.randn(100, 4)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=20)
        result = diag.sargan_test()

        assert np.isnan(result["statistic"])
        assert "exactly identified" in result["interpretation"].lower()

    # ---------------------------------------------------------------
    # format_diagnostics_report with AR tests
    # ---------------------------------------------------------------
    def test_format_diagnostics_report_with_ar_tests(self):
        """Test format_diagnostics_report including AR tests."""
        np.random.seed(42)
        n_entities = 20
        n_periods = 10
        n_obs = n_entities * n_periods

        residuals = np.random.randn(n_obs, 2)
        instruments = np.random.randn(n_obs, 8)
        entity_ids = np.repeat(np.arange(n_entities), n_periods)

        diag = GMMDiagnostics(
            residuals,
            instruments,
            n_params=4,
            n_entities=n_entities,
            entity_ids=entity_ids,
        )

        formatted = diag.format_diagnostics_report(
            include_ar_tests=True,
            max_ar_order=2,
        )

        assert isinstance(formatted, str)
        assert "AR(1)" in formatted
        assert "AR(2)" in formatted
        assert "Serial Correlation" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
