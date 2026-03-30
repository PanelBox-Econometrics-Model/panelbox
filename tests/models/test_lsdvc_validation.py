"""
Validation of panelbox LSDVC against Stata xtlsdvc reference values.

Reference: Bruno (2005) "Estimation and inference in dynamic unbalanced
panel-data models with a small number of individuals", Stata Journal,
Vol. 5, No. 4, pp. 473-500.

Stata reference values come from the appendix of Bruno (2005), using
abdata (ind==4, N=29 firms) with the command:
    xtlsdvc n w k yr1977-yr1984 if ind==4, initial(ah/ab) bias(1/2/3)

Known differences between panelbox and Stata xtlsdvc:
1. panelbox applies Kiviet (1995) bias correction ONLY to rho (the lagged
   dependent variable coefficient), using the simplified AR(1) formula:
   bias = -(1+rho)/(T-1) + higher-order terms.
   Stata applies the full matrix bias correction to ALL coefficients
   (rho and beta), using the general formula from Kiviet (1995) that
   accounts for the interaction between exogenous regressors and the
   within transformation.

2. As a consequence:
   - The rho coefficient differs slightly (panelbox corrects only for the
     AR(1) component, Stata for the full model structure)
   - The beta coefficients (w, k, year dummies) are NOT bias-corrected in
     panelbox (they stay at LSDV values), while Stata corrects them.

3. The LSDV (uncorrected) estimates match Stata exactly, confirming that
   the base estimation step is correct.

4. Bootstrap SEs are in the same order of magnitude, which is expected
   since the parametric bootstrap DGP is similar.
"""

import numpy as np
import pytest

from panelbox.datasets import load_abdata, load_grunfeld
from panelbox.models.dynamic.lsdvc import LSDVC

# =====================================================================
# Stata reference values from Bruno (2005) Stata Journal, Appendix
# Dataset: abdata, ind==4, N=29, T=7-9 (unbalanced)
# Model: xtlsdvc n w k yr1977-yr1984 if ind==4
# =====================================================================

STATA_LSDV = {
    "L1.n": 0.4056509,
    "w": -0.3541811,
    "k": 0.2541555,
}

STATA_AH_BIAS1 = {
    "L1.n": 0.5389829,
    "w": -0.3375203,
    "k": 0.2218794,
}

STATA_AH_BIAS2 = {
    "L1.n": 0.5354691,
    "w": -0.3380943,
    "k": 0.2226967,
}

STATA_AH_BIAS3 = {
    "L1.n": 0.6338054,
    "w": -0.3258186,
    "k": 0.1988694,
}

STATA_AB_BIAS3 = {
    "L1.n": 0.6360273,
    "w": -0.3256377,
    "k": 0.1988754,
}

# Stata bootstrap SEs (AB, bias(3), vcov(100))
STATA_AB_BIAS3_SE = {
    "L1.n": 0.0912651,
    "w": 0.1434720,
    "k": 0.0537594,
}


def _load_abdata_ind4():
    """Load abdata filtered to industry 4, with year dummies."""
    a = load_abdata()
    a4 = a[a["ind"] == 4].copy()
    a4["id"] = a4["id"].astype(int)
    a4["year"] = a4["year"].astype(int)
    for yr in range(1977, 1985):
        a4[f"yr{yr}"] = (a4["year"] == yr).astype(float)
    return a4


ABDATA_FORMULA = "n ~ w + k + yr1977 + yr1978 + yr1979 + yr1980 + yr1981 + yr1982 + yr1983 + yr1984"


class TestLSDVMatchesStata:
    """Verify LSDV (uncorrected) estimates match Stata exactly."""

    def test_lsdv_rho(self):
        """LSDV rho should match Stata xtreg,fe to 5 decimal places."""
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator="ab",
            bias_order=1,
        )
        results = model.fit(n_bootstrap=0)
        assert results.lsdv_params["L1.n"] == pytest.approx(STATA_LSDV["L1.n"], abs=1e-4)

    def test_lsdv_w(self):
        """LSDV coefficient on w should match Stata."""
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator="ab",
            bias_order=1,
        )
        results = model.fit(n_bootstrap=0)
        assert results.lsdv_params["w"] == pytest.approx(STATA_LSDV["w"], abs=1e-4)

    def test_lsdv_k(self):
        """LSDV coefficient on k should match Stata."""
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator="ab",
            bias_order=1,
        )
        results = model.fit(n_bootstrap=0)
        assert results.lsdv_params["k"] == pytest.approx(STATA_LSDV["k"], abs=1e-4)


class TestBiasCorrectedRhoDirection:
    """
    Verify bias correction moves rho in the correct direction.

    The Nickell bias is negative (LSDV underestimates rho), so the
    bias-corrected rho should be LARGER than the LSDV rho.
    """

    @pytest.mark.parametrize("init", ["ah", "ab"])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_rho_correction_direction(self, init, order):
        """LSDVC rho > LSDV rho (bias correction is upward)."""
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator=init,
            bias_order=order,
        )
        results = model.fit(n_bootstrap=0)
        assert results.params["L1.n"] > results.lsdv_params["L1.n"], (
            f"LSDVC rho ({results.params['L1.n']:.6f}) should be > "
            f"LSDV rho ({results.lsdv_params['L1.n']:.6f})"
        )


class TestAbdataAgainstStata:
    """
    Compare panelbox LSDVC with Stata xtlsdvc on abdata (ind==4).

    Tolerances are wider than for LSDV because panelbox uses the
    simplified AR(1) bias formula while Stata uses the full matrix
    formula. The rho coefficient is compared; beta coefficients are
    expected to differ (panelbox does not correct them).
    """

    def test_ah_bias3_rho(self):
        """
        Rho with AH init, bias(3) should be close to Stata.

        Stata: 0.6338054
        Panelbox: ~0.6439 (simplified formula, rho-only correction)
        Tolerance: 0.02 (reflects formula simplification)
        """
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator="ah",
            bias_order=3,
        )
        results = model.fit(n_bootstrap=0)
        diff = abs(results.params["L1.n"] - STATA_AH_BIAS3["L1.n"])
        assert diff < 0.02, (
            f"rho diff = {diff:.4f} (panelbox={results.params['L1.n']:.6f}, "
            f"stata={STATA_AH_BIAS3['L1.n']:.6f})"
        )

    def test_ab_bias3_rho(self):
        """
        Rho with AB init, bias(3) should be close to Stata.

        Stata: 0.6360273
        Panelbox: ~0.6439 (simplified formula)
        """
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator="ab",
            bias_order=3,
        )
        results = model.fit(n_bootstrap=0)
        diff = abs(results.params["L1.n"] - STATA_AB_BIAS3["L1.n"])
        assert diff < 0.02, (
            f"rho diff = {diff:.4f} (panelbox={results.params['L1.n']:.6f}, "
            f"stata={STATA_AB_BIAS3['L1.n']:.6f})"
        )

    def test_ah_bias1_rho_order_of_magnitude(self):
        """
        Rho with AH init, bias(1).

        Larger tolerance expected: the simplified bias(1) formula
        -(1+rho)/(T-1) overcorrects when exogenous regressors are present.
        Stata: 0.5390, Panelbox: ~0.757
        We verify the correction is in the right direction and rho < 1.
        """
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator="ah",
            bias_order=1,
        )
        results = model.fit(n_bootstrap=0)
        rho = results.params["L1.n"]
        # Correction should be upward from LSDV (0.406)
        assert rho > STATA_LSDV["L1.n"]
        # Should still be < 1 for a stationary process
        assert rho < 1.0, f"rho = {rho:.6f} >= 1 (explosive)"

    def test_initial_estimator_invariance(self):
        """
        All initial estimators should converge to the same LSDVC
        result (since bias correction is iterated to convergence).
        """
        a4 = _load_abdata_ind4()
        results_by_init = {}
        for init in ["ah", "ab"]:
            model = LSDVC(
                ABDATA_FORMULA,
                a4,
                "id",
                "year",
                initial_estimator=init,
                bias_order=3,
            )
            results_by_init[init] = model.fit(n_bootstrap=0)

        rho_ah = results_by_init["ah"].params["L1.n"]
        rho_ab = results_by_init["ab"].params["L1.n"]
        assert rho_ah == pytest.approx(rho_ab, abs=1e-6), (
            f"AH rho={rho_ah:.7f} != AB rho={rho_ab:.7f}"
        )

    def test_beta_not_corrected_documented(self):
        """
        Beta coefficients should remain at LSDV values (known limitation).

        This test documents the difference: panelbox does not apply
        bias correction to exogenous variable coefficients.
        """
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator="ab",
            bias_order=3,
        )
        results = model.fit(n_bootstrap=0)

        # Beta coefficients should be identical to LSDV
        assert results.params["w"] == pytest.approx(results.lsdv_params["w"], abs=1e-10)
        assert results.params["k"] == pytest.approx(results.lsdv_params["k"], abs=1e-10)

        # Document the Stata difference
        w_diff = abs(results.params["w"] - STATA_AB_BIAS3["w"])
        k_diff = abs(results.params["k"] - STATA_AB_BIAS3["k"])
        # These differences exist because Stata corrects beta too
        assert w_diff > 0.01, "Expected w to differ from Stata (no beta correction)"
        assert k_diff > 0.01, "Expected k to differ from Stata (no beta correction)"


class TestAbdataBootstrapSEs:
    """Compare bootstrap SEs with Stata on abdata."""

    @pytest.mark.slow
    def test_bootstrap_se_order_of_magnitude(self):
        """
        Bootstrap SEs should be in the same order of magnitude as Stata.

        Stata AB bias(3) vcov(100): SE(L1.n) = 0.091
        Tolerance: within factor of 2 (bootstrap variance is high with
        few replications, and the DGP differs slightly).
        """
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator="ab",
            bias_order=3,
        )
        results = model.fit(n_bootstrap=200, seed=42)
        se_rho = results.std_errors["L1.n"]
        stata_se_rho = STATA_AB_BIAS3_SE["L1.n"]

        # Within factor of 2
        ratio = se_rho / stata_se_rho
        assert 0.5 < ratio < 2.0, (
            f"SE ratio = {ratio:.2f} (panelbox={se_rho:.4f}, stata={stata_se_rho:.4f})"
        )


class TestGrunfeldValidation:
    """
    Validate LSDVC on Grunfeld dataset (N=10, T=20).

    No Stata reference values available for this specific model.
    Instead, we validate internal consistency and known properties.
    """

    def test_grunfeld_ab_order2_coefficients(self):
        """Verify Grunfeld AB order=2 produces reasonable coefficients."""
        data = load_grunfeld()
        model = LSDVC(
            "invest ~ value + capital",
            data,
            "firm",
            "year",
            initial_estimator="ab",
            bias_order=2,
        )
        results = model.fit(n_bootstrap=0)

        # Rho should be positive and < 1 (stationary)
        rho = results.params["L1.invest"]
        assert 0.0 < rho < 1.0, f"rho = {rho:.6f} out of (0, 1)"

        # Value and capital should be positive (investment increases with
        # firm value and capital stock)
        assert results.params["value"] > 0
        assert results.params["capital"] > 0

        # Bias correction should be upward
        assert rho > results.lsdv_params["L1.invest"]

    def test_grunfeld_all_initializers_converge(self):
        """All initial estimators should give the same LSDVC result."""
        data = load_grunfeld()
        rho_values = {}
        for init in ["ah", "ab", "bb"]:
            model = LSDVC(
                "invest ~ value + capital",
                data,
                "firm",
                "year",
                initial_estimator=init,
                bias_order=2,
            )
            results = model.fit(n_bootstrap=0)
            rho_values[init] = results.params["L1.invest"]

        # All should converge to same value
        assert rho_values["ah"] == pytest.approx(rho_values["ab"], abs=1e-5)
        assert rho_values["ab"] == pytest.approx(rho_values["bb"], abs=1e-5)

    def test_grunfeld_bias_order_hierarchy(self):
        """Higher-order corrections should refine the estimate."""
        data = load_grunfeld()
        rho_by_order = {}
        lsdv_rho = None
        for order in [1, 2, 3]:
            model = LSDVC(
                "invest ~ value + capital",
                data,
                "firm",
                "year",
                initial_estimator="ab",
                bias_order=order,
            )
            results = model.fit(n_bootstrap=0)
            rho_by_order[order] = results.params["L1.invest"]
            lsdv_rho = results.lsdv_params["L1.invest"]

        # All should correct upward from LSDV
        for order in [1, 2, 3]:
            assert rho_by_order[order] > lsdv_rho

        # Order 2 and 3 should be close to each other (refinement diminishes)
        diff_23 = abs(rho_by_order[2] - rho_by_order[3])
        diff_12 = abs(rho_by_order[1] - rho_by_order[2])
        assert diff_23 < diff_12, "Order 2->3 refinement should be smaller than 1->2"

    def test_grunfeld_reference_values(self):
        """
        Pin down Grunfeld reference values for regression testing.

        These values were computed by panelbox LSDVC and serve as
        internal regression baselines (not Stata cross-validated).
        """
        data = load_grunfeld()
        model = LSDVC(
            "invest ~ value + capital",
            data,
            "firm",
            "year",
            initial_estimator="ab",
            bias_order=2,
        )
        results = model.fit(n_bootstrap=0)

        # Regression test: pin to known values
        assert results.params["L1.invest"] == pytest.approx(0.710286, abs=1e-3)
        assert results.params["value"] == pytest.approx(0.101530, abs=1e-3)
        assert results.params["capital"] == pytest.approx(0.152632, abs=1e-3)

    @pytest.mark.slow
    def test_grunfeld_bootstrap_reproducibility(self):
        """Bootstrap with same seed should give identical results."""
        data = load_grunfeld()
        results = []
        for _ in range(2):
            model = LSDVC(
                "invest ~ value + capital",
                data,
                "firm",
                "year",
                initial_estimator="ab",
                bias_order=2,
            )
            results.append(model.fit(n_bootstrap=100, seed=123))

        np.testing.assert_array_almost_equal(
            results[0].std_errors.values,
            results[1].std_errors.values,
            decimal=10,
        )


class TestNickellBiasMagnitude:
    """
    Validate the magnitude of Nickell bias against analytical formula.

    For a balanced panel AR(1) with no exogenous regressors, the
    Nickell bias is approximately -(1+rho)/(T-1). We verify that the
    LSDV bias in our estimation is consistent with this formula.
    """

    def test_nickell_bias_grunfeld(self):
        """
        LSDV bias on Grunfeld should be approximately -(1+rho)/(T-1).

        With T=19 (after losing 1 period to lag) and rho~0.71:
        Expected bias ~ -(1+0.71)/18 ~ -0.095
        Observed: LSDV_rho=0.623, LSDVC_rho=0.710, diff=-0.087
        """
        data = load_grunfeld()
        model = LSDVC(
            "invest ~ value + capital",
            data,
            "firm",
            "year",
            initial_estimator="ab",
            bias_order=2,
        )
        results = model.fit(n_bootstrap=0)

        rho_lsdvc = results.params["L1.invest"]
        rho_lsdv = results.lsdv_params["L1.invest"]
        observed_bias = rho_lsdv - rho_lsdvc  # negative

        # Analytical Nickell bias (leading term)
        T = 19  # Grunfeld: 20 years - 1 lag
        expected_bias = -(1 + rho_lsdvc) / (T - 1)

        # Should be in the same ballpark (within 50%)
        ratio = observed_bias / expected_bias
        assert 0.5 < ratio < 2.0, (
            f"Bias ratio = {ratio:.2f} (observed={observed_bias:.4f}, expected={expected_bias:.4f})"
        )

    def test_nickell_bias_abdata(self):
        """
        LSDV bias on abdata (ind==4) should be larger than Grunfeld
        because T is smaller (T~8 vs T=20), making Nickell bias worse.
        """
        # Grunfeld
        grunfeld = load_grunfeld()
        model_g = LSDVC(
            "invest ~ value + capital",
            grunfeld,
            "firm",
            "year",
            initial_estimator="ab",
            bias_order=2,
        )
        res_g = model_g.fit(n_bootstrap=0)
        bias_g = abs(res_g.lsdv_params["L1.invest"] - res_g.params["L1.invest"])

        # abdata without year dummies (simpler model for comparison)
        a = load_abdata()
        a4 = a[a["ind"] == 4].copy()
        a4["id"] = a4["id"].astype(int)
        a4["year"] = a4["year"].astype(int)
        model_a = LSDVC(
            "n ~ w + k",
            a4,
            "id",
            "year",
            initial_estimator="ab",
            bias_order=2,
        )
        res_a = model_a.fit(n_bootstrap=0)
        bias_a = abs(res_a.lsdv_params["L1.n"] - res_a.params["L1.n"])

        # abdata has smaller T, so bias should be larger
        assert bias_a > bias_g, (
            f"abdata bias ({bias_a:.4f}) should be > Grunfeld bias ({bias_g:.4f}) due to smaller T"
        )


class TestConvergenceProperties:
    """Validate convergence behavior of iterated bias correction."""

    def test_convergence_grunfeld(self):
        """Grunfeld should converge within a few iterations."""
        data = load_grunfeld()
        model = LSDVC(
            "invest ~ value + capital",
            data,
            "firm",
            "year",
            initial_estimator="ab",
            bias_order=2,
        )
        results = model.fit(n_bootstrap=0)
        assert results.converged
        assert results.n_iterations < 20

    def test_convergence_abdata(self):
        """abdata (ind==4) should converge."""
        a4 = _load_abdata_ind4()
        model = LSDVC(
            ABDATA_FORMULA,
            a4,
            "id",
            "year",
            initial_estimator="ab",
            bias_order=3,
        )
        results = model.fit(n_bootstrap=0)
        assert results.converged
        assert results.n_iterations < 20

    def test_bb_fewer_iterations(self):
        """
        BB initial estimate should require fewer iterations since
        it is typically closer to the true rho than AH.
        """
        data = load_grunfeld()
        iters = {}
        for init in ["ah", "ab", "bb"]:
            model = LSDVC(
                "invest ~ value + capital",
                data,
                "firm",
                "year",
                initial_estimator=init,
                bias_order=2,
            )
            results = model.fit(n_bootstrap=0)
            iters[init] = results.n_iterations

        # BB should need <= iterations as AH (BB starts closer to truth)
        assert iters["bb"] <= iters["ah"], f"BB iterations ({iters['bb']}) > AH ({iters['ah']})"
