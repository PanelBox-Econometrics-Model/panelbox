"""Regression tests pinning GMM estimates to external reference implementations.

Reference values were produced on the bundled Arellano-Bond panel (abdata,
140 firms, 1976-1984, 1,031 obs) for the model

    n ~ L1.n + w + k [+ year dummies]

with GMM-style instruments lag(n, 2:9), IV-style w and k, two-step
estimation and Windmeijer-corrected standard errors.

* Difference GMM: R plm 2.6 ``pgmm(..., transformation = "d")`` with
  ``summary(robust = TRUE)``. PanelBox must match to 3 decimals.
* System GMM: pydynpd 0.2 (a Python replica of Stata's xtabond2, whose
  default h(3) weighting PanelBox follows). plm's system GMM weights the level
  equation differently and is not a valid reference for it. PanelBox still
  drops the first level-equation row per entity that xtabond2 keeps, so the
  tolerance is 2% on coefficients and 5% on standard errors.
"""

import contextlib
import io

import numpy as np
import pytest

import panelbox as pb

COMMON = {
    "dep_var": "n",
    "lags": 1,
    "id_var": "id",
    "time_var": "year",
    "exog_vars": ["w", "k"],
    "two_step": True,
    "gmm_max_lag": 9,
}

PLM_DIFFERENCE = {
    # (collapse, time_dummies): coef, se, hansen_p, n_instruments
    (True, False): {
        "coef": [0.6990, -0.7692, 0.2408],
        "se": [0.1282, 0.1042, 0.0659],
        "hansen_p": 0.002,
        "n_instruments": 9,
    },
    (True, True): {
        "coef": [0.8766, -0.4587, 0.1907],
        "se": [0.2342, 0.1698, 0.0649],
        "hansen_p": 0.244,
        "n_instruments": 16,
    },
    (False, True): {
        "coef": [0.2617, -0.3681, 0.2840],
        "se": [0.1384, 0.1378, 0.0607],
        "hansen_p": 0.050,
        "n_instruments": 37,
    },
}

PYDYNPD_SYSTEM = {
    (True, True): {
        "coef": [0.5621990, -0.2413994, 0.3547180],
        "se": [0.1871391, 0.1115897, 0.1462262],
        "hansen_p": 0.230,
        "n_instruments": 18,
    },
    (False, True): {
        "coef": [0.7556161, -0.1735291, 0.1985330],
        "se": [0.0646745, 0.0810154, 0.0486740],
        "hansen_p": 0.057,
        "n_instruments": 45,
    },
}


@pytest.fixture(scope="module")
def abdata():
    return pb.load_dataset("abdata")


def _fit(cls, data, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(data, **COMMON, **kw).fit()


@pytest.mark.parametrize(("collapse", "time_dummies"), sorted(PLM_DIFFERENCE))
def test_difference_gmm_matches_plm(abdata, collapse, time_dummies):
    ref = PLM_DIFFERENCE[(collapse, time_dummies)]
    res = _fit(pb.DifferenceGMM, abdata, collapse=collapse, time_dummies=time_dummies)

    np.testing.assert_allclose(res.params.values[:3], ref["coef"], atol=5e-4)
    np.testing.assert_allclose(res.std_errors.values[:3], ref["se"], atol=5e-4)
    assert abs(res.hansen_j.pvalue - ref["hansen_p"]) < 1e-3
    assert res.n_instruments == ref["n_instruments"]
    assert res.nobs == 751


@pytest.mark.parametrize(("collapse", "time_dummies"), sorted(PYDYNPD_SYSTEM))
def test_system_gmm_matches_xtabond2_replica(abdata, collapse, time_dummies):
    ref = PYDYNPD_SYSTEM[(collapse, time_dummies)]
    res = _fit(pb.SystemGMM, abdata, collapse=collapse, time_dummies=time_dummies)

    np.testing.assert_allclose(res.params.values[:3], ref["coef"], rtol=0.02, atol=0.01)
    np.testing.assert_allclose(res.std_errors.values[:3], ref["se"], rtol=0.05, atol=0.01)
    assert abs(res.hansen_j.pvalue - ref["hansen_p"]) < 0.01
    assert res.n_instruments == ref["n_instruments"]
    assert res.nobs == 751


def test_time_dummies_are_instrumented(abdata):
    """Year dummies must enter the instrument matrix, not only the regressors.

    Before this was fixed, collapsed system GMM with year dummies was
    reported as under-identified (9 instruments for 13 parameters).
    """
    res = _fit(pb.SystemGMM, abdata, collapse=True, time_dummies=True)
    assert res.n_instruments == 18
    assert len(res.params) == 1 + 2 + 7 + 1  # L.n, w, k, 7 year dummies, constant
