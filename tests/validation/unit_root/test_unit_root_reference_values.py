"""Regression tests pinning LLC and IPS statistics to R plm::purtest.

Reference values were computed with plm 2.6 on a panel of 50 independent
random walks of length 50 (``numpy.random.default_rng(0)``), using
``purtest(..., lags = 0)`` with ``exo = "intercept"`` / ``"trend"``.
Before the adjustment terms of Levin, Lin and Chu (2002) were implemented,
LLC returned -8.08 (p = 0.000) on this panel, rejecting a true unit root.
"""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from panelbox.validation.unit_root import IPSTest, LLCTest

PLM = {
    # (test, trend): (statistic, pvalue)
    ("llc", "c"): (-0.475, 0.317),
    ("llc", "ct"): (-1.994, 0.023),
    ("ips", "c"): (1.504, 0.934),
    ("ips", "ct"): (-0.908, 0.182),
}


@pytest.fixture(scope="module")
def random_walk_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_entities, n_periods = 50, 50
    rows = [
        (i, t, y)
        for i in range(n_entities)
        for t, y in enumerate(np.cumsum(rng.normal(size=n_periods)) + 10)
    ]
    return pd.DataFrame(rows, columns=["id", "year", "y"])


def _run(cls, data, trend):
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(data, variable="y", entity_col="id", time_col="year", trend=trend, lags=0).run()


@pytest.mark.parametrize("trend", ["c", "ct"])
def test_llc_matches_plm(random_walk_panel, trend):
    stat, pvalue = PLM[("llc", trend)]
    res = _run(LLCTest, random_walk_panel, trend)
    assert res.statistic == pytest.approx(stat, abs=0.07)
    assert res.pvalue == pytest.approx(pvalue, abs=0.01)


@pytest.mark.parametrize("trend", ["c", "ct"])
def test_ips_matches_plm(random_walk_panel, trend):
    stat, pvalue = PLM[("ips", trend)]
    res = _run(IPSTest, random_walk_panel, trend)
    assert res.statistic == pytest.approx(stat, abs=1e-3)
    assert res.pvalue == pytest.approx(pvalue, abs=1e-3)


def test_llc_does_not_reject_true_unit_root(random_walk_panel):
    res = _run(LLCTest, random_walk_panel, "c")
    assert res.pvalue > 0.05
