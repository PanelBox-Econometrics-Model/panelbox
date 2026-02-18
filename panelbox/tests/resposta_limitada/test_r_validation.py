"""
Validation tests comparing PanelBox to R benchmarks.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelbox.models.censored import PooledTobit
from panelbox.models.count import NegativeBinomial, PoissonFixedEffects, PooledPoisson
from panelbox.models.discrete.binary import FixedEffectsLogit, PooledLogit, PooledProbit

# Tolerances
COEF_RTOL = 0.05  # 5% relative tolerance for coefficients
SE_RTOL = 0.10  # 10% relative tolerance for standard errors
ME_RTOL = 0.10  # 10% for marginal effects


def load_r_results(filename):
    """Load R benchmark results from JSON."""
    path = Path(__file__).parent / "r/results" / filename
    with open(path, "r") as f:
        return json.load(f)


class TestPooledLogitVsR:
    """Compare PooledLogit to R glm(family=binomial(link='logit'))"""

    @pytest.fixture
    def data(self):
        return pd.read_csv("tests/resposta_limitada/data/binary_panel_test.csv")

    @pytest.fixture
    def r_results(self):
        return load_r_results("pooled_logit_results.json")

    def test_coefficients(self, data, r_results):
        model = PooledLogit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.params[var], r_results["coef"][var], rtol=COEF_RTOL
            ), f"Coefficient mismatch for {var}: Python={result.params[var]:.6f}, R={r_results['coef'][var]:.6f}"

    def test_standard_errors(self, data, r_results):
        model = PooledLogit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.std_errors[var], r_results["se"][var], rtol=SE_RTOL
            ), f"SE mismatch for {var}: Python={result.std_errors[var]:.6f}, R={r_results['se'][var]:.6f}"

    def test_loglikelihood(self, data, r_results):
        model = PooledLogit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        assert np.isclose(
            result.llf, r_results["loglik"], rtol=0.001
        ), f"Log-likelihood mismatch: Python={result.llf:.6f}, R={r_results['loglik']:.6f}"

    def test_marginal_effects(self, data, r_results):
        model = PooledLogit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()
        ame = model.marginal_effects(at="overall")

        r_ame = {row["factor"]: row["AME"] for row in r_results["ame"]}

        for var in ["x1", "x2"]:
            assert np.isclose(
                ame.marginal_effects[var], r_ame[var], rtol=ME_RTOL
            ), f"AME mismatch for {var}: Python={ame.marginal_effects[var]:.6f}, R={r_ame[var]:.6f}"


class TestPooledProbitVsR:
    """Compare PooledProbit to R glm(family=binomial(link='probit'))"""

    @pytest.fixture
    def data(self):
        return pd.read_csv("tests/resposta_limitada/data/binary_panel_test.csv")

    @pytest.fixture
    def r_results(self):
        return load_r_results("pooled_probit_results.json")

    def test_coefficients(self, data, r_results):
        model = PooledProbit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.params[var], r_results["coef"][var], rtol=COEF_RTOL
            ), f"Coefficient mismatch for {var}"

    def test_standard_errors(self, data, r_results):
        model = PooledProbit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.std_errors[var], r_results["se"][var], rtol=SE_RTOL
            ), f"SE mismatch for {var}"

    def test_loglikelihood(self, data, r_results):
        model = PooledProbit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        assert np.isclose(result.llf, r_results["loglik"], rtol=0.001), f"Log-likelihood mismatch"

    def test_marginal_effects(self, data, r_results):
        model = PooledProbit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()
        ame = model.marginal_effects(at="overall")

        r_ame = {row["factor"]: row["AME"] for row in r_results["ame"]}

        for var in ["x1", "x2"]:
            assert np.isclose(
                ame.marginal_effects[var], r_ame[var], rtol=ME_RTOL
            ), f"AME mismatch for {var}"


class TestFELogitVsR:
    """Compare FixedEffectsLogit to R clogit"""

    @pytest.fixture
    def data(self):
        return pd.read_csv("tests/resposta_limitada/data/binary_panel_test.csv")

    @pytest.fixture
    def r_results(self):
        return load_r_results("fe_logit_results.json")

    def test_coefficients(self, data, r_results):
        model = FixedEffectsLogit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.params[var], r_results["coef"][var], rtol=COEF_RTOL
            ), f"FE Logit coefficient mismatch for {var}"

    def test_standard_errors(self, data, r_results):
        model = FixedEffectsLogit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.std_errors[var], r_results["se"][var], rtol=SE_RTOL
            ), f"FE Logit SE mismatch for {var}"

    def test_loglikelihood(self, data, r_results):
        model = FixedEffectsLogit("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        assert np.isclose(
            result.llf, r_results["loglik"], rtol=0.001
        ), f"FE Logit log-likelihood mismatch"


class TestPooledTobitVsR:
    """Compare PooledTobit to R censReg"""

    @pytest.fixture
    def data(self):
        return pd.read_csv("tests/resposta_limitada/data/censored_panel_test.csv")

    @pytest.fixture
    def r_results(self):
        return load_r_results("pooled_tobit_results.json")

    def test_coefficients(self, data, r_results):
        model = PooledTobit("y ~ x1 + x2", data, "entity", "time", left=0)
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.params[var], r_results["coef"][var], rtol=COEF_RTOL
            ), f"Tobit coefficient mismatch for {var}"

    def test_standard_errors(self, data, r_results):
        model = PooledTobit("y ~ x1 + x2", data, "entity", "time", left=0)
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.std_errors[var], r_results["se"][var], rtol=SE_RTOL
            ), f"Tobit SE mismatch for {var}"

    def test_sigma(self, data, r_results):
        model = PooledTobit("y ~ x1 + x2", data, "entity", "time", left=0)
        result = model.fit()

        assert np.isclose(
            result.scale, r_results["sigma"], rtol=COEF_RTOL
        ), f"Sigma mismatch: Python={result.scale:.6f}, R={r_results['sigma']:.6f}"

    def test_loglikelihood(self, data, r_results):
        model = PooledTobit("y ~ x1 + x2", data, "entity", "time", left=0)
        result = model.fit()

        assert np.isclose(
            result.llf, r_results["loglik"], rtol=0.001
        ), f"Tobit log-likelihood mismatch"


class TestPooledPoissonVsR:
    """Compare PooledPoisson to R glm(family=poisson)"""

    @pytest.fixture
    def data(self):
        return pd.read_csv("tests/resposta_limitada/data/count_panel_test.csv")

    @pytest.fixture
    def r_results(self):
        return load_r_results("pooled_poisson_results.json")

    def test_coefficients(self, data, r_results):
        model = PooledPoisson("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.params[var], r_results["coef"][var], rtol=COEF_RTOL
            ), f"Poisson coefficient mismatch for {var}"

    def test_standard_errors(self, data, r_results):
        model = PooledPoisson("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.std_errors[var], r_results["se"][var], rtol=SE_RTOL
            ), f"Poisson SE mismatch for {var}"

    def test_loglikelihood(self, data, r_results):
        model = PooledPoisson("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        assert np.isclose(
            result.llf, r_results["loglik"], rtol=0.001
        ), f"Poisson log-likelihood mismatch"

    def test_marginal_effects(self, data, r_results):
        model = PooledPoisson("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()
        ame = model.marginal_effects(at="overall")

        r_ame = r_results["ame"]

        for var in ["x1", "x2"]:
            assert np.isclose(
                ame.marginal_effects[var], r_ame[var], rtol=ME_RTOL
            ), f"Poisson AME mismatch for {var}"


class TestFEPoissonVsR:
    """Compare PoissonFixedEffects to R pglm"""

    @pytest.fixture
    def data(self):
        return pd.read_csv("tests/resposta_limitada/data/count_panel_test.csv")

    @pytest.fixture
    def r_results(self):
        return load_r_results("fe_poisson_results.json")

    def test_coefficients(self, data, r_results):
        model = PoissonFixedEffects("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.params[var], r_results["coef"][var], rtol=COEF_RTOL
            ), f"FE Poisson coefficient mismatch for {var}"

    def test_standard_errors(self, data, r_results):
        model = PoissonFixedEffects("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.std_errors[var], r_results["se"][var], rtol=SE_RTOL
            ), f"FE Poisson SE mismatch for {var}"

    def test_loglikelihood(self, data, r_results):
        model = PoissonFixedEffects("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        assert np.isclose(
            result.llf, r_results["loglik"], rtol=0.001
        ), f"FE Poisson log-likelihood mismatch"


class TestNegativeBinomialVsR:
    """Compare NegativeBinomial to R glm.nb"""

    @pytest.fixture
    def data(self):
        return pd.read_csv("tests/resposta_limitada/data/count_panel_test.csv")

    @pytest.fixture
    def r_results(self):
        return load_r_results("negbin_results.json")

    def test_coefficients(self, data, r_results):
        model = NegativeBinomial("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.params[var], r_results["coef"][var], rtol=COEF_RTOL
            ), f"NegBin coefficient mismatch for {var}"

    def test_standard_errors(self, data, r_results):
        model = NegativeBinomial("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        for var in ["x1", "x2"]:
            assert np.isclose(
                result.std_errors[var], r_results["se"][var], rtol=SE_RTOL
            ), f"NegBin SE mismatch for {var}"

    def test_theta(self, data, r_results):
        model = NegativeBinomial("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        assert np.isclose(
            result.theta, r_results["theta"], rtol=0.15  # Theta pode ter mais variação
        ), f"Theta mismatch: Python={result.theta:.6f}, R={r_results['theta']:.6f}"

    def test_loglikelihood(self, data, r_results):
        model = NegativeBinomial("y ~ x1 + x2", data, "entity", "time")
        result = model.fit()

        assert np.isclose(
            result.llf, r_results["loglik"], rtol=0.001
        ), f"NegBin log-likelihood mismatch"
