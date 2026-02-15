"""
Tests for Quantile Treatment Effects.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.quantile.treatment_effects import QTEResult, QuantileTreatmentEffects
from panelbox.utils.data import PanelData


class TestQuantileTreatmentEffects:
    """Tests for QTE estimation."""

    @pytest.fixture
    def treatment_data(self):
        """Generate data with treatment effect."""
        np.random.seed(123)
        n = 500

        # Covariates
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)

        # Treatment assignment (correlated with X1)
        propensity = 1 / (1 + np.exp(-0.5 * X1))
        D = np.random.binomial(1, propensity)

        # Outcome with heterogeneous treatment effect
        # Treatment effect varies across distribution
        y0 = 2 + X1 - 0.5 * X2 + np.random.randn(n)
        treatment_effect = 1 + 0.5 * X1 + np.random.randn(n) * 0.5
        y1 = y0 + treatment_effect
        y = D * y1 + (1 - D) * y0

        df = pd.DataFrame({"y": y, "treatment": D, "X1": X1, "X2": X2})

        return df

    def test_standard_qte(self, treatment_data):
        """Test standard QTE estimation."""
        qte = QuantileTreatmentEffects(
            treatment_data, outcome="y", treatment="treatment", covariates=["X1", "X2"]
        )

        result = qte.estimate_qte(
            tau=[0.25, 0.5, 0.75], method="standard", bootstrap=True, n_boot=100
        )

        # Check result structure
        assert isinstance(result, QTEResult)
        assert len(result.qte_results) == 3
        assert result.method == "standard"

        # Check that QTE varies across quantiles (heterogeneous effect)
        qte_values = [result.qte_results[tau]["qte"] for tau in [0.25, 0.5, 0.75]]
        assert len(set(qte_values)) > 1  # Not all identical

        # Check bootstrap inference
        for tau in [0.25, 0.5, 0.75]:
            assert result.qte_results[tau]["se"] is not None
            assert result.qte_results[tau]["ci_lower"] is not None
            assert result.qte_results[tau]["ci_upper"] is not None

    def test_unconditional_qte(self, treatment_data):
        """Test unconditional QTE via RIF."""
        qte = QuantileTreatmentEffects(
            treatment_data, outcome="y", treatment="treatment", covariates=["X1", "X2"]
        )

        result = qte.estimate_qte(tau=[0.25, 0.5, 0.75], method="unconditional")

        # Check result
        assert result.method == "unconditional"
        assert len(result.qte_results) == 3

        # Should have unconditional quantile info
        for tau in [0.25, 0.5, 0.75]:
            assert "unconditional_quantile" in result.qte_results[tau]
            assert "density" in result.qte_results[tau]

    def test_did_qte(self):
        """Test difference-in-differences QTE."""
        np.random.seed(456)
        n_entities = 50
        n_time = 2

        # Generate panel data
        data_list = []
        for entity in range(n_entities):
            # Treatment group (half of entities)
            treated = entity < n_entities // 2

            for time in range(n_time):
                # Post-treatment period
                post = time == 1

                # Outcome
                y = (
                    2
                    + np.random.randn()
                    + treated * 0.5
                    + post * 0.3  # Treated group baseline difference
                    + treated * post * 1.0  # Time trend
                )  # Treatment effect

                data_list.append(
                    {"y": y, "treatment": int(treated), "entity": entity, "time": time}
                )

        df = pd.DataFrame(data_list)
        panel_data = PanelData(df, entity="entity", time="time")

        qte = QuantileTreatmentEffects(
            panel_data, outcome="y", treatment="treatment", entity_col="entity", time_col="time"
        )

        result = qte.estimate_qte(
            tau=[0.25, 0.5, 0.75], method="did", pre_post_cutoff=0.5, bootstrap=True, n_boot=100
        )

        # Check DiD results
        assert result.method == "did"
        assert len(result.qte_results) == 3

        # Check components
        for tau in [0.25, 0.5, 0.75]:
            assert "treated_change" in result.qte_results[tau]
            assert "control_change" in result.qte_results[tau]
            assert "qte" in result.qte_results[tau]

            # QTE should be around 1.0 (true effect)
            qte_val = result.qte_results[tau]["qte"]
            assert 0.5 < qte_val < 1.5  # Reasonable range

    def test_cic_qte(self):
        """Test changes-in-changes QTE."""
        np.random.seed(789)

        # Simple CiC data
        n_per_group = 100
        data_list = []

        # Control group, period 0
        y00 = np.random.randn(n_per_group)
        data_list.extend([{"y": y, "treatment": 0, "time": 0} for y in y00])

        # Control group, period 1 (shift in distribution)
        y01 = np.random.randn(n_per_group) + 0.5
        data_list.extend([{"y": y, "treatment": 0, "time": 1} for y in y01])

        # Treated group, period 0
        y10 = np.random.randn(n_per_group) + 0.3
        data_list.extend([{"y": y, "treatment": 1, "time": 0} for y in y10])

        # Treated group, period 1 (with treatment effect)
        y11 = np.random.randn(n_per_group) + 0.3 + 0.5 + 1.0  # baseline + trend + effect
        data_list.extend([{"y": y, "treatment": 1, "time": 1} for y in y11])

        df = pd.DataFrame(data_list)

        qte = QuantileTreatmentEffects(
            df, outcome="y", treatment="treatment", entity_col=None, time_col="time"
        )

        result = qte.estimate_qte(tau=[0.25, 0.5, 0.75], method="cic")

        # Check CiC results
        assert result.method == "cic"
        assert len(result.qte_results) == 3

        for tau in [0.25, 0.5, 0.75]:
            assert "q_11" in result.qte_results[tau]
            assert "q_11_counterfactual" in result.qte_results[tau]
            assert "change_control" in result.qte_results[tau]

    def test_binary_conversion(self):
        """Test automatic binary treatment conversion."""
        np.random.seed(111)
        n = 100

        df = pd.DataFrame(
            {
                "y": np.random.randn(n),
                "treatment": np.random.choice([0, 1, 2], n),  # Non-binary
                "X": np.random.randn(n),
            }
        )

        with pytest.warns(UserWarning):
            qte = QuantileTreatmentEffects(df, outcome="y", treatment="treatment", covariates=["X"])

        # Should have converted to binary
        assert len(np.unique(qte.d)) == 2
        assert set(np.unique(qte.d)) == {0, 1}

    def test_no_covariates(self, treatment_data):
        """Test QTE without covariates."""
        qte = QuantileTreatmentEffects(
            treatment_data, outcome="y", treatment="treatment", covariates=None  # No covariates
        )

        result = qte.estimate_qte(tau=0.5, method="standard", bootstrap=False)

        # Should work with intercept only
        assert result is not None
        assert 0.5 in result.qte_results

    def test_plot_qte(self, treatment_data):
        """Test QTE plotting."""
        qte = QuantileTreatmentEffects(
            treatment_data, outcome="y", treatment="treatment", covariates=["X1", "X2"]
        )

        result = qte.estimate_qte(
            tau=np.arange(0.1, 1.0, 0.1), method="standard", bootstrap=False  # Skip for speed
        )

        # Should create plot without error
        fig = qte.plot_qte(result, show_ate=True)
        assert fig is not None


class TestQTEResult:
    """Tests for QTEResult class."""

    @pytest.fixture
    def mock_results(self):
        """Create mock QTE results."""
        qte_results = {}
        for tau in [0.25, 0.5, 0.75]:
            qte_results[tau] = {
                "qte": 1.0 + 0.5 * tau,  # Increasing effect
                "se": 0.1,
                "ci_lower": 0.8 + 0.5 * tau,
                "ci_upper": 1.2 + 0.5 * tau,
            }
        return QTEResult(qte_results, method="standard")

    def test_result_summary(self, mock_results):
        """Test result summary."""
        # Should print without error
        mock_results.summary()

    def test_to_dataframe(self, mock_results):
        """Test DataFrame conversion."""
        df = mock_results.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "tau" in df.columns
        assert "qte" in df.columns
        assert "se" in df.columns

    def test_constant_effects_test(self, mock_results):
        """Test hypothesis test for constant effects."""
        test_result = mock_results.test_constant_effects()

        assert "test_statistic" in test_result
        assert "p_value" in test_result
        assert "reject_constant" in test_result

    def test_heterogeneity_detection(self):
        """Test detection of heterogeneous effects."""
        # Create results with heterogeneous effects
        qte_results = {}
        for tau in np.arange(0.1, 1.0, 0.1):
            qte_results[tau] = {
                "qte": 1.0 + 2.0 * tau,  # Strong heterogeneity
                "se": 0.1,
                "ci_lower": 0.8 + 2.0 * tau,
                "ci_upper": 1.2 + 2.0 * tau,
            }

        result = QTEResult(qte_results, method="standard")
        result.summary()

        # Should detect heterogeneity
        qte_values = [res["qte"] for res in result.qte_results.values()]
        heterogeneity = np.std(qte_values)
        assert heterogeneity > 0.5  # Substantial heterogeneity


class TestDensityEstimation:
    """Test density estimation for RIF."""

    def test_density_at_quantile(self):
        """Test kernel density estimation at quantile."""
        np.random.seed(222)
        y = np.random.randn(500)

        df = pd.DataFrame({"y": y, "treatment": np.random.binomial(1, 0.5, 500)})

        qte = QuantileTreatmentEffects(df, outcome="y", treatment="treatment")

        # Estimate density at median
        q_median = np.median(y)
        density = qte._density_at_quantile(q_median)

        # Should be close to normal density at 0
        expected_density = 1 / np.sqrt(2 * np.pi)
        assert 0.2 < density < 0.5  # Reasonable range

    def test_density_bandwidth(self):
        """Test different bandwidth specifications."""
        np.random.seed(333)
        y = np.random.randn(200)

        df = pd.DataFrame({"y": y, "treatment": np.random.binomial(1, 0.5, 200)})

        qte = QuantileTreatmentEffects(df, outcome="y", treatment="treatment")

        # Test with different bandwidths
        q = np.median(y)
        density1 = qte._density_at_quantile(q, bandwidth=0.1)
        density2 = qte._density_at_quantile(q, bandwidth=0.5)

        # Different bandwidths should give different results
        assert density1 != density2
        assert density1 > 0 and density2 > 0
