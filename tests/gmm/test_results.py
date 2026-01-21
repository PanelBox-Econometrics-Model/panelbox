"""
Unit tests for GMM results classes.
"""

import pytest
import numpy as np
import pandas as pd
from panelbox.gmm.results import TestResult, GMMResults


class TestTestResult:
    """Tests for TestResult class."""

    def test_creation(self):
        """Test basic TestResult creation."""
        result = TestResult(
            name='Hansen J-test',
            statistic=10.5,
            pvalue=0.15,
            df=5,
            distribution='chi2',
            null_hypothesis='Instruments are valid'
        )

        assert result.name == 'Hansen J-test'
        assert result.statistic == 10.5
        assert result.pvalue == 0.15
        assert result.df == 5
        assert result.conclusion == 'PASS'

    def test_hansen_conclusion_pass(self):
        """Test Hansen J-test conclusion logic for PASS."""
        result = TestResult(
            name='Hansen J-test',
            statistic=10.0,
            pvalue=0.18,  # Between 0.10 and 0.25
            df=5
        )
        assert result.conclusion == 'PASS'

    def test_hansen_conclusion_reject(self):
        """Test Hansen J-test conclusion logic for REJECT."""
        result = TestResult(
            name='Hansen J-test',
            statistic=20.0,
            pvalue=0.05,  # Below 0.10
            df=5
        )
        assert result.conclusion == 'REJECT'

    def test_hansen_conclusion_warning(self):
        """Test Hansen J-test conclusion logic for WARNING."""
        result = TestResult(
            name='Hansen J-test',
            statistic=2.0,
            pvalue=0.85,  # Above 0.25
            df=5
        )
        assert result.conclusion == 'WARNING'

    def test_ar2_conclusion_pass(self):
        """Test AR(2) test conclusion logic for PASS."""
        result = TestResult(
            name='AR(2) test',
            statistic=1.5,
            pvalue=0.13,  # Above 0.10
            distribution='normal'
        )
        assert result.conclusion == 'PASS'

    def test_ar2_conclusion_reject(self):
        """Test AR(2) test conclusion logic for REJECT."""
        result = TestResult(
            name='AR(2) test',
            statistic=2.8,
            pvalue=0.005,  # Below 0.10
            distribution='normal'
        )
        assert result.conclusion == 'REJECT'

    def test_ar1_conclusion_expected(self):
        """Test AR(1) test conclusion logic for EXPECTED."""
        result = TestResult(
            name='AR(1) test',
            statistic=-2.5,
            pvalue=0.01,  # Below 0.10 (expected for AR(1))
            distribution='normal'
        )
        assert result.conclusion == 'EXPECTED'

    def test_str_with_df(self):
        """Test string representation with degrees of freedom."""
        result = TestResult(
            name='Hansen J-test',
            statistic=10.5,
            pvalue=0.15,
            df=5
        )
        s = str(result)
        assert 'Hansen J-test' in s
        assert '10.500' in s
        assert '0.1500' in s
        assert 'df=5' in s

    def test_str_without_df(self):
        """Test string representation without degrees of freedom."""
        result = TestResult(
            name='AR(1) test',
            statistic=-2.5,
            pvalue=0.01,
            df=None
        )
        s = str(result)
        assert 'AR(1) test' in s
        assert '-2.500' in s
        assert '0.0100' in s
        assert 'df' not in s

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TestResult(
            name='Hansen J-test',
            statistic=10.5,
            pvalue=0.15,
            df=5
        )
        d = result.to_dict()

        assert d['name'] == 'Hansen J-test'
        assert d['statistic'] == 10.5
        assert d['pvalue'] == 0.15
        assert d['df'] == 5
        assert d['conclusion'] == 'PASS'


class TestGMMResults:
    """Tests for GMMResults class."""

    @pytest.fixture
    def basic_results(self):
        """Create basic GMMResults for testing."""
        params = pd.Series([0.5, 0.3], index=['L1.y', 'x'])
        std_errors = pd.Series([0.1, 0.05], index=['L1.y', 'x'])
        tvalues = pd.Series([5.0, 6.0], index=['L1.y', 'x'])
        pvalues = pd.Series([0.001, 0.0001], index=['L1.y', 'x'])

        hansen = TestResult('Hansen J-test', 10.0, 0.18, 5)
        sargan = TestResult('Sargan test', 8.0, 0.25, 5)
        ar1 = TestResult('AR(1) test', -2.5, 0.01, None)
        ar2 = TestResult('AR(2) test', 1.2, 0.23, None)

        vcov = np.array([[0.01, 0.001], [0.001, 0.0025]])

        return GMMResults(
            params=params,
            std_errors=std_errors,
            tvalues=tvalues,
            pvalues=pvalues,
            nobs=100,
            n_groups=50,
            n_instruments=20,
            n_params=2,
            hansen_j=hansen,
            sargan=sargan,
            ar1_test=ar1,
            ar2_test=ar2,
            vcov=vcov,
            weight_matrix=np.eye(20),
            converged=True,
            two_step=True,
            windmeijer_corrected=True
        )

    def test_creation(self, basic_results):
        """Test basic GMMResults creation."""
        assert basic_results.nobs == 100
        assert basic_results.n_groups == 50
        assert basic_results.n_instruments == 20
        assert basic_results.n_params == 2
        assert basic_results.converged
        assert basic_results.two_step
        assert basic_results.windmeijer_corrected

    def test_instrument_ratio(self, basic_results):
        """Test instrument ratio property."""
        assert basic_results.instrument_ratio == 20 / 50
        assert basic_results.instrument_ratio == 0.4

    def test_conf_int_default(self, basic_results):
        """Test default 95% confidence intervals."""
        ci = basic_results.conf_int()

        assert 'lower' in ci.columns
        assert 'upper' in ci.columns
        assert len(ci) == 2

        # Check that intervals make sense
        assert ci.loc['L1.y', 'lower'] < 0.5 < ci.loc['L1.y', 'upper']
        assert ci.loc['x', 'lower'] < 0.3 < ci.loc['x', 'upper']

    def test_conf_int_custom_alpha(self, basic_results):
        """Test confidence intervals with custom alpha."""
        ci_95 = basic_results.conf_int(alpha=0.05)
        ci_99 = basic_results.conf_int(alpha=0.01)

        # 99% CI should be wider than 95% CI
        width_95 = ci_95.loc['L1.y', 'upper'] - ci_95.loc['L1.y', 'lower']
        width_99 = ci_99.loc['L1.y', 'upper'] - ci_99.loc['L1.y', 'lower']
        assert width_99 > width_95

    def test_summary(self, basic_results):
        """Test summary generation."""
        summary = basic_results.summary()

        assert isinstance(summary, str)
        assert 'Difference GMM' in summary
        assert 'Number of observations' in summary
        assert 'Number of groups' in summary
        assert 'Number of instruments' in summary
        assert 'L1.y' in summary
        assert 'x' in summary
        assert 'Hansen J-test' in summary
        assert 'AR(1) test' in summary

    def test_summary_custom_title(self, basic_results):
        """Test summary with custom title."""
        summary = basic_results.summary(title='Custom GMM Model')
        assert 'Custom GMM Model' in summary

    def test_to_latex(self, basic_results):
        """Test LaTeX table generation."""
        latex = basic_results.to_latex()

        assert isinstance(latex, str)
        assert r'\begin{table}' in latex
        assert r'\end{table}' in latex
        assert r'\toprule' in latex
        assert r'\bottomrule' in latex
        assert 'L1.y' in latex or r'L1\_y' in latex
        assert '0.5000' in latex or '0.50' in latex

    def test_to_latex_custom_caption(self, basic_results):
        """Test LaTeX with custom caption and label."""
        latex = basic_results.to_latex(
            caption='My Custom Caption',
            label='tab:my_model'
        )

        assert 'My Custom Caption' in latex
        assert 'tab:my_model' in latex

    def test_to_dict(self, basic_results):
        """Test conversion to dictionary."""
        d = basic_results.to_dict()

        assert 'params' in d
        assert 'std_errors' in d
        assert 'pvalues' in d
        assert 'nobs' in d
        assert 'n_groups' in d
        assert 'n_instruments' in d
        assert 'hansen_j' in d
        assert 'instrument_ratio' in d

        assert d['nobs'] == 100
        assert d['n_groups'] == 50
        assert d['instrument_ratio'] == 0.4

    def test_repr(self, basic_results):
        """Test representation string."""
        r = repr(basic_results)

        assert 'GMMResults' in r
        assert 'nobs=100' in r
        assert 'n_groups=50' in r
        assert 'n_instruments=20' in r


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
