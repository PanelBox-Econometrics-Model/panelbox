"""Tests for standard errors utility functions."""

import numpy as np
import pytest

from panelbox.standard_errors.utils import (
    clustered_covariance,
    compute_bread,
    compute_clustered_meat,
    compute_leverage,
    compute_meat_hc,
    compute_twoway_clustered_meat,
    hc_covariance,
    sandwich_covariance,
    twoway_clustered_covariance,
)


class TestComputeLeverage:
    """Test compute_leverage function."""

    def test_leverage_basic(self):
        """Test basic leverage computation."""
        X = np.array([[1, 2], [1, 3], [1, 4]])
        leverage = compute_leverage(X)

        assert leverage.shape == (3,)
        assert np.all(leverage >= 0)
        assert np.all(leverage <= 1)

    def test_leverage_sum(self):
        """Test leverage sum equals number of parameters."""
        X = np.random.randn(20, 3)
        leverage = compute_leverage(X)

        # Sum of leverage should equal k
        assert np.isclose(leverage.sum(), 3, rtol=0.01)


class TestComputeBread:
    """Test compute_bread function."""

    def test_bread_basic(self):
        """Test basic bread computation."""
        X = np.array([[1, 2], [1, 3], [1, 4]])
        bread = compute_bread(X)

        assert bread.shape == (2, 2)
        assert np.allclose(bread, bread.T)  # Should be symmetric


class TestComputeMeatHC:
    """Test compute_meat_hc function."""

    def test_meat_hc0(self):
        """Test HC0 meat computation."""
        X = np.random.randn(10, 3)
        resid = np.random.randn(10)
        meat = compute_meat_hc(X, resid, method="HC0")

        assert meat.shape == (3, 3)
        assert np.allclose(meat, meat.T)

    def test_meat_hc1(self):
        """Test HC1 meat computation."""
        X = np.random.randn(10, 3)
        resid = np.random.randn(10)
        meat = compute_meat_hc(X, resid, method="HC1")

        assert meat.shape == (3, 3)
        assert np.allclose(meat, meat.T)

    def test_meat_hc2(self):
        """Test HC2 meat computation."""
        X = np.random.randn(10, 3)
        resid = np.random.randn(10)
        meat = compute_meat_hc(X, resid, method="HC2")

        assert meat.shape == (3, 3)
        assert np.allclose(meat, meat.T)

    def test_meat_hc3(self):
        """Test HC3 meat computation."""
        X = np.random.randn(10, 3)
        resid = np.random.randn(10)
        meat = compute_meat_hc(X, resid, method="HC3")

        assert meat.shape == (3, 3)
        assert np.allclose(meat, meat.T)

    def test_meat_invalid_method(self):
        """Test invalid HC method raises error."""
        X = np.random.randn(10, 3)
        resid = np.random.randn(10)

        with pytest.raises(ValueError, match="Unknown HC method"):
            compute_meat_hc(X, resid, method="INVALID")


class TestSandwichCovariance:
    """Test sandwich_covariance function."""

    def test_sandwich_basic(self):
        """Test basic sandwich computation."""
        bread = np.eye(3)
        meat = np.eye(3)
        cov = sandwich_covariance(bread, meat)

        assert cov.shape == (3, 3)
        assert np.allclose(cov, np.eye(3))


class TestComputeClusteredMeat:
    """Test compute_clustered_meat function."""

    def test_clustered_meat_basic(self):
        """Test basic clustered meat computation."""
        X = np.random.randn(20, 3)
        resid = np.random.randn(20)
        clusters = np.repeat([1, 2, 3, 4], 5)

        meat = compute_clustered_meat(X, resid, clusters)

        assert meat.shape == (3, 3)
        assert np.allclose(meat, meat.T)

    def test_clustered_meat_no_correction(self):
        """Test clustered meat without correction."""
        X = np.random.randn(20, 3)
        resid = np.random.randn(20)
        clusters = np.repeat([1, 2], 10)

        meat = compute_clustered_meat(X, resid, clusters, df_correction=False)

        assert meat.shape == (3, 3)


class TestComputeTwowayClusteredMeat:
    """Test compute_twoway_clustered_meat function."""

    def test_twoway_clustered_meat(self):
        """Test two-way clustered meat computation."""
        X = np.random.randn(20, 3)
        resid = np.random.randn(20)
        clusters1 = np.repeat([1, 2], 10)
        clusters2 = np.tile([1, 2, 3, 4], 5)

        meat = compute_twoway_clustered_meat(X, resid, clusters1, clusters2)

        assert meat.shape == (3, 3)


class TestHCCovariance:
    """Test hc_covariance function."""

    def test_hc_covariance_hc1(self):
        """Test HC1 covariance."""
        X = np.random.randn(20, 3)
        resid = np.random.randn(20)

        cov = hc_covariance(X, resid, method="HC1")

        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)


class TestClusteredCovariance:
    """Test clustered_covariance function."""

    def test_clustered_covariance_basic(self):
        """Test basic clustered covariance."""
        X = np.random.randn(20, 3)
        resid = np.random.randn(20)
        clusters = np.repeat([1, 2, 3, 4], 5)

        cov = clustered_covariance(X, resid, clusters)

        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)


class TestTwowayClusteredCovariance:
    """Test twoway_clustered_covariance function."""

    def test_twoway_clustered_covariance_basic(self):
        """Test basic two-way clustered covariance."""
        X = np.random.randn(20, 3)
        resid = np.random.randn(20)
        clusters1 = np.repeat([1, 2], 10)
        clusters2 = np.tile([1, 2, 3, 4], 5)

        cov = twoway_clustered_covariance(X, resid, clusters1, clusters2)

        assert cov.shape == (3, 3)
        assert np.allclose(cov, cov.T)
