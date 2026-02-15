"""
Test summary methods for all advanced models.
"""

import numpy as np
import pandas as pd

from panelbox.gmm.bias_corrected import BiasCorrectedGMM
from panelbox.gmm.cue_gmm import ContinuousUpdatedGMM
from panelbox.models.count.ppml import PPML
from panelbox.models.discrete.multinomial import MultinomialLogit
from panelbox.models.selection.heckman import PanelHeckman

# Set random seed
np.random.seed(42)


def test_gmm_summary():
    """Test CUE-GMM summary."""
    print("\n" + "=" * 60)
    print("Testing CUE-GMM summary()")
    print("=" * 60)

    # Simple test data
    n = 100
    X = np.random.randn(n, 2)
    y = X @ np.array([1.0, 0.5]) + np.random.randn(n) * 0.5

    # Define moment conditions
    def moments(params, X, y):
        resid = y - X @ params
        return X.T @ resid.reshape(-1, 1) / len(y)

    try:
        model = ContinuousUpdatedGMM(
            moment_function=lambda p: moments(p, X, y), n_params=2, n_moments=2
        )
        result = model.fit(initial_params=np.ones(2))

        if hasattr(result, "summary"):
            print(result.summary())
            print("\n✓ CUE-GMM summary() works")
        else:
            print("✗ CUE-GMM result has no summary() method")
    except Exception as e:
        print(f"✗ Error testing CUE-GMM: {e}")


def test_bias_corrected_gmm_summary():
    """Test Bias-Corrected GMM summary."""
    print("\n" + "=" * 60)
    print("Testing Bias-Corrected GMM summary()")
    print("=" * 60)

    try:
        # This would need proper panel data setup
        print("⚠ Bias-Corrected GMM requires full panel setup - skipping for now")
    except Exception as e:
        print(f"✗ Error testing Bias-Corrected GMM: {e}")


def test_heckman_summary():
    """Test Panel Heckman summary."""
    print("\n" + "=" * 60)
    print("Testing Panel Heckman summary()")
    print("=" * 60)

    try:
        # Generate test data
        n = 200
        np.random.seed(42)

        # Selection equation
        z = np.random.randn(n, 2)
        z = np.column_stack([np.ones(n), z])

        # Outcome equation
        x = np.random.randn(n, 2)
        x = np.column_stack([np.ones(n), x])

        # Generate outcomes with selection
        u = np.random.randn(n)
        e = 0.5 * u + np.random.randn(n) * 0.5

        selection_latent = z @ np.array([0.5, 1.0, -0.5]) + u
        selection = (selection_latent > 0).astype(int)

        outcome = x @ np.array([1.0, 2.0, -1.0]) + e
        outcome[selection == 0] = np.nan

        # Fit model
        model = PanelHeckman(
            endog=outcome, exog=x, selection=selection, exog_selection=z, method="two_step"
        )
        result = model.fit()

        if hasattr(result, "summary"):
            print(result.summary())
            print("\n✓ Panel Heckman summary() works")
        else:
            print("✗ Panel Heckman result has no summary() method")
    except Exception as e:
        print(f"✗ Error testing Panel Heckman: {e}")


def test_multinomial_summary():
    """Test Multinomial Logit summary."""
    print("\n" + "=" * 60)
    print("Testing Multinomial Logit summary()")
    print("=" * 60)

    try:
        # Generate test data
        n = 300
        np.random.seed(42)

        X = np.random.randn(n, 2)
        X = np.column_stack([np.ones(n), X])

        # Generate choices (3 categories)
        latent_0 = 0  # Base category
        latent_1 = X @ np.array([0.5, 1.0, -0.5]) + np.random.randn(n)
        latent_2 = X @ np.array([-0.5, -1.0, 1.0]) + np.random.randn(n)

        latents = np.column_stack([np.zeros(n), latent_1, latent_2])
        choice = np.argmax(latents, axis=1)

        # Fit model
        model = MultinomialLogit(endog=choice, exog=X, n_choices=3)
        result = model.fit()

        if hasattr(result, "summary"):
            print(result.summary())
            print("\n✓ Multinomial Logit summary() works")
        else:
            print("✗ Multinomial Logit result has no summary() method")
    except Exception as e:
        print(f"✗ Error testing Multinomial Logit: {e}")


def test_ppml_summary():
    """Test PPML summary."""
    print("\n" + "=" * 60)
    print("Testing PPML summary()")
    print("=" * 60)

    try:
        # Generate test data
        n = 200
        np.random.seed(42)

        X = np.random.randn(n, 2)
        X = np.column_stack([np.ones(n), X])

        # Generate count data
        linear_pred = X @ np.array([0.5, 1.0, -0.5])
        y = np.random.poisson(np.exp(linear_pred))

        # Create entity and time indices
        entity = np.repeat(np.arange(50), 4)
        time = np.tile(np.arange(4), 50)

        # Fit model
        model = PPML(endog=y, exog=X, entity=entity, time=time)
        result = model.fit()

        if hasattr(result, "summary"):
            print(result.summary())
            print("\n✓ PPML summary() works")
        else:
            print("✗ PPML result has no summary() method")
    except Exception as e:
        print(f"✗ Error testing PPML: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING SUMMARY METHODS FOR ADVANCED MODELS")
    print("=" * 60)

    test_gmm_summary()
    test_bias_corrected_gmm_summary()
    test_heckman_summary()
    test_multinomial_summary()
    test_ppml_summary()

    print("\n" + "=" * 60)
    print("SUMMARY TESTING COMPLETE")
    print("=" * 60)
