# vulture_whitelist.py
# False positives for Vulture — items used via reflection, frameworks, or public API

# ── TYPE_CHECKING imports (used in quoted type annotations) ──────────────
# These are imported under `if TYPE_CHECKING:` and used in string annotations
# like Optional["gpd.GeoDataFrame"]. Vulture can't trace these.
import geopandas as gpd  # noqa: used in TYPE_CHECKING for "gpd.GeoDataFrame" annotations

gpd  # ensure vulture sees usage of the gpd name

# ── Context manager protocol ─────────────────────────────────────────────
_.exc_type  # __exit__ protocol parameter
_.exc_val  # __exit__ protocol parameter
_.exc_tb  # __exit__ protocol parameter

# ── Public API parameters (accepted but not yet implemented) ─────────────
_.entity_highlight  # plot_comprehensive_summary: planned feature
_.include_series  # create_unit_root_test_plot: planned feature
_.data_type  # suggest_chart: planned feature
_.stratify  # subset_sensitivity: planned feature

# ── Stub function parameters (not yet implemented) ──────────────────────
_.estimator_func  # bootstrap_two_step_variance: raises NotImplementedError

# ── Method parameters kept for API compatibility ─────────────────────────
_.residuals_full  # difference_hansen_test: parameter in public API signature
