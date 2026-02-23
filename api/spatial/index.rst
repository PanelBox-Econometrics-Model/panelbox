Spatial Econometrics API Reference
===================================

.. currentmodule:: panelbox

This section provides complete API documentation for the spatial econometrics module in PanelBox.

Spatial Panel Models
--------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   models.spatial.SpatialLag
   models.spatial.SpatialError
   models.spatial.SpatialDurbin
   models.spatial.GeneralNestingSpatial
   models.spatial.SpatialPanelModel

Spatial Weight Matrices
------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   core.spatial_weights.SpatialWeights

Spatial Diagnostics
-------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   validation.spatial.MoranIPanelTest
   validation.spatial.LocalMoranI
   validation.spatial.LMLagTest
   validation.spatial.LMErrorTest
   validation.spatial.RobustLMLagTest
   validation.spatial.RobustLMErrorTest

.. autosummary::
   :toctree: generated/
   :template: function.rst

   validation.spatial.run_lm_tests

Spatial Effects
---------------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   effects.spatial_effects.compute_spatial_effects
   effects.spatial_effects.plot_spatial_effects

Spatial HAC Standard Errors
----------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   standard_errors.spatial_hac.SpatialHAC

Performance Optimization
------------------------

.. autosummary::
   :toctree: generated/
   :template: module.rst

   optimization.spatial_optimizations
   optimization.parallel_inference

Visualization
-------------

.. autosummary::
   :toctree: generated/
   :template: module.rst

   visualization.spatial_plots

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import panelbox as pb
   import numpy as np

   # Create spatial weight matrix
   W = pb.SpatialWeights.from_contiguity(gdf, method='queen')

   # Estimate spatial lag model
   model = pb.SpatialLag(
       formula='y ~ x1 + x2',
       data=panel_data,
       entity_col='county',
       time_col='year',
       W=W
   )

   result = model.fit(effects='fixed')
   print(result.summary())

Spatial Diagnostics
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test for spatial autocorrelation
   moran_test = pb.MoranIPanelTest(
       residuals, W, entity_ids, time_ids
   )
   moran_result = moran_test.run()

   # LM tests for model selection
   lm_results = pb.run_lm_tests(ols_result, W)
   print(f"Recommended model: {lm_results['recommendation']}")

Effects Decomposition
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For SDM model
   sdm = pb.SpatialDurbin(formula, data, entity_col, time_col, W)
   sdm_result = sdm.fit()

   # Decompose effects
   effects = pb.compute_spatial_effects(sdm_result)
   print(effects.summary())

   # Visualize
   pb.plot_spatial_effects(effects)
