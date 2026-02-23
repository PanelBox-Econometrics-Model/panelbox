Stochastic Frontier Analysis (SFA)
=====================================

Overview
--------

The ``panelbox.frontier`` module provides tools for estimating production and cost frontiers using maximum likelihood estimation with various distributional assumptions for the inefficiency term.

Main Features
~~~~~~~~~~~~~

- **Multiple Distributional Assumptions**: Half-normal, exponential, truncated normal, and gamma distributions
- **Production and Cost Frontiers**: Flexible specification of frontier type
- **Efficiency Estimation**: JLMS, Battese-Coelli, and modal estimators
- **Cross-Section and Panel Data**: Support for both cross-sectional and panel models
- **Comprehensive Diagnostics**: Convergence checks, standard errors, and model comparison tools

Model Structure
~~~~~~~~~~~~~~~

The basic stochastic frontier model is:

**Production Frontier:**

.. math::

    \ln(y_i) = x_i'\beta + v_i - u_i

**Cost Frontier:**

.. math::

    \ln(y_i) = x_i'\beta + v_i + u_i

where:

- :math:`v_i \sim N(0, \sigma^2_v)` is random noise
- :math:`u_i \geq 0` is inefficiency with specified distribution
- Technical efficiency: :math:`TE_i = \exp(-u_i) \in (0,1]`
- Cost efficiency: :math:`CE_i = \exp(u_i) \in [1,\infty)`

Classes
-------

StochasticFrontier
~~~~~~~~~~~~~~~~~~

.. autoclass:: panelbox.frontier.StochasticFrontier
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Example

   .. code-block:: python

      from panelbox.frontier import StochasticFrontier
      import pandas as pd

      # Load data
      data = pd.read_csv('production_data.csv')

      # Specify model
      sf = StochasticFrontier(
          data=data,
          depvar='log_output',
          exog=['log_labor', 'log_capital'],
          frontier='production',
          dist='half_normal'
      )

      # Estimate via MLE
      result = sf.fit(method='mle')

      # Display results
      print(result.summary())

      # Get efficiency estimates
      efficiency = result.efficiency(estimator='bc')

SFResult
~~~~~~~~

.. autoclass:: panelbox.frontier.SFResult
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Attributes

   .. attribute:: params
      :type: pandas.Series

      Estimated parameters (β, σ²_v, σ²_u, ...)

   .. attribute:: se
      :type: pandas.Series

      Standard errors

   .. attribute:: tvalues
      :type: pandas.Series

      t-statistics

   .. attribute:: pvalues
      :type: pandas.Series

      p-values

   .. attribute:: loglik
      :type: float

      Log-likelihood value

   .. attribute:: aic
      :type: float

      Akaike Information Criterion

   .. attribute:: bic
      :type: float

      Bayesian Information Criterion

   .. attribute:: sigma_v
      :type: float

      Noise standard deviation

   .. attribute:: sigma_u
      :type: float

      Inefficiency standard deviation

   .. attribute:: lambda_param
      :type: float

      λ = σ_u / σ_v

   .. attribute:: gamma
      :type: float

      γ = σ²_u / (σ²_v + σ²_u)

   .. attribute:: converged
      :type: bool

      Whether optimization converged

Enumerations
------------

FrontierType
~~~~~~~~~~~~

.. autoclass:: panelbox.frontier.FrontierType
   :members:
   :undoc-members:

   Enumeration for frontier type specification.

   .. attribute:: PRODUCTION

      Production frontier where inefficiency reduces output

   .. attribute:: COST

      Cost frontier where inefficiency increases cost

DistributionType
~~~~~~~~~~~~~~~~

.. autoclass:: panelbox.frontier.DistributionType
   :members:
   :undoc-members:

   Enumeration for inefficiency distribution.

   .. attribute:: HALF_NORMAL

      Half-normal distribution (Aigner et al. 1977)

   .. attribute:: EXPONENTIAL

      Exponential distribution (Meeusen & van den Broeck 1977)

   .. attribute:: TRUNCATED_NORMAL

      Truncated normal with location parameter μ

   .. attribute:: GAMMA

      Gamma distribution (Greene 1990)

ModelType
~~~~~~~~~

.. autoclass:: panelbox.frontier.ModelType
   :members:
   :undoc-members:

   Enumeration for panel model structure.

   .. attribute:: CROSS_SECTION

      Cross-sectional model (no panel structure)

   .. attribute:: POOLED

      Pooled panel model

   .. attribute:: PITT_LEE

      Pitt & Lee (1981) time-invariant inefficiency

   .. attribute:: BATTESE_COELLI_92

      Battese & Coelli (1992) time-varying model

   .. attribute:: BATTESE_COELLI_95

      Battese & Coelli (1995) with heterogeneity

Functions
---------

Data Validation
~~~~~~~~~~~~~~~

.. autofunction:: panelbox.frontier.validate_frontier_data

.. autofunction:: panelbox.frontier.prepare_panel_index

Examples
--------

Basic Production Frontier
~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimate a cross-sectional production frontier with half-normal inefficiency:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from panelbox.frontier import StochasticFrontier

   # Simulate Cobb-Douglas production data
   np.random.seed(42)
   n = 500

   log_labor = np.random.uniform(0, 3, n)
   log_capital = np.random.uniform(0, 3, n)

   # True parameters
   beta_0, beta_1, beta_2 = 2.0, 0.6, 0.3
   sigma_v, sigma_u = 0.1, 0.2

   # Generate errors
   v = np.random.normal(0, sigma_v, n)
   u = np.abs(np.random.normal(0, sigma_u, n))

   # Output
   log_output = beta_0 + beta_1*log_labor + beta_2*log_capital + v - u

   data = pd.DataFrame({
       'log_output': log_output,
       'log_labor': log_labor,
       'log_capital': log_capital
   })

   # Estimate
   sf = StochasticFrontier(
       data=data,
       depvar='log_output',
       exog=['log_labor', 'log_capital'],
       frontier='production',
       dist='half_normal'
   )

   result = sf.fit()
   print(result.summary())

Cost Frontier
~~~~~~~~~~~~~

Estimate a cost frontier with exponential distribution:

.. code-block:: python

   from panelbox.frontier import StochasticFrontier

   # Assume 'cost_data' contains log_cost, log_labor, log_capital
   sf = StochasticFrontier(
       data=cost_data,
       depvar='log_cost',
       exog=['log_labor', 'log_capital'],
       frontier='cost',
       dist='exponential'
   )

   result = sf.fit(method='mle')

   # Cost efficiency (CE > 1 means inefficient)
   efficiency = result.efficiency(estimator='bc')
   print(f"Mean cost efficiency: {efficiency['efficiency'].mean():.4f}")

Comparing Distributions
~~~~~~~~~~~~~~~~~~~~~~~~

Compare multiple distributional assumptions:

.. code-block:: python

   from panelbox.frontier import StochasticFrontier

   # Estimate with different distributions
   distributions = ['half_normal', 'exponential', 'truncated_normal']
   results = {}

   for dist in distributions:
       sf = StochasticFrontier(
           data=data,
           depvar='log_output',
           exog=['log_labor', 'log_capital'],
           frontier='production',
           dist=dist
       )
       results[dist] = sf.fit(method='mle', verbose=False)

   # Compare models
   comparison = results['half_normal'].compare_distributions(
       [results['exponential'], results['truncated_normal']]
   )
   print(comparison)

Efficiency Analysis
~~~~~~~~~~~~~~~~~~~

Extract and analyze efficiency estimates:

.. code-block:: python

   # Get efficiency with confidence intervals
   eff = result.efficiency(estimator='bc', ci_level=0.95)

   # Summary statistics
   print(eff[['efficiency', 'ci_lower', 'ci_upper']].describe())

   # Most efficient units
   print("\nTop 10 most efficient:")
   print(eff.nlargest(10, 'efficiency'))

   # Visualize
   import matplotlib.pyplot as plt

   plt.hist(eff['efficiency'], bins=30, alpha=0.7, edgecolor='black')
   plt.xlabel('Technical Efficiency')
   plt.ylabel('Frequency')
   plt.title(f'Efficiency Distribution (Mean: {eff["efficiency"].mean():.3f})')
   plt.show()

References
----------

**Foundational Papers:**

- Aigner, D., Lovell, C. K., & Schmidt, P. (1977). Formulation and estimation of stochastic frontier production function models. *Journal of Econometrics*, 6(1), 21-37.

- Meeusen, W., & van Den Broeck, J. (1977). Efficiency estimation from Cobb-Douglas production functions with composed error. *International Economic Review*, 435-444.

**Efficiency Estimation:**

- Jondrow, J., Lovell, C. K., Materov, I. S., & Schmidt, P. (1982). On the estimation of technical inefficiency in the stochastic frontier production function model. *Journal of Econometrics*, 19(2-3), 233-238.

- Battese, G. E., & Coelli, T. J. (1988). Prediction of firm-level technical efficiencies with a generalized frontier production function and panel data. *Journal of Econometrics*, 38(3), 387-399.

**Panel Data Models:**

- Pitt, M. M., & Lee, L. F. (1981). The measurement and sources of technical inefficiency in the Indonesian weaving industry. *Journal of Development Economics*, 9(1), 43-64.

- Battese, G. E., & Coelli, T. J. (1992). Frontier production functions, technical efficiency and panel data: with application to paddy farmers in India. *Journal of Productivity Analysis*, 3(1), 153-169.

- Battese, G. E., & Coelli, T. J. (1995). A model for technical inefficiency effects in a stochastic frontier production function for panel data. *Empirical Economics*, 20(2), 325-332.

- Cornwell, C., Schmidt, P., & Sickles, R. C. (1990). Production frontiers with cross-sectional and time-series variation in efficiency levels. *Journal of Econometrics*, 46(1-2), 185-200.

- Greene, W. (2005). Reconsidering heterogeneity in panel data estimators of the stochastic frontier model. *Journal of Econometrics*, 126(2), 269-303.

**Distributional Assumptions:**

- Stevenson, R. E. (1980). Likelihood functions for generalized stochastic frontier estimation. *Journal of Econometrics*, 13(1), 57-66.

- Greene, W. H. (1990). A gamma-distributed stochastic frontier model. *Journal of Econometrics*, 46(1-2), 141-163.

**Confidence Intervals:**

- Horrace, W. C., & Schmidt, P. (1996). Confidence statements for efficiency estimates from stochastic frontier models. *Journal of Productivity Analysis*, 7(2), 257-282.

See Also
--------

- :doc:`/examples/sfa_basic_usage` - Detailed usage examples
- :doc:`/tutorials/sfa_introduction` - Tutorial introduction to SFA
- :doc:`/theory/stochastic_frontiers` - Theoretical background
