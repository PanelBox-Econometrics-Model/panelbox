"""Version information for panelbox."""

__version__ = "0.4.3"
__author__ = "Gustavo Haase, Paulo Dourado"
__email__ = "gustavo.haase@gmail.com"
__license__ = "MIT"

# Version history
# 0.4.0 (2026-02-05): Robust Standard Errors
#                     - HC0-HC3: Heteroskedasticity-robust standard errors (White 1980, MacKinnon-White 1985)
#                     - Clustered SE: One-way and two-way clustering (Cameron-Gelbach-Miller 2011)
#                     - Driscoll-Kraay: Spatial and temporal dependence (Driscoll & Kraay 1998)
#                     - Newey-West HAC: Heteroskedasticity and autocorrelation consistent (Newey & West 1987)
#                     - PCSE: Panel-corrected standard errors (Beck & Katz 1995)
#                     - 75+ tests, ~90% coverage, integrated with FE and RE models
# 0.3.0 (2026-01-22): Advanced Robustness Analysis
#                     - PanelBootstrap: 4 bootstrap methods (pairs, wild, block, residual)
#                     - SensitivityAnalysis: 3 methods (LOO entities, LOO periods, subset)
#                     - 63 new tests, comprehensive documentation
#                     - Optional matplotlib visualization
# 0.2.0 (2026-01-21): GMM implementation complete (Difference & System GMM)
#                     - Arellano-Bond (1991) Difference GMM
#                     - Blundell-Bond (1998) System GMM
#                     - Robust to unbalanced panels
#                     - Comprehensive documentation
# 0.1.0 (Initial): Core panel data models (FE, RE, OLS)
