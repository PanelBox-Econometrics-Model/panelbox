"""Version information for panelbox."""

from __future__ import annotations

__version__ = "0.6.1"
__author__ = "Gustavo Haase, Paulo Dourado"
__email__ = "gustavo.haase@gmail.com"
__license__ = "MIT"

# Version history
# 0.6.1 (2026-02-25): Dataset Bundling for Colab Compatibility
#                     - Bundle 103 CSV datasets in the library package
#                     - New load_dataset() API with category-based organization
#                     - 80+ notebooks migrated from relative paths to load_dataset()
#                     - Notebooks now work in Google Colab with pip install panelbox
#                     - Add spatial validation tests and frontier tests
# 0.6.0 (2026-02-25): Complete Econometric Toolkit
#                     - 70+ econometric models across 13 families
#                     - 50+ diagnostic tests with validation against R/Stata
#                     - CUE-GMM, Bias-Corrected GMM, Panel Heckman
#                     - Cointegration tests (Westerlund, Pedroni, Kao)
#                     - Unit root tests (Hadri, Breitung, LLC, IPS, Fisher)
#                     - PPML, Multinomial Logit, Ordered Logit/Probit
#                     - Panel VAR, VECM, IRF, FEVD, Granger Causality
#                     - Stochastic Frontier Analysis (4-component, TFP)
#                     - Spatial models (SAR, SEM, SDM, GNS, Dynamic)
#                     - Quantile regression (FE, Canay, Location-Scale, Dynamic)
#                     - 35+ Plotly charts, HTML/LaTeX/Markdown reports
#                     - Master Report system with experiment overview
#                     - Complete documentation overhaul (200+ pages)
#                     - Google Colab tutorials for all model families
#                     - 3986 tests passing, 85-92% coverage
#                     - ValidationTest: Configurable test runner (quick, basic, full) for model validation
#                     - ComparisonTest: Test runner for multi-model comparison
#                     - Master Report: Comprehensive HTML report with experiment overview and sub-report navigation
#                     - Enhanced PanelExperiment with save_master_report() method
#                     - Full workflow integration: validation → comparison → residuals → master report
#                     - 19 new tests (9 ValidationTest + 10 ComparisonTest)
#                     - 4 integration tests for complete end-to-end workflow
#                     - Clean API for report generation and model testing
#                     - Professional master report template with responsive design
# 0.7.0 (2026-02-08): Advanced Features & Production Polish (Sprint 5)
#                     - ResidualResult: Container for residual diagnostics with 4 tests
#                       (Shapiro-Wilk, Jarque-Bera, Durbin-Watson, Ljung-Box)
#                     - Fixed chart registration system (all 35 charts now registered)
#                     - analyze_residuals() method in PanelExperiment
#                     - Professional summary() output for residual diagnostics
#                     - HTML reports now include embedded interactive charts (102.9 KB vs 77.5 KB)
#                     - Complete result container trilogy: Validation, Comparison, Residual
#                     - 16 comprehensive tests for ResidualResult
#                     - Zero warnings in console output
#                     - Production-ready package
# 0.6.0 (2026-02-08): Experiment Pattern & Result Containers
#                     - PanelExperiment: Factory-based model management with automatic storage
#                     - ValidationResult: Container for validation test results with HTML/JSON export
#                     - ComparisonResult: Container for model comparison with best model selection
#                     - One-liner workflows: validate_model(), compare_models(), fit_all_models()
#                     - Professional HTML reports with embedded Plotly charts
#                     - BaseResult abstract class for extensible result containers
#                     - Complete public API integration
#                     - 20+ tests for Experiment Pattern components
#                     - Comprehensive documentation and working examples
# 0.5.0 (2026-02-08): Comprehensive Visualization System
#                     - 28+ interactive Plotly charts for panel data analysis
#                     - Phase 6: Panel-specific visualizations (entity/time effects, between-within, structure)
#                     - Phase 7: Econometric test visualizations (ACF/PACF, unit root, cointegration, CD)
#                     - 3 professional themes (Professional, Academic, Presentation)
#                     - Registry/Factory pattern for extensibility
#                     - HTML report generation with interactive charts
#                     - Complete export system (HTML, JSON, PNG, SVG, PDF)
#                     - High-level convenience APIs for common use cases
#                     - 90+ tests, comprehensive documentation
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
