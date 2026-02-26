"""
data — Synthetic panel datasets for the validation tutorial series.

All CSV files in this directory are generated programmatically by
``utils/data_generators.py``.  To regenerate::

    python utils/data_generators.py

Available datasets
------------------
firmdata.csv              100 firms × 10 years; groupwise heteroskedasticity
macro_panel.csv           30 countries × 20 years; CD + AR(1) errors
small_panel.csv           20 entities × 10 periods; i.i.d. errors
sales_panel.csv           50 firms × 24 quarters; seasonal component
macro_ts_panel.csv        15 countries × 40 years; structural break 2008
panel_with_outliers.csv   80 firms × 8 years; ~5% injected outliers
real_firms.csv            120 firms × 5 years; natural heterogeneity
panel_comprehensive.csv   100 entities × 12 periods; rich variable set
panel_unbalanced.csv      150 entities, up to 10 periods; random attrition
"""
