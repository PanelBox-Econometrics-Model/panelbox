# Dataset Descriptions for SFA Tutorial Series

All datasets are synthetically generated using `utils/data_generation.py` with `seed=42` for reproducibility.

## Overview

| Dataset | Type | Observations | Variables | Used in |
|---------|------|-------------|-----------|---------|
| `hospital_data.csv` | Cross-section | 200 | 9 | Notebook 01 |
| `farm_data.csv` | Cross-section | 300 | 8 | Notebook 01 |
| `bank_panel.csv` | Panel (50×15) | 750 | 9 | Notebook 02 |
| `airline_panel.csv` | Panel (25×20) | 500 | 7 | Notebook 02 |
| `manufacturing_panel.csv` | Panel (100×10) | 1,000 | 8 | Notebook 03 |
| `electricity_panel.csv` | Panel (60×12) | 720 | 7 | Notebook 03 |
| `hospital_panel.csv` | Panel (80×10) | 800 | 10 | Notebook 04 |
| `school_panel.csv` | Panel (100×8) | 800 | 10 | Notebook 04 |
| `dairy_farm.csv` | Cross-section | 500 | 9 | Notebook 05 |
| `telecom_panel.csv` | Panel (40×15) | 600 | 8 | Notebook 05 |
| `brazilian_firms.csv` | Panel (500×10) | 5,000 | 12 | Notebook 06 |

---

## 1. `hospital_data.csv` — Cross-Section Hospital Efficiency

**DGP**: Cobb-Douglas production frontier with half-normal inefficiency.

| Variable | Type | Description |
|----------|------|-------------|
| `hospital_id` | int | Hospital identifier (1-200) |
| `log_output` | float | Log of output (patient-days) |
| `log_labor` | float | Log of FTE staff |
| `log_capital` | float | Log of capital value |
| `log_supplies` | float | Log of medical supplies expenditure |
| `teaching` | int | Teaching hospital dummy (0/1) |
| `urban` | int | Urban location dummy (0/1) |
| `ownership` | str | Ownership type: public, private, nonprofit |
| `beds` | int | Number of beds (original scale) |

**Frontier**: log_output = 2.0 + 0.45·log_labor + 0.30·log_capital + 0.20·log_supplies + 0.15·teaching + v - u
- v ~ N(0, 0.04), u ~ |N(0, 0.09)|
- Teaching hospitals: ~5% efficiency bonus; Private hospitals: slightly more efficient

---

## 2. `farm_data.csv` — Cross-Section Agricultural Production

**DGP**: Cobb-Douglas with exponential inefficiency.

| Variable | Type | Description |
|----------|------|-------------|
| `farm_id` | int | Farm identifier (1-300) |
| `log_output` | float | Log of crop output (tons) |
| `log_land` | float | Log of cultivated area (hectares) |
| `log_labor` | float | Log of labor hours |
| `log_fertilizer` | float | Log of fertilizer expenditure |
| `irrigation` | int | Irrigation access (0/1) |
| `region` | str | Geographic region (North/South/East/West) |
| `farm_size` | str | Size category (small/medium/large) |

**Frontier**: Cobb-Douglas with region effects on technology level.
- u ~ Exp(0.25); Irrigation reduces inefficiency by 15%

---

## 3. `bank_panel.csv` — Panel Banking Efficiency

**DGP**: Cobb-Douglas with BC92 time-varying inefficiency.

| Variable | Type | Description |
|----------|------|-------------|
| `bank_id` | int | Bank identifier (1-50) |
| `year` | int | Year (2005-2019) |
| `log_output` | float | Log of total loans + investments |
| `log_labor` | float | Log of employees |
| `log_capital` | float | Log of fixed assets |
| `log_deposits` | float | Log of total deposits |
| `ownership` | str | public, private_domestic, foreign |
| `size_category` | str | small, medium, large |
| `region` | str | Region (1-5) |

**Frontier**: u_it = u_i · exp(-0.03·(t - T))
- Foreign banks more efficient; Large banks have scale advantages

---

## 4. `airline_panel.csv` — Panel Airline Efficiency

**DGP**: Cobb-Douglas with Kumbhakar (1990) time pattern.

| Variable | Type | Description |
|----------|------|-------------|
| `airline_id` | int | Airline identifier (1-25) |
| `year` | int | Year (2000-2019) |
| `log_output` | float | Log of revenue passenger kilometers |
| `log_labor` | float | Log of employees |
| `log_fuel` | float | Log of fuel consumption |
| `log_fleet` | float | Log of aircraft count |
| `carrier_type` | str | legacy, low_cost, regional |

**Frontier**: Low-cost carriers more efficient. Efficiency improvement after 2010.

---

## 5. `manufacturing_panel.csv` — Panel Manufacturing TFP

**DGP**: Four-component model (Kumbhakar et al. 2014).

| Variable | Type | Description |
|----------|------|-------------|
| `firm_id` | int | Firm identifier (1-100) |
| `year` | int | Year (2010-2019) |
| `log_output` | float | Log of value added |
| `log_labor` | float | Log of employees |
| `log_capital` | float | Log of capital stock |
| `log_materials` | float | Log of intermediate inputs |
| `sector` | str | Manufacturing subsector (5 sectors) |
| `exporter` | int | Export activity (0/1) |

**Structure**: y_it = α + β'x + μ_i - η_i + v_it - u_it
- μ_i ~ N(0, 0.15), η_i ~ |N(0, 0.08)|, v_it ~ N(0, 0.03), u_it ~ |N(0, 0.05)|
- Exporters have lower persistent inefficiency

---

## 6. `electricity_panel.csv` — Panel Electricity Generation

**DGP**: Four-component model with fuel-type heterogeneity.

| Variable | Type | Description |
|----------|------|-------------|
| `generator_id` | int | Generator identifier (1-60) |
| `year` | int | Year (2008-2019) |
| `log_output` | float | Log of MWh generated |
| `log_labor` | float | Log of employees |
| `log_capital` | float | Log of installed capacity |
| `log_fuel` | float | Log of fuel input |
| `fuel_type` | str | coal, gas, hydro, nuclear |

**Structure**: Gas/hydro more efficient overall. Nuclear: high persistent, low transient.

---

## 7. `hospital_panel.csv` — Panel Hospital with Determinants

**DGP**: BC95 with determinants affecting inefficiency mean.

| Variable | Type | Description |
|----------|------|-------------|
| `hospital_id` | int | Hospital identifier (1-80) |
| `year` | int | Year (2010-2019) |
| `log_output` | float | Log of patient discharges |
| `log_labor` | float | Log of FTE staff |
| `log_capital` | float | Log of beds |
| `log_supplies` | float | Log of supplies |
| `teaching` | int | Teaching hospital (0/1) |
| `accreditation` | int | Quality accreditation (0/1) |
| `occupancy_rate` | float | Bed occupancy rate |
| `avg_stay` | float | Average length of stay (days) |

**Structure**: μ_it = 0.5 - 0.3·teaching - 0.25·accreditation - 0.4·occupancy_rate

---

## 8. `school_panel.csv` — Panel School Efficiency

**DGP**: Wang (2002) with location AND scale effects.

| Variable | Type | Description |
|----------|------|-------------|
| `school_id` | int | School identifier (1-100) |
| `year` | int | Year (2012-2019) |
| `log_output` | float | Log of test scores (composite) |
| `log_teachers` | float | Log of number of teachers |
| `log_budget` | float | Log of per-pupil budget |
| `log_facilities` | float | Log of facilities index |
| `teacher_experience` | float | Average teacher experience (years) |
| `class_size` | float | Average class size |
| `ses_index` | float | Socioeconomic status index |
| `school_type` | str | public, private, charter |

**Structure**: Location: μ_it = f(teacher_experience, class_size, ses_index); Scale: log(σ²_u) = g(school_type, budget)

---

## 9. `dairy_farm.csv` — Cross-Section Dairy Farm

**DGP**: Translog production function with truncated normal inefficiency.

| Variable | Type | Description |
|----------|------|-------------|
| `farm_id` | int | Farm identifier (1-500) |
| `log_milk` | float | Log of milk output (liters/year) |
| `log_cows` | float | Log of number of cows |
| `log_feed` | float | Log of feed input |
| `log_land` | float | Log of grazing area |
| `log_labor` | float | Log of labor hours |
| `organic` | int | Organic farm (0/1) |
| `breed` | str | holstein, jersey, mixed |
| `cooperative` | int | Member of cooperative (0/1) |

**Structure**: Translog with squared and interaction terms. Cooperative membership reduces inefficiency.

---

## 10. `telecom_panel.csv` — Panel Telecommunications

**DGP**: CSS model (distribution-free) with technology transitions.

| Variable | Type | Description |
|----------|------|-------------|
| `firm_id` | int | Telecom firm identifier (1-40) |
| `year` | int | Year (2005-2019) |
| `log_output` | float | Log of subscribers × revenue |
| `log_labor` | float | Log of employees |
| `log_capital` | float | Log of network capital |
| `log_spectrum` | float | Log of spectrum licenses |
| `technology` | str | 2G, 3G, 4G (time-varying) |
| `market_share` | float | Market share |

**Structure**: Technology transitions (2G→3G→4G) create structural breaks in efficiency.

---

## 11. `brazilian_firms.csv` — Case Study: Brazilian Manufacturing

**DGP**: Complex multi-component model for capstone analysis.

| Variable | Type | Description |
|----------|------|-------------|
| `firm_id` | int | Firm identifier (1-500) |
| `year` | int | Year (2010-2019) |
| `log_output` | float | Log of value added (R$) |
| `log_labor` | float | Log of employees |
| `log_capital` | float | Log of capital stock (R$) |
| `log_materials` | float | Log of raw materials (R$) |
| `sector` | str | Sector (8 sectors) |
| `region` | str | Brazilian region (N, NE, SE, S, CO) |
| `exporter` | int | Export activity (0/1) |
| `foreign_owned` | int | Foreign ownership (0/1) |
| `firm_age` | float | Years since founding |
| `r_and_d` | float | R&D intensity (% of revenue) |

**Structure**: Translog frontier + BC95 determinants + four-component + regional effects + 2015-2016 recession shock.

---

## Reproducibility

All datasets can be regenerated:

```python
from utils.data_generation import *

# Example: regenerate hospital data
df = generate_hospital_data(n_hospitals=200, seed=42)
df.to_csv("data/hospital_data.csv", index=False)
```
