# PanelBox GMM Validation Suite

Validação completa da implementação GMM do PanelBox contra Stata xtabond2.

## 📁 Estrutura

```
validation/
├── data/
│   ├── abdata.dta          # Arellano-Bond dataset (Stata format)
│   └── abdata.csv          # Arellano-Bond dataset (CSV format)
├── stata/
│   ├── 01_difference_gmm_basic.do
│   ├── 02_difference_gmm_collapsed.do
│   ├── 03_system_gmm_basic.do
│   ├── 04_system_gmm_collapsed.do
│   ├── 05_with_time_dummies.do
│   └── run_all.do          # Master script
├── python/
│   ├── replicate_01_difference_gmm_basic.py
│   ├── replicate_02_difference_gmm_collapsed.py
│   ├── replicate_03_system_gmm_basic.py
│   ├── replicate_04_system_gmm_collapsed.py
│   ├── replicate_05_with_time_dummies.py
│   ├── compare_results.py
│   └── generate_report.py
└── results/
    ├── stata/              # Stata xtabond2 outputs
    ├── python/             # PanelBox outputs
    └── comparison/         # Comparison tables
```

## 🎯 Validation Scripts

### 01: Difference GMM Basic
- **Reference:** Roodman (2009), page 106
- **Model:** `n = γ·n_{t-1} + β_w·w + β_k·k + ε`
- **Instruments:** GMM-style (lags 2 to max), IV-style (w, k)
- **Target:** Validate core Difference GMM implementation

### 02: Difference GMM Collapsed
- **Same as 01** but with `collapse` option
- **Target:** Validate instrument collapsing
- **Expected:** Fewer instruments, similar coefficients

### 03: System GMM Basic
- **Reference:** Roodman (2009), page 120
- **Model:** System GMM (difference + level equations)
- **Instruments:** 
  - Difference eq: levels of L.n (lags 2+)
  - Level eq: differences of L.n (lag 1)
- **Target:** Validate System GMM implementation

### 04: System GMM Collapsed
- **Same as 03** but with `collapse`
- **Target:** Validate System GMM with collapsed instruments

### 05: With Time Dummies
- **Model:** Includes year fixed effects
- **Challenge:** Higher dimensionality
- **Target:** Validate time dummy handling

## 🚀 Running Validation

### Prerequisites

**Stata:**
- Stata 14 or higher
- xtabond2 package (`ssc install xtabond2`)

**Python:**
- PanelBox installed
- pandas, numpy, scipy

### Step 1: Run Stata Scripts

```bash
cd validation/stata
stata -b do run_all.do
```

Generates: `validation/results/stata/*.txt`

### Step 2: Run Python Replication

```bash
cd validation/python
python replicate_01_difference_gmm_basic.py
python replicate_02_difference_gmm_collapsed.py
python replicate_03_system_gmm_basic.py
python replicate_04_system_gmm_collapsed.py
python replicate_05_with_time_dummies.py
```

Generates: `validation/results/python/*.json`

### Step 3: Compare Results

```bash
cd validation/python
python compare_results.py
```

Generates: `validation/results/comparison/*.csv`

### Step 4: Generate Report

```bash
python generate_report.py
```

Generates: `validation/VALIDATION_REPORT_STATA.md`

## 📊 Comparison Criteria

### Coefficients
- **Tolerance:** < 0.01% difference
- **Formula:** `|coef_stata - coef_python| / |coef_stata| < 0.0001`

### Standard Errors
- **Tolerance:** < 0.5% difference
- **Formula:** `|se_stata - se_python| / |se_stata| < 0.005`

### Test Statistics
- **Hansen J:** < 0.1% difference
- **AR tests:** < 1% difference

## 📝 Expected Results

### Example 01: Difference GMM Basic

**Stata xtabond2:**
```
Number of obs      =       611
Number of groups   =       140
Number of instruments =        42

                 Coef.   Std. Err.
L.n           .6861222   .1410496
w            -.6078527   .1749928
k             .3569219   .0617118

Hansen test:  chi2(39) =  28.31   Prob > chi2 = 0.905
AR(2) test:   z = -0.34            Pr > z = 0.731
```

**PanelBox (expected to match within tolerance):**
```python
Number of observations:            611
Number of groups:                  140
Number of instruments:              42

Variable            Coef.     Std.Err.        z    P>|z|
L1.n                0.686      0.141       4.87   0.000
w                  -0.608      0.175      -3.47   0.001
k                   0.357      0.062       5.78   0.000

Hansen J-test: statistic=28.31, p-value=0.905
AR(2) test: statistic=-0.34, p-value=0.731
```

## ✅ Success Criteria

- [ ] All 5 examples replicate Stata results
- [ ] Coefficients within 0.01% tolerance
- [ ] Standard errors within 0.5% tolerance
- [ ] Test statistics within 1% tolerance
- [ ] Performance within 2x of Stata
- [ ] Complete documentation of any discrepancies

## 📚 References

**Roodman, D. (2009).** "How to do xtabond2: An introduction to difference and system GMM in Stata." *The Stata Journal*, 9(1), 86-136.

**Arellano, M., & Bond, S. (1991).** "Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations." *Review of Economic Studies*, 58(2), 277-297.

**Blundell, R., & Bond, S. (1998).** "Initial conditions and moment restrictions in dynamic panel data models." *Journal of Econometrics*, 87(1), 115-143.

## 📧 Contact

For questions about this validation suite, please open an issue on GitHub.
