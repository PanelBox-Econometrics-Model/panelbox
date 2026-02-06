# PanelBox Jupyter Notebooks

Welcome to the **PanelBox Jupyter Notebooks** collection! This directory contains comprehensive, hands-on tutorials for learning panel data econometrics with PanelBox.

## 📚 Available Notebooks

### 🔴 Core Notebooks (Start Here!)

| # | Notebook | Level | Time | Description |
|---|----------|-------|------|-------------|
| 00 | [Getting Started](./00_getting_started.ipynb) | Beginner | 10 min | Your first panel data analysis in 10 minutes |
| 01 | [Static Models Complete](./01_static_models_complete.ipynb) | Intermediate | 30-40 min | All 5 static panel models with tests |
| 02 | [Dynamic GMM Complete](./02_dynamic_gmm_complete.ipynb) | Advanced | 40-50 min | Difference & System GMM - flagship feature |

### 🟡 Advanced Notebooks (Available Now!)

| # | Notebook | Level | Time | Description |
|---|----------|-------|------|-------------|
| 03 | [Validation Complete](./03_validation_complete.ipynb) | Advanced | 30-40 min | 20+ validation tests |
| 04 | [Robust Inference](./04_robust_inference.ipynb) | Advanced | 30-40 min | Bootstrap, sensitivity analysis |
| 05 | [Report Generation](./05_report_generation.ipynb) | Intermediate | 20-30 min | HTML, Markdown, LaTeX export |

### 🟢 Specialized Notebooks (Coming Soon)

| # | Notebook | Focus | Status |
|---|----------|-------|--------|
| 06 | Advanced Features | IV, custom formulas | ⏳ Planned |
| 07 | Real-World Case Study | Publication-ready analysis | ⏳ Planned |
| 08 | Unbalanced Panels | Missing data strategies | ⏳ Planned |
| 09 | Performance Optimization | Large datasets, Numba | ⏳ Planned |

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install panelbox
```

Or from source:

```bash
git clone https://github.com/PanelBox-Econometrics-Model/panelbox.git
cd panelbox
pip install -e .
```

### 2. Launch Jupyter

```bash
cd examples/jupyter
jupyter notebook
```

### 3. Start with Notebook 00

Open `00_getting_started.ipynb` and follow along!

---

## 📖 Learning Path

### For Beginners

Start here if you're new to panel data or PanelBox:

1. **[00_getting_started.ipynb](./00_getting_started.ipynb)** (10 min)
   - Install PanelBox
   - Load your first dataset
   - Estimate your first model
   - Interpret results

2. **[01_static_models_complete.ipynb](./01_static_models_complete.ipynb)** (30-40 min)
   - Learn all 5 static models
   - Understand when to use each
   - Run specification tests
   - Compare models

### For Intermediate Users

You understand panel data basics, want to learn GMM:

3. **[02_dynamic_gmm_complete.ipynb](./02_dynamic_gmm_complete.ipynb)** (40-50 min)
   - Why GMM?
   - Difference GMM (Arellano-Bond)
   - System GMM (Blundell-Bond)
   - All specification tests
   - Troubleshooting guide

### For Advanced Users

You want to master all features:

4. **[03_validation_complete.ipynb](./03_validation_complete.ipynb)** (30-40 min)
   - 20+ validation tests
   - Unit root tests (LLC, IPS, Fisher)
   - Cointegration tests (Pedroni, Kao)
   - Diagnostic tests (serial correlation, heteroskedasticity, cross-sectional dependence)
   - ValidationSuite for comprehensive testing

5. **[04_robust_inference.ipynb](./04_robust_inference.ipynb)** (30-40 min)
   - 8 types of robust SE (HC0-HC3, clustered, Driscoll-Kraay, Newey-West, PCSE)
   - 4 bootstrap methods (pairs, wild, block, residual)
   - Sensitivity analysis (leave-one-out, subset stability)
   - Outlier detection and influence diagnostics

6. **[05_report_generation.ipynb](./05_report_generation.ipynb)** (20-30 min)
   - HTML reports (interactive, styled)
   - Markdown export (GitHub-friendly)
   - LaTeX tables (publication-ready)
   - Comparison tables (multiple models)
   - Automated workflows

---

## 🎯 What's Covered

### Static Panel Models

- ✅ Pooled OLS
- ✅ Between Estimator
- ✅ Fixed Effects (Within)
- ✅ Random Effects (GLS)
- ✅ First Difference

### Dynamic Panel GMM

- ✅ Difference GMM (Arellano-Bond 1991)
- ✅ System GMM (Blundell-Bond 1998)
- ✅ Instrument selection & collapse option
- ✅ Windmeijer correction
- ✅ All GMM specification tests

### Specification Tests

- ✅ F-test (Pooled vs FE)
- ✅ Hausman test (FE vs RE)
- ✅ Hansen J-test (overidentification)
- ✅ Sargan test
- ✅ AR(1) & AR(2) tests

### Robust Standard Errors

- ✅ HC0, HC1, HC2, HC3
- ✅ Clustered (entity, time, two-way)
- ✅ Driscoll-Kraay
- ✅ Newey-West
- ✅ PCSE

### Validation Tests

- ✅ Specification tests (Hausman, RESET, Mundlak, Chow)
- ✅ Unit root (LLC, IPS, Fisher)
- ✅ Cointegration (Pedroni, Kao)
- ✅ Serial correlation (Wooldridge, Breusch-Godfrey, Baltagi-Wu)
- ✅ Heteroskedasticity (Modified Wald, Breusch-Pagan, White)
- ✅ Cross-sectional dependence (Pesaran CD, BP-LM, Frees)
- ✅ ValidationSuite (run all tests at once)

### Report Generation

- ✅ HTML reports (interactive, styled)
- ✅ Markdown export (GitHub-friendly)
- ✅ LaTeX tables (publication-ready, journal style)
- ✅ Comparison tables (multiple models side-by-side)
- ✅ Custom formatting and templates
- ✅ Automated workflows

---

## 📊 Datasets Used

All notebooks use **built-in datasets** - no external data required!

### Grunfeld Dataset
- **Used in**: Notebooks 00, 01
- **Content**: Investment data for 10 US firms (1935-1954)
- **Load**: `pb.load_grunfeld()`
- **Size**: 200 observations (balanced panel)

### Arellano-Bond Dataset
- **Used in**: Notebook 02
- **Content**: Employment data for UK firms (1979-1984)
- **Load**: `pb.load_abdata()`
- **Size**: ~1,000 observations (unbalanced panel)

---

## 💡 Tips for Success

### Before You Start

1. **Install dependencies**: `pip install panelbox jupyter matplotlib seaborn`
2. **Basic Python**: Familiarity with pandas and numpy helps
3. **Econometrics background**: Not required, but helpful

### While Working Through Notebooks

1. **Run cells sequentially**: Don't skip cells
2. **Read the narrative**: Not just the code!
3. **Experiment**: Try changing parameters
4. **Check outputs**: Make sure results make sense

### Common Issues

**Q: Import error for panelbox?**
```bash
# Make sure you're in the right environment
pip install panelbox
# Or if using development version:
pip install -e /path/to/panelbox
```

**Q: Plots not showing?**
```python
# Add this at the top of notebook:
%matplotlib inline
```

**Q: Notebook kernel dies?**
```bash
# You might need more memory for large datasets
# Or there's a bug - please report it!
```

---

## 📝 Notebook Structure

Each notebook follows this template:

1. **Introduction**
   - What you'll learn
   - Prerequisites
   - Estimated time

2. **Setup**
   - Imports
   - Configuration
   - Load data

3. **Main Content**
   - Step-by-step tutorial
   - Code + narrative
   - Visualizations

4. **Results & Interpretation**
   - What the numbers mean
   - Economic interpretation
   - Statistical inference

5. **Summary & Next Steps**
   - Key takeaways
   - Links to related notebooks
   - Further reading

---

## 🔗 Additional Resources

### PanelBox Resources

- **Documentation**: [GitHub Wiki](https://github.com/PanelBox-Econometrics-Model/panelbox/wiki)
- **API Reference**: [API Docs](https://github.com/PanelBox-Econometrics-Model/panelbox/tree/main/docs)
- **Examples (Python scripts)**: [examples/](../README.md)
- **Issues**: [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues)

### Panel Data Econometrics

**Textbooks**:
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

**Key Papers**:
- Arellano & Bond (1991): "Some Tests of Specification for Panel Data"
- Blundell & Bond (1998): "Initial Conditions and Moment Restrictions"
- Roodman (2009): "How to do xtabond2" (Stata Journal)
- Windmeijer (2005): "Finite Sample Correction for GMM"

---

## 🛠️ Development Status

### Completed ✅

**Milestone 1** (2026-02-05):
- [x] 00_getting_started.ipynb (19KB)
- [x] 01_static_models_complete.ipynb (34KB)
- [x] 02_dynamic_gmm_complete.ipynb (15KB)

**Milestone 2** (2026-02-05):
- [x] 03_validation_complete.ipynb (31KB)
- [x] 04_robust_inference.ipynb (31KB)
- [x] 05_report_generation.ipynb (34KB)

**Documentation**:
- [x] NOTEBOOKS_PLAN.md (tracking document)
- [x] README.md (this file)

**Milestone 2 Complete!** 🎉 (2026-02-05)

**Total**: 164KB of educational content covering all major features!

### Planned 📋

- [ ] 06_advanced_features.ipynb
- [ ] 07_real_world_case_study.ipynb
- [ ] 08_unbalanced_panels.ipynb
- [ ] 09_performance_optimization.ipynb

See [NOTEBOOKS_PLAN.md](./NOTEBOOKS_PLAN.md) for detailed roadmap.

---

## 🤝 Contributing

Found a bug or want to improve a notebook?

1. **Report Issues**: [GitHub Issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues)
2. **Suggest Improvements**: [GitHub Discussions](https://github.com/PanelBox-Econometrics-Model/panelbox/discussions)
3. **Submit PRs**: See [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## 📜 License

These notebooks are part of PanelBox and are licensed under the MIT License.

---

## 🎓 Citation

If you use these notebooks in your research or teaching, please cite:

```bibtex
@software{panelbox2026,
  author = {Haase, Gustavo and Dourado, Paulo},
  title = {PanelBox: Panel Data Econometrics in Python},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/PanelBox-Econometrics-Model/panelbox}
}
```

---

**Happy Learning!** 🚀

*PanelBox - Panel Data Econometrics Made Easy*

---

**Last Updated**: 2026-02-05
**Status**: Milestone 1 Complete (3/10 notebooks)
**Next Update**: After Milestone 2 completion
