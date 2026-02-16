# 🎉 Announcement: Standard Errors Tutorial Series v1.0.0

**Date**: February 16, 2026

---

## What's New?

We're excited to announce the release of **PanelBox Standard Errors Tutorial Series v1.0.0**, a comprehensive collection of Jupyter notebooks covering robust inference methods for panel data econometrics.

---

## 📚 What You Get

### 7 Complete Tutorial Notebooks

1. **Robust Fundamentals** (60-75 min) - HC0-HC3 heteroskedasticity-robust standard errors
2. **Clustering in Panels** (75-90 min) - One-way and two-way clustering
3. **HAC and Autocorrelation** (90-120 min) - Newey-West and Driscoll-Kraay
4. **Spatial Errors** (90-120 min) - Spatial HAC with distance matrices
5. **MLE Inference** (75-90 min) - Sandwich estimator and delta method
6. **Bootstrap for Quantiles** (90-120 min) - Bootstrap methods for quantile regression
7. **Methods Comparison** (90-120 min) - Comprehensive comparison and best practices

### 11 Curated Datasets

Real and synthetic datasets covering:
- Corporate investment (Grunfeld)
- Economic growth panels
- Financial markets
- Agricultural productivity
- Wage and labor economics
- Credit and health insurance
- Spatial data with coordinates

### Utility Modules

Helper functions for:
- **Plotting**: Residuals, ACF/PACF, SE comparisons, spatial kernels, forest plots
- **Diagnostics**: Heteroskedasticity tests, autocorrelation tests, spatial correlation
- **Data Generation**: Synthetic data with various correlation structures

### Complete Documentation

- Main README with series overview
- Quick Start guide for immediate use
- CHANGELOG documenting version history
- Integration with PanelBox documentation site

---

## 🎯 Who Is This For?

- **Graduate Students**: Learning panel data econometrics and robust inference
- **Researchers**: Working with panel data and needing appropriate standard errors
- **Practitioners**: Applying panel methods to real-world data
- **Instructors**: Teaching econometrics courses

---

## 🚀 Getting Started

### Installation

```bash
# Install PanelBox (requires v0.8.0 or higher)
pip install panelbox>=0.8.0

# Install additional dependencies
pip install jupyter matplotlib seaborn statsmodels scipy
```

### Clone and Run

```bash
# Clone the repository
git clone https://github.com/PanelBox-Econometrics-Model/panelbox.git
cd panelbox/examples/standard_errors

# Launch Jupyter
cd notebooks
jupyter notebook 01_robust_fundamentals.ipynb
```

### Quick Start

For a rapid introduction, see our [Quick Start Guide](QUICK_START.md).

---

## 📖 Learning Paths

Choose your path based on your needs:

### 🟢 Basic Path (3-4 hours)
Perfect for practitioners who need robust standard errors for applied work.

**01 → 02 → 07**

### 🟡 Intermediate Path (5-6 hours)
For researchers working with time series panels or autocorrelated errors.

**01 → 02 → 03 → 07**

### 🔴 Advanced Path (8-10 hours)
For researchers dealing with spatial data, MLE models, or quantile regression.

**01 → 02 → 03 → 05 → 04 → 06 → 07**

---

## 🌟 Key Features

### Progressive Learning
- Each notebook builds on the previous one
- Clear learning objectives at the start
- Exercises with solutions
- Real-world examples

### Hands-On Examples
- 11 datasets covering diverse applications
- Reproducible code with set random seeds
- Publication-ready figures
- Interactive diagnostics

### Best Practices
- Decision tree for choosing methods
- Common pitfalls and how to avoid them
- Monte Carlo simulation studies
- Sensitivity analysis

### Integration with PanelBox
- Uses PanelBox v0.8.0+ features
- HTML report generation
- LaTeX export for publications
- Consistent API across all methods

---

## 📊 Example: Quick Comparison

```python
import panelbox as pb
from panelbox.standard_errors import StandardErrorComparison

# Load data
data = pb.load_grunfeld()

# Fit model with multiple SE methods
model = pb.FixedEffects(data, 'inv ~ value + capital')
result = model.fit()

# Compare standard errors
comparison = StandardErrorComparison(result)
comparison.compare_methods(['standard', 'HC3', 'cluster_entity', 'driscoll_kraay'])
comparison.plot()
```

See **Notebook 07** for comprehensive comparisons and decision frameworks.

---

## 📍 Where to Find Everything

### Main Directory
```
examples/standard_errors/
├── notebooks/           # 7 tutorial notebooks
├── data/               # 11 datasets
├── utils/              # Helper functions
├── outputs/            # Generated figures and reports
├── README.md           # Series overview
├── QUICK_START.md      # Quick start guide
└── CHANGELOG.md        # Version history
```

### Documentation
- **Series Overview**: [docs/tutorials/standard_errors_series.md](../../docs/tutorials/standard_errors_series.md)
- **PanelBox Docs**: https://panelbox-econometrics-model.github.io/panelbox
- **API Reference**: https://panelbox-econometrics-model.github.io/panelbox/api/

---

## 💡 Use Cases

### "My data has heteroskedasticity"
👉 Start with **Notebook 01** → Use HC3 standard errors

### "I have panel data with many firms"
👉 Go to **Notebook 02** → Use entity-clustered standard errors

### "My time series panel has autocorrelation"
👉 Check **Notebook 03** → Use Driscoll-Kraay standard errors

### "I'm working with spatial/geographic data"
👉 See **Notebook 04** → Use spatial HAC with distance weights

### "I'm estimating a logit/probit model"
👉 Learn from **Notebook 05** → Use MLE sandwich estimator

### "I'm doing quantile regression"
👉 Study **Notebook 06** → Use panel bootstrap methods

### "I don't know which method to use"
👉 Jump to **Notebook 07** → Follow the decision tree

---

## 🤝 Contributing

We welcome contributions! Found an issue or have suggestions?

- **Issues**: https://github.com/PanelBox-Econometrics-Model/panelbox/issues
- **Pull Requests**: https://github.com/PanelBox-Econometrics-Model/panelbox/pulls
- **Email**: panelbox@example.com

---

## 📝 Citation

If you use these tutorials in your research or teaching, please cite:

```bibtex
@software{panelbox2026,
  title = {PanelBox: Python Library for Panel Data Econometrics},
  author = {PanelBox Development Team},
  year = {2026},
  url = {https://github.com/PanelBox-Econometrics-Model/panelbox},
  version = {0.8.0},
  note = {Standard Errors Tutorial Series v1.0.0}
}
```

---

## 🎓 Related Resources

### Academic Papers
- White (1980) - Heteroskedasticity-robust standard errors
- Arellano (1987) - Cluster-robust standard errors
- Newey & West (1987) - HAC standard errors
- Driscoll & Kraay (1998) - Panel HAC
- Cameron, Gelbach & Miller (2011) - Multiway clustering
- Conley (1999) - Spatial HAC

### Books
- Wooldridge (2010) - *Econometric Analysis of Cross Section and Panel Data*
- Cameron & Trivedi (2005) - *Microeconometrics: Methods and Applications*
- Angrist & Pischke (2009) - *Mostly Harmless Econometrics*

### Other PanelBox Tutorials
- [Getting Started](../../docs/tutorials/01_getting_started.md)
- [Static Models](../../docs/tutorials/02_static_models.md)
- [GMM Introduction](../../docs/tutorials/03_gmm_intro.md)

---

## 🔄 Version Information

- **Tutorial Series Version**: 1.0.0
- **Release Date**: February 16, 2026
- **PanelBox Version**: 0.8.0+
- **Python Version**: 3.9+
- **Git Tag**: `v1.0.0-standard-errors`

---

## 📣 Spread the Word!

Share this tutorial series with your colleagues, students, and collaborators:

- GitHub: https://github.com/PanelBox-Econometrics-Model/panelbox
- Twitter: #PanelBox #Econometrics #PanelData
- LinkedIn: PanelBox Development Team

---

## 🙏 Acknowledgments

This tutorial series was developed with contributions from:
- PanelBox Development Team
- Panel data econometrics community
- Graduate students and researchers who provided feedback

Special thanks to the authors of the academic papers that form the foundation of these methods.

---

## 🔜 What's Next?

We're continuously improving the tutorial series. Upcoming additions:

- **Video walkthroughs** for each notebook
- **Interactive exercises** with automated grading
- **Additional datasets** from diverse fields
- **Translated versions** (Spanish, Portuguese, Chinese)
- **R/Stata comparison notebooks** for cross-platform users

Stay tuned for updates!

---

## 📬 Contact

Questions? Feedback? Suggestions?

- **GitHub Issues**: https://github.com/PanelBox-Econometrics-Model/panelbox/issues
- **Email**: panelbox@example.com
- **Documentation**: https://panelbox-econometrics-model.github.io/panelbox

---

## 📄 License

This tutorial series is licensed under the MIT License.

---

**Happy Learning!** 🎓📊📈

---

*Last Updated: February 16, 2026*
