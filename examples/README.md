# PanelBox Examples and Tutorials

Welcome to the PanelBox examples repository! This directory contains comprehensive tutorials and example notebooks to help you master panel data econometrics with PanelBox.

## Quick Start

### Installation
```bash
pip install panelbox
```

### Your First Analysis
Start with the fundamentals module:
```bash
cd tutorials/01_fundamentals
jupyter notebook 01_introduction_panel_data.ipynb
```

## Tutorial Modules

### 1. Fundamentals (Start Here!)
**Time**: 3.5-4.5 hours | **Level**: Beginner to Intermediate

Learn the core concepts of panel data analysis:
- [01 - Introduction to Panel Data Structures](tutorials/01_fundamentals/01_introduction_panel_data.ipynb)
- [02 - Model Specification with Formulas](tutorials/01_fundamentals/02_formulas_specification.ipynb)
- [03 - Estimation and Results Interpretation](tutorials/01_fundamentals/03_estimation_interpretation.ipynb)
- [04 - Spatial Fundamentals](tutorials/01_fundamentals/04_spatial_fundamentals.ipynb) (Optional)

### 2. Classical Estimators (Coming Soon)
Fixed Effects, Random Effects, Between, First Difference

### 3. Dynamic GMM (Coming Soon)
Arellano-Bond, Difference GMM, System GMM

### 4. Spatial Models (Coming Soon)
Spatial Lag, Spatial Error, Spatial Panel

### 5. Advanced Topics (Coming Soon)
Quantile Regression, Stochastic Frontier Analysis, Selection Models

### 15. Visualization and Reports
**Time**: ~4 hours | **Level**: Beginner to Intermediate

Master PanelBox's visualization and report generation capabilities:
- [01 - Visualization Introduction](visualization/notebooks/01_visualization_introduction.ipynb)
- [02 - Visual Diagnostics](visualization/notebooks/02_visual_diagnostics.ipynb)
- [03 - Advanced Visualizations](visualization/notebooks/03_advanced_visualizations.ipynb)
- [04 - Automated Reports](visualization/notebooks/04_automated_reports.ipynb)

## Learning Paths

### Path A: General Panel Data Analysis
```
01_fundamentals → 02_classical_estimators → 03_gmm_dynamic
```

### Path B: Spatial Econometrics
```
01_fundamentals (notebooks 1,2,3,4) → 04_spatial_models
```

## Datasets

All datasets are located in `/datasets/` with detailed documentation.

**Included:**
- Grunfeld Investment Data (1935-1954)
- Arellano-Bond Employment Data (future)
- Sample Spatial Data (future)

See [datasets/README.md](datasets/README.md) for complete documentation.

## Directory Structure

```
examples/
├── tutorials/              # Tutorial notebooks organized by module
│   ├── 01_fundamentals/
│   ├── 02_classical_estimators/
│   ├── 03_gmm_dynamic/
│   ├── 04_spatial_models/
│   └── 05_advanced/
├── visualization/         # Series 15 — Visualization and Reports
│   ├── notebooks/         # 01-04 tutorial notebooks
│   ├── data/              # Datasets (mostly synthetic)
│   ├── outputs/           # Generated charts and reports (git-ignored)
│   ├── solutions/         # Fully executed solution notebooks
│   ├── utils/             # Shared data generators
│   └── tests/             # Sanity tests
├── datasets/              # Datasets used in tutorials
├── solutions/             # Solutions to exercises
├── scripts/               # Utility scripts
└── cheatsheets/          # Quick reference materials
```

## Prerequisites

### Required Knowledge
- Basic Python (variables, loops, functions)
- Pandas fundamentals (DataFrames, indexing)
- Introductory econometrics (linear regression, hypothesis testing)
- Basic linear algebra (matrices, vectors)

### Software Requirements
- Python 3.8+
- Jupyter Notebook or JupyterLab
- PanelBox and dependencies (installed via pip)

### Environment Setup
To verify your environment is ready:
```bash
cd scripts
python setup_environment.py
```

## Additional Resources

### 📖 Documentation
- **Main Documentation**: https://panelbox.readthedocs.io
- **API Reference**: https://panelbox.readthedocs.io/en/latest/api/
- **GMM Tutorial**: [docs/gmm/tutorial.md](../docs/gmm/tutorial.md)
- **GMM Interpretation Guide**: [docs/gmm/interpretation_guide.md](../docs/gmm/interpretation_guide.md)

### 💡 Example Scripts
Beyond tutorials, check out practical examples:
- **GMM Examples**: [examples/gmm/](gmm/)
  - Basic Difference GMM
  - System GMM
  - Unbalanced panel handling
  - Production function estimation
- **Validation Examples**: [examples/validation/](validation/)
  - Bootstrap methods
  - Cross-validation
  - Sensitivity analysis
- **Spatial Examples**: [examples/spatial/](spatial/)
  - Basic spatial models
  - Technology diffusion
  - Regional analysis

### 🔗 External Resources
- **Arellano-Bond (1991)**: Original Difference GMM paper
- **Blundell-Bond (1998)**: System GMM methodology
- **Roodman (2009)**: "How to do xtabond2" - Essential GMM guide
- **Baltagi (2021)**: Panel Data Econometrics textbook
- **Wooldridge (2010)**: Cross Section and Panel Data

## Need Help?

- **Documentation**: https://panelbox.readthedocs.io
- **Issues**: https://github.com/panelbox/panelbox/issues
- **Discussions**: https://github.com/panelbox/panelbox/discussions

## Contributing

Found a typo or want to improve an example? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test all notebooks (Restart Kernel and Run All)
5. Submit a pull request

## License

PanelBox is licensed under the BSD 3-Clause License. See LICENSE file for details.

## Citation

If you use PanelBox in your research, please cite:

```bibtex
@software{panelbox,
  title={PanelBox: A Python Library for Panel Data Econometrics},
  author={Your Name},
  year={2026},
  url={https://github.com/panelbox/panelbox}
}
```

---

**Last Updated**: 2026-02-16
**PanelBox Version**: 0.8.0+
