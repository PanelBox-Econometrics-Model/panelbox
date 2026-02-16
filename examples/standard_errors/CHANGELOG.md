# Changelog

All notable changes to the Standard Errors Tutorial Series will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-02-16

### Added

#### Directory Structure
- Created main `standard_errors/` directory with organized subdirectories
- Added `data/` folder for datasets (11 datasets planned)
- Added `notebooks/` folder for 7 tutorial notebooks
- Added `outputs/figures/` with subfolders for each notebook (01-07)
- Added `outputs/reports/html/` for generated HTML reports
- Added `utils/` package with helper modules

#### Documentation
- Created comprehensive `README.md` with:
  - Tutorial overview and learning paths
  - Dataset descriptions and applications
  - Installation and setup instructions
  - Best practices and troubleshooting
  - References to key econometric papers
  - Citation information
- Created `.gitignore` for proper version control
- Created `CHANGELOG.md` (this file)

#### Utility Modules (Placeholders)
- Created `utils/__init__.py` for package initialization
- Created `utils/plotting.py` for visualization helpers
- Created `utils/diagnostics.py` for diagnostic tests
- Created `utils/data_generators.py` for synthetic data generation

### Planned

#### Phase 2: Data Preparation (Next)
- [ ] Prepare and validate 11 datasets
- [ ] Add data dictionaries for each dataset
- [ ] Create data validation scripts

#### Phase 3: Utility Development
- [ ] Implement plotting functions
- [ ] Implement diagnostic functions
- [ ] Implement data generators
- [ ] Add unit tests for utilities

#### Phase 4: Notebook Creation
- [ ] Create 7 tutorial notebooks (01-07)
- [ ] Add exercises to each notebook
- [ ] Create solution notebooks

#### Phase 5: Quality Assurance
- [ ] Test all notebooks end-to-end
- [ ] Peer review for accuracy
- [ ] Student testing (if applicable)

---

## [Unreleased]

### To Be Added
- Tutorial notebooks (01-07)
- Datasets (11 files)
- Utility function implementations
- Example HTML reports
- Video walkthroughs (optional)

---

## Version History

- **1.0.0** (2026-02-16): Initial directory structure and documentation
- **0.1.0** (Planning): Specification document created

---

## Notes

### Compatibility
- **PanelBox**: 0.8.0+
- **Python**: 3.9+
- **Dependencies**: pandas, numpy, matplotlib, seaborn, scipy

### Known Issues
- None (initial release)

### Deprecations
- None

---

**Last Updated**: 2026-02-16
