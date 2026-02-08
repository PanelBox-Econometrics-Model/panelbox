# Sprint 5 Planning - Advanced Features

**Date**: 2026-02-08
**Status**: PLANNING
**Duration**: 5 days (estimated)

---

## 🎯 Sprint Goal

**Enhance the Experiment Pattern with advanced result containers and polish the system for production excellence**

**Focus Areas**:
1. ResidualResult container for residual diagnostics
2. Chart registration system fixes (warnings in current version)
3. Enhanced metadata and tracking
4. Performance optimization
5. Production polish

---

## 📊 Sprint Metrics

**Target**: 13-15 story points
**Estimated Time**: 8-10 hours
**Expected Velocity**: 100-120% (based on previous sprints)

---

## 🎯 User Stories Proposed

### Priority 1: Critical Fixes

#### US-016: Fix Chart Registration System (3 pts) 🔴 HIGH PRIORITY

**Problem**: Current system shows warnings:
```
Warning: Chart type 'validation_test_overview' is not registered. Available charts: none
Warning: Chart type 'comparison_coefficients' is not registered. Available charts: none
```

**As a user**, I want charts to display properly in HTML reports without warnings.

**Acceptance Criteria**:
- [ ] No chart registration warnings
- [ ] All validation charts render properly
- [ ] All comparison charts render properly
- [ ] Registry properly initialized at import time
- [ ] Tests verify charts are registered

**Value**: HIGH - Fixes production issue, improves user experience

**Tasks**:
1. Debug chart registry initialization (1h)
2. Fix chart registration in visualization/__init__.py (1h)
3. Test all chart types render correctly (1h)
4. Update tests to verify registration (30m)

---

### Priority 2: Core Features

#### US-017: ResidualResult Container (5 pts) 🟡 MEDIUM PRIORITY

**As a researcher**, I want a ResidualResult container to analyze and report on residual diagnostics.

**Acceptance Criteria**:
- [ ] `ResidualResult` class inheriting from `BaseResult`
- [ ] Stores residuals, fitted values, diagnostics
- [ ] Properties: `shapiro_test`, `durbin_watson`, `ljung_box`
- [ ] Method: `summary()` with diagnostic statistics
- [ ] Method: `save_html()` generates residual diagnostics report
- [ ] Integration with PanelExperiment: `experiment.analyze_residuals(name)`
- [ ] Tests with >85% coverage

**Value**: MEDIUM - Completes the result container trilogy

**Tasks**:
1. Create ResidualResult class (2h)
2. Implement residual diagnostic tests (1h)
3. Create residual report template (if needed) (1h)
4. Add analyze_residuals() to PanelExperiment (30m)
5. Write comprehensive tests (1h)

---

#### US-018: Enhanced Metadata Tracking (3 pts) 🟡 MEDIUM PRIORITY

**As a user**, I want better metadata tracking for reproducibility and auditing.

**Acceptance Criteria**:
- [ ] Track package version in results
- [ ] Track computation time for model fitting
- [ ] Track computation environment (Python version, OS)
- [ ] Add `experiment.get_history()` for audit trail
- [ ] Metadata included in JSON exports
- [ ] Tests verify metadata completeness

**Value**: MEDIUM - Improves reproducibility

**Tasks**:
1. Add version tracking to results (30m)
2. Add timing decorators to fit methods (1h)
3. Add environment detection (30m)
4. Implement get_history() method (1h)
5. Update tests (30m)

---

### Priority 3: Polish & Optimization

#### US-019: Performance Optimization (2 pts) 🟢 LOW PRIORITY

**As a user**, I want fast report generation even with large datasets.

**Acceptance Criteria**:
- [ ] Profile report generation performance
- [ ] Optimize ValidationTransformer
- [ ] Optimize chart generation
- [ ] Cache expensive computations
- [ ] Tests verify performance improvements

**Value**: LOW - Nice to have

**Tasks**:
1. Profile current performance (30m)
2. Optimize hot paths (1h)
3. Add caching where appropriate (1h)
4. Benchmark improvements (30m)

---

#### US-020: Export Enhancements (3 pts) 🟢 LOW PRIORITY

**As a user**, I want to export results in additional formats.

**Acceptance Criteria**:
- [ ] `result.to_latex()` - LaTeX table export
- [ ] `result.to_markdown()` - Markdown table export
- [ ] `result.to_dict(format='compact')` - Compact format
- [ ] Tests for all export formats

**Value**: LOW - Nice to have, useful for papers

**Tasks**:
1. Implement to_latex() (1h)
2. Implement to_markdown() (1h)
3. Add format parameter to to_dict() (30m)
4. Write tests (1h)

---

## 📋 Sprint Backlog (Recommended)

### Sprint 5A (MVP - 8 pts)

**Focus**: Fix critical issues + ResidualResult

1. **US-016**: Fix Chart Registration (3 pts) ✅ MUST HAVE
2. **US-017**: ResidualResult Container (5 pts) ✅ SHOULD HAVE

**Total**: 8 points
**Time**: ~5-6 hours
**Outcome**: Production-ready system with no warnings + complete result container set

---

### Sprint 5B (Full - 14 pts)

**If time permits, add**:

3. **US-018**: Enhanced Metadata (3 pts) - Nice to have
4. **US-019**: Performance Optimization (2 pts) - Polish

**Total**: 13 points
**Time**: ~8-9 hours

---

### Sprint 5C (Extended - 17 pts)

**If extra time**:

5. **US-020**: Export Enhancements (3 pts) - Polish

**Total**: 16 points
**Time**: ~11-12 hours

---

## 🎯 Recommendation: Sprint 5A (MVP)

**Focus on**: US-016 (Chart Registration Fix) + US-017 (ResidualResult)

**Why**:
1. **US-016 is critical** - Fixes production warnings
2. **US-017 completes the pattern** - ValidationResult, ComparisonResult, ResidualResult
3. **8 points is achievable** - Matches recent sprint velocities
4. **High value/time ratio** - Core features that users need

**Benefits**:
- ✅ No warnings in reports
- ✅ Complete result container set
- ✅ Clean, production-ready package
- ✅ Ready for PyPI deployment as v0.6.1 or v0.7.0

---

## 🗓️ Sprint 5 Timeline

### Day 1 (2-3 hours)
**Morning**: Sprint Planning + US-016
- Sprint planning (30m)
- Debug chart registration issue (1h)
- Fix registration system (1h)
- Test all charts (30m)

**Outcome**: Chart warnings fixed ✅

### Day 2 (3-4 hours)
**Full Day**: US-017 Part 1
- Create ResidualResult class (2h)
- Implement diagnostic tests (1h)
- Create template (if needed) (1h)

**Outcome**: ResidualResult core complete

### Day 3 (1-2 hours)
**Morning**: US-017 Part 2
- Add to PanelExperiment (30m)
- Write comprehensive tests (1h)
- Integration testing (30m)

**Outcome**: ResidualResult fully integrated ✅

### Day 4 (1 hour)
**Morning**: Polish & Documentation
- Update documentation (30m)
- Create sprint5_review.md (30m)

**Outcome**: Sprint 5 complete, documented

### Day 5 (Optional)
**If time**: US-018 or US-019
- Enhanced metadata OR
- Performance optimization

---

## 📊 Definition of Done (Sprint 5)

A user story is DONE when:
- [ ] Code implemented per acceptance criteria
- [ ] Tests written and passing (>85% coverage)
- [ ] No warnings in console output
- [ ] Integration with existing system verified
- [ ] Documentation updated (docstrings)
- [ ] Sprint review document created
- [ ] Ready for deployment

---

## 🎯 Sprint Success Criteria

Sprint 5 is successful if:

**MVP (Sprint 5A)**:
- [x] US-016: Chart registration fixed (no warnings)
- [x] US-017: ResidualResult implemented and tested
- [x] All tests passing (>85% coverage)
- [x] Integration tests verify end-to-end workflow
- [x] Documentation complete

**Full (Sprint 5B)**:
- [x] MVP criteria met
- [x] US-018: Enhanced metadata tracking
- [x] US-019: Performance optimized

**Extended (Sprint 5C)**:
- [x] Full criteria met
- [x] US-020: Export enhancements (LaTeX, Markdown)

---

## 📝 Technical Notes

### Chart Registration Issue

Current problem:
```python
# visualization/__init__.py might not be registering charts properly
# Registry.register() calls may not be executing at import time
```

**Solution approach**:
1. Check if Registry._instance is initialized
2. Verify chart classes are importing correctly
3. Ensure register() is called at module import
4. Add explicit initialization if needed

### ResidualResult Design

```python
class ResidualResult(BaseResult):
    """Container for residual diagnostics."""

    def __init__(self, model_results, residuals=None, fitted_values=None, ...):
        self.model_results = model_results
        self.residuals = residuals or model_results.resid
        self.fitted_values = fitted_values or model_results.fittedvalues

    @property
    def shapiro_test(self):
        """Shapiro-Wilk test for normality."""
        from scipy.stats import shapiro
        return shapiro(self.residuals)

    @property
    def durbin_watson(self):
        """Durbin-Watson test for autocorrelation."""
        from statsmodels.stats.stattools import durbin_watson
        return durbin_watson(self.residuals)
```

---

## 🚀 After Sprint 5

### Version Planning

**Option A**: Release v0.6.1 (patch)
- Just bug fixes (US-016)
- Quick release

**Option B**: Release v0.7.0 (minor)
- US-016 + US-017
- New feature (ResidualResult)
- Recommended

**Option C**: Continue to Sprint 6
- Add more features
- Release v0.7.0 or v1.0.0

---

## 📊 Sprint 5 vs Previous Sprints

| Sprint | Points | Features | Time | Velocity |
|--------|--------|----------|------|----------|
| Sprint 1 | 14 pts | Viz API | ~2h | 127% |
| Sprint 2 | 13 pts | Reports | ~3h | 130% |
| Sprint 3 | 13 pts | Experiment | ~2h | 100% |
| Sprint 4 | 13 pts | Results | ~3h | 100% |
| **Sprint 5A** | **8 pts** | **Fix + Residual** | **~5h** | **TBD** |
| **Sprint 5B** | **13 pts** | **+ Metadata + Perf** | **~8h** | **TBD** |

---

## ✅ Sprint 5 Decision

**Recommended**: **Sprint 5A (MVP)** - 8 points

**Focus**:
1. US-016: Fix Chart Registration (3 pts) - CRITICAL
2. US-017: ResidualResult Container (5 pts) - COMPLETE TRILOGY

**Outcome**: Production-ready v0.6.1 or v0.7.0 with:
- ✅ No warnings
- ✅ Complete result container set (Validation, Comparison, Residual)
- ✅ Clean, polished system
- ✅ Ready for PyPI

---

**Ready to start Sprint 5A?** 🚀

**First Task**: US-016 - Fix Chart Registration System
**Estimated Time**: 3 hours
**Priority**: HIGH (fixes production warnings)
