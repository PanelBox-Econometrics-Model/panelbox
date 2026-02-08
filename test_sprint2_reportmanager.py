"""
Test Sprint 2 - ReportManager Integration
==========================================

Tests the integration between ReportManager, TemplateManager, and CSSManager
to generate a complete validation report.
"""

from pathlib import Path

from panelbox.report.report_manager import ReportManager


def test_report_manager():
    """Test ReportManager with proper validation data structure."""

    print("=" * 70)
    print("SPRINT 2: Testing ReportManager Integration")
    print("=" * 70)
    print()

    # 1. Initialize ReportManager
    print("1. Initializing ReportManager...")
    report_mgr = ReportManager(enable_cache=True, minify=False)
    print(f"   ✅ ReportManager initialized")
    print(f"   - TemplateManager: {report_mgr.template_manager}")
    print(f"   - CSSManager: {report_mgr.css_manager}")
    print(f"   - AssetManager: {report_mgr.asset_manager}")
    print()

    # 2. Prepare validation data with proper structure
    print("2. Preparing validation data...")

    # Summary data (required by template)
    summary = {
        "total_tests": 5,
        "total_passed": 3,
        "total_failed": 2,
        "pass_rate": 60.0,
        "pass_rate_formatted": "60.0%",
        "has_issues": True,
        "overall_status": "warning",
        "status_message": "Some tests failed - Review recommendations",
        "tests_failed": 2,
        "tests_passed": 3,
    }

    # Model information
    model_info = {
        "model_type": "PanelOLS",
        "formula": "y ~ x1 + x2 + EntityEffects",
        "nobs": 1000,
        "nobs_formatted": "1,000",
        "n_entities": 100,
        "n_entities_formatted": "100",
        "n_periods": 10,
        "n_periods_formatted": "10",
        "balanced": True,
    }

    # Test results
    tests = {
        "specification": [
            {
                "name": "Hausman Test",
                "status": "passed",
                "statistic": 2.34,
                "pvalue": 0.31,
                "pvalue_formatted": "0.310",
                "conclusion": "Random effects appropriate",
                "description": "Tests whether random effects model is consistent",
            },
            {
                "name": "F-Test for Fixed Effects",
                "status": "failed",
                "statistic": 15.67,
                "pvalue": 0.001,
                "pvalue_formatted": "0.001",
                "conclusion": "Fixed effects needed",
                "description": "Tests presence of entity-specific effects",
            },
        ],
        "serial_correlation": [
            {
                "name": "Wooldridge Test",
                "status": "passed",
                "statistic": 1.23,
                "pvalue": 0.27,
                "pvalue_formatted": "0.270",
                "conclusion": "No serial correlation detected",
                "description": "Tests for first-order autocorrelation",
            }
        ],
        "heteroskedasticity": [
            {
                "name": "Breusch-Pagan Test",
                "status": "failed",
                "statistic": 45.78,
                "pvalue": 0.003,
                "pvalue_formatted": "0.003",
                "conclusion": "Heteroskedasticity detected",
                "description": "Tests for heteroskedastic errors",
            }
        ],
        "cross_section": [
            {
                "name": "Pesaran CD Test",
                "status": "passed",
                "statistic": 0.89,
                "pvalue": 0.37,
                "pvalue_formatted": "0.370",
                "conclusion": "No cross-sectional dependence",
                "description": "Tests for cross-sectional correlation",
            }
        ],
    }

    # Recommendations (matching template structure)
    recommendations = [
        {
            "severity": "high",
            "category": "heteroskedasticity",
            "issue": "Address Heteroskedasticity",
            "tests": ["Breusch-Pagan Test"],
            "suggestions": [
                "Use robust standard errors (HC1, HC2, or HC3) to obtain valid inference",
                "Refit model with robust covariance matrix",
                "Consider White or cluster-robust standard errors",
            ],
        },
        {
            "severity": "medium",
            "category": "specification",
            "issue": "Consider Fixed Effects",
            "tests": ["F-Test for Fixed Effects"],
            "suggestions": [
                "Use fixed effects model to account for entity-specific effects",
                "Include entity dummies if fixed effects model is not suitable",
                "Run additional specification tests (Mundlak test) to validate choice",
            ],
        },
    ]

    # Combine into full context
    validation_data = {
        "report_title": "Panel Data Validation Report - Sprint 2 Test",
        "summary": summary,
        "model_info": model_info,
        "tests": tests,
        "recommendations": recommendations,
        "charts": None,  # No charts for this simple test
    }

    print(f"   ✅ Validation data prepared")
    print(f"   - Total tests: {summary['total_tests']}")
    print(f"   - Passed: {summary['total_passed']}")
    print(f"   - Failed: {summary['total_failed']}")
    print(f"   - Model: {model_info['model_type']}")
    print()

    # 3. Generate validation report
    print("3. Generating validation report...")
    try:
        html = report_mgr.generate_report(
            report_type="validation",
            template="validation/interactive/index.html",
            context=validation_data,
            embed_assets=True,
            include_plotly=True,
        )

        print(f"   ✅ Report generated successfully")
        print(f"   - HTML size: {len(html):,} characters")
        print()

        # 4. Save report
        print("4. Saving report to file...")
        output_path = Path("/home/guhaase/projetos/panelbox/sprint2_test_report.html")
        output_path.write_text(html, encoding="utf-8")
        file_size_kb = output_path.stat().st_size / 1024

        print(f"   ✅ Report saved: {output_path}")
        print(f"   - File size: {file_size_kb:.1f} KB")
        print()

        # 5. Validate HTML structure
        print("5. Validating HTML structure...")
        checks = {
            "DOCTYPE": html.startswith("<!DOCTYPE html>"),
            "html tag": "<html" in html,
            "head tag": "<head>" in html,
            "body tag": "<body>" in html,
            "CSS embedded": "css_inline" not in html
            and len(html) > 10000,  # CSS should be compiled
            "Summary section": "Validation Summary" in html,
            "Test results": "Hausman Test" in html,
            "Recommendations": "Address Heteroskedasticity" in html,
        }

        all_passed = all(checks.values())
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}: {'PASS' if passed else 'FAIL'}")

        print()

        # 6. Final result
        print("=" * 70)
        if all_passed:
            print("✅ SPRINT 2 TEST: PASSED")
            print()
            print("ReportManager Integration Complete:")
            print("  • TemplateManager ✅")
            print("  • CSSManager ✅")
            print("  • AssetManager ✅")
            print("  • Full report generation ✅")
            print("  • HTML validation ✅")
        else:
            print("❌ SPRINT 2 TEST: FAILED")
            print("Some validation checks did not pass.")
        print("=" * 70)

        return all_passed

    except Exception as e:
        print(f"   ❌ Error generating report: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback

        print()
        print("Traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_report_manager()
    exit(0 if success else 1)
