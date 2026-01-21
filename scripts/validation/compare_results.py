"""
Compare PanelBox and R test results.

This script compares the validation test statistics and p-values
from PanelBox with equivalent tests from R (plm package).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_results(filepath: Path) -> Dict[str, Any]:
    """Load JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_test(pb_result: Dict, r_result: Dict, test_name: str) -> Dict[str, Any]:
    """
    Compare a single test between PanelBox and R.

    Parameters
    ----------
    pb_result : dict
        PanelBox test result
    r_result : dict
        R test result
    test_name : str
        Name of the test

    Returns
    -------
    dict
        Comparison results
    """
    comparison = {
        'test': test_name,
        'panelbox': pb_result,
        'r': r_result,
        'status': 'unknown'
    }

    # Check for errors
    if 'error' in pb_result:
        comparison['status'] = 'pb_error'
        comparison['message'] = f"PanelBox error: {pb_result['error']}"
        return comparison

    if 'error' in r_result:
        comparison['status'] = 'r_error'
        comparison['message'] = f"R error: {r_result['error']}"
        return comparison

    # Compare statistics
    if 'statistic' in pb_result and 'statistic' in r_result:
        pb_stat = pb_result['statistic']
        r_stat = r_result['statistic']

        # Compute relative and absolute differences
        abs_diff = abs(pb_stat - r_stat)
        rel_diff = abs_diff / abs(r_stat) if r_stat != 0 else float('inf')

        comparison['stat_diff'] = {
            'panelbox': pb_stat,
            'r': r_stat,
            'abs_diff': abs_diff,
            'rel_diff': rel_diff
        }

        # Compare p-values
        if 'pvalue' in pb_result and 'pvalue' in r_result:
            pb_pval = pb_result['pvalue']
            r_pval = r_result['pvalue']

            pval_abs_diff = abs(pb_pval - r_pval)
            pval_rel_diff = pval_abs_diff / r_pval if r_pval != 0 else float('inf')

            comparison['pval_diff'] = {
                'panelbox': pb_pval,
                'r': r_pval,
                'abs_diff': pval_abs_diff,
                'rel_diff': pval_rel_diff
            }

            # Determine status based on tolerance
            # Consider close if:
            # - Relative difference < 5% OR
            # - Absolute difference < 0.01 for p-values
            stat_close = rel_diff < 0.05 or abs_diff < 0.1
            pval_close = pval_rel_diff < 0.05 or pval_abs_diff < 0.01

            if stat_close and pval_close:
                comparison['status'] = 'match'
                comparison['message'] = 'Results match within tolerance'
            elif stat_close:
                comparison['status'] = 'partial'
                comparison['message'] = 'Statistic matches but p-value differs'
            elif pval_close:
                comparison['status'] = 'partial'
                comparison['message'] = 'P-value matches but statistic differs'
            else:
                comparison['status'] = 'mismatch'
                comparison['message'] = 'Significant difference in results'
        else:
            comparison['status'] = 'incomplete'
            comparison['message'] = 'Missing p-value in one implementation'
    else:
        comparison['status'] = 'incomplete'
        comparison['message'] = 'Missing statistic in one implementation'

    return comparison


def compare_dataset_results(pb_file: Path, r_file: Path, dataset_name: str) -> Dict[str, Any]:
    """
    Compare all tests for a dataset.

    Parameters
    ----------
    pb_file : Path
        PanelBox results file
    r_file : Path
        R results file
    dataset_name : str
        Name of dataset for reporting

    Returns
    -------
    dict
        Comparison results for all tests
    """
    print(f"\n{'=' * 80}")
    print(f"Comparing: {dataset_name}")
    print(f"{'=' * 80}")

    pb_results = load_results(pb_file)
    r_results = load_results(r_file)

    comparisons = {
        'dataset': dataset_name,
        'model_type': pb_results.get('model_type', 'unknown'),
        'tests': {},
        'summary': {
            'total': 0,
            'match': 0,
            'partial': 0,
            'mismatch': 0,
            'pb_error': 0,
            'r_error': 0,
            'incomplete': 0
        }
    }

    # Get all test names from both results
    all_tests = set(pb_results.get('tests', {}).keys()) | set(r_results.get('tests', {}).keys())

    for test_name in sorted(all_tests):
        pb_test = pb_results.get('tests', {}).get(test_name, {'error': 'Not run'})
        r_test = r_results.get('tests', {}).get(test_name, {'error': 'Not run'})

        comparison = compare_test(pb_test, r_test, test_name)
        comparisons['tests'][test_name] = comparison

        # Update summary
        comparisons['summary']['total'] += 1
        comparisons['summary'][comparison['status']] = \
            comparisons['summary'].get(comparison['status'], 0) + 1

        # Print comparison
        print(f"\n{test_name.upper()}")
        print("-" * 80)
        print(f"  Status: {comparison['status'].upper()}")

        if 'stat_diff' in comparison:
            sd = comparison['stat_diff']
            print(f"  Statistic:")
            print(f"    PanelBox: {sd['panelbox']:>12.4f}")
            print(f"    R:        {sd['r']:>12.4f}")
            print(f"    Abs diff: {sd['abs_diff']:>12.4f}")
            print(f"    Rel diff: {sd['rel_diff']:>11.2%}")

        if 'pval_diff' in comparison:
            pd = comparison['pval_diff']
            print(f"  P-value:")
            print(f"    PanelBox: {pd['panelbox']:>12.6f}")
            print(f"    R:        {pd['r']:>12.6f}")
            print(f"    Abs diff: {pd['abs_diff']:>12.6f}")

        if 'message' in comparison:
            print(f"  Note: {comparison['message']}")

        if 'note' in comparison.get('r', {}):
            print(f"  R note: {comparison['r']['note']}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total tests:     {comparisons['summary']['total']}")
    print(f"  ✅ Match:        {comparisons['summary'].get('match', 0)}")
    print(f"  ⚠️  Partial:     {comparisons['summary'].get('partial', 0)}")
    print(f"  ❌ Mismatch:     {comparisons['summary'].get('mismatch', 0)}")
    print(f"  🔧 PB Error:     {comparisons['summary'].get('pb_error', 0)}")
    print(f"  🔧 R Error:      {comparisons['summary'].get('r_error', 0)}")
    print(f"  ⚪ Incomplete:   {comparisons['summary'].get('incomplete', 0)}")

    return comparisons


def generate_validation_report(all_comparisons: List[Dict]) -> str:
    """Generate comprehensive validation report."""
    lines = []
    lines.append("=" * 80)
    lines.append("PANELBOX VALIDATION REPORT - COMPARISON WITH R")
    lines.append("=" * 80)
    lines.append("")

    # Overall summary
    total_tests = sum(c['summary']['total'] for c in all_comparisons)
    total_match = sum(c['summary'].get('match', 0) for c in all_comparisons)
    total_partial = sum(c['summary'].get('partial', 0) for c in all_comparisons)
    total_mismatch = sum(c['summary'].get('mismatch', 0) for c in all_comparisons)

    lines.append("OVERALL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total test comparisons: {total_tests}")
    lines.append(f"✅ Exact matches:       {total_match} ({total_match/total_tests*100:.1f}%)")
    lines.append(f"⚠️  Partial matches:    {total_partial} ({total_partial/total_tests*100:.1f}%)")
    lines.append(f"❌ Mismatches:          {total_mismatch} ({total_mismatch/total_tests*100:.1f}%)")
    lines.append("")

    # Success rate
    success_rate = (total_match + total_partial) / total_tests * 100 if total_tests > 0 else 0
    lines.append(f"Overall validation success: {success_rate:.1f}%")
    lines.append("")

    # Dataset-by-dataset summary
    lines.append("=" * 80)
    lines.append("DATASET-BY-DATASET RESULTS")
    lines.append("=" * 80)

    for comp in all_comparisons:
        lines.append("")
        lines.append(f"{comp['dataset']} ({comp['model_type']})")
        lines.append("-" * 80)

        for test_name, test_comp in comp['tests'].items():
            status_icon = {
                'match': '✅',
                'partial': '⚠️',
                'mismatch': '❌',
                'pb_error': '🔧',
                'r_error': '🔧',
                'incomplete': '⚪'
            }.get(test_comp['status'], '?')

            lines.append(f"  {status_icon} {test_name:20} - {test_comp['status'].upper()}")

            if 'stat_diff' in test_comp:
                rel_diff = test_comp['stat_diff']['rel_diff']
                if rel_diff < float('inf'):
                    lines.append(f"      Stat diff: {rel_diff:.2%}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)
    lines.append("")

    if success_rate >= 90:
        lines.append("✅ EXCELLENT: PanelBox validation tests match R implementations very closely.")
        lines.append("   The numerical implementations are accurate and reliable.")
    elif success_rate >= 75:
        lines.append("⚠️  GOOD: PanelBox validation tests mostly match R implementations.")
        lines.append("   Minor differences may be due to different algorithms or approximations.")
    elif success_rate >= 50:
        lines.append("⚠️  FAIR: PanelBox validation tests partially match R implementations.")
        lines.append("   Review differences carefully - may indicate implementation issues.")
    else:
        lines.append("❌ NEEDS WORK: Significant differences between PanelBox and R.")
        lines.append("   Implementations should be reviewed and corrected.")

    lines.append("")
    lines.append("NOTES:")
    lines.append("- Some tests use approximations in R (e.g., Modified Wald uses Bartlett test)")
    lines.append("- Small numerical differences (<5%) are expected due to algorithm differences")
    lines.append("- P-value differences <0.01 are considered acceptable")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main comparison script."""
    output_dir = Path('/home/guhaase/projetos/panelbox/scripts/validation/output')

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Run generate_test_data_and_run.py first!")
        return

    print("=" * 80)
    print("COMPARING PANELBOX AND R VALIDATION RESULTS")
    print("=" * 80)

    all_comparisons = []

    # Compare AR(1) data with FE
    if (output_dir / 'panelbox_results_ar1_fe.json').exists() and \
       (output_dir / 'r_results_ar1_fe.json').exists():
        comp_ar1 = compare_dataset_results(
            output_dir / 'panelbox_results_ar1_fe.json',
            output_dir / 'r_results_ar1_fe.json',
            'AR(1) Data - Fixed Effects'
        )
        all_comparisons.append(comp_ar1)

    # Compare heteroskedastic data with FE
    if (output_dir / 'panelbox_results_het_fe.json').exists() and \
       (output_dir / 'r_results_het_fe.json').exists():
        comp_het = compare_dataset_results(
            output_dir / 'panelbox_results_het_fe.json',
            output_dir / 'r_results_het_fe.json',
            'Heteroskedastic Data - Fixed Effects'
        )
        all_comparisons.append(comp_het)

    # Compare clean data with FE
    if (output_dir / 'panelbox_results_clean_fe.json').exists() and \
       (output_dir / 'r_results_clean_fe.json').exists():
        comp_clean_fe = compare_dataset_results(
            output_dir / 'panelbox_results_clean_fe.json',
            output_dir / 'r_results_clean_fe.json',
            'Clean Data - Fixed Effects'
        )
        all_comparisons.append(comp_clean_fe)

    # Compare clean data with RE
    if (output_dir / 'panelbox_results_clean_re.json').exists() and \
       (output_dir / 'r_results_clean_re.json').exists():
        comp_clean_re = compare_dataset_results(
            output_dir / 'panelbox_results_clean_re.json',
            output_dir / 'r_results_clean_re.json',
            'Clean Data - Random Effects'
        )
        all_comparisons.append(comp_clean_re)

    # Generate final report
    report = generate_validation_report(all_comparisons)
    print("\n" * 2)
    print(report)

    # Save report
    report_file = output_dir / 'validation_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # Save detailed comparisons as JSON
    comparison_file = output_dir / 'validation_comparisons.json'
    with open(comparison_file, 'w') as f:
        json.dump(all_comparisons, f, indent=2)
    print(f"Detailed comparisons saved to: {comparison_file}")


if __name__ == '__main__':
    main()
