"""
Export Charts Example

Demonstrates how to export PanelBox charts to various image formats
(PNG, SVG, PDF) using the built-in export functionality.
"""

from pathlib import Path
import numpy as np
from panelbox.visualization import (
    create_validation_charts,
    create_residual_diagnostics,
    create_comparison_charts,
    export_chart,
    export_charts,
    export_charts_multiple_formats,
)


def create_sample_validation_data():
    """Create sample validation data for examples."""
    return {
        'tests': [
            {'name': 'Hausman Test', 'statistic': 12.34, 'pvalue': 0.015,
             'result': 'Reject H0', 'category': 'specification'},
            {'name': 'Breusch-Pagan', 'statistic': 45.67, 'pvalue': 0.001,
             'result': 'Reject H0', 'category': 'heteroskedasticity'},
            {'name': 'Wooldridge AR', 'statistic': 8.91, 'pvalue': 0.045,
             'result': 'Reject H0', 'category': 'serial_correlation'},
            {'name': 'Pesaran CD', 'statistic': 2.34, 'pvalue': 0.112,
             'result': 'Fail to Reject H0', 'category': 'cross_sectional'},
        ],
        'summary': {
            'total_tests': 4,
            'tests_passed': 1,
            'tests_failed': 3,
            'pass_rate': 25.0
        },
        'categories': {
            'specification': [{'name': 'Hausman Test', 'statistic': 12.34, 'pvalue': 0.015}],
            'heteroskedasticity': [{'name': 'Breusch-Pagan', 'statistic': 45.67, 'pvalue': 0.001}],
            'serial_correlation': [{'name': 'Wooldridge AR', 'statistic': 8.91, 'pvalue': 0.045}],
            'cross_sectional': [{'name': 'Pesaran CD', 'statistic': 2.34, 'pvalue': 0.112}],
        }
    }


def example_single_chart_export():
    """Example 1: Export a single chart to different formats."""
    print("=" * 80)
    print("Example 1: Single Chart Export")
    print("=" * 80)
    print()

    # Create validation charts
    data = create_sample_validation_data()
    charts = create_validation_charts(data, include_html=False)

    # Get test overview chart
    test_overview = charts['test_overview']

    # Export as PNG
    print("Exporting test_overview chart...")
    export_chart(test_overview, 'output/exports/test_overview.png', width=1200, height=800)
    print("✓ PNG: output/exports/test_overview.png")

    # Export as SVG (vector format - scalable)
    export_chart(test_overview, 'output/exports/test_overview.svg', width=1200, height=800)
    print("✓ SVG: output/exports/test_overview.svg")

    # Export as PDF
    export_chart(test_overview, 'output/exports/test_overview.pdf', width=1200, height=800)
    print("✓ PDF: output/exports/test_overview.pdf")

    # Export high-resolution PNG (2x for retina displays)
    export_chart(test_overview, 'output/exports/test_overview_2x.png', scale=2.0)
    print("✓ High-res PNG (2x): output/exports/test_overview_2x.png")

    print()


def example_batch_export():
    """Example 2: Batch export all validation charts."""
    print("=" * 80)
    print("Example 2: Batch Export All Charts")
    print("=" * 80)
    print()

    # Create validation charts
    data = create_sample_validation_data()
    charts = create_validation_charts(data, include_html=False)

    print(f"Created {len(charts)} validation charts")
    print(f"Chart names: {', '.join(charts.keys())}")
    print()

    # Export all charts as PNG
    print("Exporting all charts as PNG...")
    paths = export_charts(
        charts,
        output_dir='output/exports/validation',
        format='png',
        width=1200,
        height=800,
        prefix='validation_'
    )

    for chart_name, path in paths.items():
        print(f"  ✓ {chart_name}: {path}")

    print()


def example_multiple_formats():
    """Example 3: Export charts in multiple formats at once."""
    print("=" * 80)
    print("Example 3: Export Multiple Formats Simultaneously")
    print("=" * 80)
    print()

    # Create validation charts
    data = create_sample_validation_data()
    charts = create_validation_charts(data, include_html=False)

    # Export in PNG, SVG, and PDF formats
    print("Exporting charts in multiple formats...")
    all_paths = export_charts_multiple_formats(
        charts,
        output_dir='output/exports/multi_format',
        formats=['png', 'svg', 'pdf'],
        width=1200,
        height=800
    )

    print()
    for format, paths in all_paths.items():
        print(f"{format.upper()} exports:")
        for chart_name, path in paths.items():
            print(f"  ✓ {chart_name}: {path}")
        print()


def example_custom_sizes():
    """Example 4: Export with custom sizes for different use cases."""
    print("=" * 80)
    print("Example 4: Custom Sizes for Different Use Cases")
    print("=" * 80)
    print()

    # Create validation charts
    data = create_sample_validation_data()
    charts = create_validation_charts(data, include_html=False)

    test_overview = charts['test_overview']

    # Presentation size (16:9 aspect ratio)
    print("Presentation size (1920x1080)...")
    export_chart(
        test_overview,
        'output/exports/sizes/presentation_16_9.png',
        width=1920,
        height=1080
    )
    print("✓ output/exports/sizes/presentation_16_9.png")

    # Publication size (high DPI)
    print("Publication size (high-res)...")
    export_chart(
        test_overview,
        'output/exports/sizes/publication_hires.png',
        width=2400,
        height=1600,
        scale=2.0
    )
    print("✓ output/exports/sizes/publication_hires.png")

    # Social media square
    print("Social media square (1200x1200)...")
    export_chart(
        test_overview,
        'output/exports/sizes/social_square.png',
        width=1200,
        height=1200
    )
    print("✓ output/exports/sizes/social_square.png")

    # Thumbnail
    print("Thumbnail (400x300)...")
    export_chart(
        test_overview,
        'output/exports/sizes/thumbnail.png',
        width=400,
        height=300
    )
    print("✓ output/exports/sizes/thumbnail.png")

    print()


def example_using_methods():
    """Example 5: Using chart object methods directly."""
    print("=" * 80)
    print("Example 5: Using Chart Object Methods Directly")
    print("=" * 80)
    print()

    # Create validation charts
    data = create_sample_validation_data()
    charts = create_validation_charts(data, include_html=False)

    test_overview = charts['test_overview']

    # Method 1: save_image()
    print("Using save_image() method...")
    test_overview.save_image('output/exports/methods/test1.png', width=1200)
    print("✓ output/exports/methods/test1.png")

    # Method 2: to_png() + manual save
    print("Using to_png() method...")
    png_bytes = test_overview.to_png(width=1200, height=800)
    Path('output/exports/methods').mkdir(parents=True, exist_ok=True)
    with open('output/exports/methods/test2.png', 'wb') as f:
        f.write(png_bytes)
    print("✓ output/exports/methods/test2.png")

    # Method 3: to_svg()
    print("Using to_svg() method...")
    svg_bytes = test_overview.to_svg(width=1200, height=800)
    with open('output/exports/methods/test3.svg', 'wb') as f:
        f.write(svg_bytes)
    print("✓ output/exports/methods/test3.svg")

    # Method 4: to_pdf()
    print("Using to_pdf() method...")
    pdf_bytes = test_overview.to_pdf(width=1200, height=800)
    with open('output/exports/methods/test4.pdf', 'wb') as f:
        f.write(pdf_bytes)
    print("✓ output/exports/methods/test4.pdf")

    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "PanelBox Chart Export Examples" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Run examples
    example_single_chart_export()
    example_batch_export()
    example_multiple_formats()
    example_custom_sizes()
    example_using_methods()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("✅ All examples completed successfully!")
    print()
    print("Exported files can be found in:")
    print("  - output/exports/")
    print("  - output/exports/validation/")
    print("  - output/exports/multi_format/")
    print("  - output/exports/sizes/")
    print("  - output/exports/methods/")
    print()
    print("Supported Formats:")
    print("  • PNG  - Raster image (default)")
    print("  • SVG  - Vector image (scalable)")
    print("  • PDF  - Portable document format")
    print("  • JPEG - Compressed raster image")
    print("  • WEBP - Modern web format")
    print()
    print("Key Features:")
    print("  • High-resolution exports (scale parameter)")
    print("  • Custom dimensions (width/height)")
    print("  • Batch export multiple charts")
    print("  • Multiple formats simultaneously")
    print("  • Direct method access on chart objects")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
