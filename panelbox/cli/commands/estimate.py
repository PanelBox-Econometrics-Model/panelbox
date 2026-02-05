"""
Estimate command for PanelBox CLI.

This module implements the 'estimate' command which allows users to estimate
panel data models from the command line.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import panelbox as pb

# Model mapping
MODEL_MAP = {
    "pooled": pb.PooledOLS,
    "fe": pb.FixedEffects,
    "fixed": pb.FixedEffects,
    "re": pb.RandomEffects,
    "random": pb.RandomEffects,
    "between": pb.BetweenEstimator,
    "fd": pb.FirstDifferenceEstimator,
    "first_diff": pb.FirstDifferenceEstimator,
    "diff_gmm": pb.DifferenceGMM,
    "sys_gmm": pb.SystemGMM,
}

# Valid covariance types
COV_TYPES = [
    "nonrobust",
    "robust",
    "hc0",
    "hc1",
    "hc2",
    "hc3",
    "clustered",
    "twoway",
    "driscoll_kraay",
    "newey_west",
    "pcse",
]


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """
    Add estimate command parser to subparsers.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object from main parser

    Returns
    -------
    argparse.ArgumentParser
        The estimate command parser
    """
    parser = subparsers.add_parser(
        "estimate",
        help="Estimate a panel data model",
        description="Estimate a panel data model from CSV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fixed Effects with robust SE
  panelbox estimate --data data.csv --model fe \\
      --formula "y ~ x1 + x2" --entity firm --time year \\
      --cov-type robust --output fe_results.pkl

  # Pooled OLS with clustered SE
  panelbox estimate --data data.csv --model pooled \\
      --formula "invest ~ value + capital" \\
      --entity firm --time year \\
      --cov-type clustered --output pooled.pkl

  # Between Estimator
  panelbox estimate --data data.csv --model between \\
      --formula "y ~ x1 + x2" --entity firm --time year \\
      --output between.pkl

  # System GMM (dynamic panel)
  panelbox estimate --data data.csv --model sys_gmm \\
      --formula "y ~ L1.y + x1 + x2" --entity firm --time year \\
      --output gmm.pkl
        """,
    )

    # Required arguments
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data file")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Model type to estimate",
    )

    parser.add_argument(
        "--formula", type=str, required=True, help='Model formula (e.g., "y ~ x1 + x2")'
    )

    parser.add_argument("--entity", type=str, required=True, help="Name of entity/panel column")

    parser.add_argument("--time", type=str, required=True, help="Name of time column")

    # Optional arguments
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Path to save results (default: no save)"
    )

    parser.add_argument(
        "--cov-type",
        type=str,
        default="nonrobust",
        choices=COV_TYPES,
        help="Covariance estimator type (default: nonrobust)",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="pickle",
        choices=["pickle", "json"],
        help="Output format (default: pickle)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

    parser.add_argument("--no-summary", action="store_true", help="Do not print summary table")

    # Set the function to call
    parser.set_defaults(func=execute)

    return parser


def load_data(filepath: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load data from CSV file.

    Parameters
    ----------
    filepath : str
        Path to CSV file
    verbose : bool, default=False
        Print verbose output

    Returns
    -------
    pd.DataFrame
        Loaded data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    if verbose:
        print(f"Loading data from: {filepath}")

    data = pd.read_csv(filepath)

    if verbose:
        print(f"  Loaded {len(data)} observations")
        print(f"  Columns: {', '.join(data.columns)}")

    return data


def execute(args: argparse.Namespace) -> int:
    """
    Execute the estimate command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Load data
        data = load_data(args.data, args.verbose)

        # Check required columns
        if args.entity not in data.columns:
            print(f"Error: Entity column '{args.entity}' not found in data", file=sys.stderr)
            print(f"Available columns: {', '.join(data.columns)}", file=sys.stderr)
            return 1

        if args.time not in data.columns:
            print(f"Error: Time column '{args.time}' not found in data", file=sys.stderr)
            print(f"Available columns: {', '.join(data.columns)}", file=sys.stderr)
            return 1

        # Get model class
        model_class = MODEL_MAP[args.model]

        if args.verbose:
            print(f"\nEstimating {model_class.__name__} model...")
            print(f"  Formula: {args.formula}")
            print(f"  Entity: {args.entity}")
            print(f"  Time: {args.time}")
            print(f"  Covariance type: {args.cov_type}")

        # Create model
        model = model_class(
            formula=args.formula, data=data, entity_col=args.entity, time_col=args.time
        )

        # Fit model
        results = model.fit(cov_type=args.cov_type)

        if args.verbose:
            print("\n✓ Estimation completed successfully")

        # Print summary
        if not args.no_summary:
            print("\n" + "=" * 80)
            print(results.summary())
            print("=" * 80)

        # Save results if output specified
        if args.output:
            output_path = Path(args.output)

            if args.verbose:
                print(f"\nSaving results to: {output_path}")

            results.save(output_path, format=args.format)

            if args.verbose:
                print("✓ Results saved successfully")

        return 0

    except Exception as e:
        print(f"Error during estimation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
