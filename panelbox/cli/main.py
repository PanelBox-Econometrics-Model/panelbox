"""
Main entry point for PanelBox CLI.

This module provides the command-line interface for PanelBox, allowing users
to estimate panel data models, run diagnostics, and generate reports from
the command line.
"""

import argparse
import sys
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser for PanelBox CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all subcommands
    """
    parser = argparse.ArgumentParser(
        prog="panelbox",
        description="PanelBox: Panel Data Econometrics in Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate a Fixed Effects model
  panelbox estimate --data data.csv --model fe \\
      --formula "y ~ x1 + x2" --entity firm --time year \\
      --output results.pkl

  # Estimate with robust standard errors
  panelbox estimate --data data.csv --model pooled \\
      --formula "y ~ x1 + x2" --entity firm --time year \\
      --cov-type robust --output results.pkl

  # Get information about a dataset
  panelbox info --data data.csv

For more information, visit: https://github.com/yourusername/panelbox
        """,
    )

    parser.add_argument("--version", action="version", version="PanelBox 0.2.0")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest="command",
        title="Available commands",
        description="Use panelbox <command> --help for more information",
        help="Command to execute",
    )

    # Import command modules
    from panelbox.cli.commands import estimate, info

    # Add estimate command
    estimate.add_parser(subparsers)

    # Add info command
    info.add_parser(subparsers)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for PanelBox CLI.

    Parameters
    ----------
    argv : list of str, optional
        Command line arguments. If None, uses sys.argv[1:]

    Returns
    -------
    int
        Exit code (0 for success, non-zero for error)
    """
    # Create parser
    parser = create_parser()

    # Parse arguments
    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)

    # If no command provided, print help
    if not args.command:
        parser.print_help()
        return 0

    # Execute the command
    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        if hasattr(args, "verbose") and args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
