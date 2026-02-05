"""
Info command for PanelBox CLI.

This module implements the 'info' command which displays information about
datasets or saved results.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import panelbox as pb


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    """
    Add info command parser to subparsers.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparsers object from main parser

    Returns
    -------
    argparse.ArgumentParser
        The info command parser
    """
    parser = subparsers.add_parser(
        "info",
        help="Display information about data or results",
        description="Display information about CSV data or saved results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show data information
  panelbox info --data data.csv

  # Show data with entity/time structure
  panelbox info --data data.csv --entity firm --time year

  # Show results information
  panelbox info --results results.pkl
        """,
    )

    # Mutually exclusive group: either data or results
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--data", type=str, help="Path to CSV data file")

    group.add_argument("--results", type=str, help="Path to results file (.pkl)")

    # Optional for data info
    parser.add_argument("--entity", type=str, help="Entity column name (for panel structure info)")

    parser.add_argument("--time", type=str, help="Time column name (for panel structure info)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Print verbose output")

    # Set the function to call
    parser.set_defaults(func=execute)

    return parser


def print_data_info(
    filepath: str, entity_col: str = None, time_col: str = None, verbose: bool = False
) -> None:
    """
    Print information about CSV data.

    Parameters
    ----------
    filepath : str
        Path to CSV file
    entity_col : str, optional
        Entity column name
    time_col : str, optional
        Time column name
    verbose : bool, default=False
        Print verbose output
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Load data
    data = pd.read_csv(filepath)

    print("=" * 80)
    print(f"Data File Information: {filepath.name}")
    print("=" * 80)

    # Basic info
    print(f"\nFile path:        {filepath}")
    print(f"File size:        {filepath.stat().st_size:,} bytes")
    print(f"Number of rows:   {len(data):,}")
    print(f"Number of cols:   {len(data.columns)}")

    # Column info
    print(f"\nColumns ({len(data.columns)}):")
    for col in data.columns:
        dtype = data[col].dtype
        n_missing = data[col].isna().sum()
        n_unique = data[col].nunique()
        print(
            f"  - {col:<20s} {str(dtype):<10s} (unique: {n_unique:>6,}, missing: {n_missing:>6,})"
        )

    # Data types summary
    print(f"\nData Types:")
    dtype_counts = data.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns")

    # Panel structure if entity and time provided
    if entity_col and time_col:
        if entity_col not in data.columns:
            print(f"\nWarning: Entity column '{entity_col}' not found", file=sys.stderr)
        elif time_col not in data.columns:
            print(f"\nWarning: Time column '{time_col}' not found", file=sys.stderr)
        else:
            print(f"\n" + "-" * 80)
            print("Panel Structure:")
            print("-" * 80)

            n_entities = data[entity_col].nunique()
            n_periods = data[time_col].nunique()

            print(f"Entity column:    {entity_col}")
            print(f"Time column:      {time_col}")
            print(f"No. entities:     {n_entities:,}")
            print(f"No. periods:      {n_periods:,}")
            print(f"Max obs/entity:   {len(data) / n_entities:.1f}")

            # Check if balanced
            obs_per_entity = data.groupby(entity_col).size()
            is_balanced = (obs_per_entity == obs_per_entity.iloc[0]).all()

            if is_balanced:
                print(
                    f"Panel type:       Balanced (all entities have {obs_per_entity.iloc[0]} obs)"
                )
            else:
                print(f"Panel type:       Unbalanced")
                print(f"  Min obs:        {obs_per_entity.min()}")
                print(f"  Max obs:        {obs_per_entity.max()}")
                print(f"  Mean obs:       {obs_per_entity.mean():.1f}")

    # Summary statistics for numeric columns
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0 and verbose:
        print(f"\n" + "-" * 80)
        print("Summary Statistics (numeric columns):")
        print("-" * 80)
        print(data[numeric_cols].describe())

    print("=" * 80)


def print_results_info(filepath: str, verbose: bool = False) -> None:
    """
    Print information about saved results.

    Parameters
    ----------
    filepath : str
        Path to results file
    verbose : bool, default=False
        Print verbose output
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    # Load results
    results = pb.PanelResults.load(filepath)

    print("=" * 80)
    print(f"Results File Information: {filepath.name}")
    print("=" * 80)

    print(f"\nFile path:        {filepath}")
    print(f"File size:        {filepath.stat().st_size:,} bytes")

    print(f"\n" + "-" * 80)
    print("Model Information:")
    print("-" * 80)
    print(f"Model type:       {results.model_type}")
    print(f"Formula:          {results.formula}")
    print(f"Covariance type:  {results.cov_type}")

    print(f"\n" + "-" * 80)
    print("Sample Information:")
    print("-" * 80)
    print(f"Observations:     {results.nobs:,}")
    print(f"Entities:         {results.n_entities:,}")
    if results.n_periods is not None:
        print(f"Time periods:     {results.n_periods:,}")
    print(f"DF model:         {results.df_model}")
    print(f"DF residual:      {results.df_resid}")

    print(f"\n" + "-" * 80)
    print("Fit Statistics:")
    print("-" * 80)
    if not pd.isna(results.rsquared):
        print(f"R-squared:        {results.rsquared:.4f}")
    if not pd.isna(results.rsquared_adj):
        print(f"Adj. R-squared:   {results.rsquared_adj:.4f}")
    if not pd.isna(results.rsquared_within):
        print(f"R² (within):      {results.rsquared_within:.4f}")
    if not pd.isna(results.rsquared_between):
        print(f"R² (between):     {results.rsquared_between:.4f}")
    if not pd.isna(results.rsquared_overall):
        print(f"R² (overall):     {results.rsquared_overall:.4f}")

    print(f"\n" + "-" * 80)
    print(f"Parameters ({len(results.params)}):")
    print("-" * 80)
    for var in results.params.index:
        coef = results.params[var]
        se = results.std_errors[var]
        t = results.tvalues[var]
        p = results.pvalues[var]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var:<15s} coef={coef:>8.4f}  se={se:>6.4f}  t={t:>6.2f}  p={p:>6.4f} {sig}")

    if verbose:
        print(f"\n" + "=" * 80)
        print("Full Summary:")
        print("=" * 80)
        print(results.summary())

    print("=" * 80)


def execute(args: argparse.Namespace) -> int:
    """
    Execute the info command.

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
        if args.data:
            print_data_info(args.data, args.entity, args.time, args.verbose)
        elif args.results:
            print_results_info(args.results, args.verbose)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
