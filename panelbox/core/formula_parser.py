"""
FormulaParser - Parser for R-style formulas for panel models.

This module provides formula parsing functionality similar to R's formula syntax,
adapted for panel data econometrics.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
import patsy
import pandas as pd
import numpy as np


class FormulaParser:
    """
    Parser for R-style formulas with panel-specific extensions.

    Supports standard R formula syntax:
    - Basic: "y ~ x1 + x2"
    - Interactions: "y ~ x1 * x2" (expands to x1 + x2 + x1:x2)
    - Transformations: "y ~ log(x1) + I(x2**2)"

    For GMM models (future implementation), supports:
    - Lags: "y ~ L(y, 1:2)" for lagged variables
    - Instruments: "y ~ x1 + x2 | gmm(y, 2:4) + iv(x1)"

    Parameters
    ----------
    formula : str
        Formula string in R-style syntax

    Attributes
    ----------
    formula : str
        Original formula string
    dependent : str
        Name of dependent variable
    regressors : List[str]
        List of regressor variable names
    has_intercept : bool
        Whether model includes intercept
    has_instruments : bool
        Whether formula includes instrument specification (for GMM)

    Examples
    --------
    >>> parser = FormulaParser("y ~ x1 + x2")
    >>> parser.parse()
    >>> print(parser.dependent)
    'y'
    >>> print(parser.regressors)
    ['x1', 'x2']
    """

    def __init__(self, formula: str):
        if not isinstance(formula, str):
            raise TypeError("formula must be a string")

        if '~' not in formula:
            raise ValueError("formula must contain '~' separating dependent and independent variables")

        self.formula = formula.strip()
        self.dependent: Optional[str] = None
        self.regressors: List[str] = []
        self.has_intercept: bool = True
        self.has_instruments: bool = False
        self._instrument_spec: Optional[str] = None
        self._parsed: bool = False

    def parse(self) -> 'FormulaParser':
        """
        Parse the formula string.

        Returns
        -------
        FormulaParser
            Self (for method chaining)

        Examples
        --------
        >>> parser = FormulaParser("y ~ x1 + x2").parse()
        """
        # Split on ~ to get LHS and RHS
        parts = self.formula.split('~')
        if len(parts) != 2:
            raise ValueError("formula must have exactly one '~'")

        lhs, rhs = parts[0].strip(), parts[1].strip()

        # Parse dependent variable (LHS)
        self.dependent = lhs

        # Check for instruments (for GMM - future)
        if '|' in rhs:
            self.has_instruments = True
            rhs_parts = rhs.split('|')
            rhs = rhs_parts[0].strip()
            self._instrument_spec = rhs_parts[1].strip()

        # Check for no intercept
        if '-1' in rhs or '- 1' in rhs or '+0' in rhs or '+ 0' in rhs:
            self.has_intercept = False
            # Remove the -1 or +0 from RHS
            rhs = re.sub(r'[+-]\s*[01]', '', rhs)

        # Store RHS for later use with patsy
        self._rhs = rhs.strip()

        # Parse regressors (will be expanded by patsy later)
        # For now, just store the basic variable names
        self.regressors = self._extract_variable_names(rhs)

        self._parsed = True
        return self

    def _extract_variable_names(self, rhs: str) -> List[str]:
        """
        Extract basic variable names from RHS.

        This is a simple extraction that doesn't handle all transformations.
        Patsy will handle the full parsing when building design matrices.

        Parameters
        ----------
        rhs : str
            Right-hand side of formula

        Returns
        -------
        List[str]
            List of variable names
        """
        # Split on + but respect parentheses
        # This is a simplified version - patsy will do the heavy lifting
        terms = re.split(r'\s*\+\s*', rhs)
        variables = []

        for term in terms:
            term = term.strip()
            if not term or term in ['-1', '- 1', '0', '1']:
                continue

            # Extract variable names from term
            # Handle simple cases: x, log(x), I(x**2), x:y, x*y
            # Check for function calls first (before checking for : or *)
            func_match = re.match(r'(?:\w+\.)*(\w+)\((.*)\)', term)
            if func_match:
                # This is a function call - extract variable from it
                var = self._extract_var_from_term(term)
                if var and var not in variables:
                    variables.append(var)
            elif ':' in term:
                # Interaction term
                parts = term.split(':')
                for part in parts:
                    var = self._extract_var_from_term(part.strip())
                    if var and var not in variables:
                        variables.append(var)
            elif '*' in term:
                # Interaction with expansion (not inside parentheses)
                parts = term.split('*')
                for part in parts:
                    var = self._extract_var_from_term(part.strip())
                    if var and var not in variables:
                        variables.append(var)
            else:
                var = self._extract_var_from_term(term)
                if var and var not in variables:
                    variables.append(var)

        return variables

    def _extract_var_from_term(self, term: str) -> Optional[str]:
        """
        Extract variable name from a single term.

        Parameters
        ----------
        term : str
            Single term from formula

        Returns
        -------
        Optional[str]
            Variable name, or None if not extractable
        """
        term = term.strip()

        # Function call like log(x), np.log(x), I(x**2)
        func_match = re.match(r'(?:\w+\.)*(\w+)\((.*)\)', term)
        if func_match:
            func_name = func_match.group(1)
            arg = func_match.group(2)

            # For I(), extract variable from expression
            if func_name == 'I':
                # Extract variable names from expression
                var_matches = re.findall(r'\b([a-zA-Z_]\w*)\b', arg)
                return var_matches[0] if var_matches else None
            else:
                # For other functions, return argument if it's a variable name
                if re.match(r'^[a-zA-Z_]\w*$', arg.strip()):
                    return arg.strip()
                return None

        # Simple variable name
        if re.match(r'^[a-zA-Z_]\w*$', term):
            return term

        return None

    def build_design_matrices(
        self,
        data: pd.DataFrame,
        return_type: str = 'dataframe'
    ) -> Tuple[Any, Any]:
        """
        Build design matrices using patsy.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing variables referenced in formula
        return_type : str, default='dataframe'
            Return type: 'dataframe', 'matrix', or 'array'
            - 'dataframe': returns pandas DataFrames
            - 'matrix': returns patsy DesignMatrix objects
            - 'array': returns numpy arrays

        Returns
        -------
        y : DataFrame, DesignMatrix, or ndarray
            Dependent variable
        X : DataFrame, DesignMatrix, or ndarray
            Design matrix for independent variables

        Examples
        --------
        >>> parser = FormulaParser("y ~ x1 + x2").parse()
        >>> y, X = parser.build_design_matrices(data)
        """
        if not self._parsed:
            self.parse()

        # Build formula for patsy
        # Patsy will handle intercept automatically unless we specify -1
        if self.has_intercept:
            patsy_formula = f"{self.dependent} ~ {self._rhs}"
        else:
            patsy_formula = f"{self.dependent} ~ {self._rhs} - 1"

        # Use patsy to build design matrices
        y_mat, X_mat = patsy.dmatrices(patsy_formula, data, return_type='dataframe')

        if return_type == 'dataframe':
            # y_mat is a DataFrame with one column, extract as Series
            y = y_mat.iloc[:, 0]
            X = X_mat
        elif return_type == 'matrix':
            # Return patsy DesignMatrix objects
            y, X = patsy.dmatrices(patsy_formula, data, return_type='matrix')
        elif return_type == 'array':
            # Return numpy arrays
            y, X = patsy.dmatrices(patsy_formula, data, return_type='dataframe')
            y = y.values.ravel()
            X = X.values
        else:
            raise ValueError("return_type must be 'dataframe', 'matrix', or 'array'")

        return y, X

    def get_variable_names(self, data: pd.DataFrame) -> List[str]:
        """
        Get the names of variables in the design matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing variables

        Returns
        -------
        List[str]
            List of column names in design matrix

        Examples
        --------
        >>> parser = FormulaParser("y ~ x1 + x2").parse()
        >>> var_names = parser.get_variable_names(data)
        >>> print(var_names)
        ['Intercept', 'x1', 'x2']
        """
        _, X = self.build_design_matrices(data, return_type='dataframe')
        return list(X.columns)

    def __repr__(self) -> str:
        """String representation."""
        if self._parsed:
            return f"FormulaParser('{self.formula}', dependent='{self.dependent}', k={len(self.regressors)})"
        else:
            return f"FormulaParser('{self.formula}', unparsed)"


def parse_formula(formula: str) -> FormulaParser:
    """
    Convenience function to parse a formula.

    Parameters
    ----------
    formula : str
        Formula string

    Returns
    -------
    FormulaParser
        Parsed formula object

    Examples
    --------
    >>> parser = parse_formula("y ~ x1 + x2")
    >>> print(parser.dependent)
    'y'
    """
    return FormulaParser(formula).parse()
