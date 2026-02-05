"""
Robustness checks for panel data models.

Provides tools to test robustness of results across different
specifications, samples, and estimators.
"""

from typing import List, Optional

import pandas as pd

from panelbox.core.results import PanelResults


class RobustnessChecker:
    """
    Robustness checking framework for panel data models.

    Parameters
    ----------
    results : PanelResults
        Base model results
    verbose : bool
        Print progress

    Examples
    --------
    >>> checker = pb.RobustnessChecker(results)
    >>> alt_specs = checker.check_alternative_specs([
    ...     "y ~ x1",
    ...     "y ~ x1 + x2",
    ...     "y ~ x1 + x2 + x3"
    ... ])
    >>> print(checker.generate_robustness_table(alt_specs))
    """

    def __init__(self, results: PanelResults, verbose: bool = True):
        self.results = results
        self.verbose = verbose
        self.model = results._model
        assert self.model is not None, "Results must have a model reference for robustness checks"
        self.data = self.model.data.data
        self.entity_col = self.model.data.entity_col
        self.time_col = self.model.data.time_col

    def check_alternative_specs(
        self, formulas: List[str], model_type: Optional[str] = None
    ) -> List[PanelResults]:
        """
        Test alternative specifications.

        Parameters
        ----------
        formulas : list of str
            Alternative model formulas
        model_type : str, optional
            Model type to use. If None, uses same as base model

        Returns
        -------
        results_list : list of PanelResults
            Results for each specification
        """
        results_list = []

        if model_type is None:
            model_class = type(self.model)
        else:
            # Import appropriate model class
            from panelbox import FixedEffects, PooledOLS, RandomEffects

            model_map = {"fe": FixedEffects, "pooled": PooledOLS, "re": RandomEffects}
            model_class = model_map.get(model_type, type(self.model))

        for formula in formulas:
            if self.verbose:
                print(f"Estimating: {formula}")

            try:
                model = model_class(formula, self.data, self.entity_col, self.time_col)
                result = model.fit(cov_type=self.results.cov_type)
                results_list.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"  Failed: {e}")
                results_list.append(None)

        return results_list

    def generate_robustness_table(
        self, results_list: List[PanelResults], parameters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate robustness table comparing specifications.

        Parameters
        ----------
        results_list : list of PanelResults
            Results to compare
        parameters : list of str, optional
            Parameters to include. If None, uses all common parameters

        Returns
        -------
        table : pd.DataFrame
            Comparison table
        """
        if parameters is None:
            # Find common parameters across all models
            param_sets = [set(r.params.index) for r in results_list if r is not None]
            parameters = sorted(set.intersection(*param_sets)) if param_sets else []

        data = []
        for i, result in enumerate(results_list, 1):
            if result is None:
                continue

            for param in parameters:
                if param in result.params.index:
                    data.append(
                        {
                            "Specification": f"({i})",
                            "Parameter": param,
                            "Coefficient": result.params[param],
                            "SE": result.std_errors[param],
                            "p-value": result.pvalues[param],
                        }
                    )

        df = pd.DataFrame(data)

        # Pivot to wide format
        if len(df) > 0:
            table = df.pivot(
                index="Parameter", columns="Specification", values=["Coefficient", "SE", "p-value"]
            )
        else:
            table = pd.DataFrame()

        return table
