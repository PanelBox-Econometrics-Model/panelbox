"""
Data transformers for panel-specific visualizations.

This module provides transformers to extract and prepare data from PanelResults
and PanelData objects for panel-specific charts.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class PanelDataTransformer:
    """
    Transformer for panel data visualization.

    This class provides static methods to extract and transform panel data
    for visualization purposes, including entity effects, time effects,
    variance decomposition, and panel structure analysis.
    """

    @staticmethod
    def extract_entity_effects(panel_results) -> Dict[str, Any]:
        """
        Extract entity fixed effects from panel estimation results.

        Parameters
        ----------
        panel_results : PanelResults
            Results from panel model estimation

        Returns
        -------
        dict
            Dictionary with keys:
            - 'entity_id': list of entity identifiers
            - 'effect': list of effect estimates (αᵢ)
            - 'std_error': list of standard errors (optional)

        Notes
        -----
        Entity effects are calculated as:
        αᵢ = ȳᵢ - X̄ᵢβ̂

        where:
        - ȳᵢ is the mean of y for entity i
        - X̄ᵢ is the mean of X for entity i
        - β̂ are the estimated coefficients
        """
        try:
            # Try to access entity effects directly
            if hasattr(panel_results, "entity_effects"):
                effects_df = panel_results.entity_effects

                return {
                    "entity_id": list(effects_df.index),
                    "effect": list(effects_df.values.flatten()),
                    "std_error": None,  # TODO: Calculate standard errors
                }

            # Alternative: calculate from residuals
            elif hasattr(panel_results, "resids") and hasattr(panel_results, "entity_info"):
                residuals = panel_results.resids
                entity_info = panel_results.entity_info

                # Group residuals by entity
                entity_ids = []
                effects = []

                for entity_id in entity_info["entity_ids"]:
                    entity_mask = entity_info["entity_labels"] == entity_id
                    entity_resids = residuals[entity_mask]

                    # Mean residual is the entity effect
                    entity_effect = np.mean(entity_resids)

                    entity_ids.append(entity_id)
                    effects.append(entity_effect)

                return {"entity_id": entity_ids, "effect": effects, "std_error": None}

            # Fallback: try to extract from model internals
            elif hasattr(panel_results, "model") and hasattr(panel_results.model, "entity_ids"):
                # Extract entity dummies if available
                entity_ids = panel_results.model.entity_ids

                # Try to get effects from params
                if hasattr(panel_results, "params"):
                    # Look for entity effect parameters
                    param_names = list(panel_results.params.index)
                    entity_params = [p for p in param_names if "entity" in p.lower() or "C(" in p]

                    if entity_params:
                        effects = [panel_results.params[p] for p in entity_params]
                        std_errors = (
                            [panel_results.std_errors[p] for p in entity_params]
                            if hasattr(panel_results, "std_errors")
                            else None
                        )

                        return {
                            "entity_id": entity_ids[: len(effects)],
                            "effect": effects,
                            "std_error": std_errors,
                        }

                # If no params found, return dummy data
                return {
                    "entity_id": list(entity_ids),
                    "effect": [0.0] * len(entity_ids),
                    "std_error": None,
                }

            else:
                raise AttributeError("Cannot extract entity effects from provided object")

        except Exception as e:
            raise ValueError(
                f"Failed to extract entity effects: {str(e)}\n"
                "Expected PanelResults object with entity_effects attribute or "
                "compatible structure."
            )

    @staticmethod
    def extract_time_effects(panel_results) -> Dict[str, Any]:
        """
        Extract time fixed effects from panel estimation results.

        Parameters
        ----------
        panel_results : PanelResults
            Results from panel model estimation

        Returns
        -------
        dict
            Dictionary with keys:
            - 'time': list of time periods
            - 'effect': list of effect estimates (λₜ)
            - 'std_error': list of standard errors (optional)

        Notes
        -----
        Time effects are calculated as:
        λₜ = ȳₜ - X̄ₜβ̂

        where:
        - ȳₜ is the mean of y at time t
        - X̄ₜ is the mean of X at time t
        - β̂ are the estimated coefficients
        """
        try:
            # Try to access time effects directly
            if hasattr(panel_results, "time_effects"):
                effects_df = panel_results.time_effects

                return {
                    "time": list(effects_df.index),
                    "effect": list(effects_df.values.flatten()),
                    "std_error": None,  # TODO: Calculate standard errors
                }

            # Alternative: calculate from residuals
            elif hasattr(panel_results, "resids") and hasattr(panel_results, "time_info"):
                residuals = panel_results.resids
                time_info = panel_results.time_info

                # Group residuals by time
                time_periods = []
                effects = []

                for time_period in time_info["time_periods"]:
                    time_mask = time_info["time_labels"] == time_period
                    time_resids = residuals[time_mask]

                    # Mean residual is the time effect
                    time_effect = np.mean(time_resids)

                    time_periods.append(time_period)
                    effects.append(time_effect)

                return {"time": time_periods, "effect": effects, "std_error": None}

            # Fallback: extract from params
            elif hasattr(panel_results, "params"):
                param_names = list(panel_results.params.index)
                time_params = [p for p in param_names if "time" in p.lower() or "year" in p.lower()]

                if time_params:
                    # Try to extract time periods from param names
                    time_periods = []
                    effects = []
                    std_errors = []

                    for param in time_params:
                        # Extract year/time from parameter name
                        # E.g., "C(year)[T.2001]" -> 2001
                        import re

                        match = re.search(r"\[T\.(\d+)\]", param)
                        if match:
                            time_periods.append(int(match.group(1)))
                        else:
                            time_periods.append(param)

                        effects.append(panel_results.params[param])
                        if hasattr(panel_results, "std_errors"):
                            std_errors.append(panel_results.std_errors[param])

                    return {
                        "time": time_periods,
                        "effect": effects,
                        "std_error": std_errors if std_errors else None,
                    }

            raise AttributeError("Cannot extract time effects from provided object")

        except Exception as e:
            raise ValueError(
                f"Failed to extract time effects: {str(e)}\n"
                "Expected PanelResults object with time_effects attribute or "
                "compatible structure."
            )

    @staticmethod
    def calculate_between_within(
        panel_data, variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate between and within variance decomposition.

        Parameters
        ----------
        panel_data : PanelData or DataFrame
            Panel data with MultiIndex (entity, time)
        variables : list of str, optional
            Variables to decompose. If None, use all numeric columns.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'variables': list of variable names
            - 'between_var': list of between variances
            - 'within_var': list of within variances
            - 'total_var': list of total variances

        Notes
        -----
        Variance decomposition:
        - Total variance: σ²_t = Var(yᵢₜ)
        - Between variance: σ²_b = Var(ȳᵢ) where ȳᵢ is entity mean
        - Within variance: σ²_w = Var(yᵢₜ - ȳᵢ)

        Property: σ²_t = σ²_b + σ²_w
        """
        try:
            # Convert to DataFrame if needed
            if hasattr(panel_data, "dataframe"):
                df = panel_data.dataframe
            elif isinstance(panel_data, pd.DataFrame):
                df = panel_data
            else:
                raise TypeError("panel_data must be PanelData or DataFrame")

            # Get numeric variables
            if variables is None:
                variables = df.select_dtypes(include=[np.number]).columns.tolist()

            # Ensure MultiIndex
            if not isinstance(df.index, pd.MultiIndex):
                raise ValueError("DataFrame must have MultiIndex (entity, time)")

            results = {"variables": [], "between_var": [], "within_var": [], "total_var": []}

            # Calculate for each variable
            for var in variables:
                if var not in df.columns:
                    continue

                # Total variance
                total_var = df[var].var()

                # Between variance: variance of entity means
                entity_means = df[var].groupby(level=0).mean()
                between_var = entity_means.var()

                # Within variance: variance of deviations from entity means
                # For each observation, subtract its entity mean
                deviations = df[var] - df[var].groupby(level=0).transform("mean")
                within_var = deviations.var()

                results["variables"].append(var)
                results["total_var"].append(total_var)
                results["between_var"].append(between_var)
                results["within_var"].append(within_var)

            return results

        except Exception as e:
            raise ValueError(
                f"Failed to calculate between-within decomposition: {str(e)}\n"
                "Expected PanelData or DataFrame with MultiIndex (entity, time)."
            )

    @staticmethod
    def analyze_panel_structure(panel_data) -> Dict[str, Any]:
        """
        Analyze panel data structure and balance.

        Parameters
        ----------
        panel_data : PanelData or DataFrame
            Panel data with MultiIndex (entity, time)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'entities': list of entity IDs
            - 'time_periods': list of time periods
            - 'presence_matrix': 2D array of 0/1 (entity × time)
            - 'is_balanced': bool
            - 'balance_percentage': float
            - 'complete_entities': list of entities with all periods
            - 'n_entities': int
            - 'n_periods': int

        Notes
        -----
        A panel is balanced if every entity has observations for every time period.
        """
        try:
            # Convert to DataFrame if needed
            if hasattr(panel_data, "dataframe"):
                df = panel_data.dataframe
            elif isinstance(panel_data, pd.DataFrame):
                df = panel_data
            else:
                raise TypeError("panel_data must be PanelData or DataFrame")

            # Ensure MultiIndex
            if not isinstance(df.index, pd.MultiIndex):
                raise ValueError("DataFrame must have MultiIndex (entity, time)")

            # Get unique entities and time periods
            entities = df.index.get_level_values(0).unique().tolist()
            time_periods = df.index.get_level_values(1).unique().tolist()

            # Sort for better visualization
            entities = sorted(entities)
            time_periods = sorted(time_periods)

            # Create presence matrix
            n_entities = len(entities)
            n_periods = len(time_periods)
            presence_matrix = np.zeros((n_entities, n_periods), dtype=int)

            for i, entity in enumerate(entities):
                for j, time_period in enumerate(time_periods):
                    if (entity, time_period) in df.index:
                        presence_matrix[i, j] = 1

            # Calculate statistics
            total_cells = n_entities * n_periods
            present_cells = np.sum(presence_matrix)
            is_balanced = present_cells == total_cells
            balance_percentage = (present_cells / total_cells) * 100

            # Find complete entities
            complete_entities = [
                entities[i] for i in range(n_entities) if np.sum(presence_matrix[i, :]) == n_periods
            ]

            # Calculate attrition rate (entities leaving over time)
            entities_per_period = np.sum(presence_matrix, axis=0)
            attrition_rate = None
            if n_periods > 1:
                # Percentage decline from first to last period
                attrition_rate = (
                    (entities_per_period[0] - entities_per_period[-1])
                    / entities_per_period[0]
                    * 100
                )

            return {
                "entities": entities,
                "time_periods": time_periods,
                "presence_matrix": presence_matrix,
                "is_balanced": is_balanced,
                "balance_percentage": balance_percentage,
                "complete_entities": complete_entities,
                "n_entities": n_entities,
                "n_periods": n_periods,
                "attrition_rate": attrition_rate,
            }

        except Exception as e:
            raise ValueError(
                f"Failed to analyze panel structure: {str(e)}\n"
                "Expected PanelData or DataFrame with MultiIndex (entity, time)."
            )

    @staticmethod
    def prepare_panel_summary(panel_data) -> Dict[str, Any]:
        """
        Prepare comprehensive summary of panel data.

        Parameters
        ----------
        panel_data : PanelData or DataFrame
            Panel data with MultiIndex (entity, time)

        Returns
        -------
        dict
            Dictionary with summary statistics including:
            - Structure analysis
            - Variance decomposition
            - Balance metrics
            - Descriptive statistics
        """
        # Combine multiple analyses
        structure = PanelDataTransformer.analyze_panel_structure(panel_data)
        between_within = PanelDataTransformer.calculate_between_within(panel_data)

        return {
            "structure": structure,
            "variance_decomposition": between_within,
        }
