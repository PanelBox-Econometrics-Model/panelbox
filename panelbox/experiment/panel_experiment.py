"""
Panel Experiment - Main interface for panel data analysis.

This module provides a high-level API for managing panel data experiments,
fitting multiple models, and comparing results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


class PanelExperiment:
    """
    Main interface for managing panel data experiments.

    This class provides a high-level API for fitting multiple panel models,
    comparing them, and generating reports. It manages model storage,
    naming, and retrieval.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with MultiIndex (entity, time) or columns for entity/time
    formula : str
        Model formula in patsy format (e.g., "y ~ x1 + x2")
    entity_col : str, optional
        Name of entity column (if not using MultiIndex)
    time_col : str, optional
        Name of time column (if not using MultiIndex)

    Attributes
    ----------
    data : pd.DataFrame
        Panel data (will be converted to MultiIndex)
    formula : str
        Model formula
    entity_col : str or None
        Entity column name
    time_col : str or None
        Time column name

    Examples
    --------
    >>> # With entity/time columns
    >>> experiment = PanelExperiment(
    ...     data=df,
    ...     formula="y ~ x1 + x2",
    ...     entity_col="firm_id",
    ...     time_col="year"
    ... )

    >>> # Fit models
    >>> experiment.fit_model('pooled_ols', name='ols')
    >>> experiment.fit_model('fixed_effects', name='fe')
    >>> experiment.fit_model('random_effects', name='re')

    >>> # List models
    >>> experiment.list_models()
    ['ols', 'fe', 're']

    >>> # Get model
    >>> fe_model = experiment.get_model('fe')
    """

    # Model type aliases
    MODEL_ALIASES = {
        "pooled": "pooled_ols",
        "pooled_ols": "pooled_ols",
        "fe": "fixed_effects",
        "fixed_effects": "fixed_effects",
        "re": "random_effects",
        "random_effects": "random_effects",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        formula: str,
        entity_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ):
        """
        Initialize PanelExperiment.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data
        formula : str
            Model formula
        entity_col : str, optional
            Entity column name
        time_col : str, optional
            Time column name
        """
        self.original_data = data.copy()
        self.formula = formula
        self.entity_col = entity_col
        self.time_col = time_col

        # Validate data
        self._validate_data()

        # Set MultiIndex if needed
        if entity_col is not None and time_col is not None:
            self.data = self.original_data.set_index([entity_col, time_col])
        else:
            self.data = self.original_data.copy()

        # Storage for fitted models
        self._models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}

        # Model counter for auto-naming
        self._model_counters: Dict[str, int] = {}

        # Experiment metadata
        self.created_at = datetime.now()

    def _validate_data(self):
        """
        Validate input data.

        Raises
        ------
        TypeError
            If data is not a DataFrame
        ValueError
            If data validation fails
        """
        if not isinstance(self.original_data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if self.original_data.empty:
            raise ValueError("data cannot be empty")

        # Check if using MultiIndex or columns
        if self.entity_col is None and self.time_col is None:
            # Must have MultiIndex
            if not isinstance(self.original_data.index, pd.MultiIndex):
                raise ValueError(
                    "If entity_col and time_col are not provided, "
                    "data must have a MultiIndex with (entity, time)"
                )
        else:
            # Check columns exist
            if self.entity_col is not None and self.entity_col not in self.original_data.columns:
                raise ValueError(f"entity_col '{self.entity_col}' not found in data")

            if self.time_col is not None and self.time_col not in self.original_data.columns:
                raise ValueError(f"time_col '{self.time_col}' not found in data")

    def fit_model(self, model_type: str, name: Optional[str] = None, **kwargs) -> Any:
        """
        Fit a panel model.

        Parameters
        ----------
        model_type : str
            Type of model to fit:
            - 'pooled_ols' or 'pooled': Pooled OLS
            - 'fixed_effects' or 'fe': Fixed Effects
            - 'random_effects' or 're': Random Effects
        name : str, optional
            Name for this model. If None, auto-generated.
        **kwargs
            Additional arguments passed to model.fit()

        Returns
        -------
        results
            Fitted model results

        Examples
        --------
        >>> experiment.fit_model('pooled_ols', name='ols')
        >>> experiment.fit_model('fixed_effects', name='fe', cov_type='clustered')
        >>> experiment.fit_model('re')  # Auto-generated name
        """
        # Resolve model type alias
        model_type_resolved = self.MODEL_ALIASES.get(model_type.lower(), model_type.lower())

        if model_type_resolved not in ["pooled_ols", "fixed_effects", "random_effects"]:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Valid types: {list(self.MODEL_ALIASES.keys())}"
            )

        # Generate name if not provided
        if name is None:
            name = self._generate_model_name(model_type_resolved)

        # Check if name already exists
        if name in self._models:
            raise ValueError(f"Model with name '{name}' already exists")

        # Create and fit model
        print(f"Fitting {model_type_resolved} model '{name}'...")
        model = self._create_model(model_type_resolved)
        results = model.fit(**kwargs)

        # Store model and metadata
        self._models[name] = results
        self._model_metadata[name] = {
            "model_type": model_type_resolved,
            "fitted_at": datetime.now(),
            "formula": self.formula,
            "kwargs": kwargs,
        }

        print(f"✅ Model '{name}' fitted successfully")

        return results

    def list_models(self) -> List[str]:
        """
        List names of all fitted models.

        Returns
        -------
        list of str
            Model names in order of fitting

        Examples
        --------
        >>> experiment.list_models()
        ['ols', 'fe', 're']
        """
        return list(self._models.keys())

    def get_model(self, name: str) -> Any:
        """
        Get a fitted model by name.

        Parameters
        ----------
        name : str
            Model name

        Returns
        -------
        results
            Fitted model results

        Raises
        ------
        KeyError
            If model name not found

        Examples
        --------
        >>> model = experiment.get_model('ols')
        >>> print(model.summary)
        """
        if name not in self._models:
            available = self.list_models()
            raise KeyError(
                f"Model '{name}' not found. "
                f"Available models: {available if available else 'none'}"
            )
        return self._models[name]

    def get_model_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a fitted model.

        Parameters
        ----------
        name : str
            Model name

        Returns
        -------
        dict
            Model metadata

        Examples
        --------
        >>> meta = experiment.get_model_metadata('ols')
        >>> print(meta['model_type'])
        'pooled_ols'
        """
        if name not in self._model_metadata:
            raise KeyError(f"Model '{name}' not found")
        return self._model_metadata[name].copy()

    def _generate_model_name(self, model_type: str) -> str:
        """
        Generate unique model name.

        Parameters
        ----------
        model_type : str
            Type of model

        Returns
        -------
        str
            Unique model name (e.g., 'pooled_ols_1', 'fixed_effects_2')
        """
        # Initialize counter for this model type if needed
        if model_type not in self._model_counters:
            self._model_counters[model_type] = 0

        # Increment counter
        self._model_counters[model_type] += 1

        # Generate name
        counter = self._model_counters[model_type]
        return f"{model_type}_{counter}"

    def _create_model(self, model_type: str) -> Any:
        """
        Factory method to create model.

        Parameters
        ----------
        model_type : str
            Type of model ('pooled_ols', 'fixed_effects', 'random_effects')

        Returns
        -------
        Model
            Unfitted model instance

        Raises
        ------
        ImportError
            If panelbox not available
        """
        # Import panelbox models
        try:
            import panelbox as pb
        except ImportError:
            raise ImportError("panelbox is required to fit models")

        # Create model based on type
        if model_type == "pooled_ols":
            if self.entity_col and self.time_col:
                model = pb.PooledOLS(
                    self.formula, self.original_data, self.entity_col, self.time_col
                )
            else:
                model = pb.PooledOLS(self.formula, self.data)

        elif model_type == "fixed_effects":
            if self.entity_col and self.time_col:
                model = pb.FixedEffects(
                    self.formula, self.original_data, self.entity_col, self.time_col
                )
            else:
                model = pb.FixedEffects(self.formula, self.data)

        elif model_type == "random_effects":
            if self.entity_col and self.time_col:
                model = pb.RandomEffects(
                    self.formula, self.original_data, self.entity_col, self.time_col
                )
            else:
                model = pb.RandomEffects(self.formula, self.data)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return model

    def validate_model(
        self, name: str, tests: str = "default", alpha: float = 0.05, verbose: bool = False
    ):
        """
        Run validation tests on a fitted model and return ValidationResult.

        This is a convenience method that combines model retrieval, validation,
        and ValidationResult creation.

        Parameters
        ----------
        name : str
            Name of fitted model to validate
        tests : str, default "default"
            Which tests to run
        alpha : float, default 0.05
            Significance level
        verbose : bool, default False
            Whether to print progress

        Returns
        -------
        ValidationResult
            ValidationResult container with test results

        Examples
        --------
        >>> experiment.fit_model('fixed_effects', name='fe')
        >>> val_result = experiment.validate_model('fe')
        >>> val_result.save_html('validation_report.html', test_type='validation')
        """
        from panelbox.experiment.results import ValidationResult

        # Get model
        model_results = self.get_model(name)

        # Create ValidationResult using factory method
        return ValidationResult.from_model_results(
            model_results=model_results,
            alpha=alpha,
            tests=tests,
            verbose=verbose,
            metadata={"experiment_formula": self.formula, "model_name": name},
        )

    def compare_models(self, model_names: Optional[List[str]] = None, **kwargs):
        """
        Compare multiple fitted models and return ComparisonResult.

        This is a convenience method that creates a ComparisonResult from
        the experiment's fitted models.

        Parameters
        ----------
        model_names : list of str, optional
            Names of models to compare. If None, compares all models.
        **kwargs
            Additional arguments passed to ComparisonResult

        Returns
        -------
        ComparisonResult
            ComparisonResult container with comparison data

        Examples
        --------
        >>> experiment.fit_model('pooled_ols', name='pooled')
        >>> experiment.fit_model('fixed_effects', name='fe')
        >>> experiment.fit_model('random_effects', name='re')
        >>>
        >>> # Compare all models
        >>> comp_result = experiment.compare_models()
        >>> comp_result.save_html('comparison.html', test_type='comparison')
        >>>
        >>> # Compare specific models
        >>> comp_result = experiment.compare_models(model_names=['fe', 're'])
        """
        from panelbox.experiment.results import ComparisonResult

        # Add experiment metadata
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}
        kwargs["metadata"]["experiment_formula"] = self.formula

        # Use ComparisonResult factory method
        return ComparisonResult.from_experiment(experiment=self, model_names=model_names, **kwargs)

    def fit_all_models(
        self, model_types: Optional[List[str]] = None, names: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Fit multiple models at once.

        This is a convenience method for fitting multiple model types with
        a single call.

        Parameters
        ----------
        model_types : list of str, optional
            Model types to fit. If None, fits ['pooled_ols', 'fixed_effects', 'random_effects']
        names : list of str, optional
            Names for each model. If None, auto-generates names.
        **kwargs
            Common kwargs passed to all fit() calls

        Returns
        -------
        dict
            Dictionary of {name: results} for all fitted models

        Examples
        --------
        >>> # Fit all three standard models
        >>> results = experiment.fit_all_models()
        >>> print(results.keys())
        dict_keys(['pooled_ols_1', 'fixed_effects_1', 'random_effects_1'])
        >>>
        >>> # Fit specific models with custom names
        >>> results = experiment.fit_all_models(
        ...     model_types=['fixed_effects', 'random_effects'],
        ...     names=['fe', 're']
        ... )
        """
        # Default to all three model types
        if model_types is None:
            model_types = ["pooled_ols", "fixed_effects", "random_effects"]

        # Validate names if provided
        if names is not None and len(names) != len(model_types):
            raise ValueError(
                f"Length of names ({len(names)}) must match "
                f"length of model_types ({len(model_types)})"
            )

        # Fit models
        results = {}
        for i, model_type in enumerate(model_types):
            name = names[i] if names is not None else None
            fitted_model = self.fit_model(model_type, name=name, **kwargs)

            # Get the actual name used (might be auto-generated)
            actual_name = self.list_models()[-1]
            results[actual_name] = fitted_model

        return results

    def analyze_residuals(self, name: str, **kwargs):
        """
        Analyze residuals of a fitted model and return ResidualResult.

        This is a convenience method that creates a ResidualResult from
        a fitted model's residuals, providing diagnostic tests and visualizations.

        Parameters
        ----------
        name : str
            Name of fitted model to analyze
        **kwargs
            Additional arguments passed to ResidualResult

        Returns
        -------
        ResidualResult
            ResidualResult container with diagnostic tests and data

        Raises
        ------
        ValueError
            If model with given name doesn't exist

        Examples
        --------
        >>> experiment.fit_model('fixed_effects', name='fe')
        >>>
        >>> # Analyze residuals
        >>> residual_result = experiment.analyze_residuals('fe')
        >>> print(residual_result.summary())
        >>>
        >>> # Save HTML report
        >>> residual_result.save_html(
        ...     'residual_diagnostics.html',
        ...     test_type='residuals',
        ...     theme='professional'
        ... )
        >>>
        >>> # Check specific tests
        >>> stat, pvalue = residual_result.shapiro_test
        >>> print(f"Normality test p-value: {pvalue:.4f}")
        >>>
        >>> dw = residual_result.durbin_watson
        >>> print(f"Durbin-Watson: {dw:.4f}")

        See Also
        --------
        validate_model : Validate model with specification tests
        compare_models : Compare multiple models
        """
        from panelbox.experiment.results import ResidualResult

        # Get model results
        model_results = self.get_model(name)

        # Add experiment metadata
        if "metadata" not in kwargs:
            kwargs["metadata"] = {}
        kwargs["metadata"]["experiment_formula"] = self.formula
        kwargs["metadata"]["model_name"] = name

        # Use ResidualResult factory method
        return ResidualResult.from_model_results(model_results=model_results, **kwargs)

    def save_master_report(
        self,
        file_path: str,
        theme: str = "professional",
        title: Optional[str] = None,
        reports: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Save master report with experiment overview and model summaries.

        The master report provides a comprehensive overview of the experiment,
        including all fitted models and links to generated sub-reports.

        Parameters
        ----------
        file_path : str
            Path where to save the master HTML report
        theme : str, default 'professional'
            Visual theme for the report. Options: 'professional', 'academic', 'presentation'
        title : str, optional
            Custom title for the report. If None, uses 'PanelBox Master Report'
        reports : list of dict, optional
            List of report information dictionaries with keys:
            - 'type': Report type ('validation', 'comparison', 'residuals')
            - 'title': Report title
            - 'description': Report description
            - 'file_path': Path to the report file

        Returns
        -------
        Path
            Path to saved master report

        Raises
        ------
        ValueError
            If no models have been fitted

        Examples
        --------
        >>> import panelbox as pb
        >>>
        >>> # Create experiment and fit models
        >>> data = pb.load_grunfeld()
        >>> experiment = pb.PanelExperiment(
        ...     data=data,
        ...     formula="invest ~ value + capital",
        ...     entity_col="firm",
        ...     time_col="year"
        ... )
        >>> experiment.fit_model('pooled_ols', name='ols')
        >>> experiment.fit_model('fixed_effects', name='fe')
        >>>
        >>> # Generate validation report
        >>> validation = experiment.validate_model('fe')
        >>> validation.save_html('validation.html', test_type='validation')
        >>>
        >>> # Generate master report
        >>> experiment.save_master_report(
        ...     'master.html',
        ...     theme='professional',
        ...     reports=[{
        ...         'type': 'validation',
        ...         'title': 'Fixed Effects Validation',
        ...         'description': 'Specification tests for FE model',
        ...         'file_path': 'validation.html'
        ...     }]
        ... )

        See Also
        --------
        validate_model : Validate model with specification tests
        compare_models : Compare multiple models
        analyze_residuals : Analyze model residuals
        """
        from pathlib import Path

        from panelbox.report import ReportManager

        # Validate that at least one model has been fitted
        if len(self._models) == 0:
            raise ValueError(
                "Cannot generate master report: no models have been fitted. "
                "Use experiment.fit_model() to fit at least one model."
            )

        # Prepare experiment information
        experiment_info = {
            "formula": self.formula,
            "n_obs": len(self.data),
            "n_entities": len(self.data.index.get_level_values(0).unique()),
            "n_time": len(self.data.index.get_level_values(1).unique()),
        }

        # Prepare models information
        models_info = []
        for name in self.list_models():
            results = self.get_model(name)
            metadata = self.get_model_metadata(name)

            model_entry = {
                "name": name,
                "type": metadata.get("model_type", "Unknown"),
                "rsquared": results.rsquared if hasattr(results, "rsquared") else 0.0,
                "aic": results.aic if hasattr(results, "aic") else 0.0,
                "bic": results.bic if hasattr(results, "bic") else 0.0,
            }

            # Add timestamp if available
            if "timestamp" in metadata:
                model_entry["timestamp"] = metadata["timestamp"].isoformat()

            models_info.append(model_entry)

        # Prepare reports list
        if reports is None:
            reports = []

        # Prepare master report context
        context = {
            "experiment_info": experiment_info,
            "models": models_info,
            "reports": reports,
            "report_title": title or "PanelBox Master Report",
        }

        # Load master CSS
        master_css_path = Path(__file__).parent.parent / "templates" / "master" / "master.css"
        if master_css_path.exists():
            context["master_css"] = master_css_path.read_text(encoding="utf-8")
        else:
            context["master_css"] = ""

        # Generate master report HTML
        report_manager = ReportManager()
        html = report_manager.generate_report(
            report_type="master",
            template="master/index.html",
            context=context,
            embed_assets=True,
            include_plotly=False,  # Master report doesn't need Plotly
        )

        # Save to file
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")

        return output_path

    def __repr__(self) -> str:
        """String representation."""
        n_models = len(self._models)
        model_list = ", ".join(self.list_models()) if n_models > 0 else "none"

        return (
            f"PanelExperiment(\n"
            f"  formula='{self.formula}',\n"
            f"  n_obs={len(self.data)},\n"
            f"  n_models={n_models},\n"
            f"  models=[{model_list}]\n"
            f")"
        )
