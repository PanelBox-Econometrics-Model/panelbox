"""Variable transformer for panel data: generates lags, diffs, logs, and more."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


class VariableTransformer:
    """
    Gera transformacoes de variaveis respeitando estrutura de painel.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame em formato long panel.
    entity_col : str
        Nome da coluna de entidade (cross-section identifier).
    time_col : str
        Nome da coluna de tempo.
    transformations : dict, optional
        Dicionario de transformacoes a aplicar. Chaves possiveis:
        - 'lag': list[int] — defasagens (ex: [1, 2, 3])
        - 'diff': bool — primeira diferenca
        - 'log': bool — logaritmo natural (ignora variaveis com x <= 0)
        - 'acum': list[int] — media movel acumulada (ex: [3, 6, 12])
        - 'growth': bool — taxa de crescimento percentual
        - 'sq': bool — quadrado da variavel
    nan_threshold : float, optional
        Fracao maxima de NaN permitida por variavel transformada.
        Default 0.30 (30%).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        transformations: dict | None = None,
        nan_threshold: float = 0.30,
    ):
        self.data = data.copy()
        self.entity_col = entity_col
        self.time_col = time_col
        self.transformations = transformations or {
            "lag": [1],
            "diff": True,
        }
        self.nan_threshold = nan_threshold

        self._transformation_map: dict[str, tuple[str, str]] = {}
        self._discarded: list[tuple[str, float]] = []
        self._multicollinear_removed: list[str] = []
        self._transformed_df: pd.DataFrame | None = None

    def transform(self, variables: list[str]) -> pd.DataFrame:
        """
        Gera todas as transformacoes para as variaveis candidatas.

        Returns DataFrame com colunas originais + transformadas.
        Convencao de nomes:
            Lag:     L1_var, L2_var, L3_var
            Diff:    D_var
            Log:     log_var
            Acum:    acum3_var, acum6_var
            Growth:  growth_var
            Sq:      sq_var
        """
        result = self.data.copy()
        self._transformation_map = {}
        self._discarded = []
        new_cols: dict[str, pd.Series] = {}

        grouped = self.data.groupby(self.entity_col)
        t = self.transformations

        for var in variables:
            if var not in self.data.columns:
                warnings.warn(f"Variavel '{var}' nao encontrada no DataFrame.", stacklevel=2)
                continue

            series = self.data[var]
            series_grouped = grouped[var]

            if "lag" in t:
                new_cols.update(self._create_lags(series_grouped, var, t["lag"]))

            if t.get("diff"):
                new_cols.update(self._create_diff(series_grouped, var))

            if t.get("log"):
                new_cols.update(self._create_log(series, var))

            if "acum" in t:
                new_cols.update(self._create_acum(series_grouped, var, t["acum"]))

            if t.get("growth"):
                new_cols.update(self._create_growth(series_grouped, var))

            if t.get("sq"):
                new_cols.update(self._create_sq(series, var))

        # Filter by NaN threshold
        n_rows = len(result)
        valid_cols: dict[str, pd.Series] = {}
        for col_name, col_series in new_cols.items():
            nan_frac = col_series.isna().sum() / n_rows
            if nan_frac > self.nan_threshold:
                self._discarded.append((col_name, nan_frac))
            else:
                valid_cols[col_name] = col_series

        # Add valid columns to result
        for col_name, col_series in valid_cols.items():
            result[col_name] = col_series.values

        # Check multicollinearity among new columns
        if len(valid_cols) > 1:
            to_remove = self._check_multicollinearity(result, threshold=0.95)
            self._multicollinear_removed = to_remove
            result = result.drop(columns=to_remove, errors="ignore")
            for col_name in to_remove:
                self._transformation_map.pop(col_name, None)

        self._transformed_df = result
        return result

    def _create_lags(self, series_grouped, var: str, lags: list[int]) -> dict[str, pd.Series]:
        """Cria defasagens por entidade via groupby.shift()."""
        cols = {}
        for lag in lags:
            col_name = f"L{lag}_{var}"
            cols[col_name] = series_grouped.shift(lag)
            self._transformation_map[col_name] = (var, f"lag{lag}")
        return cols

    def _create_diff(self, series_grouped, var: str) -> dict[str, pd.Series]:
        """Primeira diferenca por entidade."""
        col_name = f"D_{var}"
        cols = {col_name: series_grouped.diff()}
        self._transformation_map[col_name] = (var, "diff")
        return cols

    def _create_log(self, series: pd.Series, var: str) -> dict[str, pd.Series]:
        """Log natural. Retorna dict vazio para variaveis com x <= 0."""
        if (series <= 0).any():
            warnings.warn(
                f"Variavel '{var}' contem valores <= 0. Transformacao log ignorada.",
                stacklevel=2,
            )
            return {}
        col_name = f"log_{var}"
        cols = {col_name: np.log(series)}
        self._transformation_map[col_name] = (var, "log")
        return cols

    def _create_acum(self, series_grouped, var: str, windows: list[int]) -> dict[str, pd.Series]:
        """Media movel rolling por entidade."""
        cols = {}
        for w in windows:
            col_name = f"acum{w}_{var}"
            cols[col_name] = series_grouped.transform(
                lambda x: x.rolling(window=w, min_periods=w).mean()
            )
            self._transformation_map[col_name] = (var, f"acum{w}")
        return cols

    def _create_growth(self, series_grouped, var: str) -> dict[str, pd.Series]:
        """(x_t - x_{t-1}) / x_{t-1} * 100 por entidade."""
        col_name = f"growth_{var}"
        cols = {col_name: series_grouped.pct_change() * 100}
        self._transformation_map[col_name] = (var, "growth")
        return cols

    def _create_sq(self, series: pd.Series, var: str) -> dict[str, pd.Series]:
        """Quadrado da variavel."""
        col_name = f"sq_{var}"
        cols = {col_name: series**2}
        self._transformation_map[col_name] = (var, "sq")
        return cols

    def _check_multicollinearity(
        self, df_transformed: pd.DataFrame, threshold: float = 0.95
    ) -> list[str]:
        """Detect pairs with correlation above threshold and return variables to remove."""
        # Only check transformed columns
        transformed_cols = [c for c in df_transformed.columns if c in self._transformation_map]
        if len(transformed_cols) < 2:
            return []

        corr_matrix = df_transformed[transformed_cols].corr().abs()
        to_remove = set()

        for i, col_i in enumerate(transformed_cols):
            for j, col_j in enumerate(transformed_cols):
                if j <= i:
                    continue
                if col_i in to_remove or col_j in to_remove:
                    continue
                if corr_matrix.loc[col_i, col_j] > threshold:
                    # Remove the more complex one (higher lag, or later alphabetically)
                    remove = self._pick_to_remove(col_i, col_j)
                    to_remove.add(remove)

        return list(to_remove)

    def _pick_to_remove(self, col_a: str, col_b: str) -> str:
        """Pick which of two highly-correlated columns to remove (keep simpler)."""
        # Priority: original > sq > log > diff > growth > lag1 > lag2 > acum3 > ...
        complexity = {"log": 1, "sq": 1, "diff": 2, "growth": 2}

        _, type_a = self._transformation_map.get(col_a, (col_a, ""))
        _, type_b = self._transformation_map.get(col_b, (col_b, ""))

        def _score(t: str) -> int:
            if t in complexity:
                return complexity[t]
            # lag1 -> 3, lag2 -> 4, acum3 -> 5, acum6 -> 8, etc.
            for prefix in ("lag", "acum"):
                if t.startswith(prefix):
                    num = int(t[len(prefix) :])
                    return 2 + num
            return 10

        return col_b if _score(type_a) <= _score(type_b) else col_a

    def get_transformation_map(self) -> dict[str, tuple[str, str]]:
        """Return mapping of transformed name to (source_var, transformation_type)."""
        return dict(self._transformation_map)

    def get_valid_transformations(self) -> list[str]:
        """Lista de colunas transformadas sem NaN excessivo."""
        if self._transformed_df is None:
            return []
        return [c for c in self._transformed_df.columns if c in self._transformation_map]

    def summary(self) -> str:
        """Resumo: quantas transformacoes geradas, quantas descartadas."""
        total_generated = (
            len(self._transformation_map) + len(self._discarded) + len(self._multicollinear_removed)
        )
        valid = len(self._transformation_map)
        discarded_nan = len(self._discarded)
        discarded_mc = len(self._multicollinear_removed)

        lines = [
            "=== VariableTransformer Summary ===",
            f"Transformacoes geradas:     {total_generated}",
            f"Transformacoes validas:     {valid}",
            f"Descartadas por NaN > {self.nan_threshold:.0%}: {discarded_nan}",
            f"Descartadas por multicolinearidade: {discarded_mc}",
        ]

        if self._discarded:
            lines.append("\nDescartadas (NaN):")
            for col_name, nan_frac in self._discarded:
                lines.append(f"  {col_name}: {nan_frac:.1%} NaN")

        if self._multicollinear_removed:
            lines.append("\nRemovidas (multicolinearidade > 0.95):")
            for col_name in self._multicollinear_removed:
                lines.append(f"  {col_name}")

        return "\n".join(lines)
