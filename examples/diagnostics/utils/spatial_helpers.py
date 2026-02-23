"""
Spatial analysis helper functions for PanelBox diagnostics tutorials.

Provides utilities for constructing spatial weight matrices, creating
Moran scatterplots, LISA cluster maps, and formatting LM test decision
tree summaries following the Anselin & Rey (2014) procedure.

Functions
---------
- build_weight_matrix: Construct spatial weight matrix from coordinates
- plot_moran_scatterplot: Moran scatterplot (y vs Wy) with quadrant labels
- plot_lisa_map: LISA cluster map using scatter plot
- lm_decision_tree_summary: Formatted decision tree from LM test results

References
----------
Anselin, L. (1988). "Spatial Econometrics: Methods and Models."
    Kluwer Academic Publishers.
Anselin, L. (1995). "Local Indicators of Spatial Association -- LISA."
    Geographical Analysis, 27(2), 93-115.
Anselin, L., & Rey, S.J. (2014). "Modern Spatial Econometrics in Practice:
    A Guide to GeoDa, GeoDaSpace and PySAL." GeoDa Press LLC.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# 1. Spatial weight matrix construction
# ---------------------------------------------------------------------------


def build_weight_matrix(
    coordinates: np.ndarray,
    method: str = "knn",
    k: int = 5,
    threshold: Optional[float] = None,
    row_normalize: bool = True,
) -> np.ndarray:
    """
    Construct a spatial weight matrix from geographic coordinates.

    Supports two construction methods:
    - 'knn': k-nearest neighbours, where each unit is connected to its k
      closest neighbours (symmetric weights are NOT enforced; the raw knn
      graph is used).
    - 'distance': distance-band, where units within a given Euclidean
      distance threshold are connected with inverse-distance weights.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (N, 2) with spatial coordinates for each unit.
        Typically (latitude, longitude) or (x, y).
    method : {'knn', 'distance'}, default 'knn'
        Method for defining neighbours.
        - 'knn': connect each unit to its ``k`` nearest neighbours.
        - 'distance': connect units within ``threshold`` Euclidean distance.
    k : int, default 5
        Number of nearest neighbours (only used when ``method='knn'``).
        Must be at least 1 and less than N.
    threshold : float or None, default None
        Distance cutoff for the 'distance' method.  If None and
        ``method='distance'``, the 25th percentile of all pairwise
        distances is used as a sensible default.
    row_normalize : bool, default True
        If True, each row of the weight matrix is normalised to sum to 1
        (row-stochastic matrix).  Row normalisation is standard practice
        in spatial econometrics so that Wy represents a weighted average
        of neighbours.

    Returns
    -------
    W : np.ndarray
        Spatial weight matrix of shape (N, N).  Diagonal entries are
        always zero.

    Raises
    ------
    ValueError
        If ``method`` is not one of 'knn' or 'distance', if ``k`` is
        invalid, or if ``coordinates`` does not have the expected shape.

    Examples
    --------
    >>> coords = np.random.default_rng(0).uniform(size=(50, 2))
    >>> W_knn = build_weight_matrix(coords, method="knn", k=4)
    >>> W_knn.shape
    (50, 50)
    >>> np.allclose(W_knn.sum(axis=1), 1.0)
    True

    >>> W_dist = build_weight_matrix(coords, method="distance", threshold=0.3)
    >>> W_dist.diagonal().sum()
    0.0

    Notes
    -----
    For the 'knn' method, the function uses ``scipy.spatial.KDTree`` for
    efficient neighbour lookup (O(N log N) construction, O(k log N) per
    query).  For the 'distance' method, ``scipy.spatial.distance.cdist``
    computes all pairwise distances in a single vectorised call.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            "coordinates must be a 2-D array with at least 2 columns; "
            f"got shape {coordinates.shape}"
        )

    N = coordinates.shape[0]

    if method == "knn":
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if k >= N:
            raise ValueError(f"k ({k}) must be less than the number of units ({N})")

        tree = KDTree(coordinates)
        # Query k+1 neighbours because the first result is the point itself
        _distances, indices = tree.query(coordinates, k=k + 1)

        W = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j_pos in range(1, k + 1):  # skip index 0 (self)
                neighbour = indices[i, j_pos]
                W[i, neighbour] = 1.0

    elif method == "distance":
        dist_matrix = cdist(coordinates, coordinates, metric="euclidean")

        if threshold is None:
            # Use 25th percentile of positive distances as default
            positive_dists = dist_matrix[dist_matrix > 0]
            threshold = np.percentile(positive_dists, 25)

        W = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                if i != j and dist_matrix[i, j] <= threshold:
                    W[i, j] = 1.0 / dist_matrix[i, j]

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'knn' or 'distance'.")

    # Ensure zero diagonal
    np.fill_diagonal(W, 0.0)

    # Row normalisation
    if row_normalize:
        row_sums = W.sum(axis=1)
        # Avoid division by zero for isolated units
        row_sums[row_sums == 0] = 1.0
        W = W / row_sums[:, np.newaxis]

    return W


# ---------------------------------------------------------------------------
# 2. Moran scatterplot
# ---------------------------------------------------------------------------


def plot_moran_scatterplot(
    values: np.ndarray,
    W: np.ndarray,
    entity_ids: Optional[np.ndarray] = None,
    variable_name: str = "y",
    annotate_outliers: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a Moran scatterplot: standardised variable vs its spatial lag.

    The four quadrants of the scatterplot represent different types of
    spatial association:

    - **HH** (upper-right): High values surrounded by high values (clusters)
    - **LH** (upper-left): Low values surrounded by high values (spatial outliers)
    - **LL** (lower-left): Low values surrounded by low values (clusters)
    - **HL** (lower-right): High values surrounded by low values (spatial outliers)

    A linear regression line is fitted through the scatter; its slope is
    an approximation of Moran's I statistic.

    Parameters
    ----------
    values : np.ndarray
        Variable values for N spatial units (1-D array of length N).
    W : np.ndarray
        Row-normalised spatial weight matrix of shape (N, N).
    entity_ids : np.ndarray or None, default None
        Labels for each spatial unit.  Used for annotating outlier points.
        If None, integer indices are used.
    variable_name : str, default 'y'
        Name of the variable, used in axis labels and title.
    annotate_outliers : bool, default True
        If True, points in the HL and LH quadrants (spatial outliers)
        are annotated with their entity ID.
    save_path : str or None, default None
        If provided, the figure is saved to this file path (e.g.
        'moran_scatter.png').  The format is inferred from the extension.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Moran scatterplot figure.

    Notes
    -----
    The slope of the OLS regression line through the scatterplot equals
    Moran's I when the variable is standardised and W is row-normalised.
    This provides an intuitive visual interpretation of global spatial
    autocorrelation.

    References
    ----------
    Anselin, L. (1996). "The Moran Scatterplot as an ESDA Tool to Assess
        Local Instability in Spatial Association." In M. Fischer, H. Scholten,
        & D. Unwin (Eds.), Spatial Analytical Perspectives on GIS.
    """
    values = np.asarray(values, dtype=float).ravel()
    N = len(values)

    if entity_ids is None:
        entity_ids = np.arange(N)
    else:
        entity_ids = np.asarray(entity_ids)

    # Standardise
    z = (values - np.mean(values)) / np.std(values)
    Wz = W @ z

    # Fit regression line (slope ~ Moran's I)
    slope = float(z @ Wz / (z @ z))
    x_line = np.linspace(z.min() - 0.5, z.max() + 0.5, 100)
    y_line = slope * x_line

    # Assign quadrants
    quadrants = np.empty(N, dtype="U2")
    quadrants[(z >= 0) & (Wz >= 0)] = "HH"
    quadrants[(z < 0) & (Wz < 0)] = "LL"
    quadrants[(z >= 0) & (Wz < 0)] = "HL"
    quadrants[(z < 0) & (Wz >= 0)] = "LH"

    # Colour mapping
    colour_map = {
        "HH": "#d32f2f",  # red
        "LL": "#1976d2",  # blue
        "HL": "#ff9800",  # orange
        "LH": "#4fc3f7",  # light blue
    }
    colours = [colour_map[q] for q in quadrants]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(z, Wz, c=colours, alpha=0.65, edgecolors="white", s=50, zorder=3)
    ax.plot(
        x_line,
        y_line,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label=f"Slope (Moran's I) = {slope:.4f}",
        zorder=2,
    )

    # Reference lines
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="-", alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="-", alpha=0.5)

    # Quadrant labels
    x_max = max(abs(z.min()), abs(z.max())) + 0.3
    y_max = max(abs(Wz.min()), abs(Wz.max())) + 0.3
    label_kwargs = {"fontsize": 14, "fontweight": "bold", "alpha": 0.25, "ha": "center", "va": "center"}
    ax.text(x_max * 0.6, y_max * 0.7, "HH", color="#d32f2f", **label_kwargs)
    ax.text(-x_max * 0.6, -y_max * 0.7, "LL", color="#1976d2", **label_kwargs)
    ax.text(x_max * 0.6, -y_max * 0.7, "HL", color="#ff9800", **label_kwargs)
    ax.text(-x_max * 0.6, y_max * 0.7, "LH", color="#4fc3f7", **label_kwargs)

    # Annotate outliers (HL and LH)
    if annotate_outliers:
        outlier_mask = np.isin(quadrants, ["HL", "LH"])
        for idx in np.where(outlier_mask)[0]:
            ax.annotate(
                str(entity_ids[idx]),
                (z[idx], Wz[idx]),
                fontsize=7,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_xlabel(f"Standardised {variable_name} (z)", fontsize=12)
    ax.set_ylabel(f"Spatial lag of {variable_name} (Wz)", fontsize=12)
    ax.set_title(f"Moran Scatterplot: {variable_name}", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 3. LISA cluster map
# ---------------------------------------------------------------------------


def plot_lisa_map(
    local_i: np.ndarray,
    pvalues: np.ndarray,
    z_values: np.ndarray,
    Wz_values: np.ndarray,
    coordinates: np.ndarray,
    alpha: float = 0.05,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a LISA (Local Indicators of Spatial Association) cluster map.

    Each spatial unit is plotted at its geographic coordinates and coloured
    by its LISA cluster type.  Only statistically significant units
    (p-value < ``alpha``) receive a cluster colour; non-significant units
    are shown in grey.

    Colour scheme:
    - **HH** (Hot spot): red -- high values surrounded by high values
    - **LL** (Cold spot): blue -- low values surrounded by low values
    - **HL** (High outlier): orange -- high values surrounded by low values
    - **LH** (Low outlier): light blue -- low values surrounded by high values
    - **NS** (Not significant): grey

    Parameters
    ----------
    local_i : np.ndarray
        Local Moran's I statistics for each unit (length N).
    pvalues : np.ndarray
        Pseudo p-values from permutation inference (length N).
    z_values : np.ndarray
        Standardised variable values (z-scores) for each unit (length N).
    Wz_values : np.ndarray
        Spatial lag of the standardised variable for each unit (length N).
    coordinates : np.ndarray
        Geographic coordinates of shape (N, 2).  The first column is
        typically longitude (or x) and the second is latitude (or y),
        but the function treats columns 0 and 1 as horizontal and vertical
        axes respectively.  If coordinates have the form (lat, lon), the
        plot will use column 1 (lon) as x and column 0 (lat) as y.
    alpha : float, default 0.05
        Significance level for classifying cluster membership.
    save_path : str or None, default None
        If provided, the figure is saved to this file path.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The LISA cluster map figure.

    Notes
    -----
    The cluster classification follows Anselin (1995):
    - Significant & z > 0 & Wz > 0 => HH
    - Significant & z < 0 & Wz < 0 => LL
    - Significant & z > 0 & Wz < 0 => HL
    - Significant & z < 0 & Wz > 0 => LH

    References
    ----------
    Anselin, L. (1995). "Local Indicators of Spatial Association -- LISA."
        Geographical Analysis, 27(2), 93-115.
    """
    local_i = np.asarray(local_i, dtype=float)
    pvalues = np.asarray(pvalues, dtype=float)
    z_values = np.asarray(z_values, dtype=float)
    Wz_values = np.asarray(Wz_values, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)

    N = len(local_i)

    # Determine x, y from coordinates
    # Convention: if shape is (N, 2), assume (lat, lon) => x=lon, y=lat
    if coordinates.shape[1] >= 2:
        x_coords = coordinates[:, 1]  # longitude
        y_coords = coordinates[:, 0]  # latitude
    else:
        raise ValueError(f"coordinates must have at least 2 columns; got shape {coordinates.shape}")

    # Classify each unit
    colour_map = {
        "HH": "#d32f2f",  # red
        "LL": "#1976d2",  # blue
        "HL": "#ff9800",  # orange
        "LH": "#4fc3f7",  # light blue
        "NS": "#bdbdbd",  # grey
    }

    cluster_types = []
    for i in range(N):
        if pvalues[i] >= alpha:
            cluster_types.append("NS")
        elif z_values[i] >= 0 and Wz_values[i] >= 0:
            cluster_types.append("HH")
        elif z_values[i] < 0 and Wz_values[i] < 0:
            cluster_types.append("LL")
        elif z_values[i] >= 0 and Wz_values[i] < 0:
            cluster_types.append("HL")
        else:
            cluster_types.append("LH")

    cluster_types = np.array(cluster_types)
    [colour_map[ct] for ct in cluster_types]

    # Count clusters for legend
    unique_types, counts = np.unique(cluster_types, return_counts=True)
    count_dict = dict(zip(unique_types, counts))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot non-significant first (background)
    ns_mask = cluster_types == "NS"
    if ns_mask.any():
        ax.scatter(
            x_coords[ns_mask],
            y_coords[ns_mask],
            c=colour_map["NS"],
            s=30,
            alpha=0.4,
            edgecolors="white",
            linewidths=0.3,
            label=f"NS ({count_dict.get('NS', 0)})",
            zorder=2,
        )

    # Plot significant clusters on top
    plot_order = ["LL", "LH", "HL", "HH"]  # plot HH last so it's on top
    label_map = {
        "HH": "HH - Hot spot",
        "LL": "LL - Cold spot",
        "HL": "HL - High outlier",
        "LH": "LH - Low outlier",
    }

    for ctype in plot_order:
        mask = cluster_types == ctype
        if mask.any():
            ax.scatter(
                x_coords[mask],
                y_coords[mask],
                c=colour_map[ctype],
                s=60,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
                label=f"{label_map[ctype]} ({count_dict.get(ctype, 0)})",
                zorder=3,
            )

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(
        f"LISA Cluster Map (alpha = {alpha})",
        fontsize=14,
    )
    ax.legend(
        loc="best",
        fontsize=9,
        framealpha=0.9,
        title="Cluster Type",
        title_fontsize=10,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# 4. LM decision tree summary
# ---------------------------------------------------------------------------


def lm_decision_tree_summary(
    lm_results: dict,
    alpha: float = 0.05,
) -> str:
    """
    Generate a formatted decision tree summary from LM spatial test results.

    Implements the classical Anselin & Rey (2014) specification search
    procedure for choosing between spatial lag (SAR), spatial error (SEM),
    or general spatial models (SDM/GNS) based on the four LM tests.

    The decision tree proceeds as follows:

    1. If neither LM-Lag nor LM-Error is significant at level ``alpha``,
       conclude there is no evidence of spatial dependence -- an OLS model
       is appropriate.
    2. If only LM-Lag is significant, estimate a Spatial Lag (SAR) model.
    3. If only LM-Error is significant, estimate a Spatial Error (SEM) model.
    4. If both LM-Lag and LM-Error are significant, consult the robust
       versions:

       a. If only Robust LM-Lag is significant, estimate SAR.
       b. If only Robust LM-Error is significant, estimate SEM.
       c. If both robust tests are significant, estimate a Spatial Durbin
          Model (SDM) or General Nesting Spatial (GNS) model.
       d. If neither robust test is significant, choose the model whose
          standard LM statistic is larger.

    Parameters
    ----------
    lm_results : dict
        Dictionary returned by ``run_lm_tests()``.  Expected keys:

        - 'lm_lag': LMTestResult
        - 'lm_error': LMTestResult
        - 'robust_lm_lag': LMTestResult
        - 'robust_lm_error': LMTestResult
        - 'recommendation': str
        - 'reason': str
    alpha : float, default 0.05
        Significance level for hypothesis testing.

    Returns
    -------
    summary : str
        Multi-line formatted string containing:
        - A table of all four test statistics with significance markers
        - The step-by-step decision tree logic
        - The final model recommendation

    Examples
    --------
    >>> from panelbox.diagnostics.spatial_tests import run_lm_tests
    >>> results = run_lm_tests(ols_result, W, alpha=0.05)
    >>> print(lm_decision_tree_summary(results, alpha=0.05))
    ================================================================
    LM Spatial Dependence Tests -- Decision Tree
    Anselin & Rey (2014) Specification Search
    ================================================================
    ...
    """
    # Extract test results
    lm_lag = lm_results["lm_lag"]
    lm_error = lm_results["lm_error"]
    robust_lag = lm_results["robust_lm_lag"]
    robust_error = lm_results["robust_lm_error"]

    # Significance flags
    lm_lag_sig = lm_lag.pvalue < alpha
    lm_error_sig = lm_error.pvalue < alpha
    robust_lag_sig = robust_lag.pvalue < alpha
    robust_error_sig = robust_error.pvalue < alpha

    # Significance markers
    def _sig_marker(pvalue: float) -> str:
        if pvalue < 0.001:
            return "***"
        elif pvalue < 0.01:
            return "**"
        elif pvalue < 0.05:
            return "*"
        elif pvalue < 0.10:
            return "."
        return ""

    # Build the summary string
    lines = []
    sep = "=" * 64
    dash = "-" * 64

    lines.append(sep)
    lines.append("LM Spatial Dependence Tests -- Decision Tree")
    lines.append("Anselin & Rey (2014) Specification Search")
    lines.append(sep)
    lines.append("")
    lines.append(f"Significance level: alpha = {alpha}")
    lines.append("")

    # Test results table
    lines.append(dash)
    lines.append(f"{'Test':<25} {'Statistic':>12} {'p-value':>12} {'Sig.':>6}")
    lines.append(dash)

    for test, label in [
        (lm_lag, "LM-Lag"),
        (lm_error, "LM-Error"),
        (robust_lag, "Robust LM-Lag"),
        (robust_error, "Robust LM-Error"),
    ]:
        marker = _sig_marker(test.pvalue)
        lines.append(f"{label:<25} {test.statistic:>12.4f} {test.pvalue:>12.4f} {marker:>6}")

    lines.append(dash)
    lines.append("Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1")
    lines.append("")

    # Decision tree walkthrough
    lines.append(sep)
    lines.append("Decision Tree")
    lines.append(sep)
    lines.append("")

    # Step 1: Check standard LM tests
    lines.append("Step 1: Are the standard LM tests significant?")
    lines.append(
        f"  LM-Lag:   p = {lm_lag.pvalue:.4f}  -->  "
        f"{'SIGNIFICANT' if lm_lag_sig else 'not significant'}"
    )
    lines.append(
        f"  LM-Error: p = {lm_error.pvalue:.4f}  -->  "
        f"{'SIGNIFICANT' if lm_error_sig else 'not significant'}"
    )
    lines.append("")

    if not lm_lag_sig and not lm_error_sig:
        # Case 1: Neither significant
        lines.append("  Result: Neither LM test is significant.")
        lines.append("  --> No evidence of spatial dependence.")
        lines.append("")
        lines.append(dash)
        lines.append("RECOMMENDATION: OLS (no spatial model needed)")
        lines.append(
            "REASON: Neither LM-Lag nor LM-Error rejects the null "
            "hypothesis of no spatial dependence."
        )

    elif lm_lag_sig and not lm_error_sig:
        # Case 2: Only LM-Lag significant
        lines.append("  Result: Only LM-Lag is significant.")
        lines.append("  --> Evidence of spatial lag dependence.")
        lines.append("")
        lines.append(dash)
        lines.append("RECOMMENDATION: SAR (Spatial Lag Model)")
        lines.append(
            "REASON: LM-Lag is significant while LM-Error is not, "
            "indicating spatial autoregressive structure in y."
        )

    elif not lm_lag_sig and lm_error_sig:
        # Case 3: Only LM-Error significant
        lines.append("  Result: Only LM-Error is significant.")
        lines.append("  --> Evidence of spatial error dependence.")
        lines.append("")
        lines.append(dash)
        lines.append("RECOMMENDATION: SEM (Spatial Error Model)")
        lines.append(
            "REASON: LM-Error is significant while LM-Lag is not, "
            "indicating spatial autocorrelation in the error term."
        )

    else:
        # Case 4: Both significant -- need robust tests
        lines.append("  Result: BOTH LM tests are significant.")
        lines.append("  --> Proceed to Step 2: examine robust tests.")
        lines.append("")

        lines.append("Step 2: Are the robust LM tests significant?")
        lines.append(
            f"  Robust LM-Lag:   p = {robust_lag.pvalue:.4f}  -->  "
            f"{'SIGNIFICANT' if robust_lag_sig else 'not significant'}"
        )
        lines.append(
            f"  Robust LM-Error: p = {robust_error.pvalue:.4f}  -->  "
            f"{'SIGNIFICANT' if robust_error_sig else 'not significant'}"
        )
        lines.append("")

        if robust_lag_sig and not robust_error_sig:
            lines.append("  Result: Only Robust LM-Lag is significant.")
            lines.append(
                "  --> Spatial lag dependence dominates after controlling for error dependence."
            )
            lines.append("")
            lines.append(dash)
            lines.append("RECOMMENDATION: SAR (Spatial Lag Model)")
            lines.append(
                "REASON: Robust LM-Lag is significant while Robust LM-Error "
                "is not, indicating spatial lag is the primary source of "
                "spatial dependence."
            )

        elif not robust_lag_sig and robust_error_sig:
            lines.append("  Result: Only Robust LM-Error is significant.")
            lines.append(
                "  --> Spatial error dependence dominates after controlling for lag dependence."
            )
            lines.append("")
            lines.append(dash)
            lines.append("RECOMMENDATION: SEM (Spatial Error Model)")
            lines.append(
                "REASON: Robust LM-Error is significant while Robust LM-Lag "
                "is not, indicating spatial error is the primary source of "
                "spatial dependence."
            )

        elif robust_lag_sig and robust_error_sig:
            lines.append("  Result: BOTH robust tests are significant.")
            lines.append(
                "  --> Evidence of both lag and error dependence. Consider a general model."
            )
            lines.append("")
            lines.append(dash)
            lines.append(
                "RECOMMENDATION: SDM (Spatial Durbin Model) or GNS (General Nesting Spatial Model)"
            )
            lines.append(
                "REASON: Both robust tests are significant, suggesting "
                "the data exhibit both spatial lag and spatial error "
                "dependence simultaneously. A Spatial Durbin Model can "
                "capture both forms of dependence."
            )

        else:
            # Neither robust test significant -- fall back to larger statistic
            if lm_lag.statistic > lm_error.statistic:
                chosen = "SAR (Spatial Lag Model)"
                reason_detail = (
                    f"LM-Lag statistic ({lm_lag.statistic:.4f}) exceeds "
                    f"LM-Error statistic ({lm_error.statistic:.4f})"
                )
            else:
                chosen = "SEM (Spatial Error Model)"
                reason_detail = (
                    f"LM-Error statistic ({lm_error.statistic:.4f}) exceeds "
                    f"LM-Lag statistic ({lm_lag.statistic:.4f})"
                )

            lines.append("  Result: Neither robust test is significant.")
            lines.append(
                "  --> Robust tests lack power; fall back to the larger standard LM statistic."
            )
            lines.append("")
            lines.append(dash)
            lines.append(f"RECOMMENDATION: {chosen}")
            lines.append(
                f"REASON: Neither robust test is significant. Using the "
                f"standard LM statistics as a tiebreaker: {reason_detail}."
            )

    lines.append(dash)
    lines.append("")

    # Append the recommendation from run_lm_tests if available
    if "recommendation" in lm_results and "reason" in lm_results:
        lines.append(f"run_lm_tests() recommendation: {lm_results['recommendation']}")
        lines.append(f"run_lm_tests() reason:         {lm_results['reason']}")

    return "\n".join(lines)
