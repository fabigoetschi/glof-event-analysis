#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A4 — Event-table statistics: Spearman correlations, OLS models, appendix figures.

Purpose (Appendix):
  Reproducible statistical post-processing of the final event table (CSV):
    1) Enforce numeric columns
    2) Compute a simple derived index (sediment × confinement)
    3) Spearman rank correlations for selected variable pairs (CSV)
    4) OLS model comparison for BV (CSV)
    5) Compact appendix figure panels (PNG) with Spearman annotations

Inputs:
  - Event table CSV (as used in thesis), containing at least:
      catchment, bv_ratio, bq_ratio,
      q_outburst_m3s, mudflow_Q,
      v_lake_m3, mudflow_V,
      slope_median_deg, slope_10_25_frac,
      sediment_share_moraine, sediment_share_landslide, sediment_share_channel,
      valley_v_frac

Outputs (written to --outdir):
  - spearman_summary.csv
  - model_comparison_BV.csv
  - C1_hydrological_scaling.png
  - C2_slope_controls_BV.png
  - C3_sediment_availability_BV.png
  - C4_confinement_coupling.png

Dependencies:
  pip install pandas numpy matplotlib scipy statsmodels

Example:
  python A4_event_statistics_models.py \
    --infile ./06_Datenanalyse/042_Event_table.csv \
    --outdir ./06_Datenanalyse/appendix_stats
"""

from __future__ import annotations

import argparse
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import statsmodels.api as sm


CATCHMENT_ORDER = [
    "Kaskelen",
    "Kargaly",
    "Ulken Almaty",
    "Kishi Almaty",
    "Talgar",
    "Issyk",
]


# ---------------------------
# Helpers
# ---------------------------

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from DataFrame column names.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with trimmed column names.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric floats with robust cleaning.

    Handles common formatting issues:
    - empty strings / 'nan' / 'None' -> NaN
    - commas and spaces removed (e.g., '1,234' or '1 234')
    - non-parsable entries coerced to NaN

    Parameters
    ----------
    s : pandas.Series
        Input series.

    Returns
    -------
    pandas.Series
        Float series with invalid values converted to NaN.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s = s.str.replace(",", "", regex=False).str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure selected columns in a DataFrame are numeric.

    Only columns present in the DataFrame are processed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    cols : list of str
        Column names to coerce to numeric.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with selected columns converted to float.
    """
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = to_numeric_series(df[c])
    return df


def spearman_table(df: pd.DataFrame, pairs: list[tuple[str, str]], outpath: str, min_n: int = 5) -> pd.DataFrame:
    """Compute Spearman rank correlations for specified column pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table containing the variables.
    pairs : list of (str, str)
        List of (x, y) column name pairs.
    outpath : str
        Output CSV path for the correlation summary.
    min_n : int, default=5
        Minimum sample size required to compute statistics. Below this,
        rho and p are set to NaN.

    Returns
    -------
    pandas.DataFrame
        Summary table with columns: x, y, spearman_rho, p_value, n.
    """
    rows = []
    for x, y in pairs:
        d = df[[x, y]].dropna()
        n = len(d)
        if n < min_n:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(d[x], d[y], nan_policy="omit")
        rows.append({"x": x, "y": y, "spearman_rho": rho, "p_value": p, "n": n})
    out = pd.DataFrame(rows)
    out.to_csv(outpath, index=False)
    return out


def run_ols(df: pd.DataFrame, y: str, X_vars: list[str], name: str) -> dict:
    """Fit an OLS regression model and return key comparison metrics.

    Rows with missing values in any model variable are dropped.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    y : str
        Dependent variable column name.
    X_vars : list of str
        Explanatory variable column names.
    name : str
        Model name/label used in summaries.

    Returns
    -------
    dict
        Dictionary with:
        - model : str
        - n : int
        - r2 : float
        - adj_r2 : float
        - AIC : float
        - BIC : float
        - results : statsmodels.regression.linear_model.RegressionResults
    """
    d = df[[y] + X_vars].dropna()
    X = sm.add_constant(d[X_vars])
    yv = d[y]
    model = sm.OLS(yv, X).fit()
    return {
        "model": name,
        "n": int(model.nobs),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "AIC": float(model.aic),
        "BIC": float(model.bic),
        "results": model,
    }


def _annot_spearman(ax, d: pd.DataFrame, x: str, y: str, min_n: int = 5) -> None:
    """Annotate an axis with Spearman rho/p/n for two variables.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to annotate.
    d : pandas.DataFrame
        Data source.
    x, y : str
        Column names for correlation.
    min_n : int, default=5
        Minimum sample size required to compute rho and p.

    Returns
    -------
    None
    """
    dd = d[[x, y]].dropna()
    n = len(dd)
    if n >= min_n:
        rho, p = spearmanr(dd[x], dd[y], nan_policy="omit")
        txt = f"ρ={rho:.2f}\np={p:.3f}\nn={n}"
    else:
        txt = f"n={n}"
    ax.text(0.98, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)


def scatter_by_catchment(ax, df: pd.DataFrame, x: str, y: str, catchment_col: str, xlabel: str, ylabel: str, title: str) -> None:
    """Create a catchment-colored scatter plot for two variables.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    df : pandas.DataFrame
        Input table.
    x, y : str
        Column names for x/y.
    catchment_col : str
        Column used to group points (legend categories).
    xlabel, ylabel : str
        Axis labels.
    title : str
        Subplot title.

    Returns
    -------
    None
    """
    d = df[[x, y, catchment_col]].dropna()
    for cat in CATCHMENT_ORDER:
        dd = d[d[catchment_col] == cat]
        if dd.empty:
            continue
        ax.scatter(dd[x], dd[y], alpha=0.85, label=cat)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=9)


def savefig(path: str) -> None:
    """Save current Matplotlib figure to disk and close it.

    Parameters
    ----------
    path : str
        Output path. Parent directories are created if needed.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_panel_figure(df: pd.DataFrame, outpath: str, panels: list[dict], catchment_col: str, suptitle: str, ncols: int = 2, figsize=(12, 5.5)) -> None:
    """Create a multi-panel scatter figure with shared legend and Spearman annotations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table.
    outpath : str
        Output path for the PNG figure.
    panels : list of dict
        Panel definitions. Each dict must contain:
        - 'x', 'y' : column names
        - 'xlabel', 'ylabel' : labels
        - 'title' : panel title
    catchment_col : str
        Column used to group points for coloring/legend.
    suptitle : str
        Figure-level title.
    ncols : int, default=2
        Number of subplot columns.
    figsize : tuple, default=(12, 5.5)
        Figure size.

    Returns
    -------
    None
    """
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    fig.suptitle(suptitle, fontsize=14, y=0.98)

    for i, p in enumerate(panels):
        ax = axes[i]
        scatter_by_catchment(ax, df, p["x"], p["y"], catchment_col, p["xlabel"], p["ylabel"], p["title"])
        _annot_spearman(ax, df, p["x"], p["y"])
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False, fontsize=9)
        plt.tight_layout(rect=[0, 0.10, 1, 0.94])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.94])

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    """Entry point for event-table statistics and appendix figure generation.

    Reads the event table, enforces numeric types, computes derived indices,
    writes correlation/model summary CSVs, and exports appendix figure panels.

    Returns
    -------
    None
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Event table CSV")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--sep", default=";", help="CSV separator used in event table (default ';')")
    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.infile, sep=args.sep, engine="python", quoting=csv.QUOTE_MINIMAL)
    df = clean_colnames(df)

    # --- canonical column names (match thesis event table) ---
    catchment = "catchment"

    BV = "bv_ratio"
    BQ = "bq_ratio"

    Q_out = "q_outburst_m3s"
    Q_debris = "mudflow_Q"

    V_lake = "v_lake_m3"
    V_mud = "mudflow_V"

    slope_med = "slope_median_deg"
    slope_10_25 = "slope_10_25_frac"

    sed_moraine = "sediment_share_moraine"
    sed_landslide = "sediment_share_landslide"
    sed_channel = "sediment_share_channel"

    confinement = "valley_v_frac"

    # order catchments consistently (plots)
    if catchment in df.columns:
        df[catchment] = pd.Categorical(df[catchment], categories=CATCHMENT_ORDER, ordered=True)

    # numeric enforcement
    df = ensure_numeric(df, [
        BV, BQ, Q_out, Q_debris, V_lake, V_mud,
        slope_med, slope_10_25,
        sed_moraine, sed_landslide, sed_channel,
        confinement,
    ])

    # derived index: (moraine + landslide) * confinement (all in fractions)
    df["sed_conf_index"] = (
        ((df[sed_moraine].fillna(0) / 100) + (df[sed_landslide].fillna(0) / 100))
        * (df[confinement] / 100)
    )

    # -------------------------
    # Spearman table (Appendix)
    # -------------------------
    spearman_pairs = [
        (Q_out, Q_debris),
        (V_lake, V_mud),
        (slope_med, BV),
        (slope_10_25, BV),
        (sed_moraine, BV),
        (sed_landslide, BV),
        (sed_channel, BV),
        (BQ, BV),
        ("sed_conf_index", BV),
        (confinement, BV),
        (confinement, BQ),
    ]
    spearman_out = os.path.join(outdir, "spearman_summary.csv")
    spearman_table(df, spearman_pairs, spearman_out)
    print(f"[OK] Spearman table: {spearman_out}")

    # -------------------------
    # OLS model comparison (BV)
    # -------------------------
    models = [
        run_ols(df, BV, [Q_out, V_lake], "M1: Hydrology"),
        run_ols(df, BV, [Q_out, V_lake, sed_landslide], "M2: Hydro + Sediment (LS)"),
        run_ols(df, BV, [Q_out, V_lake, slope_med, confinement], "M3: Hydro + Geo"),
        run_ols(df, BV, [Q_out, V_lake, BQ], "M4: Process (BQ)"),
        run_ols(df, BV, [Q_out, V_lake, sed_landslide, confinement, "sed_conf_index"], "M5: Coupling index"),
    ]
    model_rows = [{
        "Model": m["model"],
        "n": m["n"],
        "R2": round(m["r2"], 3),
        "Adj_R2": round(m["adj_r2"], 3),
        "AIC": round(m["AIC"], 1),
        "BIC": round(m["BIC"], 1),
    } for m in models]
    model_df = pd.DataFrame(model_rows).sort_values("AIC")
    model_out = os.path.join(outdir, "model_comparison_BV.csv")
    model_df.to_csv(model_out, index=False)
    print(f"[OK] Model comparison: {model_out}")

    # -------------------------
    # Appendix panel figures
    # -------------------------
    make_panel_figure(
        df, os.path.join(outdir, "C1_hydrological_scaling.png"), [
            {"x": Q_out, "y": Q_debris, "title": "(a) Outburst vs debris-flow discharge", "xlabel": Q_out, "ylabel": Q_debris},
            {"x": V_lake, "y": V_mud, "title": "(b) Lake volume vs debris-flow volume", "xlabel": V_lake, "ylabel": V_mud},
        ],
        catchment, "Hydrological scaling relationships", ncols=2, figsize=(12, 5.5)
    )
    make_panel_figure(
        df, os.path.join(outdir, "C2_slope_controls_BV.png"), [
            {"x": slope_med, "y": BV, "title": "(a) BV vs median channel slope", "xlabel": slope_med, "ylabel": BV},
            {"x": slope_10_25, "y": BV, "title": "(b) BV vs fraction of slopes 10–25°", "xlabel": slope_10_25, "ylabel": BV},
        ],
        catchment, "Slope controls on BV", ncols=2, figsize=(12, 5.5)
    )
    make_panel_figure(
        df, os.path.join(outdir, "C3_sediment_availability_BV.png"), [
            {"x": sed_moraine, "y": BV, "title": "(a) BV vs moraine sediment share", "xlabel": sed_moraine, "ylabel": BV},
            {"x": sed_landslide, "y": BV, "title": "(b) BV vs landslide sediment share", "xlabel": sed_landslide, "ylabel": BV},
            {"x": sed_channel, "y": BV, "title": "(c) BV vs channel sediment share", "xlabel": sed_channel, "ylabel": BV},
        ],
        catchment, "Sediment availability effects on BV", ncols=3, figsize=(15, 5.5)
    )
    make_panel_figure(
        df, os.path.join(outdir, "C4_confinement_coupling.png"), [
            {"x": confinement, "y": BV, "title": "(a) BV vs V-valley share", "xlabel": confinement, "ylabel": BV},
            {"x": confinement, "y": BQ, "title": "(b) BQ vs V-valley share", "xlabel": confinement, "ylabel": BQ},
            {"x": "sed_conf_index", "y": BV, "title": "(c) Sediment–confinement index vs BV", "xlabel": "sed_conf_index", "ylabel": BV},
        ],
        catchment, "Confinement and sediment coupling effects", ncols=3, figsize=(15, 5.5)
    )

    print(f"[OK] Appendix figures written to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
