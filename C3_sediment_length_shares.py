#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C3 — Sediment source shares along a centerline (per event/date).

Purpose (Appendix):
  Intersect a stream centerline with mapped sediment-source polygons and compute
  per-event (Date) length totals and shares per sediment category (SediSource).

Inputs:
  - Sediment polygons (GeoJSON/GPKG/Shapefile) with attributes:
      * Date (event identifier; string)
      * SediSource (category; e.g., moraine/landslide/channel)
  - Centerline geometry (GeoJSON/GPKG/Shapefile): LineString/MultiLineString

Outputs:
  - CSV table: one row per Date, columns:
      total_km, <cat>_km, <cat>_share, <cat>_share_pct

Notes:
  - If sediment polygons overlap, line segments can be counted twice.
    A warning is printed if identical line pieces appear multiple times after
    overlay (a proxy for overlaps).

Dependencies:
  pip install geopandas pandas shapely

Example:
  python A3_sediment_length_shares.py \
    --sediment ./04_Sedimentanalyse/Sedimentanalyse.geojson \
    --centerline ./03_Geländeanalyse/Stream_Längsprofil_mit_Höhe.geojson \
    --out ./04_Sedimentanalyse/event_sedisource_lengthshares.csv
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import geopandas as gpd
import pandas as pd


def _ensure_projected_meter_crs(
        sed: gpd.GeoDataFrame, 
        line: gpd.GeoDataFrame
        ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Ensure both GeoDataFrames share a projected CRS in meters.

    - Reprojects `line` to `sed.crs` if they differ.
    - If CRS is geographic (lat/lon), both layers are reprojected to an
    estimated UTM CRS for length computations.

    Parameters
    ----------
    sed : geopandas.GeoDataFrame
        Sediment polygons.
    line : geopandas.GeoDataFrame
        Centerline geometry.

    Returns
    -------
    (sed_proj, line_proj) : tuple of geopandas.GeoDataFrame
        Reprojected layers in a projected CRS suitable for length calculations.

    Raises
    ------
    ValueError
        If either input has no CRS.
    """
    if sed.crs is None or line.crs is None:
        raise ValueError("CRS fehlt in Sediment oder Centerline. Beide brauchen ein CRS.")

    # unify CRS
    if line.crs != sed.crs:
        line = line.to_crs(sed.crs)

    # ensure projected CRS in meters (not geographic lat/lon)
    if getattr(sed.crs, "is_geographic", False):
        target = sed.estimate_utm_crs()
        sed = sed.to_crs(target)
        line = line.to_crs(target)

    return sed, line


def _dissolve_to_single_lines(line_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Dissolve a line layer to one geometry and explode to individual LineStrings.

    This simplifies subsequent overlay operations by working with a unified
    centerline geometry.

    Parameters
    ----------
    line_gdf : geopandas.GeoDataFrame
        Input line features.

    Returns
    -------
    geopandas.GeoDataFrame
        Exploded GeoDataFrame containing LineString parts (one per row).
    """
    geom = line_gdf.dissolve().geometry.iloc[0]
    out = gpd.GeoDataFrame({"geometry": [geom]}, crs=line_gdf.crs)
    return out.explode(index_parts=False).reset_index(drop=True)


def _warn_possible_overlaps(
        inside: gpd.GeoDataFrame, 
        date_value: str, 
        eps_m: float = 0.01
        ) -> None:
    """Warn about potential polygon overlaps that can cause double counting.

    This is a heuristic: geometries are simplified and compared by WKT to
    detect identical line pieces occurring multiple times after overlay.

    Parameters
    ----------
    inside : geopandas.GeoDataFrame
        Intersected line segments within sediment polygons.
    date_value : str
        Event/date identifier used for messaging.
    eps_m : float, default=0.01
        Simplification tolerance [m] used before WKT comparison.

    Returns
    -------
    None

    Notes
    -----
    This does not guarantee detection of all overlaps; it flags likely cases
    that could inflate length sums.
    """
    if inside.empty:
        return
    tmp = inside.copy()
    tmp["_wkt"] = tmp.geometry.apply(lambda g: g.simplify(eps_m).wkt if g is not None else "")
    dup = tmp.groupby("_wkt").filter(lambda x: len(x) > 1)
    if dup.empty:
        return
    print(f"\n[WARN] {date_value}: Possible polygon overlaps -> potential double counting.")


def compute_event_table(
    sediment_path: str,
    centerline_path: str,
    out_csv: str,
    date_col: str = "Date",
    cat_col: str = "SediSource",
    ) -> pd.DataFrame:
    """Compute per-event sediment-source length totals and shares along a centerline.

    For each event (grouped by `date_col`), the centerline is intersected with
    sediment-source polygons and line lengths are aggregated by category.

    Parameters
    ----------
    sediment_path : str
        Path to sediment polygons with attributes `date_col` and `cat_col`.
    centerline_path : str
        Path to centerline geometry (LineString/MultiLineString).
    out_csv : str
        Output CSV path.
    date_col : str, default="Date"
        Column name identifying events/dates.
    cat_col : str, default="SediSource"
        Column name identifying sediment source categories.

    Returns
    -------
    pandas.DataFrame
        Event table with one row per event/date and columns:
        - total_km
        - <category>_km
        - <category>_share
        - <category>_share_pct

    Raises
    ------
    KeyError
        If required columns are missing in the sediment layer.
    ValueError
        If CRS is missing in either layer.
    """
    sed = gpd.read_file(sediment_path)
    line_gdf = gpd.read_file(centerline_path)

    for col in (date_col, cat_col):
        if col not in sed.columns:
            raise KeyError(f"Sediment layer: missing column '{col}'. Available: {list(sed.columns)}")

    sed = sed[sed.geometry.notnull() & ~sed.geometry.is_empty].copy()
    line_gdf = line_gdf[line_gdf.geometry.notnull() & ~line_gdf.geometry.is_empty].copy()

    sed, line_gdf = _ensure_projected_meter_crs(sed, line_gdf)
    lines = _dissolve_to_single_lines(line_gdf)

    event_rows: List[Dict[str, float]] = []

    for date_value, sed_d in sed.groupby(date_col):
        # Intersect: centerline segments inside each sediment polygon
        inside = gpd.overlay(
            lines,
            sed_d[[cat_col, "geometry"]],
            how="intersection",
            keep_geom_type=True,
        )
        inside = inside[inside.geom_type.isin(["LineString", "MultiLineString"])].copy()
        if inside.empty:
            event_rows.append({"Date": str(date_value), "total_km": 0.0})
            continue

        inside["len_m"] = inside.length
        _warn_possible_overlaps(inside, str(date_value))

        by_cat = inside.groupby(cat_col, as_index=False)["len_m"].sum()
        by_cat["len_km"] = by_cat["len_m"] / 1000.0
        total_km = float(by_cat["len_km"].sum())

        row: Dict[str, float] = {"Date": str(date_value), "total_km": total_km}
        for _, r in by_cat.iterrows():
            c = str(r[cat_col]).strip()
            row[f"{c}_km"] = float(r["len_km"])

        if total_km > 0:
            for _, r in by_cat.iterrows():
                c = str(r[cat_col]).strip()
                share = float(r["len_km"]) / total_km
                row[f"{c}_share"] = share
                row[f"{c}_share_pct"] = 100.0 * share

        event_rows.append(row)

    df = pd.DataFrame(event_rows).sort_values("Date").reset_index(drop=True)
    num_cols = [c for c in df.columns if c != "Date"]
    df[num_cols] = df[num_cols].fillna(0.0)

    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote CSV: {out_csv} (rows={len(df)})")
    return df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for sediment-source length-share computation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for sediment layer, centerline, output CSV and column names.
    """
    p = argparse.ArgumentParser(description="Compute sediment-source length shares along a centerline per event/date.")
    p.add_argument("--sediment", required=True, help="Sediment polygons (GeoJSON/GPKG/Shp) with Date + SediSource")
    p.add_argument("--centerline", required=True, help="Centerline (LineString/MultiLineString)")
    p.add_argument("--out", required=True, help="Output CSV")
    p.add_argument("--date-col", default="Date")
    p.add_argument("--cat-col", default="SediSource")
    return p.parse_args()


def main() -> None:
    """Entry point for command-line execution of the sediment-share script."""
    args = parse_args()
    compute_event_table(args.sediment, args.centerline, args.out, args.date_col, args.cat_col)


if __name__ == "__main__":
    main()
