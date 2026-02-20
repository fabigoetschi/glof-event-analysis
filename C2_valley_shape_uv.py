#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C2 — Valley shape classification (U vs V) from DEM cross-sections.

Purpose (Appendix):
  Sample DEM cross-sections orthogonal to a stream centerline, compute a simple
  floor-width ratio metric, and classify each cross-section as:
    - U (wide floor)
    - V (narrow floor)
    - intermediate
    - too_shallow / invalid

Used in thesis:
  Aggregation of cross-sections yields a confinement proxy (e.g., V-valley share)
  for each event/channel.

Method (summary):
  1) Place cross-sections every `--spacing` meters along stream(s).
  2) For each cross-section, sample DEM elevations at `--step` spacing.
  3) Compute valley depth (z_max - z_min).
  4) Define floor elevation: z_floor = z_min + floor_rel_height * depth.
  5) Floor width ratio = width(z <= z_floor) / total_width.
  6) Thresholds decide U/V.

Inputs:
  - DEM GeoTIFF (projected CRS in meters recommended)
  - Stream centerlines (LineString/MultiLineString) readable by GeoPandas

Outputs:
  - CSV with one row per cross-section (metrics + class)
  - Optional GeoJSON:
      * cross-sections
      * cross-section center points

Dependencies:
  pip install geopandas rasterio shapely numpy pandas

Example:
  python A2_valley_shape_uv.py \
    --dem ./01_DEM/DEM_def.tif \
    --streams ./03_Geländeanalyse/Stream_Längsprofil_mit_Höhe.geojson \
    --outdir ./03_Geländeanalyse/valley_shape_output
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, Point


def densify_line(line: LineString, step_m: float) -> LineString:
    """Densify a line by interpolating vertices at a fixed spacing.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Input line geometry.
    step_m : float
        Target spacing between interpolated vertices [m].

    Returns
    -------
    shapely.geometry.LineString
        Densified line with additional vertices.
    """
    if line.length == 0:
        return line
    distances = np.arange(0, line.length, step_m)
    pts = [line.interpolate(float(d)) for d in distances]
    pts.append(line.interpolate(line.length))
    return LineString([(p.x, p.y) for p in pts])


def cross_section_at(
        line: LineString, 
        s: float, 
        half_width_m: float
        ) -> LineString:
    """Construct a cross-section line orthogonal to a centerline at distance s.

    The normal direction is approximated using a small forward/backward step
    to estimate the local tangent.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Stream centerline (projected CRS).
    s : float
        Along-line distance at which to place the cross-section [m].
    half_width_m : float
        Half-width of the cross-section [m]; total width is 2 * half_width_m.

    Returns
    -------
    shapely.geometry.LineString
        Cross-section line as a LineString with two endpoints.

    Notes
    -----
    If the tangent cannot be estimated (zero-length), a horizontal cross-section
    is returned.
    """
    s = max(0.0, min(float(s), line.length))
    center = line.interpolate(s)
    cx, cy = center.x, center.y

    # tangent from small forward/backward step
    ds = 1.0
    p1 = line.interpolate(max(0.0, s - ds))
    p2 = line.interpolate(min(line.length, s + ds))
    tx, ty = (p2.x - p1.x), (p2.y - p1.y)
    lt = math.hypot(tx, ty)

    if lt == 0:
        return LineString([(cx - half_width_m, cy), (cx + half_width_m, cy)])

    nx, ny = (-ty / lt), (tx / lt)  # unit normal
    left = (cx - half_width_m * nx, cy - half_width_m * ny)
    right = (cx + half_width_m * nx, cy + half_width_m * ny)
    return LineString([left, right])


def sample_dem_along_line(
        dem: rasterio.DatasetReader, 
        line: LineString, 
        step_m: float
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Sample DEM elevations along a line at approximately fixed spacing.

    Parameters
    ----------
    dem : rasterio.DatasetReader
        Open rasterio dataset for the DEM.
    line : shapely.geometry.LineString
        Sampling line in the DEM CRS.
    step_m : float
        Sampling step along the line [m].

    Returns
    -------
    dists : numpy.ndarray
        Distances along the line [m], shape (n,).
    elevs : numpy.ndarray
        Sampled elevations [m] (NaN for NoData), shape (n,).
    """
    if line.length == 0:
        return np.array([]), np.array([])
    n = max(2, int(line.length / step_m) + 1)
    dists = np.linspace(0.0, float(line.length), n)
    pts = [line.interpolate(float(d)) for d in dists]
    coords = [(p.x, p.y) for p in pts]
    nodata = dem.nodata

    z = []
    for v in dem.sample(coords):
        zz = float(v[0])
        if nodata is not None and (math.isclose(zz, nodata) or zz == nodata):
            z.append(np.nan)
        else:
            z.append(zz)

    return dists, np.asarray(z, dtype=float)


def classify_uv(
    dists: np.ndarray,
    elevs: np.ndarray,
    floor_rel_height: float,
    u_threshold: float,
    v_threshold: float,
    min_depth_m: float,
) -> Dict:
    """Classify valley cross-section shape as U, V, intermediate, or invalid.

    Method
    ------
    - Compute valley depth = z_max - z_min.
    - Define a "floor" elevation: z_floor = z_min + floor_rel_height * depth.
    - Compute floor width ratio = width(z <= z_floor) / total_width.
    - Classify based on thresholds.

    Parameters
    ----------
    dists : numpy.ndarray
        Distances along the cross-section [m].
    elevs : numpy.ndarray
        Elevations along the cross-section [m] (NaNs allowed).
    floor_rel_height : float
        Relative height (0..1) above the minimum used to define the valley floor.
    u_threshold : float
        Ratio threshold at/above which cross-sections are classified as U.
    v_threshold : float
        Ratio threshold at/below which cross-sections are classified as V.
    min_depth_m : float
        Minimum required cross-section depth [m]. Shallower sections are flagged.

    Returns
    -------
    dict
        Classification results with keys including:
        - 'class' : {'U','V','intermediate','too_shallow','invalid'}
        - 'depth', 'floor_width_ratio', 'total_width', 'floor_width', 'z_min', 'z_max'
        (some fields may be missing or NaN for invalid cases)
    """
    mask = ~np.isnan(elevs)
    if mask.sum() < 3:
        return {"class": "invalid"}

    d = dists[mask]
    z = elevs[mask]
    total_width = float(d[-1] - d[0])
    if total_width <= 0:
        return {"class": "invalid"}

    z_min = float(np.min(z))
    z_max = float(np.max(z))
    depth = z_max - z_min

    if depth < min_depth_m:
        return {"class": "too_shallow", "depth": depth}

    z_floor = z_min + floor_rel_height * depth
    floor_mask = z <= z_floor

    if floor_mask.sum() < 2:
        floor_width = 0.0
    else:
        df = d[floor_mask]
        floor_width = float(df[-1] - df[0])

    ratio = floor_width / total_width
    if ratio >= u_threshold:
        cls = "U"
    elif ratio <= v_threshold:
        cls = "V"
    else:
        cls = "intermediate"

    return {
        "class": cls,
        "depth": depth,
        "floor_width_ratio": ratio,
        "total_width": total_width,
        "floor_width": floor_width,
        "z_min": z_min,
        "z_max": z_max,
    }

def valley_class_shares_per_event(
    df: pd.DataFrame,
    event_col: str = "Date",
    class_col: str = "class",
    out_csv: str | None = None,
) -> pd.DataFrame:
    """Compute per-event shares (%) of valley-shape classes.

    This aggregates the cross-section level classifications produced by this
    script into one row per event (e.g., per Date). Shares are computed both:

    - relative to all cross-sections (including 'too_shallow'/'invalid')
    - relative to valid cross-sections only (U/V/intermediate)

    Parameters
    ----------
    df : pandas.DataFrame
        Cross-section table as created in `run()` containing at least
        `event_col` and `class_col`.
    event_col : str, default="Date"
        Column identifying the event (e.g., 'Date').
    class_col : str, default="class"
        Column containing valley classification labels.
    out_csv : str or None, default=None
        If provided, write the aggregated table to this CSV path.

    Returns
    -------
    pandas.DataFrame
        One row per event with counts and percentage shares.
    """
    if event_col not in df.columns:
        raise KeyError(f"Missing event column '{event_col}'. Available: {list(df.columns)}")
    if class_col not in df.columns:
        raise KeyError(f"Missing class column '{class_col}'. Available: {list(df.columns)}")

    d = df[[event_col, class_col]].copy()
    d[event_col] = d[event_col].astype(str).str.strip()
    d[class_col] = d[class_col].astype(str).str.strip()
    d = d[d[event_col] != ""]  # drop rows without event id

    if d.empty:
        out = pd.DataFrame(columns=[event_col])
        if out_csv:
            Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_csv, index=False)
        return out

    # counts per event x class
    counts = (
        d.groupby([event_col, class_col])
         .size()
         .unstack(fill_value=0)
         .sort_index()
    )

    # ensure canonical columns exist
    for c in ["U", "V", "intermediate", "too_shallow", "invalid"]:
        if c not in counts.columns:
            counts[c] = 0

    n_total = counts.sum(axis=1)
    valid_cols = ["U", "V", "intermediate"]
    n_valid = counts[valid_cols].sum(axis=1)

    # shares vs total
    shares_total = counts.div(n_total, axis=0) * 100.0
    shares_total = shares_total.add_suffix("_pct_total")

    # shares vs valid only (avoid divide-by-zero)
    shares_valid = counts[valid_cols].div(n_valid.replace(0, np.nan), axis=0) * 100.0
    shares_valid = shares_valid.add_suffix("_pct_valid").fillna(0.0)

    out = pd.concat(
        [
            n_total.rename("n_xsec_total"),
            n_valid.rename("n_xsec_valid"),
            shares_total[["U_pct_total", "V_pct_total", "intermediate_pct_total", "too_shallow_pct_total", "invalid_pct_total"]],
            shares_valid[["U_pct_valid", "V_pct_valid", "intermediate_pct_valid"]],
        ],
        axis=1,
    ).reset_index()

    # tidy rounding
    pct_cols = [c for c in out.columns if c.endswith("_pct_total") or c.endswith("_pct_valid")]
    out[pct_cols] = out[pct_cols].round(3)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)

    return out

def run(
    dem_path: str,
    streams_path: str,
    outdir: str,
    spacing_m: float,
    half_width_m: float,
    sample_step_m: float,
    floor_rel_height: float,
    u_threshold: float,
    v_threshold: float,
    min_depth_m: float,
    save_geojson: bool,
    ) -> None:
    """Run the valley-shape classification pipeline along stream centerlines.

    For each stream line, cross-sections are placed at a fixed spacing,
    sampled on the DEM, classified, and written to CSV (and optionally GeoJSON).

    Parameters
    ----------
    dem_path : str
        Path to the DEM GeoTIFF.
    streams_path : str
        Path to stream centerlines readable by GeoPandas.
    outdir : str
        Output directory.
    spacing_m : float
        Spacing between cross-sections along the stream [m].
    half_width_m : float
        Half-width of each cross-section [m].
    sample_step_m : float
        Sampling step along each cross-section [m].
    floor_rel_height : float
        Relative floor height (0..1) used for floor width ratio.
    u_threshold : float
        U classification threshold for the floor width ratio.
    v_threshold : float
        V classification threshold for the floor width ratio.
    min_depth_m : float
        Minimum valley depth required to classify [m].
    save_geojson : bool
        If True, write cross-section and center point GeoJSON outputs.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If streams have no CRS information.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as dem:
        streams = gpd.read_file(streams_path)
        if streams.crs is None:
            raise ValueError("Streams layer has no CRS.")

        # reproject streams to DEM CRS if needed
        if streams.crs != dem.crs:
            streams = streams.to_crs(dem.crs)

        records = []
        xsec_features = []
        center_features = []

        for feat_idx, row in streams.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # thesis-relevant IDs (kept if present)
            props = row.drop(labels=["geometry"], errors="ignore").to_dict()
            catchment = props.get("NameCatch", props.get("catchment", ""))
            glacier = props.get("Glacier", props.get("glacier", ""))
            lake = props.get("Lake", props.get("lake", ""))

            # handle LineString/MultiLineString
            lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms) if geom.geom_type == "MultiLineString" else []
            for li, line in enumerate(lines):
                if line.length == 0:
                    continue

                line_d = densify_line(line, step_m=50.0)
                n_x = max(1, int(line_d.length / spacing_m))
                s_list = np.linspace(0.5 * spacing_m, max(0.5 * spacing_m, line_d.length - 0.5 * spacing_m), n_x)

                for xi, s in enumerate(s_list):
                    xsec = cross_section_at(line_d, float(s), half_width_m)
                    dists, elevs = sample_dem_along_line(dem, xsec, sample_step_m)
                    res = classify_uv(dists, elevs, floor_rel_height, u_threshold, v_threshold, min_depth_m)

                    center = line_d.interpolate(float(s))
                    rec = {
                        "feature_index": int(feat_idx),
                        "line_index": int(li),
                        "xsec_index": int(xi),
                        "catchment": catchment,
                        "glacier": glacier,
                        "lake": lake,
                        "center_s_m": float(s),
                        "center_x": float(center.x),
                        "center_y": float(center.y),
                        **{k: res.get(k, np.nan) for k in ["class", "depth", "floor_width_ratio", "total_width", "floor_width", "z_min", "z_max"]},
                    }
                    records.append(rec)

                    if save_geojson:
                        xsec_features.append({
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": [(float(x), float(y)) for x, y in xsec.coords]},
                            "properties": {
                                "catchment": catchment,
                                "glacier": glacier,
                                "lake": lake,
                                "feature_index": int(feat_idx),
                                "line_index": int(li),
                                "xsec_index": int(xi),
                                "valley_class": res.get("class", "invalid"),
                                "depth_m": None if np.isnan(res.get("depth", np.nan)) else float(res.get("depth")),
                                "floor_width_ratio": None if np.isnan(res.get("floor_width_ratio", np.nan)) else float(res.get("floor_width_ratio")),
                            },
                        })
                        center_features.append({
                            "type": "Feature",
                            "geometry": {"type": "Point", "coordinates": [float(center.x), float(center.y)]},
                            "properties": {
                                "catchment": catchment,
                                "glacier": glacier,
                                "lake": lake,
                                "feature_index": int(feat_idx),
                                "line_index": int(li),
                                "xsec_index": int(xi),
                                "valley_class": res.get("class", "invalid"),
                            },
                        })

        df = pd.DataFrame(records)
        df.to_csv(out / "valley_shape_profiles.csv", index=False)
        print(f"[OK] Wrote CSV: {(out / 'valley_shape_profiles.csv').resolve()}")

        if save_geojson and xsec_features:
            (out / "valley_cross_sections.geojson").write_text(
                __import__("json").dumps({"type": "FeatureCollection", "features": xsec_features}, ensure_ascii=False),
                encoding="utf-8"
            )
            (out / "valley_xsec_centers.geojson").write_text(
                __import__("json").dumps({"type": "FeatureCollection", "features": center_features}, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"[OK] Wrote GeoJSON: {(out / 'valley_cross_sections.geojson').resolve()}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for valley-shape classification.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments for DEM/streams paths and classification parameters.
    """
    p = argparse.ArgumentParser(description="Classify valley shape (U/V) from DEM cross-sections along streams.")
    p.add_argument("--dem", required=True)
    p.add_argument("--streams", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--spacing", type=float, default=200.0, help="Cross-section spacing along stream [m]")
    p.add_argument("--half-width", type=float, default=250.0, help="Half-width of cross-section [m]")
    p.add_argument("--step", type=float, default=10.0, help="Sampling step along cross-section [m]")

    p.add_argument("--floor-rel", type=float, default=0.2, help="Relative floor threshold (0..1)")
    p.add_argument("--u-thresh", type=float, default=0.30)
    p.add_argument("--v-thresh", type=float, default=0.15)
    p.add_argument("--min-depth", type=float, default=10.0)

    p.add_argument("--save-geojson", action="store_true", help="Also write QGIS-ready GeoJSON outputs")
    return p.parse_args()


def main() -> None:
    """Entry point for command-line execution of the valley-shape script."""
    a = parse_args()
    run(
        dem_path=a.dem,
        streams_path=a.streams,
        outdir=a.outdir,
        spacing_m=a.spacing,
        half_width_m=a.half_width,
        sample_step_m=a.step,
        floor_rel_height=a.floor_rel,
        u_threshold=a.u_thresh,
        v_threshold=a.v_thresh,
        min_depth_m=a.min_depth,
        save_geojson=a.save_geojson,
    )


if __name__ == "__main__":
    main()
