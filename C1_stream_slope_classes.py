#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C1 — Channel slope classification along 3D stream centerlines.

Purpose (Appendix):
  From 3D stream polylines (x,y,z), compute distance-elevation profiles and local
  segment slopes, classify slope into 3 bins, and export:
    (i) per-stream summary statistics (CSV),
    (ii) per-point longitudinal profile table (CSV),
    (iii) slope-classified stream segments (GeoJSON).

Why this matters in the thesis:
  The derived slope metrics (median slope, fraction of 10–25° segments, etc.)
  are used as geomorphic controls in the event table and subsequent statistics.

Inputs:
  - 3D stream centerlines GeoJSON with LineString/MultiLineString geometry.
    Coordinates must be (x, y, z) in a projected CRS (meters).

Outputs:
  - stream_stats_slope_classes.csv
  - stream_longitudinal_profile_points.csv
  - streams_slope_classes.geojson

Dependencies:
  pip install numpy pandas matplotlib

Example:
  python A1_stream_slope_classes.py \
    --in ./03_Geländeanalyse/Stream_Längsprofil_mit_Höhe.geojson \
    --outdir ./03_Geländeanalyse/slope_output
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Core computations
# ---------------------------

def _extract_lines_3d(feature: Dict) -> List[List[Tuple[float, float, float]]]:
    """Extract 3D LineString geometries from a GeoJSON feature.

    Parameters
    ----------
    feature : dict
        A GeoJSON feature containing a geometry of type 'LineString' or
        'MultiLineString' with coordinates (x, y, z) in a projected CRS.

    Returns
    -------
    list of list of tuple(float, float, float)
        A list of 3D polylines. Each polyline is a list of (x, y, z) tuples.
        Consecutive duplicate points are removed. Polylines with < 2 points
        are discarded.

    Raises
    ------
    ValueError
        If any coordinate has fewer than 3 elements (i.e., not truly 3D).
    """
    geom = feature.get("geometry", {}) or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates")

    if gtype == "LineString":
        lines = [coords]
    elif gtype == "MultiLineString":
        lines = coords
    else:
        return []

    out = []
    for line in lines:
        if not line:
            continue
        pts = []
        for c in line:
            if len(c) < 3:
                raise ValueError("Stream coordinates must be 3D (x, y, z).")
            pts.append((float(c[0]), float(c[1]), float(c[2])))

        # remove duplicate consecutive points
        cleaned = [pts[0]]
        for p in pts[1:]:
            if p != cleaned[-1]:
                cleaned.append(p)

        if len(cleaned) >= 2:
            out.append(cleaned)

    return out


def chainage_m(xyz: List[Tuple[float, float, float]]) -> List[float]:
    """Compute cumulative planimetric distance (chainage) along a 3D polyline.

    The distance is computed in the horizontal plane using (x, y) only.

    Parameters
    ----------
    xyz : list of tuple(float, float, float)
        Polyline vertices as (x, y, z) in meters.

    Returns
    -------
    list of float
        Cumulative distance in meters for each vertex (same length as `xyz`),
        starting at 0.0.
    """
    d = [0.0]
    for i in range(1, len(xyz)):
        x0, y0, _ = xyz[i - 1]
        x1, y1, _ = xyz[i]
        d.append(d[-1] + math.dist((x0, y0), (x1, y1)))
    return d


def segment_slopes_deg(dist_m: List[float], elev_m: List[float]) -> Tuple[List[float], List[float]]:
    """Compute local slope angles (degrees) for each segment of a profile.

    Slope is computed as arctan(|dz| / dx) in degrees using the distance
    increment `dx` from `dist_m` and elevation increment `dz` from `elev_m`.
    The absolute value of dz is used (i.e., slope magnitude).

    Parameters
    ----------
    dist_m : list of float
        Cumulative distance along the polyline [m], length n.
    elev_m : list of float
        Elevation values [m], length n.

    Returns
    -------
    slopes_per_point : list of float
        Slope per vertex [deg], length n. The first element is 0.0 by
        convention (no segment before first point).
    segment_lengths : list of float
        Segment lengths [m], length n-1.

    Notes
    -----
    Segments with non-positive dx are assigned slope 0.0.
    """
    n = len(elev_m)
    slopes = [0.0]
    seglen = []
    for i in range(n - 1):
        dx = dist_m[i + 1] - dist_m[i]
        dz = elev_m[i + 1] - elev_m[i]
        if dx <= 0:
            s = 0.0
        else:
            s = math.degrees(math.atan(abs(dz) / dx))
        slopes.append(float(s))
        seglen.append(float(dx))
    return slopes, seglen


def slope_class(s_deg: float) -> str:
    """Classify a slope angle into discrete bins used in the thesis.

    Parameters
    ----------
    s_deg : float
        Slope angle in degrees.

    Returns
    -------
    str
        One of:
        - 'lt10'    for slopes < 10°
        - '10to25'  for 10° <= slopes < 25°
        - 'gt25'    for slopes >= 25°
    """
    if s_deg < 10.0:
        return "lt10"
    if s_deg < 25.0:
        return "10to25"
    return "gt25"


def merge_segments_by_class(
    xyz: List[Tuple[float, float, float]],
    seg_classes: List[str],
) -> List[Tuple[str, List[Tuple[float, float, float]]]]:
    """Merge consecutive segments with identical slope class into longer lines.

    Given a polyline and a per-segment class label, this function groups
    adjacent segments that share the same class into a list of longer
    LineStrings (as coordinate lists).

    Parameters
    ----------
    xyz : list of tuple(float, float, float)
        Polyline vertices (x, y, z), length n.
    seg_classes : list of str
        Segment classes, length n-1 (one label per segment).

    Returns
    -------
    list of (str, list of tuple(float, float, float))
        A list of (class_label, coords) groups, where coords is a list of
        vertices representing a merged LineString with >= 2 points.

    Raises
    ------
    AssertionError
        If `len(seg_classes) != len(xyz) - 1`.
    """
    assert len(seg_classes) == len(xyz) - 1
    if not seg_classes:
        return []

    groups = []
    cls = seg_classes[0]
    coords = [xyz[0]]

    for i, c in enumerate(seg_classes):
        p_next = xyz[i + 1]
        if c == cls:
            coords.append(p_next)
        else:
            groups.append((cls, coords))
            cls = c
            coords = [xyz[i], p_next]

    groups.append((cls, coords))
    return [g for g in groups if len(g[1]) >= 2]


# ---------------------------
# I/O helpers
# ---------------------------

def read_geojson(path: str) -> Dict:
    """Read a GeoJSON file into a Python dictionary.

    Parameters
    ----------
    path : str
        Path to a GeoJSON file.

    Returns
    -------
    dict
        Parsed GeoJSON object (typically a FeatureCollection).
"""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_geojson(features: List[Dict], path: str) -> None:
    """Write a list of GeoJSON features as a FeatureCollection.

    Parameters
    ----------
    features : list of dict
        GeoJSON Feature dictionaries.
    path : str
        Output path for the GeoJSON file. Parent directories are created if
        necessary.

    Returns
    -------
    None
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    p.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")


def savefig(path: str) -> None:
    """Save the current Matplotlib figure and close it.

    This helper ensures the output directory exists, applies tight layout,
    writes the figure, and closes it to free memory.

    Parameters
    ----------
    path : str
        Output file path (e.g., PNG).

    Returns
    -------
    None
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------
# Main processing
# ---------------------------

def process_streams(in_geojson: str, outdir: str, make_plots: bool = False) -> None:
    """Process 3D stream centerlines to derive slope metrics and slope classes.

    Workflow
    --------
    1) Read 3D stream geometries (LineString/MultiLineString) from GeoJSON
    2) Compute chainage, elevation profile, and segment slopes
    3) Classify slopes into bins (<10°, 10–25°, >25°)
    4) Export:
    - per-stream summary statistics (CSV)
    - per-point longitudinal profile table (CSV)
    - slope-classified segments (GeoJSON)
    5) Optionally export longitudinal-profile plots

    Parameters
    ----------
    in_geojson : str
        Path to input GeoJSON containing 3D stream centerlines (x, y, z).
    outdir : str
        Output directory.
    make_plots : bool, default=False
        If True, write a PNG longitudinal profile for each line.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the input GeoJSON contains no features.
    """
    outdir = str(outdir)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    data = read_geojson(in_geojson)
    features = data.get("features", [])
    if not features:
        raise ValueError("Input GeoJSON contains no features.")

    stats_rows = []
    point_rows = []
    seg_features = []

    for fi, feat in enumerate(features):
        props = feat.get("properties", {}) or {}
        lines = _extract_lines_3d(feat)
        if not lines:
            continue

        # metadata (keep only thesis-relevant keys if present)
        catchment = props.get("NameCatch", props.get("catchment", ""))
        glacier = props.get("Glacier", props.get("glacier", ""))
        lake = props.get("Lake", props.get("lake", ""))

        for li, xyz in enumerate(lines):
            dist = chainage_m(xyz)
            elev = [p[2] for p in xyz]
            slopes_pt, seglen = segment_slopes_deg(dist, elev)
            seg_slopes = slopes_pt[1:]
            seg_classes = [slope_class(s) for s in seg_slopes]

            # ---- per-point table ----
            for pi, (x, y, z) in enumerate(xyz):
                point_rows.append({
                    "feature": fi,
                    "line_index": li,
                    "point_index": pi,
                    "catchment": catchment,
                    "glacier": glacier,
                    "lake": lake,
                    "dist_m": round(float(dist[pi]), 3),
                    "elev_m": round(float(z), 3),
                    "slope_deg": round(float(slopes_pt[pi]), 3),
                    "slope_class": slope_class(slopes_pt[pi]) if pi > 0 else "",
                })

            # ---- summary stats (length-weighted) ----
            total_m = float(dist[-1])
            if total_m <= 0 or sum(seglen) <= 0:
                continue

            wmean = float(sum(s * l for s, l in zip(seg_slopes, seglen)) / sum(seglen))
            median = float(np.median(seg_slopes))
            maxs = float(np.max(seg_slopes))

            def frac(cls: str) -> float:
                l = sum(L for c, L in zip(seg_classes, seglen) if c == cls)
                return 100.0 * l / sum(seglen)

            stats_rows.append({
                "feature": fi,
                "line_index": li,
                "catchment": catchment,
                "glacier": glacier,
                "lake": lake,
                "length_km": round(total_m / 1000.0, 3),
                "slope_mean_deg": round(wmean, 3),
                "slope_median_deg": round(median, 3),
                "slope_max_deg": round(maxs, 3),
                "slope_lt10_frac_pct": round(frac("lt10"), 3),
                "slope_10_25_frac_pct": round(frac("10to25"), 3),
                "slope_gt25_frac_pct": round(frac("gt25"), 3),
            })

            # ---- slope-class segments (GeoJSON) ----
            for gi, (cls, coords) in enumerate(merge_segments_by_class(xyz, seg_classes)):
                seg_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[x, y] for (x, y, _z) in coords],
                    },
                    "properties": {
                        "feature": fi,
                        "line_index": li,
                        "segment_index": gi,
                        "catchment": catchment,
                        "glacier": glacier,
                        "lake": lake,
                        "slope_class": cls,
                    },
                })

            # ---- optional plot (profile) ----
            if make_plots:
                plt.figure(figsize=(8, 4.5))
                plt.plot([d / 1000 for d in dist], elev)
                plt.xlabel("Distance [km]")
                plt.ylabel("Elevation [m]")
                plt.title(f"Longitudinal profile (feature {fi}, line {li})")
                savefig(str(out / "plots" / f"profile_f{fi}_l{li}.png"))

    # Write outputs
    pd.DataFrame(stats_rows).to_csv(out / "stream_stats_slope_classes.csv", index=False)
    pd.DataFrame(point_rows).to_csv(out / "stream_longitudinal_profile_points.csv", index=False)
    write_geojson(seg_features, str(out / "streams_slope_classes.geojson"))

    print(f"[OK] Wrote outputs to: {out.resolve()}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the slope classification script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - in_geojson : str
        - outdir : str
        - plots : bool
    """
    p = argparse.ArgumentParser(description="Compute slope classes and longitudinal-profile metrics from 3D stream lines.")
    p.add_argument("--in", dest="in_geojson", required=True, help="3D stream centerlines GeoJSON")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--plots", action="store_true", help="Also write simple longitudinal-profile plots (PNG)")
    return p.parse_args()


def main() -> None:
    """Entry point for command-line execution.

    Parses arguments and runs the processing pipeline.

    Returns
    -------
    None
    """
    args = parse_args()
    process_streams(args.in_geojson, args.outdir, make_plots=args.plots)


if __name__ == "__main__":
    main()
