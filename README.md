# Appendix code

This folder contains *cleaned, thesis-relevant* versions of the four Python workflows
used to derive geomorphic metrics and run the statistical analysis.

## C1 — Stream slope classes
**File:** `A1_stream_slope_classes.py`

- Input: 3D stream centerlines GeoJSON (x,y,z)
- Output:
  - `stream_stats_slope_classes.csv` (length-weighted slope metrics per line)
  - `stream_longitudinal_profile_points.csv` (distance–elevation–slope per point)
  - `streams_slope_classes.geojson` (2D segments with slope_class for QGIS)

## C2 — Valley shape (U/V) from DEM cross-sections
**File:** `A2_valley_shape_uv.py`

- Input: DEM GeoTIFF + stream centerlines
- Output:
  - `valley_shape_profiles.csv` (per cross-section metrics + class)
  - optional GeoJSON for QGIS (`--save-geojson`)

## C3 — Sediment-source shares along the centerline
**File:** `A3_sediment_length_shares.py`

- Input: sediment polygons with `Date` and `SediSource` + centerline
- Output:
  - event-level length totals and shares per sediment category (CSV)

## C4 — Statistics / models / appendix figures
**File:** `A4_event_statistics_models.py`

- Input: final event table CSV (as used in the thesis)
- Output:
  - `spearman_summary.csv`
  - `model_comparison_BV.csv`
  - Appendix panel figures `C1`–`C4` (PNG)

## Notes for reproducibility
- Each script is self-contained and runnable from the command line.
- All file paths are passed as CLI arguments (no hard-coded project paths).
- Outputs are written into the specified output directories.
