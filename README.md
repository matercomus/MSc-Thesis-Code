# Taxi Location Data Preprocessing & Abnormality Detection

## Project Overview

This repository implements a complete pipeline for processing, cleaning, and analyzing Beijing taxi GPS trajectory data. Key components and their purposes:
  - **data.py**: ingests raw CSV logs and converts them to compressed Parquet with explicit schema for efficient I/O.
  - **utils.py**: provides core utilities for filtering (license plates, dates), feature engineering (time deltas, haversine distance, implied speeds), anomaly flagging, and period segmentation/summarization.
  - **geo.py / geo2.py**: apply geographic filters (bounding box and spatial join) to restrict data to the Beijing region.
  - **new_indicators_pipeline.py**: the main, pure Python script for the full Beijing taxi trajectory cleaning and analysis pipeline. No Marimo, no notebooks—just a clean script. Uses tqdm for progress bars and logging for status updates. It:
    - Loads filtered points from `data/filtered_points_in_beijing.parquet`.
    - Computes robust IQR-based thresholds for time gaps and speeds (on occupied trips only).
    - Computes time, distance, speed, and abnormality flags for each point.
    - Detects outliers using Isolation Forest (scikit-learn).
    - Removes all points for any taxi with a temporal gap or position jump.
    - Adds period IDs and summarizes periods, removing small periods (<3 points).
    - Attaches period IDs to cleaned points and saves to `data/cleaned_with_period_id_in_beijing.parquet`.
    - Computes per-period outlier counts and flags period-level outliers (Isolation Forest and SLD ratio).
    - Flags SLD-based outliers using IQR threshold and saves enriched period summaries to `data/periods_with_sld_ratio.parquet`.
    - All outputs are written to the `data/` folder.
  - **threshold_analysis.py**: standalone analysis script for empirical investigation of time-gap and speed distributions, devising robust thresholds (IQR-based) and generating CSV/PNG artifacts in a timestamped output folder.
  - **trajectory_viewer.py**: Streamlit application for interactive exploration of per-period and per-point data, including anomaly flags and map visualization.

**Primary outputs**:
  - All Parquet outputs are stored in the `data/` folder (which is git-ignored):
    - `data/filtered_points_in_beijing.parquet`: filtered raw points in Beijing.
    - `data/cleaned_points_in_beijing.parquet`: points after removing taxis with point-level anomalies.
    - `data/cleaned_with_period_id_in_beijing.parquet`: cleaned points annotated with period IDs.
    - `data/periods_in_beijing.parquet`, `data/periods_with_sld_ratio.parquet`: enriched per-period summaries.
  - Timestamped result directories from `threshold_analysis.py` containing detailed CSV and PNG investigation artifacts (these are not in `data/`).

## 1. Data Compression

 **File:** `data.py`
 - Convert CSV → Parquet using Polars (`scan_csv` + `sink_parquet`)
 - Default compression: **zstd** (also supports snappy, gzip, lz4)
 - Schema overrides & precise dtypes:
   - `license_plate`: Categorical
   - `timestamp`: Datetime
   - `longitude`, `latitude`: Float64
   - `instant_speed`: Float64
   - `occupancy_status`: Int8
 

## 2. Data Cleaning & Filtering

### 2.1 License-Plate Validation

 **File:** `utils.py` → `filter_chinese_license_plates`
 - Keep only valid 7-char standard plates or 8-char new-energy plates
 - Regex on Chinese character + letter + alphanumeric
 

### 2.2 Date & Time Filtering

 **File:** `utils.py` → `filter_by_date`
 - Default: year == 2019
 - Optional: arbitrary `start_date` / `end_date` range (inclusive)
 

### 2.3 Geographic Filtering (Beijing)

 **Files:** `geo.py`, `geo2.py`
 1. Bounding-box filter on lon/lat
 2. Convert to GeoPandas points + spatial-join vs Beijing polygon (GADM shapefile)
 3. Output `filtered_points_in_beijing.parquet`
 

### 2.4 Suspicious-Taxi Removal

 **File:** `new_indicators_pipeline.py`
 - Flags any taxi with a temporal gap or position jump
 - Removes **all** points for any flagged taxi
 - Output `data/cleaned_points_in_beijing.parquet`
 

## 3. Feature Engineering & Transformation

 **File:** `utils.py`
 1. **Time & Distance** (`add_time_distance_calcs`)
    - `time_diff_duration`, `time_diff_seconds`
    - Haversine distance → `distance_km`
 2. **Speed Calculation** (`add_implied_speed`)
    - `implied_speed_kph`
 3. **Abnormality Flags** (`add_abnormality_flags`)
    - `is_temporal_gap` (`>300s`)
    - `is_position_jump` (`>200 kph`)
 4. **Select Final Columns** (`select_final_columns`)
    ```
    [license_plate, timestamp, longitude, latitude,
     occupancy_status, time_diff_duration, time_diff_seconds,
     distance_km, instant_speed, implied_speed_kph,
     is_temporal_gap, is_position_jump]
    ```
 5. **Period Segmentation**
    - `add_period_id`: increment on plate or occupancy change
    - `summarize_periods`: aggregate per (plate, status, period_id)
      - `start_time`, `end_time`, `duration`, `count_rows`
      - `avg_implied_speed`, `sum_time_diff`, `sum_distance`
      - `straight_line_distance_km`, `sld_ratio`
 

## 4. Anomaly-Detection Indicators (in `new_indicators_pipeline.py`)

- **Point-Level Anomalies**:
  - *Temporal Gaps*: time_diff_seconds > IQR-based threshold (typically ~90s)
  - *Spatial Jumps*: implied_speed_kph > IQR-based threshold (typically ~140-200 kph)

- **Plate-Level Filtering**:
  - Remove any taxi with one or more point-level anomalies before period summarization

- **Period-Level Anomalies**:
  - *Isolation Forest (IF)*: flag periods with any IF-detected point outliers (`is_traj_outlier`)
  - *SLD Ratio*: sum_distance / straight_line_distance_km; periods with sld_ratio > Q3 + 1.5×IQR are flagged (`is_sld_outlier`)

## 5. Example Visualizations

The pipeline and viewer produce several types of visualizations:

- **Histograms** (from `threshold_analysis.py`):
  - Distribution of time gaps (`time_diff_seconds`) and implied speeds (`implied_speed_kph`), both raw and after IQR-based cleaning.
  - Outlier-only distributions for both features.
  - Examples (generated in the latest run, see `threshold_analysis_*/`):

    **Time Difference Distributions:**
    - Raw: ![Time Diff Raw](threshold_analysis_20250509_161016/hist_time_diff_raw.png)
    - Clean (≤ IQR): ![Time Diff Clean](threshold_analysis_20250509_161016/hist_time_diff_clean.png)
    - Outliers (> IQR): ![Time Diff Outliers](threshold_analysis_20250509_161016/hist_time_diff_outliers.png)

    **Speed Distributions:**
    - Raw: ![Speed Raw](threshold_analysis_20250509_161016/hist_speed_raw.png)
    - Clean (≤ IQR): ![Speed Clean](threshold_analysis_20250509_161016/hist_speed_clean.png)
    - Outliers (> IQR): ![Speed Outliers](threshold_analysis_20250509_161016/hist_speed_outliers.png)

- **Interactive Trajectory Viewer** (`trajectory_viewer.py`):
  - Map-based and XY plots of individual taxi trajectories for selected periods.
  - Table of period-level metadata and outlier flags.
  - Sidebar filters for SLD/IF outlier status, occupancy, and more.
  - Example screenshot:  
    *(Add your own screenshot here, e.g. `docs/example_trajectory_viewer.png`)*

## 6. Threshold Analysis Investigation

**File:** `threshold_analysis.py`
- Performs an in-depth exploration of time gaps (`time_diff_seconds`) and implied speeds (`implied_speed_kph`) on **contiguous occupied** segments (occupancy_status=1 and previous point also occupied).
- Creates a **timestamped output directory** per run (e.g. `threshold_analysis_YYYYMMDD_HHMMSS/`) containing:
  - **CSV summaries** (`describe`) for raw vs. cleaned (≤ IQR) distributions.
  - **Percentile tables** for three tiers:
    1. **Raw** (90, 95, 97.5, 99, 99.5, 99.9 percentiles)
    2. **Clean** (values ≤ IQR threshold; 90, 95, 97.5)
    3. **Outliers** (> IQR threshold; 90, 95)
  - **Histograms**:
    - Raw distribution (log scale)
    - Clean distribution (linear, IQR threshold line)
    - Outlier-only distribution
- **IQR-based thresholds** (Q3 + 1.5×IQR) on occupied segments:
  - Time gap threshold ≃ 92.5 seconds
  - Speed threshold ≃ 146.8 km/h
- **Investigation findings**:
  - Raw 99th percentile speed ≃ 810 km/h; cleaned (≤146.8 km/h) 97.5th percentile ≃ 89 km/h.
  - Extreme GPS jumps detected (e.g. taxi 京BQ9046 moves ~53 km in 1 s ⇒ ≃194 000 km/h).
  - Reveals need for a **pre-filter** on point-level anomalies (e.g. drop Δd >1 km for Δt ≤1 s or speed >200 km/h) before downstream analysis.

## 7. Testing

- All tests are located in `tests.py` in the project root (not in a `tests/` folder).
- Pytest is configured via `pyproject.toml` to discover tests in this file.
- To run tests, use:
  ```bash
  uv run pytest
  ```
  (Requires [uv](https://github.com/astral-sh/uv) and Python 3.11+)

## 8. Data Folder and Version Control

- All intermediate and output Parquet files are stored in the `data/` directory.
- The `data/` folder is included in `.gitignore` and is not tracked by git.
- If you need to share or archive data, copy files from `data/` manually.