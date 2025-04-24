# utils.py

import polars as pl
import datetime
from typing import Dict, Tuple, Optional, List, Union
import logging
import sys

# --- Constants ---
EARTH_RADIUS_KM = 6371.0
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in km using Haversine formula.
    """
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c


# --- Functions ---


def filter_chinese_license_plates(
    lazy_df: pl.LazyFrame,
    col: str = "license_plate",
) -> pl.LazyFrame:
    """Filters Chinese license plates following official standards."""
    string_col = pl.col(col).cast(pl.Utf8)

    is_valid = (
        # Standard plates (7 characters)
        (
            (string_col.str.len_chars() == 7)
            & (string_col.str.slice(0, 1).str.contains(r"[\u4e00-\u9fa5]"))
            & (string_col.str.slice(1, 1).str.contains(r"[A-HJ-NP-Z]"))
            & (string_col.str.slice(2, 5).str.contains(r"^[A-HJ-NP-Z0-9]{5}$"))
        )
        |
        # New energy plates (8 characters)
        (
            (string_col.str.len_chars() == 8)
            & (string_col.str.slice(0, 1).str.contains(r"[\u4e00-\u9fa5]"))
            & (string_col.str.slice(1, 1).str.contains(r"[A-HJ-NP-Z]"))
            & (
                string_col.str.slice(2, 6).str.contains(r"^[A-HJ-NP-Z0-9]{5}$")
            )  # Corrected slice
            & (string_col.str.slice(7, 1).str.contains(r"[DF]$"))
        )
    )
    return lazy_df.filter(pl.col(col).is_not_null() & is_valid)


def profile_data(
    ldf: pl.LazyFrame, columns: Optional[List[str]] = None
) -> Tuple[pl.DataFrame, Dict]:
    """
    Simple data profiling using Polars built-ins.
    Returns:
        - DataFrame with basic statistics
        - Dictionary with additional metrics
    """
    try:
        df = ldf.collect()

        # Handle empty DataFrame
        if df.is_empty() or len(df.columns) == 0:
            return pl.DataFrame(), {}

        # Select only specified columns if provided
        if columns:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            df = df.select(columns)

        # Get basic statistics
        describe_df = df.describe()

        # Get missing values stats
        missing_stats = df.select(
            pl.all().null_count().name.prefix("null_count_"),
            (pl.all().null_count() / pl.len() * 100).name.prefix("null_pct_"),
        )

        # Get unique counts
        unique_stats = df.select(pl.all().n_unique())

        # Combine into final output
        stats = {}
        for col in df.columns:
            stats[col] = {
                "dtype": str(df.schema[col]),
                "missing_count": missing_stats[f"null_count_{col}"][0],
                "missing_%": missing_stats[f"null_pct_{col}"][0],
                "unique": unique_stats[col][0],
            }
            describe_stats = {}
            for stat in ["mean", "std", "min", "max", "median"]:
                value = describe_df.filter(pl.col("statistic") == stat)[col].to_list()
                if value:
                    describe_stats[stat] = value[0]
            stats[col].update(describe_stats)

        # Convert to DataFrame
        stats_df = pl.DataFrame(
            [{"column": col, **vals} for col, vals in stats.items()]
        )

        return stats_df, stats

    except Exception as e:
        raise RuntimeError(f"Data profiling failed: {str(e)}") from e


def filter_by_date(
    lazy_df: pl.LazyFrame,
    timestamp_col: str = "timestamp",
    correct_year: int = 2019,
    start_date: Optional[Union[datetime.date, datetime.datetime]] = None,
    end_date: Optional[Union[datetime.date, datetime.datetime]] = None,
) -> pl.LazyFrame:
    """
    Filters rows based on the timestamp column's year or a custom date range.

    Args:
        lazy_df: Input LazyFrame
        timestamp_col: Name of the timestamp column (default: "timestamp")
        correct_year: Target year to filter by. Default is 2019. Ignored if start_date or end_date are provided.
        start_date: Start of the date range (inclusive). If provided, overrides correct_year.
        end_date: End of the date range (inclusive). If provided, overrides correct_year.

    Returns:
        Filtered LazyFrame containing rows within the specified year or date range.
    """
    # Validate timestamp column
    schema = lazy_df.collect_schema()
    if timestamp_col not in schema:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame")

    # Check if column is datetime type (works even for empty DataFrames)
    if not isinstance(schema[timestamp_col], (pl.Datetime, pl.Date)):
        raise ValueError(f"Column '{timestamp_col}' must be a datetime or date type")

    # Return empty DataFrame early if input is empty
    if lazy_df.collect().is_empty():
        return lazy_df

    # Determine filtering conditions
    conditions = []
    if start_date is not None or end_date is not None:
        if start_date is not None:
            conditions.append(pl.col(timestamp_col) >= start_date)
        if end_date is not None:
            conditions.append(pl.col(timestamp_col) <= end_date)
    else:
        conditions.append(pl.col(timestamp_col).dt.year() == correct_year)

    # Combine conditions and filter
    combined_condition = conditions[0]
    for cond in conditions[1:]:
        combined_condition = combined_condition & cond

    return lazy_df.filter(combined_condition)


def configure_logging():
    """Configure logging to stdout"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def add_time_distance_calcs(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Add time difference and Haversine distance calculations"""
    return (
        lazy_df.with_columns(
            [
                pl.col("timestamp")
                .diff(1)
                .over("license_plate")
                .cast(pl.Duration)
                .alias("time_diff_duration"),
                pl.col("latitude")
                .shift(1)
                .over("license_plate")
                .alias("prev_latitude"),
                pl.col("longitude")
                .shift(1)
                .over("license_plate")
                .alias("prev_longitude"),
            ]
        )
        .with_columns(
            [
                pl.col("time_diff_duration")
                .dt.total_seconds()
                .alias("time_diff_seconds"),
                pl.col("latitude").radians().alias("lat_rad"),
                pl.col("longitude").radians().alias("lon_rad"),
                pl.col("prev_latitude").radians().alias("prev_lat_rad"),
                pl.col("prev_longitude").radians().alias("prev_lon_rad"),
            ]
        )
        .with_columns(
            [
                ((pl.col("lat_rad") - pl.col("prev_lat_rad")) / 2)
                .sin()
                .pow(2)
                .alias("sin_dlat_half_sq"),
                ((pl.col("lon_rad") - pl.col("prev_lon_rad")) / 2)
                .sin()
                .pow(2)
                .alias("sin_dlon_half_sq"),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("sin_dlat_half_sq")
                    + pl.col("lat_rad").cos()
                    * pl.col("prev_lat_rad").cos()
                    * pl.col("sin_dlon_half_sq")
                ).alias("haversine_a")
            ]
        )
        .with_columns(
            [
                (
                    pl.lit(2.0)
                    * pl.arctan2(
                        pl.col("haversine_a").sqrt(),
                        (pl.lit(1.0) - pl.col("haversine_a")).sqrt(),
                    )
                ).alias("haversine_c")
            ]
        )
        .with_columns(
            [
                (pl.lit(EARTH_RADIUS_KM) * pl.col("haversine_c"))
                .fill_null(0.0)
                .alias("distance_km")
            ]
        )
    )


def add_implied_speed(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Calculate speed in km/h between points"""
    return lazy_df.with_columns(
        [
            pl.when(pl.col("time_diff_seconds") > 0)
            .then(pl.col("distance_km") / (pl.col("time_diff_seconds") / 3600.0))
            .otherwise(0.0)
            .alias("implied_speed_kph")
        ]
    )


def add_abnormality_flags(
    lazy_df: pl.LazyFrame, gap_threshold_sec: float, speed_threshold_kph: float
) -> pl.LazyFrame:
    """Add flags for temporal gaps and position jumps"""
    return lazy_df.with_columns(
        [
            (pl.col("time_diff_seconds") > gap_threshold_sec).alias("is_temporal_gap"),
            (pl.col("implied_speed_kph") > speed_threshold_kph).alias(
                "is_position_jump"
            ),
        ]
    )


def select_final_columns(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """Select and format final output columns"""
    return lazy_df.select(
        [
            "license_plate",
            "timestamp",
            "longitude",
            "latitude",
            "occupancy_status",
            "time_diff_duration",
            pl.col("time_diff_seconds").round(2),
            pl.col("distance_km").round(3),
            "instant_speed",
            pl.col("implied_speed_kph").round(2),
            "is_temporal_gap",
            "is_position_jump",
        ]
    )


def add_period_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    Identify continuous periods per license_plate and occupancy_status.
    Assign period IDs starting at 1 and increment on changes.
    """
    # Identify changes in license_plate or occupancy_status compared to previous row
    license_plate = pl.col("license_plate")
    occupancy_status = pl.col("occupancy_status")

    prev_license_plate = license_plate.shift(1)
    prev_occupancy_status = occupancy_status.shift(1)

    change = (license_plate != prev_license_plate) | (
        occupancy_status != prev_occupancy_status
    )

    # Mark first row as a new period and compute cumulative sum to assign period IDs
    change_filled = change.fill_null(True).cast(pl.Int32)
    period_id = change_filled.cum_sum().alias("period_id")

    return df.with_columns(period_id)


def summarize_periods(df: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate statistics per license_plate, occupancy_status, period_id.
    Expects columns: license_plate, occupancy_status, period_id,
                     timestamp, implied_speed_kph, time_diff_seconds, distance_km,
                     latitude, longitude.
    Returns start_time, end_time, duration, count_rows, avg_implied_speed, sum_time_diff,
            sum_distance, straight_line_distance_km per group.
    """
    if df.is_empty():
        # Return empty with expected schema if input empty
        return pl.DataFrame(
            {
                "license_plate": pl.Series([], dtype=pl.Utf8),
                "occupancy_status": pl.Series([], dtype=pl.Utf8),
                "period_id": pl.Series([], dtype=pl.Int32),
                "start_time": pl.Series([], dtype=pl.Datetime),
                "end_time": pl.Series([], dtype=pl.Datetime),
                "duration": pl.Series([], dtype=pl.Duration),
                "count_rows": pl.Series([], dtype=pl.UInt32),
                "avg_implied_speed": pl.Series([], dtype=pl.Float64),
                "sum_time_diff": pl.Series([], dtype=pl.Float64),
                "sum_distance": pl.Series([], dtype=pl.Float64),
                "straight_line_distance_km": pl.Series([], dtype=pl.Float64),
            }
        )

    # Determine if latitude/longitude columns are available
    has_latlon = ("latitude" in df.columns) and ("longitude" in df.columns)

    # Build aggregation expressions
    agg_exprs = [
        pl.col("timestamp").min().alias("start_time"),
        pl.col("timestamp").max().alias("end_time"),
        pl.count().alias("count_rows"),
        pl.col("implied_speed_kph").mean().alias("avg_implied_speed"),
        pl.col("time_diff_seconds").sum().alias("sum_time_diff"),
        pl.col("distance_km").sum().alias("sum_distance"),
    ]
    if has_latlon:
        agg_exprs.extend(
            [
                pl.col("latitude").first().alias("start_latitude"),
                pl.col("longitude").first().alias("start_longitude"),
                pl.col("latitude").last().alias("end_latitude"),
                pl.col("longitude").last().alias("end_longitude"),
            ]
        )

    grouped = (
        df.group_by(["license_plate", "occupancy_status", "period_id"])
        .agg(agg_exprs)
        .sort(["license_plate", "period_id"])
    )

    # Compute duration
    grouped = grouped.with_columns(
        (pl.col("end_time") - pl.col("start_time")).alias("duration")
    )

    # Compute straight line distance if latitude/longitude available
    if has_latlon:
        start_lats = grouped["start_latitude"].to_list()
        start_lons = grouped["start_longitude"].to_list()
        end_lats = grouped["end_latitude"].to_list()
        end_lons = grouped["end_longitude"].to_list()

        distances = [
            haversine_distance(lat1, lon1, lat2, lon2)
            for lat1, lon1, lat2, lon2 in zip(start_lats, start_lons, end_lats, end_lons)
        ]

        grouped = grouped.with_columns(
            pl.Series("straight_line_distance_km", distances)
        )

        # Drop intermediate latitude/longitude columns
        grouped = grouped.drop(
            ["start_latitude", "start_longitude", "end_latitude", "end_longitude"]
        )

    return grouped
