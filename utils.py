# utils.py

import polars as pl
import datetime
from typing import Dict, Tuple, Optional, List, Union


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
