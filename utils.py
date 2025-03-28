import polars as pl
from typing import Dict, Tuple, Optional, List


def filter_chinese_license_plates(
    lazy_df: pl.LazyFrame,
    col: str = "license_plate",
    regex_pattern: str = r"^[\u4e00-\u9fa5][A-Za-z]",
    min_length: int = 7,
    max_length: int = 8,
) -> pl.LazyFrame:
    """
    Efficiently filters license plates without schema resolution.
    Preserves original dtype (works with categorical/string columns).
    """
    return lazy_df.filter(
        # Cast to string only for the regex/length checks
        pl.col(col).cast(pl.String).str.contains(regex_pattern)
        & pl.col(col).cast(pl.String).str.len_chars().is_between(min_length, max_length)
    )


def profile_data(
    ldf: pl.LazyFrame, columns: Optional[List[str]] = None, verbose: bool = False
) -> Tuple[pl.DataFrame, Dict]:
    """
    Profile a Polars LazyFrame and return summary statistics.

    Args:
        ldf: Existing LazyFrame to analyze
        columns: Specific columns to profile (default: all)
        verbose: Print detailed analysis to console

    Returns:
        Tuple of (summary_df, stats_dict)

    Example:
        ldf = pl.scan_parquet("data.parquet")
        summary_df, stats = profile_data(ldf)
    """
    try:
        schema = ldf.collect_schema()
        column_names = columns or schema.names()

        # Validate columns exist
        missing_cols = set(column_names) - set(schema.names())
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        # Build dynamic selection
        selections = [pl.len().alias("total_rows")]

        for col in column_names:
            dtype = schema[col]

            # Common metrics
            selections.extend(
                [
                    pl.col(col).null_count().alias(f"null_{col}"),
                    pl.col(col).min().alias(f"min_{col}"),
                    pl.col(col).max().alias(f"max_{col}"),
                ]
            )

            # Unique counts
            if dtype in (pl.Categorical, pl.Date, pl.Datetime, pl.Duration, pl.Time):
                selections.append(pl.col(col).n_unique().alias(f"unique_{col}"))
            else:
                selections.append(pl.col(col).approx_n_unique().alias(f"unique_{col}"))

            # Numeric metrics
            if dtype.is_numeric():
                selections.extend(
                    [
                        pl.col(col).mean().alias(f"mean_{col}"),
                        pl.col(col).std().alias(f"std_{col}"),
                    ]
                )

        # Execute query
        result = ldf.select(selections).collect().row(0, named=True)
        total_rows = result["total_rows"]

        # Process results
        stats_dict = {}
        for col in column_names:
            stats = {
                "dtype": str(schema[col]),
                "missing_count": result[f"null_{col}"],
                "missing_%": (result[f"null_{col}"] / total_rows) * 100,
                "unique": result[f"unique_{col}"],
                "min": result[f"min_{col}"],
                "max": result[f"max_{col}"],
            }
            if schema[col].is_numeric():
                stats.update(
                    {"mean": result.get(f"mean_{col}"), "std": result.get(f"std_{col}")}
                )
            stats_dict[col] = stats

            if verbose:
                print(f"📊 {col} ({stats['dtype']})")
                print(
                    f"   Missing: {stats['missing_count']} ({stats['missing_%']:.1f}%)"
                )
                print(f"   Unique: {stats['unique']}")
                if "mean" in stats:
                    print(f"   Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
                print(f"   Range: [{stats['min']} → {stats['max']}]\n")

        # Create summary DataFrame
        summary_data = {
            "column": column_names,
            "dtype": [stats["dtype"] for stats in stats_dict.values()],
            "missing_count": [stats["missing_count"] for stats in stats_dict.values()],
            "missing_%": [stats["missing_%"] for stats in stats_dict.values()],
            "unique": [stats["unique"] for stats in stats_dict.values()],
            "min": [str(stats["min"]) for stats in stats_dict.values()],
            "max": [str(stats["max"]) for stats in stats_dict.values()],
        }

        numeric_stats = {
            "mean": [stats.get("mean") for stats in stats_dict.values()],
            "std": [stats.get("std") for stats in stats_dict.values()],
        }

        summary_df = pl.DataFrame(
            {
                **summary_data,
                **{k: pl.Series(v, dtype=pl.Float64) for k, v in numeric_stats.items()},
            }
        ).select(
            [
                "column",
                "dtype",
                "missing_count",
                "missing_%",
                "unique",
                "mean",
                "std",
                "min",
                "max",
            ]
        )

        return summary_df, stats_dict

    except Exception as e:
        raise RuntimeError(f"Data profiling failed: {str(e)}") from e
