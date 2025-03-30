# utils.py

import polars as pl
from typing import Dict, Tuple, Optional, List


def filter_chinese_license_plates(
    lazy_df: pl.LazyFrame,
    col: str = "license_plate",
) -> pl.LazyFrame:
    """
    Filters Chinese license plates with strict validation:
    - Standard plates: 1 Chinese char + uppercase letter + 5 alphanumerics (7 chars total)
    - New energy plates: 1 Chinese char + uppercase letter + 4 alphanumerics + D/d (8 chars total)
    - Excludes I and O to avoid confusion with numbers
    """
    # Handle potential null values first
    not_null = pl.col(col).is_not_null()

    # Standard plate pattern - must be exactly 7 characters
    # First char must be Chinese, second must be uppercase letter (excluding I and O),
    # remaining 5 must be alphanumeric (excluding I and O)
    standard_pattern = r"^[\u4e00-\u9fa5][A-HJ-NP-Z][A-HJ-NP-Z0-9]{5}$"

    # New energy plate pattern - must be exactly 8 characters
    # First char must be Chinese, second must be uppercase letter (excluding I and O),
    # next 4 must be alphanumeric (excluding I and O), last must be D or d
    new_energy_pattern = r"^[\u4e00-\u9fa5][A-HJ-NP-Z][A-HJ-NP-Z0-9]{4}[Dd]$"

    # Combine conditions
    return lazy_df.filter(
        not_null
        & (
            pl.col(col).cast(pl.String).str.contains(standard_pattern)
            | pl.col(col).cast(pl.String).str.contains(new_energy_pattern)
        )
    )


def profile_data(
    ldf: pl.LazyFrame, columns: Optional[List[str]] = None, verbose: bool = False
) -> Tuple[pl.DataFrame, Dict]:
    """
    Profile a Polars LazyFrame and return summary statistics with empty DataFrame handling.
    """
    try:
        schema = ldf.collect_schema()
        column_names = columns or schema.names()

        missing_cols = set(column_names) - set(schema.names())
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")

        selections = [pl.len().alias("total_rows")]
        for col in column_names:
            dtype = schema[col]
            selections.extend(
                [
                    pl.col(col).null_count().alias(f"null_{col}"),
                    pl.col(col).min().alias(f"min_{col}"),
                    pl.col(col).max().alias(f"max_{col}"),
                ]
            )
            if dtype in (pl.Categorical, pl.Date, pl.Datetime, pl.Duration, pl.Time):
                selections.append(pl.col(col).n_unique().alias(f"unique_{col}"))
            else:
                selections.append(pl.col(col).approx_n_unique().alias(f"unique_{col}"))
            if dtype.is_numeric():
                selections.extend(
                    [
                        pl.col(col).mean().alias(f"mean_{col}"),
                        pl.col(col).std().alias(f"std_{col}"),
                    ]
                )

        result = ldf.select(selections).collect()
        if result.is_empty():
            total_rows = 0
            row_data = {k: None for k in result.columns}
        else:
            row_data = result.row(0, named=True)
            total_rows = row_data.get("total_rows", 0)

        stats_dict = {}
        for col in column_names:
            null_count = row_data.get(f"null_{col}", 0)
            if total_rows == 0:
                missing_pct = 100.0
            else:
                missing_pct = (null_count / total_rows) * 100

            stats = {
                "dtype": str(schema[col]),
                "missing_count": null_count,
                "missing_%": missing_pct,
                "unique": row_data.get(f"unique_{col}"),
                "min": row_data.get(f"min_{col}"),
                "max": row_data.get(f"max_{col}"),
            }
            if schema[col].is_numeric():
                stats.update(
                    {
                        "mean": row_data.get(f"mean_{col}"),
                        "std": row_data.get(f"std_{col}"),
                    }
                )
            stats_dict[col] = stats

            if verbose:
                print(f"📊 {col} ({stats['dtype']})")
                print(
                    f"   Missing: {stats['missing_count']} ({stats['missing_%']:.1f}%)"
                )
                print(f"   Unique: {stats['unique']}")
                if "mean" in stats:
                    mean = stats["mean"] or 0  # Handle None
                    std = stats["std"] or 0
                    print(f"   Mean: {mean:.2f} ± {std:.2f}")
                print(f"   Range: [{stats['min']} → {stats['max']}]\n")

        summary_data = {
            "column": column_names,
            "dtype": [stats["dtype"] for stats in stats_dict.values()],
            "missing_count": [stats["missing_count"] for stats in stats_dict.values()],
            "missing_%": [stats["missing_%"] for stats in stats_dict.values()],
            "unique": [stats["unique"] for stats in stats_dict.values()],
            "min": [
                str(stats["min"]) if stats["min"] is not None else None
                for stats in stats_dict.values()
            ],
            "max": [
                str(stats["max"]) if stats["max"] is not None else None
                for stats in stats_dict.values()
            ],
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
