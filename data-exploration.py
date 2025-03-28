import marimo

__generated_with = "0.11.25"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Data Exploration I

        ## Todo:

        - [ ] Missing values
        - [ ] Remove outliers
        - [ ] Exclude data outside Beijing
        - [ ] Calc velocity
        - [ ] Len, Min, Max, Mean
        - [ ] Plots
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def _(pl):
    # File path to your Parquet file
    parquet_file_path = "2019.11.25.parquet"

    # Load the Parquet file lazily using Polars
    ldf = pl.scan_parquet(parquet_file_path)
    return ldf, parquet_file_path


@app.cell
def _(ldf, pl):
    # Get column schema efficiently
    schema = ldf.collect_schema()
    column_names = schema.names()

    # Build dynamic selection
    selections = [pl.len().alias("total_rows")]

    for col in column_names:
        dtype = schema[col]
    
        # Common metrics for all columns
        selections.extend([
            pl.col(col).null_count().alias(f"null_{col}"),
            pl.col(col).min().alias(f"min_{col}"),
            pl.col(col).max().alias(f"max_{col}")
        ])
    
        # Unique count handling
        if dtype in (pl.Categorical, pl.Date, pl.Datetime, pl.Duration, pl.Time):
            selections.append(pl.col(col).n_unique().alias(f"unique_{col}"))
        else:
            selections.append(pl.col(col).approx_n_unique().alias(f"unique_{col}"))
    
        # Numeric-specific metrics
        if dtype.is_numeric():
            selections.extend([
                pl.col(col).mean().alias(f"mean_{col}"),
                pl.col(col).std().alias(f"std_{col}")
            ])

    # Execute query
    result = ldf.select(selections).collect().row(0, named=True)

    # Process results
    total_rows = result["total_rows"]
    missing_stats = {
        col: {
            "dtype": str(schema[col]),
            "missing_count": result[f"null_{col}"],
            "missing_%": (result[f"null_{col}"] / total_rows) * 100,
            "unique": result[f"unique_{col}"],
            "min": result[f"min_{col}"],
            "max": result[f"max_{col}"],
            **({"mean": result[f"mean_{col}"], "std": result[f"std_{col}"]} 
               if schema[col].is_numeric() else {})
        }
        for col in column_names
    }

    # Print analysis
    print(f"Analyzed {total_rows:,} rows\n")
    # for col, stats in missing_stats.items():
    #     print(f"📊 {col} ({stats['dtype']})")
    #     print(f"   Missing: {stats['missing_count']} ({stats['missing_%']:.1f}%)")
    #     print(f"   Unique values: {stats['unique']}")
    #     if stats.get("mean") is not None:
    #         print(f"   Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
    #     print(f"   Range: [{stats['min']} → {stats['max']}]\n")

    # Create summary DataFrame with proper type handling
    summary_data = {
        "column": column_names,
        "dtype": [stats["dtype"] for stats in missing_stats.values()],
        "missing_count": [stats["missing_count"] for stats in missing_stats.values()],
        "missing_%": [stats["missing_%"] for stats in missing_stats.values()],
        "unique": [stats["unique"] for stats in missing_stats.values()],
        "min": [str(stats["min"]) for stats in missing_stats.values()],  # Convert to string for mixed types
        "max": [str(stats["max"]) for stats in missing_stats.values()],
    }

    # Add numeric stats as separate columns with explicit dtype
    numeric_stats = {
        "mean": [stats.get("mean") for stats in missing_stats.values()],
        "std": [stats.get("std") for stats in missing_stats.values()],
    }

    # Create final DataFrame
    summary_df = pl.DataFrame({
        **summary_data,
        **{k: pl.Series(v, dtype=pl.Float64) for k, v in numeric_stats.items()}
    })

    # Reorder columns for better readability
    summary_df = summary_df.select([
        "column", "dtype", "missing_count", "missing_%", "unique",
        "mean", "std", "min", "max"
    ])
    return (
        col,
        column_names,
        dtype,
        missing_stats,
        numeric_stats,
        result,
        schema,
        selections,
        summary_data,
        summary_df,
        total_rows,
    )


@app.cell
def _(summary_df):
    summary_df
    return


@app.cell
def _(ldf):
    # Value counts for license_plate (showing top 10 for brevity - can be memory intensive for many unique plates)
    top_license_plates_ldf = ldf.group_by("license_plate").len().sort("len", descending=True).limit(10)
    top_license_plates_df = top_license_plates_ldf.collect()
    return top_license_plates_df, top_license_plates_ldf


@app.cell
def _(top_license_plates_df):
    top_license_plates_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
