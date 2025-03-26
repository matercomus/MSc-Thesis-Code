import marimo

__generated_with = "0.11.25"
app = marimo.App(width="full")


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
def _(ldf):
    # Value counts for occupancy_status (as before)
    occupancy_counts_ldf = ldf.group_by("occupancy_status").count()
    occupancy_counts_df = occupancy_counts_ldf.collect()
    return occupancy_counts_df, occupancy_counts_ldf


@app.cell
def _(mo, occupancy_counts_df):
    mo.vstack([
        mo.Html("<h2>Value Counts of Occupancy Status</h2>"),
        occupancy_counts_df
    ])
    return


@app.cell
def _(ldf, pl):
    # Basic statistics for timestamp (min and max)
    timestamp_stats_ldf = ldf.select([pl.min("timestamp").alias("min_timestamp"), pl.max("timestamp").alias("max_timestamp")])
    timestamp_stats_df = timestamp_stats_ldf.collect()
    return timestamp_stats_df, timestamp_stats_ldf


@app.cell
def _(mo, timestamp_stats_df):
    mo.vstack([
        mo.Html("<h2>Timestamp Range</h2>"),
        timestamp_stats_df
    ])
    return


@app.cell
def _(ldf):
    # Value counts for license_plate (showing top 10 for brevity - can be memory intensive for many unique plates)
    top_license_plates_ldf = ldf.group_by("license_plate").count().sort("count", descending=True).limit(10)
    top_license_plates_df = top_license_plates_ldf.collect()
    return top_license_plates_df, top_license_plates_ldf


@app.cell
def _(mo, top_license_plates_df):
    mo.vstack([
        mo.Html("<h2>Top 10 Most Frequent License Plates</h2>"),
        top_license_plates_df
    ])
    return


@app.cell
def _(ldf, mo, pl):
    column_names = ldf.collect_schema().names()
    mo.vstack([
        mo.Html("<h2>Column Names</h2>"),
        pl.DataFrame({"Column Names": column_names})
    ])
    return (column_names,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
