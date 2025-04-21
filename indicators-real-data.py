

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl
    from utils import (
        configure_logging,
        add_time_distance_calcs,
        add_implied_speed,
        add_abnormality_flags,
        select_final_columns,
        add_period_id,
        summarize_periods,
    )
    return (
        add_abnormality_flags,
        add_implied_speed,
        add_period_id,
        add_time_distance_calcs,
        configure_logging,
        pl,
        select_final_columns,
        summarize_periods,
    )


@app.cell
def _(configure_logging, pl):
    # Configuration
    configure_logging()
    TEMPORAL_GAP_THRESHOLD = 300  # 5 minutes in seconds
    SPEED_THRESHOLD = 200.0  # km/h

    # Load data
    lazy_df = pl.scan_parquet("filtered_points_in_beijing.parquet")
    return SPEED_THRESHOLD, TEMPORAL_GAP_THRESHOLD, lazy_df


@app.cell
def _(
    SPEED_THRESHOLD,
    TEMPORAL_GAP_THRESHOLD,
    add_abnormality_flags,
    add_implied_speed,
    add_time_distance_calcs,
    lazy_df,
    select_final_columns,
):
    # Process pipeline
    results = (
        lazy_df.sort("license_plate", "timestamp")
        .pipe(add_time_distance_calcs)
        .pipe(add_implied_speed)
        .pipe(add_abnormality_flags, TEMPORAL_GAP_THRESHOLD, SPEED_THRESHOLD)
        .pipe(select_final_columns)
        .collect()
    )
    return (results,)


@app.cell
def _(results):
    # Display results
    results.head(100)
    return


@app.cell
def _(pl, results):
    results.filter(pl.col("is_temporal_gap") | pl.col("is_position_jump"))
    return


@app.cell
def _(pl, results):
    results.filter(pl.col("is_temporal_gap") & pl.col("is_position_jump"))
    return


@app.cell
def _(pl, results):
    # Compute all unique flagged license plates
    flagged_license_plates = (
        results.filter(pl.col("is_temporal_gap") | pl.col("is_position_jump"))
        .select("license_plate")
        .unique()
    )

    # Remove rows with any flagged license plate and save the cleaned lazy DF as parquet
    cleaned_lazy_df = (
        results.filter(
            ~pl.col("license_plate").is_in(
                flagged_license_plates.get_column("license_plate")
            )
        )
        .drop(["is_temporal_gap", "is_position_jump"])
        .lazy()
    )

    # Sink cleaned lazy DF
    cleaned_lazy_df.sink_parquet("cleaned_points_in_beijing.parquet")
    return (cleaned_lazy_df,)


@app.cell
def _(cleaned_lazy_df):
    cleaned_lazy_df.collect().head(50)
    return


@app.cell
def _(add_period_id, cleaned_lazy_df, summarize_periods):
    # Add period_id to cleaned data
    period_lazy_df = (
        cleaned_lazy_df.collect()
        .pipe(add_period_id)
        .pipe(summarize_periods)
        .lazy()
    )

    # Display the first few rows of the cleaned period DF
    period_lazy_df.collect().sort(
        "license_plate", "occupancy_status", "start_time"
    ).head(50)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
