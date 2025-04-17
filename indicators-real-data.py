import marimo

__generated_with = "0.12.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from utils import configure_logging, add_time_distance_calcs, add_implied_speed, add_abnormality_flags, select_final_columns
    return (
        add_abnormality_flags,
        add_implied_speed,
        add_time_distance_calcs,
        configure_logging,
        mo,
        pl,
        select_final_columns,
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
        lazy_df
        .sort("license_plate", "timestamp")
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
    results
    return


@app.cell
def _(pl, results):
    results.filter(
        pl.col("is_temporal_gap") | pl.col("is_position_jump")
    )
    return


@app.cell
def _(pl, results):
    results.filter(
        pl.col("is_temporal_gap") & pl.col("is_position_jump")
    )
    return


if __name__ == "__main__":
    app.run()
