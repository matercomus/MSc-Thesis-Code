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

    # Sample data (replace with your data loading logic)
    data = {
        "license_plate": ["A", "A", "B", "B", "A"],
        "timestamp": [
            "2023-01-01T10:00:00", "2023-01-01T10:05:00", 
            "2023-01-01T11:00:00", "2023-01-01T11:00:01",
            "2023-01-01T10:15:00"  # Temporal gap example
        ],
        "longitude": [10.0, 10.1, 20.0, 20.0001, 11.0],  # Position jump
        "latitude": [50.0, 50.1, 60.0, 60.0001, 51.0],
    }
    lazy_df = pl.LazyFrame(data).with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%S")
    )
    return SPEED_THRESHOLD, TEMPORAL_GAP_THRESHOLD, data, lazy_df


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


if __name__ == "__main__":
    app.run()
