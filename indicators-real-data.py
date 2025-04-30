

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


@app.cell
def _():
    import polars as pl

    # Enable global string cache before any Categorical operations
    pl.enable_string_cache()
    from utils import (
        configure_logging,
        add_time_distance_calcs,
        add_implied_speed,
        add_abnormality_flags,
        select_final_columns,
        add_period_id,
        summarize_periods,
        detect_outliers_pd,
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
        detect_outliers_pd,
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
    detect_outliers_pd,
):
    """
    Process pipeline and detect outliers in the trajectory data.
    """
    import pandas as pd
    import polars as pl
    # Compute base indicators
    results = (
        lazy_df.sort("license_plate", "timestamp")
        .pipe(add_time_distance_calcs)
        .pipe(add_implied_speed)
        .pipe(add_abnormality_flags, TEMPORAL_GAP_THRESHOLD, SPEED_THRESHOLD)
        .pipe(select_final_columns)
        .collect()
    )
    # Detect outliers using Isolation Forest on speed, acceleration, and direction change
    pd_df = results.to_pandas()
    pd_df["is_outlier"] = detect_outliers_pd(pd_df).values
    # Convert back to Polars
    results = pl.from_pandas(pd_df)
    return (results,)


@app.cell
def _(results):
    # Display results
    results.head(100)
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
    period_df = (
        cleaned_lazy_df.collect().pipe(add_period_id).pipe(summarize_periods)
    )
    period_lazy_df = period_df.lazy()
    return period_df, period_lazy_df


@app.cell
def _(period_lazy_df):
    # Sink periods with summarized metrics
    period_lazy_df.sink_parquet("periods_in_beijing.parquet")
    # Also save with straight-line to sum-distance ratio
    period_lazy_df.sink_parquet("periods_with_sld_ratio.parquet")
    return


@app.cell
def _(period_df):
    period_df.head(50)
    return


@app.cell
def _(cleaned_lazy_df, period_lazy_df, pl):
    # Add period_id to cleaned_lazy_df by merging with period_lazy_df based on timestamp within start and end time, and license plate
    cleaned_with_period_id_df = cleaned_lazy_df.join(
        period_lazy_df.select(
            ["license_plate", "start_time", "end_time", "period_id"]
        ),
        on=["license_plate"],
        how="left",
    ).filter(
        (pl.col("timestamp") >= pl.col("start_time"))
        & (pl.col("timestamp") <= pl.col("end_time"))
    )

    # Collect the final DataFrame with period_id column
    with_period_id_df = cleaned_with_period_id_df.collect()
    with_period_id_df.head(50)
    return (cleaned_with_period_id_df,)


@app.cell
def _(cleaned_with_period_id_df):
    cleaned_with_period_id_df.sink_parquet(
        "cleaned_with_period_id_in_beijing.parquet"
    )
    return


"""
When run as a script, bypass Marimo cycles and execute the pipeline directly.
"""
def main():
    import polars as pl
    import pandas as pd
    from utils import (
        configure_logging,
        add_time_distance_calcs,
        add_implied_speed,
        add_abnormality_flags,
        select_final_columns,
        add_period_id,
        summarize_periods,
        detect_outliers_pd,
    )

    # Setup
    pl.enable_string_cache()
    configure_logging()
    TEMPORAL_GAP_THRESHOLD = 300  # seconds
    SPEED_THRESHOLD = 200.0  # km/h

    # Load filtered points
    lazy_df = pl.scan_parquet("filtered_points_in_beijing.parquet")

    # Compute time, distance, speed, and abnormality flags
    df_base = (
        lazy_df.sort("license_plate", "timestamp")
        .pipe(add_time_distance_calcs)
        .pipe(add_implied_speed)
        .pipe(add_abnormality_flags, TEMPORAL_GAP_THRESHOLD, SPEED_THRESHOLD)
        .pipe(select_final_columns)
        .collect()
    )

    # Detect outliers in the trajectory
    pd_df = df_base.to_pandas()
    pd_df["is_outlier"] = detect_outliers_pd(pd_df).values
    results = pl.from_pandas(pd_df)

    # Remove license plates with any temporal gap or position jump
    flagged_series = results.filter(
        (pl.col("is_temporal_gap")) | (pl.col("is_position_jump"))
    ).get_column("license_plate").unique()
    flagged_list = flagged_series.to_list()
    cleaned_lazy_df = (
        results.filter(~pl.col("license_plate").is_in(flagged_list))
        .drop(["is_temporal_gap", "is_position_jump"])
        .lazy()
    )
    cleaned_lazy_df.sink_parquet("cleaned_points_in_beijing.parquet")

    # Summarize periods and compute straight-line distance ratios
    # Base period summary
    period_df = cleaned_lazy_df.collect().pipe(add_period_id).pipe(summarize_periods)
    # Attach period_id (and period time bounds) to cleaned points for outlier stats
    cleaned_with_period_id_lazy = (
        cleaned_lazy_df.join(
            period_df.lazy().select([
                "license_plate", "period_id", "start_time", "end_time"
            ]),
            on=["license_plate"], how="left"
        )
        .filter(
            (pl.col("timestamp") >= pl.col("start_time")) &
            (pl.col("timestamp") <= pl.col("end_time"))
        )
    )
    # Compute per-period outlier counts
    outlier_stats = (
        cleaned_with_period_id_lazy
        .group_by(["license_plate", "period_id"])
        .agg((pl.col("is_outlier") == -1).sum().alias("traj_outlier_count"))
        .collect()
    )
    # Join outlier stats into period summary and compute ratio and flag
    period_df = (
        period_df
        .join(outlier_stats, on=["license_plate", "period_id"], how="left")
        .with_columns([
            pl.col("traj_outlier_count").fill_null(0),
            (pl.col("traj_outlier_count") / pl.col("count_rows")).fill_null(0.0).alias("traj_outlier_ratio"),
            (pl.col("traj_outlier_count") > 0).alias("is_traj_outlier"),
        ])
    )
    # Sink enriched periods
    period_lazy_df = period_df.lazy()
    period_lazy_df.sink_parquet("periods_in_beijing.parquet")
    period_lazy_df.sink_parquet("periods_with_sld_ratio.parquet")

    # Attach period_id back to cleaned points and sink
    cleaned_with_period_id_df = cleaned_lazy_df.join(
        period_lazy_df.select(["license_plate", "start_time", "end_time", "period_id"]),
        on=["license_plate"],
        how="left",
    ).filter(
        (pl.col("timestamp") >= pl.col("start_time")) & (pl.col("timestamp") <= pl.col("end_time"))
    )
    cleaned_with_period_id_df.sink_parquet(
        "cleaned_with_period_id_in_beijing.parquet"
    )
    print("Pipeline completed: parquet files written.")

if __name__ == "__main__":
    main()
