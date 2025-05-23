"""
new_indicators_pipeline.py

A pure Python script for the full Beijing taxi trajectory cleaning and analysis pipeline.
- No Marimo, no notebooks, just a clean script.
- Uses tqdm for progress bars on long-running steps.
- Uses logging (see utils.py) for status updates.
- Outputs: cleaned_points_in_beijing.parquet, cleaned_with_period_id_in_beijing.parquet, periods_with_sld_ratio.parquet

Requirements: polars, pandas, tqdm, scikit-learn, scipy
"""

import logging
from tqdm import tqdm
import polars as pl
from utils import (
    configure_logging,
    add_time_distance_calcs,
    add_implied_speed,
    add_abnormality_flags,
    select_final_columns,
    add_period_id,
    summarize_periods,
    detect_outliers_pd,
    compute_iqr_thresholds,
    compute_generic_iqr_threshold,
)

def save_parquet(df: pl.DataFrame, path: str, label: str = None):
    df.write_parquet(path)
    msg = f"Saved {label or path} to {path} (shape: {df.shape})"
    print(msg)
    logging.info(msg)

def attach_period_id(cleaned_df: pl.DataFrame, period_df: pl.DataFrame) -> pl.DataFrame:
    period_meta = period_df.select(["license_plate", "period_id", "start_time", "end_time"])
    joined = cleaned_df.join(
        period_meta,
        on=["license_plate"],
        how="left",
    ).filter(
        (pl.col("timestamp") >= pl.col("start_time")) & (pl.col("timestamp") <= pl.col("end_time"))
    )
    return joined

def main():
    configure_logging()
    logging.info("Starting indicators pipeline (pure Python version)")

    # Load filtered points
    logging.info("Loading filtered_points_in_beijing.parquet...")
    lazy_df = pl.scan_parquet("data/filtered_points_in_beijing.parquet")

    # Compute robust thresholds using IQR on occupied trips only
    logging.info("Computing IQR-based thresholds for time gaps and speeds...")
    time_gap_th, speed_th = compute_iqr_thresholds(lazy_df)
    logging.info(f"Thresholds: time_gap={time_gap_th:.2f} sec, speed={speed_th:.2f} kph")

    # Compute base indicators and abnormality flags
    logging.info("Computing time, distance, speed, and abnormality flags...")
    base_df = (
        lazy_df.sort("license_plate", "timestamp")
        .pipe(add_time_distance_calcs)
        .pipe(add_implied_speed)
        .pipe(add_abnormality_flags, time_gap_th, speed_th)
        .pipe(select_final_columns)
    )
    logging.info("Collecting base DataFrame (this may take a while)...")
    base_df = base_df.collect()
    logging.info(f"Base DataFrame shape: {base_df.shape}")

    # Detect outliers using Isolation Forest
    logging.info("Detecting trajectory outliers (Isolation Forest)...")
    pd_df = base_df.to_pandas()
    tqdm.pandas(desc="Outlier detection")
    pd_df["is_outlier"] = detect_outliers_pd(pd_df).values
    results = pl.from_pandas(pd_df)

    # Remove license plates with any temporal gap or position jump
    logging.info("Filtering out taxis with any temporal gap or position jump...")
    flagged_plates = results.filter(
        (pl.col("is_temporal_gap")) | (pl.col("is_position_jump"))
    ).get_column("license_plate").unique().to_list()
    cleaned_df = (
        results.filter(~pl.col("license_plate").is_in(flagged_plates))
        .drop(["is_temporal_gap", "is_position_jump"])
    )
    logging.info(f"Cleaned DataFrame shape: {cleaned_df.shape}")
    save_parquet(cleaned_df, "data/cleaned_points_in_beijing.parquet", label="Cleaned points")

    # Add period_id and summarize periods
    logging.info("Adding period_id and summarizing periods...")
    period_df = cleaned_df.pipe(add_period_id).pipe(summarize_periods)
    # Remove small periods (fewer than 3 points)
    MIN_PERIOD_POINTS = 3
    period_df = period_df.filter(pl.col("count_rows") >= MIN_PERIOD_POINTS)
    logging.info(f"Period summary shape: {period_df.shape}")

    # Attach period_id (and period time bounds) to cleaned points
    logging.info("Attaching period_id to cleaned points...")
    cleaned_with_period_id = attach_period_id(cleaned_df, period_df)
    logging.info(f"Cleaned with period_id shape: {cleaned_with_period_id.shape}")
    save_parquet(cleaned_with_period_id, "data/cleaned_with_period_id_in_beijing.parquet", label="Cleaned points with period_id")

    # Compute per-period outlier counts (Isolation Forest)
    logging.info("Computing per-period outlier counts...")
    outlier_stats = (
        cleaned_with_period_id.group_by(["license_plate", "period_id"])
        .agg((pl.col("is_outlier") == -1).sum().alias("traj_outlier_count"))
    )
    # Join outlier stats into period summary and compute ratio and flag
    period_df = (
        period_df.join(outlier_stats, on=["license_plate", "period_id"], how="left")
        .with_columns([
            pl.col("traj_outlier_count").fill_null(0),
            (pl.col("traj_outlier_count") / pl.col("count_rows")).fill_null(0.0).alias("traj_outlier_ratio"),
            (pl.col("traj_outlier_count") > 0).alias("is_traj_outlier"),
        ])
    )
    # Flag SLD-based outliers using IQR threshold
    logging.info("Flagging SLD-based outliers...")
    sld_th = compute_generic_iqr_threshold(period_df.lazy(), "sld_ratio")
    period_df = period_df.with_columns(
        (pl.col("sld_ratio") > sld_th).alias("is_sld_outlier")
    )
    save_parquet(period_df, "data/periods_with_sld_ratio.parquet", label="Period summary with SLD ratio and outlier flags")

    logging.info("Pipeline completed: all parquet files written.")

if __name__ == "__main__":
    main()
