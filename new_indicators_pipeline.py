"""
new_indicators_pipeline.py

A pure Python script for the full Beijing taxi trajectory cleaning and analysis pipeline.
- No Marimo, no notebooks, just a clean script.
- Uses tqdm for progress bars on long-running steps.
- Uses logging (see utils.py) for status updates.
- Outputs: cleaned_points_in_beijing.parquet, cleaned_with_period_id_in_beijing.parquet, periods_with_sld_ratio.parquet

Requirements: polars, pandas, tqdm, scikit-learn
"""

import logging
from tqdm import tqdm
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
    compute_iqr_thresholds,
    compute_generic_iqr_threshold,
)
import osmnx as ox
import networkx as nx
from pipeline_utils import file_hash, write_meta, is_up_to_date
from pathlib import Path
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def ensure_osm_graph(osm_graph_path, place_name="Beijing, China"):
    meta_path = str(osm_graph_path) + '.meta.json'
    if Path(osm_graph_path).exists() and Path(meta_path).exists():
        print("OSM graph already exists and is up-to-date.")
        return
    print("Downloading OSM graph...")
    G = ox.graph_from_place(place_name, network_type="drive")
    ox.save_graphml(G, osm_graph_path)
    write_meta(meta_path, {
        "graphml_hash": file_hash(osm_graph_path),
        "place_name_hash": hashlib.sha256(place_name.encode()).hexdigest()
    })

def compute_network_shortest_paths_batched(
    periods_path,
    osm_graph_path,
    output_path,
    batch_size=500,
    num_workers=8,
    checkpoint_dir="network_checkpoints"
):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("network_shortest_path")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load periods
    periods_df = pd.read_parquet(periods_path)
    G = ox.load_graphml(osm_graph_path)

    # Unique key for checkpointing
    periods_df['unique_key'] = periods_df['license_plate'].astype(str) + '_' + periods_df['period_id'].astype(str)

    # Check for existing checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, "network_paths_checkpoint.parquet")
    if os.path.exists(checkpoint_file):
        done_df = pd.read_parquet(checkpoint_file)
        done_keys = set(done_df['unique_key'])
        logger.info(f"Loaded checkpoint with {len(done_keys)} completed periods.")
    else:
        done_df = pd.DataFrame()
        done_keys = set()

    # Filter out already processed periods
    to_process = periods_df[~periods_df['unique_key'].isin(done_keys)].copy()
    logger.info(f"Processing {len(to_process)} new periods (skipping {len(done_keys)} already done).")

    def get_dist(row):
        try:
            orig = ox.nearest_nodes(G, row['start_longitude'], row['start_latitude'])
            dest = ox.nearest_nodes(G, row['end_longitude'], row['end_latitude'])
            return nx.shortest_path_length(G, orig, dest, weight='length') / 1000
        except Exception as e:
            logger.warning(f"Failed for period {row['unique_key']}: {e}")
            return float('nan')

    # Process in batches
    all_results = []
    for i in tqdm(range(0, len(to_process), batch_size), desc="Batches"):
        batch = to_process.iloc[i:i+batch_size]
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(get_dist, row): idx for idx, row in batch.iterrows()}
            for future in tqdm(as_completed(future_to_idx), total=len(batch), desc=f"Batch {i//batch_size+1}"):
                idx = future_to_idx[future]
                try:
                    dist = future.result()
                except Exception as e:
                    dist = float('nan')
                results.append((idx, dist))
        # Assign results
        for idx, dist in results:
            batch.at[idx, 'network_shortest_distance'] = dist
        all_results.append(batch)
        # Save checkpoint
        if os.path.exists(checkpoint_file):
            prev = pd.read_parquet(checkpoint_file)
            pd.concat([prev, batch]).drop_duplicates('unique_key').to_parquet(checkpoint_file)
        else:
            batch.to_parquet(checkpoint_file)
        logger.info(f"Checkpointed batch {i//batch_size+1} ({len(batch)} periods).")

    # Combine with previous results
    if not done_df.empty:
        all_results.append(done_df)
    final_df = pd.concat(all_results, ignore_index=True)
    final_df['route_deviation_ratio'] = final_df['sum_distance'] / final_df['network_shortest_distance']
    final_df.to_parquet(output_path)
    logger.info(f"Saved final results to {output_path}")

    return final_df

def compute_network_outlier_flag(input_path, output_path, iqr_multiplier=1.5):
    input_paths = {"network_ratio_hash": input_path}
    if is_up_to_date(output_path, input_paths):
        print(f"{output_path} is up-to-date. Skipping computation.")
        return pd.read_parquet(output_path)
    df = pd.read_parquet(input_path)
    q1 = df['route_deviation_ratio'].quantile(0.25)
    q3 = df['route_deviation_ratio'].quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + iqr_multiplier * iqr
    df['is_network_outlier'] = df['route_deviation_ratio'] > threshold
    df.to_parquet(output_path)
    write_meta(str(output_path) + '.meta.json', {
        "network_ratio_hash": file_hash(input_path)
    })
    return df

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

    # After period_df is created and saved as data/periods_with_sld_ratio.parquet
    # 1. Ensure OSM graph
    ensure_osm_graph("beijing_drive.graphml", place_name="Beijing, China")
    # 2. Compute network shortest paths and ratios (batched, threaded, checkpointed)
    network_ratio_path = "data/periods_with_network_ratio.parquet"
    compute_network_shortest_paths_batched(
        periods_path="data/periods_with_sld_ratio.parquet",
        osm_graph_path="beijing_drive.graphml",
        output_path=network_ratio_path,
        batch_size=500,
        num_workers=8,
        checkpoint_dir="network_checkpoints"
    )
    # 3. Compute threshold and flag outliers
    final_periods_path = "data/periods_with_network_ratio_flagged.parquet"
    compute_network_outlier_flag(
        input_path=network_ratio_path,
        output_path=final_periods_path
    )

    logging.info("Pipeline completed: all parquet files written.")

if __name__ == "__main__":
    main()
