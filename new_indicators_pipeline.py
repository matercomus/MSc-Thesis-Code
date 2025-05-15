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
import numpy as np
import argparse
import glob
import shutil

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

def ensure_osm_graph(osm_graph_path, periods_path, buffer=0.01):
    meta_path = str(osm_graph_path) + '.meta.json'
    if Path(osm_graph_path).exists() and Path(meta_path).exists():
        print("OSM graph already exists and is up-to-date.")
        return
    print("Downloading OSM graph cropped to data bounding box...")
    periods = pd.read_parquet(periods_path)
    min_lat = min(periods['start_latitude'].min(), periods['end_latitude'].min())
    max_lat = max(periods['start_latitude'].max(), periods['end_latitude'].max())
    min_lon = min(periods['start_longitude'].min(), periods['end_longitude'].min())
    max_lon = max(periods['end_longitude'].max(), periods['end_longitude'].max())
    bbox = (max_lat + buffer, min_lat - buffer, max_lon + buffer, min_lon - buffer)
    G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
    ox.save_graphml(G, osm_graph_path)
    write_meta(meta_path, {
        "graphml_hash": file_hash(osm_graph_path),
        "bbox": [min_lat, max_lat, min_lon, max_lon],
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

    # Precompute/caching nearest nodes for all unique points
    all_points = set(
        list(zip(periods_df['start_longitude'], periods_df['start_latitude'])) +
        list(zip(periods_df['end_longitude'], periods_df['end_latitude']))
    )
    logger.info(f"Caching nearest OSM nodes for {len(all_points)} unique points...")
    point_to_node = {}
    for lon, lat in tqdm(all_points, desc="Nearest node cache"):
        try:
            point_to_node[(lon, lat)] = ox.nearest_nodes(G, lon, lat)
        except Exception:
            point_to_node[(lon, lat)] = None

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

    # Group by unique start/end node pairs to avoid redundant shortest path calculations
    to_process['start_node'] = [point_to_node.get((row['start_longitude'], row['start_latitude'])) for _, row in to_process.iterrows()]
    to_process['end_node'] = [point_to_node.get((row['end_longitude'], row['end_latitude'])) for _, row in to_process.iterrows()]
    node_pairs = set(zip(to_process['start_node'], to_process['end_node']))
    node_pair_to_dist = {}

    def compute_pair(pair):
        orig, dest = pair
        if orig is None or dest is None or orig == dest:
            return pair, 0.0
        try:
            dist = nx.shortest_path_length(G, orig, dest, weight='length') / 1000
            return pair, dist
        except Exception:
            return pair, float('nan')

    # Use ThreadPoolExecutor to avoid pickling issues
    logger.info(f"Computing shortest paths for {len(node_pairs)} unique node pairs...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_pair = {executor.submit(compute_pair, pair): pair for pair in node_pairs}
        for future in tqdm(as_completed(future_to_pair), total=len(node_pairs), desc="Node pairs"):
            pair, dist = future.result()
            node_pair_to_dist[pair] = dist

    # Assign distances to periods
    def get_dist(row):
        # Skip if sum_distance is very small or start/end node is None or identical
        if row['sum_distance'] < 0.01 or row['start_node'] is None or row['end_node'] is None or row['start_node'] == row['end_node']:
            return 0.0
        return node_pair_to_dist.get((row['start_node'], row['end_node']), float('nan'))

    # Process in batches
    all_results = []
    for i in tqdm(range(0, len(to_process), batch_size), desc="Batches"):
        batch = to_process.iloc[i:i+batch_size]
        batch['network_shortest_distance'] = batch.apply(get_dist, axis=1)
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

def clean_pipeline_outputs():
    print("Cleaning pipeline outputs and checkpoints...")
    # Remove all files in network_checkpoints/
    checkpoint_dir = Path("network_checkpoints")
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*"):
            try:
                f.unlink()
                print(f"Deleted checkpoint: {f}")
            except Exception as e:
                print(f"Could not delete {f}: {e}")
    # Remove all .meta.json files in data/
    for meta_file in glob.glob("data/*.meta.json"):
        try:
            os.remove(meta_file)
            print(f"Deleted meta: {meta_file}")
        except Exception as e:
            print(f"Could not delete {meta_file}: {e}")
    # Remove all intermediate Parquet files in data/ except filtered_points_in_beijing.parquet
    keep_files = {"filtered_points_in_beijing.parquet"}
    for pq_file in glob.glob("data/*.parquet"):
        if os.path.basename(pq_file) not in keep_files:
            try:
                os.remove(pq_file)
                print(f"Deleted parquet: {pq_file}")
            except Exception as e:
                print(f"Could not delete {pq_file}: {e}")
    print("Clean complete.")

def clean_step_outputs(step):
    """Clean only the outputs/meta/checkpoints for a specific pipeline step."""
    step_map = {
        "cleaned_points": [
            "data/cleaned_points_in_beijing.parquet",
            "data/cleaned_points_in_beijing.parquet.meta.json",
        ],
        "cleaned_with_period_id": [
            "data/cleaned_with_period_id_in_beijing.parquet",
            "data/cleaned_with_period_id_in_beijing.parquet.meta.json",
        ],
        "periods_with_sld_ratio": [
            "data/periods_with_sld_ratio.parquet",
            "data/periods_with_sld_ratio.parquet.meta.json",
        ],
        "network": [
            "data/periods_with_network_ratio.parquet",
            "data/periods_with_network_ratio.parquet.meta.json",
            "data/periods_with_network_ratio_flagged.parquet",
            "data/periods_with_network_ratio_flagged.parquet.meta.json",
            "network_checkpoints/",
        ],
        "osm_graph": [
            "beijing_drive.graphml",
            "beijing_drive.graphml.meta.json",
        ],
    }
    if step not in step_map:
        print(f"Unknown step: {step}. Valid steps: {list(step_map.keys())}")
        return
    print(f"Cleaning outputs for step: {step}")
    for path in step_map[step]:
        if path.endswith("/"):
            # Directory: remove all files inside
            if os.path.exists(path):
                for f in os.listdir(path):
                    fpath = os.path.join(path, f)
                    try:
                        os.remove(fpath)
                        print(f"Deleted checkpoint: {fpath}")
                    except Exception as e:
                        print(f"Could not delete {fpath}: {e}")
        else:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Deleted: {path}")
                except Exception as e:
                    print(f"Could not delete {path}: {e}")
    print("Step clean complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true", help="Remove all checkpoints, meta, and intermediate outputs before running pipeline.")
    parser.add_argument("--clean-step", type=str, help="Remove outputs/meta/checkpoints for a specific step only. Valid steps: cleaned_points, cleaned_with_period_id, periods_with_sld_ratio, network, osm_graph.")
    args = parser.parse_args()
    if args.clean and args.clean_step:
        parser.error("--clean and --clean-step cannot be used together.")
    if args.clean:
        clean_pipeline_outputs()
    if args.clean_step:
        clean_step_outputs(args.clean_step)

    configure_logging()
    logging.info("Starting indicators pipeline (pure Python version)")

    # --- Step 1: Cleaned points ---
    cleaned_points_path = "data/cleaned_points_in_beijing.parquet"
    cleaned_points_meta = cleaned_points_path + ".meta.json"
    filtered_points_path = "data/filtered_points_in_beijing.parquet"
    input_paths = {"filtered_points_hash": filtered_points_path}
    if is_up_to_date(cleaned_points_path, input_paths):
        logging.info(f"{cleaned_points_path} is up-to-date. Skipping computation.")
        cleaned_df = pl.read_parquet(cleaned_points_path)
    else:
        # Load filtered points
        logging.info("Loading filtered_points_in_beijing.parquet...")
        lazy_df = pl.scan_parquet(filtered_points_path)

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
        save_parquet(cleaned_df, cleaned_points_path, label="Cleaned points")
        write_meta(cleaned_points_meta, {"filtered_points_hash": file_hash(filtered_points_path)})

    # --- Step 2: Cleaned with period_id ---
    cleaned_with_pid_path = "data/cleaned_with_period_id_in_beijing.parquet"
    cleaned_with_pid_meta = cleaned_with_pid_path + ".meta.json"
    input_paths = {"cleaned_points_hash": cleaned_points_path}
    if is_up_to_date(cleaned_with_pid_path, input_paths):
        logging.info(f"{cleaned_with_pid_path} is up-to-date. Skipping computation.")
        cleaned_with_period_id = pl.read_parquet(cleaned_with_pid_path)
    else:
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
        save_parquet(cleaned_with_period_id, cleaned_with_pid_path, label="Cleaned points with period_id")
        write_meta(cleaned_with_pid_meta, {"cleaned_points_hash": file_hash(cleaned_points_path)})

    # --- Step 3: Period summary with SLD ratio ---
    periods_sld_path = "data/periods_with_sld_ratio.parquet"
    periods_sld_meta = periods_sld_path + ".meta.json"
    input_paths = {"cleaned_with_pid_hash": cleaned_with_pid_path}
    if is_up_to_date(periods_sld_path, input_paths):
        logging.info(f"{periods_sld_path} is up-to-date. Skipping computation.")
        period_df = pl.read_parquet(periods_sld_path)
    else:
        # Compute per-period outlier counts (Isolation Forest)
        logging.info("Computing per-period outlier counts...")
        outlier_stats = (
            cleaned_with_period_id.group_by(["license_plate", "period_id"])
            .agg((pl.col("is_outlier") == -1).sum().alias("traj_outlier_count"))
        )
        # Join outlier stats into period summary and compute ratio and flag
        period_df = (
            cleaned_with_period_id.pipe(add_period_id).pipe(summarize_periods)
            .join(outlier_stats, on=["license_plate", "period_id"], how="left")
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
        save_parquet(period_df, periods_sld_path, label="Period summary with SLD ratio and outlier flags")
        write_meta(periods_sld_meta, {"cleaned_with_pid_hash": file_hash(cleaned_with_pid_path)})

    # After period_df is created and saved as data/periods_with_sld_ratio.parquet
    # 1. Ensure OSM graph
    ensure_osm_graph("beijing_drive.graphml", periods_sld_path)
    # 2. Compute network shortest paths and ratios (batched, threaded, checkpointed)
    network_ratio_path = "data/periods_with_network_ratio.parquet"
    compute_network_shortest_paths_batched(
        periods_path=periods_sld_path,
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
